//! The generational training loop: evaluate every genome on the same
//! matches, let `neat` breed the next generation, re-evaluate the
//! champion on fresh matches, and record everything.
//!
//! A run is a pure function of its `TrainConfig`: genomes are evaluated
//! independently (so thread count cannot change results), match seeds
//! derive only from `(run seed, generation, index)`, and the population's
//! state is checkpointed after every generation, so a resumed run is
//! identical to one that never stopped.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use neat::{Genome, Population};
use rayon::prelude::*;

use super::config::TrainConfig;
use super::decisions::record_decisions;
use super::evaluate::{evaluate, match_seed, Opponents, Score};
use super::events::{
    ChampionStats, Complexity, Event, FitnessStats, GenerationEvent, OpponentStat, RunEnd,
    RunStart, SCHEMA_VERSION,
};
use super::run_dir::{BestRecord, Checkpoint, HallMember, RunDir, TrainError};
use crate::{NeatStrategy, Strategy, FEATURE_COUNT, FEATURE_NAMES, FEATURE_SET_VERSION};

/// One member of the opponent pool.
#[derive(Clone)]
pub struct Opponent {
    pub name: String,
    pub strategy: Arc<dyn Strategy>,
}

/// Receives progress while a run executes. All methods default to doing
/// nothing; implementors only override what they show. Called on the
/// thread that called `Trainer::run`, between evaluation steps.
pub trait TrainObserver {
    fn on_start(&mut self, _start: &RunStart) {}
    /// `done` of `total` genomes of the current generation evaluated.
    fn on_eval_progress(&mut self, _generation: u32, _done: usize, _total: usize) {}
    fn on_generation(&mut self, _event: &GenerationEvent) {}
    fn on_finish(&mut self, _end: &RunEnd) {}
}

pub struct Trainer {
    config: TrainConfig,
    opponents: Vec<Opponent>,
    dir: RunDir,
    population: Population,
    best: Option<BestRecord>,
    hall: Vec<HallMember>,
    total_rounds: u64,
    elapsed_before: f64,
    resumed_from: Option<u32>,
}

/// Genomes are evaluated in this many parallel batches, with a progress
/// callback between batches.
const PROGRESS_STEPS: usize = 10;

/// Decisions recorded for each new best champion.
const DECISIONS_PER_BEST: usize = 12;

/// Training matches of `generation`: different every generation, so a
/// genome cannot be tuned to one set of deals.
fn training_seeds(config: &TrainConfig, generation: u32) -> Vec<u64> {
    (0..config.matches_per_genome as u64)
        .map(|i| match_seed(config.seed, 2 * u64::from(generation), i))
        .collect()
}

/// The fixed set every generation's champion is compared on: the same
/// deals every time (so champions are compared like for like) and never
/// part of training.
fn reeval_seeds(config: &TrainConfig) -> Vec<u64> {
    (0..config.reeval_matches as u64)
        .map(|i| match_seed(config.seed, u64::MAX - 1, i))
        .collect()
}

/// The set the champion *candidates* are ranked on. It is separate from
/// `reeval_seeds` so that the champion's reported score is measured on
/// matches that played no part in choosing it (a score measured on the
/// selection matches would be inflated by the selection).
fn selection_seeds(config: &TrainConfig) -> Vec<u64> {
    (0..config.reeval_matches as u64)
        .map(|i| match_seed(config.seed, u64::MAX - 4, i))
        .collect()
}

/// A second fixed set, used only to confirm a new best champion: it was
/// not used to pick it, so its score is not inflated by the selection.
fn heldout_seeds(config: &TrainConfig) -> Vec<u64> {
    (0..config.reeval_matches as u64 * 2)
        .map(|i| match_seed(config.seed, u64::MAX - 2, i))
        .collect()
}

fn strategy_for(genome: &Genome) -> Arc<dyn Strategy> {
    Arc::new(
        NeatStrategy::new("candidate", genome).expect("trained genomes use this build's features"),
    )
}

/// The indices of the `k` highest values, best first (ties: lower index).
fn top_indices(values: &[f64], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    order.truncate(k);
    order
}

#[allow(clippy::cast_precision_loss)] // counts are far below 2^52
fn fitness_stats(fitness: &[f64]) -> FitnessStats {
    let count = fitness.len() as f64;
    let mean = fitness.iter().sum::<f64>() / count;
    let mut sorted = fitness.to_vec();
    sorted.sort_by(f64::total_cmp);
    let (min, best) = (sorted[0], sorted[sorted.len() - 1]);
    let median = if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        f64::midpoint(sorted[sorted.len() / 2 - 1], sorted[sorted.len() / 2])
    };
    let std_dev = (fitness.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / count).sqrt();
    let mut histogram = vec![0u32; 10];
    for &value in fitness {
        let bucket = if best > min {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let bucket = (((value - min) / (best - min)) * 10.0) as usize;
            bucket.min(9)
        } else {
            0
        };
        histogram[bucket] += 1;
    }
    FitnessStats {
        best,
        mean,
        median,
        min,
        std_dev,
        histogram,
    }
}

impl Trainer {
    /// Starts a new run in `dir`.
    ///
    /// # Errors
    ///
    /// `TrainError::Config` for an invalid setup (or a directory that
    /// already holds a run), `TrainError::Io` if files cannot be written.
    pub fn new(
        config: TrainConfig,
        opponents: Vec<Opponent>,
        dir: &Path,
    ) -> Result<Self, TrainError> {
        config.validate().map_err(TrainError::Config)?;
        if opponents.len() != config.opponent_specs.len() {
            return Err(TrainError::Config(format!(
                "{} opponents were built for {} specs",
                opponents.len(),
                config.opponent_specs.len()
            )));
        }
        let dir = RunDir::create_new(dir)?;
        dir.write_config(&config)?;
        let population = Population::new(
            FEATURE_COUNT,
            config.neat.clone(),
            match_seed(config.seed, u64::MAX, 0),
        )
        .map_err(|e| TrainError::Config(e.to_string()))?;
        let trainer = Self {
            config,
            opponents,
            dir,
            population,
            best: None,
            hall: Vec::new(),
            total_rounds: 0,
            elapsed_before: 0.0,
            resumed_from: None,
        };
        // A run killed during generation 0 must still be resumable (and
        // must not look like a run to overwrite), so the directory holds a
        // checkpoint from the very start.
        trainer.dir.write_checkpoint(&trainer.checkpoint(0.0))?;
        Ok(trainer)
    }

    fn checkpoint(&self, elapsed_secs: f64) -> Checkpoint {
        Checkpoint {
            schema_version: SCHEMA_VERSION,
            feature_count: FEATURE_COUNT,
            feature_set_version: FEATURE_SET_VERSION,
            config: self.config.clone(),
            opponent_names: self.opponents.iter().map(|o| o.name.clone()).collect(),
            population: self.population.snapshot(),
            best: self.best.clone(),
            hall_of_fame: self.hall.clone(),
            total_rounds: self.total_rounds,
            elapsed_secs,
        }
    }

    /// Continues the run in `dir` from its last checkpoint. `generations`
    /// optionally raises (or lowers) the total to run up to.
    ///
    /// # Errors
    ///
    /// `TrainError::Checkpoint` for unusable files, `TrainError::Mismatch`
    /// if `opponents` are not the pool the run started with.
    pub fn resume(
        dir: &Path,
        opponents: Vec<Opponent>,
        generations: Option<u32>,
    ) -> Result<Self, TrainError> {
        let dir = RunDir::open_existing(dir)?;
        let checkpoint = dir.read_checkpoint()?;
        if checkpoint.feature_count != FEATURE_COUNT
            || checkpoint.feature_set_version != FEATURE_SET_VERSION
        {
            return Err(TrainError::Mismatch(format!(
                "the run was trained with {} features (feature-set version {}) but this build has {} (version {}); \
                 its genomes cannot be continued",
                checkpoint.feature_count, checkpoint.feature_set_version, FEATURE_COUNT, FEATURE_SET_VERSION
            )));
        }
        let names: Vec<&str> = opponents.iter().map(|o| o.name.as_str()).collect();
        if names
            != checkpoint
                .opponent_names
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        {
            return Err(TrainError::Mismatch(format!(
                "the run used opponents {:?} but {:?} were given",
                checkpoint.opponent_names, names
            )));
        }
        let mut config = checkpoint.config;
        if let Some(total) = generations {
            config.generations = total;
        }
        config.validate().map_err(TrainError::Config)?;
        let population = Population::restore(checkpoint.population)
            .map_err(|e| TrainError::Checkpoint(e.to_string()))?;
        let completed = population.generation();
        dir.truncate_events_from(completed)?;
        Ok(Self {
            config,
            opponents,
            dir,
            population,
            best: checkpoint.best,
            hall: checkpoint.hall_of_fame,
            total_rounds: checkpoint.total_rounds,
            elapsed_before: checkpoint.elapsed_secs,
            resumed_from: Some(completed),
        })
    }

    #[must_use]
    pub fn config(&self) -> &TrainConfig {
        &self.config
    }

    #[must_use]
    pub fn completed_generations(&self) -> u32 {
        self.population.generation()
    }

    /// Runs generations until `config.generations` are complete.
    ///
    /// # Errors
    ///
    /// `TrainError::Io` if a run file cannot be written.
    pub fn run(&mut self, observer: &mut dyn TrainObserver) -> Result<RunEnd, TrainError> {
        let started = Instant::now();
        let start = RunStart {
            schema_version: SCHEMA_VERSION,
            config: self.config.clone(),
            opponents: self.opponents.iter().map(|o| o.name.clone()).collect(),
            feature_names: FEATURE_NAMES.iter().map(|&n| n.to_owned()).collect(),
            resumed_from_generation: self.resumed_from,
        };
        self.dir
            .append_event(&Event::RunStart(Box::new(start.clone())))?;
        observer.on_start(&start);

        while self.population.generation() < self.config.generations {
            self.step(observer, started)?;
        }

        let end = RunEnd {
            generations_completed: self.population.generation(),
            best_generation: self.best.as_ref().map(|b| b.generation),
            best_reeval: self.best.as_ref().map(|b| b.reeval.clone()),
            best_heldout: self.best.as_ref().map(|b| b.heldout.clone()),
            elapsed_secs: self.elapsed_before + started.elapsed().as_secs_f64(),
        };
        self.dir.append_event(&Event::RunEnd(end.clone()))?;
        observer.on_finish(&end);
        Ok(end)
    }

    /// The fixed opponents only: what champions are compared against, so
    /// scores stay comparable across generations as the hall changes.
    fn fixed_pool(&self) -> Vec<Arc<dyn Strategy>> {
        self.opponents.iter().map(|o| o.strategy.clone()).collect()
    }

    fn hall_pool(&self) -> Vec<Arc<dyn Strategy>> {
        self.hall
            .iter()
            .map(|member| -> Arc<dyn Strategy> {
                Arc::new(
                    NeatStrategy::new(format!("HoF({})", member.generation), &member.genome)
                        .expect("hall members use this build's features"),
                )
            })
            .collect()
    }

    /// What genomes train against: the fixed opponents plus the hall.
    fn training_pool(&self) -> Vec<Arc<dyn Strategy>> {
        let mut pool = self.fixed_pool();
        pool.extend(self.hall_pool());
        pool
    }

    fn evaluate_population(&self, generation: u32, observer: &mut dyn TrainObserver) -> Vec<f64> {
        let table = self.config.table();
        let pool = self.training_pool();
        let seeds = training_seeds(&self.config, generation);
        let genomes = self.population.genomes();
        let batch = genomes.len().div_ceil(PROGRESS_STEPS);
        let mut fitness = Vec::with_capacity(genomes.len());
        for chunk in genomes.chunks(batch) {
            let scores: Vec<f64> = chunk
                .par_iter()
                .map(|genome| {
                    evaluate(
                        &strategy_for(genome),
                        &table,
                        Opponents::Mixed(&pool),
                        &seeds,
                    )
                    .mean
                })
                .collect();
            fitness.extend(scores);
            observer.on_eval_progress(generation, fitness.len(), genomes.len());
        }
        fitness
    }

    /// Picks the generation's champion: the best of the `k` genomes with
    /// the highest training fitness, ranked on the selection matches.
    /// Returns its index and its rank by training fitness (0 = the training
    /// best). With one candidate no matches are played.
    fn select_champion(&self, fitness: &[f64]) -> (usize, usize) {
        let ranked = top_indices(fitness, self.config.champion_candidates);
        if ranked.len() == 1 {
            return (ranked[0], 0);
        }
        let table = self.config.table();
        let pool = self.fixed_pool();
        let seeds = selection_seeds(&self.config);
        let genomes = self.population.genomes();
        let scores: Vec<Score> = ranked
            .par_iter()
            .map(|&index| {
                evaluate(
                    &strategy_for(&genomes[index]),
                    &table,
                    Opponents::Mixed(&pool),
                    &seeds,
                )
            })
            .collect();
        let winner = scores.iter().enumerate().fold(0, |best, (rank, score)| {
            if score.mean > scores[best].mean {
                rank
            } else {
                best
            }
        });
        (ranked[winner], winner)
    }

    /// The champion on the fixed matches: against the mixed fixed pool,
    /// against each fixed opponent alone and, once the hall has members,
    /// against the hall alone. These matches played no part in choosing it.
    fn reevaluate(&self, champion: &Genome) -> (Score, Vec<OpponentStat>, Option<Score>) {
        let table = self.config.table();
        let seeds = reeval_seeds(&self.config);
        let candidate = strategy_for(champion);
        let mixed = evaluate(
            &candidate,
            &table,
            Opponents::Mixed(&self.fixed_pool()),
            &seeds,
        );
        let per_opponent = self
            .opponents
            .par_iter()
            .map(|opponent| OpponentStat {
                name: opponent.name.clone(),
                score: evaluate(
                    &candidate,
                    &table,
                    Opponents::Only(&opponent.strategy),
                    &seeds,
                )
                .into(),
            })
            .collect();
        let hall = self.hall_pool();
        let hall_score = (!hall.is_empty())
            .then(|| evaluate(&candidate, &table, Opponents::Mixed(&hall), &seeds));
        (mixed, per_opponent, hall_score)
    }

    /// A few real decisions of the champion (for the dashboard's decision
    /// inspector), from one held-out match.
    fn sample_decisions(
        &self,
        generation: u32,
        champion: &Genome,
    ) -> super::decisions::DecisionFile {
        let pool: Vec<Arc<dyn Strategy>> =
            self.opponents.iter().map(|o| o.strategy.clone()).collect();
        record_decisions(
            champion,
            &self.config.table(),
            &pool,
            heldout_seeds(&self.config)[0],
            DECISIONS_PER_BEST,
            generation,
        )
    }

    /// The champion's score on the held-out matches (see `heldout_seeds`).
    fn confirm(&self, champion: &Genome) -> Score {
        let pool: Vec<Arc<dyn Strategy>> =
            self.opponents.iter().map(|o| o.strategy.clone()).collect();
        evaluate(
            &strategy_for(champion),
            &self.config.table(),
            Opponents::Mixed(&pool),
            &heldout_seeds(&self.config),
        )
    }

    fn step(
        &mut self,
        observer: &mut dyn TrainObserver,
        started: Instant,
    ) -> Result<(), TrainError> {
        let step_started = Instant::now();
        let generation = self.population.generation();

        let fitness = self.evaluate_population(generation, observer);
        let (champion_index, training_rank) = self.select_champion(&fitness);
        // `advance` replaces the genomes, so take the champion first.
        let champion = self.population.genomes()[champion_index].clone();
        let hall_generations: Vec<u32> = self.hall.iter().map(|m| m.generation).collect();
        let stats = fitness_stats(&fitness);
        self.population.set_fitness(fitness.clone());
        let report = self.population.advance();

        let (reeval, opponents, hall_score) = self.reevaluate(&champion);
        let is_new_best = self
            .best
            .as_ref()
            .is_none_or(|b| reeval.mean > b.reeval.mean);
        let genome_file = self.dir.write_champion(generation, &champion)?;
        let heldout = if is_new_best {
            self.dir.write_best(&champion)?;
            let heldout = self.confirm(&champion);
            self.dir
                .write_decisions(&self.sample_decisions(generation, &champion))?;
            self.best = Some(BestRecord {
                generation,
                reeval: reeval.clone().into(),
                heldout: heldout.clone().into(),
            });
            Some(heldout)
        } else {
            None
        };

        // The hall takes a fresh champion every `interval` generations.
        if self.config.hall_of_fame_size > 0
            && generation > 0
            && generation.is_multiple_of(self.config.hall_of_fame_interval)
        {
            self.hall.push(HallMember {
                generation,
                genome: champion.clone(),
            });
            while self.hall.len() > self.config.hall_of_fame_size {
                self.hall.remove(0);
            }
        }

        let table = self.config.table();
        let rounds_per_match = table.rounds as u64;
        let rounds_evaluated = rounds_per_match
            * (self.config.matches_per_genome as u64 * self.config.neat.population_size as u64
                + self.config.reeval_matches as u64
                    * (if self.config.champion_candidates > 1 {
                        self.config.champion_candidates as u64
                    } else {
                        0
                    } + 1
                        + self.opponents.len() as u64
                        + u64::from(hall_score.is_some()))
                // A new best also plays one match to record its decisions.
                + heldout.as_ref().map_or(0, |h| h.matches as u64 + 1));
        self.total_rounds += rounds_evaluated;
        let generation_secs = step_started.elapsed().as_secs_f64();
        let elapsed_secs = self.elapsed_before + started.elapsed().as_secs_f64();

        let event = GenerationEvent {
            generation,
            elapsed_secs,
            generation_secs,
            rounds_evaluated,
            total_rounds: self.total_rounds,
            #[allow(clippy::cast_precision_loss)]
            rounds_per_sec: rounds_evaluated as f64 / generation_secs.max(1e-9),
            fitness: stats,
            champion: ChampionStats {
                train_fitness: fitness[champion_index],
                reeval: reeval.into(),
                heldout: heldout.map(Into::into),
                training_rank,
                hidden_nodes: champion.hidden_count(),
                enabled_connections: champion.enabled_connection_count(),
                genome_file,
                is_new_best,
            },
            opponents,
            hall_of_fame: hall_generations,
            hall_score: hall_score.map(Into::into),
            species: report.species,
            compatibility_threshold: report.compatibility_threshold,
            complexity: Complexity {
                mean_hidden_nodes: report.mean_hidden_nodes,
                mean_enabled_connections: report.mean_enabled_connections,
                innovation_count: report.innovation_count,
            },
        };

        // The checkpoint comes last: if the process dies earlier, resume
        // redoes this generation and trims the duplicate event.
        self.dir
            .append_event(&Event::Generation(Box::new(event.clone())))?;
        self.dir.write_checkpoint(&self.checkpoint(elapsed_secs))?;
        observer.on_generation(&event);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::test_support::sample;
    use super::*;

    #[test]
    fn top_indices_orders_best_first_with_ties_going_to_the_lower_index() {
        let values = [0.5, 0.9, 0.5, -1.0, 0.9];
        assert_eq!(top_indices(&values, 3), vec![1, 4, 0]);
        assert_eq!(top_indices(&values, 1), vec![1]);
        assert_eq!(top_indices(&values, 99), vec![1, 4, 0, 2, 3]);
        assert!(top_indices(&[], 3).is_empty());
    }

    #[test]
    fn champion_selection_has_its_own_seed_stream() {
        // The candidates are ranked on `selection_seeds`; the champion's
        // reported score is then measured on `reeval_seeds`. If they shared
        // matches, the reported score would be inflated by the very
        // selection that chose the champion.
        let config = sample();
        let selection = selection_seeds(&config);
        assert_eq!(selection.len(), config.reeval_matches);
        assert_eq!(
            selection,
            selection_seeds(&config),
            "fixed, not per generation"
        );
        for seed in &selection {
            assert!(!reeval_seeds(&config).contains(seed));
            assert!(!heldout_seeds(&config).contains(seed));
            for generation in 0..200 {
                assert!(!training_seeds(&config, generation).contains(seed));
            }
        }
    }

    #[test]
    fn champions_are_compared_on_one_fixed_seed_set_that_training_never_uses() {
        let config = sample();
        let fixed = reeval_seeds(&config);
        let held_out = heldout_seeds(&config);
        assert_eq!(fixed.len(), config.reeval_matches);
        assert_eq!(held_out.len(), config.reeval_matches * 2);
        // The same every generation (paired comparison between champions).
        assert_eq!(fixed, reeval_seeds(&config));
        // Disjoint from each other and from every generation's training matches.
        for seed in fixed.iter().chain(&held_out) {
            assert_eq!(
                fixed.iter().chain(&held_out).filter(|s| *s == seed).count(),
                1,
                "{seed}"
            );
            for generation in 0..200 {
                assert!(!training_seeds(&config, generation).contains(seed));
            }
        }
    }
}
