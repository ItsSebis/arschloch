//! The generational training loop: evaluate every genome on the same
//! matches, let `neat` breed the next generation, re-evaluate the
//! champion on fresh matches, and record everything.
//!
//! The scores, champions and genomes of a run are a function of its
//! `TrainConfig` alone: genomes are evaluated independently (so thread
//! count cannot change results), match seeds derive only from
//! `(run seed, generation, index)`, and the population's state is
//! checkpointed after every generation, so a resumed run is identical to
//! one that never stopped. Only the wall-clock fields of the events
//! (elapsed time, rounds per second, stage timings) vary between runs.
//!
//! The stage functions live in `stages`, the seed sets in `seeds`, the
//! parallel pass with progress reports in `progress`.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use neat::{Genome, Population};

mod progress;
mod seeds;
mod stages;
mod stats;
#[cfg(test)]
mod tests;

use self::stats::fitness_stats;
use super::config::TrainConfig;
use super::evaluate::match_seed;
use super::events::{
    ChampionStats, Complexity, Event, GenerationEvent, RunEnd, RunStart, StageTimings,
    SCHEMA_VERSION,
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
    warm_started_from: Option<String>,
}

/// Decisions recorded for each new best champion.
const DECISIONS_PER_BEST: usize = 12;

fn strategy_for(genome: &Genome) -> Arc<dyn Strategy> {
    Arc::new(
        NeatStrategy::new("candidate", genome).expect("trained genomes use this build's features"),
    )
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
        Self::start(config, opponents, dir, None)
    }

    /// Starts a new run in `dir` from the final population of the run in
    /// `source`, which is only read. The new run has its own settings,
    /// seed and generation count (from 0); the population size must match
    /// the source's because its genomes are kept as they are.
    ///
    /// # Errors
    ///
    /// `TrainError::Checkpoint` if `source` holds no usable run,
    /// `TrainError::Mismatch` if it was trained on another feature set or
    /// has another population size, otherwise as `new`. Nothing is
    /// created in `dir` when the source is refused.
    pub fn new_from(
        config: TrainConfig,
        opponents: Vec<Opponent>,
        dir: &Path,
        source: &Path,
    ) -> Result<Self, TrainError> {
        Self::start(config, opponents, dir, Some(source))
    }

    /// Reads the run in `source` and checks that a warm start of a run
    /// with `population_size` genomes can be built on it, without writing
    /// anything. Callers use it to refuse a bad source before creating
    /// any file.
    ///
    /// # Errors
    ///
    /// `TrainError::Checkpoint` if `source` holds no usable run,
    /// `TrainError::Mismatch` for another feature set or population size.
    pub fn check_warm_source(
        source: &Path,
        population_size: usize,
    ) -> Result<Checkpoint, TrainError> {
        let checkpoint = RunDir::open_existing(source)?.read_checkpoint()?;
        if checkpoint.feature_count != FEATURE_COUNT
            || checkpoint.feature_set_version != FEATURE_SET_VERSION
        {
            return Err(TrainError::Mismatch(format!(
                "{} was trained with {} features (feature-set version {}) but this build has {} (version {}); \
                 its genomes cannot be built on, though `--resume` can still continue it with the old build",
                source.display(),
                checkpoint.feature_count,
                checkpoint.feature_set_version,
                FEATURE_COUNT,
                FEATURE_SET_VERSION
            )));
        }
        let source_size = checkpoint.config.neat.population_size;
        if source_size != population_size {
            return Err(TrainError::Mismatch(format!(
                "{} has a population of {source_size} but this run asks for {population_size}; \
                 a warm start keeps the genomes it is given",
                source.display()
            )));
        }
        Ok(checkpoint)
    }

    fn start(
        config: TrainConfig,
        opponents: Vec<Opponent>,
        dir: &Path,
        source: Option<&Path>,
    ) -> Result<Self, TrainError> {
        config.validate().map_err(TrainError::Config)?;
        if opponents.len() != config.opponent_specs.len() {
            return Err(TrainError::Config(format!(
                "{} opponents were built for {} specs",
                opponents.len(),
                config.opponent_specs.len()
            )));
        }
        let seed = match_seed(config.seed, u64::MAX, 0);
        // Everything that can refuse the source happens before the new
        // directory is touched.
        let (population, warm_started_from) = if let Some(source) = source {
            let checkpoint = Self::check_warm_source(source, config.neat.population_size)?;
            let population =
                Population::warm_start(checkpoint.population, config.neat.clone(), seed)
                    .map_err(|e| TrainError::Checkpoint(e.to_string()))?;
            (population, Some(source.display().to_string()))
        } else {
            let population = Population::new(FEATURE_COUNT, config.neat.clone(), seed)
                .map_err(|e| TrainError::Config(e.to_string()))?;
            (population, None)
        };
        let dir = RunDir::create_new(dir)?;
        dir.write_config(&config)?;
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
            warm_started_from,
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
            warm_started_from: None,
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
            warm_started_from: self.warm_started_from.clone(),
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

    fn step(
        &mut self,
        observer: &mut dyn TrainObserver,
        started: Instant,
    ) -> Result<(), TrainError> {
        let step_started = Instant::now();
        let generation = self.population.generation();
        let mut timings = StageTimings::default();

        let stage = Instant::now();
        let (fitness, skill_terms) = self.evaluate_population(generation, observer);
        timings.training_evaluation = stage.elapsed().as_secs_f64();
        let stage = Instant::now();
        let (champion_index, training_rank) = self.select_champion(&fitness);
        timings.champion_selection = stage.elapsed().as_secs_f64();
        // `advance` replaces the genomes, so take the champion first.
        let champion = self.population.genomes()[champion_index].clone();
        let hall_generations: Vec<u32> = self.hall.iter().map(|m| m.generation).collect();
        let stats = fitness_stats(&fitness, skill_terms.as_deref());
        let stage = Instant::now();
        self.population.set_fitness(fitness.clone());
        let report = self.population.advance();
        timings.speciation_and_reproduction = stage.elapsed().as_secs_f64();

        let (reeval, opponents, hall_score) = self.reevaluate(&champion, &mut timings);
        // File writes are timed on their own and added up; the confirmation
        // and the decision sample (which also play matches) are separate.
        let stage = Instant::now();
        let is_new_best = self
            .best
            .as_ref()
            .is_none_or(|b| reeval.mean > b.reeval.mean);
        let genome_file = self.dir.write_champion(generation, &champion)?;
        let mut files_secs = stage.elapsed().as_secs_f64();
        let heldout = if is_new_best {
            Some(self.record_new_best(
                generation,
                &champion,
                &reeval,
                &mut timings,
                &mut files_secs,
            )?)
        } else {
            None
        };

        let stage = Instant::now();
        self.update_hall(generation, &champion);
        timings.hall_of_fame += stage.elapsed().as_secs_f64();
        timings.checkpoint_and_files = files_secs;

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
        // Neither `generation_secs` nor the timings include writing the
        // event line and the checkpoint (they come after the event is built).
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
                skill_term: skill_terms.as_ref().map(|t| t[champion_index]),
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
            timings: Some(timings),
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
