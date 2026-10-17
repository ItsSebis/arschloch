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
//! Starting and resuming a run live in `setup`, the stage functions in
//! `stages`, the seed sets in `seeds`, the parallel pass with progress
//! reports in `progress`.

use std::sync::Arc;
use std::time::Instant;

use neat::{Genome, Population};

mod progress;
mod seeds;
mod setup;
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
use crate::{FeatureSet, NeatStrategy, Strategy, FEATURE_SET_VERSION};

/// One member of the opponent pool.
#[derive(Clone)]
pub struct Opponent {
    pub name: String,
    pub strategy: Arc<dyn Strategy>,
}

/// Receives progress while a run executes. All methods default to doing
/// nothing; implementors only override what they show. Called on the
/// thread that called `Trainer::run`, while the evaluation runs.
pub trait TrainObserver {
    fn on_start(&mut self, _start: &RunStart) {}
    /// `done` of `total` genomes of the current generation evaluated.
    fn on_eval_progress(&mut self, _generation: u32, _done: usize, _total: usize) {}
    fn on_generation(&mut self, _event: &GenerationEvent) {}
    fn on_finish(&mut self, _end: &RunEnd) {}
    /// Asked before every generation: `true` ends the run there (the same as reaching
    /// the generation limit, so `--resume` continues it). Never true by default.
    fn should_stop(&mut self) -> bool {
        false
    }
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
            feature_names: self
                .config
                .features
                .names()
                .iter()
                .map(|&n| n.to_owned())
                .collect(),
            resumed_from_generation: self.resumed_from,
            warm_started_from: self.warm_started_from.clone(),
        };
        self.dir
            .append_event(&Event::RunStart(Box::new(start.clone())))?;
        observer.on_start(&start);

        while self.population.generation() < self.config.generations && !observer.should_stop() {
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
        let mut eval = self.evaluate_population(generation, observer);
        // Racing is part of the training evaluation (and of its timing).
        let racing = self.race(generation, &mut eval);
        timings.training_evaluation = stage.elapsed().as_secs_f64();
        let (fitness, skill_terms) = (eval.fitness, eval.skill);
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
            .is_none_or(|b| reeval.fitness > b.reeval.fitness_value());
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
                + heldout.as_ref().map_or(0, |h| h.matches as u64 + 1))
            // Racing's extra matches.
            + racing
                .as_ref()
                .map_or(0, |r| rounds_per_match * (r.raced_final * r.extra_matches) as u64);
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
            racing,
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
