//! The generational loop: speciate, rank, allocate offspring, reproduce.
//!
//! The caller owns evaluation. A generation is:
//! `genomes()` -> evaluate each -> `set_fitness(..)` -> `advance()`.
//! Higher fitness is better; fitness may be negative.
//!
//! Everything random flows through one seeded `Xoshiro256PlusPlus` (a
//! generator whose state serializes, which is what makes a run resumable), and species and
//! genomes are always visited in index order, so a run is a pure
//! function of `(config, num_inputs, seed, fitness values)`.

use rand::rngs::Xoshiro256PlusPlus;
use rand::seq::IndexedRandom;
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::config::NeatConfig;
use crate::crossover::crossover;
use crate::error::NeatError;
use crate::genome::Genome;
use crate::innovation::InnovationTracker;
use crate::mutation::mutate;
use crate::species::{assign, Species, SpeciesStats};

/// Numbers describing the generation that `advance` just consumed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationReport {
    pub generation: u32,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub median_fitness: f64,
    pub min_fitness: f64,
    /// Index into the generation's `genomes()` of the fittest genome
    /// (first one on ties).
    pub champion_index: usize,
    pub species: Vec<SpeciesStats>,
    /// The speciation threshold this generation was grouped with.
    pub compatibility_threshold: f64,
    pub mean_hidden_nodes: f64,
    pub mean_enabled_connections: f64,
    pub innovation_count: usize,
}

/// Everything needed to continue a run exactly where it stopped: the
/// generator state, innovation registry, species bookkeeping and the
/// current genomes. Taken *between* generations (after `advance`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationState {
    config: NeatConfig,
    rng: Xoshiro256PlusPlus,
    tracker: InnovationTracker,
    genomes: Vec<Genome>,
    species: Vec<Species>,
    next_species_id: u32,
    threshold: f64,
    generation: u32,
    best: Option<(Genome, f64)>,
}

pub struct Population {
    config: NeatConfig,
    rng: Xoshiro256PlusPlus,
    tracker: InnovationTracker,
    genomes: Vec<Genome>,
    fitness: Option<Vec<f64>>,
    species: Vec<Species>,
    next_species_id: u32,
    threshold: f64,
    generation: u32,
    best: Option<(Genome, f64)>,
}

impl Population {
    /// Generation 0: `config.population_size` minimal genomes (inputs and
    /// bias wired to the output) with independently random weights.
    ///
    /// # Errors
    ///
    /// Returns `NeatError::InvalidConfig` if `config` fails validation or
    /// `num_inputs` is zero.
    ///
    /// # Panics
    ///
    /// Panics if `num_inputs` does not fit in a `u32` node id.
    pub fn new(num_inputs: usize, config: NeatConfig, seed: u64) -> Result<Self, NeatError> {
        config.validate()?;
        if num_inputs == 0 {
            return Err(NeatError::InvalidConfig(
                "num_inputs must be at least 1".into(),
            ));
        }
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut tracker =
            InnovationTracker::new(u32::try_from(num_inputs).expect("input count fits u32") + 2);
        let genomes = (0..config.population_size)
            .map(|_| Genome::minimal(num_inputs, &mut tracker, &config, &mut rng))
            .collect();
        Ok(Self {
            threshold: config.compatibility_threshold,
            config,
            rng,
            tracker,
            genomes,
            fitness: None,
            species: Vec::new(),
            next_species_id: 0,
            generation: 0,
            best: None,
        })
    }

    /// Captures the population between generations; see `PopulationState`.
    ///
    /// # Panics
    ///
    /// Panics if fitness has been recorded but not yet consumed by
    /// `advance`: that half-finished generation cannot be resumed.
    #[must_use]
    pub fn snapshot(&self) -> PopulationState {
        assert!(
            self.fitness.is_none(),
            "snapshot between generations: fitness is pending"
        );
        PopulationState {
            config: self.config.clone(),
            rng: self.rng.clone(),
            tracker: self.tracker.clone(),
            genomes: self.genomes.clone(),
            species: self.species.clone(),
            next_species_id: self.next_species_id,
            threshold: self.threshold,
            generation: self.generation,
            best: self.best.clone(),
        }
    }

    /// Rebuilds a population from a snapshot (for example one read back
    /// from JSON), checking it is internally consistent.
    ///
    /// # Errors
    ///
    /// Returns `NeatError::InvalidConfig` if the config is invalid or the
    /// genome count differs from `population_size`, and
    /// `NeatError::InvalidGenome` if the genomes disagree on their input
    /// count.
    pub fn restore(state: PopulationState) -> Result<Self, NeatError> {
        state.config.validate()?;
        if state.genomes.len() != state.config.population_size {
            return Err(NeatError::InvalidConfig(format!(
                "snapshot holds {} genomes but population_size is {}",
                state.genomes.len(),
                state.config.population_size
            )));
        }
        let inputs = state.genomes[0].num_inputs();
        if state.genomes.iter().any(|g| g.num_inputs() != inputs) {
            return Err(NeatError::InvalidGenome(
                "snapshot genomes disagree on their input count".into(),
            ));
        }
        Ok(Self {
            config: state.config,
            rng: state.rng,
            tracker: state.tracker,
            genomes: state.genomes,
            fitness: None,
            species: state.species,
            next_species_id: state.next_species_id,
            threshold: state.threshold,
            generation: state.generation,
            best: state.best,
        })
    }

    /// Starts a *new* run from an earlier one's snapshot: the genomes,
    /// innovation history, species and threshold carry over, while the
    /// generation counter, the best-so-far and the random generator start
    /// afresh (`seed`) and the settings are `config`'s. The population
    /// size cannot change, because the genomes are kept as they are.
    ///
    /// # Errors
    ///
    /// `NeatError::InvalidConfig` if `config` fails validation or its
    /// `population_size` differs from the snapshot's; `InvalidGenome` as
    /// for `restore`.
    pub fn warm_start(
        state: PopulationState,
        config: NeatConfig,
        seed: u64,
    ) -> Result<Self, NeatError> {
        config.validate()?;
        if config.population_size != state.genomes.len() {
            return Err(NeatError::InvalidConfig(format!(
                "a warm start keeps the snapshot's {} genomes but population_size is {}",
                state.genomes.len(),
                config.population_size
            )));
        }
        let mut population = Self::restore(PopulationState { config, ..state })?;
        population.generation = 0;
        population.best = None;
        // Each species' record and stagnation were measured on the old run's
        // fitness; against a different opponent pool they would make every
        // species look stagnant (or protected) for the wrong reason.
        for species in &mut population.species {
            species.best_fitness = f64::NEG_INFINITY;
            species.stagnation = 0;
            species.age = 0;
        }
        population.rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        Ok(population)
    }

    #[must_use]
    pub fn genomes(&self) -> &[Genome] {
        &self.genomes
    }

    #[must_use]
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// The fittest genome seen in any generation so far, and its fitness.
    #[must_use]
    pub fn best(&self) -> Option<(&Genome, f64)> {
        self.best
            .as_ref()
            .map(|(genome, fitness)| (genome, *fitness))
    }

    #[must_use]
    pub fn config(&self) -> &NeatConfig {
        &self.config
    }

    /// Records the fitness of `genomes()`, in the same order.
    ///
    /// # Panics
    ///
    /// Panics if the length differs from the population's or any value is
    /// not finite: both are bugs in the evaluator, and NaN would silently
    /// corrupt every ranking.
    pub fn set_fitness(&mut self, fitness: Vec<f64>) {
        assert_eq!(fitness.len(), self.genomes.len(), "one fitness per genome");
        assert!(
            fitness.iter().all(|f| f.is_finite()),
            "fitness must be finite"
        );
        self.fitness = Some(fitness);
    }

    /// Consumes the recorded fitness: updates species bookkeeping and the
    /// all-time best, then replaces `genomes()` with the next generation.
    ///
    /// # Panics
    ///
    /// Panics if `set_fitness` was not called for this generation.
    pub fn advance(&mut self) -> GenerationReport {
        let fitness = self
            .fitness
            .take()
            .expect("set_fitness must be called before advance");

        assign(
            &mut self.species,
            &self.genomes,
            self.threshold,
            &self.config,
            &mut self.next_species_id,
            &mut self.rng,
        );
        let threshold_used = self.threshold;
        self.update_species_records(&fitness);
        let champion_index = argmax(&fitness);
        if self
            .best
            .as_ref()
            .is_none_or(|(_, f)| fitness[champion_index] > *f)
        {
            self.best = Some((
                self.genomes[champion_index].clone(),
                fitness[champion_index],
            ));
        }
        let report = self.report(&fitness, champion_index, threshold_used);

        let quotas = self.offspring_quotas(&fitness);
        let next = self.reproduce(&fitness, &quotas);
        // A species that got no offspring is finished. Keeping it would
        // let its stale representative capture healthy offspring next
        // generation (assignment is first-fit), and they would inherit
        // its stagnation and be culled with it.
        let mut quota_iter = quotas.iter();
        self.species
            .retain(|_| quota_iter.next().is_some_and(|&quota| quota > 0));
        self.adapt_threshold();
        self.genomes = next;
        self.generation += 1;
        report
    }

    fn update_species_records(&mut self, fitness: &[f64]) {
        for species in &mut self.species {
            let best = species
                .members
                .iter()
                .map(|&m| fitness[m])
                .fold(f64::NEG_INFINITY, f64::max);
            if best > species.best_fitness {
                species.best_fitness = best;
                species.stagnation = 0;
            } else {
                species.stagnation += 1;
            }
            species.age += 1;
        }
    }

    fn report(&self, fitness: &[f64], champion_index: usize, threshold: f64) -> GenerationReport {
        let count = to_f64(fitness.len());
        let mut sorted = fitness.to_vec();
        sorted.sort_by(f64::total_cmp);
        let median = if sorted.len() % 2 == 1 {
            sorted[sorted.len() / 2]
        } else {
            f64::midpoint(sorted[sorted.len() / 2 - 1], sorted[sorted.len() / 2])
        };
        let species = self
            .species
            .iter()
            .map(|s| SpeciesStats {
                id: s.id,
                size: s.members.len(),
                age: s.age,
                generation_best: s
                    .members
                    .iter()
                    .map(|&m| fitness[m])
                    .fold(f64::NEG_INFINITY, f64::max),
                mean_fitness: s.members.iter().map(|&m| fitness[m]).sum::<f64>()
                    / to_f64(s.members.len()),
                best_fitness: s.best_fitness,
                stagnation: s.stagnation,
            })
            .collect();
        GenerationReport {
            generation: self.generation,
            best_fitness: fitness[champion_index],
            mean_fitness: fitness.iter().sum::<f64>() / count,
            median_fitness: median,
            min_fitness: sorted[0],
            champion_index,
            species,
            compatibility_threshold: threshold,
            mean_hidden_nodes: self
                .genomes
                .iter()
                .map(|g| to_f64(g.hidden_count()))
                .sum::<f64>()
                / count,
            mean_enabled_connections: self
                .genomes
                .iter()
                .map(|g| to_f64(g.enabled_connection_count()))
                .sum::<f64>()
                / count,
            innovation_count: self.tracker.innovation_count(),
        }
    }

    /// Offspring per species (index-aligned with `self.species`), summing
    /// to the population size. Species compete on mean fitness (shifted
    /// to be positive), which is what explicit fitness sharing amounts to;
    /// stagnant species get nothing unless they hold the best fitness.
    fn offspring_quotas(&self, fitness: &[f64]) -> Vec<usize> {
        let floor = fitness.iter().copied().fold(f64::INFINITY, f64::min);
        // Two species are exempt from culling: the one holding the best
        // fitness ever recorded, and the one holding this generation's
        // champion. Fitness is noisy, so a species' record is the luckiest
        // sample it ever had; without the second exemption a stagnant
        // species could be culled while holding the current best genome.
        let record_holder = self
            .species
            .iter()
            .enumerate()
            .fold((0, f64::NEG_INFINITY), |best, (i, s)| {
                if s.best_fitness > best.1 {
                    (i, s.best_fitness)
                } else {
                    best
                }
            })
            .0;
        let champion = argmax(fitness);
        let champions_species = self
            .species
            .iter()
            .position(|s| s.members.contains(&champion));
        let is_protected = |i: usize| i == record_holder || Some(i) == champions_species;
        let weights: Vec<f64> = self
            .species
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if s.stagnation >= self.config.stagnation_limit && !is_protected(i) {
                    return 0.0;
                }
                let shifted: f64 = s.members.iter().map(|&m| fitness[m] - floor + 1e-6).sum();
                shifted / to_f64(s.members.len())
            })
            .collect();
        allocate(&weights, self.config.population_size)
    }

    fn reproduce(&mut self, fitness: &[f64], quotas: &[usize]) -> Vec<Genome> {
        let mut next = Vec::with_capacity(self.config.population_size);
        for (species, &quota) in self.species.iter().zip(quotas) {
            if quota == 0 {
                continue;
            }
            let mut ranked = species.members.clone();
            ranked.sort_by(|&a, &b| fitness[b].total_cmp(&fitness[a]));
            let mut produced = 0;
            if ranked.len() >= self.config.elitism_min_species_size {
                next.push(self.genomes[ranked[0]].clone());
                produced += 1;
            }
            let survivors = &ranked[..survivor_count(ranked.len(), self.config.survival_fraction)];
            while produced < quota {
                let mut child =
                    if survivors.len() >= 2 && self.rng.random_bool(self.config.crossover_rate) {
                        let first = *survivors.choose(&mut self.rng).expect("non-empty");
                        let mut second = first;
                        while second == first {
                            second = *survivors.choose(&mut self.rng).expect("non-empty");
                        }
                        let (fitter, other) = if fitness[first] >= fitness[second] {
                            (first, second)
                        } else {
                            (second, first)
                        };
                        crossover(
                            &self.genomes[fitter],
                            &self.genomes[other],
                            &self.config,
                            &mut self.rng,
                        )
                    } else {
                        let parent = *survivors.choose(&mut self.rng).expect("non-empty");
                        self.genomes[parent].clone()
                    };
                mutate(&mut child, &mut self.tracker, &self.config, &mut self.rng);
                next.push(child);
                produced += 1;
            }
        }
        debug_assert_eq!(next.len(), self.config.population_size);
        next
    }

    /// Nudges the threshold toward the configured species count, so a
    /// population that has become too uniform (or too fragmented) is
    /// re-grouped at a usable granularity.
    fn adapt_threshold(&mut self) {
        if self.species.len() < self.config.target_species {
            self.threshold = (self.threshold - self.config.threshold_step)
                .max(self.config.min_compatibility_threshold);
        } else if self.species.len() > self.config.target_species {
            self.threshold += self.config.threshold_step;
        }
    }
}

/// How many of a species' best members may become parents: at least one.
fn survivor_count(size: usize, fraction: f64) -> usize {
    let wanted = (to_f64(size) * fraction).ceil();
    // `wanted` is in [0, size], so the conversion cannot lose anything.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let wanted = wanted as usize;
    wanted.clamp(1, size)
}

fn to_f64(n: usize) -> f64 {
    // Populations are far below 2^52, where usize -> f64 stays exact.
    #[allow(clippy::cast_precision_loss)]
    let value = n as f64;
    value
}

/// Index of the largest value (the first one on ties).
fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 {
                (i, v)
            } else {
                best
            }
        })
        .0
}

/// Splits `total` seats proportionally to `weights` (largest-remainder
/// method), so the result always sums to `total` and zero-weight entries
/// get nothing.
///
/// # Panics
///
/// Panics if no weight is positive.
fn allocate(weights: &[f64], total: usize) -> Vec<usize> {
    let sum: f64 = weights.iter().sum();
    assert!(sum > 0.0, "at least one species must be able to reproduce");
    let exact: Vec<f64> = weights.iter().map(|w| w / sum * to_f64(total)).collect();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let mut seats: Vec<usize> = exact.iter().map(|e| e.floor() as usize).collect();
    let mut by_remainder: Vec<usize> = (0..weights.len()).filter(|&i| weights[i] > 0.0).collect();
    by_remainder
        .sort_by(|&a, &b| (exact[b] - exact[b].floor()).total_cmp(&(exact[a] - exact[a].floor())));
    let mut leftover = total - seats.iter().sum::<usize>();
    for &index in by_remainder.iter().cycle() {
        if leftover == 0 {
            break;
        }
        seats[index] += 1;
        leftover -= 1;
    }
    seats
}

#[cfg(test)]
mod tests;
