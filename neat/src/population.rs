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
        let mut population = Self::restore(PopulationState {
            config,
            ..state
        })?;
        population.generation = 0;
        population.best = None;
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
mod tests {
    use super::*;

    fn tiny_config() -> NeatConfig {
        NeatConfig {
            population_size: 30,
            ..NeatConfig::default()
        }
    }

    #[test]
    fn allocate_sums_to_the_total_and_respects_proportions() {
        assert_eq!(allocate(&[1.0, 1.0, 2.0], 8), vec![2, 2, 4]);
        let uneven = allocate(&[1.0, 1.0, 1.0], 10);
        assert_eq!(uneven.iter().sum::<usize>(), 10);
        assert!(uneven.iter().all(|&n| (3..=4).contains(&n)));
    }

    #[test]
    fn allocate_never_seats_a_zero_weight_entry() {
        let seats = allocate(&[0.0, 3.0, 0.0, 1.0], 7);
        assert_eq!(seats[0], 0);
        assert_eq!(seats[2], 0);
        assert_eq!(seats.iter().sum::<usize>(), 7);
    }

    #[test]
    fn survivor_count_keeps_at_least_one_and_at_most_all() {
        assert_eq!(survivor_count(1, 0.2), 1);
        assert_eq!(survivor_count(10, 0.2), 2);
        assert_eq!(survivor_count(10, 1.0), 10);
        assert_eq!(survivor_count(3, 0.0), 1);
    }

    #[test]
    fn new_population_has_the_configured_size_and_valid_genomes() {
        let population = Population::new(3, tiny_config(), 1).unwrap();
        assert_eq!(population.genomes().len(), 30);
        assert_eq!(population.generation(), 0);
        assert!(population.best().is_none());
        assert!(population.genomes().iter().all(|g| g.num_inputs() == 3));
    }

    #[test]
    fn invalid_setups_are_rejected() {
        assert!(Population::new(0, tiny_config(), 1).is_err());
        let bad = NeatConfig {
            population_size: 1,
            ..tiny_config()
        };
        assert!(Population::new(2, bad, 1).is_err());
    }

    #[test]
    fn advance_keeps_the_population_size_constant() {
        let mut population = Population::new(2, tiny_config(), 7).unwrap();
        for generation in 0..10 {
            let fitness = (0..30).map(|i| f64::from(i % 7) - 3.0).collect();
            population.set_fitness(fitness);
            let report = population.advance();
            assert_eq!(report.generation, generation);
            assert_eq!(population.genomes().len(), 30);
            assert_eq!(report.species.iter().map(|s| s.size).sum::<usize>(), 30);
        }
        assert_eq!(population.generation(), 10);
    }

    #[test]
    fn report_statistics_describe_the_evaluated_generation() {
        let mut population = Population::new(2, tiny_config(), 3).unwrap();
        let fitness: Vec<f64> = (0..30).map(f64::from).collect();
        population.set_fitness(fitness);
        let report = population.advance();
        assert!((report.best_fitness - 29.0).abs() < f64::EPSILON);
        assert!((report.min_fitness - 0.0).abs() < f64::EPSILON);
        assert!((report.mean_fitness - 14.5).abs() < 1e-12);
        assert!((report.median_fitness - 14.5).abs() < 1e-12);
        assert_eq!(report.champion_index, 29);
        let (best, fitness) = population.best().unwrap();
        assert!((fitness - 29.0).abs() < f64::EPSILON);
        assert_eq!(best.num_inputs(), 2);
    }

    #[test]
    fn best_never_regresses() {
        let mut population = Population::new(2, tiny_config(), 3).unwrap();
        population.set_fitness((0..30).map(f64::from).collect());
        population.advance();
        population.set_fitness(vec![-5.0; 30]);
        population.advance();
        assert!((population.best().unwrap().1 - 29.0).abs() < f64::EPSILON);
    }

    #[test]
    fn the_champion_survives_unchanged_in_a_large_species() {
        let config = NeatConfig {
            population_size: 20,
            compatibility_threshold: 1000.0,
            ..NeatConfig::default()
        };
        let mut population = Population::new(2, config, 5).unwrap();
        let champion = population.genomes()[4].clone();
        let mut fitness = vec![0.0; 20];
        fitness[4] = 10.0;
        population.set_fitness(fitness);
        population.advance();
        assert!(population.genomes().contains(&champion));
    }

    #[test]
    fn stagnant_species_stop_reproducing_but_the_best_one_never_does() {
        let config = NeatConfig {
            population_size: 40,
            compatibility_threshold: 0.0001,
            min_compatibility_threshold: 0.0001,
            stagnation_limit: 2,
            ..NeatConfig::default()
        };
        let mut population = Population::new(2, config, 9).unwrap();
        for _ in 0..6 {
            // Constant fitness: nothing ever improves, every species
            // stagnates, yet the population must keep its size.
            population.set_fitness(vec![1.0; 40]);
            population.advance();
            assert_eq!(population.genomes().len(), 40);
        }
    }

    #[test]
    #[should_panic(expected = "set_fitness must be called")]
    fn advancing_without_fitness_panics() {
        Population::new(2, tiny_config(), 1).unwrap().advance();
    }

    #[test]
    #[should_panic(expected = "finite")]
    fn nan_fitness_panics() {
        let mut population = Population::new(2, tiny_config(), 1).unwrap();
        let mut fitness = vec![0.0; 30];
        fitness[3] = f64::NAN;
        population.set_fitness(fitness);
    }

    #[test]
    #[should_panic(expected = "one fitness per genome")]
    fn wrong_fitness_length_panics() {
        Population::new(2, tiny_config(), 1)
            .unwrap()
            .set_fitness(vec![0.0; 3]);
    }

    #[test]
    fn the_smallest_legal_population_keeps_evolving() {
        let config = NeatConfig {
            population_size: 2,
            compatibility_threshold: 0.0001,
            min_compatibility_threshold: 0.0001,
            ..NeatConfig::default()
        };
        let mut population = Population::new(2, config, 1).unwrap();
        for generation in 0..20 {
            population.set_fitness(vec![f64::from(generation), -1.0]);
            population.advance();
            assert_eq!(population.genomes().len(), 2);
        }
    }

    #[test]
    fn a_wide_input_layer_works() {
        let mut population = Population::new(40, tiny_config(), 2).unwrap();
        for _ in 0..3 {
            population.set_fitness((0..30).map(f64::from).collect());
            population.advance();
        }
        assert!(population.genomes().iter().all(|g| g.num_inputs() == 40));
    }

    #[test]
    fn every_genome_stays_valid_across_many_mixed_lineage_generations() {
        // High structural mutation rates make lineages that discovered
        // different hidden nodes interbreed constantly, which is where
        // out-of-order node ids and cycles would show up.
        let config = NeatConfig {
            population_size: 60,
            add_node_rate: 0.4,
            add_connection_rate: 0.4,
            toggle_enable_rate: 0.2,
            ..NeatConfig::default()
        };
        let mut population = Population::new(3, config, 11).unwrap();
        for generation in 0..40 {
            for genome in population.genomes() {
                Genome::from_parts(
                    genome.num_inputs(),
                    genome.nodes().to_vec(),
                    genome.connections().to_vec(),
                )
                .unwrap_or_else(|error| panic!("generation {generation}: {error}"));
            }
            let fitness = (0..60)
                .map(|i| f64::from((i * 13 + generation) % 17))
                .collect();
            population.set_fitness(fitness);
            population.advance();
        }
    }

    #[test]
    fn a_culled_species_does_not_survive_to_capture_next_generations_offspring() {
        let config = NeatConfig {
            population_size: 30,
            compatibility_threshold: 0.0001,
            min_compatibility_threshold: 0.0001,
            stagnation_limit: 3,
            ..NeatConfig::default()
        };
        let mut population = Population::new(2, config, 4).unwrap();
        population.set_fitness(vec![1.0; 30]);
        population.advance();
        assert!(population.species.len() > 2, "need several species");
        // Species 1 holds the best fitness (protected). Species 0 is
        // hopelessly stagnant, and is first in line to capture any
        // genome near its representative (first-fit assignment), which is
        // exactly what lets a zombie species swallow healthy offspring.
        population.species[1].best_fitness = 100.0;
        population.species[0].best_fitness = 50.0;
        population.species[0].stagnation = 99;
        population.species[0].representative = population.genomes()[0].clone();
        population.species[1].representative = population.genomes()[1].clone();
        let doomed = population.species[0].id;
        // The champion (genome 1) is in the protected species, not the doomed one.
        let mut fitness = vec![1.0; 30];
        fitness[1] = 2.0;
        population.set_fitness(fitness);
        population.advance();
        assert!(
            population.species.iter().all(|s| s.id != doomed),
            "a species with no offspring quota must be dropped, not kept as a trap"
        );
    }

    #[test]
    fn the_species_holding_the_generations_champion_is_never_culled() {
        // Fitness is noisy, so a species' record is the luckiest sample it
        // ever had and a stagnant species can easily hold *this*
        // generation's best genome. Culling it would delete the champion's
        // whole lineage.
        let config = NeatConfig {
            population_size: 30,
            compatibility_threshold: 0.0001,
            min_compatibility_threshold: 0.0001,
            stagnation_limit: 3,
            ..NeatConfig::default()
        };
        let mut population = Population::new(2, config, 4).unwrap();
        population.set_fitness(vec![1.0; 30]);
        population.advance();
        assert!(population.species.len() > 2, "need several species");
        population.species[1].best_fitness = 100.0; // holds the all-time record
        population.species[0].best_fitness = 50.0;
        population.species[0].stagnation = 99;
        population.species[0].representative = population.genomes()[0].clone();
        population.species[1].representative = population.genomes()[1].clone();
        let champions_species = population.species[0].id;
        let mut fitness = vec![1.0; 30];
        fitness[0] = 5.0; // genome 0 (species 0) is this generation's champion
        population.set_fitness(fitness);
        population.advance();
        assert!(
            population.species.iter().any(|s| s.id == champions_species),
            "the champion's species must keep reproducing"
        );
    }

    #[test]
    fn default_config_forms_several_species_within_the_first_generations() {
        // Random initial weights put genomes ~0.3 apart; a threshold far
        // above that would hold the whole population in one species for
        // dozens of generations, with no protection for new structure.
        let probe = [0.3, -0.7, 0.1, 0.9, -0.2, 0.5, -0.4, 0.8, 0.0, -0.6];
        let mut population = Population::new(10, NeatConfig::default(), 21).unwrap();
        let mut scratch = Vec::new();
        let mut species_at_generation_14 = 0;
        for generation in 0..15 {
            let fitness: Vec<f64> = population
                .genomes()
                .iter()
                .map(|g| crate::Network::compile(g).activate(&probe, &mut scratch))
                .collect();
            population.set_fitness(fitness);
            let report = population.advance();
            if generation == 14 {
                species_at_generation_14 = report.species.len();
            }
        }
        assert!(
            species_at_generation_14 >= 3,
            "only {species_at_generation_14} species after 15 generations"
        );
    }

    #[test]
    fn default_config_keeps_the_species_count_stable() {
        // Fitness is a network's output on a fixed probe input: smooth,
        // deterministic, and it rewards drifting weights, so the
        // population keeps changing structure like a real run.
        let probe = [0.3, -0.7, 0.1, 0.9, -0.2, 0.5, -0.4, 0.8, 0.0, -0.6];
        let config = NeatConfig::default();
        let target = config.target_species;
        let mut population = Population::new(10, config, 21).unwrap();
        let mut scratch = Vec::new();
        for generation in 0..200 {
            let fitness: Vec<f64> = population
                .genomes()
                .iter()
                .map(|g| crate::Network::compile(g).activate(&probe, &mut scratch))
                .collect();
            population.set_fitness(fitness);
            let report = population.advance();
            if generation >= 100 {
                let count = report.species.len();
                assert!(
                    (2..=3 * target).contains(&count),
                    "generation {generation}: {count} species (target {target})"
                );
            }
        }
    }

    fn run_generations(population: &mut Population, from: u32, to: u32) {
        for generation in from..to {
            let fitness = (0..30)
                .map(|i| f64::from((i * 7 + generation) % 11))
                .collect();
            population.set_fitness(fitness);
            population.advance();
        }
    }

    #[test]
    fn a_warm_start_keeps_the_genomes_but_restarts_the_counters() {
        let mut pop = Population::new(2, tiny_config(), 5).unwrap();
        run_generations(&mut pop, 0, 3);
        let genomes_before = pop.genomes().to_vec();
        let warm = Population::warm_start(pop.snapshot(), tiny_config(), 99).unwrap();
        assert_eq!(warm.generation(), 0);
        assert!(warm.best().is_none());
        assert_eq!(warm.genomes(), &genomes_before[..]);
    }

    #[test]
    fn a_warm_start_with_another_population_size_is_refused() {
        let pop = Population::new(2, tiny_config(), 1).unwrap();
        let bigger = NeatConfig {
            population_size: 31,
            ..NeatConfig::default()
        };
        assert!(matches!(
            Population::warm_start(pop.snapshot(), bigger, 1),
            Err(NeatError::InvalidConfig(_))
        ));
    }

    #[test]
    fn a_warm_start_is_deterministic_in_its_seed_and_differs_between_seeds() {
        let mut pop = Population::new(2, tiny_config(), 5).unwrap();
        run_generations(&mut pop, 0, 3);
        let state = pop.snapshot();
        let advance = |seed| {
            let mut warm = Population::warm_start(state.clone(), tiny_config(), seed).unwrap();
            run_generations(&mut warm, 0, 2);
            serde_json::to_string(&warm.snapshot()).unwrap()
        };
        assert_eq!(advance(7), advance(7));
        assert_ne!(advance(7), advance(8));
    }

    #[test]
    fn a_restored_snapshot_continues_exactly_like_the_original() {
        let mut straight = Population::new(2, tiny_config(), 5).unwrap();
        run_generations(&mut straight, 0, 8);

        let mut first_half = Population::new(2, tiny_config(), 5).unwrap();
        run_generations(&mut first_half, 0, 4);
        // Through JSON, as a checkpoint file would be.
        let json = serde_json::to_string(&first_half.snapshot()).unwrap();
        let mut resumed = Population::restore(serde_json::from_str(&json).unwrap()).unwrap();
        assert_eq!(resumed.generation(), 4);
        run_generations(&mut resumed, 4, 8);

        assert_eq!(
            serde_json::to_string(&resumed.snapshot()).unwrap(),
            serde_json::to_string(&straight.snapshot()).unwrap()
        );
        assert_eq!(
            serde_json::to_string(resumed.genomes()).unwrap(),
            serde_json::to_string(straight.genomes()).unwrap()
        );
    }

    #[test]
    fn restore_rejects_a_snapshot_that_does_not_fit_its_config() {
        let population = Population::new(2, tiny_config(), 1).unwrap();
        let mut value = serde_json::to_value(population.snapshot()).unwrap();
        value["genomes"].as_array_mut().unwrap().pop();
        let state = serde_json::from_value(value).unwrap();
        assert!(Population::restore(state).is_err());
    }

    #[test]
    #[should_panic(expected = "fitness is pending")]
    fn snapshotting_mid_generation_panics() {
        let mut population = Population::new(2, tiny_config(), 1).unwrap();
        population.set_fitness(vec![0.0; 30]);
        let _ = population.snapshot();
    }

    #[test]
    fn same_seed_and_fitness_give_identical_populations() {
        let run = || {
            let mut population = Population::new(2, tiny_config(), 42).unwrap();
            for _ in 0..8 {
                let fitness = (0..30).map(|i| f64::from((i * 7) % 11)).collect();
                population.set_fitness(fitness);
                population.advance();
            }
            serde_json::to_string(population.genomes()).unwrap()
        };
        assert_eq!(run(), run());
    }
}
