//! Tunable parameters of a NEAT run, with the defaults from the original
//! NEAT paper's XOR experiments where one exists.

use serde::{Deserialize, Serialize};

use crate::error::NeatError;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeatConfig {
    /// Genomes per generation (kept constant).
    pub population_size: usize,

    /// New and replacement weights are drawn uniformly from
    /// `[-weight_init_range, weight_init_range]`.
    pub weight_init_range: f64,
    /// Weights are clamped to `[-weight_limit, weight_limit]`.
    pub weight_limit: f64,
    /// Chance per offspring that its weights are mutated at all.
    pub weight_mutate_rate: f64,
    /// When weights mutate: chance per weight of a small perturbation
    /// (otherwise the weight is replaced outright).
    pub weight_perturb_rate: f64,
    /// A perturbation is uniform in `[-power, power]`.
    pub weight_perturb_power: f64,
    pub add_connection_rate: f64,
    /// Random endpoint pairs tried before giving up on an
    /// add-connection mutation (most pairs are rejected: already
    /// connected, or would close a cycle).
    pub add_connection_attempts: usize,
    pub add_node_rate: f64,
    pub toggle_enable_rate: f64,

    /// Chance an offspring is produced by crossover instead of cloning.
    pub crossover_rate: f64,
    /// Chance a gene disabled in either parent stays disabled in the child.
    pub disabled_gene_rate: f64,

    /// Compatibility distance coefficients (excess, disjoint, mean
    /// weight difference).
    pub excess_coefficient: f64,
    pub disjoint_coefficient: f64,
    pub weight_difference_coefficient: f64,
    /// Starting speciation threshold; adapted to hold `target_species`.
    /// Distances here are small (random initial weights in `[-1, 1]` put
    /// two genomes about 0.3 apart), so a threshold of 3.0 would keep a
    /// whole run in one species for dozens of generations.
    pub compatibility_threshold: f64,
    pub min_compatibility_threshold: f64,
    /// How far the threshold moves per generation toward `target_species`.
    /// Must be small against typical distances (a step of 0.3 made the
    /// species count swing between ~5 and ~60 every generation).
    pub threshold_step: f64,
    pub target_species: usize,

    /// Generations without improvement before a species stops
    /// reproducing (the species holding the best fitness is exempt).
    pub stagnation_limit: u32,
    /// Fraction of each species (best first) that may become parents.
    pub survival_fraction: f64,
    /// Species at least this large copy their best member unchanged.
    pub elitism_min_species_size: usize,
}

impl Default for NeatConfig {
    fn default() -> Self {
        Self {
            population_size: 150,
            weight_init_range: 1.0,
            weight_limit: 8.0,
            weight_mutate_rate: 0.8,
            weight_perturb_rate: 0.9,
            weight_perturb_power: 0.5,
            add_connection_rate: 0.1,
            add_connection_attempts: 20,
            add_node_rate: 0.05,
            toggle_enable_rate: 0.01,
            crossover_rate: 0.75,
            disabled_gene_rate: 0.75,
            excess_coefficient: 1.0,
            disjoint_coefficient: 1.0,
            weight_difference_coefficient: 0.4,
            compatibility_threshold: 0.5,
            min_compatibility_threshold: 0.1,
            threshold_step: 0.05,
            target_species: 8,
            stagnation_limit: 15,
            survival_fraction: 0.2,
            elitism_min_species_size: 5,
        }
    }
}

impl NeatConfig {
    /// # Errors
    ///
    /// Returns `NeatError::InvalidConfig` naming the first offending
    /// field.
    pub fn validate(&self) -> Result<(), NeatError> {
        let rates = [
            ("weight_mutate_rate", self.weight_mutate_rate),
            ("weight_perturb_rate", self.weight_perturb_rate),
            ("add_connection_rate", self.add_connection_rate),
            ("add_node_rate", self.add_node_rate),
            ("toggle_enable_rate", self.toggle_enable_rate),
            ("crossover_rate", self.crossover_rate),
            ("disabled_gene_rate", self.disabled_gene_rate),
            ("survival_fraction", self.survival_fraction),
        ];
        for (name, value) in rates {
            if !(0.0..=1.0).contains(&value) {
                return Err(invalid(name, "must be within [0, 1]"));
            }
        }
        let positives = [
            ("weight_init_range", self.weight_init_range),
            ("weight_limit", self.weight_limit),
            ("weight_perturb_power", self.weight_perturb_power),
            ("compatibility_threshold", self.compatibility_threshold),
            (
                "min_compatibility_threshold",
                self.min_compatibility_threshold,
            ),
            ("threshold_step", self.threshold_step),
        ];
        for (name, value) in positives {
            if !(value.is_finite() && value > 0.0) {
                return Err(invalid(name, "must be a positive number"));
            }
        }
        if self.population_size < 2 {
            return Err(invalid("population_size", "must be at least 2"));
        }
        if self.target_species == 0 {
            return Err(invalid("target_species", "must be at least 1"));
        }
        if self.add_connection_attempts == 0 {
            return Err(invalid("add_connection_attempts", "must be at least 1"));
        }
        Ok(())
    }
}

fn invalid(field: &str, reason: &str) -> NeatError {
    NeatError::InvalidConfig(format!("{field} {reason}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        assert_eq!(NeatConfig::default().validate(), Ok(()));
    }

    #[test]
    fn out_of_range_rate_is_rejected_by_name() {
        let config = NeatConfig {
            crossover_rate: 1.5,
            ..NeatConfig::default()
        };
        let error = config.validate().unwrap_err();
        assert!(error.to_string().contains("crossover_rate"), "{error}");
    }

    #[test]
    fn tiny_population_is_rejected() {
        let config = NeatConfig {
            population_size: 1,
            ..NeatConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn nan_parameters_are_rejected() {
        let config = NeatConfig {
            weight_limit: f64::NAN,
            ..NeatConfig::default()
        };
        assert!(config.validate().is_err());
    }
}
