//! Speciation: grouping genomes by structural similarity so a new
//! structure competes mainly with its own kind while it is still being
//! tuned, instead of being outcompeted by established topologies.

use rand::seq::IndexedRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::config::NeatConfig;
use crate::genome::Genome;

/// `c1 * excess / N + c2 * disjoint / N + c3 * mean weight difference
/// of matching genes`, with `N` the larger genome's gene count (at least
/// 1). Symmetric; zero between identical genomes.
#[must_use]
pub fn compatibility_distance(left: &Genome, right: &Genome, config: &NeatConfig) -> f64 {
    let (left_genes, right_genes) = (left.connections(), right.connections());
    let left_max = left_genes.last().map(|c| c.innovation);
    let right_max = right_genes.last().map(|c| c.innovation);
    let (mut matching, mut weight_difference, mut disjoint, mut excess) = (0u32, 0.0, 0u32, 0u32);

    let (mut l, mut r) = (0, 0);
    while l < left_genes.len() || r < right_genes.len() {
        match (left_genes.get(l), right_genes.get(r)) {
            (Some(lg), Some(rg)) if lg.innovation == rg.innovation => {
                matching += 1;
                weight_difference += (lg.weight - rg.weight).abs();
                l += 1;
                r += 1;
            }
            (Some(lg), rg) if rg.is_none_or(|rg| lg.innovation < rg.innovation) => {
                count_unmatched(lg.innovation, right_max, &mut disjoint, &mut excess);
                l += 1;
            }
            (_, Some(rg)) => {
                count_unmatched(rg.innovation, left_max, &mut disjoint, &mut excess);
                r += 1;
            }
            (_, None) => unreachable!("loop condition"),
        }
    }

    let n = f64::from(
        u32::try_from(left_genes.len().max(right_genes.len()).max(1)).unwrap_or(u32::MAX),
    );
    let mean_weight_difference = if matching == 0 {
        0.0
    } else {
        weight_difference / f64::from(matching)
    };
    config.excess_coefficient * f64::from(excess) / n
        + config.disjoint_coefficient * f64::from(disjoint) / n
        + config.weight_difference_coefficient * mean_weight_difference
}

/// A gene present in only one genome is *excess* if it lies beyond the
/// other genome's highest innovation, otherwise *disjoint*.
fn count_unmatched(innovation: u32, other_max: Option<u32>, disjoint: &mut u32, excess: &mut u32) {
    if other_max.is_some_and(|max| innovation < max) {
        *disjoint += 1;
    } else {
        *excess += 1;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Species {
    pub id: u32,
    pub representative: Genome,
    /// Indices into the population's genome list.
    pub members: Vec<usize>,
    pub best_fitness: f64,
    pub stagnation: u32,
    pub age: u32,
}

/// Public per-species numbers for one evaluated generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeciesStats {
    pub id: u32,
    pub size: usize,
    pub age: u32,
    /// Best member fitness this generation.
    pub generation_best: f64,
    pub mean_fitness: f64,
    /// Best fitness the species has ever reached.
    pub best_fitness: f64,
    /// Generations since `best_fitness` last improved.
    pub stagnation: u32,
}

/// Re-assigns every genome to the first species whose representative is
/// within `threshold`, founding a new species when none is. Empty species
/// are dropped, and each survivor's representative becomes a random
/// current member.
pub(crate) fn assign<R: Rng + ?Sized>(
    species: &mut Vec<Species>,
    genomes: &[Genome],
    threshold: f64,
    config: &NeatConfig,
    next_species_id: &mut u32,
    rng: &mut R,
) {
    for s in species.iter_mut() {
        s.members.clear();
    }
    for (index, genome) in genomes.iter().enumerate() {
        let home = species
            .iter()
            .position(|s| compatibility_distance(genome, &s.representative, config) < threshold);
        if let Some(position) = home {
            species[position].members.push(index);
        } else {
            species.push(Species {
                id: *next_species_id,
                representative: genome.clone(),
                members: vec![index],
                best_fitness: f64::NEG_INFINITY,
                stagnation: 0,
                age: 0,
            });
            *next_species_id += 1;
        }
    }
    species.retain(|s| !s.members.is_empty());
    for s in species.iter_mut() {
        let pick = *s.members.choose(rng).expect("members are non-empty");
        s.representative = genomes[pick].clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::test_support::{minimal, rng};
    use crate::mutation::{add_connection, add_node};

    #[test]
    fn identical_genomes_are_at_distance_zero() {
        let (genome, _) = minimal(3, 1);
        assert!(
            compatibility_distance(&genome, &genome, &NeatConfig::default()).abs() < f64::EPSILON
        );
    }

    #[test]
    fn weight_only_differences_scale_with_the_weight_coefficient() {
        let config = NeatConfig::default();
        let (a, _) = minimal(2, 1);
        let mut b = a.clone();
        for connection in b.connections_mut() {
            connection.weight = a.connections()[0].weight; // arbitrary but fixed
        }
        let expected_mean: f64 = a
            .connections()
            .iter()
            .map(|c| (c.weight - a.connections()[0].weight).abs())
            .sum::<f64>()
            / 3.0;
        let distance = compatibility_distance(&a, &b, &config);
        assert!((distance - config.weight_difference_coefficient * expected_mean).abs() < 1e-12);
    }

    #[test]
    fn structural_difference_counts_disjoint_and_excess_genes() {
        let config = NeatConfig {
            weight_difference_coefficient: 0.0,
            ..NeatConfig::default()
        };
        let (a, mut tracker) = minimal(1, 1);
        let mut b = a.clone();
        // One split adds one hidden node: the original gene is still
        // shared (disabled), and two new genes lie beyond a's last
        // innovation, so they are excess.
        assert!(add_node(&mut b, &mut tracker, &config, &mut rng(2)));
        let distance = compatibility_distance(&a, &b, &config);
        assert_eq!(b.connections().len(), 4);
        assert!((distance - 2.0 / 4.0).abs() < 1e-12, "{distance}");
    }

    #[test]
    fn distance_is_symmetric() {
        let config = NeatConfig::default();
        let (mut a, mut tracker) = minimal(3, 1);
        let mut b = a.clone();
        let mut random = rng(4);
        for _ in 0..10 {
            add_node(&mut a, &mut tracker, &config, &mut random);
            add_connection(&mut b, &mut tracker, &config, &mut random);
            add_node(&mut b, &mut tracker, &config, &mut random);
        }
        let forward = compatibility_distance(&a, &b, &config);
        let backward = compatibility_distance(&b, &a, &config);
        assert!((forward - backward).abs() < 1e-12);
        assert!(forward > 0.0);
    }

    #[test]
    fn assign_puts_similar_genomes_together_and_founds_new_species() {
        let config = NeatConfig {
            weight_difference_coefficient: 0.0,
            ..NeatConfig::default()
        };
        let (base, mut tracker) = minimal(2, 1);
        let mut diverged = base.clone();
        let mut random = rng(3);
        for _ in 0..15 {
            add_node(&mut diverged, &mut tracker, &config, &mut random);
        }
        let genomes = vec![base.clone(), base.clone(), diverged.clone(), base, diverged];
        let mut species = Vec::new();
        let mut next_id = 0;
        // 15 splits add 30 excess genes to a 33-gene genome: distance ~0.91.
        assign(
            &mut species,
            &genomes,
            0.5,
            &config,
            &mut next_id,
            &mut rng(1),
        );
        assert_eq!(species.len(), 2);
        assert_eq!(species[0].members, vec![0, 1, 3]);
        assert_eq!(species[1].members, vec![2, 4]);
        assert_eq!(next_id, 2);
    }

    #[test]
    fn assign_keeps_species_ids_and_drops_empty_ones() {
        let config = NeatConfig::default();
        let (base, _) = minimal(2, 1);
        let mut species = Vec::new();
        let mut next_id = 0;
        assign(
            &mut species,
            &[base.clone(), base.clone()],
            100.0,
            &config,
            &mut next_id,
            &mut rng(1),
        );
        let original_id = species[0].id;
        // A second species that nobody joins disappears.
        species.push(Species {
            id: 77,
            representative: base.clone(),
            members: vec![],
            best_fitness: 0.0,
            stagnation: 0,
            age: 0,
        });
        // Threshold 0 puts nobody in the old species... but base is
        // distance 0 from itself, which is not < 0, so a fresh one forms.
        assign(
            &mut species,
            &[base],
            0.0,
            &config,
            &mut next_id,
            &mut rng(1),
        );
        assert_eq!(species.len(), 1);
        assert_ne!(species[0].id, original_id);
        assert_eq!(species[0].id, 1);
    }
}
