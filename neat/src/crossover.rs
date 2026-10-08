//! Crossover: aligns two genomes by innovation number.
//!
//! The child's structure is exactly the fitter parent's (matching,
//! disjoint and excess genes all come from it), so it inherits that
//! parent's acyclicity; only the weights and enabled flags of *matching*
//! genes are mixed with the other parent.

use rand::{Rng, RngExt};

use crate::config::NeatConfig;
use crate::genome::Genome;

/// `fitter` is the parent with the higher fitness (the caller breaks
/// ties however it likes). Both parents must have the same input count.
///
/// # Panics
///
/// Panics if the parents have different input counts.
pub fn crossover<R: Rng + ?Sized>(
    fitter: &Genome,
    other: &Genome,
    config: &NeatConfig,
    rng: &mut R,
) -> Genome {
    assert_eq!(
        fitter.num_inputs(),
        other.num_inputs(),
        "crossover needs genomes over the same inputs"
    );
    let connections = fitter
        .connections()
        .iter()
        .map(|gene| {
            let mut child = *gene;
            if let Some(matching) = other.connection_by_innovation(gene.innovation) {
                if rng.random_bool(0.5) {
                    child.weight = matching.weight;
                }
                child.enabled = if gene.enabled && matching.enabled {
                    true
                } else {
                    !rng.random_bool(config.disabled_gene_rate)
                };
            }
            child
        })
        .collect();
    Genome::from_parts(fitter.num_inputs(), fitter.nodes().to_vec(), connections)
        .expect("a child that copies the fitter parent's structure keeps every invariant")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::test_support::{minimal, rng};
    use crate::mutation::{add_connection, add_node, mutate_weights};

    fn diverged_pair() -> (Genome, Genome, NeatConfig) {
        let config = NeatConfig::default();
        let (mut a, mut tracker) = minimal(3, 1);
        let mut b = a.clone();
        let mut random = rng(2);
        // Give the shared genes different weights in the two parents.
        mutate_weights(&mut b, &config, &mut random);
        for _ in 0..6 {
            add_node(&mut a, &mut tracker, &config, &mut random);
            add_connection(&mut b, &mut tracker, &config, &mut random);
            add_node(&mut b, &mut tracker, &config, &mut random);
        }
        (a, b, config)
    }

    #[test]
    fn child_has_exactly_the_fitter_parents_structure() {
        let (a, b, config) = diverged_pair();
        let child = crossover(&a, &b, &config, &mut rng(3));
        let innovations = |g: &Genome| {
            g.connections()
                .iter()
                .map(|c| c.innovation)
                .collect::<Vec<_>>()
        };
        assert_eq!(innovations(&child), innovations(&a));
        assert_eq!(child.nodes(), a.nodes());
    }

    #[test]
    fn matching_genes_take_weights_from_either_parent() {
        let (a, b, config) = diverged_pair();
        let mut from_a = 0;
        let mut from_b = 0;
        for seed in 0..40 {
            let child = crossover(&a, &b, &config, &mut rng(seed));
            for gene in child.connections() {
                let (Some(ga), Some(gb)) = (
                    a.connection_by_innovation(gene.innovation),
                    b.connection_by_innovation(gene.innovation),
                ) else {
                    continue;
                };
                if (ga.weight - gb.weight).abs() < f64::EPSILON {
                    continue;
                }
                if (gene.weight - ga.weight).abs() < f64::EPSILON {
                    from_a += 1;
                } else if (gene.weight - gb.weight).abs() < f64::EPSILON {
                    from_b += 1;
                }
            }
        }
        assert!(from_a > 20 && from_b > 20, "a: {from_a}, b: {from_b}");
    }

    #[test]
    fn a_gene_disabled_in_either_parent_is_usually_disabled_in_the_child() {
        let config = NeatConfig::default();
        let (a, _) = minimal(2, 1);
        let mut b = a.clone();
        b.connections_mut()[0].enabled = false;
        let trials = 400;
        let disabled = (0..trials)
            .filter(|&seed| !crossover(&a, &b, &config, &mut rng(seed)).connections()[0].enabled)
            .count();
        // Expect ~75% of 400 = 300.
        assert!((250..=350).contains(&disabled), "{disabled}");
    }

    #[test]
    fn crossing_a_genome_with_itself_reproduces_it() {
        let config = NeatConfig::default();
        let (a, _) = minimal(2, 1);
        assert_eq!(crossover(&a, &a, &config, &mut rng(1)), a);
    }

    #[test]
    #[should_panic(expected = "same inputs")]
    fn mismatched_input_counts_panic() {
        let config = NeatConfig::default();
        let (a, _) = minimal(2, 1);
        let (b, _) = minimal(3, 1);
        let _ = crossover(&a, &b, &config, &mut rng(1));
    }
}
