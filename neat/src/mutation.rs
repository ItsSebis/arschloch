//! Mutation operators. Each keeps the genome invariants (see
//! `crate::genome`): the structural ones consult the innovation tracker
//! so identical changes made independently get identical gene ids.

use rand::seq::{IndexedMutRandom, IndexedRandom};
use rand::{Rng, RngExt};

use crate::config::NeatConfig;
use crate::genome::{ConnectionGene, Genome, NodeKind};
use crate::innovation::InnovationTracker;

/// Applies every operator, each with its own configured probability.
pub fn mutate<R: Rng + ?Sized>(
    genome: &mut Genome,
    tracker: &mut InnovationTracker,
    config: &NeatConfig,
    rng: &mut R,
) {
    if rng.random_bool(config.weight_mutate_rate) {
        mutate_weights(genome, config, rng);
    }
    if rng.random_bool(config.add_connection_rate) {
        add_connection(genome, tracker, config, rng);
    }
    if rng.random_bool(config.add_node_rate) {
        add_node(genome, tracker, config, rng);
    }
    if rng.random_bool(config.toggle_enable_rate) {
        toggle_enable(genome, rng);
    }
}

/// Perturbs each weight slightly, or (rarely) replaces it outright.
pub fn mutate_weights<R: Rng + ?Sized>(genome: &mut Genome, config: &NeatConfig, rng: &mut R) {
    for connection in genome.connections_mut() {
        let weight = if rng.random_bool(config.weight_perturb_rate) {
            connection.weight
                + rng.random_range(-config.weight_perturb_power..=config.weight_perturb_power)
        } else {
            rng.random_range(-config.weight_init_range..=config.weight_init_range)
        };
        connection.weight = weight.clamp(-config.weight_limit, config.weight_limit);
    }
}

/// Connects two previously unconnected nodes with a random weight.
/// Returns whether a connection was added; a genome with no free
/// feedforward pair left (or unlucky draws) is left unchanged.
pub fn add_connection<R: Rng + ?Sized>(
    genome: &mut Genome,
    tracker: &mut InnovationTracker,
    config: &NeatConfig,
    rng: &mut R,
) -> bool {
    let sources: Vec<u32> = genome
        .nodes()
        .iter()
        .filter(|n| n.kind != NodeKind::Output)
        .map(|n| n.id)
        .collect();
    let targets: Vec<u32> = genome
        .nodes()
        .iter()
        .filter(|n| matches!(n.kind, NodeKind::Hidden | NodeKind::Output))
        .map(|n| n.id)
        .collect();
    for _ in 0..config.add_connection_attempts {
        let (Some(&from), Some(&to)) = (sources.choose(rng), targets.choose(rng)) else {
            return false;
        };
        if genome.has_connection(from, to) || genome.would_create_cycle(from, to) {
            continue;
        }
        genome.insert_connection(ConnectionGene {
            innovation: tracker.connection(from, to),
            from,
            to,
            weight: rng.random_range(-config.weight_init_range..=config.weight_init_range),
            enabled: true,
        });
        return true;
    }
    false
}

/// Splits a random enabled connection with a new hidden node: the old
/// connection is disabled, the incoming half gets weight 1 and the
/// outgoing half keeps the old weight, so behavior is initially almost
/// unchanged. Returns whether a split happened (no enabled connection,
/// or the node already exists in this genome from an earlier split of
/// the same connection that was later re-enabled, means no change).
pub fn add_node<R: Rng + ?Sized>(
    genome: &mut Genome,
    tracker: &mut InnovationTracker,
    _config: &NeatConfig,
    rng: &mut R,
) -> bool {
    let enabled: Vec<ConnectionGene> = genome
        .connections()
        .iter()
        .filter(|c| c.enabled)
        .copied()
        .collect();
    let Some(&old) = enabled.choose(rng) else {
        return false;
    };
    let split = tracker.split(old.innovation, old.from, old.to);
    if genome.node(split.node_id).is_some() {
        return false;
    }
    for connection in genome.connections_mut() {
        if connection.innovation == old.innovation {
            connection.enabled = false;
        }
    }
    genome.insert_hidden_node(split.node_id);
    genome.insert_connection(ConnectionGene {
        innovation: split.in_innovation,
        from: old.from,
        to: split.node_id,
        weight: 1.0,
        enabled: true,
    });
    genome.insert_connection(ConnectionGene {
        innovation: split.out_innovation,
        from: split.node_id,
        to: old.to,
        weight: old.weight,
        enabled: true,
    });
    true
}

/// Flips one random connection between enabled and disabled.
pub fn toggle_enable<R: Rng + ?Sized>(genome: &mut Genome, rng: &mut R) {
    if let Some(connection) = genome.connections_mut().choose_mut(rng) {
        connection.enabled = !connection.enabled;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::test_support::{minimal, rng};

    fn config() -> NeatConfig {
        NeatConfig::default()
    }

    #[test]
    fn add_node_splits_a_connection_and_preserves_the_path() {
        let (mut genome, mut tracker) = minimal(2, 1);
        let before = genome.connections().len();
        assert!(add_node(&mut genome, &mut tracker, &config(), &mut rng(5)));
        assert_eq!(genome.hidden_count(), 1);
        assert_eq!(genome.connections().len(), before + 2);
        let disabled: Vec<_> = genome.connections().iter().filter(|c| !c.enabled).collect();
        assert_eq!(disabled.len(), 1);
        let hidden = genome.nodes().last().unwrap().id;
        let into = genome
            .connections()
            .iter()
            .find(|c| c.to == hidden)
            .unwrap();
        let out = genome
            .connections()
            .iter()
            .find(|c| c.from == hidden)
            .unwrap();
        assert!((into.weight - 1.0).abs() < f64::EPSILON);
        assert!((out.weight - disabled[0].weight).abs() < f64::EPSILON);
        assert_eq!((into.from, out.to), (disabled[0].from, disabled[0].to));
    }

    #[test]
    fn identical_splits_in_two_genomes_share_ids() {
        let (mut a, mut tracker) = minimal(1, 1);
        let mut b = a.clone();
        // One input, one bias: seed the same pick in both genomes.
        add_node(&mut a, &mut tracker, &config(), &mut rng(9));
        add_node(&mut b, &mut tracker, &config(), &mut rng(9));
        assert_eq!(a.nodes(), b.nodes());
        let innovations = |g: &Genome| {
            g.connections()
                .iter()
                .map(|c| c.innovation)
                .collect::<Vec<_>>()
        };
        assert_eq!(innovations(&a), innovations(&b));
    }

    #[test]
    fn add_connection_never_breaks_genome_invariants() {
        let (mut genome, mut tracker) = minimal(3, 1);
        let cfg = config();
        let mut random = rng(11);
        for _ in 0..200 {
            add_node(&mut genome, &mut tracker, &cfg, &mut random);
            add_connection(&mut genome, &mut tracker, &cfg, &mut random);
            // from_parts re-validates acyclicity, kinds, uniqueness.
            Genome::from_parts(
                genome.num_inputs(),
                genome.nodes().to_vec(),
                genome.connections().to_vec(),
            )
            .expect("mutation preserved every invariant");
        }
        assert!(genome.hidden_count() > 5);
    }

    #[test]
    fn add_connection_gives_up_on_a_saturated_genome() {
        let (mut genome, mut tracker) = minimal(1, 1);
        // Inputs are already wired to the only target (the output).
        assert!(!add_connection(
            &mut genome,
            &mut tracker,
            &config(),
            &mut rng(1)
        ));
    }

    #[test]
    fn toggle_flips_exactly_one_connection() {
        let (mut genome, _) = minimal(2, 1);
        toggle_enable(&mut genome, &mut rng(3));
        assert_eq!(
            genome.enabled_connection_count(),
            genome.connections().len() - 1
        );
    }

    #[test]
    fn weight_mutation_stays_within_the_limit() {
        let (mut genome, _) = minimal(2, 1);
        let cfg = NeatConfig {
            weight_perturb_power: 100.0,
            weight_limit: 2.0,
            ..config()
        };
        let mut random = rng(4);
        for _ in 0..50 {
            mutate_weights(&mut genome, &cfg, &mut random);
        }
        assert!(genome.connections().iter().all(|c| c.weight.abs() <= 2.0));
    }

    #[test]
    fn mutate_with_all_rates_zero_changes_nothing() {
        let (mut genome, mut tracker) = minimal(2, 1);
        let original = genome.clone();
        let cfg = NeatConfig {
            weight_mutate_rate: 0.0,
            add_connection_rate: 0.0,
            add_node_rate: 0.0,
            toggle_enable_rate: 0.0,
            ..config()
        };
        mutate(&mut genome, &mut tracker, &cfg, &mut rng(1));
        assert_eq!(genome, original);
    }
}
