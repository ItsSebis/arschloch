//! End-to-end check that the generational loop actually learns: XOR is
//! the classic NEAT benchmark because it needs a hidden node, so solving
//! it exercises add-node, crossover, speciation and the evaluator
//! together.

use neat::{NeatConfig, Network, Population};

const CASES: [([f64; 2], f64); 4] = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
];

/// `4 - total squared error`, mapping the network's `tanh` output from
/// `(-1, 1)` onto `(0, 1)`; 4.0 is perfect.
fn xor_fitness(network: &Network) -> f64 {
    let mut scratch = Vec::new();
    let error: f64 = CASES
        .iter()
        .map(|(inputs, target)| {
            let probability = f64::midpoint(network.activate(inputs, &mut scratch), 1.0);
            (probability - target).powi(2)
        })
        .sum();
    4.0 - error
}

fn solves_xor(network: &Network) -> bool {
    let mut scratch = Vec::new();
    CASES
        .iter()
        .all(|(inputs, target)| (network.activate(inputs, &mut scratch) > 0.0) == (*target > 0.5))
}

/// Evolves until a genome classifies all four cases correctly, returning
/// the generation it happened in.
fn generations_to_solve(seed: u64, limit: u32) -> Option<u32> {
    let mut population = Population::new(2, NeatConfig::default(), seed).unwrap();
    for generation in 0..limit {
        let networks: Vec<Network> = population.genomes().iter().map(Network::compile).collect();
        if networks.iter().any(solves_xor) {
            return Some(generation);
        }
        population.set_fitness(networks.iter().map(xor_fitness).collect());
        population.advance();
    }
    None
}

#[test]
fn evolves_a_network_that_solves_xor() {
    // Seeds 0..30 all solve within 30 generations; the bound here leaves
    // ample headroom so only a real regression in the loop trips it.
    for seed in 0..5 {
        let solved_at = generations_to_solve(seed, 100);
        assert!(
            solved_at.is_some(),
            "seed {seed} never solved XOR in 100 generations"
        );
    }
}

#[test]
fn xor_runs_are_reproducible() {
    assert_eq!(generations_to_solve(3, 200), generations_to_solve(3, 200));
}
