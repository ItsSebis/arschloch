//! Helpers shared by the NEAT integration tests.

// Each test crate that includes this module uses only some helpers.
#![allow(dead_code)]

use neat::{ConnectionGene, Genome, NodeGene, NodeKind};
use sim::FEATURE_COUNT;

/// A genome scoring `sum(weight * feature)` for the given
/// `(feature index, weight)` pairs, wired straight to the output.
pub fn linear_genome(weights: &[(usize, f64)]) -> Genome {
    let id = |n: usize| u32::try_from(n).expect("small ids");
    let mut nodes: Vec<NodeGene> = (0..FEATURE_COUNT)
        .map(|n| NodeGene {
            id: id(n),
            kind: NodeKind::Input,
        })
        .collect();
    nodes.push(NodeGene {
        id: id(FEATURE_COUNT),
        kind: NodeKind::Bias,
    });
    nodes.push(NodeGene {
        id: id(FEATURE_COUNT + 1),
        kind: NodeKind::Output,
    });
    let connections = weights
        .iter()
        .enumerate()
        .map(|(innovation, &(feature, weight))| ConnectionGene {
            innovation: id(innovation),
            from: id(feature),
            to: id(FEATURE_COUNT + 1),
            weight,
            enabled: true,
        })
        .collect();
    Genome::from_parts(FEATURE_COUNT, nodes, connections).expect("a valid linear genome")
}

/// `LowestLegal` expressed as a network: never pass when a play exists
/// (`is_pass`, feature 0), prefer small combos (`combo_size`, 1), then
/// the weakest card (`top_strength`, 2). The weights keep every score
/// well inside tanh's non-saturated range (size steps 0.25 outweigh the
/// whole strength range 0.2).
pub fn lowest_legal_genome() -> Genome {
    linear_genome(&[(0, -3.0), (1, -2.0), (2, -0.2)])
}
