//! A genome compiled into a fast, immutable feedforward evaluator.
//!
//! Inputs are copied through, the bias node is always `1.0`, and every
//! hidden and output node computes `tanh(sum of weight * source)` over
//! its enabled incoming connections (`score` exposes the output's sum
//! before the `tanh`). `Network` holds no mutable state, so
//! one instance can be shared across threads; callers supply the scratch
//! buffer.

use crate::genome::{Genome, NodeKind};

#[derive(Debug, Clone, PartialEq)]
pub struct Network {
    num_inputs: usize,
    node_count: usize,
    bias_index: usize,
    output_index: usize,
    /// Non-input nodes in an order where every source is computed before
    /// its targets: `(node index, [(source index, weight)])`.
    steps: Vec<(usize, Vec<(usize, f64)>)>,
}

impl Network {
    /// Compiles the enabled connections of `genome`.
    ///
    /// # Panics
    ///
    /// Panics if the enabled connections contain a cycle, which a valid
    /// `Genome` cannot (see its module docs).
    #[must_use]
    pub fn compile(genome: &Genome) -> Self {
        let nodes = genome.nodes();
        let index_of = |id: u32| {
            nodes
                .binary_search_by_key(&id, |n| n.id)
                .expect("validated genomes only reference existing nodes")
        };
        let mut incoming: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nodes.len()];
        for connection in genome.connections().iter().filter(|c| c.enabled) {
            incoming[index_of(connection.to)].push((index_of(connection.from), connection.weight));
        }

        // Kahn's algorithm over the enabled connections.
        let mut pending: Vec<usize> = incoming.iter().map(Vec::len).collect();
        let mut ready: Vec<usize> = (0..nodes.len()).filter(|&i| pending[i] == 0).collect();
        let mut steps = Vec::new();
        let mut visited = 0;
        while let Some(index) = ready.pop() {
            visited += 1;
            if !matches!(nodes[index].kind, NodeKind::Input | NodeKind::Bias) {
                steps.push((index, std::mem::take(&mut incoming[index])));
            }
            for (target, sources) in incoming.iter().enumerate() {
                let edges = sources.iter().filter(|(s, _)| *s == index).count();
                if edges > 0 {
                    pending[target] -= edges;
                    if pending[target] == 0 {
                        ready.push(target);
                    }
                }
            }
        }
        assert_eq!(visited, nodes.len(), "enabled connections must be acyclic");

        Self {
            num_inputs: genome.num_inputs(),
            node_count: nodes.len(),
            bias_index: genome.num_inputs(),
            output_index: genome.num_inputs() + 1,
            steps,
        }
    }

    #[must_use]
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// Evaluates the network: the output node's `tanh` activation, in
    /// `(-1, 1)`. `scratch` is reused across calls to avoid allocating;
    /// its prior contents are irrelevant.
    ///
    /// # Panics
    ///
    /// Panics if `inputs.len()` is not `num_inputs()`.
    pub fn activate(&self, inputs: &[f64], scratch: &mut Vec<f64>) -> f64 {
        self.evaluate(inputs, scratch).tanh()
    }

    /// The output node's weighted sum *before* `tanh`: the same ordering
    /// as `activate`, but it never saturates. Compare candidates by this,
    /// not by `activate`: `tanh(x)` is exactly `1.0` in `f64` for `x`
    /// above about 18.7, so distinct large scores would tie.
    ///
    /// # Panics
    ///
    /// Panics if `inputs.len()` is not `num_inputs()`.
    pub fn score(&self, inputs: &[f64], scratch: &mut Vec<f64>) -> f64 {
        self.evaluate(inputs, scratch)
    }

    /// Runs every node and returns the output node's pre-activation sum
    /// (hidden nodes still squash with `tanh`).
    fn evaluate(&self, inputs: &[f64], scratch: &mut Vec<f64>) -> f64 {
        assert_eq!(inputs.len(), self.num_inputs, "wrong number of inputs");
        scratch.clear();
        scratch.resize(self.node_count, 0.0);
        scratch[..self.num_inputs].copy_from_slice(inputs);
        scratch[self.bias_index] = 1.0;
        let mut output_sum = 0.0;
        for (target, sources) in &self.steps {
            let sum: f64 = sources
                .iter()
                .map(|&(source, weight)| scratch[source] * weight)
                .sum();
            scratch[*target] = sum.tanh();
            if *target == self.output_index {
                output_sum = sum;
            }
        }
        output_sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::test_support::minimal;
    use crate::genome::{ConnectionGene, NodeGene};

    fn gene(innovation: u32, from: u32, to: u32, weight: f64, enabled: bool) -> ConnectionGene {
        ConnectionGene {
            innovation,
            from,
            to,
            weight,
            enabled,
        }
    }

    #[test]
    fn direct_connections_compute_tanh_of_the_weighted_sum() {
        let (genome, _) = minimal(2, 1);
        let network = Network::compile(&genome);
        let w: Vec<f64> = genome.connections().iter().map(|c| c.weight).collect();
        let expected = (0.5 * w[0] + -0.25 * w[1] + w[2]).tanh();
        let got = network.activate(&[0.5, -0.25], &mut Vec::new());
        assert!((got - expected).abs() < 1e-12, "{got} vs {expected}");
    }

    #[test]
    fn hidden_node_feeds_the_output() {
        // 1 input (id 0), bias (1), output (2), hidden (3):
        // 0 -> 3 (w 2), 3 -> 2 (w -1.5); no bias connection.
        let nodes = vec![
            NodeGene {
                id: 0,
                kind: NodeKind::Input,
            },
            NodeGene {
                id: 1,
                kind: NodeKind::Bias,
            },
            NodeGene {
                id: 2,
                kind: NodeKind::Output,
            },
            NodeGene {
                id: 3,
                kind: NodeKind::Hidden,
            },
        ];
        let genome = Genome::from_parts(
            1,
            nodes,
            vec![gene(0, 0, 3, 2.0, true), gene(1, 3, 2, -1.5, true)],
        )
        .unwrap();
        let network = Network::compile(&genome);
        let got = network.activate(&[0.3], &mut Vec::new());
        let expected = (-1.5 * (2.0_f64 * 0.3).tanh()).tanh();
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn disabled_connections_do_not_contribute() {
        let (mut genome, _) = minimal(1, 1);
        for connection in genome.connections_mut() {
            connection.enabled = false;
        }
        let network = Network::compile(&genome);
        assert!(network.activate(&[1.0], &mut Vec::new()).abs() < f64::EPSILON);
    }

    #[test]
    fn evaluation_is_independent_of_scratch_contents() {
        let (genome, _) = minimal(3, 5);
        let network = Network::compile(&genome);
        let mut dirty = vec![99.0; 40];
        let a = network.activate(&[0.1, 0.2, 0.3], &mut dirty);
        let b = network.activate(&[0.1, 0.2, 0.3], &mut Vec::new());
        assert!((a - b).abs() < f64::EPSILON);
    }

    #[test]
    fn compiles_every_mutated_genome() {
        use crate::genome::test_support::rng;
        use crate::mutation::mutate;
        let config = crate::NeatConfig {
            add_node_rate: 0.5,
            add_connection_rate: 0.5,
            toggle_enable_rate: 0.5,
            ..crate::NeatConfig::default()
        };
        let (mut genome, mut tracker) = minimal(4, 1);
        let mut random = rng(2);
        for _ in 0..100 {
            mutate(&mut genome, &mut tracker, &config, &mut random);
            let output =
                Network::compile(&genome).activate(&[0.1, -0.2, 0.3, 0.4], &mut Vec::new());
            assert!(output.is_finite() && output.abs() <= 1.0);
        }
    }

    fn one_input_network(weight: f64) -> Network {
        let nodes = vec![
            NodeGene {
                id: 0,
                kind: NodeKind::Input,
            },
            NodeGene {
                id: 1,
                kind: NodeKind::Bias,
            },
            NodeGene {
                id: 2,
                kind: NodeKind::Output,
            },
        ];
        Network::compile(&Genome::from_parts(1, nodes, vec![gene(0, 0, 2, weight, true)]).unwrap())
    }

    #[test]
    fn score_is_the_unsquashed_output_and_activate_squashes_it() {
        let network = one_input_network(0.8);
        let score = network.score(&[0.5], &mut Vec::new());
        assert!((score - 0.4).abs() < 1e-12, "{score}");
        let activated = network.activate(&[0.5], &mut Vec::new());
        assert!((activated - score.tanh()).abs() < 1e-12);
    }

    #[test]
    fn score_keeps_ordering_where_activate_saturates() {
        // tanh(40) and tanh(41) are both exactly 1.0 in f64, which would
        // make a strategy comparing activations see a tie.
        let (low, high) = (one_input_network(40.0), one_input_network(41.0));
        let mut scratch = Vec::new();
        assert!(
            (low.activate(&[1.0], &mut scratch) - high.activate(&[1.0], &mut scratch)).abs()
                < f64::EPSILON
        );
        assert!(high.score(&[1.0], &mut scratch) > low.score(&[1.0], &mut scratch));
    }

    #[test]
    fn network_can_be_shared_across_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Network>();
    }

    #[test]
    #[should_panic(expected = "wrong number of inputs")]
    fn wrong_input_length_panics() {
        let (genome, _) = minimal(2, 1);
        Network::compile(&genome).activate(&[1.0], &mut Vec::new());
    }
}
