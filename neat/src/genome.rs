//! The genotype: node genes and connection genes, plus the structural
//! invariants every genome in this crate satisfies.
//!
//! Invariants (checked by `Genome::from_parts`, and therefore by JSON
//! loading; maintained by construction everywhere else):
//! - node ids ascend; ids `0..num_inputs` are inputs, `num_inputs` is the
//!   bias, `num_inputs + 1` the single output, anything above is hidden;
//! - connections ascend by innovation number, no `(from, to)` pair
//!   repeats, no connection ends in an input/bias or starts at the output;
//! - weights are finite;
//! - the graph of *all* connections, enabled or not, is acyclic. Counting
//!   disabled genes means re-enabling one can never close a cycle, so
//!   every network compiled from a genome is feedforward.

use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};

use crate::config::NeatConfig;
use crate::error::NeatError;
use crate::innovation::InnovationTracker;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    Input,
    Bias,
    Output,
    Hidden,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeGene {
    pub id: u32,
    pub kind: NodeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConnectionGene {
    pub innovation: u32,
    pub from: u32,
    pub to: u32,
    pub weight: f64,
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "GenomeParts")]
pub struct Genome {
    num_inputs: usize,
    nodes: Vec<NodeGene>,
    connections: Vec<ConnectionGene>,
}

/// The raw shape of a genome in JSON, validated on the way in.
#[derive(Deserialize)]
struct GenomeParts {
    num_inputs: usize,
    nodes: Vec<NodeGene>,
    connections: Vec<ConnectionGene>,
}

impl TryFrom<GenomeParts> for Genome {
    type Error = NeatError;

    fn try_from(parts: GenomeParts) -> Result<Self, NeatError> {
        Self::from_parts(parts.num_inputs, parts.nodes, parts.connections)
    }
}

fn bad(reason: impl Into<String>) -> NeatError {
    NeatError::InvalidGenome(reason.into())
}

impl Genome {
    /// The starting topology: every input and the bias wired straight to
    /// the single output with random weights, no hidden nodes.
    ///
    /// # Panics
    ///
    /// Panics if `num_inputs` does not fit in a `u32` node id.
    pub fn minimal<R: Rng + ?Sized>(
        num_inputs: usize,
        tracker: &mut InnovationTracker,
        config: &NeatConfig,
        rng: &mut R,
    ) -> Self {
        let inputs = u32::try_from(num_inputs).expect("input count fits in a u32 node id");
        let (bias, output) = (inputs, inputs + 1);
        let mut nodes: Vec<NodeGene> = (0..inputs)
            .map(|id| NodeGene {
                id,
                kind: NodeKind::Input,
            })
            .collect();
        nodes.push(NodeGene {
            id: bias,
            kind: NodeKind::Bias,
        });
        nodes.push(NodeGene {
            id: output,
            kind: NodeKind::Output,
        });
        let connections = (0..=bias)
            .map(|from| ConnectionGene {
                innovation: tracker.connection(from, output),
                from,
                to: output,
                weight: rng.random_range(-config.weight_init_range..=config.weight_init_range),
                enabled: true,
            })
            .collect();
        Self::from_parts(num_inputs, nodes, connections)
            .expect("a fully connected input layer satisfies every genome invariant")
    }

    /// Builds a genome from raw genes, checking every invariant listed in
    /// the module docs. Connections are accepted in any order and sorted
    /// by innovation number.
    ///
    /// # Errors
    ///
    /// Returns `NeatError::InvalidGenome` naming the broken invariant.
    pub fn from_parts(
        num_inputs: usize,
        nodes: Vec<NodeGene>,
        mut connections: Vec<ConnectionGene>,
    ) -> Result<Self, NeatError> {
        connections.sort_by_key(|c| c.innovation);
        let genome = Self {
            num_inputs,
            nodes,
            connections,
        };
        genome.validate()?;
        Ok(genome)
    }

    fn validate(&self) -> Result<(), NeatError> {
        let fixed = self
            .num_inputs
            .checked_add(2)
            .filter(|&count| u32::try_from(count).is_ok())
            .ok_or_else(|| bad("num_inputs is too large for u32 node ids"))?;
        if self.nodes.len() < fixed {
            return Err(bad("missing input, bias or output nodes"));
        }
        for (index, node) in self.nodes.iter().enumerate() {
            let expected = match index {
                i if i < self.num_inputs => NodeKind::Input,
                i if i == self.num_inputs => NodeKind::Bias,
                i if i == self.num_inputs + 1 => NodeKind::Output,
                _ => NodeKind::Hidden,
            };
            if node.kind != expected {
                return Err(bad(format!(
                    "node {} should be {expected:?}, found {:?}",
                    node.id, node.kind
                )));
            }
            if index < fixed && usize::try_from(node.id) != Ok(index) {
                return Err(bad(format!(
                    "fixed node at index {index} has id {}",
                    node.id
                )));
            }
            if index > 0 && self.nodes[index - 1].id >= node.id {
                return Err(bad("node ids must strictly ascend"));
            }
        }
        for pair in self.connections.windows(2) {
            if pair[0].innovation == pair[1].innovation {
                return Err(bad(format!("duplicate innovation {}", pair[0].innovation)));
            }
        }
        for (i, connection) in self.connections.iter().enumerate() {
            if !connection.weight.is_finite() {
                return Err(bad(format!(
                    "innovation {} has a non-finite weight",
                    connection.innovation
                )));
            }
            let from = self
                .node(connection.from)
                .ok_or_else(|| bad(format!("connection from unknown node {}", connection.from)))?;
            let to = self
                .node(connection.to)
                .ok_or_else(|| bad(format!("connection to unknown node {}", connection.to)))?;
            if matches!(to.kind, NodeKind::Input | NodeKind::Bias) || from.kind == NodeKind::Output
            {
                return Err(bad(format!(
                    "connection {} -> {} runs against the feedforward direction",
                    from.id, to.id
                )));
            }
            if self.connections[..i]
                .iter()
                .any(|other| other.from == connection.from && other.to == connection.to)
            {
                return Err(bad(format!("{} -> {} appears twice", from.id, to.id)));
            }
        }
        if self.has_cycle() {
            return Err(bad("connections contain a cycle"));
        }
        Ok(())
    }

    #[must_use]
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    #[must_use]
    pub fn nodes(&self) -> &[NodeGene] {
        &self.nodes
    }

    /// Ascending by innovation number.
    #[must_use]
    pub fn connections(&self) -> &[ConnectionGene] {
        &self.connections
    }

    #[must_use]
    pub fn hidden_count(&self) -> usize {
        self.nodes.len() - self.num_inputs - 2
    }

    #[must_use]
    pub fn enabled_connection_count(&self) -> usize {
        self.connections.iter().filter(|c| c.enabled).count()
    }

    /// Id of the single output node.
    #[must_use]
    pub fn output_id(&self) -> u32 {
        self.nodes[self.num_inputs + 1].id
    }

    pub(crate) fn node(&self, id: u32) -> Option<&NodeGene> {
        self.nodes
            .binary_search_by_key(&id, |n| n.id)
            .ok()
            .map(|index| &self.nodes[index])
    }

    pub(crate) fn connection_by_innovation(&self, innovation: u32) -> Option<&ConnectionGene> {
        self.connections
            .binary_search_by_key(&innovation, |c| c.innovation)
            .ok()
            .map(|index| &self.connections[index])
    }

    pub(crate) fn has_connection(&self, from: u32, to: u32) -> bool {
        self.connections
            .iter()
            .any(|c| c.from == from && c.to == to)
    }

    pub(crate) fn connections_mut(&mut self) -> &mut [ConnectionGene] {
        &mut self.connections
    }

    /// Appends a hidden node. The caller guarantees `id` is above every
    /// existing id (the tracker allocates ids in ascending order, but a
    /// genome can be missing a lower hidden id another lineage has, so
    /// the sorted position is looked up rather than assumed).
    pub(crate) fn insert_hidden_node(&mut self, id: u32) {
        let position = self.nodes.partition_point(|n| n.id < id);
        self.nodes.insert(
            position,
            NodeGene {
                id,
                kind: NodeKind::Hidden,
            },
        );
    }

    pub(crate) fn insert_connection(&mut self, connection: ConnectionGene) {
        let position = self
            .connections
            .partition_point(|c| c.innovation < connection.innovation);
        self.connections.insert(position, connection);
    }

    /// Would adding `from -> to` close a cycle? True when `to` already
    /// reaches `from` (or they are the same node), counting disabled
    /// connections too.
    pub(crate) fn would_create_cycle(&self, from: u32, to: u32) -> bool {
        if from == to {
            return true;
        }
        let mut stack = vec![to];
        let mut seen = vec![to];
        while let Some(current) = stack.pop() {
            for connection in self.connections.iter().filter(|c| c.from == current) {
                if connection.to == from {
                    return true;
                }
                if !seen.contains(&connection.to) {
                    seen.push(connection.to);
                    stack.push(connection.to);
                }
            }
        }
        false
    }

    fn has_cycle(&self) -> bool {
        // Kahn's algorithm: a graph is acyclic exactly when repeatedly
        // removing nodes without incoming edges removes every node.
        let mut in_degree = vec![0usize; self.nodes.len()];
        for connection in &self.connections {
            if let Ok(index) = self.nodes.binary_search_by_key(&connection.to, |n| n.id) {
                in_degree[index] += 1;
            }
        }
        let mut ready: Vec<usize> = (0..self.nodes.len())
            .filter(|&i| in_degree[i] == 0)
            .collect();
        let mut removed = 0;
        while let Some(index) = ready.pop() {
            removed += 1;
            let id = self.nodes[index].id;
            for connection in self.connections.iter().filter(|c| c.from == id) {
                if let Ok(target) = self.nodes.binary_search_by_key(&connection.to, |n| n.id) {
                    in_degree[target] -= 1;
                    if in_degree[target] == 0 {
                        ready.push(target);
                    }
                }
            }
        }
        removed != self.nodes.len()
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use rand::SeedableRng;

    use super::*;

    pub fn rng(seed: u64) -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(seed)
    }

    /// Tracker and minimal genome over `num_inputs` inputs, as
    /// `Population::new` would build them.
    pub fn minimal(num_inputs: usize, seed: u64) -> (Genome, InnovationTracker) {
        let mut tracker = InnovationTracker::new(u32::try_from(num_inputs).unwrap() + 2);
        let genome = Genome::minimal(
            num_inputs,
            &mut tracker,
            &NeatConfig::default(),
            &mut rng(seed),
        );
        (genome, tracker)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::minimal;
    use super::*;

    #[test]
    fn minimal_genome_wires_inputs_and_bias_to_the_output() {
        let (genome, _) = minimal(3, 1);
        assert_eq!(genome.num_inputs(), 3);
        assert_eq!(genome.nodes().len(), 5);
        assert_eq!(genome.hidden_count(), 0);
        assert_eq!(genome.output_id(), 4);
        assert_eq!(genome.connections().len(), 4);
        assert!(genome.connections().iter().all(|c| c.enabled && c.to == 4));
        assert!(genome.connections().iter().all(|c| c.weight.abs() <= 1.0));
    }

    #[test]
    fn minimal_genomes_from_one_tracker_share_innovation_numbers() {
        let mut tracker = InnovationTracker::new(5);
        let config = NeatConfig::default();
        let a = Genome::minimal(3, &mut tracker, &config, &mut super::test_support::rng(1));
        let b = Genome::minimal(3, &mut tracker, &config, &mut super::test_support::rng(2));
        let ids = |g: &Genome| {
            g.connections()
                .iter()
                .map(|c| c.innovation)
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(&a), ids(&b));
        assert!((a.connections()[0].weight - b.connections()[0].weight).abs() > f64::EPSILON);
    }

    #[test]
    fn hidden_nodes_stay_sorted_when_ids_arrive_out_of_order() {
        let (mut genome, _) = minimal(1, 1);
        for id in [9, 5, 7] {
            genome.insert_hidden_node(id);
        }
        let ids: Vec<u32> = genome.nodes().iter().map(|n| n.id).collect();
        assert_eq!(ids, vec![0, 1, 2, 5, 7, 9]);
        assert_eq!(genome.hidden_count(), 3);
    }

    #[test]
    fn json_with_an_absurd_input_count_is_rejected_not_a_panic() {
        for count in ["18446744073709551615", "18446744073709551614", "4294967295"] {
            let json = format!(r#"{{"num_inputs":{count},"nodes":[],"connections":[]}}"#);
            assert!(serde_json::from_str::<Genome>(&json).is_err(), "{count}");
        }
    }

    #[test]
    fn json_round_trip_is_bit_exact_for_arbitrary_weights() {
        // Weights are arbitrary f64s after mutation; a saved champion
        // must play exactly like the one that was evaluated, so the text
        // form has to preserve every bit.
        let config = NeatConfig::default();
        let mut random = super::test_support::rng(77);
        for seed in 0..200 {
            let (mut genome, _) = minimal(5, seed);
            for _ in 0..3 {
                crate::mutation::mutate_weights(&mut genome, &config, &mut random);
            }
            let json = serde_json::to_string(&genome).unwrap();
            let restored: Genome = serde_json::from_str(&json).unwrap();
            assert_eq!(restored, genome, "seed {seed}");
        }
    }

    #[test]
    fn json_round_trip_is_lossless() {
        let (genome, _) = minimal(2, 7);
        let json = serde_json::to_string(&genome).unwrap();
        let restored: Genome = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, genome);
    }

    #[test]
    fn json_with_a_cycle_is_rejected() {
        let (genome, _) = minimal(1, 3);
        let mut value: serde_json::Value = serde_json::to_value(&genome).unwrap();
        // Two hidden nodes, 3 and 4, feeding each other.
        value["nodes"].as_array_mut().unwrap().extend([
            serde_json::json!({"id": 3, "kind": "Hidden"}),
            serde_json::json!({"id": 4, "kind": "Hidden"}),
        ]);
        value["connections"].as_array_mut().unwrap().extend([
            serde_json::json!({"innovation": 10, "from": 3, "to": 4, "weight": 1.0, "enabled": true}),
            serde_json::json!({"innovation": 11, "from": 4, "to": 3, "weight": 1.0, "enabled": true}),
        ]);
        let error = serde_json::from_value::<Genome>(value).unwrap_err();
        assert!(error.to_string().contains("cycle"), "{error}");
    }

    #[test]
    fn json_with_a_dangling_connection_is_rejected() {
        let (genome, _) = minimal(1, 3);
        let mut value: serde_json::Value = serde_json::to_value(&genome).unwrap();
        value["connections"].as_array_mut().unwrap().push(
            serde_json::json!({"innovation": 10, "from": 0, "to": 99, "weight": 1.0, "enabled": true}),
        );
        assert!(serde_json::from_value::<Genome>(value).is_err());
    }

    #[test]
    fn json_with_a_non_finite_weight_is_rejected() {
        let (genome, _) = minimal(1, 3);
        let mut parts = (
            genome.num_inputs(),
            genome.nodes().to_vec(),
            genome.connections().to_vec(),
        );
        parts.2[0].weight = f64::NAN;
        assert!(Genome::from_parts(parts.0, parts.1, parts.2).is_err());
    }

    #[test]
    fn connection_into_an_input_is_rejected() {
        let (genome, _) = minimal(2, 3);
        let mut connections = genome.connections().to_vec();
        connections.push(ConnectionGene {
            innovation: 50,
            from: 0,
            to: 1,
            weight: 0.5,
            enabled: true,
        });
        assert!(Genome::from_parts(2, genome.nodes().to_vec(), connections).is_err());
    }

    #[test]
    fn would_create_cycle_sees_disabled_connections_too() {
        let (mut genome, mut tracker) = minimal(1, 3);
        // 0 -> 3 -> 2 (output), with the first leg disabled.
        genome.insert_hidden_node(3);
        genome.insert_connection(ConnectionGene {
            innovation: tracker.connection(0, 3),
            from: 0,
            to: 3,
            weight: 1.0,
            enabled: false,
        });
        genome.insert_connection(ConnectionGene {
            innovation: tracker.connection(3, 2),
            from: 3,
            to: 2,
            weight: 1.0,
            enabled: true,
        });
        assert!(genome.would_create_cycle(3, 0));
        assert!(!genome.would_create_cycle(1, 3));
        assert!(genome.would_create_cycle(3, 3));
    }
}
