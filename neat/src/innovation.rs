//! Historical markings: the run-wide registry that gives every distinct
//! structural change a stable number, which is what lets crossover and
//! speciation line two genomes' genes up.
//!
//! The registry is global for the whole run (not reset per generation):
//! the same `(from, to)` connection always gets the same innovation
//! number, and splitting the same connection always yields the same
//! hidden node id. That keeps ids small and makes two lineages that
//! independently discover the same structure compatible.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// What splitting a connection with a new hidden node produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Split {
    pub node_id: u32,
    /// Innovation of the new `from -> node` connection.
    pub in_innovation: u32,
    /// Innovation of the new `node -> to` connection.
    pub out_innovation: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "TrackerRecord", into = "TrackerRecord")]
pub struct InnovationTracker {
    next_innovation: u32,
    next_node_id: u32,
    connections: HashMap<(u32, u32), u32>,
    splits: HashMap<u32, Split>,
}

impl InnovationTracker {
    /// `first_hidden_id` is the first node id not reserved for the fixed
    /// input/bias/output nodes.
    #[must_use]
    pub fn new(first_hidden_id: u32) -> Self {
        Self {
            next_innovation: 0,
            next_node_id: first_hidden_id,
            connections: HashMap::new(),
            splits: HashMap::new(),
        }
    }

    /// The innovation number of the `from -> to` connection, allocating
    /// one the first time this pair is seen.
    pub fn connection(&mut self, from: u32, to: u32) -> u32 {
        if let Some(&innovation) = self.connections.get(&(from, to)) {
            return innovation;
        }
        let innovation = self.next_innovation;
        self.next_innovation += 1;
        self.connections.insert((from, to), innovation);
        innovation
    }

    /// The hidden node and two connections that splitting the connection
    /// `innovation` (running `from -> to`) creates, allocating them the
    /// first time this connection is split.
    pub fn split(&mut self, innovation: u32, from: u32, to: u32) -> Split {
        if let Some(&split) = self.splits.get(&innovation) {
            return split;
        }
        let node_id = self.next_node_id;
        self.next_node_id += 1;
        let split = Split {
            node_id,
            in_innovation: self.connection(from, node_id),
            out_innovation: self.connection(node_id, to),
        };
        self.splits.insert(innovation, split);
        split
    }

    /// Total distinct connections registered so far.
    #[must_use]
    pub fn innovation_count(&self) -> usize {
        self.connections.len()
    }
}

/// Serialized form: sorted vectors, so the JSON is stable and has no
/// tuple map keys (which JSON cannot express).
#[derive(Serialize, Deserialize)]
struct TrackerRecord {
    next_innovation: u32,
    next_node_id: u32,
    /// `(from, to, innovation)`
    connections: Vec<(u32, u32, u32)>,
    /// `(split connection, node, in innovation, out innovation)`
    splits: Vec<(u32, u32, u32, u32)>,
}

impl From<InnovationTracker> for TrackerRecord {
    fn from(tracker: InnovationTracker) -> Self {
        let mut connections: Vec<_> = tracker
            .connections
            .into_iter()
            .map(|((from, to), innovation)| (from, to, innovation))
            .collect();
        connections.sort_unstable();
        let mut splits: Vec<_> = tracker
            .splits
            .into_iter()
            .map(|(on, s)| (on, s.node_id, s.in_innovation, s.out_innovation))
            .collect();
        splits.sort_unstable();
        Self {
            next_innovation: tracker.next_innovation,
            next_node_id: tracker.next_node_id,
            connections,
            splits,
        }
    }
}

impl From<TrackerRecord> for InnovationTracker {
    fn from(record: TrackerRecord) -> Self {
        Self {
            next_innovation: record.next_innovation,
            next_node_id: record.next_node_id,
            connections: record
                .connections
                .into_iter()
                .map(|(from, to, innovation)| ((from, to), innovation))
                .collect(),
            splits: record
                .splits
                .into_iter()
                .map(|(on, node_id, in_innovation, out_innovation)| {
                    (
                        on,
                        Split {
                            node_id,
                            in_innovation,
                            out_innovation,
                        },
                    )
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_connection_gets_the_same_innovation_every_time() {
        let mut tracker = InnovationTracker::new(10);
        let first = tracker.connection(0, 9);
        let other = tracker.connection(1, 9);
        assert_ne!(first, other);
        assert_eq!(tracker.connection(0, 9), first);
        assert_eq!(tracker.innovation_count(), 2);
    }

    #[test]
    fn splitting_the_same_connection_twice_returns_the_same_node() {
        let mut tracker = InnovationTracker::new(10);
        let innovation = tracker.connection(0, 9);
        let first = tracker.split(innovation, 0, 9);
        assert_eq!(first.node_id, 10);
        assert_eq!(tracker.split(innovation, 0, 9), first);
        let other = tracker.connection(1, 9);
        assert_eq!(tracker.split(other, 1, 9).node_id, 11);
    }

    #[test]
    fn split_connections_are_registered_like_any_other() {
        let mut tracker = InnovationTracker::new(10);
        let innovation = tracker.connection(0, 9);
        let split = tracker.split(innovation, 0, 9);
        assert_eq!(tracker.connection(0, split.node_id), split.in_innovation);
        assert_eq!(tracker.connection(split.node_id, 9), split.out_innovation);
    }

    #[test]
    fn json_round_trip_preserves_every_future_allocation() {
        let mut tracker = InnovationTracker::new(10);
        let innovation = tracker.connection(0, 9);
        tracker.split(innovation, 0, 9);
        let json = serde_json::to_string(&tracker).unwrap();
        let mut restored: InnovationTracker = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, tracker);
        assert_eq!(restored.connection(3, 4), tracker.connection(3, 4));
        assert_eq!(restored.split(5, 3, 4), tracker.split(5, 3, 4));
    }
}
