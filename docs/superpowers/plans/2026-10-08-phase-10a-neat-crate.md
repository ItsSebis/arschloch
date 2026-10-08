# Phase 10a — `neat` Crate Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `neat`, a generic, game-agnostic NEAT library (genomes with
historical markings, mutation, crossover, speciation, a feedforward
evaluator and a generational `Population` loop) that provably learns
XOR, as the foundation for Phases 10b-10e.

**Architecture:** A new workspace crate `neat` with no dependency on
`engine` or `sim`. A `Genome` keeps its structural invariants (ordered
nodes, no cycles even among disabled genes, fixed input/bias/output
layout) so every network compiled from it is feedforward. An
`InnovationTracker` gives structural changes run-wide ids. `Population`
owns the loop but not the evaluation: the caller reads `genomes()`,
reports fitness with `set_fitness`, and calls `advance`, which returns a
`GenerationReport` (the numbers Phase 10c will stream to the terminal and
`events.jsonl`). One seeded `StdRng` drives everything, so a run is
reproducible.

**Tech Stack:** Rust workspace; `rand 0.10.3` (`Rng`, `RngExt`,
`seq::{IndexedRandom, IndexedMutRandom}`, `StdRng`), `serde 1.0.229`
with derive, `serde_json 1.0.151`. All three are already in
`Cargo.lock` via `sim`.

**Spec:** `docs/superpowers/specs/2026-10-08-neat-engine-design.md`,
sections 3 (architecture), 5 (NEAT algorithm specifics) and 10 (testing,
the `neat`-crate bullet). This plan implements phase "10a" of its
section 11.

## How this plan was prepared

Every code block below was written into a scratch copy of the workspace,
compiled, formatted with `cargo fmt`, linted with the workspace's
pedantic clippy (`-D warnings`) and tested: 60 unit tests plus 2
integration tests, all passing. Executors should copy the code as
written. If a step's result differs from its "Expected", stop and
investigate instead of editing code until it passes.

## Global Constraints

- New crate `neat` depends only on `rand = "0.10.3"`, `serde = { version = "1.0.229", features = ["derive"] }` and `serde_json = "1.0.151"`: no other new dependency, and **no dependency on `engine` or `sim`** (spec section 3: `neat` has no game knowledge; the dependency direction is `sim -> neat`, added in 10b).
- `[lints] workspace = true` in `neat/Cargo.toml`: `unsafe_code = "forbid"`, clippy pedantic as warnings, and the phase gate runs clippy with `-D warnings`.
- Randomness only through `rand` traits, from the one `StdRng` that `Population` owns, seeded from the run seed. No `thread_rng`, no wall-clock, and no iteration over a `HashMap` where order could affect results (the tracker's maps are used for lookup only; its JSON form is sorted).
- Higher fitness is better; fitness may be negative; fitness must be finite (NaN panics, as it would silently corrupt ranking).
- Genome invariants (module docs of `neat/src/genome.rs`) are checked by `Genome::from_parts` and therefore by JSON loading; hand-edited or stale genome JSON must be rejected with `NeatError::InvalidGenome`, never accepted and never a panic.
- Defaults come from `NeatConfig::default()` (population 150, `target_species` 8, compatibility coefficients 1.0/1.0/0.4, threshold 3.0, stagnation limit 15, survival fraction 0.2, elitism for species of 5 or more).
- Activation is `tanh` for hidden and output nodes; the bias node is `1.0`; one output node.
- Existing crates and baseline strategies are not modified (`docs/baselines/pre-neat/` stays valid).
- Per-task verification is `cargo test -p neat`. Between Task 3 and Task 8 the non-test build can report `dead_code` warnings for items that later tasks start using; they disappear by Task 8 and the full gate (fmt, clippy `-D warnings`, whole-workspace tests) runs in Task 10.

## Review Focus

Failure modes the spec implies but a straightforward implementation tends to miss, each pinned by a named test:

1. **Smallest legal population (2 genomes) with many species:** quotas must still sum to the population size. Test: `population::tests::the_smallest_legal_population_keeps_evolving`.
2. **A wide input layer** (the game network will have dozens of features): nothing assumes a handful of inputs. Test: `population::tests::a_wide_input_layer_works`.
3. **Lineages that discovered different hidden nodes interbreeding** (out-of-order hidden node ids, accidental cycles): Tests: `genome::tests::hidden_nodes_stay_sorted_when_ids_arrive_out_of_order` and `population::tests::every_genome_stays_valid_across_many_mixed_lineage_generations`.
4. **A stale or hand-edited genome file** (cycle, dangling endpoint, NaN weight, wrong node kind): rejected with an error, not accepted. Tests: `genome::tests::json_with_a_cycle_is_rejected`, `json_with_a_dangling_connection_is_rejected`, `json_with_a_non_finite_weight_is_rejected`, `connection_into_an_input_is_rejected`.
5. **One network shared by many parallel matches** (10b/10c evaluate genomes with rayon): `Network` must be `Send + Sync` with no interior state. Test: `network::tests::network_can_be_shared_across_threads`.

Also pinned: constant and all-negative fitness (no division by zero, population size preserved), the all-time best never regressing, same seed giving an identical population, and the XOR end-to-end check.

---

## File Structure

```
Cargo.toml                     # modify: add "neat" to workspace members
neat/Cargo.toml                # new
neat/src/lib.rs                # module list + re-exports
neat/src/error.rs              # NeatError
neat/src/config.rs             # NeatConfig (+ validation)
neat/src/innovation.rs         # InnovationTracker (run-wide historical markings)
neat/src/genome.rs             # Genome, NodeGene, ConnectionGene, invariants, JSON
neat/src/mutation.rs           # weight / add-connection / add-node / toggle operators
neat/src/crossover.rs          # innovation-aligned crossover
neat/src/network.rs            # Genome -> immutable feedforward evaluator
neat/src/species.rs            # compatibility distance, speciation, SpeciesStats
neat/src/population.rs         # generational loop, quotas, GenerationReport
neat/tests/xor.rs              # end-to-end: evolve XOR
docs/ROADMAP.md, docs/ARCHITECTURE.md, docs/BUILDING.md   # modify (Task 10)
```

Not in this phase (deliberately): opponent pools, fitness functions,
game features, `NeatStrategy`, the CLI `train` subcommand, checkpoint
files, the dashboard, and "mutation counts by kind that improved
fitness" from the spec's event schema (that needs per-offspring
attribution, which 10c adds around `Population::advance`).

---


### Task 1: Crate scaffold, `NeatError`, `NeatConfig`
**Files:**
- Modify: `Cargo.toml`
- Create: `neat/Cargo.toml`, `neat/src/lib.rs`, `neat/src/error.rs`, `neat/src/config.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: `neat::NeatError { InvalidConfig(String), InvalidGenome(String) }` (Display, Error, Clone, PartialEq, Eq); `neat::NeatConfig` (all-pub fields, `Default`, `Serialize`/`Deserialize`, `fn validate(&self) -> Result<(), NeatError>`).

- [ ] **Step 1: Add the crate to the workspace**

In the root `Cargo.toml`, change the members line to:

```toml
members = ["engine", "sim", "cli", "web", "neat"]
```

Create `neat/Cargo.toml`:

```toml
[package]
name = "neat"
version.workspace = true
edition.workspace = true
license.workspace = true

[lints]
workspace = true

[dependencies]
rand = "0.10.3"
serde = { version = "1.0.229", features = ["derive"] }
serde_json = "1.0.151"
```

- [ ] **Step 2: Write the failing tests**

Create `neat/src/lib.rs`:

```rust
//! A generic NEAT (neuroevolution of augmenting topologies) library:
//! genomes, historical markings, mutation, crossover, speciation, a
//! feedforward evaluator and the generational loop. It knows nothing
//! about cards or games; callers evaluate genomes and report fitness.
//! See docs/superpowers/specs/2026-10-08-neat-engine-design.md.

pub mod config;
pub mod error;

pub use config::NeatConfig;
pub use error::NeatError;
```

Create `neat/src/error.rs` containing only this test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_names_the_kind_and_reason() {
        let config = NeatError::InvalidConfig("population_size is 0".into());
        let genome = NeatError::InvalidGenome("cycle".into());
        assert_eq!(
            config.to_string(),
            "invalid NEAT config: population_size is 0"
        );
        assert_eq!(genome.to_string(), "invalid genome: cycle");
    }
}
```

Create `neat/src/config.rs` containing only this test module:

```rust
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test -p neat`

Expected: FAIL (compile errors: `NeatError` and `NeatConfig` are not defined yet).

- [ ] **Step 4: Implement `NeatError`**

Insert at the top of `neat/src/error.rs`, above `#[cfg(test)]`:

```rust
//! The one error type of the `neat` crate.

use std::fmt;

/// Everything that can go wrong at this crate's boundaries: a bad
/// configuration, or a genome (built by hand or read from JSON) that
/// breaks a structural invariant. Internal code treats invariant
/// violations as bugs and panics instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NeatError {
    InvalidConfig(String),
    InvalidGenome(String),
}

impl fmt::Display for NeatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(reason) => write!(f, "invalid NEAT config: {reason}"),
            Self::InvalidGenome(reason) => write!(f, "invalid genome: {reason}"),
        }
    }
}

impl std::error::Error for NeatError {}
```

- [ ] **Step 5: Implement `NeatConfig`**

Insert at the top of `neat/src/config.rs`, above `#[cfg(test)]`:

```rust
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
    pub compatibility_threshold: f64,
    pub min_compatibility_threshold: f64,
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
            compatibility_threshold: 3.0,
            min_compatibility_threshold: 0.3,
            threshold_step: 0.3,
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo fmt -p neat && cargo test -p neat`

Expected: PASS (5 tests: 1 in `error`, 4 in `config`).

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock neat
git commit -m "$(cat <<'EOF'
neat: add the crate with NeatError and NeatConfig (Phase 10a)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
)"
```


### Task 2: `InnovationTracker`
**Files:**
- Create: `neat/src/innovation.rs`
- Modify: `neat/src/lib.rs`

**Interfaces:**
- Consumes: nothing from earlier tasks (std `HashMap`, `serde`).
- Produces: `InnovationTracker::new(first_hidden_id: u32) -> Self`; `fn connection(&mut self, from: u32, to: u32) -> u32`; `fn split(&mut self, innovation: u32, from: u32, to: u32) -> Split`; `fn innovation_count(&self) -> usize`; `Split { node_id, in_innovation, out_innovation }` (all `u32`); the tracker is `Clone + PartialEq + Serialize + Deserialize` (sorted-vector JSON form).

- [ ] **Step 1: Write the failing tests**

Create `neat/src/innovation.rs` containing only this test module, and wire the module into the crate:

Add these lines to `neat/src/lib.rs`, placing each `pub mod` in the alphabetical list of modules and each `pub use` in the alphabetical list of re-exports (Task 8 shows the final file):

```rust
pub mod innovation;

pub use innovation::InnovationTracker;
```

```rust
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p neat innovation::`

Expected: FAIL (compile errors: the items the tests use are not defined yet).

- [ ] **Step 3: Implement**

Insert this at the very top of `neat/src/innovation.rs`, above the `#[cfg(test)]` line:

```rust
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo fmt -p neat && cargo test -p neat innovation::`

Expected: PASS. If anything fails, stop and investigate: this code was compiled and tested as written, so a difference points at a typo or an environment problem, not a design change.

- [ ] **Step 5: Commit**

```bash
git add neat/src/innovation.rs neat/src/lib.rs
git commit -m "$(cat <<'EOF'
neat: add the run-wide InnovationTracker (Phase 10a)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
)"
```


### Task 3: `Genome` and its invariants
**Files:**
- Create: `neat/src/genome.rs`
- Modify: `neat/src/lib.rs`

**Interfaces:**
- Consumes: `NeatConfig` (`weight_init_range`), `NeatError::InvalidGenome`, `InnovationTracker::connection`.
- Produces: `Genome` (private fields; `Clone + PartialEq + Serialize`, `Deserialize` via validation); `NodeKind { Input, Bias, Output, Hidden }`; `NodeGene { id: u32, kind }`; `ConnectionGene { innovation: u32, from: u32, to: u32, weight: f64, enabled: bool }`; `Genome::minimal(num_inputs, &mut InnovationTracker, &NeatConfig, &mut R) -> Genome`; `Genome::from_parts(num_inputs, Vec<NodeGene>, Vec<ConnectionGene>) -> Result<Genome, NeatError>`; accessors `num_inputs()`, `nodes()`, `connections()`, `hidden_count()`, `enabled_connection_count()`, `output_id()`; crate-private helpers `node`, `connection_by_innovation`, `has_connection`, `connections_mut`, `insert_hidden_node`, `insert_connection`, `would_create_cycle`; test-only `genome::test_support::{rng, minimal}` used by later tasks' tests.

- [ ] **Step 1: Write the failing tests**

Create `neat/src/genome.rs` containing only this test module, and wire the module into the crate:

Add these lines to `neat/src/lib.rs`, placing each `pub mod` in the alphabetical list of modules and each `pub use` in the alphabetical list of re-exports (Task 8 shows the final file):

```rust
pub mod genome;

pub use genome::{ConnectionGene, Genome, NodeGene, NodeKind};
```

```rust
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p neat genome::`

Expected: FAIL (compile errors: the items the tests use are not defined yet).

- [ ] **Step 3: Implement**

Insert this at the very top of `neat/src/genome.rs`, above the `#[cfg(test)]` line:

```rust
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
        let fixed = self.num_inputs + 2;
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo fmt -p neat && cargo test -p neat genome::`

Expected: PASS. If anything fails, stop and investigate: this code was compiled and tested as written, so a difference points at a typo or an environment problem, not a design change.

- [ ] **Step 5: Commit**

```bash
git add neat/src/genome.rs neat/src/lib.rs
git commit -m "$(cat <<'EOF'
neat: add Genome with enforced structural invariants (Phase 10a)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
)"
```


### Task 4: Mutation operators
**Files:**
- Create: `neat/src/mutation.rs`
- Modify: `neat/src/lib.rs`

**Interfaces:**
- Consumes: `Genome` crate-private helpers and `test_support` (Task 3), `InnovationTracker::{connection, split}` (Task 2), `NeatConfig` rates.
- Produces: `mutation::mutate(&mut Genome, &mut InnovationTracker, &NeatConfig, &mut R)`; `mutate_weights(&mut Genome, &NeatConfig, &mut R)`; `add_connection(..) -> bool`; `add_node(..) -> bool`; `toggle_enable(&mut Genome, &mut R)`. All preserve the genome invariants.

- [ ] **Step 1: Write the failing tests**

Create `neat/src/mutation.rs` containing only this test module, and wire the module into the crate:

Add these lines to `neat/src/lib.rs`, placing each `pub mod` in the alphabetical list of modules and each `pub use` in the alphabetical list of re-exports (Task 8 shows the final file):

```rust
pub mod mutation;
```

```rust
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p neat mutation::`

Expected: FAIL (compile errors: the items the tests use are not defined yet).

- [ ] **Step 3: Implement**

Insert this at the very top of `neat/src/mutation.rs`, above the `#[cfg(test)]` line:

```rust
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo fmt -p neat && cargo test -p neat mutation::`

Expected: PASS. If anything fails, stop and investigate: this code was compiled and tested as written, so a difference points at a typo or an environment problem, not a design change.

- [ ] **Step 5: Commit**

```bash
git add neat/src/mutation.rs neat/src/lib.rs
git commit -m "$(cat <<'EOF'
neat: add mutation operators (Phase 10a)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
)"
```


### Task 5: Crossover
**Files:**
- Create: `neat/src/crossover.rs`
- Modify: `neat/src/lib.rs`

**Interfaces:**
- Consumes: `Genome` accessors/helpers and `test_support` (Task 3), mutation operators for the tests (Task 4), `NeatConfig::disabled_gene_rate`.
- Produces: `crossover::crossover(fitter: &Genome, other: &Genome, &NeatConfig, &mut R) -> Genome` (child structure equals `fitter`'s; panics on differing input counts).

- [ ] **Step 1: Write the failing tests**

Create `neat/src/crossover.rs` containing only this test module, and wire the module into the crate:

Add these lines to `neat/src/lib.rs`, placing each `pub mod` in the alphabetical list of modules and each `pub use` in the alphabetical list of re-exports (Task 8 shows the final file):

```rust
pub mod crossover;
```

```rust
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p neat crossover::`

Expected: FAIL (compile errors: the items the tests use are not defined yet).

- [ ] **Step 3: Implement**

Insert this at the very top of `neat/src/crossover.rs`, above the `#[cfg(test)]` line:

```rust
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo fmt -p neat && cargo test -p neat crossover::`

Expected: PASS. If anything fails, stop and investigate: this code was compiled and tested as written, so a difference points at a typo or an environment problem, not a design change.

- [ ] **Step 5: Commit**

```bash
git add neat/src/crossover.rs neat/src/lib.rs
git commit -m "$(cat <<'EOF'
neat: add innovation-aligned crossover (Phase 10a)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
)"
```


### Task 6: Feedforward `Network`
**Files:**
- Create: `neat/src/network.rs`
- Modify: `neat/src/lib.rs`

**Interfaces:**
- Consumes: `Genome` accessors and `test_support` (Task 3), `mutate` for one test (Task 4).
- Produces: `Network::compile(&Genome) -> Network` (`Clone + PartialEq`, `Send + Sync`); `fn num_inputs(&self) -> usize`; `fn activate(&self, inputs: &[f64], scratch: &mut Vec<f64>) -> f64`.

- [ ] **Step 1: Write the failing tests**

Create `neat/src/network.rs` containing only this test module, and wire the module into the crate:

Add these lines to `neat/src/lib.rs`, placing each `pub mod` in the alphabetical list of modules and each `pub use` in the alphabetical list of re-exports (Task 8 shows the final file):

```rust
pub mod network;

pub use network::Network;
```

```rust
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p neat network::`

Expected: FAIL (compile errors: the items the tests use are not defined yet).

- [ ] **Step 3: Implement**

Insert this at the very top of `neat/src/network.rs`, above the `#[cfg(test)]` line:

```rust
//! A genome compiled into a fast, immutable feedforward evaluator.
//!
//! Inputs are copied through, the bias node is always `1.0`, and every
//! hidden and output node computes `tanh(sum of weight * source)` over
//! its enabled incoming connections. `Network` holds no mutable state, so
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

    /// Evaluates the network. `scratch` is reused across calls to avoid
    /// allocating; its prior contents are irrelevant.
    ///
    /// # Panics
    ///
    /// Panics if `inputs.len()` is not `num_inputs()`.
    pub fn activate(&self, inputs: &[f64], scratch: &mut Vec<f64>) -> f64 {
        assert_eq!(inputs.len(), self.num_inputs, "wrong number of inputs");
        scratch.clear();
        scratch.resize(self.node_count, 0.0);
        scratch[..self.num_inputs].copy_from_slice(inputs);
        scratch[self.bias_index] = 1.0;
        for (target, sources) in &self.steps {
            let sum: f64 = sources
                .iter()
                .map(|&(source, weight)| scratch[source] * weight)
                .sum();
            scratch[*target] = sum.tanh();
        }
        scratch[self.output_index]
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo fmt -p neat && cargo test -p neat network::`

Expected: PASS. If anything fails, stop and investigate: this code was compiled and tested as written, so a difference points at a typo or an environment problem, not a design change.

- [ ] **Step 5: Commit**

```bash
git add neat/src/network.rs neat/src/lib.rs
git commit -m "$(cat <<'EOF'
neat: add the feedforward Network evaluator (Phase 10a)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
)"
```


### Task 7: Compatibility distance and speciation
**Files:**
- Create: `neat/src/species.rs`
- Modify: `neat/src/lib.rs`

**Interfaces:**
- Consumes: `Genome` accessors and `test_support` (Task 3), `add_node`/`add_connection` for tests (Task 4), `NeatConfig` coefficients.
- Produces: `species::compatibility_distance(&Genome, &Genome, &NeatConfig) -> f64`; `SpeciesStats { id, size, age, generation_best, mean_fitness, best_fitness, stagnation }` (`Serialize`); crate-private `Species` and `assign(&mut Vec<Species>, &[Genome], threshold, &NeatConfig, &mut u32, &mut R)`.

- [ ] **Step 1: Write the failing tests**

Create `neat/src/species.rs` containing only this test module, and wire the module into the crate:

Add these lines to `neat/src/lib.rs`, placing each `pub mod` in the alphabetical list of modules and each `pub use` in the alphabetical list of re-exports (Task 8 shows the final file):

```rust
pub mod species;

pub use species::SpeciesStats;
```

```rust
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p neat species::`

Expected: FAIL (compile errors: the items the tests use are not defined yet).

- [ ] **Step 3: Implement**

Insert this at the very top of `neat/src/species.rs`, above the `#[cfg(test)]` line:

```rust
//! Speciation: grouping genomes by structural similarity so a new
//! structure competes mainly with its own kind while it is still being
//! tuned, instead of being outcompeted by established topologies.

use rand::seq::IndexedRandom;
use rand::Rng;
use serde::Serialize;

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

#[derive(Debug, Clone)]
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
#[derive(Debug, Clone, PartialEq, Serialize)]
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo fmt -p neat && cargo test -p neat species::`

Expected: PASS. If anything fails, stop and investigate: this code was compiled and tested as written, so a difference points at a typo or an environment problem, not a design change.

- [ ] **Step 5: Commit**

```bash
git add neat/src/species.rs neat/src/lib.rs
git commit -m "$(cat <<'EOF'
neat: add compatibility distance and speciation (Phase 10a)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
)"
```


### Task 8: `Population`: quotas, reproduction, the generational loop
**Files:**
- Create: `neat/src/population.rs`
- Modify: `neat/src/lib.rs`

**Interfaces:**
- Consumes: Everything above: `Genome::minimal`, `mutate`, `crossover`, `assign`/`Species`/`SpeciesStats`, `InnovationTracker`, `NeatConfig`, `NeatError`.
- Produces: `Population::new(num_inputs: usize, NeatConfig, seed: u64) -> Result<Population, NeatError>`; `genomes() -> &[Genome]`; `generation() -> u32`; `best() -> Option<(&Genome, f64)>`; `config() -> &NeatConfig`; `set_fitness(Vec<f64>)`; `advance() -> GenerationReport`; `GenerationReport { generation, best_fitness, mean_fitness, median_fitness, min_fitness, champion_index, species: Vec<SpeciesStats>, compatibility_threshold, mean_hidden_nodes, mean_enabled_connections, innovation_count }` (`Serialize`).

- [ ] **Step 1: Write the failing tests**

Create `neat/src/population.rs` containing only this test module, and wire the module into the crate:

Add these lines to `neat/src/lib.rs`, placing each `pub mod` in the alphabetical list of modules and each `pub use` in the alphabetical list of re-exports (Task 8 shows the final file):

```rust
pub mod population;

pub use population::{GenerationReport, Population};
```

```rust
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p neat population::`

Expected: FAIL (compile errors: the items the tests use are not defined yet).

- [ ] **Step 3: Implement**

Insert this at the very top of `neat/src/population.rs`, above the `#[cfg(test)]` line:

```rust
//! The generational loop: speciate, rank, allocate offspring, reproduce.
//!
//! The caller owns evaluation. A generation is:
//! `genomes()` -> evaluate each -> `set_fitness(..)` -> `advance()`.
//! Higher fitness is better; fitness may be negative.
//!
//! Everything random flows through one seeded `StdRng`, and species and
//! genomes are always visited in index order, so a run is a pure
//! function of `(config, num_inputs, seed, fitness values)`.

use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand::{RngExt, SeedableRng};
use serde::Serialize;

use crate::config::NeatConfig;
use crate::crossover::crossover;
use crate::error::NeatError;
use crate::genome::Genome;
use crate::innovation::InnovationTracker;
use crate::mutation::mutate;
use crate::species::{assign, Species, SpeciesStats};

/// Numbers describing the generation that `advance` just consumed.
#[derive(Debug, Clone, PartialEq, Serialize)]
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

pub struct Population {
    config: NeatConfig,
    rng: StdRng,
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
        let mut rng = StdRng::seed_from_u64(seed);
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
        let protected = self
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
        let weights: Vec<f64> = self
            .species
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if s.stagnation >= self.config.stagnation_limit && i != protected {
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo fmt -p neat && cargo test -p neat population::`

Expected: PASS. If anything fails, stop and investigate: this code was compiled and tested as written, so a difference points at a typo or an environment problem, not a design change.

- [ ] **Step 5: Check `lib.rs` against the final layout**

`neat/src/lib.rs` should now read exactly:

```rust
//! A generic NEAT (neuroevolution of augmenting topologies) library:
//! genomes, historical markings, mutation, crossover, speciation, a
//! feedforward evaluator and the generational loop. It knows nothing
//! about cards or games; callers evaluate genomes and report fitness.
//! See docs/superpowers/specs/2026-10-08-neat-engine-design.md.

pub mod config;
pub mod crossover;
pub mod error;
pub mod genome;
pub mod innovation;
pub mod mutation;
pub mod network;
pub mod population;
pub mod species;

pub use config::NeatConfig;
pub use error::NeatError;
pub use genome::{ConnectionGene, Genome, NodeGene, NodeKind};
pub use innovation::InnovationTracker;
pub use network::Network;
pub use population::{GenerationReport, Population};
pub use species::SpeciesStats;
```

Run: `cargo test -p neat` once more after fixing the order if it differs.

- [ ] **Step 6: Commit**

```bash
git add neat/src/population.rs neat/src/lib.rs
git commit -m "$(cat <<'EOF'
neat: add Population, the generational NEAT loop (Phase 10a)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
)"
```


### Task 9: End-to-end: evolve XOR
**Files:**
- Create: `neat/tests/xor.rs`

**Interfaces:**
- Consumes: `neat::{NeatConfig, Network, Population}` (public API only).
- Produces: nothing (integration test). It documents the intended driver loop for Phase 10c: compile networks, evaluate, `set_fitness`, `advance`.

- [ ] **Step 1: Write the integration test**

Create `neat/tests/xor.rs`:

```rust
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
```

- [ ] **Step 2: Run it**

Run: `cargo test -p neat --test xor`

Expected: PASS in well under a second. During preparation, all 30 seeds from 0 to 29 solved XOR within 30 generations; this test requires 5 seeds within 100, so a failure means the loop regressed (or a step above was mistyped), not that the seed was unlucky.

Note: this test is a regression guard added after the loop exists, so it cannot be seen failing first; to see it fail meaningfully, temporarily make `Population::reproduce` skip `mutate` and confirm the test fails, then revert.

- [ ] **Step 3: Commit**

```bash
git add neat/tests/xor.rs
git commit -m "$(cat <<'EOF'
neat: add an end-to-end XOR evolution test (Phase 10a)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
)"
```


### Task 10: Docs, full gate, wrap-up

**Files:**
- Modify: `docs/ROADMAP.md`, `docs/ARCHITECTURE.md`, `docs/BUILDING.md`

**Interfaces:**
- Consumes: the finished `neat` crate.
- Produces: docs that say 10a is done and how to use/test the crate.

- [ ] **Step 1: Mark 10a done in the roadmap**

In `docs/ROADMAP.md`, in the Phase 10 bullet that begins ``- 10a: generic `neat` crate.``, insert `(done)` after `10a`, so it reads ``- 10a (done): generic `neat` crate.`` (the rest of that bullet, covering 10b-10e, stays as is).

- [ ] **Step 2: Update the architecture doc**

In `docs/ARCHITECTURE.md`, change the heading to ``### `neat` (Phase 10; core done in 10a)`` (it currently ends in `(planned, Phase 10)`) and add this paragraph below its existing text:

```markdown
Public surface: `Population` (own the generational loop; the caller
evaluates `genomes()`, calls `set_fitness`, then `advance`, which returns
a `GenerationReport`), `Genome` (validated, JSON-serializable),
`Network` (immutable, `Send + Sync` feedforward evaluator with a
caller-supplied scratch buffer), `InnovationTracker`, `NeatConfig` and
`NeatError`. Randomness comes only from the population's seeded
`StdRng`.
```

- [ ] **Step 3: Document the test command**

In `docs/BUILDING.md`, after the "Native build & test" code block's explanatory paragraph, add:

```markdown
The `neat` crate has no game dependency: `cargo test -p neat` runs its
unit tests and the XOR end-to-end evolution test in about a second.
```

- [ ] **Step 4: Run the whole phase-done gate**

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: all clean; `neat` contributes 60 unit tests and 2 integration tests. If clippy reports `dead_code` or other warnings, they indicate a step was skipped or mistyped: compare against the code in the task named by the warning.

- [ ] **Step 5: Confirm nothing else changed**

Run: `git status --short` and `git diff --stat HEAD~9 -- engine sim cli web docs/baselines`

Expected: no changes under `engine/`, `sim/`, `cli/`, `web/` or `docs/baselines/` (the `HEAD~9` base is the commit before Task 1; adjust the number if tasks were committed differently).

- [ ] **Step 6: Commit**

```bash
git add docs/ROADMAP.md docs/ARCHITECTURE.md docs/BUILDING.md
git commit -m "$(cat <<'EOF'
docs: mark Phase 10a done and document the neat crate

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review (done while writing)

- **Spec coverage (sections 5 and 10, `neat` bullet):** innovation numbers and stable ids (Task 2); genome with fixed input/bias/output and minimal start topology (Task 3); weight, add-connection (cycle-free), add-node and toggle mutations (Task 4); innovation-aligned crossover with 75% disabled-gene rule and fitter-parent excess/disjoint (Task 5); `tanh` feedforward evaluation matching a hand-computed example (Task 6); compatibility distance, adaptive threshold, fitness sharing via mean-shifted species weights, elitism, stagnation culling with the best species protected (Tasks 7-8); genome JSON round-trip (Task 3); every genome in exactly one species (Tasks 7-8); XOR learns (Task 9). Spec items deliberately left to later phases are listed under "Not in this phase".
- **Placeholder scan:** none; every code step contains the complete code.
- **Type consistency:** names and signatures in each task's Interfaces block match the code (`Genome::minimal`, `from_parts`, `Population::new/set_fitness/advance`, `GenerationReport` fields, `SpeciesStats` fields).
- **Review Focus:** all five lines map to named tests above.
