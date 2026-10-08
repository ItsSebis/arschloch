# Phase 10c — Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `cli train`: evolve a NEAT player against a pool of opponents,
generation by generation, with a row per generation on the terminal, a
machine-readable event log, champion genome files that play in the normal
simulator, and exact checkpoint/resume.

**Architecture:** `neat::Population` becomes snapshot/restorable (its RNG
switches to a serializable xoshiro256++). A new `sim::training` module
holds evaluation (common random numbers: every genome in a generation
plays the same deals, seats and opponents), `TrainConfig`, the event
schema, the run directory (atomic checkpoints) and `Trainer`, which
reports through a `TrainObserver` trait. `cli train` parses arguments and
renders events. The existing flat `cli` argument parser is untouched:
`cli train ...` is dispatched before it.

**Tech Stack:** Rust workspace; `neat` (10a), `NeatStrategy`/`GenomeFile`
(10b), `rayon` (already used), `rand 0.10.3` with its `serde` feature
(a feature flag of a crate already in the lockfile).

**Spec:** `docs/superpowers/specs/2026-10-08-neat-engine-design.md`,
sections 6 (fitness and evaluation), 7 (training CLI), 7a (live
monitoring: the terminal and event-log parts) and 8 (determinism). Task
9 records, as section 7b, what was built and where it deliberately
differs.

## How this plan was prepared

The code below was written into a scratch copy of the workspace first and
passed `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D
warnings` and `cargo test --workspace` (about 380 tests). It was also run
for real: a 30-generation training through the built binary (population
80, about 35 s), the champion re-measured through the ordinary simulator
(agreeing with the trainer's own numbers: +0.57/+0.61/+0.54 against
+0.59/+0.62/+0.56 for lowest-legal/endgame-denial/adaptive), and a
SIGKILL-and-resume whose final state was byte-identical to an
uninterrupted run. Those two real scenarios are Task 8's scripts. The
steps are applied by one executor from the same data that renders this
document.

## Global Constraints

- Dependency direction stays `cli -> sim -> neat`, `sim -> engine`; `neat` depends on neither. `cli` now depends on `neat` directly (to build a `NeatConfig`).
- No new crates. `rand` gains its `serde` feature in `neat` (for `Xoshiro256PlusPlus`).
- `Strategy`, `TurnContext`, `run_match`, `run_batch`, the hand-written strategies and the flat `cli` simulation arguments are **not modified**. `docs/baselines/pre-neat/` keeps its numbers (Task 9 only adds a caveats paragraph).
- A run is a pure function of its `TrainConfig`: all randomness comes from the seed (match seeds from `(run seed, generation, index)`, evolution from the population's xoshiro256++), genomes are evaluated independently so thread count cannot change results, and nothing depends on wall-clock time except fields explicitly named `*_secs`/`rounds_per_sec` in events.
- Evaluation uses common random numbers and **sampled** opponents (not cyclic rotation): table position relative to the other players is a large effect, and rotation preserves neighbour order.
- Higher fitness is better. Scores are mean finishing-role scores, +1 (best role) to -1 (worst role).
- Per generation the persistence order is: champion genome files, event line, then the checkpoint last. Files other than `events.jsonl` are written atomically (temp file, fsync, rename). A new run never overwrites an existing run directory.
- The event schema (`SCHEMA_VERSION = 1`) changes additively only; a removal or rename bumps the version, and the checkpoint reader refuses other versions.
- Genome files written by training are ordinary `GenomeFile`s (feature names and feature-set version included).
- Opponent pool names must be unique (results are grouped by name); the default pool is `lowest-legal`, `endgame-denial`, `adaptive:reading,tempo,bully`.
- Between tasks 2 and 6 the non-test build can report `dead_code`/unused warnings for items later tasks start using; those are expected. Any other warning is a defect.
- Per-task verification is `cargo test -p <crate>`; the full gate (`cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`) runs in Task 9.

## Review Focus

Failure modes the spec implies but a straightforward implementation tends to miss, each pinned by a named test or scenario:

1. **Resume must equal never stopping, including after a hard kill:** `training_run::resuming_is_identical_to_never_stopping`, `a_crash_between_the_event_and_the_checkpoint_is_repaired_on_resume`, and Task 8's `scenario2.sh` (SIGKILL).
2. **Results must not depend on thread count or timing:** `training_run::results_do_not_depend_on_the_thread_count` (1 vs 4 threads).
3. **A trained champion's score must be real:** `scenario1.sh` re-measures the saved `best.json` through the ordinary simulator and requires agreement with the trainer's own numbers and a clearly better-than-even result; `training_run::training_improves_play` requires the population mean to rise, not just one lucky genome.
4. **Operator mistakes must fail loudly and early without panics or damage:** overwriting an existing run, resuming a missing run or with a different opponent pool, duplicate or malformed `--opponent`, conflicting flags with `--resume`, invalid settings (`training_run::resuming_with_a_different_pool_or_no_run_is_refused`, `invalid_setups_are_refused_before_anything_is_written`, `train_smoke::bad_input_fails_with_a_clear_message_and_no_panic`, `a_new_run_never_overwrites_an_existing_one`, `run_dir::a_corrupt_checkpoint_is_an_error_not_a_panic`).
5. **Speciation must actually engage early:** `neat::population::tests::default_config_forms_several_species_within_the_first_generations` (fails with the old threshold of 3.0: one species for dozens of generations).

Also pinned: scores run +1..-1 at every table size, placements add up and reproduce the mean (`evaluate::tests`), the event log trims cleanly (`run_dir::tests::events_append_as_lines_and_truncation_drops_the_tail`), the rendered rows show every headline number (`train_output::tests`), and `--quiet` prints only the summary.

---

## File Structure

```
neat/Cargo.toml, neat/src/{config,species,population,lib}.rs     # Task 1
sim/src/lib.rs                                                    # `pub mod training;`
sim/src/training/mod.rs                                           # module root and re-exports
sim/src/training/evaluate.rs     # Task 2: scores, common random numbers
sim/src/training/config.rs       # Task 3: TrainConfig
sim/src/training/events.rs       # Task 4: the event stream
sim/src/training/run_dir.rs      # Task 5: files, atomic checkpoints
sim/src/training/trainer.rs      # Task 6: Trainer, TrainObserver
sim/tests/training_run.rs        # Task 6: end-to-end trainer tests
cli/Cargo.toml, cli/src/main.rs  # Task 7
cli/src/train_args.rs, cli/src/train_output.rs, cli/src/train.rs   # Task 7
cli/tests/train_smoke.rs         # Task 8
docs/ROADMAP.md, docs/ARCHITECTURE.md, docs/BUILDING.md, docs/baselines/pre-neat/README.md, the spec   # Task 9
```

Not in this phase: the browser dashboard, decision samples, hall-of-fame/co-evolution and curriculum (10e), evolving the exchange step, the role feature, and the full how-to-train guide.

---

### Task 1: `neat`: resumable population state and a sane speciation threshold

**Files:**
- Modify: `neat/Cargo.toml`, `neat/src/config.rs`, `neat/src/species.rs`, `neat/src/population.rs`, `neat/src/lib.rs`

**Why:** Checkpoint/resume (spec section 7) needs the whole population state, including the RNG, to round-trip through JSON. And a real training probe showed the starting threshold of 3.0 is ~10x the distance scale (random initial weights put genomes ~0.3 apart), so a whole run sat in one species for dozens of generations.

**Interfaces:**
- Consumes: `neat::Population`, `InnovationTracker`, `Species`, `Genome` (all already serializable except the RNG and `Species`).
- Produces: `PopulationState` (serde), `Population::snapshot(&self) -> PopulationState`, `Population::restore(PopulationState) -> Result<Population, NeatError>`; `SpeciesStats` and `GenerationReport` gain `Deserialize`; `Population` uses `rand::rngs::Xoshiro256PlusPlus` (its state serializes; `StdRng`'s does not); `NeatConfig::default()` threshold 0.5 / min 0.1.

- [ ] **Step 1: Edit**

In `neat/src/population.rs` (Tests first: resume equivalence, restore validation, snapshot-mid-generation), replace:

```rust
    #[test]
    fn same_seed_and_fitness_give_identical_populations() {
```

with:

```rust
    fn run_generations(population: &mut Population, from: u32, to: u32) {
        for generation in from..to {
            let fitness = (0..30)
                .map(|i| f64::from((i * 7 + generation) % 11))
                .collect();
            population.set_fitness(fitness);
            population.advance();
        }
    }

    #[test]
    fn a_restored_snapshot_continues_exactly_like_the_original() {
        let mut straight = Population::new(2, tiny_config(), 5).unwrap();
        run_generations(&mut straight, 0, 8);

        let mut first_half = Population::new(2, tiny_config(), 5).unwrap();
        run_generations(&mut first_half, 0, 4);
        // Through JSON, as a checkpoint file would be.
        let json = serde_json::to_string(&first_half.snapshot()).unwrap();
        let mut resumed = Population::restore(serde_json::from_str(&json).unwrap()).unwrap();
        assert_eq!(resumed.generation(), 4);
        run_generations(&mut resumed, 4, 8);

        assert_eq!(
            serde_json::to_string(&resumed.snapshot()).unwrap(),
            serde_json::to_string(&straight.snapshot()).unwrap()
        );
        assert_eq!(
            serde_json::to_string(resumed.genomes()).unwrap(),
            serde_json::to_string(straight.genomes()).unwrap()
        );
    }

    #[test]
    fn restore_rejects_a_snapshot_that_does_not_fit_its_config() {
        let population = Population::new(2, tiny_config(), 1).unwrap();
        let mut value = serde_json::to_value(population.snapshot()).unwrap();
        value["genomes"].as_array_mut().unwrap().pop();
        let state = serde_json::from_value(value).unwrap();
        assert!(Population::restore(state).is_err());
    }

    #[test]
    #[should_panic(expected = "fitness is pending")]
    fn snapshotting_mid_generation_panics() {
        let mut population = Population::new(2, tiny_config(), 1).unwrap();
        population.set_fitness(vec![0.0; 30]);
        let _ = population.snapshot();
    }

    #[test]
    fn same_seed_and_fitness_give_identical_populations() {
```

- [ ] **Step 2: Edit**

In `neat/src/population.rs` (Test first: the default threshold forms species early), replace:

```rust
    #[test]
    fn default_config_keeps_the_species_count_stable() {
```

with:

```rust
    #[test]
    fn default_config_forms_several_species_within_the_first_generations() {
        // Random initial weights put genomes ~0.3 apart; a threshold far
        // above that would hold the whole population in one species for
        // dozens of generations, with no protection for new structure.
        let probe = [0.3, -0.7, 0.1, 0.9, -0.2, 0.5, -0.4, 0.8, 0.0, -0.6];
        let mut population = Population::new(10, NeatConfig::default(), 21).unwrap();
        let mut scratch = Vec::new();
        let mut species_at_generation_14 = 0;
        for generation in 0..15 {
            let fitness: Vec<f64> = population
                .genomes()
                .iter()
                .map(|g| crate::Network::compile(g).activate(&probe, &mut scratch))
                .collect();
            population.set_fitness(fitness);
            let report = population.advance();
            if generation == 14 {
                species_at_generation_14 = report.species.len();
            }
        }
        assert!(
            species_at_generation_14 >= 3,
            "only {species_at_generation_14} species after 15 generations"
        );
    }

    #[test]
    fn default_config_keeps_the_species_count_stable() {
```

- [ ] **Step 3: Run (expect failure)**

Run: `cargo test -p neat`

Expected: FAIL (compile errors: `snapshot`, `restore` are not defined).

- [ ] **Step 4: Edit**

In `neat/Cargo.toml` (rand's `serde` feature makes `Xoshiro256PlusPlus` (de)serializable), replace:

```toml
rand = "0.10.3"
```

with:

```toml
rand = { version = "0.10.3", features = ["serde"] }
```

- [ ] **Step 5: Edit**

In `neat/src/species.rs`, replace:

```rust
use serde::Serialize;
```

with:

```rust
use serde::{Deserialize, Serialize};
```

- [ ] **Step 6: Edit**

In `neat/src/species.rs`, replace:

```rust
#[derive(Debug, Clone)]
pub(crate) struct Species {
```

with:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Species {
```

- [ ] **Step 7: Edit**

In `neat/src/species.rs`, replace:

```rust
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpeciesStats
```

with:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeciesStats
```

- [ ] **Step 8: Edit**

In `neat/src/config.rs`, replace:

```rust
compatibility_threshold: 3.0,
```

with:

```rust
compatibility_threshold: 0.5,
```

- [ ] **Step 9: Edit**

In `neat/src/config.rs`, replace:

```rust
min_compatibility_threshold: 0.3,
```

with:

```rust
min_compatibility_threshold: 0.1,
```

- [ ] **Step 10: Edit**

In `neat/src/config.rs`, replace:

```rust
    /// Starting speciation threshold; adapted to hold `target_species`.
```

with:

```rust
    /// Starting speciation threshold; adapted to hold `target_species`.
    /// Distances here are small (random initial weights in `[-1, 1]` put
    /// two genomes about 0.3 apart), so a threshold of 3.0 would keep a
    /// whole run in one species for dozens of generations.
```

- [ ] **Step 11: Edit**

In `neat/src/population.rs`, replace:

```rust
use rand::rngs::StdRng;
```

with:

```rust
use rand::rngs::Xoshiro256PlusPlus;
```

- [ ] **Step 12: Edit**

In `neat/src/population.rs`, replace:

```rust
use serde::Serialize;
```

with:

```rust
use serde::{Deserialize, Serialize};
```

- [ ] **Step 13: Edit**

In `neat/src/population.rs`, replace:

```rust
//! Everything random flows through one seeded `StdRng`,
```

with:

```rust
//! Everything random flows through one seeded `Xoshiro256PlusPlus` (a
//! generator whose state serializes, which is what makes a run resumable),
```

- [ ] **Step 14: Edit**

In `neat/src/population.rs`, replace:

```rust
    rng: StdRng,
    tracker: InnovationTracker,
    genomes: Vec<Genome>,
    fitness:
```

with:

```rust
    rng: Xoshiro256PlusPlus,
    tracker: InnovationTracker,
    genomes: Vec<Genome>,
    fitness:
```

- [ ] **Step 15: Edit**

In `neat/src/population.rs`, replace:

```rust
let mut rng = StdRng::seed_from_u64(seed);
```

with:

```rust
let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
```

- [ ] **Step 16: Edit**

In `neat/src/population.rs`, replace:

```rust
pub struct Population {
```

with:

```rust
/// Everything needed to continue a run exactly where it stopped: the
/// generator state, innovation registry, species bookkeeping and the
/// current genomes. Taken *between* generations (after `advance`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationState {
    config: NeatConfig,
    rng: Xoshiro256PlusPlus,
    tracker: InnovationTracker,
    genomes: Vec<Genome>,
    species: Vec<Species>,
    next_species_id: u32,
    threshold: f64,
    generation: u32,
    best: Option<(Genome, f64)>,
}

pub struct Population {
```

- [ ] **Step 17: Edit**

In `neat/src/population.rs`, replace:

```rust
    #[must_use]
    pub fn genomes
```

with:

```rust
    /// Captures the population between generations; see `PopulationState`.
    ///
    /// # Panics
    ///
    /// Panics if fitness has been recorded but not yet consumed by
    /// `advance`: that half-finished generation cannot be resumed.
    #[must_use]
    pub fn snapshot(&self) -> PopulationState {
        assert!(
            self.fitness.is_none(),
            "snapshot between generations: fitness is pending"
        );
        PopulationState {
            config: self.config.clone(),
            rng: self.rng.clone(),
            tracker: self.tracker.clone(),
            genomes: self.genomes.clone(),
            species: self.species.clone(),
            next_species_id: self.next_species_id,
            threshold: self.threshold,
            generation: self.generation,
            best: self.best.clone(),
        }
    }

    /// Rebuilds a population from a snapshot (for example one read back
    /// from JSON), checking it is internally consistent.
    ///
    /// # Errors
    ///
    /// Returns `NeatError::InvalidConfig` if the config is invalid or the
    /// genome count differs from `population_size`, and
    /// `NeatError::InvalidGenome` if the genomes disagree on their input
    /// count.
    pub fn restore(state: PopulationState) -> Result<Self, NeatError> {
        state.config.validate()?;
        if state.genomes.len() != state.config.population_size {
            return Err(NeatError::InvalidConfig(format!(
                "snapshot holds {} genomes but population_size is {}",
                state.genomes.len(),
                state.config.population_size
            )));
        }
        let inputs = state.genomes[0].num_inputs();
        if state.genomes.iter().any(|g| g.num_inputs() != inputs) {
            return Err(NeatError::InvalidGenome(
                "snapshot genomes disagree on their input count".into(),
            ));
        }
        Ok(Self {
            config: state.config,
            rng: state.rng,
            tracker: state.tracker,
            genomes: state.genomes,
            fitness: None,
            species: state.species,
            next_species_id: state.next_species_id,
            threshold: state.threshold,
            generation: state.generation,
            best: state.best,
        })
    }

    #[must_use]
    pub fn genomes
```

- [ ] **Step 18: Edit**

In `neat/src/population.rs`, replace:

```rust
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GenerationReport
```

with:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationReport
```

- [ ] **Step 19: Edit**

In `neat/src/lib.rs`, replace:

```rust
pub use population::{GenerationReport, Population};
```

with:

```rust
pub use population::{GenerationReport, Population, PopulationState};
```

- [ ] **Step 20: Run (expect success)**

Run: `cargo fmt -p neat && cargo test -p neat`

Expected: PASS (all `neat` tests including the 4 new ones and XOR).

- [ ] **Step 21: Edit**

In `neat/src/config.rs` (Sensitivity check: temporarily restore the old threshold), replace:

```rust
compatibility_threshold: 0.5,
```

with:

```rust
compatibility_threshold: 3.0,
```

- [ ] **Step 22: Run (expect failure)**

Run: `cargo test -p neat forms_several`

Expected: FAIL: `only 1 species after 15 generations`.

- [ ] **Step 23: Edit**

In `neat/src/config.rs` (Revert the deliberate break), replace:

```rust
compatibility_threshold: 3.0,
```

with:

```rust
compatibility_threshold: 0.5,
```

- [ ] **Step 24: Run (expect success)**

Run: `cargo test -p neat`

Expected: PASS.

- [ ] **Step 25: Commit**

```bash
git add neat Cargo.lock
git commit -F - <<'EOF'
neat: resumable Population state and a speciation threshold at the distance scale (Phase 10c)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F3sC3JdcgTFAox87bBuikA
EOF
```


### Task 2: Fitness evaluation

**Files:**
- Modify: `sim/src/lib.rs`
- Create: `sim/src/training/mod.rs`, `sim/src/training/evaluate.rs`

**Why:** Common random numbers: the seat the candidate sits in and the opponents it faces are a pure function of the match seed, so every genome in a generation plays the same deals, seats and opponents. Opponents are *sampled* per match (not cyclically rotated): a real run showed table position relative to the other players is a large effect, and cyclic rotation keeps neighbour order fixed.

**Interfaces:**
- Consumes: `sim::{run_match, MatchConfig, Strategy}`, `engine::roles_for_player_count`, `rand::rngs::Xoshiro256PlusPlus`.
- Produces: `TableSpec { player_count, deck_variant, duplicate_rule, rounds }`; `Opponents::{Mixed(&[Arc<dyn Strategy>]), Only(&Arc<dyn Strategy>)}`; `Score { mean, std_error, matches, placements }`; `evaluate(&Arc<dyn Strategy>, &TableSpec, Opponents, &[u64]) -> Score`; `match_seed(run_seed, stream, index) -> u64`; `role_score(Role, player_count) -> f64` (+1 best role .. -1 worst).

- [ ] **Step 1: Edit**

In `sim/src/lib.rs`, replace:

```rust
pub mod strategy;
```

with:

```rust
pub mod strategy;
pub mod training;
```

- [ ] **Step 2: Create file**

Create `sim/src/training/mod.rs` (Module root (grows in Tasks 3-6)):

```rust
//! Training evolved players: fitness evaluation, the run's files and
//! event stream, and the generational loop. See
//! docs/superpowers/specs/2026-10-08-neat-engine-design.md, sections 6-8.

pub mod evaluate;

pub use evaluate::{evaluate, match_seed, role_score, Opponents, Score, TableSpec};
```

- [ ] **Step 3: Create file**

Create `sim/src/training/evaluate.rs` (Tests first):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LowestLegal, RandomLegal};

    const TABLE: TableSpec = TableSpec {
        player_count: 4,
        deck_variant: DeckVariant::Single,
        duplicate_rule: DuplicateRule::FirstDealtWins,
        rounds: 4,
    };

    fn seeds(count: u64) -> Vec<u64> {
        (0..count).map(|i| match_seed(7, 0, i)).collect()
    }

    #[test]
    fn role_scores_run_from_plus_one_to_minus_one_at_every_table_size() {
        for players in 3..=6 {
            let roles = roles_for_player_count(players).unwrap();
            let scores: Vec<f64> = roles.iter().map(|&r| role_score(r, players)).collect();
            assert!((scores[0] - 1.0).abs() < 1e-12, "{players}: {scores:?}");
            assert!(
                (scores[roles.len() - 1] + 1.0).abs() < 1e-12,
                "{players}: {scores:?}"
            );
            assert!(
                scores.windows(2).all(|w| w[0] > w[1]),
                "strictly decreasing: {scores:?}"
            );
        }
    }

    #[test]
    fn match_seeds_are_stable_and_differ_by_stream_and_index() {
        assert_eq!(match_seed(1, 2, 3), match_seed(1, 2, 3));
        let all = [
            match_seed(1, 2, 3),
            match_seed(1, 2, 4),
            match_seed(1, 3, 3),
            match_seed(2, 2, 3),
        ];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn evaluation_is_deterministic() {
        let candidate: Arc<dyn Strategy> = Arc::new(RandomLegal);
        let pool: Vec<Arc<dyn Strategy>> = vec![Arc::new(LowestLegal), Arc::new(RandomLegal)];
        let run = || evaluate(&candidate, &TABLE, Opponents::Mixed(&pool), &seeds(20));
        assert_eq!(run(), run());
    }

    #[test]
    fn placements_account_for_every_round_and_the_mean_matches_them() {
        let candidate: Arc<dyn Strategy> = Arc::new(LowestLegal);
        let opponent: Arc<dyn Strategy> = Arc::new(RandomLegal);
        let score = evaluate(&candidate, &TABLE, Opponents::Only(&opponent), &seeds(30));
        assert_eq!(score.matches, 30);
        assert_eq!(score.placements.len(), 4);
        assert_eq!(score.placements.iter().sum::<u64>(), 30 * 4);
        // Rebuild the mean from the placement counts.
        let total: f64 = score
            .placements
            .iter()
            .enumerate()
            .map(|(place, &n)| {
                f64::from(u32::try_from(n).unwrap())
                    * (1.0 - 2.0 * f64::from(u32::try_from(place).unwrap()) / 3.0)
            })
            .sum();
        assert!(
            (total / 120.0 - score.mean).abs() < 1e-9,
            "{} vs {total}",
            score.mean
        );
        assert!(score.std_error > 0.0);
    }

    #[test]
    fn a_sensible_player_beats_random_and_mirrors_itself() {
        let lowest: Arc<dyn Strategy> = Arc::new(LowestLegal);
        let random: Arc<dyn Strategy> = Arc::new(RandomLegal);
        let many = seeds(150);
        let versus_random = evaluate(&lowest, &TABLE, Opponents::Only(&random), &many);
        assert!(versus_random.mean > 0.4, "{versus_random:?}");
        let mirror = evaluate(&lowest, &TABLE, Opponents::Only(&lowest), &many);
        assert!(
            mirror.mean.abs() < 0.2,
            "a player against copies of itself is about even: {mirror:?}"
        );
        let reverse = evaluate(&random, &TABLE, Opponents::Only(&lowest), &many);
        assert!(reverse.mean < -0.4, "{reverse:?}");
    }

    #[test]
    fn a_single_match_has_no_standard_error() {
        let candidate: Arc<dyn Strategy> = Arc::new(LowestLegal);
        let score = evaluate(&candidate, &TABLE, Opponents::Only(&candidate), &seeds(1));
        assert!(score.std_error.abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "at least one match")]
    fn evaluating_zero_matches_panics() {
        let candidate: Arc<dyn Strategy> = Arc::new(LowestLegal);
        let _ = evaluate(&candidate, &TABLE, Opponents::Only(&candidate), &[]);
    }
}
```

- [ ] **Step 4: Run (expect failure)**

Run: `cargo test -p sim training::evaluate`

Expected: FAIL (compile errors: `evaluate`, `TableSpec` ... are not defined).

- [ ] **Step 5: Implement**

Insert at the very top of `sim/src/training/evaluate.rs`, above the `#[cfg(test)]` line:

```rust
//! Fitness evaluation: how well one candidate plays against opponents.
//!
//! A candidate's score is its mean finishing-role score over many
//! matches, from +1 (always President) to -1 (always last). The seat it
//! sits in and the opponents it faces are a pure function of the match
//! seed, so every genome in a generation can be evaluated on exactly the
//! same deals, seats and opponents ("common random numbers"): score
//! differences then reflect the genome, not the luck of the shuffle.

use std::sync::Arc;

use engine::{roles_for_player_count, DeckVariant, DuplicateRule, Role};
use rand::rngs::Xoshiro256PlusPlus;
use rand::{RngExt, SeedableRng};

use crate::{run_match, MatchConfig, Strategy};

/// What a candidate plays: the table and the length of each match.
#[derive(Debug, Clone, Copy)]
pub struct TableSpec {
    pub player_count: u8,
    pub deck_variant: DeckVariant,
    pub duplicate_rule: DuplicateRule,
    pub rounds: usize,
}

/// Who fills the other seats.
#[derive(Clone, Copy)]
pub enum Opponents<'a> {
    /// Each other seat independently draws from the pool, per match.
    Mixed(&'a [Arc<dyn Strategy>]),
    /// Every other seat plays this one strategy.
    Only(&'a Arc<dyn Strategy>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    pub mean: f64,
    /// Standard error of `mean` across matches (0 for a single match).
    pub std_error: f64,
    pub matches: usize,
    /// How often the candidate finished in each place, best first
    /// (`placements[0]` is President), over every round of every match.
    pub placements: Vec<u64>,
}

/// The `SplitMix64` finalizer: a cheap, well-mixed 64-bit hash.
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// The seed of match `index` in `stream` (for example generation `g`'s
/// training matches, or its champion re-evaluation) of a run.
#[must_use]
pub fn match_seed(run_seed: u64, stream: u64, index: u64) -> u64 {
    mix(mix(run_seed ^ mix(stream)) ^ index)
}

/// `+1` for the best role of a table of `player_count`, `-1` for the
/// worst, linear in between.
///
/// # Panics
///
/// Panics if `player_count` is not a supported table size (3-6).
#[must_use]
pub fn role_score(role: Role, player_count: u8) -> f64 {
    let roles = roles_for_player_count(player_count).expect("supported table size");
    let rank = roles
        .iter()
        .position(|&r| r == role)
        .expect("role belongs to this table");
    #[allow(clippy::cast_precision_loss)] // table sizes are 3-6
    let (rank, last) = (rank as f64, (roles.len() - 1) as f64);
    1.0 - 2.0 * rank / last
}

/// Mean role score of `candidate` over one match per seed in `seeds`.
///
/// # Panics
///
/// Panics if `seeds` is empty, the pool is empty, or the table size is
/// unsupported.
pub fn evaluate(
    candidate: &Arc<dyn Strategy>,
    table: &TableSpec,
    opponents: Opponents<'_>,
    seeds: &[u64],
) -> Score {
    assert!(!seeds.is_empty(), "evaluation needs at least one match");
    let players = usize::from(table.player_count);
    let roles = roles_for_player_count(table.player_count).expect("supported table size");
    let mut placements = vec![0u64; players];
    let mut per_match = Vec::with_capacity(seeds.len());
    for (index, &seed) in seeds.iter().enumerate() {
        let seat = index % players;
        let mut pick = Xoshiro256PlusPlus::seed_from_u64(mix(seed ^ 0x5EA7));
        let strategies: Vec<Arc<dyn Strategy>> = (0..players)
            .map(|s| {
                if s == seat {
                    return candidate.clone();
                }
                match opponents {
                    Opponents::Only(only) => only.clone(),
                    Opponents::Mixed(pool) => pool[pick.random_range(0..pool.len())].clone(),
                }
            })
            .collect();
        let result = run_match(
            &MatchConfig {
                player_count: table.player_count,
                deck_variant: table.deck_variant,
                duplicate_rule: table.duplicate_rule,
                rounds: table.rounds,
                seed,
            },
            &strategies,
        );
        let mut total = 0.0;
        for round in &result.role_history {
            total += role_score(round[seat], table.player_count);
            let place = roles
                .iter()
                .position(|&r| r == round[seat])
                .expect("role belongs to this table");
            placements[place] += 1;
        }
        #[allow(clippy::cast_precision_loss)]
        per_match.push(total / result.role_history.len() as f64);
    }
    #[allow(clippy::cast_precision_loss)]
    let count = per_match.len() as f64;
    let mean = per_match.iter().sum::<f64>() / count;
    let std_error = if per_match.len() < 2 {
        0.0
    } else {
        let variance = per_match.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / (count - 1.0);
        (variance / count).sqrt()
    };
    Score {
        mean,
        std_error,
        matches: per_match.len(),
        placements,
    }
}
```

- [ ] **Step 6: Run (expect success)**

Run: `cargo fmt -p sim && cargo test -p sim training::evaluate`

Expected: PASS (7 tests: scores span +1..-1 at every table size; seeds stable and distinct; deterministic; placements add up and reproduce the mean; a sensible player beats random and is even against itself).

- [ ] **Step 7: Commit**

```bash
git add sim
git commit -F - <<'EOF'
sim: add fitness evaluation with common random numbers (Phase 10c)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F3sC3JdcgTFAox87bBuikA
EOF
```


### Task 3: `TrainConfig`

**Files:**
- Modify: `sim/src/training/mod.rs`
- Create: `sim/src/training/config.rs`

**Interfaces:**
- Consumes: `neat::NeatConfig`, `TableSpec` (Task 2).
- Produces: `TrainConfig { seed, player_count, deck, duplicate_rule, rounds_per_match, matches_per_genome, reeval_matches, generations, neat, opponent_specs }` (serde; `validate() -> Result<(), String>`; crate-private `table() -> TableSpec`); `DeckChoice`, `DuplicateChoice`; test-only `config::test_support::sample()`.

- [ ] **Step 1: Edit**

In `sim/src/training/mod.rs`, replace:

```rust
pub mod evaluate;
```

with:

```rust
pub mod config;
pub mod evaluate;
```

- [ ] **Step 2: Edit**

In `sim/src/training/mod.rs`, replace:

```rust
pub use evaluate::{
```

with:

```rust
pub use config::{DeckChoice, DuplicateChoice, TrainConfig};
pub use evaluate::{
```

- [ ] **Step 3: Create file**

Create `sim/src/training/config.rs` (Tests first):

```rust
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    pub fn sample() -> TrainConfig {
        TrainConfig {
            seed: 1,
            player_count: 4,
            deck: DeckChoice::Single,
            duplicate_rule: DuplicateChoice::FirstDealtWins,
            rounds_per_match: 4,
            matches_per_genome: 6,
            reeval_matches: 8,
            generations: 3,
            neat: NeatConfig {
                population_size: 12,
                ..NeatConfig::default()
            },
            opponent_specs: vec!["lowest-legal".into()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::sample;
    use super::*;

    #[test]
    fn the_sample_config_is_valid_and_round_trips_through_json() {
        let config = sample();
        assert_eq!(config.validate(), Ok(()));
        let restored: TrainConfig =
            serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
        assert_eq!(restored, config);
    }

    #[test]
    fn each_invalid_setting_is_named() {
        let cases: Vec<(TrainConfig, &str)> = vec![
            (
                TrainConfig {
                    player_count: 2,
                    ..sample()
                },
                "player_count",
            ),
            (
                TrainConfig {
                    rounds_per_match: 0,
                    ..sample()
                },
                "rounds_per_match",
            ),
            (
                TrainConfig {
                    matches_per_genome: 0,
                    ..sample()
                },
                "matches_per_genome",
            ),
            (
                TrainConfig {
                    reeval_matches: 0,
                    ..sample()
                },
                "reeval_matches",
            ),
            (
                TrainConfig {
                    generations: 0,
                    ..sample()
                },
                "generations",
            ),
            (
                TrainConfig {
                    opponent_specs: vec![],
                    ..sample()
                },
                "pool is empty",
            ),
        ];
        for (config, expected) in cases {
            let error = config.validate().unwrap_err();
            assert!(error.contains(expected), "{expected}: {error}");
        }
    }

    #[test]
    fn table_converts_to_engine_types() {
        let table = TrainConfig {
            deck: DeckChoice::Double,
            duplicate_rule: DuplicateChoice::LastDealtWins,
            ..sample()
        }
        .table();
        assert_eq!(table.deck_variant, DeckVariant::Double);
        assert_eq!(table.duplicate_rule, DuplicateRule::LastDealtWins);
        assert_eq!(table.rounds, 4);
    }
}
```

- [ ] **Step 4: Run (expect failure)**

Run: `cargo test -p sim training::config`

Expected: FAIL (compile errors: `TrainConfig` is not defined).

- [ ] **Step 5: Implement**

Insert at the very top of `sim/src/training/config.rs`, above the `#[cfg(test)]` line:

```rust
//! The settings of a training run, saved with it so a run can be resumed
//! (and later understood) without remembering the command line.

use engine::{DeckVariant, DuplicateRule};
use neat::NeatConfig;
use serde::{Deserialize, Serialize};

use super::evaluate::TableSpec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeckChoice {
    Single,
    Double,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DuplicateChoice {
    FirstDealtWins,
    LastDealtWins,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainConfig {
    /// Seeds everything: the initial population, every evolution step and
    /// every evaluation match.
    pub seed: u64,
    pub player_count: u8,
    pub deck: DeckChoice,
    pub duplicate_rule: DuplicateChoice,
    /// Rounds per evaluation match (role carry-over between rounds).
    pub rounds_per_match: usize,
    /// Matches each genome plays per generation. Every genome plays the
    /// same matches (same deals, seats and opponents).
    pub matches_per_genome: usize,
    /// Matches in each fresh-seed re-evaluation of the generation's
    /// champion, against the mixed pool and against each opponent alone.
    pub reeval_matches: usize,
    /// Total generations to run (a resumed run continues up to this).
    pub generations: u32,
    pub neat: NeatConfig,
    /// The opponent pool as the `--strategy`-style specs that built it,
    /// so a resume can rebuild exactly the same pool.
    pub opponent_specs: Vec<String>,
}

impl TrainConfig {
    pub(crate) fn table(&self) -> TableSpec {
        TableSpec {
            player_count: self.player_count,
            deck_variant: match self.deck {
                DeckChoice::Single => DeckVariant::Single,
                DeckChoice::Double => DeckVariant::Double,
            },
            duplicate_rule: match self.duplicate_rule {
                DuplicateChoice::FirstDealtWins => DuplicateRule::FirstDealtWins,
                DuplicateChoice::LastDealtWins => DuplicateRule::LastDealtWins,
            },
            rounds: self.rounds_per_match,
        }
    }

    /// # Errors
    ///
    /// Returns a message naming the first invalid setting.
    pub fn validate(&self) -> Result<(), String> {
        if !(3..=6).contains(&self.player_count) {
            return Err(format!(
                "player_count {} is not a table size (3-6)",
                self.player_count
            ));
        }
        for (name, value) in [
            ("rounds_per_match", self.rounds_per_match),
            ("matches_per_genome", self.matches_per_genome),
            ("reeval_matches", self.reeval_matches),
        ] {
            if value == 0 {
                return Err(format!("{name} must be at least 1"));
            }
        }
        if self.generations == 0 {
            return Err("generations must be at least 1".into());
        }
        if self.opponent_specs.is_empty() {
            return Err("the opponent pool is empty".into());
        }
        self.neat.validate().map_err(|e| e.to_string())
    }
}
```

- [ ] **Step 6: Run (expect success)**

Run: `cargo fmt -p sim && cargo test -p sim training::config`

Expected: PASS (3 tests: valid and JSON-stable; every invalid setting named; engine conversions).

- [ ] **Step 7: Commit**

```bash
git add sim
git commit -F - <<'EOF'
sim: add TrainConfig (Phase 10c)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F3sC3JdcgTFAox87bBuikA
EOF
```


### Task 4: The event stream

**Files:**
- Modify: `sim/src/training/mod.rs`
- Create: `sim/src/training/events.rs`

**Interfaces:**
- Consumes: `Score` (Task 2), `TrainConfig` (Task 3), `neat::SpeciesStats` (now `Deserialize`, Task 1).
- Produces: `Event::{RunStart(Box<RunStart>), Generation(Box<GenerationEvent>), RunEnd(RunEnd)}` (JSON, tagged `type`, one per line in `events.jsonl`); `GenerationEvent`, `FitnessStats`, `ChampionStats`, `OpponentStat`, `Complexity`, `ScoreStat` (from `Score`), `SCHEMA_VERSION = 1`. Additive changes only; bump the version for removals or renames.

- [ ] **Step 1: Edit**

In `sim/src/training/mod.rs`, replace:

```rust
pub mod evaluate;
```

with:

```rust
pub mod evaluate;
pub mod events;
```

- [ ] **Step 2: Edit**

In `sim/src/training/mod.rs`, replace:

```rust
pub use evaluate::{evaluate, match_seed, role_score, Opponents, Score, TableSpec};
```

with:

```rust
pub use evaluate::{evaluate, match_seed, role_score, Opponents, Score, TableSpec};
pub use events::{Event, GenerationEvent, RunEnd, RunStart, ScoreStat, SCHEMA_VERSION};
```

- [ ] **Step 3: Create file**

Create `sim/src/training/events.rs` (Tests first):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn stat() -> ScoreStat {
        ScoreStat {
            mean: 0.5,
            std_error: 0.02,
            matches: 10,
            placements: vec![5, 3, 1, 1],
        }
    }

    #[test]
    fn events_are_tagged_by_type_and_round_trip() {
        let end = Event::RunEnd(RunEnd {
            generations_completed: 3,
            best_generation: Some(2),
            best_reeval: Some(stat()),
            elapsed_secs: 1.5,
        });
        let line = serde_json::to_string(&end).unwrap();
        assert!(line.contains(r#""type":"run_end""#), "{line}");
        assert!(!line.contains('\n'), "one event is one line");
        assert_eq!(serde_json::from_str::<Event>(&line).unwrap(), end);
    }

    #[test]
    fn a_score_converts_to_a_stat_without_losing_anything() {
        let score = Score {
            mean: -0.25,
            std_error: 0.1,
            matches: 7,
            placements: vec![1, 2, 3, 4],
        };
        let stat = ScoreStat::from(score.clone());
        assert_eq!(
            (stat.mean, stat.std_error, stat.matches),
            (score.mean, score.std_error, score.matches)
        );
        assert_eq!(stat.placements, score.placements);
    }
}
```

- [ ] **Step 4: Run (expect failure)**

Run: `cargo test -p sim training::events`

Expected: FAIL (compile errors: `Event`, `ScoreStat` ... are not defined).

- [ ] **Step 5: Implement**

Insert at the very top of `sim/src/training/events.rs`, above the `#[cfg(test)]` line:

```rust
//! The training event stream: one JSON object per line in
//! `events.jsonl`. Everything the terminal, the dashboard and later
//! analysis show comes from these events, so they cannot disagree.
//! Changes must be additive, with `SCHEMA_VERSION` bumped for any
//! removal or rename.

use neat::SpeciesStats;
use serde::{Deserialize, Serialize};

use super::config::TrainConfig;
use super::evaluate::Score;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    RunStart(Box<RunStart>),
    Generation(Box<GenerationEvent>),
    RunEnd(RunEnd),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunStart {
    pub schema_version: u32,
    pub config: TrainConfig,
    /// Display names of the opponent pool, in pool order.
    pub opponents: Vec<String>,
    pub feature_names: Vec<String>,
    /// `Some(generation)` when this start is a resume.
    pub resumed_from_generation: Option<u32>,
}

/// A candidate's score with the statistics around it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreStat {
    pub mean: f64,
    pub std_error: f64,
    pub matches: usize,
    /// Finishing places, best first, over every round played.
    pub placements: Vec<u64>,
}

impl From<Score> for ScoreStat {
    fn from(score: Score) -> Self {
        Self {
            mean: score.mean,
            std_error: score.std_error,
            matches: score.matches,
            placements: score.placements,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitnessStats {
    pub best: f64,
    pub mean: f64,
    pub median: f64,
    pub min: f64,
    pub std_dev: f64,
    /// Ten equal-width buckets from `min` to `best`.
    pub histogram: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChampionStats {
    /// The champion's fitness in the generation that selected it.
    pub train_fitness: f64,
    /// Its score on fresh matches against the mixed pool: the honest
    /// number (the training fitness is inflated by selection).
    pub reeval: ScoreStat,
    pub hidden_nodes: usize,
    pub enabled_connections: usize,
    /// File name (inside the run directory) of this champion's genome.
    pub genome_file: String,
    pub is_new_best: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpponentStat {
    pub name: String,
    /// The champion against tables made only of this opponent.
    pub score: ScoreStat,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Complexity {
    pub mean_hidden_nodes: f64,
    pub mean_enabled_connections: f64,
    pub innovation_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationEvent {
    pub generation: u32,
    /// Wall time since the run was first started (summed across resumes).
    pub elapsed_secs: f64,
    pub generation_secs: f64,
    /// Rounds played this generation, evaluation plus re-evaluation.
    pub rounds_evaluated: u64,
    pub total_rounds: u64,
    pub rounds_per_sec: f64,
    pub fitness: FitnessStats,
    pub champion: ChampionStats,
    pub opponents: Vec<OpponentStat>,
    pub species: Vec<SpeciesStats>,
    pub compatibility_threshold: f64,
    pub complexity: Complexity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunEnd {
    pub generations_completed: u32,
    pub best_generation: Option<u32>,
    pub best_reeval: Option<ScoreStat>,
    pub elapsed_secs: f64,
}
```

- [ ] **Step 6: Run (expect success)**

Run: `cargo fmt -p sim && cargo test -p sim training::events`

Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add sim
git commit -F - <<'EOF'
sim: define the training event stream (Phase 10c)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F3sC3JdcgTFAox87bBuikA
EOF
```


### Task 5: The run directory: atomic checkpoints, event log, champions

**Files:**
- Modify: `sim/src/training/mod.rs`
- Create: `sim/src/training/run_dir.rs`

**Interfaces:**
- Consumes: `TrainConfig`, `Event`, `ScoreStat`, `SCHEMA_VERSION`, `neat::{Genome, PopulationState}`, `sim::GenomeFile`.
- Produces: `RunDir::{create_new, open_existing, path, write_config, append_event, truncate_events_from, write_checkpoint, read_checkpoint, write_champion, write_best}`; `Checkpoint`, `BestRecord`; `load_config(&Path) -> Result<TrainConfig, TrainError>`; `TrainError::{Io, Config, Checkpoint, Mismatch}` (Display). Files: `config.json`, `events.jsonl`, `checkpoint.json` (replaced atomically: write temp, fsync, rename), `gen-NNNN.json`, `best.json` (both playable `GenomeFile`s).

- [ ] **Step 1: Edit**

In `sim/src/training/mod.rs`, replace:

```rust
pub mod evaluate;
pub mod events;
```

with:

```rust
pub mod evaluate;
pub mod events;
pub mod run_dir;
```

- [ ] **Step 2: Edit**

In `sim/src/training/mod.rs`, replace:

```rust
pub use events::{Event, GenerationEvent, RunEnd, RunStart, ScoreStat, SCHEMA_VERSION};
```

with:

```rust
pub use events::{Event, GenerationEvent, RunEnd, RunStart, ScoreStat, SCHEMA_VERSION};
pub use run_dir::{load_config, RunDir, TrainError};
```

- [ ] **Step 3: Create file**

Create `sim/src/training/run_dir.rs` (Tests first):

```rust
#[cfg(test)]
mod tests {
    use neat::{NeatConfig, Population};

    use super::super::config::test_support::sample;
    use super::super::events::{RunEnd, RunStart};
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("arschloch-rundir-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn checkpoint() -> Checkpoint {
        let population = Population::new(
            crate::FEATURE_COUNT,
            NeatConfig {
                population_size: 4,
                ..NeatConfig::default()
            },
            1,
        )
        .unwrap();
        Checkpoint {
            schema_version: SCHEMA_VERSION,
            config: sample(),
            opponent_names: vec!["LowestLegal".into()],
            population: population.snapshot(),
            best: None,
            total_rounds: 10,
            elapsed_secs: 1.0,
        }
    }

    fn generation_event(generation: u32) -> Event {
        let stat = ScoreStat {
            mean: 0.0,
            std_error: 0.0,
            matches: 1,
            placements: vec![1],
        };
        Event::Generation(Box::new(super::super::events::GenerationEvent {
            generation,
            elapsed_secs: 0.0,
            generation_secs: 0.0,
            rounds_evaluated: 0,
            total_rounds: 0,
            rounds_per_sec: 0.0,
            fitness: super::super::events::FitnessStats {
                best: 0.0,
                mean: 0.0,
                median: 0.0,
                min: 0.0,
                std_dev: 0.0,
                histogram: vec![],
            },
            champion: super::super::events::ChampionStats {
                train_fitness: 0.0,
                reeval: stat,
                hidden_nodes: 0,
                enabled_connections: 0,
                genome_file: String::new(),
                is_new_best: false,
            },
            opponents: vec![],
            species: vec![],
            compatibility_threshold: 0.5,
            complexity: super::super::events::Complexity {
                mean_hidden_nodes: 0.0,
                mean_enabled_connections: 0.0,
                innovation_count: 0,
            },
        }))
    }

    #[test]
    fn a_new_run_refuses_a_directory_that_already_holds_one() {
        let dir = temp_dir("refuse");
        let run = RunDir::create_new(&dir).unwrap();
        run.write_checkpoint(&checkpoint()).unwrap();
        let error = RunDir::create_new(&dir).err().unwrap();
        assert!(matches!(error, TrainError::Config(_)), "{error}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn opening_a_directory_without_a_checkpoint_fails_clearly() {
        let dir = temp_dir("nocheckpoint");
        fs::create_dir_all(&dir).unwrap();
        let error = RunDir::open_existing(&dir).err().unwrap();
        assert!(error.to_string().contains("no checkpoint.json"), "{error}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_checkpoint_survives_a_round_trip_and_leaves_no_temporary_file() {
        let dir = temp_dir("roundtrip");
        let run = RunDir::create_new(&dir).unwrap();
        let original = checkpoint();
        run.write_checkpoint(&original).unwrap();
        let loaded = run.read_checkpoint().unwrap();
        assert_eq!(loaded.config, original.config);
        assert_eq!(
            serde_json::to_string(&loaded.population).unwrap(),
            serde_json::to_string(&original.population).unwrap()
        );
        assert!(!run.path("checkpoint.json.tmp").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn load_config_reads_the_runs_settings_back() {
        let dir = temp_dir("loadconfig");
        let run = RunDir::create_new(&dir).unwrap();
        run.write_checkpoint(&checkpoint()).unwrap();
        assert_eq!(load_config(&dir).unwrap(), sample());
        assert!(load_config(&dir.join("missing")).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_corrupt_checkpoint_is_an_error_not_a_panic() {
        let dir = temp_dir("corrupt");
        let run = RunDir::create_new(&dir).unwrap();
        fs::write(run.path("checkpoint.json"), "{ truncated").unwrap();
        assert!(matches!(
            run.read_checkpoint(),
            Err(TrainError::Checkpoint(_))
        ));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn champions_are_playable_genome_files() {
        let dir = temp_dir("champions");
        let run = RunDir::create_new(&dir).unwrap();
        let genome = neat::Population::restore(checkpoint().population)
            .unwrap()
            .genomes()[0]
            .clone();
        let name = run.write_champion(7, &genome).unwrap();
        assert_eq!(name, "gen-0007.json");
        run.write_best(&genome).unwrap();
        assert!(crate::NeatStrategy::from_file(&run.path(&name)).is_ok());
        assert!(crate::NeatStrategy::from_file(&run.path("best.json")).is_ok());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn events_append_as_lines_and_truncation_drops_the_tail() {
        let dir = temp_dir("events");
        let run = RunDir::create_new(&dir).unwrap();
        run.append_event(&Event::RunStart(Box::new(RunStart {
            schema_version: SCHEMA_VERSION,
            config: sample(),
            opponents: vec![],
            feature_names: vec![],
            resumed_from_generation: None,
        })))
        .unwrap();
        for generation in 0..4 {
            run.append_event(&generation_event(generation)).unwrap();
        }
        run.append_event(&Event::RunEnd(RunEnd {
            generations_completed: 4,
            best_generation: None,
            best_reeval: None,
            elapsed_secs: 0.0,
        }))
        .unwrap();
        run.truncate_events_from(2).unwrap();
        let text = fs::read_to_string(run.path("events.jsonl")).unwrap();
        let events: Vec<Event> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(events.len(), 3, "start + generations 0 and 1");
        assert!(matches!(events[0], Event::RunStart(_)));
        assert!(matches!(&events[2], Event::Generation(e) if e.generation == 1));
        fs::remove_dir_all(&dir).unwrap();
    }
}
```

- [ ] **Step 4: Run (expect failure)**

Run: `cargo test -p sim training::run_dir`

Expected: FAIL (compile errors: `RunDir`, `Checkpoint` ... are not defined).

- [ ] **Step 5: Implement**

Insert at the very top of `sim/src/training/run_dir.rs`, above the `#[cfg(test)]` line:

```rust
//! The files of a training run, all inside one directory:
//!
//! - `config.json`: the run's settings (informational; the checkpoint is
//!   authoritative for resuming);
//! - `events.jsonl`: the event stream;
//! - `checkpoint.json`: everything needed to resume, replaced atomically
//!   after every generation;
//! - `gen-NNNN.json`: each generation's champion, as a `GenomeFile`
//!   playable with `--strategy neat:PATH`;
//! - `best.json`: the champion with the best fresh-seed score so far.

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use neat::{Genome, PopulationState};
use serde::{Deserialize, Serialize};

use super::config::TrainConfig;
use super::events::{Event, ScoreStat, SCHEMA_VERSION};
use crate::GenomeFile;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainError {
    Io(String),
    /// A setting is invalid.
    Config(String),
    /// A run directory's files are unusable.
    Checkpoint(String),
    /// A resume does not match the run it continues.
    Mismatch(String),
}

impl fmt::Display for TrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(reason) => write!(f, "file error: {reason}"),
            Self::Config(reason) => write!(f, "invalid training setup: {reason}"),
            Self::Checkpoint(reason) => write!(f, "unusable run directory: {reason}"),
            Self::Mismatch(reason) => write!(f, "cannot resume: {reason}"),
        }
    }
}

impl std::error::Error for TrainError {}

fn io_error(path: &Path, error: &std::io::Error) -> TrainError {
    TrainError::Io(format!("{}: {error}", path.display()))
}

/// The best champion so far, judged on fresh matches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BestRecord {
    pub generation: u32,
    pub reeval: ScoreStat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub schema_version: u32,
    pub config: TrainConfig,
    pub opponent_names: Vec<String>,
    pub population: PopulationState,
    pub best: Option<BestRecord>,
    pub total_rounds: u64,
    pub elapsed_secs: f64,
}

/// Reads the settings of the run in `dir` from its checkpoint (what a
/// resume needs before it can rebuild the opponent pool).
///
/// # Errors
///
/// `TrainError::Checkpoint` if `dir` holds no usable checkpoint.
pub fn load_config(dir: &Path) -> Result<TrainConfig, TrainError> {
    Ok(RunDir::open_existing(dir)?.read_checkpoint()?.config)
}

pub struct RunDir {
    root: PathBuf,
}

impl RunDir {
    /// Starts a new run directory, creating it if needed.
    ///
    /// # Errors
    ///
    /// Refuses a directory that already holds a run (use resume instead,
    /// or pick another directory), so a run is never overwritten.
    pub fn create_new(root: &Path) -> Result<Self, TrainError> {
        fs::create_dir_all(root).map_err(|e| io_error(root, &e))?;
        let dir = Self {
            root: root.to_owned(),
        };
        for existing in ["checkpoint.json", "events.jsonl"] {
            if dir.path(existing).exists() {
                return Err(TrainError::Config(format!(
                    "{} already holds a run; resume it or choose another directory",
                    root.display()
                )));
            }
        }
        Ok(dir)
    }

    /// Opens a directory that holds a run.
    ///
    /// # Errors
    ///
    /// Returns `TrainError::Checkpoint` if there is no checkpoint.
    pub fn open_existing(root: &Path) -> Result<Self, TrainError> {
        let dir = Self {
            root: root.to_owned(),
        };
        if !dir.path("checkpoint.json").exists() {
            return Err(TrainError::Checkpoint(format!(
                "{} has no checkpoint.json",
                root.display()
            )));
        }
        Ok(dir)
    }

    #[must_use]
    pub fn path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    fn write_atomically(&self, name: &str, contents: &str) -> Result<(), TrainError> {
        let target = self.path(name);
        let temporary = self.path(&format!("{name}.tmp"));
        let mut file = File::create(&temporary).map_err(|e| io_error(&temporary, &e))?;
        file.write_all(contents.as_bytes())
            .map_err(|e| io_error(&temporary, &e))?;
        file.sync_all().map_err(|e| io_error(&temporary, &e))?;
        fs::rename(&temporary, &target).map_err(|e| io_error(&target, &e))
    }

    /// # Errors
    ///
    /// `TrainError::Io` if the file cannot be written.
    pub fn write_config(&self, config: &TrainConfig) -> Result<(), TrainError> {
        let text =
            serde_json::to_string_pretty(config).map_err(|e| TrainError::Config(e.to_string()))?;
        self.write_atomically("config.json", &text)
    }

    /// Appends one event as a line and flushes it, so a watcher sees it
    /// immediately.
    ///
    /// # Errors
    ///
    /// `TrainError::Io` if the log cannot be written.
    pub fn append_event(&self, event: &Event) -> Result<(), TrainError> {
        let path = self.path("events.jsonl");
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| io_error(&path, &e))?;
        let mut line =
            serde_json::to_string(event).map_err(|e| TrainError::Checkpoint(e.to_string()))?;
        line.push('\n');
        file.write_all(line.as_bytes())
            .map_err(|e| io_error(&path, &e))
    }

    /// Drops generation events at or after `generation` (and any run-end
    /// marker), keeping the log consistent with a checkpoint when a run
    /// died between writing an event and its checkpoint.
    ///
    /// # Errors
    ///
    /// `TrainError::Io` if the log cannot be read or rewritten.
    pub fn truncate_events_from(&self, generation: u32) -> Result<(), TrainError> {
        let path = self.path("events.jsonl");
        let Ok(text) = fs::read_to_string(&path) else {
            return Ok(());
        };
        let mut kept = String::new();
        for line in text.lines() {
            let keep = match serde_json::from_str::<Event>(line) {
                Ok(Event::Generation(e)) => e.generation < generation,
                Ok(Event::RunEnd(_)) => false,
                _ => true,
            };
            if keep {
                kept.push_str(line);
                kept.push('\n');
            }
        }
        self.write_atomically("events.jsonl", &kept)
    }

    /// # Errors
    ///
    /// `TrainError::Io` if the checkpoint cannot be written.
    pub fn write_checkpoint(&self, checkpoint: &Checkpoint) -> Result<(), TrainError> {
        let text =
            serde_json::to_string(checkpoint).map_err(|e| TrainError::Checkpoint(e.to_string()))?;
        self.write_atomically("checkpoint.json", &text)
    }

    /// # Errors
    ///
    /// `TrainError::Checkpoint` if the file is unreadable or malformed.
    pub fn read_checkpoint(&self) -> Result<Checkpoint, TrainError> {
        let path = self.path("checkpoint.json");
        let text = fs::read_to_string(&path).map_err(|e| io_error(&path, &e))?;
        let checkpoint: Checkpoint = serde_json::from_str(&text)
            .map_err(|e| TrainError::Checkpoint(format!("{}: {e}", path.display())))?;
        if checkpoint.schema_version != SCHEMA_VERSION {
            return Err(TrainError::Checkpoint(format!(
                "schema version {} (this build reads {SCHEMA_VERSION})",
                checkpoint.schema_version
            )));
        }
        Ok(checkpoint)
    }

    /// Writes `gen-NNNN.json` and returns its file name.
    ///
    /// # Errors
    ///
    /// `TrainError::Io` on write failure, `TrainError::Config` if the
    /// genome does not fit this build's features.
    pub fn write_champion(&self, generation: u32, genome: &Genome) -> Result<String, TrainError> {
        let name = format!("gen-{generation:04}.json");
        self.write_genome(&name, genome)?;
        Ok(name)
    }

    /// # Errors
    ///
    /// As `write_champion`.
    pub fn write_best(&self, genome: &Genome) -> Result<(), TrainError> {
        self.write_genome("best.json", genome)
    }

    fn write_genome(&self, name: &str, genome: &Genome) -> Result<(), TrainError> {
        let file =
            GenomeFile::new(genome.clone()).map_err(|e| TrainError::Config(e.to_string()))?;
        let text =
            serde_json::to_string_pretty(&file).map_err(|e| TrainError::Config(e.to_string()))?;
        self.write_atomically(name, &text)
    }
}
```

- [ ] **Step 6: Run (expect success)**

Run: `cargo fmt -p sim && cargo test -p sim training::run_dir`

Expected: PASS (8 tests: refuses to overwrite a run; missing checkpoint named; round trip without a leftover temp file; corrupt checkpoint is an error; champions are playable genome files; events append as lines and truncation drops the tail; `load_config`).

- [ ] **Step 7: Commit**

```bash
git add sim
git commit -F - <<'EOF'
sim: add the training run directory (Phase 10c)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F3sC3JdcgTFAox87bBuikA
EOF
```


### Task 6: `Trainer`: the generational loop, checkpoints and resume

**Files:**
- Modify: `sim/src/training/mod.rs`
- Create: `sim/src/training/trainer.rs`, `sim/tests/training_run.rs`

**Why:** Order of persistence per generation: champion files, event line, then the checkpoint *last*, so a crash in between is repaired on resume (the duplicate event is trimmed and the generation redone). Genomes are evaluated independently (rayon, order-preserving), so results cannot depend on thread count.

**Interfaces:**
- Consumes: Everything above plus `NeatStrategy`, `FEATURE_COUNT`, `FEATURE_NAMES` (10b) and `neat::Population`.
- Produces: `Trainer::{new(config, Vec<Opponent>, &Path), resume(&Path, Vec<Opponent>, Option<u32>), run(&mut dyn TrainObserver) -> Result<RunEnd, TrainError>, config(), completed_generations()}`; `Opponent { name, strategy }`; `trait TrainObserver { on_start, on_eval_progress(generation, done, total), on_generation, on_finish }` (all default no-ops).

- [ ] **Step 1: Create file**

Create `sim/tests/training_run.rs` (Tests first: the integration tests describe the whole trainer):

```rust
//! End-to-end tests of the trainer: the files a run leaves behind,
//! exact resume, independence from thread count, crash repair and, most
//! importantly, that training actually improves play.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use neat::NeatConfig;
use sim::training::{
    DeckChoice, DuplicateChoice, Event, Opponent, TrainConfig, TrainError, TrainObserver, Trainer,
};
use sim::{LowestLegal, NeatStrategy, RandomLegal};

fn dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("arschloch-train-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    path
}

fn opponents() -> Vec<Opponent> {
    vec![
        Opponent {
            name: "LowestLegal".into(),
            strategy: Arc::new(LowestLegal),
        },
        Opponent {
            name: "RandomLegal".into(),
            strategy: Arc::new(RandomLegal),
        },
    ]
}

fn config(generations: u32) -> TrainConfig {
    TrainConfig {
        seed: 3,
        player_count: 4,
        deck: DeckChoice::Single,
        duplicate_rule: DuplicateChoice::FirstDealtWins,
        rounds_per_match: 4,
        matches_per_genome: 8,
        reeval_matches: 12,
        generations,
        neat: NeatConfig {
            population_size: 16,
            ..NeatConfig::default()
        },
        opponent_specs: vec!["lowest-legal".into(), "random-legal".into()],
    }
}

fn events(dir: &Path) -> Vec<Event> {
    fs::read_to_string(dir.join("events.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).expect("every line is an event"))
        .collect()
}

fn checkpoint_population(dir: &Path) -> serde_json::Value {
    let text = fs::read_to_string(dir.join("checkpoint.json")).unwrap();
    serde_json::from_str::<serde_json::Value>(&text).unwrap()["population"].clone()
}

#[derive(Default)]
struct Recorder {
    starts: usize,
    progress: usize,
    last_progress: (usize, usize),
    generations: Vec<u32>,
    finished: bool,
}

impl TrainObserver for Recorder {
    fn on_start(&mut self, _: &sim::training::RunStart) {
        self.starts += 1;
    }
    fn on_eval_progress(&mut self, _: u32, done: usize, total: usize) {
        self.progress += 1;
        self.last_progress = (done, total);
    }
    fn on_generation(&mut self, event: &sim::training::GenerationEvent) {
        self.generations.push(event.generation);
    }
    fn on_finish(&mut self, _: &sim::training::RunEnd) {
        self.finished = true;
    }
}

#[test]
fn a_short_run_leaves_every_artifact_and_notifies_the_observer() {
    let run = dir("artifacts");
    let mut observer = Recorder::default();
    let end = Trainer::new(config(3), opponents(), &run)
        .unwrap()
        .run(&mut observer)
        .unwrap();

    assert_eq!(end.generations_completed, 3);
    assert_eq!(
        (
            observer.starts,
            observer.generations.clone(),
            observer.finished
        ),
        (1, vec![0, 1, 2], true)
    );
    assert_eq!(
        observer.last_progress,
        (16, 16),
        "progress reaches the whole population"
    );
    assert!(
        observer.progress >= 3 * 2,
        "several progress steps per generation"
    );

    for file in [
        "config.json",
        "events.jsonl",
        "checkpoint.json",
        "best.json",
        "gen-0000.json",
        "gen-0001.json",
        "gen-0002.json",
    ] {
        assert!(run.join(file).exists(), "{file} missing");
    }
    assert!(!run.join("checkpoint.json.tmp").exists());

    let log = events(&run);
    assert_eq!(log.len(), 5, "start, 3 generations, end");
    let Event::Generation(first) = &log[1] else {
        panic!("expected a generation event")
    };
    assert_eq!(first.generation, 0);
    assert_eq!(first.opponents.len(), 2);
    assert_eq!(first.opponents[0].name, "LowestLegal");
    assert_eq!(first.fitness.histogram.iter().sum::<u32>(), 16);
    assert_eq!(first.champion.reeval.matches, 12);
    assert!(
        first.champion.is_new_best,
        "the first champion is the first best"
    );
    assert_eq!(first.champion.genome_file, "gen-0000.json");
    // 16 genomes x 8 matches + 12 x (1 + 2 opponents), 4 rounds each.
    assert_eq!(first.rounds_evaluated, 4 * (16 * 8 + 12 * 3));
    assert!(matches!(log[4], Event::RunEnd(_)));

    assert!(NeatStrategy::from_file(&run.join("best.json")).is_ok());
    assert!(NeatStrategy::from_file(&run.join("gen-0002.json")).is_ok());
    fs::remove_dir_all(&run).unwrap();
}

#[test]
fn resuming_is_identical_to_never_stopping() {
    let straight = dir("straight");
    Trainer::new(config(4), opponents(), &straight)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();

    let split = dir("split");
    Trainer::new(config(2), opponents(), &split)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let mut resumed = Trainer::resume(&split, opponents(), Some(4)).unwrap();
    assert_eq!(resumed.completed_generations(), 2);
    let mut observer = Recorder::default();
    let end = resumed.run(&mut observer).unwrap();
    assert_eq!(end.generations_completed, 4);
    assert_eq!(observer.generations, vec![2, 3]);

    assert_eq!(
        checkpoint_population(&split),
        checkpoint_population(&straight)
    );
    for file in ["best.json", "gen-0002.json", "gen-0003.json"] {
        assert_eq!(
            fs::read_to_string(split.join(file)).unwrap(),
            fs::read_to_string(straight.join(file)).unwrap(),
            "{file}"
        );
    }
    // Each generation appears once in the resumed log, in order.
    let generations: Vec<u32> = events(&split)
        .iter()
        .filter_map(|e| {
            if let Event::Generation(g) = e {
                Some(g.generation)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(generations, vec![0, 1, 2, 3]);
    fs::remove_dir_all(&straight).unwrap();
    fs::remove_dir_all(&split).unwrap();
}

#[test]
fn results_do_not_depend_on_the_thread_count() {
    let run_with = |threads: usize, name: &str| {
        let run = dir(name);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            Trainer::new(config(3), opponents(), &run)
                .unwrap()
                .run(&mut Recorder::default())
                .unwrap();
        });
        let population = checkpoint_population(&run);
        let best = fs::read_to_string(run.join("best.json")).unwrap();
        fs::remove_dir_all(&run).unwrap();
        (population, best)
    };
    assert_eq!(run_with(1, "threads1"), run_with(4, "threads4"));
}

#[test]
fn a_crash_between_the_event_and_the_checkpoint_is_repaired_on_resume() {
    let run = dir("crash");
    Trainer::new(config(2), opponents(), &run)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    // Simulate dying after generation 2's event was logged but before its
    // checkpoint: duplicate the last generation event with generation 2.
    let log = events(&run);
    let Event::Generation(last) = log
        .iter()
        .rev()
        .find(|e| matches!(e, Event::Generation(_)))
        .unwrap()
        .clone()
    else {
        unreachable!()
    };
    let mut phantom = *last;
    phantom.generation = 2;
    let line = serde_json::to_string(&Event::Generation(Box::new(phantom))).unwrap();
    let mut text = fs::read_to_string(run.join("events.jsonl")).unwrap();
    text.push_str(&line);
    text.push('\n');
    fs::write(run.join("events.jsonl"), text).unwrap();

    Trainer::resume(&run, opponents(), Some(3))
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let generations: Vec<u32> = events(&run)
        .iter()
        .filter_map(|e| {
            if let Event::Generation(g) = e {
                Some(g.generation)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(
        generations,
        vec![0, 1, 2],
        "the phantom was replaced by the real generation 2"
    );
    fs::remove_dir_all(&run).unwrap();
}

#[test]
fn resuming_with_a_different_pool_or_no_run_is_refused() {
    let run = dir("mismatch");
    Trainer::new(config(1), opponents(), &run)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let wrong = vec![Opponent {
        name: "GreedyHighest".into(),
        strategy: Arc::new(RandomLegal),
    }];
    let error = Trainer::resume(&run, wrong, None).err().unwrap();
    assert!(matches!(error, TrainError::Mismatch(_)), "{error}");
    let error = Trainer::new(config(1), opponents(), &run).err().unwrap();
    assert!(
        matches!(error, TrainError::Config(_)),
        "a run is never overwritten: {error}"
    );
    let error = Trainer::resume(&dir("nothing-here"), opponents(), None)
        .err()
        .unwrap();
    assert!(matches!(error, TrainError::Checkpoint(_)), "{error}");
    fs::remove_dir_all(&run).unwrap();
}

#[test]
fn invalid_setups_are_refused_before_anything_is_written() {
    let run = dir("invalid");
    let bad = TrainConfig {
        matches_per_genome: 0,
        ..config(1)
    };
    assert!(matches!(
        Trainer::new(bad, opponents(), &run).err().unwrap(),
        TrainError::Config(_)
    ));
    let few = vec![opponents().remove(0)];
    assert!(matches!(
        Trainer::new(config(1), few, &run).err().unwrap(),
        TrainError::Config(_)
    ));
    assert!(!run.join("checkpoint.json").exists());
    let _ = fs::remove_dir_all(&run);
}

#[test]
fn training_improves_play() {
    let run = dir("improves");
    let config = TrainConfig {
        matches_per_genome: 14,
        reeval_matches: 40,
        neat: NeatConfig {
            population_size: 30,
            ..NeatConfig::default()
        },
        ..config(10)
    };
    Trainer::new(config, opponents(), &run)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let generations: Vec<_> = events(&run)
        .into_iter()
        .filter_map(|e| {
            if let Event::Generation(g) = e {
                Some(g)
            } else {
                None
            }
        })
        .collect();
    let first = &generations[0];
    let last = &generations[generations.len() - 1];
    // Random networks start badly (most play poorly); evolution must lift
    // the whole population, not just find one lucky genome.
    assert!(
        last.fitness.mean > first.fitness.mean + 0.25,
        "mean fitness {} -> {}",
        first.fitness.mean,
        last.fitness.mean
    );
    // And the best champion beats random play and mirrors-or-beats the
    // lowest-legal baseline on fresh matches.
    let best = generations
        .iter()
        .map(|g| &g.champion.reeval)
        .max_by(|a, b| a.mean.total_cmp(&b.mean))
        .unwrap();
    assert!(best.mean > 0.3, "best fresh-match score {}", best.mean);
    fs::remove_dir_all(&run).unwrap();
}
```

- [ ] **Step 2: Run (expect failure)**

Run: `cargo test -p sim --test training_run`

Expected: FAIL (compile errors: `Trainer`, `Opponent`, `TrainObserver` are not defined).

- [ ] **Step 3: Edit**

In `sim/src/training/mod.rs`, replace:

```rust
pub mod run_dir;
```

with:

```rust
pub mod run_dir;
pub mod trainer;
```

- [ ] **Step 4: Edit**

In `sim/src/training/mod.rs`, replace:

```rust
pub use run_dir::{load_config, RunDir, TrainError};
```

with:

```rust
pub use run_dir::{load_config, RunDir, TrainError};
pub use trainer::{Opponent, TrainObserver, Trainer};
```

- [ ] **Step 5: Create file**

Create `sim/src/training/trainer.rs` (Implement):

```rust
//! The generational training loop: evaluate every genome on the same
//! matches, let `neat` breed the next generation, re-evaluate the
//! champion on fresh matches, and record everything.
//!
//! A run is a pure function of its `TrainConfig`: genomes are evaluated
//! independently (so thread count cannot change results), match seeds
//! derive only from `(run seed, generation, index)`, and the population's
//! state is checkpointed after every generation, so a resumed run is
//! identical to one that never stopped.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use neat::{Genome, Population};
use rayon::prelude::*;

use super::config::TrainConfig;
use super::evaluate::{evaluate, match_seed, Opponents, Score};
use super::events::{
    ChampionStats, Complexity, Event, FitnessStats, GenerationEvent, OpponentStat, RunEnd,
    RunStart, SCHEMA_VERSION,
};
use super::run_dir::{BestRecord, Checkpoint, RunDir, TrainError};
use crate::{NeatStrategy, Strategy, FEATURE_COUNT, FEATURE_NAMES};

/// One member of the opponent pool.
#[derive(Clone)]
pub struct Opponent {
    pub name: String,
    pub strategy: Arc<dyn Strategy>,
}

/// Receives progress while a run executes. All methods default to doing
/// nothing; implementors only override what they show. Called on the
/// thread that called `Trainer::run`, between evaluation steps.
pub trait TrainObserver {
    fn on_start(&mut self, _start: &RunStart) {}
    /// `done` of `total` genomes of the current generation evaluated.
    fn on_eval_progress(&mut self, _generation: u32, _done: usize, _total: usize) {}
    fn on_generation(&mut self, _event: &GenerationEvent) {}
    fn on_finish(&mut self, _end: &RunEnd) {}
}

pub struct Trainer {
    config: TrainConfig,
    opponents: Vec<Opponent>,
    dir: RunDir,
    population: Population,
    best: Option<BestRecord>,
    total_rounds: u64,
    elapsed_before: f64,
    resumed_from: Option<u32>,
}

/// Genomes are evaluated in this many parallel batches, with a progress
/// callback between batches.
const PROGRESS_STEPS: usize = 10;

fn strategy_for(genome: &Genome) -> Arc<dyn Strategy> {
    Arc::new(
        NeatStrategy::new("candidate", genome).expect("trained genomes use this build's features"),
    )
}

fn position_of_best(values: &[f64]) -> usize {
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

#[allow(clippy::cast_precision_loss)] // counts are far below 2^52
fn fitness_stats(fitness: &[f64]) -> FitnessStats {
    let count = fitness.len() as f64;
    let mean = fitness.iter().sum::<f64>() / count;
    let mut sorted = fitness.to_vec();
    sorted.sort_by(f64::total_cmp);
    let (min, best) = (sorted[0], sorted[sorted.len() - 1]);
    let median = if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        f64::midpoint(sorted[sorted.len() / 2 - 1], sorted[sorted.len() / 2])
    };
    let std_dev = (fitness.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / count).sqrt();
    let mut histogram = vec![0u32; 10];
    for &value in fitness {
        let bucket = if best > min {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let bucket = (((value - min) / (best - min)) * 10.0) as usize;
            bucket.min(9)
        } else {
            0
        };
        histogram[bucket] += 1;
    }
    FitnessStats {
        best,
        mean,
        median,
        min,
        std_dev,
        histogram,
    }
}

impl Trainer {
    /// Starts a new run in `dir`.
    ///
    /// # Errors
    ///
    /// `TrainError::Config` for an invalid setup (or a directory that
    /// already holds a run), `TrainError::Io` if files cannot be written.
    pub fn new(
        config: TrainConfig,
        opponents: Vec<Opponent>,
        dir: &Path,
    ) -> Result<Self, TrainError> {
        config.validate().map_err(TrainError::Config)?;
        if opponents.len() != config.opponent_specs.len() {
            return Err(TrainError::Config(format!(
                "{} opponents were built for {} specs",
                opponents.len(),
                config.opponent_specs.len()
            )));
        }
        let dir = RunDir::create_new(dir)?;
        dir.write_config(&config)?;
        let population = Population::new(
            FEATURE_COUNT,
            config.neat.clone(),
            match_seed(config.seed, u64::MAX, 0),
        )
        .map_err(|e| TrainError::Config(e.to_string()))?;
        Ok(Self {
            config,
            opponents,
            dir,
            population,
            best: None,
            total_rounds: 0,
            elapsed_before: 0.0,
            resumed_from: None,
        })
    }

    /// Continues the run in `dir` from its last checkpoint. `generations`
    /// optionally raises (or lowers) the total to run up to.
    ///
    /// # Errors
    ///
    /// `TrainError::Checkpoint` for unusable files, `TrainError::Mismatch`
    /// if `opponents` are not the pool the run started with.
    pub fn resume(
        dir: &Path,
        opponents: Vec<Opponent>,
        generations: Option<u32>,
    ) -> Result<Self, TrainError> {
        let dir = RunDir::open_existing(dir)?;
        let checkpoint = dir.read_checkpoint()?;
        let names: Vec<&str> = opponents.iter().map(|o| o.name.as_str()).collect();
        if names
            != checkpoint
                .opponent_names
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        {
            return Err(TrainError::Mismatch(format!(
                "the run used opponents {:?} but {:?} were given",
                checkpoint.opponent_names, names
            )));
        }
        let mut config = checkpoint.config;
        if let Some(total) = generations {
            config.generations = total;
        }
        config.validate().map_err(TrainError::Config)?;
        let population = Population::restore(checkpoint.population)
            .map_err(|e| TrainError::Checkpoint(e.to_string()))?;
        let completed = population.generation();
        dir.truncate_events_from(completed)?;
        Ok(Self {
            config,
            opponents,
            dir,
            population,
            best: checkpoint.best,
            total_rounds: checkpoint.total_rounds,
            elapsed_before: checkpoint.elapsed_secs,
            resumed_from: Some(completed),
        })
    }

    #[must_use]
    pub fn config(&self) -> &TrainConfig {
        &self.config
    }

    #[must_use]
    pub fn completed_generations(&self) -> u32 {
        self.population.generation()
    }

    /// Runs generations until `config.generations` are complete.
    ///
    /// # Errors
    ///
    /// `TrainError::Io` if a run file cannot be written.
    pub fn run(&mut self, observer: &mut dyn TrainObserver) -> Result<RunEnd, TrainError> {
        let started = Instant::now();
        let start = RunStart {
            schema_version: SCHEMA_VERSION,
            config: self.config.clone(),
            opponents: self.opponents.iter().map(|o| o.name.clone()).collect(),
            feature_names: FEATURE_NAMES.iter().map(|&n| n.to_owned()).collect(),
            resumed_from_generation: self.resumed_from,
        };
        self.dir
            .append_event(&Event::RunStart(Box::new(start.clone())))?;
        observer.on_start(&start);

        while self.population.generation() < self.config.generations {
            self.step(observer, started)?;
        }

        let end = RunEnd {
            generations_completed: self.population.generation(),
            best_generation: self.best.as_ref().map(|b| b.generation),
            best_reeval: self.best.as_ref().map(|b| b.reeval.clone()),
            elapsed_secs: self.elapsed_before + started.elapsed().as_secs_f64(),
        };
        self.dir.append_event(&Event::RunEnd(end.clone()))?;
        observer.on_finish(&end);
        Ok(end)
    }

    fn evaluate_population(&self, generation: u32, observer: &mut dyn TrainObserver) -> Vec<f64> {
        let table = self.config.table();
        let pool: Vec<Arc<dyn Strategy>> =
            self.opponents.iter().map(|o| o.strategy.clone()).collect();
        let seeds: Vec<u64> = (0..self.config.matches_per_genome as u64)
            .map(|i| match_seed(self.config.seed, 2 * u64::from(generation), i))
            .collect();
        let genomes = self.population.genomes();
        let batch = genomes.len().div_ceil(PROGRESS_STEPS);
        let mut fitness = Vec::with_capacity(genomes.len());
        for chunk in genomes.chunks(batch) {
            let scores: Vec<f64> = chunk
                .par_iter()
                .map(|genome| {
                    evaluate(
                        &strategy_for(genome),
                        &table,
                        Opponents::Mixed(&pool),
                        &seeds,
                    )
                    .mean
                })
                .collect();
            fitness.extend(scores);
            observer.on_eval_progress(generation, fitness.len(), genomes.len());
        }
        fitness
    }

    fn reevaluate(&self, generation: u32, champion: &Genome) -> (Score, Vec<OpponentStat>) {
        let table = self.config.table();
        let pool: Vec<Arc<dyn Strategy>> =
            self.opponents.iter().map(|o| o.strategy.clone()).collect();
        let seeds: Vec<u64> = (0..self.config.reeval_matches as u64)
            .map(|i| match_seed(self.config.seed, 2 * u64::from(generation) + 1, i))
            .collect();
        let candidate = strategy_for(champion);
        let mixed = evaluate(&candidate, &table, Opponents::Mixed(&pool), &seeds);
        let per_opponent = self
            .opponents
            .par_iter()
            .map(|opponent| OpponentStat {
                name: opponent.name.clone(),
                score: evaluate(
                    &candidate,
                    &table,
                    Opponents::Only(&opponent.strategy),
                    &seeds,
                )
                .into(),
            })
            .collect();
        (mixed, per_opponent)
    }

    fn step(
        &mut self,
        observer: &mut dyn TrainObserver,
        started: Instant,
    ) -> Result<(), TrainError> {
        let step_started = Instant::now();
        let generation = self.population.generation();

        let fitness = self.evaluate_population(generation, observer);
        let champion_index = position_of_best(&fitness);
        // `advance` replaces the genomes, so take the champion first.
        let champion = self.population.genomes()[champion_index].clone();
        let stats = fitness_stats(&fitness);
        self.population.set_fitness(fitness.clone());
        let report = self.population.advance();

        let (reeval, opponents) = self.reevaluate(generation, &champion);
        let is_new_best = self
            .best
            .as_ref()
            .is_none_or(|b| reeval.mean > b.reeval.mean);
        let genome_file = self.dir.write_champion(generation, &champion)?;
        if is_new_best {
            self.dir.write_best(&champion)?;
            self.best = Some(BestRecord {
                generation,
                reeval: reeval.clone().into(),
            });
        }

        let table = self.config.table();
        let rounds_per_match = table.rounds as u64;
        let rounds_evaluated = rounds_per_match
            * (self.config.matches_per_genome as u64 * self.config.neat.population_size as u64
                + self.config.reeval_matches as u64 * (1 + self.opponents.len() as u64));
        self.total_rounds += rounds_evaluated;
        let generation_secs = step_started.elapsed().as_secs_f64();
        let elapsed_secs = self.elapsed_before + started.elapsed().as_secs_f64();

        let event = GenerationEvent {
            generation,
            elapsed_secs,
            generation_secs,
            rounds_evaluated,
            total_rounds: self.total_rounds,
            #[allow(clippy::cast_precision_loss)]
            rounds_per_sec: rounds_evaluated as f64 / generation_secs.max(1e-9),
            fitness: stats,
            champion: ChampionStats {
                train_fitness: fitness[champion_index],
                reeval: reeval.into(),
                hidden_nodes: champion.hidden_count(),
                enabled_connections: champion.enabled_connection_count(),
                genome_file,
                is_new_best,
            },
            opponents,
            species: report.species,
            compatibility_threshold: report.compatibility_threshold,
            complexity: Complexity {
                mean_hidden_nodes: report.mean_hidden_nodes,
                mean_enabled_connections: report.mean_enabled_connections,
                innovation_count: report.innovation_count,
            },
        };

        // The checkpoint comes last: if the process dies earlier, resume
        // redoes this generation and trims the duplicate event.
        self.dir
            .append_event(&Event::Generation(Box::new(event.clone())))?;
        self.dir.write_checkpoint(&Checkpoint {
            schema_version: SCHEMA_VERSION,
            config: self.config.clone(),
            opponent_names: self.opponents.iter().map(|o| o.name.clone()).collect(),
            population: self.population.snapshot(),
            best: self.best.clone(),
            total_rounds: self.total_rounds,
            elapsed_secs,
        })?;
        observer.on_generation(&event);
        Ok(())
    }
}
```

- [ ] **Step 6: Run (expect success)**

Run: `cargo fmt -p sim && cargo test -p sim --test training_run`

Expected: PASS (7 tests, about 15 s in a debug build): every artifact and observer callback; resuming equals never stopping (population state, best and champion files byte-identical, each generation logged once); results independent of thread count (1 vs 4); a crash between event and checkpoint is repaired; refusals (different pool, existing run, no run, invalid setup); and `training_improves_play` (mean fitness rises by more than 0.25 within 10 generations and the best champion scores above 0.3 on fresh matches).

- [ ] **Step 7: Commit**

```bash
git add sim
git commit -F - <<'EOF'
sim: add Trainer, the resumable generational loop (Phase 10c)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F3sC3JdcgTFAox87bBuikA
EOF
```


### Task 7: CLI: `cli train`

**Files:**
- Modify: `cli/Cargo.toml`, `cli/src/main.rs`
- Create: `cli/src/train_args.rs`, `cli/src/train_output.rs`, `cli/src/train.rs`

**Interfaces:**
- Consumes: `sim::training::*` (Tasks 2-6), `StrategyArg`/`DeckVariantArg`/`DuplicateRuleArg` (existing `cli/src/args.rs`), `neat::NeatConfig`.
- Produces: `cli train --out DIR [--resume] [--player-count N] [--population N] [--generations N] [--matches-per-genome N] [--reeval-matches N] [--rounds N] [--seed N] [--opponent SPEC]... [--target-species N] [--threads N] [--quiet]`. Terminal: banner with an opponent legend, one row per generation (selection best/mean, the champion's fresh score with its standard error, species count, champion size, the champion's score against each opponent alone, rounds/s, ETA, `*` on a new best), a refreshed progress line on a terminal (stderr), and a summary. Pure renderers `render_banner`, `render_header`, `render_row`, `render_summary`, `format_duration`, `format_rate`.

- [ ] **Step 1: Edit**

In `cli/Cargo.toml` (`neat` becomes a normal dependency (train builds a `NeatConfig`)), replace:

```toml

[dev-dependencies]
neat = { version = "0.1.0", path = "../neat" }
```

with:

```toml

```

- [ ] **Step 2: Edit**

In `cli/Cargo.toml`, replace:

```toml
engine = { version = "0.1.0", path = "../engine" }
```

with:

```toml
engine = { version = "0.1.0", path = "../engine" }
neat = { version = "0.1.0", path = "../neat" }
```

- [ ] **Step 3: Edit**

In `cli/src/main.rs`, replace:

```rust
mod args;
mod output;
mod summary;
```

with:

```rust
mod args;
mod output;
mod summary;
mod train;
mod train_args;
mod train_output;
```

- [ ] **Step 4: Create file**

Create `cli/src/train_args.rs` (Tests first: the file contains only its test module):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<TrainArgs, clap::Error> {
        TrainArgs::try_parse_from(std::iter::once("cli train").chain(args.iter().copied()))
    }

    #[test]
    fn only_out_is_required_and_the_defaults_are_sensible() {
        let args = parse(&["--out", "runs/a"]).unwrap();
        assert_eq!(
            (args.player_count, args.population, args.rounds),
            (4, 150, 8)
        );
        assert_eq!(args.generations, None);
        assert!(args.opponent.is_empty());
        assert!(!args.resume && !args.quiet);
    }

    #[test]
    fn out_is_required() {
        assert!(parse(&[]).is_err());
    }

    #[test]
    fn resume_accepts_only_generations_threads_and_quiet() {
        assert!(parse(&[
            "--out",
            "d",
            "--resume",
            "--generations",
            "50",
            "--threads",
            "2",
            "--quiet"
        ])
        .is_ok());
        for conflicting in [
            ["--population", "10"],
            ["--seed", "3"],
            ["--opponent", "lowest-legal"],
            ["--player-count", "5"],
        ] {
            let mut args = vec!["--out", "d", "--resume"];
            args.extend(conflicting);
            assert!(parse(&args).is_err(), "{conflicting:?}");
        }
    }

    #[test]
    fn out_of_range_values_are_rejected() {
        assert!(parse(&["--out", "d", "--player-count", "2"]).is_err());
        assert!(parse(&["--out", "d", "--population", "1"]).is_err());
        assert!(parse(&["--out", "d", "--generations", "0"]).is_err());
    }

    #[test]
    fn opponents_can_be_repeated() {
        let args = parse(&[
            "--out",
            "d",
            "--opponent",
            "lowest-legal",
            "--opponent",
            "adaptive:counting",
        ])
        .unwrap();
        assert_eq!(args.opponent, vec!["lowest-legal", "adaptive:counting"]);
    }
}
```

- [ ] **Step 5: Create file**

Create `cli/src/train_output.rs` (Tests first):

```rust
#[cfg(test)]
mod tests {
    use sim::training::events::{ChampionStats, Complexity, FitnessStats, OpponentStat};
    use sim::training::{ScoreStat, SCHEMA_VERSION};

    use super::*;

    fn stat(mean: f64) -> ScoreStat {
        ScoreStat {
            mean,
            std_error: 0.021,
            matches: 100,
            placements: vec![1, 2, 3, 4],
        }
    }

    fn event(is_new_best: bool) -> GenerationEvent {
        GenerationEvent {
            generation: 42,
            elapsed_secs: 100.0,
            generation_secs: 2.0,
            rounds_evaluated: 168_000,
            total_rounds: 1_000_000,
            rounds_per_sec: 84_000.0,
            fitness: FitnessStats {
                best: 0.412,
                mean: 0.188,
                median: 0.2,
                min: -0.5,
                std_dev: 0.1,
                histogram: vec![0; 10],
            },
            champion: ChampionStats {
                train_fitness: 0.412,
                reeval: stat(0.397),
                hidden_nodes: 7,
                enabled_connections: 23,
                genome_file: "gen-0042.json".into(),
                is_new_best,
            },
            opponents: vec![
                OpponentStat {
                    name: "LowestLegal".into(),
                    score: stat(0.61),
                },
                OpponentStat {
                    name: "Adaptive".into(),
                    score: stat(-0.05),
                },
            ],
            species: vec![],
            compatibility_threshold: 0.5,
            complexity: Complexity {
                mean_hidden_nodes: 1.0,
                mean_enabled_connections: 20.0,
                innovation_count: 30,
            },
        }
    }

    #[test]
    fn a_row_shows_every_headline_number() {
        let row = render_row(&event(false), Some(2470.0));
        for expected in [
            "42", "+0.412", "+0.188", "+0.397", "±0.021", "7/23", "+0.61", "-0.05", "84k",
            "0:41:10",
        ] {
            assert!(row.contains(expected), "{expected} missing from: {row}");
        }
        assert!(!row.ends_with('*'));
    }

    #[test]
    fn a_new_best_is_marked_and_a_missing_eta_is_dashes() {
        let row = render_row(&event(true), None);
        assert!(row.ends_with(" *"), "{row}");
        assert!(row.contains("--:--:--"));
    }

    #[test]
    fn the_header_has_one_column_per_opponent() {
        let header = render_header(3);
        assert!(header.contains("o1") && header.contains("o2") && header.contains("o3"));
        assert!(!header.contains("o4"));
        assert!(header.contains("ETA") && header.contains("champion"));
    }

    #[test]
    fn durations_and_rates_format_compactly() {
        assert_eq!(format_duration(0.0), "0:00:00");
        assert_eq!(format_duration(61.4), "0:01:01");
        assert_eq!(format_duration(3_725.0), "1:02:05");
        assert_eq!(format_duration(-5.0), "0:00:00");
        assert_eq!(format_rate(842.0), "842");
        assert_eq!(format_rate(1_234.0), "1.2k");
        assert_eq!(format_rate(84_000.0), "84k");
        assert_eq!(format_rate(2_500_000.0), "2.5M");
    }

    #[test]
    fn the_summary_names_the_best_generation_and_how_to_play_it() {
        let end = RunEnd {
            generations_completed: 100,
            best_generation: Some(87),
            best_reeval: Some(stat(0.652)),
            elapsed_secs: 5025.0,
        };
        let text = render_summary(&end, std::path::Path::new("runs/a"));
        assert!(text.contains("100 generations in 1:23:45"), "{text}");
        assert!(
            text.contains("generation 87") && text.contains("+0.652"),
            "{text}"
        );
        assert!(text.contains("neat:runs/a/best.json"), "{text}");
        let none = RunEnd {
            best_generation: None,
            best_reeval: None,
            ..end
        };
        assert!(!render_summary(&none, std::path::Path::new("x")).contains("best champion"));
    }

    #[test]
    fn the_banner_lists_opponents_and_flags_a_resume() {
        let start = RunStart {
            schema_version: SCHEMA_VERSION,
            config: sim_config(),
            opponents: vec!["LowestLegal".into(), "Adaptive(x)".into()],
            feature_names: vec![],
            resumed_from_generation: Some(12),
        };
        let text = render_banner(&start, std::path::Path::new("runs/a"));
        assert!(text.contains("o1=LowestLegal  o2=Adaptive(x)"), "{text}");
        assert!(text.contains("resumed from generation 12"), "{text}");
        assert!(text.contains("4 players"), "{text}");
    }

    fn sim_config() -> sim::training::TrainConfig {
        sim::training::TrainConfig {
            seed: 0,
            player_count: 4,
            deck: sim::training::DeckChoice::Single,
            duplicate_rule: sim::training::DuplicateChoice::FirstDealtWins,
            rounds_per_match: 8,
            matches_per_genome: 100,
            reeval_matches: 200,
            generations: 100,
            neat: neat::NeatConfig::default(),
            opponent_specs: vec![],
        }
    }
}
```

- [ ] **Step 6: Create file**

Create `cli/src/train.rs` (Tests first):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_pool_builds_with_distinct_names() {
        let specs: Vec<String> = DEFAULT_OPPONENTS.iter().map(|&s| s.to_owned()).collect();
        let pool = build_opponents(&specs).unwrap();
        let names: Vec<&str> = pool.iter().map(|o| o.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "LowestLegal",
                "EndgameDenial",
                "Adaptive(reading,tempo,bully)"
            ]
        );
    }

    #[test]
    fn a_bad_spec_names_the_offending_option() {
        let error = build_opponents(&["nonsense".to_owned()]).err().unwrap();
        assert!(
            format!("{error:#}").contains("--opponent `nonsense`"),
            "{error:#}"
        );
    }

    #[test]
    fn a_duplicate_opponent_is_rejected() {
        let specs = vec!["lowest-legal".to_owned(), "lowest-legal".to_owned()];
        let error = build_opponents(&specs).err().unwrap().to_string();
        assert!(error.contains("duplicates `LowestLegal`"), "{error}");
    }
}
```

- [ ] **Step 7: Run (expect failure)**

Run: `cargo test -p cli`

Expected: FAIL (compile errors: `TrainArgs`, `render_row`, `build_opponents` ... are not defined).

- [ ] **Step 8: Implement**

Insert at the very top of `cli/src/train_args.rs`, above the `#[cfg(test)]` line:

```rust
//! Arguments of `cli train`.

use std::path::PathBuf;

use clap::Parser;

use crate::args::{DeckVariantArg, DuplicateRuleArg};

/// The opponents a run trains against unless `--opponent` is given: the
/// strongest hand-written strategies as of the pre-NEAT baseline
/// (`CardCounter` is omitted: it plays identically to `LowestLegal`).
pub const DEFAULT_OPPONENTS: [&str; 3] = [
    "lowest-legal",
    "endgame-denial",
    "adaptive:reading,tempo,bully",
];

/// Evolve a NEAT player, generation by generation.
#[derive(Parser, Debug)]
#[command(
    name = "cli train",
    about = "Evolve a NEAT player against a pool of opponents, saving every champion",
    long_about = "Evolve a NEAT player against a pool of opponents. Each generation every \
genome plays the same matches against opponents drawn from the pool; the next generation is \
bred from the best, and the generation's champion is re-evaluated on fresh matches. Everything \
is written to --out: events.jsonl (one JSON event per generation), checkpoint.json (resume \
point), best.json and gen-NNNN.json (champions, playable with `cli --strategy neat:PATH`)."
)]
pub struct TrainArgs {
    /// Run directory (created if needed; a new run refuses a directory
    /// that already holds one).
    #[arg(long)]
    pub out: PathBuf,

    /// Continue the run in --out from its last checkpoint. Only
    /// --generations, --threads and --quiet may accompany it: the run's
    /// other settings are read from its checkpoint.
    #[arg(
        long,
        conflicts_with_all = [
            "player_count", "deck_variant", "duplicate_rule", "rounds", "population",
            "matches_per_genome", "reeval_matches", "seed", "opponent", "target_species"
        ]
    )]
    pub resume: bool,

    /// Table size (3-6 seats).
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u8).range(3..=6))]
    pub player_count: u8,

    #[arg(long, value_enum, default_value_t = DeckVariantArg::Single)]
    pub deck_variant: DeckVariantArg,

    #[arg(long, value_enum, default_value_t = DuplicateRuleArg::FirstDealtWins)]
    pub duplicate_rule: DuplicateRuleArg,

    /// Rounds per evaluation match (role carry-over between rounds).
    #[arg(long, default_value_t = 8, value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(1..))]
    pub rounds: usize,

    /// Genomes per generation.
    #[arg(long, default_value_t = 150, value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(2..))]
    pub population: usize,

    /// Total generations to run (default 100). With --resume, the new
    /// total to run up to.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub generations: Option<u32>,

    /// Matches each genome plays per generation (all genomes play the
    /// same ones).
    #[arg(long, default_value_t = 100, value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(1..))]
    pub matches_per_genome: usize,

    /// Matches in each fresh re-evaluation of a generation's champion.
    #[arg(long, default_value_t = 200, value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(1..))]
    pub reeval_matches: usize,

    /// Seeds everything; the same seed reproduces the same run.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// An opponent in the training pool, in `--strategy` syntax (for
    /// example `adaptive:reading,tempo,bully` or `neat:PATH`); repeat for
    /// several. Default: lowest-legal, endgame-denial and
    /// adaptive:reading,tempo,bully.
    #[arg(long, value_name = "SPEC")]
    pub opponent: Vec<String>,

    /// How many species the speciation threshold steers toward.
    #[arg(long, default_value_t = 8, value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(1..))]
    pub target_species: usize,

    /// Rayon thread-pool size. 0 lets rayon pick its own default.
    #[arg(long, default_value_t = 0)]
    pub threads: usize,

    /// Print only the final summary (the event log is still written).
    #[arg(long)]
    pub quiet: bool,
}
```

- [ ] **Step 9: Implement**

Insert at the very top of `cli/src/train_output.rs`, above the `#[cfg(test)]` line:

```rust
//! What `cli train` prints while it runs: a banner, one line per
//! generation, a refreshed progress line on a terminal, and a summary.
//! The line renderers are pure functions of an event so they can be
//! tested without capturing stdout.

use std::fmt::Write as _;
use std::io::{IsTerminal, Write as _};
use std::path::PathBuf;

use sim::training::{GenerationEvent, RunEnd, RunStart, TrainObserver};

/// How often (in generations) the column header is repeated.
const HEADER_EVERY: u32 = 20;

#[must_use]
pub fn format_duration(secs: f64) -> String {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let total = secs.max(0.0).round() as u64;
    format!(
        "{}:{:02}:{:02}",
        total / 3600,
        total % 3600 / 60,
        total % 60
    )
}

/// Rounds per second, compactly (`842`, `12.3k`, `1.2M`).
#[must_use]
pub fn format_rate(per_sec: f64) -> String {
    if per_sec >= 1e6 {
        format!("{:.1}M", per_sec / 1e6)
    } else if per_sec >= 1e4 {
        format!("{:.0}k", per_sec / 1e3)
    } else if per_sec >= 1e3 {
        format!("{:.1}k", per_sec / 1e3)
    } else {
        format!("{per_sec:.0}")
    }
}

#[must_use]
pub fn render_banner(start: &RunStart, out: &std::path::Path) -> String {
    let config = &start.config;
    let mut text = String::new();
    let _ = writeln!(
        text,
        "training: {} players, {:?} deck, {:?} | population {}, {} generations | {} matches x {} rounds per genome, seed {}",
        config.player_count,
        config.deck,
        config.duplicate_rule,
        config.neat.population_size,
        config.generations,
        config.matches_per_genome,
        config.rounds_per_match,
        config.seed
    );
    let legend: Vec<String> = start
        .opponents
        .iter()
        .enumerate()
        .map(|(i, name)| format!("o{}={name}", i + 1))
        .collect();
    let _ = writeln!(text, "opponents: {}", legend.join("  "));
    let _ = write!(text, "output: {}", out.display());
    if let Some(generation) = start.resumed_from_generation {
        let _ = write!(text, "  (resumed from generation {generation})");
    }
    text
}

#[must_use]
pub fn render_header(opponent_count: usize) -> String {
    let mut text = String::from("  gen    best    mean  champion (fresh)  spc  nodes/conn");
    for i in 1..=opponent_count {
        let _ = write!(text, "   o{i:<3}");
    }
    text.push_str("  rounds/s      ETA");
    text
}

/// One generation as a line: selection fitness, the champion's fresh
/// score with its standard error, species count, champion size, the
/// champion's score against each opponent alone, throughput and ETA. A
/// trailing `*` marks a new best champion.
#[must_use]
pub fn render_row(event: &GenerationEvent, eta_secs: Option<f64>) -> String {
    let champion = &event.champion;
    let mut text = format!(
        "{:>5} {:>+7.3} {:>+7.3}  {:>+7.3} ±{:<6.3}  {:>3}  {:>4}/{:<5}",
        event.generation,
        event.fitness.best,
        event.fitness.mean,
        champion.reeval.mean,
        champion.reeval.std_error,
        event.species.len(),
        champion.hidden_nodes,
        champion.enabled_connections,
    );
    for opponent in &event.opponents {
        let _ = write!(text, " {:>+6.2}", opponent.score.mean);
    }
    let _ = write!(
        text,
        "  {:>8}  {:>8}",
        format_rate(event.rounds_per_sec),
        eta_secs.map_or_else(|| "--:--:--".to_owned(), format_duration)
    );
    if champion.is_new_best {
        text.push_str(" *");
    }
    text
}

#[must_use]
pub fn render_summary(end: &RunEnd, out: &std::path::Path) -> String {
    let mut text = format!(
        "done: {} generations in {}",
        end.generations_completed,
        format_duration(end.elapsed_secs)
    );
    if let (Some(generation), Some(score)) = (end.best_generation, &end.best_reeval) {
        let _ = write!(
            text,
            "\nbest champion: generation {generation}, score {:+.3} ±{:.3} against the pool (fresh matches)\nplay it: cli --player-count 4 --matches 1000 --strategy neat:{} --strategy lowest-legal --strategy lowest-legal --strategy lowest-legal",
            score.mean,
            score.std_error,
            out.join("best.json").display()
        );
    }
    text
}

pub struct TerminalObserver {
    quiet: bool,
    out: PathBuf,
    live_progress: bool,
    progress_visible: bool,
    opponent_count: usize,
    total_generations: u32,
    first_generation: Option<u32>,
    generation_secs: Vec<f64>,
}

impl TerminalObserver {
    #[must_use]
    pub fn new(out: PathBuf, quiet: bool) -> Self {
        Self {
            quiet,
            out,
            live_progress: std::io::stderr().is_terminal(),
            progress_visible: false,
            opponent_count: 0,
            total_generations: 0,
            first_generation: None,
            generation_secs: Vec::new(),
        }
    }

    fn clear_progress(&mut self) {
        if self.progress_visible {
            eprint!("\r{:60}\r", "");
            self.progress_visible = false;
        }
    }

    fn eta(&self, event: &GenerationEvent) -> Option<f64> {
        if self.generation_secs.is_empty() {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let average = self.generation_secs.iter().sum::<f64>() / self.generation_secs.len() as f64;
        let remaining = self.total_generations.saturating_sub(event.generation + 1);
        Some(average * f64::from(remaining))
    }
}

impl TrainObserver for TerminalObserver {
    fn on_start(&mut self, start: &RunStart) {
        self.opponent_count = start.opponents.len();
        self.total_generations = start.config.generations;
        if !self.quiet {
            println!(
                "{}\n\n{}",
                render_banner(start, &self.out),
                render_header(self.opponent_count)
            );
        }
    }

    fn on_eval_progress(&mut self, generation: u32, done: usize, total: usize) {
        if self.quiet || !self.live_progress {
            return;
        }
        eprint!("\r  generation {generation}: evaluated {done}/{total} genomes ");
        let _ = std::io::stderr().flush();
        self.progress_visible = true;
    }

    fn on_generation(&mut self, event: &GenerationEvent) {
        self.clear_progress();
        self.generation_secs.push(event.generation_secs);
        let first = *self.first_generation.get_or_insert(event.generation);
        if self.quiet {
            return;
        }
        if (event.generation - first).is_multiple_of(HEADER_EVERY) && event.generation != first {
            println!("{}", render_header(self.opponent_count));
        }
        println!("{}", render_row(event, self.eta(event)));
    }

    fn on_finish(&mut self, end: &RunEnd) {
        self.clear_progress();
        println!("\n{}", render_summary(end, &self.out));
    }
}
```

- [ ] **Step 10: Implement**

Insert at the very top of `cli/src/train.rs`, above the `#[cfg(test)]` line:

```rust
//! `cli train`: wires the arguments to `sim::training::Trainer`.

use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use sim::training::{load_config, DeckChoice, DuplicateChoice, Opponent, TrainConfig, Trainer};

use crate::args::{DeckVariantArg, DuplicateRuleArg, StrategyArg};
use crate::train_args::{TrainArgs, DEFAULT_OPPONENTS};
use crate::train_output::TerminalObserver;

/// Builds the opponent pool from `--strategy`-style specs, rejecting two
/// opponents with the same display name (their results would merge).
fn build_opponents(specs: &[String]) -> anyhow::Result<Vec<Opponent>> {
    let mut opponents: Vec<Opponent> = Vec::new();
    for spec in specs {
        let strategy: Arc<dyn sim::Strategy> = spec
            .parse::<StrategyArg>()
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("invalid --opponent `{spec}`"))?
            .build();
        let name = strategy.name().to_owned();
        anyhow::ensure!(
            opponents.iter().all(|o| o.name != name),
            "--opponent `{spec}` duplicates `{name}`, which is already in the pool"
        );
        opponents.push(Opponent { name, strategy });
    }
    Ok(opponents)
}

fn new_config(args: &TrainArgs, specs: Vec<String>) -> TrainConfig {
    TrainConfig {
        seed: args.seed,
        player_count: args.player_count,
        deck: match args.deck_variant {
            DeckVariantArg::Single => DeckChoice::Single,
            DeckVariantArg::Double => DeckChoice::Double,
        },
        duplicate_rule: match args.duplicate_rule {
            DuplicateRuleArg::FirstDealtWins => DuplicateChoice::FirstDealtWins,
            DuplicateRuleArg::LastDealtWins => DuplicateChoice::LastDealtWins,
        },
        rounds_per_match: args.rounds,
        matches_per_genome: args.matches_per_genome,
        reeval_matches: args.reeval_matches,
        generations: args.generations.unwrap_or(100),
        neat: neat::NeatConfig {
            population_size: args.population,
            target_species: args.target_species,
            ..neat::NeatConfig::default()
        },
        opponent_specs: specs,
    }
}

pub fn run(raw_args: impl Iterator<Item = String>) -> anyhow::Result<()> {
    let args = TrainArgs::parse_from(std::iter::once("cli train".to_owned()).chain(raw_args));
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .context("failed to configure thread pool")?;
    }

    let mut trainer = if args.resume {
        let config = load_config(&args.out)?;
        let opponents = build_opponents(&config.opponent_specs)?;
        Trainer::resume(&args.out, opponents, args.generations)?
    } else {
        let specs: Vec<String> = if args.opponent.is_empty() {
            DEFAULT_OPPONENTS.iter().map(|&s| s.to_owned()).collect()
        } else {
            args.opponent.clone()
        };
        let opponents = build_opponents(&specs)?;
        Trainer::new(new_config(&args, specs), opponents, &args.out)?
    };

    let mut observer = TerminalObserver::new(args.out.clone(), args.quiet);
    trainer.run(&mut observer)?;
    Ok(())
}
```

- [ ] **Step 11: Edit**

In `cli/src/main.rs` (Dispatch `cli train` before the existing flat argument parser, which is unchanged), replace:

```rust
fn main() -> anyhow::Result<()> {
    let args = args::Args::parse();
```

with:

```rust
fn main() -> anyhow::Result<()> {
    // `cli train ...` evolves a player; everything else is a simulation run.
    if std::env::args().nth(1).as_deref() == Some("train") {
        return train::run(std::env::args().skip(2));
    }
    let args = args::Args::parse();
```

- [ ] **Step 12: Run (expect success)**

Run: `cargo fmt -p cli && cargo test -p cli --bins`

Expected: PASS (the existing args tests plus 5 for `TrainArgs`, 6 for the renderers and 3 for the opponent pool).

- [ ] **Step 13: Commit**

```bash
git add cli Cargo.lock
git commit -F - <<'EOF'
cli: add `cli train` with per-generation terminal output (Phase 10c)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F3sC3JdcgTFAox87bBuikA
EOF
```


### Task 8: Real-scenario verification

**Files:**
- Create: `cli/tests/train_smoke.rs`

**Why:** Unit and integration tests show each part behaves. These run the real binary the way a user will: a real training run, an independent measurement of the champion through the ordinary simulator, and a SIGKILL-and-resume.

**Interfaces:**
- Consumes: The built `cli` binary.
- Produces: Evidence that training works end to end on real runs (no new API).

- [ ] **Step 1: Create file**

Create `cli/tests/train_smoke.rs` (Smoke tests of the real binary):

```rust
//! End-to-end tests of `cli train`: the real binary, real files.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn run_dir(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("arschloch-cli-train-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_cli"))
        .args(args)
        .output()
        .expect("failed to run cli binary")
}

fn train(out: &Path, extra: &[&str]) -> Output {
    let mut args = vec![
        "train",
        "--out",
        out.to_str().unwrap(),
        "--population",
        "12",
        "--matches-per-genome",
        "4",
        "--reeval-matches",
        "6",
        "--rounds",
        "3",
        "--threads",
        "2",
    ];
    args.extend_from_slice(extra);
    cli(&args)
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// The generation numbers of the per-generation rows in `stdout`.
fn row_generations(stdout: &str) -> Vec<u32> {
    stdout
        .lines()
        .filter(|line| line.contains('±'))
        .filter_map(|line| line.split_whitespace().next()?.parse().ok())
        .collect()
}

#[test]
fn a_run_prints_a_row_per_generation_and_leaves_playable_champions() {
    let out = run_dir("run");
    let result = train(
        &out,
        &[
            "--generations",
            "3",
            "--opponent",
            "lowest-legal",
            "--opponent",
            "random-legal",
        ],
    );
    assert!(result.status.success(), "stderr: {}", text(&result.stderr));
    let stdout = text(&result.stdout);
    assert!(
        stdout.contains("o1=LowestLegal  o2=RandomLegal"),
        "{stdout}"
    );
    assert_eq!(row_generations(&stdout), vec![0, 1, 2], "{stdout}");
    assert!(stdout.contains("done: 3 generations"), "{stdout}");
    assert!(stdout.contains("best champion: generation"), "{stdout}");

    for file in [
        "config.json",
        "events.jsonl",
        "checkpoint.json",
        "best.json",
        "gen-0000.json",
        "gen-0002.json",
    ] {
        assert!(out.join(file).exists(), "{file} missing");
    }
    let events = std::fs::read_to_string(out.join("events.jsonl")).unwrap();
    assert_eq!(events.lines().count(), 5, "start, 3 generations, end");
    for line in events.lines() {
        serde_json::from_str::<serde_json::Value>(line).expect("every event line is JSON");
    }

    // The champion plays in an ordinary simulation run.
    let best = out.join("best.json");
    let played = cli(&[
        "--player-count",
        "4",
        "--matches",
        "10",
        "--rounds",
        "2",
        "--output",
        "/dev/null",
        "--strategy",
        &format!("neat:{}", best.display()),
        "--strategy",
        "lowest-legal",
        "--strategy",
        "lowest-legal",
        "--strategy",
        "lowest-legal",
    ]);
    assert!(played.status.success(), "stderr: {}", text(&played.stderr));
    assert!(text(&played.stdout).contains("Neat(best)"));
    std::fs::remove_dir_all(&out).unwrap();
}

#[test]
fn resume_continues_from_the_checkpoint_and_prints_only_new_generations() {
    let out = run_dir("resume");
    assert!(train(&out, &["--generations", "2"]).status.success());
    let result = cli(&[
        "train",
        "--out",
        out.to_str().unwrap(),
        "--resume",
        "--generations",
        "4",
        "--threads",
        "2",
    ]);
    assert!(result.status.success(), "stderr: {}", text(&result.stderr));
    let stdout = text(&result.stdout);
    assert!(stdout.contains("resumed from generation 2"), "{stdout}");
    assert_eq!(row_generations(&stdout), vec![2, 3], "{stdout}");
    assert!(stdout.contains("done: 4 generations"), "{stdout}");
    let events = std::fs::read_to_string(out.join("events.jsonl")).unwrap();
    let generations = events
        .lines()
        .filter(|l| l.contains(r#""type":"generation""#))
        .count();
    assert_eq!(generations, 4, "each generation is logged exactly once");
    std::fs::remove_dir_all(&out).unwrap();
}

#[test]
fn a_new_run_never_overwrites_an_existing_one() {
    let out = run_dir("overwrite");
    assert!(train(&out, &["--generations", "1"]).status.success());
    let again = train(&out, &["--generations", "1"]);
    assert!(!again.status.success());
    assert!(
        text(&again.stderr).contains("already holds a run"),
        "{}",
        text(&again.stderr)
    );
    std::fs::remove_dir_all(&out).unwrap();
}

#[test]
fn bad_input_fails_with_a_clear_message_and_no_panic() {
    let out = run_dir("bad");
    let bad_opponent = train(&out, &["--opponent", "nonsense"]);
    assert!(!bad_opponent.status.success());
    assert!(
        text(&bad_opponent.stderr).contains("--opponent `nonsense`"),
        "{}",
        text(&bad_opponent.stderr)
    );

    let duplicate = train(
        &out,
        &["--opponent", "lowest-legal", "--opponent", "lowest-legal"],
    );
    assert!(
        text(&duplicate.stderr).contains("duplicates `LowestLegal`"),
        "{}",
        text(&duplicate.stderr)
    );

    let conflict = cli(&[
        "train",
        "--out",
        out.to_str().unwrap(),
        "--resume",
        "--population",
        "9",
    ]);
    assert!(!conflict.status.success());
    assert!(
        text(&conflict.stderr).contains("cannot be used with"),
        "{}",
        text(&conflict.stderr)
    );

    let nothing = cli(&["train", "--out", out.to_str().unwrap(), "--resume"]);
    assert!(!nothing.status.success());
    assert!(
        text(&nothing.stderr).contains("no checkpoint.json"),
        "{}",
        text(&nothing.stderr)
    );
    for output in [&bad_opponent, &duplicate, &conflict, &nothing] {
        assert!(!text(&output.stderr).contains("panicked"));
    }
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn quiet_prints_only_the_summary() {
    let out = run_dir("quiet");
    let result = train(&out, &["--generations", "2", "--quiet"]);
    assert!(result.status.success());
    let stdout = text(&result.stdout);
    assert!(row_generations(&stdout).is_empty(), "{stdout}");
    assert!(!stdout.contains("opponents:"), "{stdout}");
    assert!(stdout.contains("done: 2 generations"), "{stdout}");
    assert_eq!(
        std::fs::read_to_string(out.join("events.jsonl"))
            .unwrap()
            .lines()
            .count(),
        4
    );
    std::fs::remove_dir_all(&out).unwrap();
}
```

- [ ] **Step 2: Run (expect success)**

Run: `cargo fmt -p cli && cargo test -p cli --test train_smoke`

Expected: PASS (5 tests: a run prints a row per generation and leaves champions that play in an ordinary simulation; `--resume` continues and prints only new generations; a new run never overwrites an existing one; bad opponents, duplicates, conflicting flags and a missing run fail with clear messages and no panic; `--quiet` prints only the summary). These test existing behaviour of the code above, so they cannot be seen failing first.

- [ ] **Step 3: Run (expect success)**

Run: `cargo build --release -p cli`

Expected: PASS.

- [ ] **Step 4: Real scenario**

Save as `scenario1.sh` (anywhere outside the repository, for example the system temp directory) and run it from the repository root with `bash scenario1.sh`:

```bash
#!/usr/bin/env bash
# Real scenario 1: train for real, then verify the champion through a
# different code path. The trainer reports each champion's fresh-match
# score against every opponent; here the saved best.json plays ordinary
# tables (sim::run_batch + sim::aggregate) and the role counts are turned
# into the same -1..+1 score. The two measurements must agree (within
# noise) and show a clearly better-than-even player.
set -euo pipefail
CLI=target/release/cli
OUT=$(mktemp -d)/run
$CLI train --out "$OUT" --population 80 --generations 30 --matches-per-genome 50 \
  --reeval-matches 100 --rounds 6 --seed 1 | tail -n 14
python3 - "$OUT" <<'PY'
import json, sys
events = [json.loads(l) for l in open(sys.argv[1] + "/events.jsonl")]
gens = [e for e in events if e["type"] == "generation"]
first, last = gens[0], gens[-1]
print(f"mean fitness gen0 {first['fitness']['mean']:+.3f} -> gen{last['generation']} {last['fitness']['mean']:+.3f}")
assert last["fitness"]["mean"] > first["fitness"]["mean"] + 0.4, "population did not improve"
PY
score() { awk '/^Neat\(/ {p=$2; v=$3; va=$4; a=$5; t=p+v+va+a; printf "%+.3f (President %.1f%%, last %.1f%%)", (p - a + (v - va)/3)/t, 100*p/t, 100*a/t}'; }
fail=0
for opp in lowest-legal endgame-denial "adaptive:reading,tempo,bully"; do
  line=$($CLI --player-count 4 --matches 2000 --rounds 6 --seed 424242 --threads 4 --output /dev/null \
    --strategy neat:"$OUT"/best.json --strategy "$opp" --strategy "$opp" --strategy "$opp" | score)
  echo "independent check vs 3x $opp: $line"
  value=${line%% *}
  python3 -c "import sys; sys.exit(0 if float('$value') > 0.3 else 1)" || fail=1
done
exit $fail
```

Expected: a 30-generation run prints its rows and summary; mean fitness rises by more than 0.4; and the independent check scores the champion above +0.3 against 3x each of lowest-legal, endgame-denial and adaptive:reading,tempo,bully (in preparation: +0.57, +0.61, +0.54, matching the trainer's own +0.59, +0.62, +0.56). Exit status 0.

- [ ] **Step 5: Real scenario**

Save as `scenario2.sh` (anywhere outside the repository, for example the system temp directory) and run it from the repository root with `bash scenario2.sh`:

```bash
#!/usr/bin/env bash
# Real scenario 2: kill a running training with SIGKILL mid-generation,
# resume it, and require the final state to be byte-identical to a run
# with the same seed that was never interrupted.
set -euo pipefail
CLI=target/release/cli
BASE=$(mktemp -d)
ARGS=(--population 40 --generations 14 --matches-per-genome 40 --reeval-matches 60 --rounds 6 --seed 5 --threads 3 --quiet)
$CLI train --out "$BASE/straight" "${ARGS[@]}" > /dev/null
$CLI train --out "$BASE/killed" "${ARGS[@]}" > /dev/null 2>&1 &
PID=$!
sleep 6
kill -9 $PID
wait $PID 2>/dev/null || true
echo "killed after $(grep -c '"type":"generation"' "$BASE/killed/events.jsonl") logged generations"
$CLI train --out "$BASE/killed" --resume --threads 3 --quiet > /dev/null
python3 - "$BASE" <<'PY'
import json, sys
base = sys.argv[1]
a = json.load(open(base + "/straight/checkpoint.json"))["population"]
b = json.load(open(base + "/killed/checkpoint.json"))["population"]
assert a == b, "final population states differ"
assert open(base + "/straight/best.json").read() == open(base + "/killed/best.json").read(), "best.json differs"
gens = [json.loads(l)["generation"] for l in open(base + "/killed/events.jsonl") if '"type":"generation"' in l]
assert gens == list(range(14)), gens
print("resumed run is byte-identical to the uninterrupted run; every generation logged once")
PY
```

Expected: `killed after N logged generations` (0 < N < 14), then `resumed run is byte-identical to the uninterrupted run; every generation logged once`. Exit status 0.

- [ ] **Step 6: Commit**

```bash
git add cli/tests/train_smoke.rs
git commit -F - <<'EOF'
cli: smoke-test cli train end to end (Phase 10c)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F3sC3JdcgTFAox87bBuikA
EOF
```


### Task 9: Docs and the full gate

**Files:**
- Modify: `docs/ROADMAP.md`, `docs/BUILDING.md`, `docs/ARCHITECTURE.md`, `docs/baselines/pre-neat/README.md`, `docs/superpowers/specs/2026-10-08-neat-engine-design.md`

**Interfaces:**
- Consumes: Everything above.
- Produces: Docs that match the code; a clean phase-done gate. (The full how-to-train guide is deferred, as requested; BUILDING.md gets a quick reference only.)

- [ ] **Step 1: Edit**

In `docs/ROADMAP.md` (Roadmap), replace:

```markdown
10c: fitness evaluation
```

with:

```markdown
10c (done): fitness evaluation
```

- [ ] **Step 2: Edit**

In `docs/superpowers/specs/2026-10-08-neat-engine-design.md` (Spec: record what was built and why it differs), replace:

```markdown
## 8. Determinism and performance
```

with:

```markdown
## 7b. As built in Phase 10c

- **Where:** evaluation, run files, events and the loop live in
  `sim/src/training/`; `cli train` only parses arguments and prints
  (`cli/src/train*.rs`). Run directory: `config.json`, `events.jsonl`,
  `checkpoint.json` (replaced atomically after every generation),
  `best.json`, `gen-NNNN.json`.
- **Evaluation:** a genome's score is its mean finishing-role score
  (+1 President .. -1 last) over `matches_per_genome` matches. Match
  seeds derive from `(run seed, generation, index)`, so every genome in
  a generation plays the same deals, seats and opponents. The candidate's
  seat rotates with the match index, but opponents are *sampled* from the
  pool per match, not rotated: table position relative to the other
  players is a large effect (a clone of `LowestLegal` and `LowestLegal`
  exchange their results exactly when their seats swap), and cyclic
  rotation keeps neighbour order fixed.
- **Re-evaluation:** each generation's champion is replayed on fresh
  seeds against the mixed pool and against every opponent alone; the
  event carries the score, its standard error and the placement counts.
  `best.json` follows this fresh score, not the selection fitness.
- **Default pool:** `lowest-legal`, `endgame-denial`,
  `adaptive:reading,tempo,bully`. `CardCounter` is omitted: it plays
  identically to `LowestLegal` in every table tried (0 of 200 all-same
  tables differ), so it adds nothing.
- **Deviations from section 7a, deliberately:** the event carries
  per-opponent placement counts rather than the full Phase 4 statistics
  (voluntary-pass and retention rates); the dashboard (10d) may add them.
  No decision samples are stored yet (10d's inspector).
- **Speciation:** the default compatibility threshold is 0.5 (not 3.0):
  random initial weights put genomes about 0.3 apart.
- **Resume:** the population state (including its xoshiro256++ generator)
  is checkpointed; resuming equals never stopping, also after SIGKILL.
  A crash between a generation's event and its checkpoint is repaired by
  trimming the duplicate event on resume.
- **Measured:** about 23k rounds/s on 4 cores in release; a 30-generation
  run (population 80, 50 matches x 6 rounds) takes about 35 s and its
  champion beats three copies of each default opponent with a score of
  about +0.55 to +0.6 (President in roughly half of all rounds),
  confirmed through the ordinary simulator.

## 8. Determinism and performance
```

- [ ] **Step 3: Edit**

In `docs/ARCHITECTURE.md`, replace:

```markdown
### `web` (starts in Phase 10d
```

with:

```markdown
`sim::training` (Phase 10c) holds fitness evaluation (`evaluate`,
common random numbers), the run's files (`RunDir`: atomic checkpoints,
`events.jsonl`, champion genome files), the event schema (`Event`) and
the resumable generational loop (`Trainer`, reporting through the
`TrainObserver` trait). `cli train` prints a row per generation from
those events; the dashboard (10d) will read the same files.

### `web` (starts in Phase 10d
```

- [ ] **Step 4: Edit**

In `docs/BUILDING.md`, replace:

```markdown
The `neat` crate has no game dependency:
```

with:

```markdown
Quick reference for training (a full guide comes later):

```bash
cargo build --release -p cli
target/release/cli train --out runs/first --generations 50     # new run
target/release/cli train --out runs/first --resume --generations 100
target/release/cli --player-count 4 --matches 1000 \
  --strategy neat:runs/first/best.json --strategy lowest-legal \
  --strategy lowest-legal --strategy lowest-legal
```

`cli train --help` lists every option. A run is reproducible from
`--seed`, resumable after any interruption, and independent of
`--threads`.

The `neat` crate has no game dependency:
```

- [ ] **Step 5: Edit**

In `docs/baselines/pre-neat/README.md`, replace:

```markdown
If a baseline strategy ever has to change,
```

with:

```markdown
**Caveats found while building Phase 10c.** (1) `CardCounter` plays
identically to `LowestLegal` (0 of 200 all-same tables differ), so the
differences between their rows in `mixed-field_*` are *not* skill. (2)
`run_batch` rotates seats cyclically, which keeps neighbour order, so
mixed-field rows carry a table-position bias (swapping two identical
players' seats swaps their results). The `vs-lowest-legal_*` files, where
one contender faces identical opponents, are the cleaner comparison.

If a baseline strategy ever has to change,
```

- [ ] **Step 6: Run (expect success)**

Run: `cargo fmt --check`

Expected: PASS.

- [ ] **Step 7: Run (expect success)**

Run: `cargo clippy --workspace --all-targets -- -D warnings`

Expected: Clean.

- [ ] **Step 8: Run (expect success)**

Run: `cargo test --workspace`

Expected: PASS: every suite green (see the ledger for the count).

- [ ] **Step 9: Commit**

```bash
git add docs Cargo.lock
git commit -F - <<'EOF'
docs: Phase 10c done; spec as-built notes, baseline caveats, training quick reference

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F3sC3JdcgTFAox87bBuikA
EOF
```


---

## Self-Review (done while writing)

- **Spec coverage (sections 6, 7, 7a terminal/log parts, 8):** common random numbers, fresh-seed champion re-evaluation with standard error, per-opponent scores, sampled opponents (Task 2, 6); `train` subcommand with checkpoint/resume (Tasks 5-7); terminal row per generation with ETA and a refreshed progress line, `events.jsonl` (Tasks 4, 6, 7); determinism independent of thread count and exact resume (Task 6, scenario 2). Deliberate differences are written into the spec by Task 9 (section 7b): no full Phase 4 statistics or decision samples yet, no complexity-pressure tie-breaking (10e), default pool without `CardCounter`.
- **Placeholder scan:** none; every step carries its code.
- **Type consistency:** `Trainer`, `Opponent`, `TrainObserver`, `TrainConfig`, `Event`, `Score`/`ScoreStat`, `RunDir`/`Checkpoint`/`TrainError`, `PopulationState` and `NeatStrategy` are used with the same signatures in every task that names them.
- **Review Focus:** all five lines map to named tests or scenarios above.
