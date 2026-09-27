# Phase 4 — Deeper Statistics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three new statistical signals to `sim` (role-sustainment,
luck-vs-skill outcome variance, per-strategy voluntary-pass
diversification plus a new strategy that exercises it), surface them
through `cli`'s existing summary table and JSON output, and add a
top-level `README.md` that documents the whole project end-to-end.

**Architecture:** `sim::MatchResult`'s two match-wide pass counters
become per-seat vectors so per-strategy breakdowns are possible;
`sim::statistics::aggregate` gains three new output fields computed from
that (and from the already-existing `role_history`); a new
`sim::HoldBackPairs` strategy gives the diversification signal something
non-trivial to measure; `cli` registers the new strategy and prints the
three new numbers.

**Tech Stack:** Rust workspace (`engine`, `sim`, `cli`), `serde`/
`serde_json`, `clap`, `rayon` — no new dependencies this phase.

**Spec:** `docs/ROADMAP.md`, "Phase 4 — Deeper statistics"; the native
plan-mode design doc this plan expands
(`/home/sebi/.claude/plans/plan-phase-1-kind-globe.md`, same content,
already approved).

## Global Constraints

- No changes to `engine`'s public API or dependency list this phase —
  every new signal is computable from data `sim` already has
  (`role_history`, per-seat move choices) or from existing `engine`
  functions (`roles_for_player_count`).
- No new external dependencies (`cargo add` not needed this phase).
- `MatchResult`'s JSON shape changes (`pass_count`/`voluntary_pass_count`
  scalars become `pass_counts`/`voluntary_pass_counts` vectors) — this is
  an intentional, allowed breaking change to the on-disk schema (see
  plan Context: no consumer has been built against it yet).
- Tasks 1-3 touch only `sim`; verify them with `cargo test -p sim` and
  `cargo fmt --check -p sim` (not full-workspace clippy — `cli` isn't
  updated to match `sim`'s new `Statistics` shape until Task 4, so a
  workspace-wide clippy/build before then is expected to still pass
  since `cli` doesn't read the new fields yet, but there's no need to
  chase full-workspace warnings on every intermediate task). Tasks 4-5
  verify with the full workspace gate (`cargo fmt --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace`).
- Every new public field/type keeps a doc comment pointing at the
  ROADMAP.md phase/section it implements, matching the existing style in
  `sim/src/statistics.rs` and `sim/src/match_result.rs`.

## Review Focus

- A batch smaller than `2 * strategies.len()` matches yields every
  `first_round_placement_variance_by_strategy` entry as `None` — the
  `cli` summary must print something readable ("not enough data"), never
  a crash or a literal `None`/`null`. (Task 2 tests the aggregation side;
  Task 4 tests the display side.)
- A match configured with `rounds == 1` contributes zero
  round-to-round transition windows, so `role_retention_by_strategy` can
  end up with no entry at all for a strategy that only ever appeared in
  single-round matches — any consumer computing a percentage (`cli`'s
  summary) must treat a missing entry, or `held == 0`, as "no data"
  rather than dividing by zero into `NaN`. (Task 2 and Task 4.)
- Two seats running the exact same strategy (e.g.
  `--strategy lowest-legal --strategy lowest-legal`) must aggregate into
  one shared bucket per strategy name for every new field, consistent
  with the existing (Phase 2) `role_counts_by_strategy` behavior — not
  silently split by seat index. (Task 2.)
- `HoldBackPairs` must never panic and must degrade to ordinary play when
  following with no legal `Play` at all, and when leading or following
  with no rank held more than once anywhere in the offered moves. (Task 3.)
- A `--matches` count that isn't an exact multiple of the number of
  `--strategy` flags (`sim::run_batch`'s seat-rotation cycle length)
  still produces well-formed, non-panicking statistics — a partial final
  rotation cycle is a smaller-but-valid seating group, not a case that
  needs special-casing or filtering out. (Task 2.)

---

### Task 1: Per-seat pass counters in `MatchResult`

**Files:**
- Modify: `sim/src/match_result.rs`
- Modify: `sim/src/match_runner.rs`
- Modify: `sim/src/statistics.rs` (only the two lines that read the old
  scalar fields, plus the test helper's signature — no new fields yet)
- Modify: `sim/tests/small_batch.rs`
- Modify: `sim/tests/multi_config.rs`

**Interfaces:**
- Consumes: nothing new (reshapes existing fields).
- Produces: `MatchResult::pass_counts: Vec<u32>` and
  `MatchResult::voluntary_pass_counts: Vec<u32>`, one entry per seat
  (replacing the old `pass_count: u32`/`voluntary_pass_count: u32`
  scalars). Task 2's `aggregate` rewrite and Task 3/4 all build on this
  shape.

- [ ] **Step 1: Change `MatchResult`'s fields**

Replace the whole struct body in `sim/src/match_result.rs`:

```rust
//! The outcome of one simulated match: role history and per-seat
//! move-shape counters consumed by `crate::statistics::aggregate`.

use engine::Role;

#[derive(Debug, Clone, serde::Serialize)]
pub struct MatchResult {
    pub player_count: u8,
    /// One entry per seat (index = `SeatId`).
    pub strategy_names: Vec<String>,
    /// One entry per round played, each seat-indexed.
    pub role_history: Vec<Vec<Role>>,
    pub trick_count: u32,
    /// One entry per seat: how many passes that seat submitted across the
    /// whole match.
    pub pass_counts: Vec<u32>,
    /// One entry per seat: how many of that seat's passes were voluntary
    /// (a `Move::Play` was also legal — see docs/ROADMAP.md, Phase 4,
    /// "Strategy diversification").
    pub voluntary_pass_counts: Vec<u32>,
}
```

- [ ] **Step 2: Update `run_match` to track per seat**

In `sim/src/match_runner.rs`, find the accumulator declarations (near the
top of `run_match`, currently reading):

```rust
    let mut trick_count = 0u32;
    let mut pass_count = 0u32;
    let mut voluntary_pass_count = 0u32;
```

Replace with:

```rust
    let mut trick_count = 0u32;
    let mut pass_counts = vec![0u32; usize::from(config.player_count)];
    let mut voluntary_pass_counts = vec![0u32; usize::from(config.player_count)];
```

Find the pass-counting block inside the trick loop (currently):

```rust
            if chosen == Move::Pass {
                pass_count += 1;
                if legal_moves.iter().any(|mv| matches!(mv, Move::Play(_))) {
                    voluntary_pass_count += 1;
                }
            }
```

Replace with:

```rust
            if chosen == Move::Pass {
                pass_counts[usize::from(seat)] += 1;
                if legal_moves.iter().any(|mv| matches!(mv, Move::Play(_))) {
                    voluntary_pass_counts[usize::from(seat)] += 1;
                }
            }
```

Find the final `MatchResult` construction (currently):

```rust
    MatchResult {
        player_count: config.player_count,
        strategy_names: strategies.iter().map(|s| s.name().to_string()).collect(),
        role_history,
        trick_count,
        pass_count,
        voluntary_pass_count,
    }
```

Replace the last two fields:

```rust
    MatchResult {
        player_count: config.player_count,
        strategy_names: strategies.iter().map(|s| s.name().to_string()).collect(),
        role_history,
        trick_count,
        pass_counts,
        voluntary_pass_counts,
    }
```

- [ ] **Step 3: Fix `match_runner.rs`'s own test**

In its `#[cfg(test)] mod tests`, the
`identical_config_and_strategies_are_fully_deterministic` test has:

```rust
        assert_eq!(first.pass_count, second.pass_count);
```

Change to:

```rust
        assert_eq!(first.pass_counts, second.pass_counts);
```

- [ ] **Step 4: Fix `sim/tests/small_batch.rs`**

Same rename, in `identical_seeds_produce_identical_results`:

```rust
    assert_eq!(first.pass_count, second.pass_count);
```

becomes

```rust
    assert_eq!(first.pass_counts, second.pass_counts);
```

- [ ] **Step 5: Fix `sim/tests/multi_config.rs`**

In `every_table_size_and_deck_variant_plays_to_completion_with_valid_roles`,
replace:

```rust
                assert!(result.trick_count >= 3, "{label}");
                assert!(result.voluntary_pass_count <= result.pass_count, "{label}");
```

with:

```rust
                assert!(result.trick_count >= 3, "{label}");
                for seat in 0..result.pass_counts.len() {
                    assert!(
                        result.voluntary_pass_counts[seat] <= result.pass_counts[seat],
                        "{label}"
                    );
                }
```

- [ ] **Step 6: Update `sim/src/statistics.rs`'s totals computation and test helper**

`aggregate`'s per-match loop currently has:

```rust
        total_passes += u64::from(result.pass_count);
        total_voluntary_passes += u64::from(result.voluntary_pass_count);
```

Replace with:

```rust
        for seat in 0..result.pass_counts.len() {
            total_passes += u64::from(result.pass_counts[seat]);
            total_voluntary_passes += u64::from(result.voluntary_pass_counts[seat]);
        }
```

The test module's `result(...)` helper currently takes scalar
`pass_count: u32, voluntary: u32`. Change its signature and body to:

```rust
    fn result(
        strategy_names: Vec<&str>,
        roles: Vec<Role>,
        pass_counts: Vec<u32>,
        voluntary_pass_counts: Vec<u32>,
    ) -> MatchResult {
        MatchResult {
            player_count: u8::try_from(strategy_names.len()).unwrap(),
            strategy_names: strategy_names.into_iter().map(String::from).collect(),
            role_history: vec![roles],
            trick_count: 0,
            pass_counts,
            voluntary_pass_counts,
        }
    }
```

Update the two existing tests' call sites. In
`aggregate_counts_matches_and_roles_by_strategy`:

```rust
        let results = vec![
            result(
                vec!["LowestLegal", "GreedyHighest"],
                vec![Role::President, Role::Arschloch],
                vec![5, 5],
                vec![1, 1],
            ),
            result(
                vec!["LowestLegal", "GreedyHighest"],
                vec![Role::Arschloch, Role::President],
                vec![5, 5],
                vec![4, 4],
            ),
        ];
```

(Total pass = 20, total voluntary = 10 across both matches — same 0.5
pooled rate the test already asserts.)

In `aggregate_of_no_passes_reports_zero_rate`:

```rust
        let results = vec![result(vec!["LowestLegal"], vec![Role::President], vec![0], vec![0])];
```

- [ ] **Step 7: Run tests**

Run: `cargo test -p sim`
Expected: PASS (all existing `sim` tests, `MatchResult`'s new shape
compiles and round-trips).

- [ ] **Step 8: Format and commit**

Run: `cargo fmt --check -p sim` (fix if needed, then re-run), then:

```bash
git add sim/src/match_result.rs sim/src/match_runner.rs sim/src/statistics.rs sim/tests/small_batch.rs sim/tests/multi_config.rs
git commit -m "sim: track pass counts per seat instead of per match"
```

---

### Task 2: The three new `Statistics` signals

**Files:**
- Modify: `sim/src/statistics.rs` (full rewrite of `aggregate` and
  `Statistics`, plus new tests)
- Modify: `sim/src/lib.rs` (re-export `RoleRetention`)
- Modify: `sim/tests/small_batch.rs` (assert the new JSON keys exist)

**Interfaces:**
- Consumes: `MatchResult::{pass_counts, voluntary_pass_counts,
  role_history, strategy_names, player_count}` (Task 1's shape);
  `engine::roles_for_player_count`.
- Produces: `Statistics::{voluntary_pass_rate_by_strategy,
  role_retention_by_strategy, first_round_placement_variance_by_strategy}`
  and the new `RoleRetention` type — Task 4's `cli::summary` reads all
  four by name.

- [ ] **Step 1: Replace `sim/src/statistics.rs` in full**

```rust
//! Aggregating a batch of `MatchResult`s into per-strategy role counts,
//! diversification, role-sustainment, and luck-vs-skill signals. See
//! docs/ROADMAP.md, Phase 2 and 4.

use std::collections::{BTreeMap, HashMap};

use engine::{roles_for_player_count, Role};

use crate::match_result::MatchResult;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct RoleRetention {
    /// Round-to-next-round transitions where a strategy held this role
    /// going in (and a next round existed to check).
    pub held: u32,
    /// Of `held`, how many kept the same role the very next round.
    pub retained_next_round: u32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Statistics {
    pub matches_played: usize,
    pub role_counts_by_strategy: BTreeMap<String, BTreeMap<Role, u32>>,
    /// `sum(voluntary_pass_counts) / sum(pass_counts)` across every match
    /// and seat (`0.0` if no passes occurred at all).
    pub voluntary_pass_rate: f64,
    /// Same ratio, broken out per strategy name (docs/ROADMAP.md, Phase 4,
    /// "Strategy diversification").
    pub voluntary_pass_rate_by_strategy: BTreeMap<String, f64>,
    /// Role-sustainment: of the times a strategy held a role with at
    /// least one more round left in the match, how often it held the
    /// *same* role the next round too (docs/ROADMAP.md, Phase 4,
    /// "Role-sustainment tracking"). A strategy that only ever appeared
    /// in single-round matches has no entry here at all (there is no
    /// "next round" to check).
    pub role_retention_by_strategy: BTreeMap<String, BTreeMap<Role, RoleRetention>>,
    /// Luck-vs-skill: variance of a strategy's first-round finishing
    /// placement (0 = the table's best role, per
    /// `engine::roles_for_player_count`'s order) across matches that
    /// seated it identically (the same `MatchResult::strategy_names`)
    /// but necessarily shuffled differently (docs/ROADMAP.md, Phase 4,
    /// "Luck-vs-skill signal"). `None` if every seating this strategy
    /// appeared in had fewer than 2 matches to compare (nothing to take a
    /// variance of) — e.g. any batch smaller than roughly
    /// `2 * strategies.len()` matches.
    pub first_round_placement_variance_by_strategy: BTreeMap<String, Option<f64>>,
}

/// Aggregates `results` into `Statistics`. Every round in every match
/// contributes one role count per seat, credited to that seat's
/// strategy (by name — seats using the same strategy type share a
/// bucket, for every field below, not just `role_counts_by_strategy`).
#[must_use]
pub fn aggregate(results: &[MatchResult]) -> Statistics {
    let mut role_counts_by_strategy: BTreeMap<String, BTreeMap<Role, u32>> = BTreeMap::new();
    let mut total_passes = 0u64;
    let mut total_voluntary_passes = 0u64;
    let mut passes_by_strategy: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let mut retention_by_strategy: BTreeMap<String, BTreeMap<Role, RoleRetention>> =
        BTreeMap::new();
    // Keyed by the exact seating (MatchResult::strategy_names) so that,
    // within one key, the only thing that differs between matches is the
    // shuffle.
    let mut placements_by_seating: HashMap<Vec<String>, BTreeMap<String, Vec<f64>>> =
        HashMap::new();

    for result in results {
        for round_roles in &result.role_history {
            for (seat, &role) in round_roles.iter().enumerate() {
                let strategy_name = result.strategy_names[seat].clone();
                *role_counts_by_strategy
                    .entry(strategy_name)
                    .or_default()
                    .entry(role)
                    .or_insert(0) += 1;
            }
        }

        for (seat, strategy_name) in result.strategy_names.iter().enumerate() {
            let voluntary = u64::from(result.voluntary_pass_counts[seat]);
            let total = u64::from(result.pass_counts[seat]);
            total_passes += total;
            total_voluntary_passes += voluntary;
            let entry = passes_by_strategy
                .entry(strategy_name.clone())
                .or_insert((0, 0));
            entry.0 += voluntary;
            entry.1 += total;
        }

        for window in result.role_history.windows(2) {
            let (current, next) = (&window[0], &window[1]);
            for (seat, &role) in current.iter().enumerate() {
                let retention = retention_by_strategy
                    .entry(result.strategy_names[seat].clone())
                    .or_default()
                    .entry(role)
                    .or_default();
                retention.held += 1;
                if next[seat] == role {
                    retention.retained_next_round += 1;
                }
            }
        }

        if let Some(first_round) = result.role_history.first() {
            let roles = roles_for_player_count(result.player_count)
                .expect("MatchResult always comes from a supported player_count");
            let bucket = placements_by_seating
                .entry(result.strategy_names.clone())
                .or_default();
            for (seat, &role) in first_round.iter().enumerate() {
                let placement_index = roles
                    .iter()
                    .position(|&r| r == role)
                    .expect("assign_roles only assigns roles from this table's own role list");
                #[allow(clippy::cast_precision_loss)]
                let placement = placement_index as f64;
                bucket
                    .entry(result.strategy_names[seat].clone())
                    .or_default()
                    .push(placement);
            }
        }
    }

    let voluntary_pass_rate = ratio(total_voluntary_passes, total_passes);
    let voluntary_pass_rate_by_strategy = passes_by_strategy
        .into_iter()
        .map(|(name, (voluntary, total))| (name, ratio(voluntary, total)))
        .collect();

    // Pooled variance across every seating a strategy appeared in
    // (weighted by degrees of freedom), skipping any seating with fewer
    // than 2 matches to compare.
    let mut variance_weighted: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    for bucket in placements_by_seating.values() {
        for (strategy_name, placements) in bucket {
            if placements.len() < 2 {
                continue;
            }
            #[allow(clippy::cast_precision_loss)]
            let weight = (placements.len() - 1) as f64;
            let entry = variance_weighted
                .entry(strategy_name.clone())
                .or_insert((0.0, 0.0));
            entry.0 += sample_variance(placements) * weight;
            entry.1 += weight;
        }
    }
    let mut first_round_placement_variance_by_strategy: BTreeMap<String, Option<f64>> =
        role_counts_by_strategy
            .keys()
            .map(|name| (name.clone(), None))
            .collect();
    for (name, (weighted_sum, weight)) in variance_weighted {
        first_round_placement_variance_by_strategy.insert(name, Some(weighted_sum / weight));
    }

    Statistics {
        matches_played: results.len(),
        role_counts_by_strategy,
        voluntary_pass_rate,
        voluntary_pass_rate_by_strategy,
        role_retention_by_strategy: retention_by_strategy,
        first_round_placement_variance_by_strategy,
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        // Precision loss only matters above 2^53 passes — far beyond any
        // simulated batch this project runs.
        #[allow(clippy::cast_precision_loss)]
        {
            numerator as f64 / denominator as f64
        }
    }
}

/// Sample variance (Bessel's correction, `n - 1` divisor).
///
/// # Panics
///
/// Never in practice: every caller only invokes this with
/// `values.len() >= 2`.
fn sample_variance(values: &[f64]) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(
        strategy_names: Vec<&str>,
        role_history: Vec<Vec<Role>>,
        pass_counts: Vec<u32>,
        voluntary_pass_counts: Vec<u32>,
    ) -> MatchResult {
        MatchResult {
            player_count: u8::try_from(strategy_names.len()).unwrap(),
            strategy_names: strategy_names.into_iter().map(String::from).collect(),
            role_history,
            trick_count: 0,
            pass_counts,
            voluntary_pass_counts,
        }
    }

    #[test]
    fn aggregate_counts_matches_and_roles_by_strategy() {
        let results = vec![
            result(
                vec!["LowestLegal", "GreedyHighest"],
                vec![vec![Role::President, Role::Arschloch]],
                vec![5, 5],
                vec![1, 1],
            ),
            result(
                vec!["LowestLegal", "GreedyHighest"],
                vec![vec![Role::Arschloch, Role::President]],
                vec![5, 5],
                vec![4, 4],
            ),
        ];
        let stats = aggregate(&results);
        assert_eq!(stats.matches_played, 2);
        assert_eq!(
            stats.role_counts_by_strategy["LowestLegal"][&Role::President],
            1
        );
        assert_eq!(
            stats.role_counts_by_strategy["LowestLegal"][&Role::Arschloch],
            1
        );
        assert!((stats.voluntary_pass_rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn aggregate_of_no_passes_reports_zero_rate() {
        let results = vec![result(
            vec!["LowestLegal"],
            vec![vec![Role::President]],
            vec![0],
            vec![0],
        )];
        let stats = aggregate(&results);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(stats.voluntary_pass_rate, 0.0);
        }
    }

    #[test]
    fn voluntary_pass_rate_by_strategy_distinguishes_strategies() {
        let results = vec![
            result(
                vec!["LowestLegal", "GreedyHighest"],
                vec![vec![Role::President, Role::Arschloch]],
                vec![10, 10],
                vec![0, 10],
            ),
            result(
                vec!["LowestLegal", "GreedyHighest"],
                vec![vec![Role::Arschloch, Role::President]],
                vec![10, 10],
                vec![0, 10],
            ),
        ];
        let stats = aggregate(&results);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(stats.voluntary_pass_rate_by_strategy["LowestLegal"], 0.0);
            assert_eq!(stats.voluntary_pass_rate_by_strategy["GreedyHighest"], 1.0);
        }
    }

    #[test]
    fn role_retention_counts_round_to_round_transitions() {
        let results = vec![result(
            vec!["LowestLegal", "RandomLegal"],
            vec![
                vec![Role::President, Role::Arschloch],
                vec![Role::President, Role::Arschloch],
                vec![Role::Arschloch, Role::President],
            ],
            vec![0, 0],
            vec![0, 0],
        )];
        let stats = aggregate(&results);
        assert_eq!(
            stats.role_retention_by_strategy["LowestLegal"][&Role::President],
            RoleRetention {
                held: 2,
                retained_next_round: 1
            }
        );
        assert_eq!(
            stats.role_retention_by_strategy["RandomLegal"][&Role::Arschloch],
            RoleRetention {
                held: 2,
                retained_next_round: 1
            }
        );
    }

    #[test]
    fn single_round_matches_produce_no_retention_entries() {
        let results = vec![result(
            vec!["LowestLegal"],
            vec![vec![Role::President]],
            vec![0],
            vec![0],
        )];
        let stats = aggregate(&results);
        assert!(stats.role_retention_by_strategy.is_empty());
    }

    #[test]
    fn first_round_placement_variance_is_computed_across_repeated_identical_seatings() {
        let results = vec![
            result(
                vec!["LowestLegal", "GreedyHighest", "RandomLegal"],
                vec![vec![Role::President, Role::Dorftrottel, Role::Arschloch]],
                vec![0, 0, 0],
                vec![0, 0, 0],
            ),
            result(
                vec!["LowestLegal", "GreedyHighest", "RandomLegal"],
                vec![vec![Role::Arschloch, Role::Dorftrottel, Role::President]],
                vec![0, 0, 0],
                vec![0, 0, 0],
            ),
        ];
        let stats = aggregate(&results);
        // Placements (0=President, 1=Dorftrottel, 2=Arschloch): LowestLegal
        // saw [0, 2], GreedyHighest saw [1, 1], RandomLegal saw [2, 0].
        let lowest_legal_variance =
            stats.first_round_placement_variance_by_strategy["LowestLegal"]
                .expect("seating repeated twice");
        assert!((lowest_legal_variance - 2.0).abs() < 1e-9);
        let greedy_highest_variance =
            stats.first_round_placement_variance_by_strategy["GreedyHighest"]
                .expect("seating repeated twice");
        assert!((greedy_highest_variance - 0.0).abs() < 1e-9);
        let random_legal_variance = stats.first_round_placement_variance_by_strategy
            ["RandomLegal"]
            .expect("seating repeated twice");
        assert!((random_legal_variance - 2.0).abs() < 1e-9);
    }

    #[test]
    fn first_round_placement_variance_is_none_for_a_seating_seen_only_once() {
        let results = vec![result(
            vec!["LowestLegal", "GreedyHighest", "RandomLegal"],
            vec![vec![Role::President, Role::Dorftrottel, Role::Arschloch]],
            vec![0, 0, 0],
            vec![0, 0, 0],
        )];
        let stats = aggregate(&results);
        assert_eq!(
            stats.first_round_placement_variance_by_strategy["LowestLegal"],
            None
        );
        assert_eq!(
            stats.first_round_placement_variance_by_strategy["GreedyHighest"],
            None
        );
        assert_eq!(
            stats.first_round_placement_variance_by_strategy["RandomLegal"],
            None
        );
    }

    #[test]
    fn a_strategy_used_in_two_seats_shares_one_bucket_across_every_field() {
        let results = vec![result(
            vec!["LowestLegal", "LowestLegal"],
            vec![
                vec![Role::President, Role::Arschloch],
                vec![Role::President, Role::Arschloch],
            ],
            vec![3, 5],
            vec![1, 2],
        )];
        let stats = aggregate(&results);
        // Both seats' role assignments land in the same "LowestLegal" bucket.
        assert_eq!(
            stats.role_counts_by_strategy["LowestLegal"][&Role::President],
            1
        );
        assert_eq!(
            stats.role_counts_by_strategy["LowestLegal"][&Role::Arschloch],
            1
        );
        // Both seats' pass counts pool into one rate: (1+2)/(3+5) = 0.375.
        assert!(
            (stats.voluntary_pass_rate_by_strategy["LowestLegal"] - 0.375).abs() < f64::EPSILON
        );
    }
}
```

- [ ] **Step 2: Re-export `RoleRetention`**

In `sim/src/lib.rs`, change:

```rust
pub use statistics::{aggregate, Statistics};
```

to:

```rust
pub use statistics::{aggregate, RoleRetention, Statistics};
```

- [ ] **Step 3: Extend the small-batch integration test**

In `sim/tests/small_batch.rs`, after the existing
`assert!(json.contains("matches_played"));` line in
`a_small_batch_produces_well_shaped_results`, add:

```rust
    assert!(json.contains("voluntary_pass_rate_by_strategy"));
    assert!(json.contains("role_retention_by_strategy"));
    assert!(json.contains("first_round_placement_variance_by_strategy"));
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p sim`
Expected: PASS (all `sim` unit and integration tests, including the new
ones above).

- [ ] **Step 5: Format and commit**

Run: `cargo fmt --check -p sim` (fix if needed, then re-run), then:

```bash
git add sim/src/statistics.rs sim/src/lib.rs sim/tests/small_batch.rs
git commit -m "sim: add role-sustainment, luck-vs-skill, and per-strategy pass-rate statistics"
```

---

### Task 3: The `HoldBackPairs` strategy

**Files:**
- Create: `sim/src/strategies/hold_back_pairs.rs`
- Modify: `sim/src/strategies/mod.rs`
- Modify: `sim/src/lib.rs`
- Modify: `sim/tests/multi_config.rs`

**Interfaces:**
- Consumes: `sim::strategies::LowestLegal` (delegated to while leading);
  `engine::{Card, DuplicateRule, Move, Rank}`.
- Produces: `sim::HoldBackPairs` (implements `sim::Strategy`) — Task 4's
  `cli::args::StrategyArg` gets a matching variant.

- [ ] **Step 1: Write `sim/src/strategies/hold_back_pairs.rs`**

```rust
//! Follows without splitting a same-rank reserve when a non-splitting
//! beating play is available; passes rather than split one when every
//! beating play would. Leading is unaffected (see docs/ROADMAP.md, Phase
//! 4, "Strategy diversification") — this strategy differs from
//! `LowestLegal` only in how it follows.

use std::collections::HashMap;

use engine::{Card, DuplicateRule, Move, Rank};

use crate::strategies::LowestLegal;
use crate::strategy::Strategy;

#[derive(Debug, Clone, Copy, Default)]
pub struct HoldBackPairs;

impl Strategy for HoldBackPairs {
    fn name(&self) -> &'static str {
        "HoldBackPairs"
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        rng: &mut dyn rand::Rng,
    ) -> Move {
        if !legal_moves.contains(&Move::Pass) {
            // Leading never offers Pass (engine::Round::legal_moves), so
            // there's no split-vs-preserve choice to make here.
            return LowestLegal.choose_play(legal_moves, duplicate_rule, rng);
        }

        // Following: every candidate combo already shares the current
        // combo's size, so two candidates at the same rank mean the hand
        // holds more of that rank than this play uses — playing either
        // strands the rest.
        let plays: Vec<(Rank, Card)> = legal_moves
            .iter()
            .filter_map(|mv| match mv {
                Move::Play(combo) => {
                    let top = combo.top_card(duplicate_rule);
                    Some((top.rank, top))
                }
                Move::Pass => None,
            })
            .collect();
        let mut candidates_per_rank: HashMap<Rank, u8> = HashMap::new();
        for &(rank, _) in &plays {
            *candidates_per_rank.entry(rank).or_insert(0) += 1;
        }

        let safe_top_card = plays
            .iter()
            .filter(|(rank, _)| candidates_per_rank[rank] == 1)
            .map(|&(_, card)| card)
            .min_by(|a, b| a.compare(b, duplicate_rule));

        match safe_top_card {
            Some(card) => legal_moves
                .iter()
                .find(|mv| {
                    matches!(mv, Move::Play(combo) if combo.top_card(duplicate_rule) == card)
                })
                .expect("safe_top_card was derived from a Play move in legal_moves")
                .clone(),
            None => Move::Pass,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::{Combo, Suit};
    use rand::SeedableRng;

    fn test_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(0)
    }

    fn single(rank: Rank, suit: Suit) -> Move {
        Move::Play(Combo::new(vec![Card::new(rank, suit, 0)]).unwrap())
    }

    #[test]
    fn leading_defers_to_lowest_legal() {
        let legal = vec![
            single(Rank::Nine, Suit::Clubs),
            single(Rank::Six, Suit::Diamonds),
        ];
        let chosen =
            HoldBackPairs.choose_play(&legal, DuplicateRule::FirstDealtWins, &mut test_rng());
        assert_eq!(chosen, single(Rank::Six, Suit::Diamonds));
    }

    #[test]
    fn following_prefers_a_rank_with_no_spare_over_a_lower_rank_that_has_one() {
        // Six is offered twice (a spare Six exists in hand), Nine once
        // (no spare) — HoldBackPairs must pick Nine even though Six is
        // lower, to avoid splitting the pair.
        let legal = vec![
            Move::Pass,
            single(Rank::Six, Suit::Diamonds),
            single(Rank::Six, Suit::Spades),
            single(Rank::Nine, Suit::Clubs),
        ];
        let chosen =
            HoldBackPairs.choose_play(&legal, DuplicateRule::FirstDealtWins, &mut test_rng());
        assert_eq!(chosen, single(Rank::Nine, Suit::Clubs));
    }

    #[test]
    fn following_passes_rather_than_split_the_only_beating_rank() {
        let legal = vec![
            Move::Pass,
            single(Rank::Six, Suit::Diamonds),
            single(Rank::Six, Suit::Spades),
        ];
        let chosen =
            HoldBackPairs.choose_play(&legal, DuplicateRule::FirstDealtWins, &mut test_rng());
        assert_eq!(chosen, Move::Pass);
    }

    #[test]
    fn following_with_no_spares_anywhere_plays_the_lowest_like_lowest_legal() {
        let legal = vec![
            Move::Pass,
            single(Rank::Nine, Suit::Clubs),
            single(Rank::Six, Suit::Diamonds),
        ];
        let chosen =
            HoldBackPairs.choose_play(&legal, DuplicateRule::FirstDealtWins, &mut test_rng());
        assert_eq!(chosen, single(Rank::Six, Suit::Diamonds));
    }

    #[test]
    fn following_with_no_legal_play_at_all_passes() {
        let legal = vec![Move::Pass];
        let chosen =
            HoldBackPairs.choose_play(&legal, DuplicateRule::FirstDealtWins, &mut test_rng());
        assert_eq!(chosen, Move::Pass);
    }
}
```

- [ ] **Step 2: Run the new tests**

Run: `cargo test -p sim strategies::hold_back_pairs`
Expected: FAIL with "module `hold_back_pairs` not found" (module not yet
registered — confirms the test file alone doesn't compile in isolation).

- [ ] **Step 3: Register the module**

In `sim/src/strategies/mod.rs`, change:

```rust
mod greedy_highest;
mod lowest_legal;
mod random_legal;

pub use greedy_highest::GreedyHighest;
pub use lowest_legal::LowestLegal;
pub use random_legal::RandomLegal;
```

to:

```rust
mod greedy_highest;
mod hold_back_pairs;
mod lowest_legal;
mod random_legal;

pub use greedy_highest::GreedyHighest;
pub use hold_back_pairs::HoldBackPairs;
pub use lowest_legal::LowestLegal;
pub use random_legal::RandomLegal;
```

In `sim/src/lib.rs`, change:

```rust
pub use strategies::{GreedyHighest, LowestLegal, RandomLegal};
```

to:

```rust
pub use strategies::{GreedyHighest, HoldBackPairs, LowestLegal, RandomLegal};
```

- [ ] **Step 4: Run tests again**

Run: `cargo test -p sim`
Expected: PASS (all `sim` tests, including the 5 new `HoldBackPairs`
tests).

- [ ] **Step 5: Exercise it in the full-game integration sweep**

In `sim/tests/multi_config.rs`, change:

```rust
use sim::{run_batch, GreedyHighest, LowestLegal, MatchConfig, RandomLegal, Strategy};

fn baseline_strategies(player_count: u8) -> Vec<Arc<dyn Strategy>> {
    let pool: [Arc<dyn Strategy>; 3] = [
        Arc::new(LowestLegal),
        Arc::new(RandomLegal),
        Arc::new(GreedyHighest),
    ];
```

to:

```rust
use sim::{run_batch, GreedyHighest, HoldBackPairs, LowestLegal, MatchConfig, RandomLegal, Strategy};

fn baseline_strategies(player_count: u8) -> Vec<Arc<dyn Strategy>> {
    let pool: [Arc<dyn Strategy>; 4] = [
        Arc::new(LowestLegal),
        Arc::new(RandomLegal),
        Arc::new(GreedyHighest),
        Arc::new(HoldBackPairs),
    ];
```

- [ ] **Step 6: Run the full sweep**

Run: `cargo test -p sim`
Expected: PASS (the table-size/deck-variant sweep now also exercises
`HoldBackPairs` across every supported table size and deck variant,
without panicking).

- [ ] **Step 7: Format and commit**

Run: `cargo fmt --check -p sim` (fix if needed, then re-run), then:

```bash
git add sim/src/strategies/hold_back_pairs.rs sim/src/strategies/mod.rs sim/src/lib.rs sim/tests/multi_config.rs
git commit -m "sim: add HoldBackPairs, a strategy that avoids splitting a same-rank reserve"
```

---

### Task 4: Surface the new statistics and strategy in `cli`

**Files:**
- Modify: `cli/src/args.rs`
- Modify: `cli/src/summary.rs`
- Modify: `cli/tests/smoke.rs`

**Interfaces:**
- Consumes: `sim::HoldBackPairs` (Task 3); `sim::Statistics`'s four
  fields (Task 2); `sim::RoleRetention`.
- Produces: `--strategy hold-back-pairs` as a valid CLI value; the
  extended stdout summary text Task 5's `README.md` describes.

- [ ] **Step 1: Register the new strategy in `cli/src/args.rs`**

Change the `StrategyArg` enum:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum StrategyArg {
    LowestLegal,
    GreedyHighest,
    RandomLegal,
    HoldBackPairs,
}
```

Change its `build` method:

```rust
impl StrategyArg {
    /// Builds the concrete strategy instance this variant names.
    #[must_use]
    pub fn build(self) -> Arc<dyn sim::Strategy> {
        match self {
            StrategyArg::LowestLegal => Arc::new(sim::LowestLegal),
            StrategyArg::GreedyHighest => Arc::new(sim::GreedyHighest),
            StrategyArg::RandomLegal => Arc::new(sim::RandomLegal),
            StrategyArg::HoldBackPairs => Arc::new(sim::HoldBackPairs),
        }
    }
}
```

Update the existing test:

```rust
    #[test]
    fn strategy_arg_builds_matching_strategy_names() {
        assert_eq!(StrategyArg::LowestLegal.build().name(), "LowestLegal");
        assert_eq!(StrategyArg::GreedyHighest.build().name(), "GreedyHighest");
        assert_eq!(StrategyArg::RandomLegal.build().name(), "RandomLegal");
        assert_eq!(StrategyArg::HoldBackPairs.build().name(), "HoldBackPairs");
    }
```

Also update the `Args` doc comment on `strategies` (currently ends "...
Must supply exactly `player_count`.") to add a trailing sentence naming
the valid values, so `--help` output and the doc comment agree with
`docs/BUILDING.md` (Task 5):

```rust
    /// One per seat, setting the seating for the batch's first match,
    /// in seat order (e.g. `--strategy lowest-legal --strategy
    /// greedy-highest`). `sim::run_batch` then rotates this seating by
    /// one position for each subsequent match, so which physical seat
    /// each strategy occupies varies across the batch (this cancels
    /// `deal`'s documented uneven-remainder seat bias — see
    /// docs/RULES.md, "Players & Deck"). Must supply exactly
    /// `player_count`. Valid values: `lowest-legal`, `greedy-highest`,
    /// `random-legal`, `hold-back-pairs`.
    #[arg(long = "strategy", value_enum, required = true)]
    pub strategies: Vec<StrategyArg>,
```

- [ ] **Step 2: Run the args tests**

Run: `cargo test -p cli args::`
Expected: PASS.

- [ ] **Step 3: Extend `cli/src/summary.rs`'s output**

Replace the whole file:

```rust
//! A human-readable summary table for stdout: run configuration, one row
//! per strategy of role counts, and diversification/role-sustainment/
//! luck-vs-skill signals broken out per strategy.

use std::fmt::Write as _;

use crate::args::Args;

/// Renders `statistics` (and the run configuration in `args`) as a
/// human-readable report. Pure and allocation-only, so it's directly
/// unit-testable without capturing stdout.
#[must_use]
pub fn render_summary(args: &Args, statistics: &sim::Statistics) -> String {
    let mut out = String::new();

    let deck_variant = format!("{:?}", args.deck_variant);
    let duplicate_rule = format!("{:?}", args.duplicate_rule);
    let _ = writeln!(
        out,
        "{} players, {deck_variant} deck, {duplicate_rule}, {} matches x {} rounds, seed {}, {} threads",
        args.player_count, args.matches, args.rounds, args.seed, args.threads
    );
    let _ = writeln!(out);

    let roles = engine::roles_for_player_count(args.player_count)
        .expect("clap already validated player_count is 3-6");

    let _ = write!(out, "{:<16}", "Strategy");
    for role in roles {
        let role_label = format!("{role:?}");
        let _ = write!(out, "{role_label:<16}");
    }
    let _ = writeln!(out);

    for (strategy_name, counts_by_role) in &statistics.role_counts_by_strategy {
        let _ = write!(out, "{strategy_name:<16}");
        for role in roles {
            let count = counts_by_role.get(role).copied().unwrap_or(0);
            let _ = write!(out, "{count:<16}");
        }
        let _ = writeln!(out);
    }
    let _ = writeln!(out);

    let pass_rate_percent = statistics.voluntary_pass_rate * 100.0;
    let _ = writeln!(out, "Voluntary pass rate: {pass_rate_percent:.2}%");

    let _ = writeln!(out);
    let _ = writeln!(out, "Voluntary pass rate by strategy:");
    for strategy_name in statistics.role_counts_by_strategy.keys() {
        match statistics.voluntary_pass_rate_by_strategy.get(strategy_name) {
            Some(rate) => {
                let _ = writeln!(out, "  {strategy_name:<16}{:.2}%", rate * 100.0);
            }
            None => {
                let _ = writeln!(out, "  {strategy_name:<16}(not enough data)");
            }
        }
    }

    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "President retention (stayed President the very next round):"
    );
    for strategy_name in statistics.role_counts_by_strategy.keys() {
        let retention = statistics
            .role_retention_by_strategy
            .get(strategy_name)
            .and_then(|by_role| by_role.get(&engine::Role::President));
        match retention {
            Some(r) if r.held > 0 => {
                let rate = f64::from(r.retained_next_round) / f64::from(r.held) * 100.0;
                let _ = writeln!(
                    out,
                    "  {strategy_name:<16}{rate:.2}% ({}/{})",
                    r.retained_next_round, r.held
                );
            }
            _ => {
                let _ = writeln!(out, "  {strategy_name:<16}(not enough data)");
            }
        }
    }

    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "First-round placement variance (lower = more skill-driven, higher = more luck-driven):"
    );
    for strategy_name in statistics.role_counts_by_strategy.keys() {
        let variance = statistics
            .first_round_placement_variance_by_strategy
            .get(strategy_name)
            .copied()
            .flatten();
        match variance {
            Some(v) => {
                let _ = writeln!(out, "  {strategy_name:<16}{v:.2}");
            }
            None => {
                let _ = writeln!(out, "  {strategy_name:<16}(not enough data)");
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{DeckVariantArg, DuplicateRuleArg, StrategyArg};
    use engine::Role;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn sample_args() -> Args {
        Args {
            player_count: 3,
            deck_variant: DeckVariantArg::Single,
            duplicate_rule: DuplicateRuleArg::FirstDealtWins,
            matches: 10,
            rounds: 2,
            strategies: vec![
                StrategyArg::LowestLegal,
                StrategyArg::GreedyHighest,
                StrategyArg::RandomLegal,
            ],
            threads: 0,
            seed: 42,
            output: PathBuf::from("results.json"),
        }
    }

    fn sample_statistics() -> sim::Statistics {
        let mut lowest_legal_counts = BTreeMap::new();
        lowest_legal_counts.insert(Role::President, 4);
        lowest_legal_counts.insert(Role::Arschloch, 6);
        let mut role_counts = BTreeMap::new();
        role_counts.insert("LowestLegal".to_string(), lowest_legal_counts);

        sim::Statistics {
            matches_played: 10,
            role_counts_by_strategy: role_counts,
            voluntary_pass_rate: 0.125,
            voluntary_pass_rate_by_strategy: BTreeMap::from([("LowestLegal".to_string(), 0.1)]),
            role_retention_by_strategy: BTreeMap::from([(
                "LowestLegal".to_string(),
                BTreeMap::from([(
                    Role::President,
                    sim::RoleRetention {
                        held: 4,
                        retained_next_round: 3,
                    },
                )]),
            )]),
            first_round_placement_variance_by_strategy: BTreeMap::from([(
                "LowestLegal".to_string(),
                None,
            )]),
        }
    }

    #[test]
    fn render_summary_includes_strategy_and_role_names() {
        let summary = render_summary(&sample_args(), &sample_statistics());
        assert!(summary.contains("LowestLegal"));
        assert!(summary.contains("President"));
        assert!(summary.contains("Dorftrottel"));
        assert!(summary.contains("Arschloch"));
    }

    #[test]
    fn render_summary_includes_formatted_pass_rate() {
        let summary = render_summary(&sample_args(), &sample_statistics());
        assert!(summary.contains("Voluntary pass rate: 12.50%"));
    }

    #[test]
    fn render_summary_includes_per_strategy_pass_rate() {
        let summary = render_summary(&sample_args(), &sample_statistics());
        assert!(summary.contains("10.00%"));
    }

    #[test]
    fn render_summary_includes_president_retention() {
        let summary = render_summary(&sample_args(), &sample_statistics());
        assert!(summary.contains("75.00% (3/4)"));
    }

    #[test]
    fn render_summary_shows_not_enough_data_for_missing_variance() {
        let summary = render_summary(&sample_args(), &sample_statistics());
        assert!(summary.contains("(not enough data)"));
    }
}
```

- [ ] **Step 4: Run the summary tests**

Run: `cargo test -p cli summary::`
Expected: PASS.

- [ ] **Step 5: Extend the end-to-end smoke test**

In `cli/tests/smoke.rs`, change `happy_path_run_produces_valid_json_and_summary`
to exercise all four strategies (bump to 4 players) and check the new
JSON keys:

```rust
#[test]
fn happy_path_run_produces_valid_json_and_summary() {
    let output_path = std::env::temp_dir().join(format!(
        "arschloch-cli-smoke-happy-{}.json",
        std::process::id()
    ));

    let output = Command::new(env!("CARGO_BIN_EXE_cli"))
        .args([
            "--player-count",
            "4",
            "--matches",
            "5",
            "--rounds",
            "2",
            "--strategy",
            "lowest-legal",
            "--strategy",
            "greedy-highest",
            "--strategy",
            "random-legal",
            "--strategy",
            "hold-back-pairs",
            "--seed",
            "1",
            "--output",
        ])
        .arg(&output_path)
        .output()
        .expect("failed to run cli binary");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("LowestLegal"));
    assert!(stdout.contains("GreedyHighest"));
    assert!(stdout.contains("RandomLegal"));
    assert!(stdout.contains("HoldBackPairs"));
    assert!(stdout.contains("Voluntary pass rate"));
    assert!(stdout.contains("President retention"));
    assert!(stdout.contains("First-round placement variance"));

    let contents = std::fs::read_to_string(&output_path).expect("output file should exist");
    let value: serde_json::Value =
        serde_json::from_str(&contents).expect("output should be valid JSON");
    assert!(value.get("matches").is_some());
    let statistics = value.get("statistics").expect("statistics key present");
    assert!(statistics.get("voluntary_pass_rate_by_strategy").is_some());
    assert!(statistics.get("role_retention_by_strategy").is_some());
    assert!(statistics
        .get("first_round_placement_variance_by_strategy")
        .is_some());

    std::fs::remove_file(&output_path).ok();
}
```

(`mismatched_strategy_count_fails` is unchanged.)

- [ ] **Step 6: Run the full workspace gate**

Run: `cargo fmt --check`
Expected: clean (fix and re-run if not).

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean.

Run: `cargo test --workspace`
Expected: PASS, every crate.

- [ ] **Step 7: Commit**

```bash
git add cli/src/args.rs cli/src/summary.rs cli/tests/smoke.rs
git commit -m "cli: add hold-back-pairs strategy and print the new Phase 4 statistics"
```

---

### Task 5: Documentation — top-level `README.md` and doc corrections

**Files:**
- Create: `README.md` (repo root)
- Modify: `docs/BUILDING.md`
- Modify: `docs/ARCHITECTURE.md`

**Interfaces:**
- Consumes: the finished CLI surface and `Statistics` shape from Tasks
  1-4 (this task only documents already-implemented behavior — no code
  changes).
- Produces: nothing consumed by other tasks (this is the last task).

- [ ] **Step 1: Write `README.md`**

```markdown
# Arschloch Simulator

A multi-threaded simulator for *Arschloch* ("Asshole"), the German
shedding-type card game — play out thousands of matches between simple
AI strategies and see who tends to end up President, and why.

## Quickstart

```bash
cargo build --release
cargo run -p cli -- \
  --player-count 4 \
  --matches 1000 \
  --rounds 10 \
  --strategy lowest-legal \
  --strategy greedy-highest \
  --strategy random-legal \
  --strategy hold-back-pairs \
  --seed 1 \
  --output results.json
```

This simulates 1000 independent matches (10 rounds each, with role
carry-over between rounds within a match) at a 4-player table, one seat
per named strategy, and produces two things:

- `results.json` — every match's full result plus aggregated statistics.
- A human-readable summary table printed to stdout.

See `docs/BUILDING.md` for the full flag reference (deck variant,
duplicate-card rule, thread count, and cross-compiling a Windows `.exe`).

## What you get

### The strategies

- `lowest-legal` — always plays the smallest, lowest-ranked legal combo;
  the most conservative baseline.
- `greedy-highest` — always plays the largest, highest-ranked legal
  combo; the most aggressive baseline.
- `random-legal` — picks uniformly among every legal move, including
  passing when it doesn't have to.
- `hold-back-pairs` — plays like `lowest-legal`, except while following:
  if every beating play would break up cards of a rank it holds more of
  than this play needs, it passes instead, to keep that reserve intact
  for a later lead.

### The stdout summary

- **Role counts by strategy** — across every round simulated, how often
  each strategy ended up in each role (President down to Arschloch). This
  is the headline number: which strategy tends to win?
- **Voluntary pass rate (pooled, then per strategy)** — how often a seat
  passed even though it had a legal play available. A strategy with a
  much higher rate than the always-play baselines is deliberately trading
  away an immediate play for something else (see `hold-back-pairs`
  above); a rate near zero means "always play if it can."
- **President retention** — of the times a strategy became President,
  what fraction stayed President the very next round? This only means
  something for matches with more than one round, since roles only carry
  over round-to-round within a single match. A rate far above chance
  suggests the role — and the winner's-advantage card exchange that comes
  with it — is genuinely self-reinforcing for that strategy; a rate close
  to how often it becomes President at all suggests it's closer to
  random.
- **First-round placement variance** — how much a strategy's very first
  round's outcome (before any role carry-over) swings between matches
  that seated it identically (same seat, same opponents) but shuffled the
  deck differently. Low variance means the shuffle barely matters for
  this strategy (its play style dominates the outcome — a "skill"
  signal); high variance means the shuffle matters a lot (a "luck"
  signal). This needs a big enough batch to repeat a seating at least
  twice — see `docs/BUILDING.md` for the exact threshold — otherwise it
  prints "(not enough data)".

### The JSON file

`results.json` has two top-level keys: `matches` (one entry per simulated
match) and `statistics` (the aggregated numbers behind the stdout table).
Both are `sim`'s own Rust types, serialized as-is — see the doc comments
on `sim::MatchResult` and `sim::Statistics` in `sim/src/` for the exact
field names and meanings.

## Learn more

| Doc | What's in it |
|---|---|
| [`docs/RULES.md`](docs/RULES.md) | The authoritative game rules this simulator implements, including house-rule decisions. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | How the `engine`/`sim`/`cli`/`web` crates fit together. |
| [`docs/BUILDING.md`](docs/BUILDING.md) | Full build, test, and CLI flag reference; cross-compiling a Windows `.exe`. |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | What's built, what's planned, what's explicitly out of scope. |
| [`docs/CODING_GUIDELINES.md`](docs/CODING_GUIDELINES.md) | Conventions for contributing. |
```

- [ ] **Step 2: Update `docs/BUILDING.md`**

In the "Running the simulator" section, change the `--strategy` bullet's
last sentence from:

```markdown
  Valid values: `lowest-legal`, `greedy-highest`, `random-legal`.
```

to:

```markdown
  Valid values: `lowest-legal`, `greedy-highest`, `random-legal`,
  `hold-back-pairs`.
```

Immediately after the bullet that starts "Two things happen:", add a new
bullet:

```markdown
- The aggregated statistics include a first-round placement variance per
  strategy (`docs/ROADMAP.md`, Phase 4, "Luck-vs-skill signal"), which
  reports `null` until the batch repeats at least one exact seating (the
  same strategies in the same seats) twice. `sim::run_batch` cycles
  seatings every `--strategy`-flag-count matches, so pick `--matches` at
  least `2 * (number of --strategy flags)` if you want this number
  populated.
```

- [ ] **Step 3: Update `docs/ARCHITECTURE.md`**

In the `sim` section, change:

```markdown
- `Strategy` trait: `choose_play(&self, legal_moves, duplicate_rule,
  rng) -> Move`. `legal_moves` (from `engine::Round::legal_moves`) already
  encodes every card a candidate move would use, so no separate `hand`
  parameter is needed; `rng` is threaded through explicitly per call
  (rather than owned by the strategy) so a single `Arc<dyn Strategy>` can
  be shared read-only across parallel matches while staying fully
  deterministic per match seed. Three baseline implementations this
  phase: `LowestLegal`, `RandomLegal`, `GreedyHighest`.
  `choose_exchange_cards` doesn't exist yet — `exchange()`'s naive
  top/bottom-N tie-break (Phase 1) is applied directly by the match
  runner; strategy-aware exchange selection is Phase 5's job
  ("smart exchange"), not introduced early as speculative generality.
- A match runner that drives `engine`'s state machine to completion using
  each seat's `Strategy`.
- Multi-threading via `rayon`: independent matches have no shared mutable
  state, so a batch of simulated matches is a `par_iter` map over match
  seeds/configs, with no locks needed.
- `MatchResult` (per-round role history, strategy tag per seat, trick/pass
  counts) and `Statistics` (aggregated across a batch, Phase 2): role
  counts by strategy name, and a pooled voluntary-pass rate (passes
  submitted while a beating play was also legal, summed across every
  seat and match — not yet broken out per strategy). Outcome variance
  (a proxy for luck vs. skill) and a per-strategy diversification signal
  are Phase 4's job, not built yet. Serialized with `serde`/`serde_json`
  so a later web phase can consume the same output format.
```

to:

```markdown
- `Strategy` trait: `choose_play(&self, legal_moves, duplicate_rule,
  rng) -> Move`. `legal_moves` (from `engine::Round::legal_moves`) already
  encodes every card a candidate move would use, so no separate `hand`
  parameter is needed; `rng` is threaded through explicitly per call
  (rather than owned by the strategy) so a single `Arc<dyn Strategy>` can
  be shared read-only across parallel matches while staying fully
  deterministic per match seed. Three baseline implementations from
  Phase 2: `LowestLegal`, `RandomLegal`, `GreedyHighest`. A fourth,
  `HoldBackPairs` (Phase 4), plays identically while leading but
  deliberately passes rather than split up a same-rank reserve while
  following, as a diversification comparison point against the three
  always-play baselines. `choose_exchange_cards` doesn't exist yet —
  `exchange()`'s naive top/bottom-N tie-break (Phase 1) is applied
  directly by the match runner; strategy-aware exchange selection is
  Phase 5's job ("smart exchange"), not introduced early as speculative
  generality.
- A match runner that drives `engine`'s state machine to completion using
  each seat's `Strategy`.
- Multi-threading via `rayon`: independent matches have no shared mutable
  state, so a batch of simulated matches is a `par_iter` map over match
  seeds/configs, with no locks needed.
- `MatchResult` (per-round role history, strategy tag per seat, per-seat
  pass counts) and `Statistics` (aggregated across a batch): role counts
  by strategy name, a pooled and per-strategy voluntary-pass rate (passes
  submitted while a beating play was also legal — Phase 4, "strategy
  diversification"), per-strategy role-sustainment counts (how often a
  role is still held the very next round — Phase 4, "role-sustainment
  tracking"), and a per-strategy first-round placement variance across
  matches with an identical seating but a different shuffle (Phase 4,
  "luck-vs-skill signal"). Serialized with `serde`/`serde_json` so a
  later web phase can consume the same output format.
```

- [ ] **Step 4: Review the docs as a first-time reader**

Read `README.md` top to bottom as if seeing this repo for the first
time: confirm the Quickstart command matches the actual `cli` flags
(cross-check against `cli/src/args.rs`), and confirm every claim about
what a statistic means matches Task 2's doc comments.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/BUILDING.md docs/ARCHITECTURE.md
git commit -m "docs: add top-level README, document Phase 4 statistics and hold-back-pairs"
```
