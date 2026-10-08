# Phase 10b — `NeatStrategy` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make an evolved NEAT genome a first-class player: `NeatStrategy`
scores every legal move with a network and plays the best one, genomes
are saved in files that remember the feature set they were trained on,
and `--strategy neat:PATH` runs a trained genome in the normal simulator
next to every hand-written strategy.

**Architecture:** `sim` gains a module `strategies/neat_player/` with
three files: `features.rs` (20 features per candidate move, computed
from `TurnContext`, with the per-turn parts computed once), 
`genome_file.rs` (`GenomeFile`: genome + feature names + format version,
refusing files from a different build) and `mod.rs` (`NeatStrategy`).
`sim` depends on `neat` (new one-way edge; `neat` never depends on
`sim`). The CLI parses `neat:PATH` into `StrategyArg::Neat`, loading and
validating the file at argument-parse time. The existing `Strategy`
trait, `TurnContext` and every hand-written strategy are unchanged.

**Tech Stack:** Rust workspace; `neat` (Phase 10a), `serde_json 1.0.151`
with the `float_roundtrip` feature (a feature flag of a crate already in
the lockfile, no new crate).

**Spec:** `docs/superpowers/specs/2026-10-08-neat-engine-design.md`,
sections 3, 4 (move scoring, features, exchange) and 10 (testing).
Deviation recorded in Task 7: the spec's "role" feature is deferred.

## How this plan was prepared

The code below was written into a scratch copy of the workspace first and
passed `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D
warnings` and `cargo test --workspace` (325 tests) before being turned
into steps. It was also run as a user would: a saved genome file through
the real `cli` binary, and a speed measurement (a table of four
`NeatStrategy` seats runs at about 1.9x the cost of four `LowestLegal`
seats). The steps are applied by one executor from the same data that
renders this document, so the plan and what runs cannot drift apart.

## Global Constraints

- Dependency direction: `cli -> sim -> engine` and `sim -> neat`. `neat` must not depend on `sim`, `engine` or `cli`. `cli` gets `neat` only as a dev-dependency (tests build genome files).
- No new crates. `serde_json` gains `features = ["float_roundtrip"]` in `neat`, `sim` and `cli`.
- `Strategy`, `TurnContext`, `OpponentHand`, `run_match`, `run_batch` and all hand-written strategies are **not modified**. `docs/baselines/pre-neat/` stays valid.
- `FEATURE_NAMES` (20 names, this order) is a contract with saved genomes: renaming, adding, removing or reordering any feature invalidates every trained genome and must change the names list so old files are refused.
- `NeatStrategy` is stateless (immutable compiled `Network` plus a name), `Send + Sync`, and never draws from `rng`. Ties between equal scores keep the first-listed legal move.
- Every feature is finite for every reachable state (no division by zero when no opponent is active or nothing is unseen) and roughly within `[0, 1]`.
- Genome files must be loaded with validation: wrong format version, wrong feature names, wrong input count or an invalid genome is an error (`GenomeFileError`), never a panic and never silently accepted.
- The exchange step uses `strategies::take_highest_naive` (spec: v1 reuses it).
- Between Task 2 and Task 4 the non-test build reports `dead_code` warnings for items that later tasks start using; those are expected and disappear by Task 4. Any other warning is a defect.
- Per-task verification is `cargo test -p <crate>`; the full gate (`cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`) runs in Task 7.

## Review Focus

Failure modes the spec implies but a straightforward implementation tends to miss, each pinned by a named test:

1. **A stale or malformed genome file** (feature set changed, wrong version, wrong input count, garbage, missing file) must fail loudly *before* any match runs, with a message naming the problem. Tests: `genome_file::tests::a_file_trained_on_other_features_is_refused_with_both_lists_named`, `an_unknown_format_version_is_refused`, `garbage_and_invalid_genomes_are_parse_errors_not_panics`, `a_missing_file_is_an_io_error_naming_the_path`; CLI `a_stale_genome_file_is_refused_at_parse_time`, `a_missing_genome_file_fails_with_a_clear_message`, `a_garbage_genome_file_fails_without_a_panic`.
2. **Saved weights must be bit-exact** so a champion plays exactly like the genome that was evaluated. Test: `neat::genome::tests::json_round_trip_is_bit_exact_for_arbitrary_weights` (Task 1) and `genome_file::tests::save_then_load_round_trips`.
3. **Every table size, both decks, both duplicate rules, evolved topologies with hidden nodes, mixed with other strategies:** only legal moves, no panics. Test: `neat_random_genomes::evolved_networks_always_play_legal_moves_everywhere`. (Double-deck duplicates tie on `top_strength`; ties keep the first-listed move, still legal.)
4. **Ties and score saturation:** exactly equal scores must resolve deterministically, and hand-wired genomes must stay inside tanh's non-saturated range. Tests: `tests::exact_ties_keep_the_first_listed_move`; the weights documented on `common::lowest_legal_genome` make `neat_equivalence` pass.
5. **One strategy instance shared by many parallel matches, deterministic results.** Tests: `tests::the_strategy_can_be_shared_across_threads`, `neat_random_genomes::batches_with_networks_are_deterministic_and_thread_safe`.

Also pinned: `a_lowest_legal_network_plays_identically_to_lowest_legal` (full matches at 3-6 players, both duplicate rules, single deck) with a sensitivity check in Task 5 that deliberately breaks a feature and watches it fail.

---

## File Structure

```
neat/Cargo.toml, neat/src/genome.rs      # Task 1: bit-exact JSON
sim/Cargo.toml                           # neat dependency, float_roundtrip
sim/src/strategies/mod.rs                # module + re-exports
sim/src/lib.rs                           # re-exports
sim/src/strategies/neat_player/mod.rs           # NeatStrategy
sim/src/strategies/neat_player/features.rs      # FEATURE_NAMES, TurnSummary
sim/src/strategies/neat_player/genome_file.rs   # GenomeFile, GenomeFileError
sim/tests/common/mod.rs                  # linear_genome, lowest_legal_genome
sim/tests/neat_equivalence.rs            # equals LowestLegal, in full matches
sim/tests/neat_random_genomes.rs         # evolved networks play legal moves everywhere
cli/Cargo.toml, cli/src/args.rs, cli/tests/smoke.rs   # neat:PATH
docs/ROADMAP.md, docs/BUILDING.md, docs/ARCHITECTURE.md, docs/superpowers/specs/2026-10-08-neat-engine-design.md
```

Not in this phase: fitness evaluation, the `train` subcommand, checkpointing, the dashboard, evolving the exchange step, the role feature.

---

### Task 1: Bit-exact JSON for genome weights (fixes a 10a gap)

**Files:**
- Modify: `neat/src/genome.rs` (one test), `neat/Cargo.toml`

**Why:** `serde_json` parses floats on a fast path that can be off by one ULP unless its `float_roundtrip` feature is on. 10a's single-genome round-trip test did not notice; a many-genome test does.

**Interfaces:**
- Consumes: `neat::Genome` JSON, `neat::mutation::mutate_weights`, `genome::test_support`.
- Produces: Genome JSON that round-trips every `f64` weight exactly (a saved champion plays exactly like the evaluated one). Later tasks rely on this for genome files.

- [ ] **Step 1: Edit**

In `neat/src/genome.rs` (Add the failing test above `json_round_trip_is_lossless`), replace:

```rust
    #[test]
    fn json_round_trip_is_lossless() {
```

with:

```rust
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
```

- [ ] **Step 2: Run (expect failure)**

Run: `cargo test -p neat bit_exact`

Expected: FAIL: `assertion `left == right` failed: seed 0` (weights differ in the last bits after the round trip).

- [ ] **Step 3: Edit**

In `neat/Cargo.toml` (Turn on exact float parsing), replace:

```toml
serde_json = "1.0.151"
```

with:

```toml
serde_json = { version = "1.0.151", features = ["float_roundtrip"] }
```

- [ ] **Step 4: Run (expect success)**

Run: `cargo test -p neat`

Expected: PASS (64 unit tests + 2 XOR).

- [ ] **Step 5: Commit**

```bash
git add neat
git commit -F - <<'EOF'
neat: make genome JSON bit-exact with serde_json float_roundtrip (Phase 10b)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
```


### Task 2: Candidate features

**Files:**
- Modify: `sim/Cargo.toml`, `sim/src/strategies/mod.rs`
- Create: `sim/src/strategies/neat_player/mod.rs`, `sim/src/strategies/neat_player/features.rs`

**Interfaces:**
- Consumes: `sim::strategy::{TurnContext, OpponentHand}`, `sim::hand_reading::PassCeilings` (`cannot_beat`, `read_pass_ceilings` in tests), `engine::{Card, Combo, Move, Rank, rank_groups, DuplicateRule}`.
- Produces: `FEATURE_NAMES: [&str; 20]`, `FEATURE_COUNT: usize`, `TurnSummary::new(&TurnContext, DuplicateRule) -> TurnSummary`, `TurnSummary::features(&self, &Move) -> [f64; FEATURE_COUNT]`. The names and their order are a contract with saved genome files.

- [ ] **Step 1: Edit**

In `sim/Cargo.toml` (`sim` now depends on `neat` (new one-way edge `sim -> neat`; `neat` never depends on `sim`)), replace:

```toml
engine = { path = "../engine" }
```

with:

```toml
engine = { path = "../engine" }
neat = { path = "../neat" }
```

- [ ] **Step 2: Edit**

In `sim/Cargo.toml`, replace:

```toml
serde_json = "1.0.151"
```

with:

```toml
serde_json = { version = "1.0.151", features = ["float_roundtrip"] }
```

- [ ] **Step 3: Create file**

Create `sim/src/strategies/neat_player/mod.rs` (Create the module root (it grows in Tasks 3 and 4)):

```rust
//! `NeatStrategy`: plays by scoring every legal move with an evolved
//! neural network and choosing the highest score. See
//! docs/superpowers/specs/2026-10-08-neat-engine-design.md, section 4.

mod features;

pub use features::{TurnSummary, FEATURE_COUNT, FEATURE_NAMES};
```

- [ ] **Step 4: Edit**

In `sim/src/strategies/mod.rs`, replace:

```rust
mod lowest_legal;
```

with:

```rust
mod lowest_legal;
mod neat_player;
```

- [ ] **Step 5: Create file**

Create `sim/src/strategies/neat_player/features.rs` (Write the tests first: the file contains only its test module):

```rust
#[cfg(test)]
mod tests {
    use engine::{Combo, Rank, Suit};

    use super::*;
    use crate::hand_reading::{read_pass_ceilings, PassCeilings};

    const RULE: DuplicateRule = DuplicateRule::FirstDealtWins;

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit, 0)
    }

    fn play(cards: &[Card]) -> Move {
        Move::Play(Combo::new(cards.to_vec()).unwrap())
    }

    fn opponent(
        seat: u8,
        hand_size: usize,
        active: bool,
        pass_ceilings: PassCeilings,
    ) -> OpponentHand {
        OpponentHand {
            seat,
            hand_size,
            active,
            pass_ceilings,
        }
    }

    /// Hand: 3d 3h 9s Ac. Unseen: 4d 9c Kh Kd Ks.
    struct Scenario {
        hand: Vec<Card>,
        unseen: Vec<Card>,
        opponents: Vec<OpponentHand>,
    }

    fn scenario() -> Scenario {
        Scenario {
            hand: vec![
                card(Rank::Three, Suit::Diamonds),
                card(Rank::Three, Suit::Hearts),
                card(Rank::Nine, Suit::Spades),
                card(Rank::Ace, Suit::Clubs),
            ],
            unseen: vec![
                card(Rank::Four, Suit::Diamonds),
                card(Rank::Nine, Suit::Clubs),
                card(Rank::King, Suit::Hearts),
                card(Rank::King, Suit::Diamonds),
                card(Rank::King, Suit::Spades),
            ],
            opponents: vec![
                opponent(1, 3, true, PassCeilings::default()),
                opponent(2, 2, true, PassCeilings::default()),
                opponent(3, 0, false, PassCeilings::default()),
            ],
        }
    }

    fn context<'a>(s: &'a Scenario, table: Option<&'a Combo>) -> TurnContext<'a> {
        TurnContext {
            seat: 0,
            hand: &s.hand,
            opponents: s.opponents.clone(),
            unseen_cards: s.unseen.clone(),
            own_pass_ceilings: PassCeilings::default(),
            current_combo: table,
        }
    }

    fn approx(actual: f64, expected: f64, name: &str) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "{name}: {actual} vs {expected}"
        );
    }

    #[test]
    fn names_are_unique_and_match_the_count() {
        assert_eq!(FEATURE_NAMES.len(), FEATURE_COUNT);
        let mut sorted = FEATURE_NAMES.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), FEATURE_COUNT);
    }

    #[test]
    fn state_features_describe_the_table_and_opponents() {
        let s = scenario();
        let ctx = context(&s, None);
        let f = TurnSummary::new(&ctx, RULE).features(&play(&[card(Rank::Nine, Suit::Spades)]));
        approx(f[8], 1.0, "is_leading");
        approx(f[9], 0.0, "table_combo_size");
        approx(f[10], 4.0 / 13.0, "hand_size");
        approx(f[11], 2.0 / 4.0, "grouped_fraction (3d,3h)");
        approx(f[12], 2.0 / 4.0, "singleton_fraction (9s, Ac)");
        approx(f[13], 2.0 / 5.0, "active_opponents (inactive one ignored)");
        approx(f[14], 2.0 / 13.0, "min_opponent_hand");
        approx(f[15], 2.5 / 13.0, "mean_opponent_hand");
        approx(f[16], 1.0, "opponent_close (a seat holds 2 cards)");
    }

    #[test]
    fn following_a_combo_sets_the_table_features() {
        let s = scenario();
        let on_table = Combo::new(vec![card(Rank::Five, Suit::Hearts)]).unwrap();
        let ctx = context(&s, Some(&on_table));
        let f = TurnSummary::new(&ctx, RULE).features(&Move::Pass);
        approx(f[8], 0.0, "is_leading");
        approx(f[9], 1.0 / 8.0, "table_combo_size");
    }

    #[test]
    fn passing_has_only_state_features_and_the_pass_flag() {
        let s = scenario();
        let ctx = context(&s, None);
        let summary = TurnSummary::new(&ctx, RULE);
        let f = summary.features(&Move::Pass);
        approx(f[0], 1.0, "is_pass");
        approx(f[7], 1.0, "hand_left_after (hand unchanged)");
        for index in [1, 2, 3, 4, 5, 6, 17, 18, 19] {
            approx(f[index], 0.0, FEATURE_NAMES[index]);
        }
    }

    #[test]
    fn playing_the_hands_top_single_sets_the_play_features() {
        let s = scenario();
        let ctx = context(&s, None);
        let f = TurnSummary::new(&ctx, RULE).features(&play(&[card(Rank::Ace, Suit::Clubs)]));
        approx(f[0], 0.0, "is_pass");
        approx(f[1], 1.0 / 8.0, "combo_size");
        approx(f[2], 1.0, "top_strength (Ac is the strongest card)");
        approx(f[3], 3.0 / 4.0, "hand_below_top");
        approx(f[4], 1.0, "uses_top_group");
        approx(f[5], 0.0, "splits_group");
        approx(f[6], 0.0, "empties_hand");
        approx(f[7], 3.0 / 4.0, "hand_left_after");
        approx(f[17], 0.0, "unseen_outranking (nothing beats Ac)");
        approx(f[18], 0.0, "unseen_beaters");
    }

    #[test]
    fn splitting_a_pair_and_emptying_the_hand_are_flagged() {
        let s = scenario();
        let ctx = context(&s, None);
        let summary = TurnSummary::new(&ctx, RULE);
        let split = summary.features(&play(&[card(Rank::Three, Suit::Hearts)]));
        approx(split[5], 1.0, "splits_group (a single from the 3s pair)");
        let whole = summary.features(&play(&[
            card(Rank::Three, Suit::Diamonds),
            card(Rank::Three, Suit::Hearts),
        ]));
        approx(whole[5], 0.0, "splits_group (the whole pair)");
        approx(whole[1], 2.0 / 8.0, "combo_size");

        let last_card = Scenario {
            hand: vec![card(Rank::Two, Suit::Clubs)],
            unseen: Vec::new(),
            opponents: vec![opponent(1, 5, true, PassCeilings::default())],
        };
        let ctx = context(&last_card, None);
        let f = TurnSummary::new(&ctx, RULE).features(&play(&[card(Rank::Two, Suit::Clubs)]));
        approx(f[6], 1.0, "empties_hand");
        approx(f[7], 0.0, "hand_left_after");
        approx(f[17], 0.0, "unseen_outranking with nothing unseen");
    }

    #[test]
    fn unseen_features_count_cards_and_ranks_that_beat_the_play() {
        let s = scenario();
        let ctx = context(&s, None);
        let summary = TurnSummary::new(&ctx, RULE);
        // Playing 3d: every unseen card (5) outranks it; the four
        // distinct unseen ranks (4, 9, K) -> 3 ranks can beat a single.
        let low = summary.features(&play(&[card(Rank::Three, Suit::Diamonds)]));
        approx(low[17], 1.0, "unseen_outranking");
        approx(low[18], 3.0 / 13.0, "unseen_beaters (4, 9, K)");
        // Playing 9s: 9c (higher suit) and the three kings beat it, 4d does not.
        let nine = summary.features(&play(&[card(Rank::Nine, Suit::Spades)]));
        approx(nine[17], 4.0 / 5.0, "unseen_outranking");
        approx(nine[18], 2.0 / 13.0, "unseen_beaters (9c, K)");
        // A pair can only be beaten by a rank with 2+ unseen cards: just kings.
        let pair = summary.features(&play(&[
            card(Rank::Three, Suit::Diamonds),
            card(Rank::Three, Suit::Hearts),
        ]));
        approx(pair[18], 1.0 / 13.0, "unseen_beaters for a pair");
    }

    #[test]
    fn opponents_locked_out_counts_seats_whose_passes_prove_they_cannot_beat() {
        let mut s = scenario();
        // Seat 1 passed against the King of Hearts: it cannot beat any
        // single at or above that card. Seat 2 has said nothing.
        let king = Combo::new(vec![card(Rank::King, Suit::Hearts)]).unwrap();
        let ceilings = read_pass_ceilings(4, &[], &[(1, king, 0)], RULE);
        s.opponents[0].pass_ceilings = ceilings[1];
        let ctx = context(&s, None);
        let summary = TurnSummary::new(&ctx, RULE);
        let ace = summary.features(&play(&[card(Rank::Ace, Suit::Clubs)]));
        approx(
            ace[19],
            1.0 / 2.0,
            "one of two active opponents is locked out",
        );
        let nine = summary.features(&play(&[card(Rank::Nine, Suit::Spades)]));
        approx(
            nine[19],
            0.0,
            "9s is below the passed King, so nobody is locked out",
        );
    }

    #[test]
    fn no_active_opponents_gives_zero_not_nan() {
        let mut s = scenario();
        s.opponents.clear();
        let ctx = context(&s, None);
        let f = TurnSummary::new(&ctx, RULE).features(&play(&[card(Rank::Ace, Suit::Clubs)]));
        assert!(f.iter().all(|v| v.is_finite()));
        approx(f[13], 0.0, "active_opponents");
        approx(f[14], 0.0, "min_opponent_hand");
        approx(f[19], 0.0, "opponents_locked_out");
    }

    #[test]
    fn strength_follows_the_house_card_order() {
        let low = card(Rank::Two, Suit::Diamonds);
        let same_rank_higher_suit = card(Rank::Two, Suit::Clubs);
        let next_rank = card(Rank::Three, Suit::Diamonds);
        assert!(strength(low) < strength(same_rank_higher_suit));
        assert!(strength(same_rank_higher_suit) < strength(next_rank));
        approx(strength(low), 0.0, "weakest");
        approx(strength(card(Rank::Ace, Suit::Clubs)), 1.0, "strongest");
    }
}
```

- [ ] **Step 6: Run (expect failure)**

Run: `cargo test -p sim neat_player`

Expected: FAIL (compile errors: `TurnSummary`, `FEATURE_NAMES` etc. are not defined).

- [ ] **Step 7: Implement**

Insert at the very top of `sim/src/strategies/neat_player/features.rs`, above the `#[cfg(test)]` line (Implement, above the `#[cfg(test)]` line):

```rust
//! The network's view of a decision: one fixed-size feature vector per
//! candidate move (see docs/superpowers/specs/2026-10-08-neat-engine-design.md,
//! section 4).
//!
//! Every value is a deterministic function of `(candidate, TurnContext,
//! duplicate_rule)` and lies roughly in `[0, 1]` (hand sizes can exceed 1
//! in the double deck). The set and order below are a *contract* with
//! saved genomes: changing either invalidates every trained genome, so
//! `FEATURE_NAMES` is stored in each genome file and checked on load.

use std::cmp::Ordering;

use engine::{rank_groups, Card, DuplicateRule, Move, Rank};

use crate::strategy::{OpponentHand, TurnContext};

pub const FEATURE_NAMES: [&str; 20] = [
    "is_pass",
    "combo_size",
    "top_strength",
    "hand_below_top",
    "uses_top_group",
    "splits_group",
    "empties_hand",
    "hand_left_after",
    "is_leading",
    "table_combo_size",
    "hand_size",
    "grouped_fraction",
    "singleton_fraction",
    "active_opponents",
    "min_opponent_hand",
    "mean_opponent_hand",
    "opponent_close",
    "unseen_outranking",
    "unseen_beaters",
    "opponents_locked_out",
];

pub const FEATURE_COUNT: usize = FEATURE_NAMES.len();

/// Hand sizes are scaled by a typical single-deck hand (52 / 4).
const HAND_SCALE: f64 = 13.0;
/// The largest combo any seat can field (double deck, 8 of a rank).
const COMBO_SCALE: f64 = 8.0;
const RANK_COUNT: f64 = 13.0;
/// An opponent this close to going out makes the round urgent.
const CLOSE_HAND: usize = 2;

#[allow(clippy::cast_precision_loss)] // counts here are tiny (< 2^52)
fn count(n: usize) -> f64 {
    n as f64
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    count(numerator) / count(denominator.max(1))
}

fn flag(condition: bool) -> f64 {
    f64::from(u8::from(condition))
}

/// Position of `card` in the 52-card order (rank first, then suit), in
/// `[0, 1]`. Ignores the double-deck duplicate tiebreak, which has no
/// strength meaning of its own.
fn strength(card: Card) -> f64 {
    f64::from(card.rank as u8 * 4 + card.suit as u8) / 51.0
}

/// Everything about the turn that does not depend on the candidate move,
/// computed once and reused for every candidate.
pub struct TurnSummary<'a> {
    duplicate_rule: DuplicateRule,
    hand: &'a [Card],
    unseen: &'a [Card],
    unseen_groups: Vec<Vec<Card>>,
    hand_group_sizes: Vec<(Rank, usize)>,
    max_hand_rank: Option<Rank>,
    active_opponents: Vec<&'a OpponentHand>,
    state: [f64; FEATURE_COUNT],
}

impl<'a> TurnSummary<'a> {
    #[must_use]
    pub fn new(context: &'a TurnContext<'a>, duplicate_rule: DuplicateRule) -> Self {
        let hand = context.hand;
        let groups = rank_groups(hand);
        let hand_group_sizes: Vec<(Rank, usize)> =
            groups.iter().map(|g| (g[0].rank, g.len())).collect();
        let active_opponents: Vec<&OpponentHand> =
            context.opponents.iter().filter(|o| o.active).collect();
        let hands: Vec<usize> = active_opponents.iter().map(|o| o.hand_size).collect();

        let mut state = [0.0; FEATURE_COUNT];
        state[8] = flag(context.current_combo.is_none());
        state[9] = context
            .current_combo
            .map_or(0.0, |c| count(c.size()) / COMBO_SCALE);
        state[10] = count(hand.len()) / HAND_SCALE;
        state[11] = ratio(
            groups.iter().filter(|g| g.len() >= 2).map(Vec::len).sum(),
            hand.len(),
        );
        state[12] = ratio(groups.iter().filter(|g| g.len() == 1).count(), hand.len());
        state[13] = count(active_opponents.len()) / 5.0;
        state[14] = hands.iter().min().map_or(0.0, |&h| count(h) / HAND_SCALE);
        state[15] = if hands.is_empty() {
            0.0
        } else {
            count(hands.iter().sum()) / count(hands.len()) / HAND_SCALE
        };
        state[16] = flag(hands.iter().any(|&h| h <= CLOSE_HAND));

        Self {
            duplicate_rule,
            hand,
            unseen: &context.unseen_cards,
            unseen_groups: rank_groups(&context.unseen_cards),
            max_hand_rank: hand.iter().map(|c| c.rank).max(),
            hand_group_sizes,
            active_opponents,
            state,
        }
    }

    /// The feature vector for playing or passing `candidate` this turn.
    #[must_use]
    pub fn features(&self, candidate: &Move) -> [f64; FEATURE_COUNT] {
        let mut f = self.state;
        match candidate {
            Move::Pass => {
                f[0] = 1.0;
                f[7] = 1.0;
            }
            Move::Play(combo) => {
                let rule = self.duplicate_rule;
                let top = combo.top_card(rule);
                let size = combo.size();
                f[1] = count(size) / COMBO_SCALE;
                f[2] = strength(top);
                f[3] = ratio(
                    self.hand
                        .iter()
                        .filter(|c| c.compare(&top, rule) == Ordering::Less)
                        .count(),
                    self.hand.len(),
                );
                f[4] = flag(self.max_hand_rank == Some(top.rank));
                let group_size = self
                    .hand_group_sizes
                    .iter()
                    .find(|(rank, _)| *rank == top.rank)
                    .map_or(size, |&(_, n)| n);
                f[5] = flag(group_size > size);
                f[6] = flag(size >= self.hand.len());
                f[7] = ratio(self.hand.len().saturating_sub(size), self.hand.len());
                f[17] = ratio(
                    self.unseen
                        .iter()
                        .filter(|c| c.compare(&top, rule) == Ordering::Greater)
                        .count(),
                    self.unseen.len(),
                );
                let beating_ranks = self
                    .unseen_groups
                    .iter()
                    .filter(|group| {
                        group.len() >= size
                            && group
                                .iter()
                                .any(|c| c.compare(&top, rule) == Ordering::Greater)
                    })
                    .count();
                f[18] = count(beating_ranks) / RANK_COUNT;
                f[19] = ratio(
                    self.active_opponents
                        .iter()
                        .filter(|o| o.pass_ceilings.cannot_beat(size, top, rule))
                        .count(),
                    self.active_opponents.len(),
                );
            }
        }
        f
    }
}
```

- [ ] **Step 8: Run (expect success)**

Run: `cargo fmt -p sim && cargo test -p sim neat_player`

Expected: PASS (10 feature tests).

- [ ] **Step 9: Commit**

```bash
git add sim/Cargo.toml sim/src/strategies/mod.rs sim/src/strategies/neat_player Cargo.lock
git commit -F - <<'EOF'
sim: add NEAT candidate features (Phase 10b)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
```


### Task 3: `GenomeFile`: a genome plus the feature set it was trained on

**Files:**
- Modify: `sim/src/strategies/neat_player/mod.rs`
- Create: `sim/src/strategies/neat_player/genome_file.rs`

**Interfaces:**
- Consumes: `FEATURE_NAMES`, `FEATURE_COUNT` (Task 2); `neat::Genome` JSON (Task 1).
- Produces: `GenomeFile { format_version, feature_names, genome }` with `new(Genome) -> Result`, `from_json(&str)`, `load(&Path)`, `save(&self, &Path)`; `GenomeFileError { Io, Parse, Mismatch }` (Display); `FORMAT_VERSION: u32`.

- [ ] **Step 1: Edit**

In `sim/src/strategies/neat_player/mod.rs`, replace:

```rust
mod features;
```

with:

```rust
mod features;
mod genome_file;
```

- [ ] **Step 2: Edit**

In `sim/src/strategies/neat_player/mod.rs`, replace:

```rust
pub use features::{TurnSummary, FEATURE_COUNT, FEATURE_NAMES};
```

with:

```rust
pub use features::{TurnSummary, FEATURE_COUNT, FEATURE_NAMES};
pub use genome_file::{GenomeFile, GenomeFileError, FORMAT_VERSION};
```

- [ ] **Step 3: Create file**

Create `sim/src/strategies/neat_player/genome_file.rs` (Tests first):

```rust
#[cfg(test)]
mod tests {
    use neat::{InnovationTracker, NeatConfig};
    use rand::SeedableRng;

    use super::*;

    fn sample_genome() -> Genome {
        let mut tracker = InnovationTracker::new(u32::try_from(FEATURE_COUNT).unwrap() + 2);
        Genome::minimal(
            FEATURE_COUNT,
            &mut tracker,
            &NeatConfig::default(),
            &mut rand::rngs::StdRng::seed_from_u64(1),
        )
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "arschloch-genome-{name}-{}.json",
            std::process::id()
        ))
    }

    #[test]
    fn save_then_load_round_trips() {
        let file = GenomeFile::new(sample_genome()).unwrap();
        let path = temp_path("roundtrip");
        file.save(&path).unwrap();
        let loaded = GenomeFile::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(loaded, file);
    }

    #[test]
    fn a_genome_with_the_wrong_input_count_is_refused() {
        let mut tracker = InnovationTracker::new(5);
        let genome = Genome::minimal(
            3,
            &mut tracker,
            &NeatConfig::default(),
            &mut rand::rngs::StdRng::seed_from_u64(1),
        );
        let error = GenomeFile::new(genome).unwrap_err();
        assert!(matches!(error, GenomeFileError::Mismatch(_)), "{error}");
    }

    #[test]
    fn a_file_trained_on_other_features_is_refused_with_both_lists_named() {
        let mut file = GenomeFile::new(sample_genome()).unwrap();
        file.feature_names[0] = "something_else".into();
        let error = GenomeFile::from_json(&serde_json::to_string(&file).unwrap()).unwrap_err();
        let GenomeFileError::Mismatch(reason) = &error else {
            panic!("expected Mismatch, got {error}");
        };
        assert!(
            reason.contains("something_else") && reason.contains("is_pass"),
            "{reason}"
        );
    }

    #[test]
    fn an_unknown_format_version_is_refused() {
        let mut file = GenomeFile::new(sample_genome()).unwrap();
        file.format_version = 99;
        let error = GenomeFile::from_json(&serde_json::to_string(&file).unwrap()).unwrap_err();
        assert!(matches!(error, GenomeFileError::Mismatch(_)), "{error}");
    }

    #[test]
    fn garbage_and_invalid_genomes_are_parse_errors_not_panics() {
        assert!(matches!(
            GenomeFile::from_json("not json"),
            Err(GenomeFileError::Parse(_))
        ));
        let file = GenomeFile::new(sample_genome()).unwrap();
        let mut value = serde_json::to_value(&file).unwrap();
        value["genome"]["connections"][0]["to"] = serde_json::json!(9999);
        assert!(matches!(
            GenomeFile::from_json(&value.to_string()),
            Err(GenomeFileError::Parse(_))
        ));
    }

    #[test]
    fn a_missing_file_is_an_io_error_naming_the_path() {
        let error = GenomeFile::load(Path::new("/nonexistent/genome.json")).unwrap_err();
        let GenomeFileError::Io(reason) = &error else {
            panic!("expected Io, got {error}");
        };
        assert!(reason.contains("/nonexistent/genome.json"), "{reason}");
    }
}
```

- [ ] **Step 4: Run (expect failure)**

Run: `cargo test -p sim genome_file`

Expected: FAIL (compile errors: `GenomeFile` is not defined).

- [ ] **Step 5: Implement**

Insert at the very top of `sim/src/strategies/neat_player/genome_file.rs`, above the `#[cfg(test)]` line:

```rust
//! The on-disk form of a trained genome.
//!
//! A genome is only meaningful together with the feature set it was
//! trained against, so the file records the feature names and the load
//! refuses a file whose features differ from this build's: a stale
//! genome fails loudly instead of silently playing garbage.

use std::fmt;
use std::path::Path;

use neat::Genome;
use serde::{Deserialize, Serialize};

use super::features::{FEATURE_COUNT, FEATURE_NAMES};

/// Bumped when the file layout (not the feature set) changes.
pub const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenomeFile {
    pub format_version: u32,
    pub feature_names: Vec<String>,
    pub genome: Genome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenomeFileError {
    Io(String),
    Parse(String),
    /// The file is well-formed but does not fit this build.
    Mismatch(String),
}

impl fmt::Display for GenomeFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(reason) => write!(f, "cannot access genome file: {reason}"),
            Self::Parse(reason) => write!(f, "malformed genome file: {reason}"),
            Self::Mismatch(reason) => write!(f, "genome file does not fit this build: {reason}"),
        }
    }
}

impl std::error::Error for GenomeFileError {}

impl GenomeFile {
    /// Wraps `genome` for saving.
    ///
    /// # Errors
    ///
    /// Returns `GenomeFileError::Mismatch` if the genome does not take
    /// exactly `FEATURE_COUNT` inputs.
    pub fn new(genome: Genome) -> Result<Self, GenomeFileError> {
        let file = Self {
            format_version: FORMAT_VERSION,
            feature_names: FEATURE_NAMES.iter().map(|&n| n.to_owned()).collect(),
            genome,
        };
        file.check_fits_this_build()?;
        Ok(file)
    }

    fn check_fits_this_build(&self) -> Result<(), GenomeFileError> {
        if self.format_version != FORMAT_VERSION {
            return Err(GenomeFileError::Mismatch(format!(
                "format version {} (this build reads {FORMAT_VERSION})",
                self.format_version
            )));
        }
        if self.feature_names != FEATURE_NAMES {
            return Err(GenomeFileError::Mismatch(format!(
                "trained on features {:?}, this build uses {FEATURE_NAMES:?}",
                self.feature_names
            )));
        }
        if self.genome.num_inputs() != FEATURE_COUNT {
            return Err(GenomeFileError::Mismatch(format!(
                "genome takes {} inputs, this build has {FEATURE_COUNT} features",
                self.genome.num_inputs()
            )));
        }
        Ok(())
    }

    /// Parses and checks a genome file's JSON text.
    ///
    /// # Errors
    ///
    /// `Parse` for malformed JSON or a structurally invalid genome,
    /// `Mismatch` for a file trained against a different feature set.
    pub fn from_json(text: &str) -> Result<Self, GenomeFileError> {
        let file: Self =
            serde_json::from_str(text).map_err(|e| GenomeFileError::Parse(e.to_string()))?;
        file.check_fits_this_build()?;
        Ok(file)
    }

    /// # Errors
    ///
    /// `Io` if the file cannot be read, otherwise as `from_json`.
    pub fn load(path: &Path) -> Result<Self, GenomeFileError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| GenomeFileError::Io(format!("{}: {e}", path.display())))?;
        Self::from_json(&text)
    }

    /// Writes pretty-printed JSON, replacing any existing file.
    ///
    /// # Errors
    ///
    /// `Io` if the file cannot be written.
    pub fn save(&self, path: &Path) -> Result<(), GenomeFileError> {
        let text = serde_json::to_string_pretty(self)
            .map_err(|e| GenomeFileError::Parse(e.to_string()))?;
        std::fs::write(path, text)
            .map_err(|e| GenomeFileError::Io(format!("{}: {e}", path.display())))
    }
}
```

- [ ] **Step 6: Run (expect success)**

Run: `cargo fmt -p sim && cargo test -p sim genome_file`

Expected: PASS (6 tests).

- [ ] **Step 7: Commit**

```bash
git add sim/src/strategies/neat_player
git commit -F - <<'EOF'
sim: add GenomeFile, a genome with its feature set (Phase 10b)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
```


### Task 4: `NeatStrategy`

**Files:**
- Modify: `sim/src/strategies/neat_player/mod.rs`, `sim/src/strategies/mod.rs`, `sim/src/lib.rs`

**Interfaces:**
- Consumes: `TurnSummary`, `FEATURE_COUNT` (Task 2), `GenomeFile`, `GenomeFileError` (Task 3), `neat::{Genome, Network}`, `sim::strategy::{Strategy, TurnContext}`, `strategies::take_highest_naive`.
- Produces: `NeatStrategy::new(name, &Genome) -> Result<Self, GenomeFileError>`, `NeatStrategy::from_file(&Path) -> Result<Self, GenomeFileError>` (named `Neat(<file stem>)`), `impl Strategy for NeatStrategy` (argmax of the network score over `legal_moves`, first listed wins ties; exchange = give up the highest cards). Re-exported from `sim`: `NeatStrategy`, `GenomeFile`, `GenomeFileError`, `TurnSummary`, `FEATURE_COUNT`, `FEATURE_NAMES`.

- [ ] **Step 1: Append**

Append to the end of `sim/src/strategies/neat_player/mod.rs` (Tests first: append this test module to the end of `mod.rs`):

```rust

#[cfg(test)]
mod tests {
    use engine::{Combo, Rank, Suit};
    use neat::{ConnectionGene, NodeGene, NodeKind};
    use rand::SeedableRng;

    use super::*;
    use crate::hand_reading::PassCeilings;

    /// A genome scoring `sum(weight * feature)` for the given
    /// `(feature index, weight)` pairs, wired directly to the output.
    pub(crate) fn linear_genome(weights: &[(usize, f64)]) -> Genome {
        let inputs = FEATURE_COUNT;
        let mut nodes: Vec<NodeGene> = (0..inputs)
            .map(|id| NodeGene {
                id: u32::try_from(id).unwrap(),
                kind: NodeKind::Input,
            })
            .collect();
        nodes.push(NodeGene {
            id: u32::try_from(inputs).unwrap(),
            kind: NodeKind::Bias,
        });
        nodes.push(NodeGene {
            id: u32::try_from(inputs + 1).unwrap(),
            kind: NodeKind::Output,
        });
        let connections = weights
            .iter()
            .enumerate()
            .map(|(innovation, &(feature, weight))| ConnectionGene {
                innovation: u32::try_from(innovation).unwrap(),
                from: u32::try_from(feature).unwrap(),
                to: u32::try_from(inputs + 1).unwrap(),
                weight,
                enabled: true,
            })
            .collect();
        Genome::from_parts(inputs, nodes, connections).unwrap()
    }

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit, 0)
    }

    fn single(rank: Rank, suit: Suit) -> Move {
        Move::Play(Combo::new(vec![card(rank, suit)]).unwrap())
    }

    fn context(hand: &[Card]) -> TurnContext<'_> {
        TurnContext {
            seat: 0,
            hand,
            opponents: Vec::new(),
            unseen_cards: Vec::new(),
            own_pass_ceilings: PassCeilings::default(),
            current_combo: None,
        }
    }

    fn rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(0)
    }

    const RULE: DuplicateRule = DuplicateRule::FirstDealtWins;

    #[test]
    fn a_negative_strength_weight_picks_the_weakest_card() {
        let strategy = NeatStrategy::new("t", &linear_genome(&[(2, -1.0)])).unwrap();
        let hand = [
            card(Rank::Nine, Suit::Spades),
            card(Rank::Four, Suit::Hearts),
        ];
        let moves = vec![
            single(Rank::Nine, Suit::Spades),
            single(Rank::Four, Suit::Hearts),
        ];
        let chosen = strategy.choose_play(&moves, RULE, &context(&hand), &mut rng());
        assert_eq!(chosen, single(Rank::Four, Suit::Hearts));
    }

    #[test]
    fn a_positive_strength_weight_picks_the_strongest_card() {
        let strategy = NeatStrategy::new("t", &linear_genome(&[(2, 1.0)])).unwrap();
        let hand = [
            card(Rank::Nine, Suit::Spades),
            card(Rank::Four, Suit::Hearts),
        ];
        let moves = vec![
            single(Rank::Nine, Suit::Spades),
            single(Rank::Four, Suit::Hearts),
        ];
        let chosen = strategy.choose_play(&moves, RULE, &context(&hand), &mut rng());
        assert_eq!(chosen, single(Rank::Nine, Suit::Spades));
    }

    #[test]
    fn the_pass_flag_weight_decides_between_passing_and_playing() {
        let hand = [card(Rank::Nine, Suit::Spades)];
        let on_table = Combo::new(vec![card(Rank::Four, Suit::Hearts)]).unwrap();
        let mut ctx = context(&hand);
        ctx.current_combo = Some(&on_table);
        let moves = vec![single(Rank::Nine, Suit::Spades), Move::Pass];
        let keen = NeatStrategy::new("t", &linear_genome(&[(0, -1.0)])).unwrap();
        assert_eq!(
            keen.choose_play(&moves, RULE, &ctx, &mut rng()),
            single(Rank::Nine, Suit::Spades)
        );
        let shy = NeatStrategy::new("t", &linear_genome(&[(0, 1.0)])).unwrap();
        assert_eq!(shy.choose_play(&moves, RULE, &ctx, &mut rng()), Move::Pass);
    }

    #[test]
    fn exact_ties_keep_the_first_listed_move() {
        // No connections: every candidate scores tanh(0) = 0.
        let strategy = NeatStrategy::new("t", &linear_genome(&[])).unwrap();
        let hand = [
            card(Rank::Nine, Suit::Spades),
            card(Rank::Four, Suit::Hearts),
        ];
        let moves = vec![
            single(Rank::Nine, Suit::Spades),
            single(Rank::Four, Suit::Hearts),
        ];
        let chosen = strategy.choose_play(&moves, RULE, &context(&hand), &mut rng());
        assert_eq!(chosen, moves[0]);
    }

    #[test]
    fn a_genome_for_a_different_feature_count_is_refused() {
        let mut tracker = neat::InnovationTracker::new(5);
        let small = Genome::minimal(
            3,
            &mut tracker,
            &neat::NeatConfig::default(),
            &mut rand::rngs::StdRng::seed_from_u64(1),
        );
        assert!(matches!(
            NeatStrategy::new("t", &small),
            Err(GenomeFileError::Mismatch(_))
        ));
    }

    #[test]
    fn from_file_names_the_player_after_the_file() {
        let file = GenomeFile::new(linear_genome(&[(2, -1.0)])).unwrap();
        let path = std::env::temp_dir().join(format!("my-champion-{}.json", std::process::id()));
        file.save(&path).unwrap();
        let strategy = NeatStrategy::from_file(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(
            strategy.name().starts_with("Neat(my-champion-"),
            "{}",
            strategy.name()
        );
    }

    #[test]
    fn the_exchange_gives_up_the_highest_cards() {
        let strategy = NeatStrategy::new("t", &linear_genome(&[])).unwrap();
        let hand = [
            card(Rank::Two, Suit::Clubs),
            card(Rank::Ace, Suit::Clubs),
            card(Rank::King, Suit::Hearts),
        ];
        let given = strategy.choose_exchange_cards(&hand, 2, RULE, &mut rng());
        assert_eq!(given.len(), 2);
        assert!(given.contains(&card(Rank::Ace, Suit::Clubs)));
        assert!(given.contains(&card(Rank::King, Suit::Hearts)));
    }

    #[test]
    fn the_strategy_can_be_shared_across_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NeatStrategy>();
    }
}
```

- [ ] **Step 2: Run (expect failure)**

Run: `cargo test -p sim neat_player::tests`

Expected: FAIL (compile errors: `NeatStrategy` is not defined).

- [ ] **Step 3: Implement**

In `sim/src/strategies/neat_player/mod.rs`, replace everything above the first `#[cfg(test)]` line (Replace everything above the `#[cfg(test)]` line with the final module header and implementation) with:

```rust
//! `NeatStrategy`: plays by scoring every legal move with an evolved
//! neural network and choosing the highest score. See
//! docs/superpowers/specs/2026-10-08-neat-engine-design.md, section 4.
//!
//! The network never produces a move, only a number per candidate, so
//! the chosen move is always one `engine` reported as legal. The
//! strategy holds an immutable compiled network and no other state, so
//! one instance is shared across parallel matches.

mod features;
mod genome_file;

use std::cmp::Ordering;
use std::path::Path;

use engine::{Card, DuplicateRule, Move};
use neat::{Genome, Network};

pub use features::{TurnSummary, FEATURE_COUNT, FEATURE_NAMES};
pub use genome_file::{GenomeFile, GenomeFileError, FORMAT_VERSION};

use crate::strategy::{Strategy, TurnContext};

#[derive(Debug, Clone)]
pub struct NeatStrategy {
    name: String,
    network: Network,
}

impl NeatStrategy {
    /// Builds a player from `genome`, reported under `name` in results.
    ///
    /// # Errors
    ///
    /// Returns `GenomeFileError::Mismatch` if the genome does not take
    /// exactly `FEATURE_COUNT` inputs.
    pub fn new(name: impl Into<String>, genome: &Genome) -> Result<Self, GenomeFileError> {
        if genome.num_inputs() != FEATURE_COUNT {
            return Err(GenomeFileError::Mismatch(format!(
                "genome takes {} inputs, this build has {FEATURE_COUNT} features",
                genome.num_inputs()
            )));
        }
        Ok(Self {
            name: name.into(),
            network: Network::compile(genome),
        })
    }

    /// Loads a genome file and names the player `Neat(<file stem>)`.
    ///
    /// # Errors
    ///
    /// Any `GenomeFileError` from loading or checking the file.
    pub fn from_file(path: &Path) -> Result<Self, GenomeFileError> {
        let file = GenomeFile::load(path)?;
        let stem = path
            .file_stem()
            .map_or_else(|| "genome".into(), |s| s.to_string_lossy());
        Self::new(format!("Neat({stem})"), &file.genome)
    }
}

impl Strategy for NeatStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        context: &TurnContext<'_>,
        _rng: &mut dyn rand::Rng,
    ) -> Move {
        let summary = TurnSummary::new(context, duplicate_rule);
        let mut scratch = Vec::new();
        let mut best: Option<(&Move, f64)> = None;
        for candidate in legal_moves {
            let score = self
                .network
                .activate(&summary.features(candidate), &mut scratch);
            // Strictly better only: ties keep the earlier-listed move, so
            // the choice is a pure function of the legal-move order.
            if best.is_none_or(|(_, top)| score.total_cmp(&top) == Ordering::Greater) {
                best = Some((candidate, score));
            }
        }
        best.expect("a seat to move always has at least one legal move")
            .0
            .clone()
    }

    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        duplicate_rule: DuplicateRule,
        _rng: &mut dyn rand::Rng,
    ) -> Vec<Card> {
        crate::strategies::take_highest_naive(hand, count, duplicate_rule)
    }
}
```

- [ ] **Step 4: Edit**

In `sim/src/strategies/mod.rs`, replace:

```rust
pub use lowest_legal::LowestLegal;
```

with:

```rust
pub use lowest_legal::LowestLegal;
pub use neat_player::{
    GenomeFile, GenomeFileError, NeatStrategy, TurnSummary, FEATURE_COUNT, FEATURE_NAMES,
    FORMAT_VERSION,
};
```

- [ ] **Step 5: Edit**

In `sim/src/lib.rs`, replace:

```rust
pub use strategies::{
    Adaptive, AdaptiveConfig, CardCounter, DenialMode, EndgameDenial, GreedyHighest, HoldBackPairs,
    LowestLegal, RandomLegal,
};
```

with:

```rust
pub use strategies::{
    Adaptive, AdaptiveConfig, CardCounter, DenialMode, EndgameDenial, GenomeFile, GenomeFileError,
    GreedyHighest, HoldBackPairs, LowestLegal, NeatStrategy, RandomLegal, TurnSummary,
    FEATURE_COUNT, FEATURE_NAMES,
};
```

- [ ] **Step 6: Run (expect success)**

Run: `cargo fmt -p sim && cargo test -p sim neat_player`

Expected: PASS (24 tests: features, genome_file, strategy).

- [ ] **Step 7: Commit**

```bash
git add sim
git commit -F - <<'EOF'
sim: add NeatStrategy, a network-scored player (Phase 10b)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
```


### Task 5: Real-scenario tests: equivalence to `LowestLegal`, evolved networks everywhere

**Files:**
- Create: `sim/tests/common/mod.rs`, `sim/tests/neat_equivalence.rs`, `sim/tests/neat_random_genomes.rs`

**Why:** Unit tests show each feature has the value its name claims. These show the pieces work *together* in real games: a network wired to mean "lowest legal" must play exactly like `LowestLegal`, and evolved networks with hidden nodes must always pick legal moves at every table size, deck and duplicate rule.

**Interfaces:**
- Consumes: `NeatStrategy`, `run_match`, `run_batch`, `LowestLegal`, `CardCounter`, `RandomLegal`, `neat::Population` (for evolved topologies).
- Produces: Evidence the whole stack plays correctly in complete matches (no new API).

- [ ] **Step 1: Create file**

Create `sim/tests/common/mod.rs`:

```rust
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
```

- [ ] **Step 2: Create file**

Create `sim/tests/neat_equivalence.rs`:

```rust
//! End-to-end proof that features, scoring and move choice are wired
//! correctly: a network hand-wired to mean "lowest legal" must play
//! *exactly* like `LowestLegal` in complete matches. One wrong sign, a
//! swapped feature or a broken tie-break changes some move in some match
//! and fails this test.

mod common;

use std::sync::Arc;

use engine::{DeckVariant, DuplicateRule};
use sim::{run_match, LowestLegal, MatchConfig, NeatStrategy, Strategy};

fn table(player_count: u8, make: &dyn Fn() -> Arc<dyn Strategy>) -> Vec<Arc<dyn Strategy>> {
    (0..player_count).map(|_| make()).collect()
}

#[test]
fn a_lowest_legal_network_plays_identically_to_lowest_legal() {
    let neat: Arc<dyn Strategy> =
        Arc::new(NeatStrategy::new("Neat(test)", &common::lowest_legal_genome()).unwrap());
    let reference: Arc<dyn Strategy> = Arc::new(LowestLegal);
    for player_count in 3..=6 {
        for duplicate_rule in [DuplicateRule::FirstDealtWins, DuplicateRule::LastDealtWins] {
            for seed in 0..40 {
                let config = MatchConfig {
                    player_count,
                    deck_variant: DeckVariant::Single,
                    duplicate_rule,
                    rounds: 6,
                    seed,
                };
                let a = run_match(&config, &table(player_count, &|| neat.clone()));
                let b = run_match(&config, &table(player_count, &|| reference.clone()));
                let context = format!("{player_count} players, {duplicate_rule:?}, seed {seed}");
                assert_eq!(a.role_history, b.role_history, "{context}");
                assert_eq!(a.trick_count, b.trick_count, "{context}");
                assert_eq!(a.pass_counts, b.pass_counts, "{context}");
                assert_eq!(
                    a.voluntary_pass_counts, b.voluntary_pass_counts,
                    "{context}"
                );
            }
        }
    }
}

#[test]
fn flipping_the_strength_weight_changes_play() {
    // Guards the equivalence test above against passing vacuously: a
    // network that prefers the *strongest* card must differ from
    // LowestLegal somewhere.
    let greedy: Arc<dyn Strategy> = Arc::new(
        NeatStrategy::new(
            "Neat(greedy)",
            &common::linear_genome(&[(0, -3.0), (1, -2.0), (2, 0.2)]),
        )
        .unwrap(),
    );
    let reference: Arc<dyn Strategy> = Arc::new(LowestLegal);
    let differs = (0..40).any(|seed| {
        let config = MatchConfig {
            player_count: 4,
            deck_variant: DeckVariant::Single,
            duplicate_rule: DuplicateRule::FirstDealtWins,
            rounds: 6,
            seed,
        };
        let a = run_match(&config, &table(4, &|| greedy.clone()));
        let b = run_match(&config, &table(4, &|| reference.clone()));
        a.role_history != b.role_history || a.trick_count != b.trick_count
    });
    assert!(differs);
}
```

- [ ] **Step 3: Create file**

Create `sim/tests/neat_random_genomes.rs`:

```rust
//! Robustness: genomes with arbitrary evolved topology must always play
//! legal moves, at every table size, deck and duplicate rule, alongside
//! other strategies, in parallel, and deterministically. `run_match`
//! panics if a strategy ever picks a move `engine` did not offer, so
//! simply completing the matches is the assertion.

use std::sync::Arc;

use engine::{DeckVariant, DuplicateRule};
use neat::{NeatConfig, Network, Population};
use rand::{RngExt, SeedableRng};
use sim::{
    run_batch, CardCounter, LowestLegal, MatchConfig, NeatStrategy, RandomLegal, Strategy,
    FEATURE_COUNT,
};

/// Genomes with real hidden structure: evolve a population for a while
/// against random fitness, so topologies diverge, then take a spread.
fn diverse_genomes(count: usize) -> Vec<neat::Genome> {
    let config = NeatConfig {
        population_size: 60,
        add_node_rate: 0.3,
        add_connection_rate: 0.3,
        toggle_enable_rate: 0.1,
        ..NeatConfig::default()
    };
    let mut population = Population::new(FEATURE_COUNT, config, 5).unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(9);
    for _ in 0..25 {
        let fitness = (0..60).map(|_| rng.random_range(0.0..1.0)).collect();
        population.set_fitness(fitness);
        population.advance();
    }
    population.genomes().iter().take(count).cloned().collect()
}

fn configs(player_count: u8, deck_variant: DeckVariant, rule: DuplicateRule) -> Vec<MatchConfig> {
    (0..12)
        .map(|seed| MatchConfig {
            player_count,
            deck_variant,
            duplicate_rule: rule,
            rounds: 4,
            seed,
        })
        .collect()
}

fn seats(player_count: u8, genomes: &[neat::Genome]) -> Vec<Arc<dyn Strategy>> {
    (0..usize::from(player_count))
        .map(|seat| -> Arc<dyn Strategy> {
            match seat % 4 {
                0 | 1 => Arc::new(
                    NeatStrategy::new(format!("Neat({seat})"), &genomes[seat % genomes.len()])
                        .unwrap(),
                ),
                2 => Arc::new(LowestLegal),
                _ => Arc::new(CardCounter),
            }
        })
        .collect()
}

#[test]
fn evolved_networks_always_play_legal_moves_everywhere() {
    let genomes = diverse_genomes(8);
    assert!(
        genomes.iter().any(|g| g.hidden_count() > 0),
        "the scenario needs real hidden structure"
    );
    for player_count in 3..=6 {
        for deck in [DeckVariant::Single, DeckVariant::Double] {
            for rule in [DuplicateRule::FirstDealtWins, DuplicateRule::LastDealtWins] {
                let results = run_batch(
                    &configs(player_count, deck, rule),
                    &seats(player_count, &genomes),
                );
                assert_eq!(results.len(), 12);
                for result in &results {
                    assert_eq!(result.role_history.len(), 4);
                }
            }
        }
    }
}

#[test]
fn batches_with_networks_are_deterministic_and_thread_safe() {
    let genomes = diverse_genomes(8);
    let run = || {
        let results = run_batch(
            &configs(5, DeckVariant::Single, DuplicateRule::FirstDealtWins),
            &seats(5, &genomes),
        );
        results
            .iter()
            .map(|r| (r.role_history.clone(), r.trick_count, r.pass_counts.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(run(), run());
}

#[test]
fn random_networks_do_not_just_play_like_random_legal() {
    // A sanity check that the scenario exercises the network: the
    // evolved genomes' choices must differ from a uniformly random
    // player's. (Same table, same seeds: only the strategies differ.)
    let genomes = diverse_genomes(4);
    let neat_table: Vec<Arc<dyn Strategy>> = (0..4)
        .map(|i| -> Arc<dyn Strategy> { Arc::new(NeatStrategy::new("n", &genomes[i]).unwrap()) })
        .collect();
    let random_table: Vec<Arc<dyn Strategy>> = (0..4)
        .map(|_| -> Arc<dyn Strategy> { Arc::new(RandomLegal) })
        .collect();
    let config = MatchConfig {
        player_count: 4,
        deck_variant: DeckVariant::Single,
        duplicate_rule: DuplicateRule::FirstDealtWins,
        rounds: 6,
        seed: 3,
    };
    let a = sim::run_match(&config, &neat_table);
    let b = sim::run_match(&config, &random_table);
    assert!(a.role_history != b.role_history || a.pass_counts != b.pass_counts);
    // And the network itself is real: compiled output is bounded.
    let network = Network::compile(&genomes[0]);
    let out = network.activate(&[0.5; FEATURE_COUNT], &mut Vec::new());
    assert!(out.abs() <= 1.0);
}
```

- [ ] **Step 4: Run (expect success)**

Run: `cargo fmt -p sim && cargo test -p sim --test neat_equivalence --test neat_random_genomes`

Expected: PASS (2 + 3 tests; a few seconds in a debug build). These are integration tests over existing code, so they cannot be seen failing first; the next three steps prove the equivalence test is sensitive.

- [ ] **Step 5: Edit**

In `sim/src/strategies/neat_player/features.rs` (Sensitivity check: temporarily invert the strength feature), replace:

```rust
f[2] = strength(top);
```

with:

```rust
f[2] = 1.0 - strength(top);
```

- [ ] **Step 6: Run (expect failure)**

Run: `cargo test -p sim --test neat_equivalence`

Expected: FAIL: `a_lowest_legal_network_plays_identically_to_lowest_legal` reports differing role histories, proving the test notices a wrong feature.

- [ ] **Step 7: Edit**

In `sim/src/strategies/neat_player/features.rs` (Revert the deliberate break), replace:

```rust
f[2] = 1.0 - strength(top);
```

with:

```rust
f[2] = strength(top);
```

- [ ] **Step 8: Run (expect success)**

Run: `cargo test -p sim`

Expected: PASS (whole `sim` suite).

- [ ] **Step 9: Commit**

```bash
git add sim/tests
git commit -F - <<'EOF'
sim: add real-scenario tests for NeatStrategy (Phase 10b)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
```


### Task 6: CLI: `--strategy neat:PATH`

**Files:**
- Modify: `cli/Cargo.toml`, `cli/src/args.rs`, `cli/tests/smoke.rs`

**Interfaces:**
- Consumes: `sim::{NeatStrategy, GenomeFile, FEATURE_COUNT}` (Task 4), `neat::{Population, NeatConfig}` (dev-dependency, to make genome files in tests).
- Produces: `StrategyArg::Neat(NeatSpec)`; the spec `neat:PATH` loads and validates the genome file while arguments are parsed, so a missing, malformed or stale file is an argument error before any match runs. The player is reported as `Neat(<file stem>)`.

- [ ] **Step 1: Edit**

In `cli/Cargo.toml`, replace:

```toml
serde_json = "1.0.151"
```

with:

```toml
serde_json = { version = "1.0.151", features = ["float_roundtrip"] }
```

- [ ] **Step 2: Append**

Append to the end of `cli/Cargo.toml` (Dev-dependency for building genome files in tests):

```toml

[dev-dependencies]
neat = { version = "0.1.0", path = "../neat" }
```

- [ ] **Step 3: Edit**

In `cli/src/args.rs` (Tests first: add these to the `tests` module, above `deck_variant_arg_converts_to_engine_type`), replace:

```rust
    #[test]
    fn deck_variant_arg_converts_to_engine_type() {
```

with:

```rust
    fn genome_file(name: &str) -> PathBuf {
        let mut population = neat::Population::new(
            sim::FEATURE_COUNT,
            neat::NeatConfig {
                population_size: 4,
                ..neat::NeatConfig::default()
            },
            1,
        )
        .unwrap();
        let genome = population.genomes()[0].clone();
        population.set_fitness(vec![0.0; 4]);
        let path = std::env::temp_dir().join(format!("{name}-{}.json", std::process::id()));
        sim::GenomeFile::new(genome).unwrap().save(&path).unwrap();
        path
    }

    #[test]
    fn parses_a_neat_spec_and_names_the_player_after_the_file() {
        let path = genome_file("champ");
        let arg: StrategyArg = format!("neat:{}", path.display()).parse().unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(arg, StrategyArg::Neat(_)));
        assert!(arg.build().name().starts_with("Neat(champ-"));
    }

    #[test]
    fn a_neat_spec_without_a_path_is_rejected() {
        for spec in ["neat", "neat:", "neat:   "] {
            let error = spec.parse::<StrategyArg>().unwrap_err();
            assert!(error.contains("neat:PATH"), "{spec}: {error}");
        }
    }

    #[test]
    fn a_missing_genome_file_is_an_argument_error_naming_the_path() {
        let error = "neat:/nonexistent/champ.json".parse::<StrategyArg>().unwrap_err();
        assert!(error.contains("/nonexistent/champ.json"), "{error}");
    }

    #[test]
    fn a_stale_genome_file_is_refused_at_parse_time() {
        let path = genome_file("stale");
        let mut value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        value["feature_names"][0] = serde_json::json!("renamed_feature");
        std::fs::write(&path, value.to_string()).unwrap();
        let error = format!("neat:{}", path.display())
            .parse::<StrategyArg>()
            .unwrap_err();
        std::fs::remove_file(&path).unwrap();
        assert!(error.contains("does not fit this build"), "{error}");
    }

    #[test]
    fn deck_variant_arg_converts_to_engine_type() {
```

- [ ] **Step 4: Run (expect failure)**

Run: `cargo test -p cli neat`

Expected: FAIL (compile errors: `StrategyArg::Neat` does not exist).

- [ ] **Step 5: Edit**

In `cli/src/args.rs` (Help text), replace:

```rust
    /// `adaptive:OPTIONS` where `OPTIONS` is a comma-separated modifier
    /// list parsed by `sim::AdaptiveConfig`'s `FromStr` (e.g.
    /// `counting`, `reading,deception=0.2`, `none`). For example, a
    /// three-seat table:
```

with:

```rust
    /// `adaptive:OPTIONS` where `OPTIONS` is a comma-separated modifier
    /// list parsed by `sim::AdaptiveConfig`'s `FromStr` (e.g.
    /// `counting`, `reading,deception=0.2`, `none`), or
    /// `neat:PATH`, a trained genome file (see `sim::GenomeFile`),
    /// reported in results as `Neat(<file name without extension>)`. For
    /// example, a three-seat table:
```

- [ ] **Step 6: Edit**

In `cli/src/args.rs`, replace:

```rust
    /// Each value is a spec, `SPEC := FIXED | "adaptive" | "adaptive:"
    /// OPTIONS`: either
```

with:

```rust
    /// Each value is a spec, `SPEC := FIXED | "adaptive" | "adaptive:"
    /// OPTIONS | "neat:" PATH`: either
```

- [ ] **Step 7: Edit**

In `cli/src/args.rs`, replace:

```rust
pub enum StrategyArg {
    Fixed(FixedStrategy),
    Adaptive(sim::AdaptiveConfig),
}
```

with:

```rust
pub enum StrategyArg {
    Fixed(FixedStrategy),
    Adaptive(sim::AdaptiveConfig),
    Neat(NeatSpec),
}

/// A `neat:PATH` spec. The genome file is loaded and checked while the
/// argument is parsed, so a missing, malformed or stale file is reported
/// as an argument error before any simulation starts.
#[derive(Clone, Debug)]
pub struct NeatSpec {
    path: PathBuf,
    strategy: Arc<sim::NeatStrategy>,
}

impl PartialEq for NeatSpec {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
    }
}
```

- [ ] **Step 8: Edit**

In `cli/src/args.rs`, replace:

```rust
        if options.is_some() {
            return Err(format!(
                "strategy `{head}` takes no options (only `adaptive:` does)"
            ));
        }
```

with:

```rust
        if head == "neat" {
            let path = options
                .map(str::trim)
                .filter(|p| !p.is_empty())
                .ok_or("`neat` needs a genome file: neat:PATH")?;
            let path = PathBuf::from(path);
            let strategy = sim::NeatStrategy::from_file(&path).map_err(|e| e.to_string())?;
            return Ok(Self::Neat(NeatSpec {
                path,
                strategy: Arc::new(strategy),
            }));
        }
        if options.is_some() {
            return Err(format!(
                "strategy `{head}` takes no options (only `adaptive:` and `neat:` do)"
            ));
        }
```

- [ ] **Step 9: Edit**

In `cli/src/args.rs`, replace:

```rust
expected one of: {}, adaptive[:OPTIONS]"
```

with:

```rust
expected one of: {}, adaptive[:OPTIONS], neat:PATH"
```

- [ ] **Step 10: Edit**

In `cli/src/args.rs`, replace:

```rust
            Self::Adaptive(config) => Arc::new(sim::Adaptive::new(config)),
```

with:

```rust
            Self::Adaptive(config) => Arc::new(sim::Adaptive::new(config)),
            Self::Neat(spec) => spec.strategy,
```

- [ ] **Step 11: Run (expect success)**

Run: `cargo fmt -p cli && cargo test -p cli --bins`

Expected: PASS (the existing args tests plus 4 new ones).

- [ ] **Step 12: Append**

Append to the end of `cli/tests/smoke.rs` (Smoke tests that run the real built binary: add to the end of `cli/tests/smoke.rs`):

```rust

fn write_genome_file(name: &str) -> std::path::PathBuf {
    let population = neat::Population::new(
        sim::FEATURE_COUNT,
        neat::NeatConfig {
            population_size: 4,
            ..neat::NeatConfig::default()
        },
        1,
    )
    .unwrap();
    let path = std::env::temp_dir().join(format!("{name}-{}.json", std::process::id()));
    sim::GenomeFile::new(population.genomes()[0].clone())
        .unwrap()
        .save(&path)
        .unwrap();
    path
}

#[test]
fn a_trained_genome_plays_in_a_normal_run() {
    let genome = write_genome_file("smoke-champion");
    let output_path = std::env::temp_dir().join(format!("smoke-neat-{}.json", std::process::id()));
    let output = Command::new(env!("CARGO_BIN_EXE_cli"))
        .args(["--player-count", "4", "--matches", "20", "--rounds", "3", "--seed", "5"])
        .args(["--strategy", &format!("neat:{}", genome.display())])
        .args(["--strategy", "lowest-legal", "--strategy", "card-counter"])
        .args(["--strategy", "adaptive:reading,tempo,bully", "--output"])
        .arg(&output_path)
        .output()
        .expect("failed to run cli binary");
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stem = genome.file_stem().unwrap().to_string_lossy().into_owned();
    assert!(stdout.contains(&format!("Neat({stem})")), "{stdout}");
    assert!(stdout.contains("LowestLegal") && stdout.contains("CardCounter"));
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&output_path).unwrap()).unwrap();
    assert_eq!(json["matches"].as_array().unwrap().len(), 20);

    std::fs::remove_file(&genome).ok();
    std::fs::remove_file(&output_path).ok();
}

#[test]
fn a_missing_genome_file_fails_with_a_clear_message() {
    let output = Command::new(env!("CARGO_BIN_EXE_cli"))
        .args(["--player-count", "3", "--matches", "1"])
        .args(["--strategy", "neat:/nonexistent/champion.json"])
        .args(["--strategy", "lowest-legal", "--strategy", "lowest-legal"])
        .output()
        .expect("failed to run cli binary");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("/nonexistent/champion.json"), "{stderr}");
}

#[test]
fn a_garbage_genome_file_fails_without_a_panic() {
    let path = std::env::temp_dir().join(format!("smoke-garbage-{}.json", std::process::id()));
    std::fs::write(&path, "{ definitely not a genome").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_cli"))
        .args(["--player-count", "3", "--matches", "1"])
        .args(["--strategy", &format!("neat:{}", path.display())])
        .args(["--strategy", "lowest-legal", "--strategy", "lowest-legal"])
        .output()
        .expect("failed to run cli binary");
    std::fs::remove_file(&path).ok();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("malformed genome file"), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
}
```

- [ ] **Step 13: Run (expect success)**

Run: `cargo fmt -p cli && cargo test -p cli`

Expected: PASS (3 new smoke tests: a trained genome plays a normal run; a missing file and a garbage file fail with clear messages, no panic).

- [ ] **Step 14: Commit**

```bash
git add cli Cargo.lock
git commit -F - <<'EOF'
cli: accept --strategy neat:PATH genome files (Phase 10b)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
```


### Task 7: Docs and the full gate

**Files:**
- Modify: `docs/ROADMAP.md`, `docs/BUILDING.md`, `docs/ARCHITECTURE.md`, `docs/superpowers/specs/2026-10-08-neat-engine-design.md`

**Interfaces:**
- Consumes: Everything above.
- Produces: Docs that match the code; a clean phase-done gate.

- [ ] **Step 1: Edit**

In `docs/ROADMAP.md` (Roadmap: mark 10b done (it is the second sentence of the Phase 10 bullet that already says `10a (done)`)), replace:

```markdown
10b: `NeatStrategy`
```

with:

```markdown
10b (done): `NeatStrategy`
```

- [ ] **Step 2: Edit**

In `docs/superpowers/specs/2026-10-08-neat-engine-design.md` (Spec: record the deviation (the role feature is not in the 20 features of 10b)), replace:

```markdown
- role: previous-round role as a normalized ordinal (if available).
```

with:

```markdown
- role: *deferred to v2.* `TurnContext` does not carry the acting
  seat's role, and adding it touches every strategy's context
  construction; it only matters once the exchange step is evolved too.
```

- [ ] **Step 3: Edit**

In `docs/BUILDING.md`, replace:

```markdown
The `neat` crate has no game dependency:
```

with:

```markdown
Trained genomes play like any other strategy via `--strategy
neat:PATH` (the player appears in results as `Neat(<file name>)`); the
file records the feature set it was trained on, and a file from a
different build is refused with an error instead of silently misplaying.

The `neat` crate has no game dependency:
```

- [ ] **Step 4: Edit**

In `docs/ARCHITECTURE.md`, replace:

```markdown
### `web` (starts in Phase 10d
```

with:

```markdown
`sim` depends on `neat` for `NeatStrategy`
(`sim/src/strategies/neat_player/`): per turn it builds 20 features
per legal move (including pass), scores each with the compiled network
and plays the highest. `GenomeFile` stores a genome with the feature
names it was trained on.

### `web` (starts in Phase 10d
```

- [ ] **Step 5: Run (expect success)**

Run: `cargo fmt --check`

Expected: PASS.

- [ ] **Step 6: Run (expect success)**

Run: `cargo clippy --workspace --all-targets -- -D warnings`

Expected: Clean.

- [ ] **Step 7: Run (expect success)**

Run: `cargo test --workspace`

Expected: PASS: every suite green (see the ledger for the count).

- [ ] **Step 8: Commit**

```bash
git add docs Cargo.lock
git commit -F - <<'EOF'
docs: Phase 10b done; document neat:PATH and the deferred role feature

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
```


---

## Self-Review (done while writing)

- **Spec coverage (sections 3, 4, 10):** network scores each candidate including pass and plays the argmax (Task 4); the feature list (Task 2, 20 features; `role` deferred with the reason recorded in the spec by Task 7); feature names stored in each genome file so old genomes are rejected (Task 3); exchange reuses `take_highest_naive` (Task 4); `neat:<genome.json>` spec (Task 6); `NeatStrategy` only ever returns a move from `legal_moves` (property exercised by the evolved-network scenario test, Task 5); identical context gives identical result (determinism test, Task 5).
- **Placeholder scan:** none; every step carries its code.
- **Type consistency:** `NeatStrategy::new(name, &Genome)`, `from_file(&Path)`, `GenomeFile::{new, from_json, load, save}`, `TurnSummary::{new, features}` and `FEATURE_COUNT`/`FEATURE_NAMES` are used with the same signatures in Tasks 2-6.
- **Review Focus:** all five lines map to named tests above.
