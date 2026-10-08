# Phase 7 — Adaptive Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship one configurable strategy, `Adaptive`, whose base
behavior is `LowestLegal` with three independently-toggleable
modifiers (card-counting, pass-based hand-reading endgame denial, and
deception), exposed via a fully-parameterized per-seat CLI grammar, so
Phase 4's statistics can show which modifier combination actually beats
plain `LowestLegal` — the strategy that beat every existing strategy in
an empirical batch run.

**Architecture:** `engine::Round` gains a pass history mirroring its
existing play history. A new `sim::hand_reading` module reduces both
histories into per-seat "pass ceilings" (the lowest top card a seat is
known unable to beat, per combo size) each turn. `sim::TurnContext`
gains these ceilings plus the current table combo. `sim::Strategy::
name()` changes from `&'static str` to `&str` so a configurable
strategy's name can reflect its actual configuration. A new `sim::
strategies::adaptive` module assembles the three modifiers (delegating
to the *existing* `CardCounter`/`GreedyHighest`/`LowestLegal` strategy
structs wherever their exact behavior is reused, never duplicating
their logic) into one `Strategy` impl. `cli` gains a `FromStr`-based
per-seat strategy spec grammar.

**Tech Stack:** Rust workspace (`engine`, `sim`, `cli`), `rand 0.10.3`
(already a dependency; this phase is the first to use `rand::RngExt::
random_bool`, confirmed present against the installed crate source).

**Spec:** The native-plan-mode design doc this plan transcribes (not a
separate spec file — this plan is the spec for this phase, same
convention as Phases 4-6). That doc itself synthesizes an Explore
agent's findings and two Opus-model design agents' algorithm/
architecture proposals, with every reconciliation decision between them
called out explicitly in that doc's Context section — read it
(`docs/superpowers/plans/2026-09-27-phase-7-adaptive-strategy.md`'s own
git history has the native-mode doc's exact text if you need the full
reasoning trail; this plan's task briefs below are the actionable
distillation of it).

## Global Constraints

- No new Cargo dependencies.
- `engine` stays dependency-free of `sim`/`cli` — `Round`'s new
  `pass_history` field/accessor is a plain data accessor, never
  anything strategy-aware.
- `CardCounter`, `EndgameDenial`, `HoldBackPairs`, `GreedyHighest`,
  `LowestLegal`, `RandomLegal` — **none of their existing algorithms
  change**. `Adaptive` reuses them by direct delegation (calling their
  `choose_play`/`choose_exchange_cards` as ordinary `Strategy` trait
  methods on their zero-field unit structs), exactly how `HoldBackPairs`
  already delegates to `LowestLegal` and `EndgameDenial` already
  delegates to `GreedyHighest`/`LowestLegal` today. Do not extract
  their logic into shared helper functions, and do not modify their
  files' logic — only their `choose_play`/`name()` *signatures* change,
  mechanically, in Task 3.
- `Strategy::choose_exchange_cards` is **not** touched by the
  hand-reading/pass-ceiling work — same reasoning as Phase 6 (no play
  history exists between rounds).
- `Adaptive` must be genuinely stateless-per-call apart from its own
  immutable `config`/`name` fields (no interior mutability) — `Strategy`
  instances are shared read-only via `Arc` across parallel matches.
- A disabled modifier (`counting: false`, `denial: Off`,
  `deception_rate: 0.0`) must never draw from `rng` — this is what
  makes `Adaptive(none)`'s output byte-identical to plain `LowestLegal`
  for the same seed (verified by Task 7's equivalence tests).
- Task 1 is verified with `cargo test -p engine`. Tasks 2-7 are
  verified with `cargo test -p sim`. Task 8 touches `cli` and Task 9
  touches docs/integration; both use the full workspace gate
  (`cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D
  warnings`, `cargo test --workspace`).
- `rand::RngExt::random_bool(&mut self, p: f64) -> bool` is confirmed
  present in this workspace's pinned `rand 0.10.3` and object-safe for
  `&mut dyn rand::Rng` (a blanket `impl<R: Rng + ?Sized> RngExt for R
  {}` covers it) — verified directly against the installed crate
  source at `~/.cargo/registry/src/.../rand-0.10.3/src/rng.rs`, not
  assumed. `use rand::RngExt;` brings the method into scope.

## Review Focus

- **`ReadingParams`/`horizon` must NOT reappear.** An earlier design
  draft included a separate hand-reading-specific threshold distinct
  from `close`; the reconciled design drops it entirely
  (`DenialMode::HandReading { close: usize }` only). If Task 4's
  implementer (working from a design report rather than this plan)
  reintroduces it, that's a spec deviation to flag.
- **The `Denial` enum stays private to `adaptive::denial`.** Only
  `respond(...) -> Option<Move>` is `pub(super)`. A reviewer should
  check `Adaptive::choose_play` never matches on `Denial` variants
  directly — that would mean the module boundary leaked.
- **Suit precision in `hand_reading::PassCeilings`.** This project has
  now hit the same class of bug twice (Phase 6's final review caught
  `CardCounter` comparing by `Rank` instead of the full `Card::compare`
  total order). `PassCeilings` must store/compare `Card`s, never
  `Rank`s, and Task 2's "suit precision" test must genuinely fail if
  someone naively keys it by `Rank`.
- **Refutation ordering.** `read_pass_ceilings`'s reverse sweep depends
  on `plays_before` correctly distinguishing "this play happened before
  the pass" from "this play happened after the pass" — Task 1's test
  must include both orderings (a refuting play recorded after the pass,
  and a non-refuting play recorded before it) since an off-by-one here
  would silently produce correct-looking output on one ordering only.
- **Determinism**: no rng draw may happen when a modifier is disabled
  (Global Constraints). Task 7's equivalence tests are the enforcement
  mechanism for this — don't weaken or skip them under time pressure,
  they're the single most important tests in this plan.

---

### Task 1: `engine::Round` — pass history

**Files:**
- Modify: `engine/src/round.rs`

**Interfaces:**
- Produces: `pub fn Round::pass_history(&self) -> &[(SeatId, Combo,
  usize)]` (the `usize` is `play_history().len()` at the moment of that
  pass).
- Consumes: nothing new — `Combo` already derives `Clone`.

- [ ] **Step 1: Read the current file first**

Read `engine/src/round.rs` in full. Confirm `apply_pass`'s exact current
body (added last phase, should be exactly two lines: compute `active`,
then the `if let Some(new_leader) = self.trick.record_pass(...)` block)
and that `current_combo` is genuinely `Some` and unmodified at the top
of `apply_pass` — `submit_move`'s `CannotPassOnLead` check already
guarantees this, but verify it yourself rather than trusting this brief.

- [ ] **Step 2: Write the failing tests**

Add to `round.rs`'s existing `#[cfg(test)] mod tests` (reuse the
existing `card(rank, suit)`/`combo(cards)` helpers):

```rust
#[test]
fn pass_history_records_the_combo_and_play_count_at_pass_time() {
    let hands = vec![
        vec![card(Rank::Eight, Suit::Clubs), card(Rank::Ten, Suit::Clubs)],
        vec![card(Rank::Seven, Suit::Clubs), card(Rank::Six, Suit::Clubs)],
        vec![card(Rank::Nine, Suit::Clubs), card(Rank::Five, Suit::Clubs)],
    ];
    let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
    assert_eq!(round.pass_history(), &[]);

    let lead = combo(vec![card(Rank::Eight, Suit::Clubs)]);
    round.submit_move(0, Move::Play(lead.clone())).unwrap(); // play_history now len 1
    round.submit_move(1, Move::Pass).unwrap();
    assert_eq!(round.pass_history(), &[(1, lead.clone(), 1)]);

    let beat = combo(vec![card(Rank::Nine, Suit::Clubs)]);
    round.submit_move(2, Move::Play(beat)).unwrap(); // play_history now len 2
    // Trick resolves (both non-leaders acted); seat 0 leads again with
    // its remaining card.
    let second_lead = combo(vec![card(Rank::Ten, Suit::Clubs)]);
    round.submit_move(0, Move::Play(second_lead.clone())).unwrap(); // len 3
    round.submit_move(1, Move::Pass).unwrap();
    assert_eq!(
        round.pass_history(),
        &[(1, lead, 1), (1, second_lead, 3)],
        "each pass is tagged with play_history().len() at that exact moment"
    );
}
```

Double-check the exact turn order/trick-resolution mechanics against
this file's existing `everyone_passing_returns_the_lead_to_the_original_leader`
test before relying on the fixture above — adjust seat numbers/hand
contents if the real turn sequence differs from what's assumed here,
but keep the core assertion (two passes, with `plays_before` reflecting
the real play count at each).

- [ ] **Step 3: Run tests to verify they fail to compile**

Run: `cargo test -p engine`
Expected: compile error — `pass_history` doesn't exist yet.

- [ ] **Step 4: Add the field, accessor, and the one-line push**

Add `pass_history: Vec<(SeatId, Combo, usize)>` to the `Round` struct,
and `pass_history: Vec::new()` to `Round::new`'s constructor.

In `apply_pass`, add before the existing `if let Some(new_leader) = ...`
block:

```rust
if let Some(combo) = &self.current_combo {
    self.pass_history.push((seat, combo.clone(), self.play_history.len()));
}
```

Add the accessor:

```rust
/// Every pass this round, in order, tagged with the combo it declined
/// to beat and `play_history().len()` at that moment (so a later
/// reduction can tell whether this seat *subsequently* played
/// something that would have beaten it — see `sim::hand_reading`).
#[must_use]
pub fn pass_history(&self) -> &[(SeatId, Combo, usize)] {
    &self.pass_history
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p engine`
Expected: PASS — all pre-existing `Round` tests unchanged and green,
plus the new test.

- [ ] **Step 6: Lint and format**

Run: `cargo fmt -p engine` then `cargo clippy -p engine --all-targets -- -D warnings`

- [ ] **Step 7: Commit**

```bash
git add engine/src/round.rs
git commit -m "engine: track pass history on Round"
```

---

### Task 2: `sim::hand_reading` — pass-ceiling reduction (new module)

**Files:**
- Create: `sim/src/hand_reading.rs`
- Modify: `sim/src/lib.rs` (declare the module, re-export `PassCeilings`)

**Interfaces:**
- Produces: `pub struct PassCeilings` (with `ceiling`/`cannot_beat`
  methods), `pub fn read_pass_ceilings(player_count: usize,
  play_history: &[(SeatId, Combo)], pass_history: &[(SeatId, Combo,
  usize)], duplicate_rule: DuplicateRule) -> Vec<PassCeilings>`.
- Consumes: `engine::{Card, Combo, DuplicateRule, SeatId}`,
  `engine::Round::play_history()`/`pass_history()` (Task 1) — but only
  via whatever caller assembles these arguments (Task 3), not directly
  in this module.

- [ ] **Step 1: Write the failing tests**

Create `sim/src/hand_reading.rs` with its test module first (construct
`Card`/`Combo`/`SeatId` values using whatever helper idiom `engine`'s
own test modules use — check `engine/src/card.rs`'s or `round.rs`'s
`card(rank, suit)` pattern and mirror it here, or define a local
equivalent):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use engine::{Rank, Suit};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit, 0)
    }

    fn combo(cards: Vec<Card>) -> Combo {
        Combo::new(cards).unwrap()
    }

    #[test]
    fn a_pass_at_size_one_bounds_size_two_as_well() {
        let pass_history = vec![(0u8, combo(vec![card(Rank::Nine, Suit::Diamonds)]), 0)];
        let ceilings = read_pass_ceilings(2, &[], &pass_history, DuplicateRule::FirstDealtWins);
        let nine = card(Rank::Nine, Suit::Diamonds);
        assert_eq!(ceilings[0].ceiling(1, DuplicateRule::FirstDealtWins), Some(nine));
        assert_eq!(ceilings[0].ceiling(2, DuplicateRule::FirstDealtWins), Some(nine));
    }

    #[test]
    fn a_pass_at_size_two_says_nothing_about_size_one() {
        let pair = combo(vec![
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Seven, Suit::Diamonds),
        ]);
        let pass_history = vec![(0u8, pair, 0)];
        let ceilings = read_pass_ceilings(2, &[], &pass_history, DuplicateRule::FirstDealtWins);
        assert_eq!(ceilings[0].ceiling(1, DuplicateRule::FirstDealtWins), None);
    }

    #[test]
    fn suit_precision_a_pass_against_one_suit_does_not_cover_a_higher_suit_same_rank() {
        // engine's Suit order: Diamonds < Hearts < Spades < Clubs.
        let pass_history = vec![(0u8, combo(vec![card(Rank::King, Suit::Hearts)]), 0)];
        let ceilings = read_pass_ceilings(2, &[], &pass_history, DuplicateRule::FirstDealtWins);
        let king_hearts = card(Rank::King, Suit::Hearts);
        let king_clubs = card(Rank::King, Suit::Clubs);
        assert!(ceilings[0].cannot_beat(1, king_hearts, DuplicateRule::FirstDealtWins));
        assert!(
            !ceilings[0].cannot_beat(1, king_clubs, DuplicateRule::FirstDealtWins),
            "a same-rank higher-suit card must NOT be considered covered by this pass \
             (a Rank-keyed ceiling would wrongly say it is)"
        );
    }

    #[test]
    fn a_later_play_above_the_passed_card_refutes_the_pass() {
        let nine = combo(vec![card(Rank::Nine, Suit::Diamonds)]);
        let king = combo(vec![card(Rank::King, Suit::Clubs)]);
        let pass_history = vec![(0u8, nine, 0)]; // plays_before: 0
        let play_history_refuting = vec![(0u8, king.clone())]; // this play is index 0, so plays_before(0) <= 0 means it happened AFTER
        let ceilings = read_pass_ceilings(
            2, &play_history_refuting, &pass_history, DuplicateRule::FirstDealtWins,
        );
        assert_eq!(
            ceilings[0].ceiling(1, DuplicateRule::FirstDealtWins), None,
            "seat 0 later played a King, proving the earlier pass was not honest"
        );
    }

    #[test]
    fn a_play_recorded_before_the_pass_does_not_refute_it() {
        let king = combo(vec![card(Rank::King, Suit::Clubs)]);
        let nine = combo(vec![card(Rank::Nine, Suit::Diamonds)]);
        let play_history = vec![(0u8, king)]; // index 0, happened before
        let pass_history = vec![(0u8, nine.clone(), 1)]; // plays_before: 1 (i.e. after that one play)
        let ceilings = read_pass_ceilings(2, &play_history, &pass_history, DuplicateRule::FirstDealtWins);
        assert_eq!(
            ceilings[0].ceiling(1, DuplicateRule::FirstDealtWins),
            Some(card(Rank::Nine, Suit::Diamonds)),
            "the King was played BEFORE this pass, so it doesn't refute it"
        );
    }

    #[test]
    fn ceilings_are_tracked_independently_per_seat() {
        let pass_history = vec![
            (0u8, combo(vec![card(Rank::Nine, Suit::Diamonds)]), 0),
            (1u8, combo(vec![card(Rank::Six, Suit::Diamonds)]), 0),
        ];
        let ceilings = read_pass_ceilings(2, &[], &pass_history, DuplicateRule::FirstDealtWins);
        assert_eq!(
            ceilings[0].ceiling(1, DuplicateRule::FirstDealtWins),
            Some(card(Rank::Nine, Suit::Diamonds))
        );
        assert_eq!(
            ceilings[1].ceiling(1, DuplicateRule::FirstDealtWins),
            Some(card(Rank::Six, Suit::Diamonds))
        );
    }
}
```

- [ ] **Step 2: Run tests to verify they fail to compile**

Run: `cargo test -p sim --lib hand_reading`
Expected: fails — the module doesn't exist yet.

- [ ] **Step 3: Implement `PassCeilings` and `read_pass_ceilings`**

Above the test module:

```rust
//! Pass-based hand reading (docs/ROADMAP.md, Phase 7).
//!
//! A seat that passed against a size-`s` combo topped by `c` held no
//! size-`s` combo topped above `c` at that moment; hands only shrink
//! (no draw pile), so that stays true for the rest of the round. By
//! upward closure it also holds no *larger* combo topped above `c` (any
//! such combo contains a size-`s` sub-combo with the same top card), so
//! a seat's ceiling at size `s` is the lowest top card it has passed
//! against at any size `<= s`. Ceilings are `Card`s, not `Rank`s:
//! passing on 9♦ says nothing about 9♣ — the same lesson `CardCounter`
//! learned the hard way (see its own doc comment).
//!
//! **Refutation.** Not every pass is honest — `HoldBackPairs` passes on
//! a single when every beating play would split a pair, `RandomLegal`
//! passes at random, and `Adaptive`'s deception modifier passes on
//! purpose. A pass is dropped (never recorded, or overwritten) once the
//! same seat later plays a combo of the same or larger size topped
//! above the passed-on card — proof it held a beater at the time. An
//! honest pass can never be refuted, since hands only shrink; this only
//! ever removes false information, never true information.

use std::cmp::Ordering;

use engine::{Card, Combo, DuplicateRule, SeatId};

/// The largest combo any seat can field: 8 copies of one rank in the
/// double deck.
pub const MAX_COMBO_SIZE: usize = 8;

/// Per-size "can't beat" facts about one seat, derived from its
/// unrefuted passes. `by_size[s - 1]` is the lowest top card passed
/// against at exactly size `s`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PassCeilings {
    by_size: [Option<Card>; MAX_COMBO_SIZE],
}

impl PassCeilings {
    /// The lowest top card this seat is known unable to beat at
    /// `size`, applying upward closure. `None`: nothing is known at
    /// this size.
    #[must_use]
    pub fn ceiling(&self, size: usize, duplicate_rule: DuplicateRule) -> Option<Card> {
        self.by_size
            .iter()
            .take(size)
            .flatten()
            .copied()
            .min_by(|a, b| a.compare(b, duplicate_rule))
    }

    /// Whether this seat is known unable to beat a size-`size` combo
    /// topped by `top`.
    #[must_use]
    pub fn cannot_beat(&self, size: usize, top: Card, duplicate_rule: DuplicateRule) -> bool {
        self.ceiling(size, duplicate_rule)
            .is_some_and(|c| top.compare(&c, duplicate_rule) != Ordering::Less)
    }
}

/// Reduces a round's play/pass history to one `PassCeilings` per seat,
/// in a single reverse sweep: walking backward, `highest_later[seat][s
/// - 1]` is the highest top card `seat` played at size `>= s` *after*
/// the pass currently being examined — exactly the refutation test.
///
/// `pass_history` entries are `(seat, combo, plays_before)` exactly as
/// `engine::Round::pass_history()` returns them (`plays_before` is
/// `play_history.len()` at the moment of that pass).
#[must_use]
pub fn read_pass_ceilings(
    player_count: usize,
    play_history: &[(SeatId, Combo)],
    pass_history: &[(SeatId, Combo, usize)],
    duplicate_rule: DuplicateRule,
) -> Vec<PassCeilings> {
    let mut ceilings = vec![PassCeilings::default(); player_count];
    let mut highest_later = vec![[None::<Card>; MAX_COMBO_SIZE]; player_count];
    let mut play_index = play_history.len();

    for (seat, combo, plays_before) in pass_history.iter().rev() {
        while play_index > *plays_before {
            play_index -= 1;
            let (play_seat, play_combo) = &play_history[play_index];
            let top = play_combo.top_card(duplicate_rule);
            for slot in &mut highest_later[usize::from(*play_seat)][..play_combo.size()] {
                if slot.is_none_or(|h| top.compare(&h, duplicate_rule) == Ordering::Greater) {
                    *slot = Some(top);
                }
            }
        }

        let seat_idx = usize::from(*seat);
        let size = combo.size();
        let passed_top = combo.top_card(duplicate_rule);
        let refuted = highest_later[seat_idx][size - 1]
            .is_some_and(|h| h.compare(&passed_top, duplicate_rule) == Ordering::Greater);
        if !refuted {
            let slot = &mut ceilings[seat_idx].by_size[size - 1];
            if slot.is_none_or(|c| passed_top.compare(&c, duplicate_rule) == Ordering::Less) {
                *slot = Some(passed_top);
            }
        }
    }
    ceilings
}
```

(This uses a plain index counter, `play_index`, instead of a
`Peekable` iterator to avoid the borrow/pattern-matching subtleties of
iterating `play_history` while also indexing into `highest_later` — get
this compiling as written or restructure minimally if it doesn't;
either way the five tests above are the actual contract.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p sim --lib hand_reading`
Expected: PASS, all 6 tests green.

- [ ] **Step 5: Declare and re-export the module**

In `sim/src/lib.rs`, add `mod hand_reading;` and `pub use hand_reading::
PassCeilings;` (re-export `read_pass_ceilings` too if this file's
existing convention re-exports free functions alongside types — check
how `exchange`/`assign_roles`-style functions are re-exported elsewhere
in this codebase's `lib.rs` files and match it; if in doubt, re-export
both).

- [ ] **Step 6: Run the full `sim` suite, lint and format**

Run: `cargo test -p sim`, then `cargo fmt -p sim`, then `cargo clippy -p sim --all-targets -- -D warnings`

- [ ] **Step 7: Commit**

```bash
git add sim/src/hand_reading.rs sim/src/lib.rs
git commit -m "sim: add hand_reading, a pure reduction of pass history into per-seat ceilings"
```

---

### Task 3: `TurnContext`/`OpponentHand` extension + `Strategy::name()` signature change + migrate all 9 implementors

**Files:**
- Modify: `sim/src/strategy.rs`
- Modify: `sim/src/match_runner.rs`
- Modify: `sim/src/strategies/lowest_legal.rs`
- Modify: `sim/src/strategies/greedy_highest.rs`
- Modify: `sim/src/strategies/random_legal.rs`
- Modify: `sim/src/strategies/hold_back_pairs.rs`
- Modify: `sim/src/strategies/card_counter.rs`
- Modify: `sim/src/strategies/endgame_denial.rs`

**Interfaces:**
- Consumes: `PassCeilings`/`read_pass_ceilings` (Task 2).
- Produces: `OpponentHand.pass_ceilings: PassCeilings`,
  `TurnContext.own_pass_ceilings: PassCeilings`,
  `TurnContext.current_combo: Option<&'a Combo>`,
  `Strategy::name(&self) -> &str` (was `&'static str`).

**Why this is one task, not several:** adding fields to `TurnContext`
breaks every existing `TurnContext { .. }` struct literal (in
`match_runner.rs`'s context-assembly helper and every strategy file's
own test-only `empty_context()`-style helper), and changing `name()`'s
return type breaks every `impl Strategy`. Both changes touch the same
broad file set for the same reason — splitting them would leave an
intermediate uncompilable state, the exact mistake this project's Phase
6 planning already had to catch and fix once for a different pair of
changes.

- [ ] **Step 1: Read the current files first**

Read `sim/src/strategy.rs`, `sim/src/match_runner.rs` (both `run_match`
and its context-assembly helper — find its actual current name, it may
not be exactly `turn_context_for`), and all six strategy files listed
above, in full.

- [ ] **Step 2: Extend `OpponentHand`/`TurnContext` and change `name()`'s signature**

In `sim/src/strategy.rs`:

```rust
use crate::hand_reading::PassCeilings;

#[derive(Debug, Clone, Copy)]
pub struct OpponentHand {
    pub seat: SeatId,
    pub hand_size: usize,
    pub active: bool,
    pub pass_ceilings: PassCeilings,
}

pub struct TurnContext<'a> {
    pub seat: SeatId,
    pub hand: &'a [Card],
    pub opponents: Vec<OpponentHand>,
    pub unseen_cards: Vec<Card>,
    /// This seat's own pass ceilings, as read by anyone else — needed
    /// by `Adaptive`'s deception modifier (a later task) to avoid a
    /// redundant bluff.
    pub own_pass_ceilings: PassCeilings,
    /// The combo currently on the table, or `None` if this seat must
    /// lead (mirrors `engine::Round::current_combo`).
    pub current_combo: Option<&'a Combo>,
}

pub trait Strategy: Send + Sync {
    /// A short, stable name used to group results by strategy. Owned
    /// content, not necessarily `'static` — a configurable strategy's
    /// name reflects its actual configuration.
    fn name(&self) -> &str;
    // choose_play/choose_exchange_cards signatures unchanged
}
```

(Add `Combo` to this file's `use engine::{...};` import if not already
present.)

- [ ] **Step 3: Run tests to confirm the expected build break**

Run: `cargo test -p sim`
Expected: many compile errors — every `impl Strategy` block's `name()`
no longer matches, and every `TurnContext { .. }`/`OpponentHand { .. }`
struct literal is missing fields. This confirms both changes took
effect.

- [ ] **Step 4: Fix `name()` in all 6 real strategy files**

In `lowest_legal.rs`, `greedy_highest.rs`, `random_legal.rs`,
`hold_back_pairs.rs`, `card_counter.rs`, `endgame_denial.rs`: change
`fn name(&self) -> &'static str` to `fn name(&self) -> &str`. Bodies are
unchanged (a `&'static str` literal already coerces to `&str` fine).

- [ ] **Step 5: Update `match_runner.rs`'s context-assembly helper and its 3 test-only `Strategy` impls**

Update the 3 test-only `Strategy` implementors inside `match_runner.rs`'s
own `#[cfg(test)] mod tests` the same way (`name()` signature).

In the production context-assembly helper (wherever `TurnContext { .. }`
is built each turn), add the ceiling computation and thread the two new
fields through:

```rust
let all_ceilings = crate::hand_reading::read_pass_ceilings(
    usize::from(config.player_count),
    round.play_history(),
    round.pass_history(),
    config.duplicate_rule,
);
let opponents: Vec<OpponentHand> = (0..config.player_count)
    .filter(|&s| s != seat)
    .map(|s| OpponentHand {
        seat: s,
        hand_size: round.hand_size(s),
        active: round.is_active(s),
        pass_ceilings: all_ceilings[usize::from(s)],
    })
    .collect();
let own_pass_ceilings = all_ceilings[usize::from(seat)];
let current_combo = round.current_combo();
```

...then add `own_pass_ceilings`, `current_combo` to the `TurnContext {
.. }` literal alongside the existing fields, and `pass_ceilings:
all_ceilings[...]` to each `OpponentHand { .. }` literal (as shown
above).

- [ ] **Step 6: Fix every `empty_context()`-style test helper**

In each of the 6 strategy files' test modules, extend their
`TurnContext`/`OpponentHand` literal-construction helper with
`own_pass_ceilings: PassCeilings::default()`, `current_combo: None`, and
`pass_ceilings: PassCeilings::default()` on any `OpponentHand` literal.
Do the same for `match_runner.rs`'s own test module wherever it builds
`TurnContext`/`OpponentHand` literals directly (not through the
production helper).

- [ ] **Step 7: Write a wiring-correctness test**

Add one test in `match_runner.rs`'s test module — a scripted multi-round
match where at least one seat passes on a size-1 combo and later the
opposing seat's `OpponentHand.pass_ceilings` (as seen by another seat's
`choose_play`, captured via a test-only `Strategy` that records what it
was given) reflects a non-default `PassCeilings`. This is a regression
guard for the *wiring*, not just each piece standalone — Phase 6's
final review specifically flagged the value of this kind of test.

- [ ] **Step 8: Run tests to verify everything compiles and passes**

Run: `cargo test -p sim`
Expected: PASS — every pre-existing test across all 7 modified files
green, plus the new wiring test.

- [ ] **Step 9: Lint and format**

Run: `cargo fmt -p sim` then `cargo clippy -p sim --all-targets -- -D warnings`

- [ ] **Step 10: Commit**

```bash
git add sim/src/strategy.rs sim/src/match_runner.rs sim/src/strategies/lowest_legal.rs sim/src/strategies/greedy_highest.rs sim/src/strategies/random_legal.rs sim/src/strategies/hold_back_pairs.rs sim/src/strategies/card_counter.rs sim/src/strategies/endgame_denial.rs
git commit -m "sim: extend TurnContext with pass ceilings; Strategy::name() returns &str"
```

---

### Task 4: `sim::strategies::adaptive::config` — `AdaptiveConfig` (new module)

**Files:**
- Create: `sim/src/strategies/adaptive/config.rs`
- Create: `sim/src/strategies/adaptive/mod.rs` (module declaration only
  for now — `mod config; pub use config::{AdaptiveConfig, DenialMode};`
  — Task 7 fills in the rest of this file)
- Modify: `sim/src/strategies/mod.rs` (`mod adaptive; pub use
  adaptive::{Adaptive, AdaptiveConfig, DenialMode};` — the `Adaptive`
  re-export will forward-reference Task 7's not-yet-written struct;
  either stub a minimal placeholder `Adaptive` here that Task 7
  replaces, or hold this line back until Task 7 — pick whichever keeps
  this task's own tests compiling cleanly and say which you chose in
  your report)

**Interfaces:**
- Produces: `AdaptiveConfig`, `DenialMode`, `AdaptiveConfigError`, plus
  `Display`/`FromStr` impls for `AdaptiveConfig`.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing tests**

Add to `config.rs`'s own `#[cfg(test)] mod tests`:

```rust
#[test]
fn display_then_parse_round_trips_for_representative_configs() {
    let configs = [
        AdaptiveConfig::NONE,
        AdaptiveConfig::default(),
        AdaptiveConfig { counting: true, denial: DenialMode::Off, deception_rate: 0.0 },
        AdaptiveConfig { counting: false, denial: DenialMode::HandSize { close: 2 }, deception_rate: 0.0 },
        AdaptiveConfig { counting: false, denial: DenialMode::HandSize { close: 3 }, deception_rate: 0.0 },
        AdaptiveConfig { counting: false, denial: DenialMode::HandReading { close: 2 }, deception_rate: 0.0 },
        AdaptiveConfig { counting: true, denial: DenialMode::Off, deception_rate: 0.2 },
    ];
    for cfg in configs {
        let rendered = cfg.to_string();
        let parsed: AdaptiveConfig = rendered.parse().unwrap_or_else(|e| {
            panic!("failed to parse rendered config {rendered:?}: {e}")
        });
        assert_eq!(parsed, cfg, "round-trip mismatch for {rendered:?}");
    }
}

#[test]
fn parses_none_and_bare_flags() {
    assert_eq!("none".parse::<AdaptiveConfig>().unwrap(), AdaptiveConfig::NONE);
    assert_eq!(
        "counting".parse::<AdaptiveConfig>().unwrap(),
        AdaptiveConfig { counting: true, denial: DenialMode::Off, deception_rate: 0.0 }
    );
}

#[test]
fn reading_implies_denial() {
    let cfg: AdaptiveConfig = "reading".parse().unwrap();
    assert!(matches!(cfg.denial, DenialMode::HandReading { .. }));
}

#[test]
fn rejects_empty_option_list() {
    assert!("".parse::<AdaptiveConfig>().is_err());
}

#[test]
fn rejects_duplicate_key() {
    assert!("counting,counting".parse::<AdaptiveConfig>().is_err());
}

#[test]
fn rejects_unknown_key() {
    assert!("bogus".parse::<AdaptiveConfig>().is_err());
}

#[test]
fn rejects_flag_given_a_value() {
    assert!("counting=1".parse::<AdaptiveConfig>().is_err());
}

#[test]
fn rejects_parameter_given_no_value() {
    assert!("deception".parse::<AdaptiveConfig>().is_err());
    assert!("close".parse::<AdaptiveConfig>().is_err());
}

#[test]
fn rejects_out_of_range_deception_rate() {
    assert!("deception=1.5".parse::<AdaptiveConfig>().is_err());
    assert!("deception=-0.1".parse::<AdaptiveConfig>().is_err());
    assert!("deception=nan".parse::<AdaptiveConfig>().is_err());
}

#[test]
fn rejects_unparseable_numbers() {
    assert!("close=abc".parse::<AdaptiveConfig>().is_err());
    assert!("deception=abc".parse::<AdaptiveConfig>().is_err());
}

#[test]
fn rejects_close_without_denial_or_reading() {
    assert!("close=3".parse::<AdaptiveConfig>().is_err());
    assert!("counting,close=3".parse::<AdaptiveConfig>().is_ok().then_some(()).is_none()
        || "counting,close=3".parse::<AdaptiveConfig>().is_err());
}

#[test]
fn rejects_close_of_zero() {
    assert!("denial,close=0".parse::<AdaptiveConfig>().is_err());
}
```

(The double-check in `rejects_close_without_denial_or_reading`'s second
assertion is redundant with the first line's intent — simplify it to a
single clean `assert!("counting,close=3".parse::<AdaptiveConfig>().is_err());`
when you write this for real; it's written awkwardly above only because
this brief is being cautious about not over-specifying exact `Result`
shapes.)

- [ ] **Step 2: Run tests to verify they fail to compile**

Run: `cargo test -p sim --lib adaptive::config`
Expected: fails — the module doesn't exist yet.

- [ ] **Step 3: Implement `AdaptiveConfig`**

```rust
// sim/src/strategies/adaptive/config.rs
use std::fmt;
use std::str::FromStr;

pub const DEFAULT_CLOSE: usize = 2; // matches EndgameDenial::CLOSE_TO_FINISHING

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DenialMode {
    Off,
    HandSize { close: usize },
    HandReading { close: usize },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveConfig {
    pub counting: bool,
    pub denial: DenialMode,
    pub deception_rate: f64,
}

impl AdaptiveConfig {
    pub const NONE: Self = Self { counting: false, denial: DenialMode::Off, deception_rate: 0.0 };

    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.deception_rate.is_finite() && (0.0..=1.0).contains(&self.deception_rate)
    }
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            counting: true,
            denial: DenialMode::HandReading { close: DEFAULT_CLOSE },
            deception_rate: 0.0,
        }
    }
}

impl fmt::Display for AdaptiveConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut items: Vec<String> = Vec::new();
        if self.counting {
            items.push("counting".into());
        }
        let close_item = |c: usize| (c != DEFAULT_CLOSE).then(|| format!("close={c}"));
        match self.denial {
            DenialMode::Off => {}
            DenialMode::HandSize { close } => {
                items.push("denial".into());
                items.extend(close_item(close));
            }
            DenialMode::HandReading { close } => {
                items.push("reading".into());
                items.extend(close_item(close));
            }
        }
        if self.deception_rate > 0.0 {
            items.push(format!("deception={}", self.deception_rate));
        }
        if items.is_empty() {
            f.write_str("none")
        } else {
            f.write_str(&items.join(","))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveConfigError(String);

impl fmt::Display for AdaptiveConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for AdaptiveConfigError {}

impl FromStr for AdaptiveConfig {
    type Err = AdaptiveConfigError;

    fn from_str(options: &str) -> Result<Self, Self::Err> {
        let err = |msg: String| AdaptiveConfigError(msg);
        if options.trim().is_empty() {
            return Err(err("empty option list (use `none` for no modifiers)".into()));
        }
        if options.trim() == "none" {
            return Ok(Self::NONE);
        }

        let mut counting = false;
        let mut reading = false;
        let mut denial = false;
        let mut close: Option<usize> = None;
        let mut deception_rate = 0.0f64;
        let mut seen_keys: Vec<&str> = Vec::new();

        for item in options.split(',') {
            let item = item.trim();
            if item.is_empty() {
                return Err(err("empty option between commas".into()));
            }
            let (key, value) = match item.split_once('=') {
                Some((k, v)) => (k.trim(), Some(v.trim())),
                None => (item, None),
            };
            if seen_keys.contains(&key) {
                return Err(err(format!("option `{key}` given more than once")));
            }
            seen_keys.push(key);

            match (key, value) {
                ("counting", None) => counting = true,
                ("denial", None) => denial = true,
                ("reading", None) => reading = true,
                ("counting" | "denial" | "reading", Some(_)) => {
                    return Err(err(format!("`{key}` doesn't take a value")));
                }
                ("close", Some(v)) => {
                    let parsed = v.parse::<usize>().map_err(|_| {
                        err(format!("`close` must be a whole number >= 1, got `{v}`"))
                    })?;
                    if parsed == 0 {
                        return Err(err("`close` must be >= 1".into()));
                    }
                    close = Some(parsed);
                }
                ("deception", Some(v)) => {
                    let rate: f64 = v
                        .parse()
                        .map_err(|_| err(format!("`deception` must be a number, got `{v}`")))?;
                    if !rate.is_finite() || !(0.0..=1.0).contains(&rate) {
                        return Err(err(format!("`deception` must be in [0, 1], got `{v}`")));
                    }
                    deception_rate = rate;
                }
                ("close" | "deception", None) => {
                    return Err(err(format!("`{key}` requires a value, e.g. `{key}=...`")));
                }
                (other, _) => {
                    return Err(err(format!(
                        "unknown option `{other}` (expected one of: counting, denial, \
                         reading, deception=<rate>, close=<n>)"
                    )));
                }
            }
        }

        if close.is_some() && !denial && !reading {
            return Err(err("`close` requires `denial` or `reading` to also be set".into()));
        }

        let denial_mode = if reading {
            DenialMode::HandReading { close: close.unwrap_or(DEFAULT_CLOSE) }
        } else if denial {
            DenialMode::HandSize { close: close.unwrap_or(DEFAULT_CLOSE) }
        } else {
            DenialMode::Off
        };

        Ok(Self { counting, denial: denial_mode, deception_rate })
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p sim --lib adaptive::config`
Expected: PASS, all tests green.

- [ ] **Step 5: Wire up the module declarations**

Create `sim/src/strategies/adaptive/mod.rs`:

```rust
mod config;

pub use config::{AdaptiveConfig, DenialMode};
```

In `sim/src/strategies/mod.rs`, add `mod adaptive;` and `pub use
adaptive::{AdaptiveConfig, DenialMode};` (hold back the `Adaptive`
struct re-export itself until Task 7, when it exists — don't stub a
placeholder, since that risks diverging from Task 7's real design;
just don't re-export a name that doesn't exist yet).

- [ ] **Step 6: Run the full `sim` suite, lint and format**

Run: `cargo test -p sim`, then `cargo fmt -p sim`, then `cargo clippy -p sim --all-targets -- -D warnings`

- [ ] **Step 7: Commit**

```bash
git add sim/src/strategies/adaptive/ sim/src/strategies/mod.rs
git commit -m "sim: add AdaptiveConfig with its Display/FromStr grammar"
```

---

### Task 5: `sim::strategies::adaptive::denial` — targeted endgame denial

**Files:**
- Create: `sim/src/strategies/adaptive/denial.rs`
- Modify: `sim/src/strategies/adaptive/mod.rs` (`mod denial;`)

**Interfaces:**
- Consumes: `TurnContext`/`OpponentHand.pass_ceilings` (Task 3),
  `DenialMode` (Task 4), `GreedyHighest`/`LowestLegal` (existing,
  unmodified — direct delegation, per Global Constraints).
- Produces: `pub(super) fn respond(mode: DenialMode, legal_moves:
  &[Move], duplicate_rule: DuplicateRule, context: &TurnContext<'_>,
  rng: &mut dyn rand::Rng) -> Option<Move>`.

- [ ] **Step 1: Write the failing tests**

Add to `denial.rs`'s own test module (construct `TurnContext`/
`OpponentHand` fixtures directly as struct literals, matching the
established idiom from `sim/src/strategies/card_counter.rs`'s/
`endgame_denial.rs`'s own test modules — reuse whatever `card(rank,
suit)` helper they use):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use engine::{Combo, DuplicateRule, Move, Rank, Suit};
    use crate::hand_reading::PassCeilings;
    use crate::strategy::OpponentHand;

    fn card(rank: Rank, suit: Suit) -> engine::Card {
        engine::Card::new(rank, suit, 0)
    }

    fn play(cards: Vec<engine::Card>) -> Move {
        Move::Play(Combo::new(cards).unwrap())
    }

    fn ceilings_from(passes: &[(usize, engine::Card)]) -> PassCeilings {
        let pass_history: Vec<_> = passes
            .iter()
            .map(|&(size, top)| {
                let cards = vec![top; size]; // same-rank filler for size >1; size-1 tests only need one card
                (0u8, Combo::new(cards).unwrap(), 0usize)
            })
            .collect();
        crate::hand_reading::read_pass_ceilings(2, &[], &pass_history, DuplicateRule::FirstDealtWins)[0]
    }

    fn context_with_threat(hand_size: usize, pass_ceilings: PassCeilings, unseen: Vec<engine::Card>) -> TurnContext<'static> {
        TurnContext {
            seat: 0,
            hand: &[],
            opponents: vec![OpponentHand { seat: 1, hand_size, active: true, pass_ceilings }],
            unseen_cards: unseen,
            own_pass_ceilings: PassCeilings::default(),
            current_combo: None,
        }
    }

    #[test]
    fn a_reads_the_ceiling_and_locks_with_the_cheaper_card() {
        let ceilings = ceilings_from(&[(1, card(Rank::Jack, Suit::Spades))]);
        let context = context_with_threat(2, ceilings, vec![card(Rank::Queen, Suit::Diamonds), card(Rank::Ace, Suit::Diamonds)]);
        let legal = vec![Move::Pass, play(vec![card(Rank::King, Suit::Hearts)]), play(vec![card(Rank::Ace, Suit::Clubs)])];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let reading = respond(DenialMode::HandReading { close: 2 }, &legal, DuplicateRule::FirstDealtWins, &context, &mut rng);
        assert_eq!(reading, Some(play(vec![card(Rank::King, Suit::Hearts)])));

        let hand_size_mode = respond(DenialMode::HandSize { close: 2 }, &legal, DuplicateRule::FirstDealtWins, &context, &mut rng);
        assert_eq!(hand_size_mode, Some(play(vec![card(Rank::Ace, Suit::Clubs)])), "HandSize must reproduce EndgameDenial's own GreedyHighest push, diverging from HandReading here");
    }

    #[test]
    fn b_no_pass_history_falls_back_to_the_only_unbeatable_play() {
        let context = context_with_threat(2, PassCeilings::default(), vec![card(Rank::Ace, Suit::Diamonds)]);
        let legal = vec![Move::Pass, play(vec![card(Rank::King, Suit::Hearts)]), play(vec![card(Rank::Ace, Suit::Clubs)])];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let result = respond(DenialMode::HandReading { close: 2 }, &legal, DuplicateRule::FirstDealtWins, &context, &mut rng);
        assert_eq!(result, Some(play(vec![card(Rank::Ace, Suit::Clubs)])));
    }

    #[test]
    fn c_suit_precision_in_the_lock_check() {
        let ceilings = ceilings_from(&[(1, card(Rank::King, Suit::Hearts))]);
        let context = context_with_threat(1, ceilings, vec![card(Rank::Ace, Suit::Diamonds)]);
        let legal = vec![
            play(vec![card(Rank::Five, Suit::Diamonds)]),
            play(vec![card(Rank::King, Suit::Diamonds)]),
            play(vec![card(Rank::King, Suit::Spades)]),
        ];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let result = respond(DenialMode::HandReading { close: 2 }, &legal, DuplicateRule::FirstDealtWins, &context, &mut rng);
        assert_eq!(
            result,
            Some(play(vec![card(Rank::King, Suit::Spades)])),
            "King of Diamonds must NOT be treated as locking-out (ceiling is King of Hearts, \
             and Diamonds < Hearts in this game's suit order, so King-Diamonds does not beat it)"
        );
    }

    #[test]
    fn d_size_lockout_with_no_pass_history() {
        let context = context_with_threat(1, PassCeilings::default(), vec![card(Rank::Ace, Suit::Diamonds)]);
        let legal = vec![
            play(vec![card(Rank::Four, Suit::Diamonds)]),
            play(vec![card(Rank::Six, Suit::Clubs), card(Rank::Six, Suit::Diamonds)]),
            play(vec![card(Rank::Jack, Suit::Clubs), card(Rank::Jack, Suit::Diamonds)]),
        ];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let result = respond(DenialMode::HandReading { close: 2 }, &legal, DuplicateRule::FirstDealtWins, &context, &mut rng);
        assert_eq!(
            result,
            Some(play(vec![card(Rank::Six, Suit::Clubs), card(Rank::Six, Suit::Diamonds)])),
            "threat has only 1 card, so it can't field any size-2 combo at all — \
             the lowest size-2 play already locks it out"
        );
    }

    #[test]
    fn no_active_threat_falls_through_to_the_caller() {
        let context = context_with_threat(10, PassCeilings::default(), vec![]);
        let legal = vec![play(vec![card(Rank::Six, Suit::Clubs)])];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        assert_eq!(respond(DenialMode::HandReading { close: 2 }, &legal, DuplicateRule::FirstDealtWins, &context, &mut rng), None);
        assert_eq!(respond(DenialMode::HandSize { close: 2 }, &legal, DuplicateRule::FirstDealtWins, &context, &mut rng), None);
    }

    #[test]
    fn denial_off_always_returns_none() {
        let context = context_with_threat(1, PassCeilings::default(), vec![]);
        let legal = vec![play(vec![card(Rank::Six, Suit::Clubs)])];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        assert_eq!(respond(DenialMode::Off, &legal, DuplicateRule::FirstDealtWins, &context, &mut rng), None);
    }
}
```

Add `use rand::SeedableRng;` as needed. Double-check every fixture's
`unseen_cards`/`pass_ceilings` combination against `targeted_denial`'s
actual logic by hand before trusting the expected outcomes above — they
were hand-traced during planning, but re-verify against your own
implementation as you write it, since a subtle fixture error here would
be easy to miss (this is the single most algorithmically dense task in
the plan).

- [ ] **Step 2: Run tests to verify they fail to compile**

Run: `cargo test -p sim --lib adaptive::denial`
Expected: fails — the module/function don't exist yet.

- [ ] **Step 3: Implement `targeted_denial` and `respond`**

```rust
// sim/src/strategies/adaptive/denial.rs
//! Endgame denial, in both of `Adaptive`'s modes.
//!
//! `HandSize` reproduces `EndgameDenial`'s exact trigger and response
//! (any active opponent's hand_size <= `close` -> full GreedyHighest
//! push, else caller falls through to base selection) by direct
//! delegation — no logic duplicated, no behavior drift, so
//! `Adaptive(denial)` stays an honest baseline comparison against
//! `EndgameDenial` itself.
//!
//! `HandReading` asks a narrower question than a blanket push: which
//! legal plays can a close-to-finishing opponent *provably* not beat,
//! using their unrefuted pass ceilings plus two other sound public
//! facts (the combo is bigger than their whole hand; no unseen card
//! beats it at all)? Taking the *lowest* such play denies the threat as
//! surely as a highest-card push while spending less — the working
//! theory (docs/ROADMAP.md, Phase 7) is that `EndgameDenial` loses to
//! `LowestLegal` precisely because it spends high cards it doesn't
//! need to. When following and the table combo is already provably
//! safe against every threat, every legal beater qualifies, so this
//! reduces to `LowestLegal`'s own choice: aggression relaxes exactly
//! when it stops being necessary. Only falls back to a full push
//! (`GreedyHighest`) when nothing is provably safe.

use std::cmp::Ordering;

use engine::{Card, DuplicateRule, Move};

use crate::strategies::{GreedyHighest, LowestLegal};
use crate::strategy::{Strategy, TurnContext};

use super::config::DenialMode;

fn any_threat(context: &TurnContext<'_>, close: usize) -> bool {
    context.opponents.iter().any(|o| o.active && o.hand_size <= close)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Denial {
    NoThreat,
    Lock(Move),
    Unproven,
}

fn targeted_denial(
    legal_moves: &[Move],
    duplicate_rule: DuplicateRule,
    context: &TurnContext<'_>,
    close: usize,
) -> Denial {
    let threats: Vec<_> = context
        .opponents
        .iter()
        .filter(|o| o.active && o.hand_size <= close)
        .collect();
    if threats.is_empty() {
        return Denial::NoThreat;
    }

    let highest_unseen = context
        .unseen_cards
        .iter()
        .copied()
        .max_by(|a, b| a.compare(b, duplicate_rule));
    let locks_out_every_threat = |size: usize, top: Card| {
        highest_unseen.is_none_or(|h| top.compare(&h, duplicate_rule) != Ordering::Less)
            || threats.iter().all(|t| {
                size > t.hand_size || t.pass_ceilings.cannot_beat(size, top, duplicate_rule)
            })
    };

    legal_moves
        .iter()
        .filter_map(|mv| match mv {
            Move::Play(combo) => Some((combo.size(), combo.top_card(duplicate_rule), mv)),
            Move::Pass => None,
        })
        .filter(|&(size, top, _)| locks_out_every_threat(size, top))
        .min_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.compare(&b.1, duplicate_rule)))
        .map_or(Denial::Unproven, |(_, _, mv)| Denial::Lock(mv.clone()))
}

pub(super) fn respond(
    mode: DenialMode,
    legal_moves: &[Move],
    duplicate_rule: DuplicateRule,
    context: &TurnContext<'_>,
    rng: &mut dyn rand::Rng,
) -> Option<Move> {
    match mode {
        DenialMode::Off => None,
        DenialMode::HandSize { close } => {
            if any_threat(context, close) {
                Some(GreedyHighest.choose_play(legal_moves, duplicate_rule, context, rng))
            } else {
                None
            }
        }
        DenialMode::HandReading { close } => {
            match targeted_denial(legal_moves, duplicate_rule, context, close) {
                Denial::NoThreat => None,
                Denial::Lock(mv) => Some(mv),
                Denial::Unproven => {
                    Some(GreedyHighest.choose_play(legal_moves, duplicate_rule, context, rng))
                }
            }
        }
    }
}
```

(`LowestLegal` is imported but not directly called in this file —
`respond`'s `None` return is what causes `Adaptive` (Task 7) to fall
through to its own `LowestLegal`/`CardCounter` base selection. Drop the
unused import if clippy flags it, or keep it only if some path here
does end up needing it once you're implementing for real.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p sim --lib adaptive::denial`
Expected: PASS, all 6 tests green.

- [ ] **Step 5: Declare the module**

In `sim/src/strategies/adaptive/mod.rs`, add `mod denial;`.

- [ ] **Step 6: Run the full `sim` suite, lint and format**

Run: `cargo test -p sim`, then `cargo fmt -p sim`, then `cargo clippy -p sim --all-targets -- -D warnings`

- [ ] **Step 7: Commit**

```bash
git add sim/src/strategies/adaptive/denial.rs sim/src/strategies/adaptive/mod.rs
git commit -m "sim: add adaptive::denial, targeted endgame denial using pass ceilings"
```

---

### Task 6: `sim::strategies::adaptive::deception` — bluff-passing

**Files:**
- Create: `sim/src/strategies/adaptive/deception.rs`
- Modify: `sim/src/strategies/adaptive/mod.rs` (`mod deception;`)

**Interfaces:**
- Consumes: `TurnContext` (Task 3), `config::DEFAULT_CLOSE` (Task 4).
- Produces: `pub(super) fn should_bluff_pass(rate: f64, legal_moves:
  &[Move], duplicate_rule: DuplicateRule, context: &TurnContext<'_>,
  rng: &mut dyn rand::Rng) -> bool`.

- [ ] **Step 1: Write the failing tests**

Add to `deception.rs`'s own test module (reuse the `card`/`play`/
`TurnContext`-construction idiom from Task 5's `denial.rs` tests):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use engine::{Combo, DuplicateRule, Move, Rank, Suit};
    use rand::SeedableRng;
    use crate::hand_reading::{read_pass_ceilings, PassCeilings};
    use crate::strategy::OpponentHand;

    fn card(rank: Rank, suit: Suit) -> engine::Card {
        engine::Card::new(rank, suit, 0)
    }
    fn play(cards: Vec<engine::Card>) -> Move {
        Move::Play(Combo::new(cards).unwrap())
    }
    fn ten_diamonds() -> Combo {
        Combo::new(vec![card(Rank::Ten, Suit::Diamonds)]).unwrap()
    }

    /// `combo`: the combo currently on the table (this seat is
    /// following it). Takes it by reference so callers can vary
    /// `current_combo`/`own_pass_ceilings` independently per test.
    fn qualifying_context(combo: &Combo, own_pass_ceilings: PassCeilings) -> TurnContext<'_> {
        TurnContext {
            seat: 0,
            hand: &[
                card(Rank::Two, Suit::Clubs), card(Rank::Three, Suit::Clubs),
                card(Rank::Four, Suit::Clubs), card(Rank::Five, Suit::Clubs),
                card(Rank::King, Suit::Clubs), card(Rank::Six, Suit::Clubs),
            ],
            opponents: vec![OpponentHand { seat: 1, hand_size: 10, active: true, pass_ceilings: PassCeilings::default() }],
            unseen_cards: vec![],
            own_pass_ceilings,
            current_combo: Some(combo),
        }
    }

    #[test]
    fn one_never_bluffs_while_pressing() {
        let table = ten_diamonds();
        let mut context = qualifying_context(&table, PassCeilings::default());
        context.opponents = vec![OpponentHand { seat: 1, hand_size: 2, active: true, pass_ceilings: PassCeilings::default() }];
        let legal = vec![Move::Pass, play(vec![card(Rank::King, Suit::Clubs)])];
        for seed in 0..500u64 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            assert!(!should_bluff_pass(0.9, &legal, DuplicateRule::FirstDealtWins, &context, &mut rng));
        }
    }

    #[test]
    fn two_never_bluffs_when_more_than_one_beating_rank_exists() {
        let table = ten_diamonds();
        let context = qualifying_context(&table, PassCeilings::default());
        let legal = vec![Move::Pass, play(vec![card(Rank::Jack, Suit::Clubs)]), play(vec![card(Rank::King, Suit::Clubs)])];
        for seed in 0..500u64 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            assert!(!should_bluff_pass(0.9, &legal, DuplicateRule::FirstDealtWins, &context, &mut rng));
        }
    }

    #[test]
    fn three_never_bluffs_when_forced_or_leading_or_redundant() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let table = ten_diamonds();

        // forced: legal_moves is Pass only
        let forced_ctx = qualifying_context(&table, PassCeilings::default());
        assert!(!should_bluff_pass(0.9, &[Move::Pass], DuplicateRule::FirstDealtWins, &forced_ctx, &mut rng));

        // leading: no current combo
        let mut leading_ctx = qualifying_context(&table, PassCeilings::default());
        leading_ctx.current_combo = None;
        let legal = vec![play(vec![card(Rank::King, Suit::Clubs)])];
        assert!(!should_bluff_pass(0.9, &legal, DuplicateRule::FirstDealtWins, &leading_ctx, &mut rng));

        // redundant: own_pass_ceilings already covers this exact combo
        // — build it the same way `read_pass_ceilings` would from a
        // real prior pass against a Ten at size 1.
        let prior_pass = vec![(0u8, ten_diamonds(), 0usize)];
        let own_ceilings = read_pass_ceilings(2, &[], &prior_pass, DuplicateRule::FirstDealtWins)[0];
        let redundant_ctx = qualifying_context(&table, own_ceilings);
        let legal = vec![Move::Pass, play(vec![card(Rank::King, Suit::Clubs)])];
        assert!(!should_bluff_pass(0.9, &legal, DuplicateRule::FirstDealtWins, &redundant_ctx, &mut rng));
    }

    #[test]
    fn four_fires_at_roughly_the_configured_rate() {
        let table = ten_diamonds();
        let context = qualifying_context(&table, PassCeilings::default());
        let legal = vec![Move::Pass, play(vec![card(Rank::King, Suit::Clubs)])];
        let fires = (0..1000u64)
            .filter(|&seed| {
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                should_bluff_pass(0.25, &legal, DuplicateRule::FirstDealtWins, &context, &mut rng)
            })
            .count();
        // Statistical, not exact: rate=0.25 over 1000 draws lands in
        // this band with overwhelming probability. If this ever
        // flakes, widen the band — don't chase exact determinism on a
        // Bernoulli draw.
        assert!((150..=350).contains(&fires), "expected roughly 250 fires out of 1000, got {fires}");
    }

    // No standalone "rate <= 0.0 never draws from rng" unit test here
    // — writing a custom panic-on-any-use `Rng` impl against rand
    // 0.10.3's redesigned trait hierarchy (`Rng: TryRng<Error =
    // Infallible>`, quite different from the simpler `RngCore` of
    // older rand versions) isn't worth the complexity. The real,
    // sufficient proof is Task 7's
    // `adaptive_none_matches_lowest_legal_exactly` equivalence test: a
    // single spurious draw here would advance the shared match-wide
    // `rng`'s state and desynchronize every later decision in that
    // match, which that test's byte-for-byte `MatchResult` comparison
    // would catch immediately.
}
```

- [ ] **Step 2: Run tests to verify they fail to compile**

Run: `cargo test -p sim --lib adaptive::deception`
Expected: fails — the module/function don't exist yet.

- [ ] **Step 3: Implement `should_bluff_pass`**

```rust
// sim/src/strategies/adaptive/deception.rs
//! Deception: occasionally passes while holding a beater, to plant a
//! false pass ceiling in an opponent's hand reading (`crate::hand_reading`).
//!
//! Bluffs only when *exactly one* rank in hand beats the table combo.
//! The lie ("nothing above this at this size") then survives until
//! that rank is finally played — which a LowestLegal-ordered hand does
//! last, exactly when a hand-reading opponent might otherwise choose a
//! `Lock` against this seat (see `denial::targeted_denial`). A
//! low-rank bluff would be refuted by the very next higher single this
//! seat plays and is worthless; this is why the trigger requires the
//! bluff to be about this seat's *only* beating rank, not any rank.
//!
//! Never bluffs while any active opponent is itself close to finishing
//! (denying is more valuable than deceiving), with a short hand (the
//! lie needs a future round-portion to pay off in), or when the public
//! ceiling already covers this exact combo (a redundant bluff costs a
//! tempo for no new misinformation).
//!
//! **Honest trade-off, stated plainly**: every bluff gives up a trick
//! this seat could have won — the same "hoard for safety, pay with
//! speed" trade this project's empirical batch run already showed
//! losing for `HoldBackPairs`/`CardCounter`. The payoff exists only
//! against opponents actually doing pass-based hand-reading, later in
//! the same match. Expect neutral-to-negative net placement for the
//! deceiver in tables without hand-reading opponents; the batch stats
//! decide whether it's worth it in mixed tables.

use rand::RngExt;

use engine::{DuplicateRule, Move};

use crate::strategy::TurnContext;

/// No bluffing below this many cards: the lie needs a future to pay off in.
const MIN_HAND_TO_BLUFF: usize = 5;

pub(super) fn should_bluff_pass(
    rate: f64,
    legal_moves: &[Move],
    duplicate_rule: DuplicateRule,
    context: &TurnContext<'_>,
    rng: &mut dyn rand::Rng,
) -> bool {
    if rate <= 0.0 {
        return false;
    }
    let Some(current) = context.current_combo else {
        return false; // leading: Pass isn't legal
    };
    let pressing = context
        .opponents
        .iter()
        .any(|o| o.active && o.hand_size <= super::config::DEFAULT_CLOSE);
    if pressing || context.hand.len() < MIN_HAND_TO_BLUFF {
        return false;
    }

    let mut beating_ranks = legal_moves.iter().filter_map(|mv| match mv {
        Move::Play(combo) => Some(combo.top_card(duplicate_rule).rank),
        Move::Pass => None,
    });
    let Some(only_rank) = beating_ranks.next() else {
        return false; // forced pass already: nothing to lie about
    };
    if beating_ranks.any(|rank| rank != only_rank) {
        return false; // more than one beating rank: a lie here is cheaply refuted
    }

    let passed_top = current.top_card(duplicate_rule);
    if context
        .own_pass_ceilings
        .cannot_beat(current.size(), passed_top, duplicate_rule)
    {
        return false; // public read is already at or below this: redundant
    }
    rng.random_bool(rate)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p sim --lib adaptive::deception`
Expected: PASS, all 5 tests green.

- [ ] **Step 5: Declare the module**

In `sim/src/strategies/adaptive/mod.rs`, add `mod deception;`.

- [ ] **Step 6: Run the full `sim` suite, lint and format**

Run: `cargo test -p sim`, then `cargo fmt -p sim`, then `cargo clippy -p sim --all-targets -- -D warnings`

- [ ] **Step 7: Commit**

```bash
git add sim/src/strategies/adaptive/deception.rs sim/src/strategies/adaptive/mod.rs
git commit -m "sim: add adaptive::deception, bluff-passing to corrupt hand reading"
```

---

### Task 7: `sim::strategies::adaptive` — the `Adaptive` strategy (assembles Tasks 4-6)

**Files:**
- Modify: `sim/src/strategies/adaptive/mod.rs`
- Modify: `sim/src/strategies/mod.rs` (add `Adaptive` to the re-export)
- Modify: `sim/src/lib.rs` (re-export `Adaptive`, `AdaptiveConfig`,
  `DenialMode` at the crate root, alongside the existing strategy
  re-exports)

**Interfaces:**
- Consumes: `AdaptiveConfig`/`DenialMode` (Task 4), `denial::respond`
  (Task 5), `deception::should_bluff_pass` (Task 6),
  `CardCounter`/`LowestLegal` (existing, unmodified).
- Produces: `pub struct Adaptive` implementing `Strategy`.

- [ ] **Step 1: Write the equivalence/integration tests first**

These are the most important tests in the whole phase — add to
`sim/src/strategies/adaptive/mod.rs`'s own test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use engine::DeckVariant;
    use crate::{MatchConfig, run_batch};

    fn configs(seeds: impl Iterator<Item = u64>, player_count: u8) -> Vec<MatchConfig> {
        seeds
            .map(|seed| MatchConfig {
                player_count,
                deck_variant: DeckVariant::Single,
                duplicate_rule: DuplicateRule::FirstDealtWins,
                rounds: 5,
                seed,
            })
            .collect()
    }

    #[test]
    fn adaptive_none_matches_lowest_legal_exactly() {
        let cfgs = configs(0..30, 4);
        let adaptive: Arc<dyn Strategy> = Arc::new(Adaptive::new(AdaptiveConfig::NONE));
        let lowest: Arc<dyn Strategy> = Arc::new(LowestLegal);
        let a = run_batch(&cfgs, &vec![adaptive; 4]);
        let b = run_batch(&cfgs, &vec![lowest; 4]);
        for (ra, rb) in a.iter().zip(&b) {
            assert_eq!(ra.role_history, rb.role_history);
            assert_eq!(ra.pass_counts, rb.pass_counts);
            assert_eq!(ra.trick_count, rb.trick_count);
        }
    }

    #[test]
    fn adaptive_counting_only_matches_card_counter_exactly() {
        let cfgs = configs(0..30, 4);
        let config = AdaptiveConfig { counting: true, denial: DenialMode::Off, deception_rate: 0.0 };
        let adaptive: Arc<dyn Strategy> = Arc::new(Adaptive::new(config));
        let counter: Arc<dyn Strategy> = Arc::new(CardCounter);
        let a = run_batch(&cfgs, &vec![adaptive; 4]);
        let b = run_batch(&cfgs, &vec![counter; 4]);
        for (ra, rb) in a.iter().zip(&b) {
            assert_eq!(ra.role_history, rb.role_history);
        }
    }

    #[test]
    fn adaptive_hand_size_denial_matches_endgame_denial_exactly() {
        let cfgs = configs(0..30, 4);
        let config = AdaptiveConfig { counting: false, denial: DenialMode::HandSize { close: 2 }, deception_rate: 0.0 };
        let adaptive: Arc<dyn Strategy> = Arc::new(Adaptive::new(config));
        let denier: Arc<dyn Strategy> = Arc::new(EndgameDenial);
        let a = run_batch(&cfgs, &vec![adaptive; 4]);
        let b = run_batch(&cfgs, &vec![denier; 4]);
        for (ra, rb) in a.iter().zip(&b) {
            assert_eq!(ra.role_history, rb.role_history);
        }
    }

    #[test]
    fn adaptive_default_runs_to_completion_at_every_table_size() {
        for player_count in [3u8, 4, 5, 6] {
            let cfgs = configs(0..20, player_count);
            let adaptive: Arc<dyn Strategy> = Arc::new(Adaptive::default());
            let strategies = vec![adaptive; usize::from(player_count)];
            let results = run_batch(&cfgs, &strategies);
            assert_eq!(results.len(), 20, "player_count {player_count}");
        }
    }

    #[test]
    fn name_reflects_configuration() {
        assert_eq!(Adaptive::new(AdaptiveConfig::NONE).name(), "Adaptive(none)");
        assert!(Adaptive::new(AdaptiveConfig { counting: true, denial: DenialMode::Off, deception_rate: 0.0 }).name().contains("counting"));
        assert!(Adaptive::new(AdaptiveConfig { counting: false, denial: DenialMode::HandReading { close: 2 }, deception_rate: 0.0 }).name().contains("reading"));
    }
}
```

Check `MatchConfig`'s real field names/types and `run_batch`'s real
signature against `sim/tests/multi_config.rs` before trusting the
sketch above — this plan's earlier phases already had to correct
similar assumptions.

- [ ] **Step 2: Run tests to verify they fail to compile**

Run: `cargo test -p sim --lib adaptive`
Expected: fails — `Adaptive` doesn't exist yet.

- [ ] **Step 3: Implement `Adaptive`**

```rust
// sim/src/strategies/adaptive/mod.rs
mod config;
mod deception;
mod denial;

pub use config::{AdaptiveConfig, DenialMode};

use engine::{Card, DuplicateRule, Move};

use crate::strategies::{CardCounter, LowestLegal};
use crate::strategy::{Strategy, TurnContext};

#[derive(Debug, Clone)]
pub struct Adaptive {
    config: AdaptiveConfig,
    name: String,
}

impl Adaptive {
    /// # Panics
    /// If `config` is invalid (`AdaptiveConfig::is_valid`). The CLI
    /// validates during parsing, so reaching this invalid is a
    /// programming error, not user input.
    #[must_use]
    pub fn new(config: AdaptiveConfig) -> Self {
        assert!(config.is_valid(), "invalid AdaptiveConfig: {config:?}");
        Self { name: format!("Adaptive({config})"), config }
    }

    #[must_use]
    pub fn config(&self) -> &AdaptiveConfig {
        &self.config
    }
}

impl Default for Adaptive {
    fn default() -> Self {
        Self::new(AdaptiveConfig::default())
    }
}

impl Strategy for Adaptive {
    fn name(&self) -> &str {
        &self.name
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        context: &TurnContext<'_>,
        rng: &mut dyn rand::Rng,
    ) -> Move {
        if !legal_moves.iter().any(|m| matches!(m, Move::Play(_))) {
            return Move::Pass;
        }

        if let Some(mv) = denial::respond(self.config.denial, legal_moves, duplicate_rule, context, rng) {
            return mv;
        }

        let base = if self.config.counting {
            CardCounter.choose_play(legal_moves, duplicate_rule, context, rng)
        } else {
            LowestLegal.choose_play(legal_moves, duplicate_rule, context, rng)
        };

        let hand_len = context.hand.len();
        let could_finish_instead = legal_moves
            .iter()
            .any(|m| matches!(m, Move::Play(c) if c.size() == hand_len));
        if legal_moves.len() > 1
            && !could_finish_instead
            && deception::should_bluff_pass(self.config.deception_rate, legal_moves, duplicate_rule, context, rng)
        {
            return Move::Pass;
        }

        base
    }

    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        duplicate_rule: DuplicateRule,
        rng: &mut dyn rand::Rng,
    ) -> Vec<Card> {
        LowestLegal.choose_exchange_cards(hand, count, duplicate_rule, rng)
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p sim --lib adaptive`
Expected: PASS, all tests green, **including the three equivalence
tests** — if any of these three fail, do not weaken the assertion or
adjust the equivalence target; the discrepancy is a real bug in either
`Adaptive`'s control flow or one of the delegated-to modifiers, and
must be root-caused before proceeding.

- [ ] **Step 5: Re-export `Adaptive`**

In `sim/src/strategies/mod.rs`, extend the existing `pub use adaptive::
{...};` line to include `Adaptive`. In `sim/src/lib.rs`, add `Adaptive`
to the crate-root strategy re-export list.

- [ ] **Step 6: Run the full `sim` suite, lint and format**

Run: `cargo test -p sim`, then `cargo fmt -p sim`, then `cargo clippy -p sim --all-targets -- -D warnings`

- [ ] **Step 7: Commit**

```bash
git add sim/src/strategies/adaptive/mod.rs sim/src/strategies/mod.rs sim/src/lib.rs
git commit -m "sim: add Adaptive, assembling counting/denial/deception over LowestLegal"
```

---

### Task 8: `cli` — parameterized `--strategy` grammar

**Files:**
- Modify: `cli/src/args.rs`
- Modify: `cli/src/summary.rs`

**Interfaces:**
- Consumes: `sim::{Adaptive, AdaptiveConfig}` (Task 7).
- Produces: `StrategyArg::{Fixed(FixedStrategy), Adaptive(AdaptiveConfig)}`
  with a `FromStr` impl implementing the grammar `SPEC := FIXED |
  "adaptive" | "adaptive:" OPTIONS`.

- [ ] **Step 1: Read the current file first**

Read `cli/src/args.rs` in full (the current `StrategyArg` enum,
`build()`, and existing tests) and `cli/src/summary.rs` (its fixed
`{:<16}` strategy-name column width).

- [ ] **Step 2: Write the failing tests**

Extend `cli/src/args.rs`'s existing test module:

```rust
#[test]
fn parses_fixed_strategy_names_unchanged() {
    assert_eq!("lowest-legal".parse::<StrategyArg>().unwrap(), StrategyArg::Fixed(FixedStrategy::LowestLegal));
    assert_eq!("card-counter".parse::<StrategyArg>().unwrap(), StrategyArg::Fixed(FixedStrategy::CardCounter));
}

#[test]
fn parses_bare_adaptive_to_defaults() {
    assert_eq!(
        "adaptive".parse::<StrategyArg>().unwrap(),
        StrategyArg::Adaptive(sim::AdaptiveConfig::default())
    );
}

#[test]
fn parses_configured_adaptive() {
    let parsed = "adaptive:counting".parse::<StrategyArg>().unwrap();
    assert_eq!(
        parsed,
        StrategyArg::Adaptive(sim::AdaptiveConfig {
            counting: true,
            denial: sim::DenialMode::Off,
            deception_rate: 0.0,
        })
    );
}

#[test]
fn rejects_options_on_a_fixed_strategy() {
    assert!("lowest-legal:counting".parse::<StrategyArg>().is_err());
}

#[test]
fn rejects_unknown_strategy_name() {
    assert!("nonexistent".parse::<StrategyArg>().is_err());
}

#[test]
fn build_produces_an_adaptive_strategy_instance() {
    let arg: StrategyArg = "adaptive:reading,deception=0.2".parse().unwrap();
    let strategy = arg.build();
    assert!(strategy.name().starts_with("Adaptive("));
}
```

(Confirm the exact existing test naming/style in `args.rs` first and
match it — these are meant to extend, not replace, its current test
module.)

- [ ] **Step 3: Run tests to verify they fail to compile**

Run: `cargo test -p cli`
Expected: fails — `StrategyArg`'s shape doesn't match yet.

- [ ] **Step 4: Replace `StrategyArg` with the tagged enum + grammar**

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum FixedStrategy {
    LowestLegal, GreedyHighest, RandomLegal, HoldBackPairs, CardCounter, EndgameDenial,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StrategyArg {
    Fixed(FixedStrategy),
    Adaptive(sim::AdaptiveConfig),
}

impl std::str::FromStr for StrategyArg {
    type Err = String;

    fn from_str(spec: &str) -> Result<Self, String> {
        let spec = spec.trim();
        let (head, options) = match spec.split_once(':') {
            Some((h, o)) => (h.trim(), Some(o)),
            None => (spec, None),
        };
        if head == "adaptive" {
            return match options {
                None => Ok(Self::Adaptive(sim::AdaptiveConfig::default())),
                Some(o) => o.parse().map(Self::Adaptive).map_err(|e| e.to_string()),
            };
        }
        if options.is_some() {
            return Err(format!("strategy `{head}` takes no options (only `adaptive:` does)"));
        }
        <FixedStrategy as clap::ValueEnum>::from_str(head, false)
            .map(Self::Fixed)
            .map_err(|_| format!(
                "unknown strategy `{head}`; expected one of: {}, adaptive[:OPTIONS]",
                fixed_strategy_names(),
            ))
    }
}

fn fixed_strategy_names() -> String {
    use clap::ValueEnum;
    FixedStrategy::value_variants()
        .iter()
        .filter_map(clap::ValueEnum::to_possible_value)
        .map(|p| p.get_name().to_owned())
        .collect::<Vec<_>>()
        .join(", ")
}

impl StrategyArg {
    #[must_use]
    pub fn build(self) -> Arc<dyn sim::Strategy> {
        match self {
            Self::Fixed(FixedStrategy::LowestLegal) => Arc::new(sim::LowestLegal),
            Self::Fixed(FixedStrategy::GreedyHighest) => Arc::new(sim::GreedyHighest),
            Self::Fixed(FixedStrategy::RandomLegal) => Arc::new(sim::RandomLegal),
            Self::Fixed(FixedStrategy::HoldBackPairs) => Arc::new(sim::HoldBackPairs),
            Self::Fixed(FixedStrategy::CardCounter) => Arc::new(sim::CardCounter),
            Self::Fixed(FixedStrategy::EndgameDenial) => Arc::new(sim::EndgameDenial),
            Self::Adaptive(config) => Arc::new(sim::Adaptive::new(config)),
        }
    }
}
```

Update `Args::strategies`'s attribute (drop `value_enum`, it falls back
to `FromStr` automatically) and doc comment to spell out the full
grammar with a worked example (three seats: `lowest-legal`,
`adaptive:counting`, `adaptive:reading,deception=0.2`):

```rust
#[arg(long = "strategy", required = true, value_name = "SPEC")]
pub strategies: Vec<StrategyArg>,
```

- [ ] **Step 5: Fix `cli/src/summary.rs`'s column width**

Replace the hardcoded `{:<16}` strategy-name formatting with a width
computed from the longest actual name in
`statistics.role_counts_by_strategy.keys()` (an `Adaptive(...)` name
can exceed 16 characters).

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test -p cli`
Expected: PASS, all tests green including the new ones.

- [ ] **Step 7: Lint and format**

Run: `cargo fmt -p cli` then `cargo clippy -p cli --all-targets -- -D warnings`

- [ ] **Step 8: Commit**

```bash
git add cli/src/args.rs cli/src/summary.rs
git commit -m "cli: parameterize --strategy with a per-seat adaptive spec grammar"
```

---

### Task 9: Integration coverage + docs

**Files:**
- Modify: `sim/tests/multi_config.rs`
- Modify: `docs/ROADMAP.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `README.md`
- Modify: `docs/BUILDING.md`
- Modify: `web/src/lib.rs` (and any other stray "Phase 7" reference)

**Interfaces:**
- Consumes: `sim::Adaptive` (Task 7), `cli`'s new grammar (Task 8).

- [ ] **Step 1: Read the current files first**

Read `sim/tests/multi_config.rs`'s `baseline_strategies` pool/rotation
formula, and `docs/ROADMAP.md`'s current Phase 7 ("Web interface")
entry, before editing either.

- [ ] **Step 2: Grow `multi_config.rs`'s strategy pool**

Add `Adaptive::default()` to the sweep pool (7 entries total now),
re-deriving the rotation-offset formula by hand for the new pool length
at every table size (3-6) — do not assume the existing formula
generalizes silently; Phase 6's Task 5 already had to fix a coverage
gap once when this pool grew from 4 to 6.

- [ ] **Step 3: Run the sim integration test**

Run: `cargo test -p sim --test multi_config`
Expected: PASS across all four table sizes with the 7-entry pool.

- [ ] **Step 4: Update `docs/ROADMAP.md`**

Insert the new Phase 7 entry (summarized from this plan's Context —
`Round`'s pass history, `sim::hand_reading`, `TurnContext`'s new
fields, `Adaptive` and its three modifiers, the CLI grammar) before the
current "Phase 7 — Web interface", and renumber that entry to "Phase
8".

- [ ] **Step 5: Sweep every other "Phase 7" reference**

Search `docs/*.md`, `README.md`, and `web/src/lib.rs` for the literal
text "Phase 7" and update any referring to the web-interface phase to
say "Phase 8" — do this proactively now; Phase 6's final review had to
catch exactly this kind of stale reference (`web/src/lib.rs`) as a gap
last time.

- [ ] **Step 6: Update `docs/ARCHITECTURE.md`, `README.md`, `docs/BUILDING.md`**

Describe `Round`'s pass history, `sim::hand_reading`, `TurnContext`'s
new fields, and `Adaptive` (config, all three modifiers, the CLI
grammar) in `docs/ARCHITECTURE.md`'s existing section style. Add an
`adaptive` entry to `README.md`'s strategies list with a one-line
grammar mention and a pointer to `docs/BUILDING.md` for the full
grammar. Extend `docs/BUILDING.md`'s `--strategy` reference with the
full grammar and the three-seat worked example from Task 8.

- [ ] **Step 7: Full workspace verification**

Run, in order: `cargo fmt --check`; `cargo clippy --workspace
--all-targets -- -D warnings`; `cargo test --workspace`. All three must
be clean. Then the manual sanity check:
`cargo run -p cli -- --player-count 3 --matches 40 --rounds 5
--strategy lowest-legal --strategy adaptive:counting --strategy
"adaptive:reading,deception=0.2" --seed 1 --output /tmp/results.json`
— confirm distinct names appear in stdout/JSON for the two `adaptive`
seats and the run completes without panicking.

- [ ] **Step 8: Commit**

```bash
git add sim/tests/multi_config.rs docs/ROADMAP.md docs/ARCHITECTURE.md README.md docs/BUILDING.md web/src/lib.rs
git commit -m "sim: add Adaptive to the multi-config sweep; docs: describe Phase 7"
```
