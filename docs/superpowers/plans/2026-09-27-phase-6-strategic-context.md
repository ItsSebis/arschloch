# Phase 6 — Strategic Context, Card Counting, and Endgame Denial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `sim::Strategy` implementations enough game-state
context to do real card counting and endgame denial play, then ship
two new, deliberately separate strategies (`CardCounter`,
`EndgameDenial`) built on that context.

**Architecture:** `engine::Round` gains a play history (cards actually
played this round) and a per-seat hand-size accessor. `sim::Strategy::
choose_play` gains one new parameter, a `TurnContext` bundling the
acting seat's own hand, every other seat's hand size/active status,
and the exact multiset of unseen cards — assembled fresh every turn by
`match_runner::run_match`, not cached. All six existing `choose_play`
implementors (four real strategies, two test-only ones inside
`match_runner.rs`) get the new parameter mechanically; only the two
new strategies actually use it. `choose_exchange_cards` is
deliberately untouched this phase (see Task 2's brief for why).

**Tech Stack:** Rust workspace (`engine`, `sim`, `cli`), no new
dependencies.

**Spec:** The native-plan-mode design doc this plan transcribes (not a
separate spec file — this plan is the spec for this phase, same
convention as Phases 4 and 5). Two research/design subagents worked out
the concrete interface and the two strategies' algorithms during
planning; every code block below was hand-traced against its own test
cases before this plan was approved.

## Global Constraints

- No new Cargo dependencies.
- `engine` stays dependency-free of `sim`/`cli` — `Round`'s new methods
  are plain data accessors (a hand-size read, a play-history read),
  never anything strategy-aware.
- `choose_exchange_cards` is **not** touched this phase — do not add
  `TurnContext` to it, do not change its signature, in any task.
- `HoldBackPairs`'s `choose_play` **algorithm** does not change — only
  its signature gains the new, ignored `context` parameter.
- Both new strategies (`CardCounter`, `EndgameDenial`) must be
  stateless (`#[derive(Debug, Clone, Copy, Default)]`, no interior
  mutability) since `Strategy` instances are shared read-only via `Arc`
  across parallel matches.
- Task 1 is verified with `cargo test -p engine` (plus `cargo clippy -p
  engine --all-targets -- -D warnings`). Tasks 2-4 are verified with
  `cargo test -p sim` (plus `cargo clippy -p sim --all-targets -- -D
  warnings`) — Task 2 includes `match_runner.rs`'s own call site, but
  that's still entirely within `sim`, so no cross-crate gate is needed
  yet. Task 5 touches `cli` and docs and uses the full workspace gate
  (`cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D
  warnings`, `cargo test --workspace`).
- `Option::is_none_or` is available on this workspace's toolchain
  (`rustc 1.98.1`, well past its 1.82 stabilization) — use it directly,
  no fallback needed.

## Review Focus

- **Off-by-one in `opponents`**: the acting seat must be genuinely
  excluded from its own `context.opponents` list. Task 2's own tests
  must assert `opponents.len() == player_count - 1` and that no entry's
  `seat` equals the acting seat.
- **Empty-`unseen_cards` edge case**: both new strategies must handle
  `unseen_cards.is_empty()` cleanly (no panic). Covered by `CardCounter`
  Task 3's "everything is precious" test and `EndgameDenial` Task 4's
  tests (which use `unseen_cards` only indirectly via `opponents`, so
  Task 3's coverage is what actually exercises this for `CardCounter`
  specifically — don't drop that test).
- **The two hidden test-only `Strategy` implementors** in
  `match_runner.rs` (`LeadCounter`, `ExchangeCallCounter`) must be
  migrated in Task 2, not discovered later — Phase 5's Task 2 already
  hit this exact gap once (a test-only impl missing from a task's file
  list, found only because the crate failed to compile).
- **Determinism regression**: `run_match`'s existing determinism tests
  (identical config + seed → identical results) must still pass
  unchanged after Task 2 threads `TurnContext` through the trick loop —
  verify by running the suite, don't assume it from reading the diff.
- **`hand_size`/`active` consistency**: a seat's `hand_size == 0` must
  always imply `active == false` and vice versa (per `Round::is_active`
  already meaning "hand is non-empty"). Any fixture in Task 4's tests
  using `active: false` must also use `hand_size: 0` — an inconsistent
  fixture (e.g. `active: false` with a nonzero `hand_size`) tests a
  state the real game can never produce.

---

### Task 1: `engine::Round` — play history + hand-size accessor

**Files:**
- Modify: `engine/src/round.rs`

**Interfaces:**
- Produces: `pub fn Round::play_history(&self) -> &[(SeatId, Combo)]`,
  `pub fn Round::hand_size(&self, seat: SeatId) -> usize`. Both are
  pure reads of data `Round` already stores.
- Consumes: nothing new — `Combo` already derives `Clone` (confirmed:
  `engine/src/combo.rs`'s `#[derive(Debug, Clone, PartialEq, Eq)]`).

- [ ] **Step 1: Read the current file first**

Read `engine/src/round.rs` in full. Confirm `apply_play` (currently at
lines 209-230) has the exact shape assumed below — in particular, that
`self.current_combo = Some(combo);` is the line the combo is moved
into, so the new history-push line goes immediately before it.

- [ ] **Step 2: Write the failing tests**

Add to `round.rs`'s existing `#[cfg(test)] mod tests` (reuse the
existing `card(rank, suit)` and `combo(cards)` helper functions already
defined there — do not redefine them):

```rust
#[test]
fn play_history_records_plays_in_order_and_omits_passes() {
    let hands = vec![
        vec![card(Rank::Eight, Suit::Clubs), card(Rank::Ten, Suit::Clubs)],
        vec![card(Rank::Seven, Suit::Clubs), card(Rank::Six, Suit::Clubs)],
        vec![card(Rank::Nine, Suit::Clubs), card(Rank::Five, Suit::Clubs)],
    ];
    let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
    assert_eq!(round.play_history(), &[]);

    let first_play = combo(vec![card(Rank::Eight, Suit::Clubs)]);
    round.submit_move(0, Move::Play(first_play.clone())).unwrap();
    assert_eq!(round.play_history(), &[(0, first_play.clone())]);

    round.submit_move(1, Move::Pass).unwrap();
    assert_eq!(
        round.play_history(),
        &[(0, first_play.clone())],
        "a pass must not appear in play_history"
    );

    let second_play = combo(vec![card(Rank::Nine, Suit::Clubs)]);
    round.submit_move(2, Move::Play(second_play.clone())).unwrap();
    assert_eq!(
        round.play_history(),
        &[(0, first_play), (2, second_play)]
    );
}

#[test]
fn hand_size_matches_hand_len_and_decreases_after_a_play() {
    let hands = vec![
        vec![card(Rank::Eight, Suit::Clubs), card(Rank::Ten, Suit::Clubs)],
        vec![card(Rank::Seven, Suit::Clubs)],
        vec![card(Rank::Nine, Suit::Clubs)],
    ];
    let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
    assert_eq!(round.hand_size(0), 2);
    assert_eq!(round.hand_size(0), round.hand(0).len());

    round
        .submit_move(0, Move::Play(combo(vec![card(Rank::Eight, Suit::Clubs)])))
        .unwrap();
    assert_eq!(round.hand_size(0), 1);
}
```

- [ ] **Step 3: Run tests to verify they fail to compile**

Run: `cargo test -p engine`
Expected: compile error — `play_history`/`hand_size` don't exist yet.

- [ ] **Step 4: Add the field, accessors, and the one-line history push**

Add `play_history: Vec<(SeatId, Combo)>` to the `Round` struct, and
`play_history: Vec::new()` to `Round::new`'s constructor. Add the two
public accessors:

```rust
/// Every combo played so far this round, in play order, tagged with
/// the seat that played it. Passes are omitted — a pass never removes
/// a card from any hand, so it carries nothing a card-counting
/// strategy needs (docs/ROADMAP.md, Phase 6).
#[must_use]
pub fn play_history(&self) -> &[(SeatId, Combo)] {
    &self.play_history
}

/// `seat`'s current hand size only (not contents) — the one piece of
/// information about *other* seats this genre treats as public.
///
/// # Panics
///
/// Panics if `seat` is not a valid seat for this round (same
/// unchecked-indexing convention as `hand`).
#[must_use]
pub fn hand_size(&self, seat: SeatId) -> usize {
    self.hands[usize::from(seat)].len()
}
```

In `apply_play`, add one line immediately before `self.current_combo =
Some(combo);`:

```rust
self.play_history.push((seat, combo.clone()));
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p engine`
Expected: PASS — all pre-existing `Round` tests unchanged and green,
plus the two new tests.

- [ ] **Step 6: Lint and format**

Run: `cargo fmt -p engine` then `cargo clippy -p engine --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 7: Commit**

```bash
git add engine/src/round.rs
git commit -m "engine: track play history and expose per-seat hand size on Round"
```

---

### Task 2: `sim::strategy.rs` + `match_runner.rs` — `TurnContext`, migrate all six `choose_play` implementors, and wire real context assembly

**Files:**
- Modify: `sim/src/strategy.rs`
- Modify: `sim/src/strategies/lowest_legal.rs`
- Modify: `sim/src/strategies/greedy_highest.rs`
- Modify: `sim/src/strategies/random_legal.rs`
- Modify: `sim/src/strategies/hold_back_pairs.rs`
- Modify: `sim/src/match_runner.rs` (its two test-only `Strategy`
  implementors, `LeadCounter` and `ExchangeCallCounter`, **and** the
  production `run_match` trick loop's own `choose_play` call site —
  see the note below on why these can't be split into separate tasks)
- Modify: `sim/src/lib.rs` (re-export `OpponentHand`, `TurnContext`)

**Interfaces:**
- Produces: `pub struct OpponentHand { pub seat: SeatId, pub hand_size:
  usize, pub active: bool }`, `pub struct TurnContext<'a> { pub seat:
  SeatId, pub hand: &'a [Card], pub opponents: Vec<OpponentHand>, pub
  unseen_cards: Vec<Card> }`, and `Strategy::choose_play`'s new
  signature: `fn choose_play(&self, legal_moves: &[Move],
  duplicate_rule: DuplicateRule, context: &TurnContext<'_>, rng: &mut
  dyn rand::Rng) -> Move`.
- Consumes: `Round::play_history()`, `Round::hand_size()` (Task 1).

**Why this is one task, not two:** an earlier draft of this plan split
"define `TurnContext`+migrate implementors" from "wire it into
`run_match`'s call site" into two separate tasks. That doesn't work:
changing `Strategy::choose_play`'s signature immediately breaks
`run_match`'s own production call site (`sim/src/match_runner.rs`,
the trick loop itself, not just its test-only `Strategy` impls) — it
would still be calling the old 3-argument form. There is no way to
migrate the trait and leave that call site broken but "not yet in
scope" without leaving the crate uncompilable at the end of a task,
which no per-task review gate should ever have to pass through. So
this task does both: change the signature, migrate every implementor
(real and test-only), *and* wire the real context assembly into
`run_match`'s call site, in one commit.

Why `choose_exchange_cards` is untouched this phase: between rounds
there's no play history yet, and "unseen" would always just be "the
full deck minus my hand" — already computable today from what
`choose_exchange_cards` receives. The concept only becomes useful
mid-round. Do not add `TurnContext` to it.

- [ ] **Step 1: Read the current files first**

Read `sim/src/strategy.rs`, `sim/src/match_runner.rs` (in full,
including its `run_match` function and its `#[cfg(test)] mod tests`),
and the four strategy files listed above. Confirm each strategy's
current `impl Strategy for ...` block, `run_match`'s exact current
trick-loop shape (where `deal`/`exchange` happen before `Round::new`,
and the existing `choose_play` call site), and imports before editing.

- [ ] **Step 2: Add `OpponentHand`, `TurnContext`, and the new trait signature**

In `sim/src/strategy.rs`, add `SeatId` to the `use engine::{...};`
import line, then add:

```rust
/// A seat other than the one currently acting, and what's publicly
/// known about it: its current hand size and whether it's still in
/// the round (a finished seat's hand size is always `0`, but `active`
/// is spelled out so strategies never have to re-derive it).
#[derive(Debug, Clone, Copy)]
pub struct OpponentHand {
    pub seat: SeatId,
    pub hand_size: usize,
    pub active: bool,
}

/// Everything beyond `legal_moves` a `Strategy` needs for card
/// counting and endgame denial (docs/ROADMAP.md, Phase 6). Built fresh
/// on the stack every turn by `match_runner::run_match` — cheap at
/// this project's match sizes, so there's no caching/incremental-
/// update machinery here.
pub struct TurnContext<'a> {
    /// The acting seat, for context that needs to know who's asking.
    pub seat: SeatId,
    /// This seat's full remaining hand.
    pub hand: &'a [Card],
    /// Every *other* seat, in seat order.
    pub opponents: Vec<OpponentHand>,
    /// The exact multiset of cards neither in `hand` nor played by
    /// anyone yet this round — i.e. every card some other still-active
    /// seat currently holds. Deterministic: this is a closed-deck game
    /// with no draw pile.
    pub unseen_cards: Vec<Card>,
}
```

Change `choose_play`'s signature in the `Strategy` trait to:

```rust
fn choose_play(
    &self,
    legal_moves: &[Move],
    duplicate_rule: DuplicateRule,
    context: &TurnContext<'_>,
    rng: &mut dyn rand::Rng,
) -> Move;
```

Leave `choose_exchange_cards` exactly as it is.

- [ ] **Step 3: Run tests to confirm the expected build break**

Run: `cargo test -p sim`
Expected: compile errors — every `impl Strategy for ...` block (six of
them, across four strategy files plus the two test-only ones in
`match_runner.rs`) is now missing the updated `choose_play` signature,
and `run_match`'s own `choose_play` call site no longer matches either.
This confirms the trait change took effect.

- [ ] **Step 4: Migrate `LowestLegal` and `GreedyHighest`**

In both `sim/src/strategies/lowest_legal.rs` and `sim/src/strategies/
greedy_highest.rs`, update `choose_play`'s signature to insert
`context: &TurnContext<'_>` as the third parameter (after
`duplicate_rule`, before `rng`), prefixed `_context` since neither
strategy uses it. Add `TurnContext` to each file's `use crate::
strategy::{...};` import. Do not change either strategy's logic or
existing tests.

- [ ] **Step 5: Migrate `RandomLegal`**

Same signature change (`_context`) in `sim/src/strategies/
random_legal.rs`. No logic change.

- [ ] **Step 6: Migrate `HoldBackPairs`**

Same signature change (`_context`) in `sim/src/strategies/
hold_back_pairs.rs`. Its existing "don't split a reserve" logic already
derives everything it needs from `legal_moves`'s per-rank candidate
count — **do not change its algorithm**, only its signature.

- [ ] **Step 7: Migrate the two test-only implementors in `match_runner.rs`**

Find `LeadCounter` and `ExchangeCallCounter` inside `sim/src/
match_runner.rs`'s own `#[cfg(test)] mod tests`. Update both
`choose_play` signatures the same way (`_context`, or `context` if
either already needs to reference it for an existing test purpose —
read each one's current body first; if unsure whether it needs the
real context, use `_context` and let the test suite tell you if
something breaks).

- [ ] **Step 8: Write a new test for `TurnContext` correctness**

Add to `match_runner.rs`'s existing `#[cfg(test)] mod tests`, following
the same external-`Arc`-plus-clone pattern `ExchangeCallCounter`
already uses (add `AtomicBool` to this module's existing `use std::
sync::atomic::{...};` import alongside `AtomicU32`):

```rust
/// Records whether any seat's `TurnContext.opponents` ever included
/// that seat itself, or had the wrong length — checked once after the
/// match completes (docs/ROADMAP.md, Phase 6, Review Focus).
struct TurnContextChecker {
    player_count: u8,
    violation: Arc<AtomicBool>,
}

impl Strategy for TurnContextChecker {
    fn name(&self) -> &'static str {
        "TurnContextChecker"
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        context: &TurnContext<'_>,
        rng: &mut dyn rand::Rng,
    ) -> Move {
        let wrong_length = context.opponents.len() != usize::from(self.player_count) - 1;
        let includes_self = context.opponents.iter().any(|o| o.seat == context.seat);
        if wrong_length || includes_self {
            self.violation.store(true, Ordering::Relaxed);
        }
        crate::strategies::LowestLegal.choose_play(legal_moves, duplicate_rule, context, rng)
    }

    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        duplicate_rule: DuplicateRule,
        rng: &mut dyn rand::Rng,
    ) -> Vec<Card> {
        crate::strategies::LowestLegal.choose_exchange_cards(hand, count, duplicate_rule, rng)
    }
}

#[test]
fn turn_context_excludes_the_acting_seat_from_opponents() {
    let config = MatchConfig {
        player_count: 4,
        deck_variant: DeckVariant::Single,
        duplicate_rule: DuplicateRule::FirstDealtWins,
        rounds: 3,
        seed: 7,
    };
    let violation = Arc::new(AtomicBool::new(false));
    let strategies: Vec<Arc<dyn Strategy>> = (0..4)
        .map(|_| {
            Arc::new(TurnContextChecker {
                player_count: config.player_count,
                violation: violation.clone(),
            }) as Arc<dyn Strategy>
        })
        .collect();
    run_match(&config, &strategies);
    assert!(!violation.load(Ordering::Relaxed));
}
```

This is the Review Focus item on off-by-one exclusion — write this
test before implementing Step 9, and confirm it still fails to compile
(the production call site still isn't passing a `TurnContext`) before
making it pass.

- [ ] **Step 9: Capture the round's dealt deck and assemble the context each turn**

Right after `hands` is dealt/exchanged and before `Round::new` moves
it, capture:

```rust
let round_deck: Vec<Card> = hands.iter().flatten().copied().collect();
```

Inside the trick loop, before the existing `choose_play` call, insert:

```rust
let opponents: Vec<OpponentHand> = (0..config.player_count)
    .filter(|&s| s != seat)
    .map(|s| OpponentHand {
        seat: s,
        hand_size: round.hand_size(s),
        active: round.is_active(s),
    })
    .collect();

let mut unseen_cards = round_deck.clone();
for card in round.hand(seat).iter().chain(
    round.play_history().iter().flat_map(|(_, combo)| combo.cards()),
) {
    if let Some(pos) = unseen_cards.iter().position(|c| c == card) {
        unseen_cards.remove(pos);
    }
}

let context = TurnContext {
    seat,
    hand: round.hand(seat),
    opponents,
    unseen_cards,
};
```

Update the existing `choose_play` call to pass `&context` as the third
argument (before `&mut rng`). Everything else in the loop (pass-count
tracking, `round.submit_move(...)`) is unchanged. If the borrow checker
rejects `context` (which borrows `round` immutably) coexisting with the
later `round.submit_move(...)` (a mutable borrow), this should resolve
itself via NLL since `context` isn't used after the `choose_play` call
— if it doesn't, report the exact compiler error rather than
restructuring broadly.

- [ ] **Step 10: Run tests to verify everything compiles and passes**

Run: `cargo test -p sim`
Expected: PASS — every pre-existing test across all six migrated files
unchanged and green, plus the new `turn_context_excludes_the_acting_
seat_from_opponents` test, **specifically including**
`identical_config_and_strategies_are_fully_deterministic` and
`identical_seeds_produce_identical_results` (Review Focus: determinism
must not regress).

- [ ] **Step 11: Re-export from `sim/src/lib.rs`**

Add `OpponentHand` and `TurnContext` to `sim/src/lib.rs`'s existing
re-export list alongside `Strategy`.

- [ ] **Step 12: Run tests once more**

Run: `cargo test -p sim`
Expected: PASS (confirms the Step 11 re-export didn't break anything).

- [ ] **Step 13: Lint and format**

Run: `cargo fmt -p sim` then `cargo clippy -p sim --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 14: Commit**

```bash
git add sim/src/strategy.rs sim/src/strategies/lowest_legal.rs sim/src/strategies/greedy_highest.rs sim/src/strategies/random_legal.rs sim/src/strategies/hold_back_pairs.rs sim/src/match_runner.rs sim/src/lib.rs
git commit -m "sim: add TurnContext, migrate all Strategy implementors, and wire it into run_match"
```

---

### Task 3: `CardCounter` — `sim/src/strategies/card_counter.rs` (new)

**Files:**
- Create: `sim/src/strategies/card_counter.rs`
- Modify: `sim/src/strategies/mod.rs`

**Interfaces:**
- Consumes: `TurnContext` (Task 2), `crate::strategies::
  take_highest_naive` (already exists, from Phase 5).
- Produces: `pub struct CardCounter` implementing `Strategy`, re-
  exported from `sim`.

- [ ] **Step 1: Write the failing tests**

Create `sim/src/strategies/card_counter.rs` with its test module first
(reuse whatever `card(rank, suit)` helper and `TurnContext`/
`OpponentHand` construction idiom the other strategy files in this
directory already use for their own `choose_play` tests — read
`sim/src/strategies/lowest_legal.rs`'s test module for the exact
pattern before writing these):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::OpponentHand;

    fn context_with_unseen(unseen_cards: Vec<Card>) -> TurnContext<'static> {
        TurnContext {
            seat: 0,
            hand: &[],
            opponents: vec![OpponentHand { seat: 1, hand_size: 5, active: true }],
            unseen_cards,
        }
    }

    #[test]
    fn leading_saves_a_precious_single_ace_for_a_non_precious_pair_of_threes() {
        let legal_moves = vec![
            Move::Play(Combo::new(vec![card(Rank::Ace, Suit::Clubs)]).unwrap()),
            Move::Play(Combo::new(vec![
                card(Rank::Three, Suit::Clubs),
                card(Rank::Three, Suit::Diamonds),
            ]).unwrap()),
        ];
        let context = context_with_unseen(vec![card(Rank::Nine, Suit::Hearts)]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let chosen = CardCounter.choose_play(
            &legal_moves,
            DuplicateRule::FirstDealtWins,
            &context,
            &mut rng,
        );
        assert_eq!(
            chosen,
            Move::Play(Combo::new(vec![
                card(Rank::Three, Suit::Clubs),
                card(Rank::Three, Suit::Diamonds),
            ]).unwrap())
        );
    }

    #[test]
    fn leading_with_no_precious_option_picks_lowest_like_lowest_legal() {
        let legal_moves = vec![
            Move::Play(Combo::new(vec![card(Rank::Six, Suit::Clubs)]).unwrap()),
            Move::Play(Combo::new(vec![card(Rank::Nine, Suit::Clubs)]).unwrap()),
        ];
        let context = context_with_unseen(vec![card(Rank::King, Suit::Hearts)]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let chosen = CardCounter.choose_play(
            &legal_moves,
            DuplicateRule::FirstDealtWins,
            &context,
            &mut rng,
        );
        assert_eq!(
            chosen,
            Move::Play(Combo::new(vec![card(Rank::Six, Suit::Clubs)]).unwrap())
        );
    }

    #[test]
    fn leading_when_everything_is_precious_still_plays_the_lowest() {
        let legal_moves = vec![
            Move::Play(Combo::new(vec![card(Rank::King, Suit::Clubs)]).unwrap()),
            Move::Play(Combo::new(vec![card(Rank::Ace, Suit::Clubs)]).unwrap()),
        ];
        let context = context_with_unseen(vec![]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let chosen = CardCounter.choose_play(
            &legal_moves,
            DuplicateRule::FirstDealtWins,
            &context,
            &mut rng,
        );
        assert_eq!(
            chosen,
            Move::Play(Combo::new(vec![card(Rank::King, Suit::Clubs)]).unwrap())
        );
    }

    #[test]
    fn following_plays_the_only_legal_beater_even_when_it_is_precious() {
        let legal_moves = vec![
            Move::Pass,
            Move::Play(Combo::new(vec![card(Rank::King, Suit::Clubs)]).unwrap()),
        ];
        let context = context_with_unseen(vec![]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let chosen = CardCounter.choose_play(
            &legal_moves,
            DuplicateRule::FirstDealtWins,
            &context,
            &mut rng,
        );
        assert_eq!(
            chosen,
            Move::Play(Combo::new(vec![card(Rank::King, Suit::Clubs)]).unwrap())
        );
    }
}
```

Add `use rand::SeedableRng;` if not already implied, and a `card(rank,
suit)` free function matching this directory's existing convention
(`Card::new(rank, suit, 0)`).

- [ ] **Step 2: Run tests to verify they fail to compile**

Run: `cargo test -p sim --lib card_counter`
Expected: fails — `CardCounter` doesn't exist yet, and
`sim/src/strategies/mod.rs` doesn't declare the module.

- [ ] **Step 3: Implement `CardCounter`**

Above the test module in `sim/src/strategies/card_counter.rs`:

```rust
//! A pure card-counting strategy: every decision is driven by exact
//! knowledge of which cards remain unseen (unplayed and not in this
//! seat's own hand) — this is a closed deck with no draw pile, so
//! "unseen" is exact, not an estimate (docs/ROADMAP.md, Phase 6).
//!
//! **Precious combos.** A legal combo of rank `R` is "precious" when no
//! unseen card outranks it. This is a deliberately cheap, rank-only
//! scarcity check, not a full analysis of whether a beating combo is
//! actually *formable*: a combo of size 2+ can only be beaten by
//! another same-size combo, which needs enough unseen copies of some
//! higher rank to exist — a single unseen higher-ranked card is enough
//! to mark a rank non-precious here even if no one could ever actually
//! assemble a same-size beater from it. That means this rule is
//! deliberately conservative (it may treat some truly-unbeatable
//! combos as "spendable"), never the reverse — it never calls a combo
//! precious that could really be beaten. This project's existing
//! strategies are cheap heuristics, not search, so this conservatism
//! is accepted rather than counting per-rank unseen quantities to
//! determine formability.
//!
//! Preciousness is upward-closed in rank (if a low rank is precious,
//! every higher rank is too), so this signal only ever changes the
//! outcome relative to `LowestLegal` while *leading*, where combo size
//! can differ across candidates (`LowestLegal` breaks ties by size
//! first, so it would spend a small precious card immediately). While
//! *following*, size is fixed by the table, so this strategy
//! deliberately degenerates to `LowestLegal`'s choice — leading always
//! requires a play (never `Pass`), so holding a precious combo back
//! only changes *which* combo leads, never *whether* one does, and
//! precious combos get forced out once every legal lead is precious
//! (typically late in the round) — this never stalls emptying the
//! hand.

use engine::{Card, DuplicateRule, Move, Rank};

use crate::strategy::{Strategy, TurnContext};

#[derive(Debug, Clone, Copy, Default)]
pub struct CardCounter;

impl Strategy for CardCounter {
    fn name(&self) -> &'static str {
        "CardCounter"
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        context: &TurnContext<'_>,
        _rng: &mut dyn rand::Rng,
    ) -> Move {
        let highest_unseen_rank = context.unseen_cards.iter().map(|c| c.rank).max();
        let is_precious =
            |rank: Rank| highest_unseen_rank.is_none_or(|highest| rank >= highest);

        let plays: Vec<(usize, Card, bool, &Move)> = legal_moves
            .iter()
            .filter_map(|mv| match mv {
                Move::Play(combo) => {
                    let top = combo.top_card(duplicate_rule);
                    Some((combo.size(), top, is_precious(top.rank), mv))
                }
                Move::Pass => None,
            })
            .collect();

        let lowest = |pool: &[(usize, Card, bool, &Move)]| {
            pool.iter()
                .min_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.compare(&b.1, duplicate_rule)))
                .map(|&(_, _, _, mv)| mv.clone())
        };

        let non_precious: Vec<_> = plays.iter().copied().filter(|&(_, _, p, _)| !p).collect();
        lowest(&non_precious)
            .or_else(|| lowest(&plays))
            .unwrap_or(Move::Pass)
    }

    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        duplicate_rule: DuplicateRule,
        _rng: &mut dyn rand::Rng,
    ) -> Vec<Card> {
        // Card-counting-aware exchange selection is out of this
        // phase's scope (Global Constraints) — reuse the naive
        // highest-N give-up.
        crate::strategies::take_highest_naive(hand, count, duplicate_rule)
    }
}
```

In `sim/src/strategies/mod.rs`, add `mod card_counter;` and `pub use
card_counter::CardCounter;`.

In `sim/src/lib.rs`, add `CardCounter` to the existing strategy
re-export list.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p sim --lib card_counter`
Expected: PASS, all 4 tests green.

- [ ] **Step 5: Run the full `sim` suite**

Run: `cargo test -p sim`
Expected: PASS — nothing else regressed.

- [ ] **Step 6: Lint and format**

Run: `cargo fmt -p sim` then `cargo clippy -p sim --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 7: Commit**

```bash
git add sim/src/strategies/card_counter.rs sim/src/strategies/mod.rs sim/src/lib.rs
git commit -m "sim: add CardCounter, a strategy that holds back unbeatable combos"
```

---

### Task 4: `EndgameDenial` — `sim/src/strategies/endgame_denial.rs` (new)

**Files:**
- Create: `sim/src/strategies/endgame_denial.rs`
- Modify: `sim/src/strategies/mod.rs`

**Interfaces:**
- Consumes: `TurnContext`, `OpponentHand` (Task 2), `GreedyHighest`,
  `LowestLegal` (both pre-existing, both already migrated to the new
  `choose_play` signature by Task 2).
- Produces: `pub struct EndgameDenial` implementing `Strategy`,
  re-exported from `sim`.

- [ ] **Step 1: Write the failing tests**

Create `sim/src/strategies/endgame_denial.rs` with its test module
first, matching the same `card(rank, suit)`/`TurnContext` construction
idiom Task 3 used:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::OpponentHand;

    fn context_with_opponent(hand_size: usize, active: bool) -> TurnContext<'static> {
        TurnContext {
            seat: 0,
            hand: &[],
            opponents: vec![OpponentHand { seat: 1, hand_size, active }],
            unseen_cards: vec![],
        }
    }

    #[test]
    fn no_close_opponent_leads_low_like_lowest_legal() {
        let legal_moves = vec![
            Move::Play(Combo::new(vec![card(Rank::Six, Suit::Clubs)]).unwrap()),
            Move::Play(Combo::new(vec![card(Rank::Nine, Suit::Clubs)]).unwrap()),
        ];
        let context = context_with_opponent(10, true);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let chosen = EndgameDenial.choose_play(
            &legal_moves,
            DuplicateRule::FirstDealtWins,
            &context,
            &mut rng,
        );
        assert_eq!(
            chosen,
            Move::Play(Combo::new(vec![card(Rank::Six, Suit::Clubs)]).unwrap())
        );
    }

    #[test]
    fn close_opponent_leads_high_to_retain_control() {
        let legal_moves = vec![
            Move::Play(Combo::new(vec![card(Rank::Six, Suit::Clubs)]).unwrap()),
            Move::Play(Combo::new(vec![card(Rank::Nine, Suit::Clubs)]).unwrap()),
        ];
        let context = context_with_opponent(2, true);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let chosen = EndgameDenial.choose_play(
            &legal_moves,
            DuplicateRule::FirstDealtWins,
            &context,
            &mut rng,
        );
        assert_eq!(
            chosen,
            Move::Play(Combo::new(vec![card(Rank::Nine, Suit::Clubs)]).unwrap())
        );
    }

    #[test]
    fn close_opponent_never_passes_when_a_beating_play_is_legal() {
        let legal_moves = vec![
            Move::Pass,
            Move::Play(Combo::new(vec![card(Rank::Eight, Suit::Clubs)]).unwrap()),
        ];
        let context = context_with_opponent(1, true);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let chosen = EndgameDenial.choose_play(
            &legal_moves,
            DuplicateRule::FirstDealtWins,
            &context,
            &mut rng,
        );
        assert_eq!(
            chosen,
            Move::Play(Combo::new(vec![card(Rank::Eight, Suit::Clubs)]).unwrap())
        );
    }

    #[test]
    fn a_finished_opponents_low_hand_size_does_not_count_as_close() {
        let legal_moves = vec![
            Move::Play(Combo::new(vec![card(Rank::Six, Suit::Clubs)]).unwrap()),
            Move::Play(Combo::new(vec![card(Rank::Nine, Suit::Clubs)]).unwrap()),
        ];
        let context = context_with_opponent(0, false);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let chosen = EndgameDenial.choose_play(
            &legal_moves,
            DuplicateRule::FirstDealtWins,
            &context,
            &mut rng,
        );
        assert_eq!(
            chosen,
            Move::Play(Combo::new(vec![card(Rank::Six, Suit::Clubs)]).unwrap())
        );
    }
}
```

- [ ] **Step 2: Run tests to verify they fail to compile**

Run: `cargo test -p sim --lib endgame_denial`
Expected: fails — `EndgameDenial` doesn't exist yet.

- [ ] **Step 3: Implement `EndgameDenial`**

```rust
//! A pure endgame-denial strategy: every decision is driven by
//! opponents' hand sizes, never by card counting (docs/ROADMAP.md,
//! Phase 6).
//!
//! `CLOSE_TO_FINISHING` (2 cards) is the "about to finish" threshold —
//! see docs/RULES.md's per-player-count deal sizes (as low as ~8-9
//! cards/seat at 6 players with a single deck, up to the mid-30s at 3
//! players with a double deck): 2 cards is deep into the final
//! stretch at every supported table size and deck variant, and larger
//! deals only make it a smaller, still-urgent fraction. 2 rather than
//! 1 gives this seat one extra turn of lead-time to act before the
//! low-hand opponent is already gone.
//!
//! Whenever at least one still-active opponent is at or below that
//! threshold, this seat switches to "deny control": never pass when a
//! legal beating play exists, and prefer the highest legal play — a
//! higher combo is harder for a near-empty hand to match, and winning
//! the trick means leading (and choosing the shape of) the next one,
//! which can lock out an opponent who can't field that shape. That
//! combination is exactly `GreedyHighest`'s existing behavior, so
//! danger mode delegates to it rather than re-deriving the same rule
//! — a cheap proxy for "retain control," not a simulation of future
//! turns. Otherwise (no opponent close), there's nothing yet to deny,
//! so it conserves instead by delegating to `LowestLegal`.

use engine::{Card, DuplicateRule, Move};

use crate::strategies::{GreedyHighest, LowestLegal};
use crate::strategy::{Strategy, TurnContext};

/// A still-active opponent at or below this many cards is "close to
/// finishing" — see the module doc comment for the threshold
/// rationale.
const CLOSE_TO_FINISHING: usize = 2;

#[derive(Debug, Clone, Copy, Default)]
pub struct EndgameDenial;

impl Strategy for EndgameDenial {
    fn name(&self) -> &'static str {
        "EndgameDenial"
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        context: &TurnContext<'_>,
        rng: &mut dyn rand::Rng,
    ) -> Move {
        let danger = context
            .opponents
            .iter()
            .any(|o| o.active && o.hand_size <= CLOSE_TO_FINISHING);
        if danger {
            GreedyHighest.choose_play(legal_moves, duplicate_rule, context, rng)
        } else {
            LowestLegal.choose_play(legal_moves, duplicate_rule, context, rng)
        }
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

In `sim/src/strategies/mod.rs`, add `mod endgame_denial;` and `pub use
endgame_denial::EndgameDenial;`. In `sim/src/lib.rs`, add
`EndgameDenial` to the strategy re-export list.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p sim --lib endgame_denial`
Expected: PASS, all 4 tests green.

- [ ] **Step 5: Run the full `sim` suite**

Run: `cargo test -p sim`
Expected: PASS.

- [ ] **Step 6: Lint and format**

Run: `cargo fmt -p sim` then `cargo clippy -p sim --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 7: Commit**

```bash
git add sim/src/strategies/endgame_denial.rs sim/src/strategies/mod.rs sim/src/lib.rs
git commit -m "sim: add EndgameDenial, a strategy that denies control to opponents close to finishing"
```

---

### Task 5: `cli` wiring, integration coverage, and docs

**Files:**
- Modify: `cli/src/args.rs`
- Modify: `sim/tests/multi_config.rs` (grow the strategy pool)
- Modify: `docs/ROADMAP.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: `sim::{CardCounter, EndgameDenial}` (Tasks 3-4).

- [ ] **Step 1: Read the current files first**

Read `cli/src/args.rs` in full (the exact pattern for adding a
strategy: a `StrategyArg` variant, a `build()` match arm, a doc-comment
mention, a test) and `sim/tests/multi_config.rs` in full (its
`baseline_strategies` helper and rotation formula).

- [ ] **Step 2: `cli/src/args.rs`**

Add `StrategyArg::CardCounter` and `StrategyArg::EndgameDenial`
variants (clap renders them `card-counter`/`endgame-denial` in
kebab-case automatically), their `build()` match arms (`Arc::new(sim::
CardCounter)`, `Arc::new(sim::EndgameDenial)`), extend the
`--strategy` field's doc comment to mention both, and extend the
existing `strategy_arg_builds_matching_strategy_names`-style test with
two more cases.

- [ ] **Step 3: Run cli tests**

Run: `cargo test -p cli`
Expected: PASS.

- [ ] **Step 4: Grow `sim/tests/multi_config.rs`'s strategy pool**

Extend `baseline_strategies`'s pool from 4 to 6 entries (adding
`CardCounter` and `EndgameDenial`), and update its rotation formula's
pool-length arithmetic (`pool.len()`) accordingly — the existing
`.cycle().skip(player_count % pool.len()).take(player_count)` formula
already generalizes to a longer pool without further changes, since it
reads `pool.len()` rather than a hardcoded `4`; confirm this by reading
the current formula before assuming it, and adjust if it's hardcoded
anywhere.

- [ ] **Step 5: Run the sim integration test**

Run: `cargo test -p sim --test multi_config`
Expected: PASS across all four table sizes with the 6-entry pool.

- [ ] **Step 6: Update `docs/ROADMAP.md`**

Read the current file. Insert a new phase entry before "Phase 6 — Web
interface", titled "Phase 6 — Strategic context, card counting, and
endgame denial", summarizing: `engine::Round` gains a play history and
hand-size accessor; `Strategy::choose_play` gains a `TurnContext`
(own hand, opponents' hand sizes, exact unseen-card multiset);
`CardCounter` holds back currently-unbeatable combos; `EndgameDenial`
switches to aggressive play whenever an active opponent's hand size is
low, to deny them easy tricks — kept as two separate strategies so
Phase 4's existing statistics can show which technique helps. Renumber
the existing "Phase 6 — Web interface" to "Phase 7", and its "Parked"
section reference if any explicit phase numbers appear there too.

- [ ] **Step 7: Check for other explicit "Phase 6" references**

Search `docs/*.md` and `README.md` for the literal text "Phase 6" and
update any that refer to the web-interface phase to say "Phase 7"
instead, now that this new phase has taken the number 6.

- [ ] **Step 8: Update `docs/ARCHITECTURE.md`**

Add a short description of `TurnContext`, `Round`'s play-history/
hand-size accessors, and the two new strategies to the existing
`engine`/`sim` section prose, matching that section's existing style
and level of detail.

- [ ] **Step 9: Update `README.md`**

Add `card-counter` and `endgame-denial` to the strategies list, one
line each, matching the existing `hold-back-pairs` entry's length and
style.

- [ ] **Step 10: Full workspace verification**

Run, in order: `cargo fmt --check`; `cargo clippy --workspace
--all-targets -- -D warnings`; `cargo test --workspace`. All three must
be clean. Then run the manual sanity check: `cargo run -p cli --
--player-count 4 --matches 40 --rounds 5 --strategy card-counter
--strategy endgame-denial --strategy lowest-legal --strategy
greedy-highest --seed 1 --output /tmp/results.json` and confirm it
exits successfully with both new strategy names appearing in stdout
and the JSON output.

- [ ] **Step 11: Commit**

```bash
git add cli/src/args.rs sim/tests/multi_config.rs docs/ROADMAP.md docs/ARCHITECTURE.md README.md
git commit -m "cli: add card-counter and endgame-denial strategies; docs: describe Phase 6"
```
