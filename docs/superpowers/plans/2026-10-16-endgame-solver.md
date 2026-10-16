# Endgame Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A search-based strategy, `EndgameSolver`, that plays exactly (max^n, over every consistent distribution of the unseen cards) once few cards are unseen and defers to a fallback strategy before that, plus the tools to learn from it (`cli solve`, `cli solver-gap`).

**Architecture:** `engine` gains one constructor, `Round::from_position`, to rebuild a mid-trick position. A new `sim::solver` module turns a `TurnContext` into an `Observation`, enumerates the consistent deals of the unseen cards, solves each deal exactly with a clone-based max^n search over `engine::Round`, and averages per move. `EndgameSolver` wraps that as a `Strategy` with a fallback; the CLI registers it and adds two commands.

**Tech Stack:** Rust workspace (`engine`, `sim`, `cli`), `rand` 0.10, `rayon`, `clap`; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-16-endgame-solver-design.md` (read its last section, "Revisions after reading the code", first: it overrides earlier statements).

## Global Constraints

- Rust edition 2021, `unsafe_code = "forbid"`, clippy `pedantic` is warn-level: new code must be warning-free under `cargo clippy --workspace --all-targets` (see `docs/CODING_GUIDELINES.md`).
- Everything is opt-in: no existing strategy, golden, reference output or bench checksum may change. `cargo test --workspace` must stay green after every task.
- Supported pass rule for solving: `PassRule::Final` only; under `Free` the solver defers to its fallback.
- The solver must never fail a move: any error path returns the fallback's move.
- The solver may only return a move contained in the `legal_moves` slice it was given.
- Results must be deterministic for identical inputs (no `HashMap` iteration order, no wall-clock, no thread-dependent state in results).
- Role score is linear: `role_score = 1 - 2 * place / (n - 1)` with `place` 0-based (`sim/src/stats_catalog.rs`).
- Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Work in the worktree `/home/sebi/Documents/Code/arschloch-wt-solver` on branch `endgame-solver`; do not push or merge to `main` without asking the repo owner.

## Review Focus

Failure modes the spec implies but a straight reading of the tasks would not test; each has a pinned test in the named task.

1. Table played under `PassRule::Free` -> solver must defer to the fallback, not panic or mis-solve (Task 5).
2. Pass ceilings that contradict every possible deal (stale or dishonest passes) -> `NoConsistentDeal`, fallback used (Task 4).
3. One or more seats already out of the round: places of the remaining seats must be offset by the number of seats out (Task 4).
4. Double deck with `LastDealtWins`: duplicate cards are distinct; the rebuilt round must offer exactly the moves the match runner offered (Task 2).
5. A position with exactly one legal move, or where the solver's own seat leads a fresh trick vs. follows (Task 5).
6. Determinism when the solver sits in a `run_batch` run with several threads (Task 6).

---

## File Structure

- Modify `engine/src/trick.rs` — add `Trick::from_parts` (crate-private).
- Modify `engine/src/round.rs` — add `Round::from_position`.
- Modify `engine/src/round/tests.rs` — tests for `from_position`.
- Create `sim/src/solver/mod.rs` — module wiring, re-exports, `SolverConfig`, `StrategyName`.
- Create `sim/src/solver/observation.rs` — `Observation` (what a seat knows) and rebuilding a `Round` for one deal.
- Create `sim/src/solver/deals.rs` — consistent-deal enumeration and counting.
- Create `sim/src/solver/search.rs` — exact max^n search of one deal.
- Create `sim/src/solver/report.rs` — `analyze`, `MoveReport`, `SolveError`.
- Create `sim/src/solver/notation.rs` — card tokens (`9H`, `10S`, `AC`) shared by `cli solve` and reports.
- Create `sim/src/solver/gap.rs` — disagreement benchmark.
- Create `sim/src/strategies/endgame_solver.rs` — the `Strategy`.
- Modify `sim/src/lib.rs`, `sim/src/strategies/mod.rs` — module/re-exports.
- Modify `cli/src/args.rs` (+ `cli/src/args/tests.rs`) — `endgame-solver[:OPTIONS]` spec.
- Create `cli/src/solve.rs`, `cli/src/solver_gap.rs`; modify `cli/src/main.rs` — commands.
- Create `docs/SOLVER.md`; modify `docs/ROADMAP.md`, `docs/ARCHITECTURE.md`.

---

### Task 1: `Round::from_position` (engine)

**Files:**
- Modify: `engine/src/trick.rs` (add `from_parts` in `impl Trick`, after `new`)
- Modify: `engine/src/round.rs` (add `from_position` in `impl Round`, after `with_pass_rule`)
- Test: `engine/src/round/tests.rs` (append; the file already has `use super::*;`, `use crate::card::{Rank, Suit};` and helpers `card(rank, suit)` and `combo(Vec<Card>)`)

**Interfaces:**
- Consumes: existing `Trick`, `Round` internals.
- Produces: `Round::from_position(hands: Vec<Vec<Card>>, duplicate_rule: DuplicateRule, turn: SeatId, on_table: Option<(SeatId, Combo)>, passed: &[bool]) -> Option<Round>`. Always `PassRule::Final`. `hands` is by seat; a seat with an empty hand is out. `on_table` is `(holder, combo)` of the trick's current play, `None` when `turn` leads a fresh trick. `passed[s]` is true for seats that passed in this trick. The play and pass histories of the result are empty; `finishing_order` holds the empty-handed seats in seat order (their true order is unknowable and irrelevant); `current_trick_leader()` returns the holder (or `turn` on a fresh trick), an approximation.

- [ ] **Step 1: Write the failing tests** (append to `engine/src/round/tests.rs`)

```rust
fn hands3() -> Vec<Vec<Card>> {
    vec![
        vec![card(Rank::Three, Suit::Clubs), card(Rank::Nine, Suit::Clubs)],
        vec![card(Rank::Six, Suit::Clubs), card(Rank::Ten, Suit::Clubs)],
        vec![card(Rank::Four, Suit::Clubs), card(Rank::King, Suit::Clubs)],
    ]
}

#[test]
fn from_position_rejects_inconsistent_positions() {
    let rule = DuplicateRule::FirstDealtWins;
    let five = combo(vec![card(Rank::Five, Suit::Hearts)]);
    // two seats only
    assert!(Round::from_position(vec![hands3()[0].clone(), hands3()[1].clone()], rule, 0, None, &[false; 2]).is_none());
    // turn out of range / turn's hand empty
    assert!(Round::from_position(hands3(), rule, 3, None, &[false; 3]).is_none());
    let mut with_empty = hands3();
    with_empty[1].clear();
    assert!(Round::from_position(with_empty.clone(), rule, 1, None, &[false; 3]).is_none());
    // passed vector of the wrong length
    assert!(Round::from_position(hands3(), rule, 0, None, &[false; 2]).is_none());
    // the seat to move cannot have passed
    assert!(Round::from_position(hands3(), rule, 1, Some((0, five)), &[false, true, false]).is_none());
    // holder cannot be the seat to move, nor a seat that passed
    assert!(Round::from_position(hands3(), rule, 1, Some((1, five)), &[false; 3]).is_none());
    assert!(Round::from_position(hands3(), rule, 1, Some((0, five)), &[true, false, false]).is_none());
    // someone passed on a fresh trick
    assert!(Round::from_position(hands3(), rule, 0, None, &[false, false, true]).is_none());
    // a seat that is out cannot have passed
    assert!(Round::from_position(with_empty, rule, 0, Some((2, five)), &[false, true, false]).is_none());
}

#[test]
fn from_position_mid_trick_plays_on_like_a_live_round() {
    let rule = DuplicateRule::FirstDealtWins;
    let five = combo(vec![card(Rank::Five, Suit::Hearts)]);
    // Seat 0 played a 5; seat 1 to move.
    let mut round = Round::from_position(hands3(), rule, 1, Some((0, five)), &[false; 3]).unwrap();
    assert_eq!(round.seat_to_move(), Some(1));
    assert_eq!(round.current_combo(), Some(&five));
    let moves = round.legal_moves();
    assert!(moves.contains(&Move::Pass));
    assert!(moves.contains(&Move::Play(combo(vec![card(Rank::Six, Suit::Clubs)]))));
    assert!(moves.contains(&Move::Play(combo(vec![card(Rank::Ten, Suit::Clubs)]))));
    assert!(!moves.contains(&Move::Play(combo(vec![card(Rank::Three, Suit::Clubs)]))));

    // Seat 1 passes (final: out of the trick), seat 2 passes: seat 0 leads again.
    round.submit_move(1, Move::Pass).unwrap();
    assert!(round.has_passed(1));
    assert_eq!(round.seat_to_move(), Some(2));
    round.submit_move(2, Move::Pass).unwrap();
    assert_eq!(round.current_combo(), None);
    assert_eq!(round.seat_to_move(), Some(0));
}

#[test]
fn from_position_with_a_seat_already_out_finishes_the_round() {
    let rule = DuplicateRule::FirstDealtWins;
    let mut hands = hands3();
    hands[0].clear(); // seat 0 is out
    let mut round = Round::from_position(hands, rule, 1, None, &[false; 3]).unwrap();
    // Seat 1 leads and plays its last-but-one card, seat 2 passes, seat 1 leads again.
    round.submit_move(1, Move::Play(combo(vec![card(Rank::Six, Suit::Clubs)]))).unwrap();
    round.submit_move(2, Move::Pass).unwrap();
    round.submit_move(1, Move::Play(combo(vec![card(Rank::Ten, Suit::Clubs)]))).unwrap();
    assert!(round.is_complete());
    // Seat 0 was out before, seat 1 emptied next, seat 2 is last.
    assert_eq!(round.finishing_order(), &[0, 1, 2]);
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p engine from_position`
Expected: FAIL (compile error: no function `from_position`).

- [ ] **Step 3: Implement**

In `engine/src/trick.rs`, inside `impl Trick` after `new`:

```rust
    /// A trick in progress under `PassRule::Final`, rebuilt from what a seat
    /// can observe: who leads/holds, whose turn it is and who has passed.
    /// `winner` is the holder of the combo on the table (`None` on a fresh
    /// trick).
    pub(crate) fn from_parts(
        leader: SeatId,
        turn: SeatId,
        winner: Option<SeatId>,
        passed: [bool; 6],
    ) -> Self {
        Self {
            leader,
            turn,
            winner,
            passes_since_last_play: 0,
            required_passes: 0,
            rule: PassRule::Final,
            passed,
        }
    }
```

In `engine/src/round.rs`, inside `impl Round` after `with_pass_rule`:

```rust
    /// Rebuilds a round in progress under [`PassRule::Final`] from what a
    /// seat can observe, for search: `hands` by seat (an empty hand is a seat
    /// that is already out), `turn` the seat to move, `on_table` the holder
    /// and combo of the trick's current play (`None`: `turn` leads a fresh
    /// trick) and `passed` by seat (seats out of the trick).
    ///
    /// The play and pass histories start empty, the seats already out are
    /// entered in the finishing order in seat order (their real order is not
    /// observable and does not affect the places of the seats still playing),
    /// and [`Round::current_trick_leader`] returns the holder (or `turn` on
    /// a fresh trick). Returns `None` for an inconsistent position: not 3-6
    /// seats, `passed` of the wrong length, fewer than two seats still in,
    /// `turn` out of range, out, or having passed, a holder equal to `turn`
    /// or having passed, a passed seat that is out, or passes on a fresh
    /// trick.
    #[must_use]
    pub fn from_position(
        hands: Vec<Vec<Card>>,
        duplicate_rule: DuplicateRule,
        turn: SeatId,
        on_table: Option<(SeatId, Combo)>,
        passed: &[bool],
    ) -> Option<Self> {
        let n = hands.len();
        if !(3..=6).contains(&n) || passed.len() != n || usize::from(turn) >= n {
            return None;
        }
        let turn_index = usize::from(turn);
        if hands[turn_index].is_empty() || passed[turn_index] {
            return None;
        }
        if hands.iter().filter(|hand| !hand.is_empty()).count() < 2 {
            return None;
        }
        if passed
            .iter()
            .zip(&hands)
            .any(|(&has_passed, hand)| has_passed && hand.is_empty())
        {
            return None;
        }
        let holder = match on_table {
            Some((holder, _)) => {
                let holder_index = usize::from(holder);
                if holder_index >= n || holder == turn || passed[holder_index] {
                    return None;
                }
                Some(holder)
            }
            None => {
                if passed.iter().any(|&has_passed| has_passed) {
                    return None;
                }
                None
            }
        };
        let mut passed_seats = [false; 6];
        passed_seats[..n].copy_from_slice(passed);
        let finishing_order = (0..n)
            .filter(|&seat| hands[seat].is_empty())
            .map(|seat| SeatId::try_from(seat).expect("seats are at most 6"))
            .collect();
        Some(Self {
            hands,
            duplicate_rule,
            finishing_order,
            current_combo: on_table.map(|(_, combo)| combo),
            trick: Trick::from_parts(holder.unwrap_or(turn), turn, holder, passed_seats),
            pass_rule: PassRule::Final,
            play_history: Vec::new(),
            pass_history: Vec::new(),
        })
    }
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p engine && cargo clippy -p engine --all-targets`
Expected: all PASS, no warnings. (If a test's expectation about `finishing_order` fails, re-read `check_for_single_seat_remaining`: the last remaining seat is appended automatically when one seat is left.)

- [ ] **Step 5: Commit**

```bash
git add engine/src/trick.rs engine/src/round.rs engine/src/round/tests.rs
git commit -m "engine: Round::from_position rebuilds a mid-trick position for search

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `Observation`, notation and the equivalence check (sim)

**Files:**
- Create: `sim/src/solver/mod.rs`, `sim/src/solver/observation.rs`, `sim/src/solver/notation.rs`
- Modify: `sim/src/lib.rs` (add `pub mod solver;` in alphabetical position after `pub mod skill;`)

**Interfaces:**
- Consumes: `TurnContext`, `OpponentHand`, `TableInfo` (`sim/src/strategy.rs`), `PassCeilings` (`sim/src/hand_reading.rs`), `Round::from_position` (Task 1).
- Produces:
  - `pub struct Observation { pub duplicate_rule: DuplicateRule, pub seat: SeatId, pub player_count: u8, pub hand: Vec<Card>, pub hand_sizes: Vec<usize>, pub in_trick: Vec<bool>, pub ceilings: Vec<PassCeilings>, pub unseen: Vec<Card>, pub on_table: Option<(SeatId, Combo)> }` — all `Vec`s indexed by seat (own seat included; own `ceilings` entry is `PassCeilings::default()`).
  - `Observation::from_context(context: &TurnContext<'_>, duplicate_rule: DuplicateRule) -> Option<Observation>` — `None` unless: `table.player_count` in 3..=6, `table.pass_rule == PassRule::Final`, `opponents.len() + 1 == player_count`, `unseen_cards.len()` equals the sum of the opponents' hand sizes, and `combo_holder.is_some() == current_combo.is_some()`.
  - `Observation::seats_out(&self) -> usize` (seats with hand size 0).
  - `Observation::round_for(&self, deal: &[Vec<Card>]) -> Option<Round>` — `deal` is by seat: for every seat the hand it holds in this deal (own seat: `self.hand`, seats out: empty).
  - `notation::card_token(&Card) -> String` (`"9H"`, `"10S"`, `"AC"`, ranks `2-9,10,J,Q,K,A`, suits `D H S C`) and `notation::parse_card_token(&str) -> Option<(Rank, Suit)>` (case-insensitive).

- [ ] **Step 1: Create the module skeleton**

`sim/src/solver/mod.rs`:

```rust
//! Exact endgame search (docs/superpowers/specs/2026-10-16-endgame-solver-design.md).
//!
//! From one seat's point of view: enumerate every distribution of the
//! unseen cards the other seats could hold (`deals`), solve each one exactly
//! (`search`, max^n) and average per move (`report`).

pub mod notation;
pub mod observation;

pub use observation::Observation;
```

Add `pub mod solver;` to `sim/src/lib.rs`.

`sim/src/solver/notation.rs`:

```rust
//! Short card names for reports and the `solve` command: `9H`, `10S`, `AC`.

use engine::{Card, Rank, Suit};

const RANKS: [(Rank, &str); 13] = [
    (Rank::Two, "2"),
    (Rank::Three, "3"),
    (Rank::Four, "4"),
    (Rank::Five, "5"),
    (Rank::Six, "6"),
    (Rank::Seven, "7"),
    (Rank::Eight, "8"),
    (Rank::Nine, "9"),
    (Rank::Ten, "10"),
    (Rank::Jack, "J"),
    (Rank::Queen, "Q"),
    (Rank::King, "K"),
    (Rank::Ace, "A"),
];

const SUITS: [(Suit, char); 4] = [
    (Suit::Diamonds, 'D'),
    (Suit::Hearts, 'H'),
    (Suit::Spades, 'S'),
    (Suit::Clubs, 'C'),
];

/// `9H`, `10S`, `AC`.
#[must_use]
pub fn card_token(card: &Card) -> String {
    let rank = RANKS.iter().find(|(r, _)| *r == card.rank).map_or("?", |(_, t)| *t);
    let suit = SUITS.iter().find(|(s, _)| *s == card.suit).map_or('?', |(_, c)| *c);
    format!("{rank}{suit}")
}

/// Inverse of [`card_token`], case-insensitive.
#[must_use]
pub fn parse_card_token(token: &str) -> Option<(Rank, Suit)> {
    let token = token.trim().to_ascii_uppercase();
    let suit_char = token.chars().last()?;
    let rank_text = &token[..token.len() - suit_char.len_utf8()];
    let suit = SUITS.iter().find(|(_, c)| *c == suit_char).map(|(s, _)| *s)?;
    let rank = RANKS.iter().find(|(_, t)| *t == rank_text).map(|(r, _)| *r)?;
    Some((rank, suit))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_round_trip_for_every_card() {
        for variant in [engine::DeckVariant::Single, engine::DeckVariant::Double] {
            for card in engine::standard_deck(variant) {
                assert_eq!(
                    parse_card_token(&card_token(&card)),
                    Some((card.rank, card.suit))
                );
            }
        }
    }

    #[test]
    fn parsing_is_case_insensitive_and_rejects_garbage() {
        assert_eq!(parse_card_token("10s"), Some((Rank::Ten, Suit::Spades)));
        assert_eq!(parse_card_token("ac"), Some((Rank::Ace, Suit::Clubs)));
        assert_eq!(parse_card_token("1H"), None);
        assert_eq!(parse_card_token("9X"), None);
        assert_eq!(parse_card_token(""), None);
    }
}
```

- [ ] **Step 2: Write the failing equivalence test and the unit tests** in `sim/src/solver/observation.rs`

```rust
//! What one seat knows at its turn, as the solver needs it.

use engine::{Card, Combo, DuplicateRule, PassRule, Round, SeatId};

use crate::hand_reading::PassCeilings;
use crate::strategy::TurnContext;

/// A seat's view of the round. Every `Vec` is indexed by seat and includes
/// the seat itself.
#[derive(Debug, Clone)]
pub struct Observation {
    pub duplicate_rule: DuplicateRule,
    pub seat: SeatId,
    pub player_count: u8,
    pub hand: Vec<Card>,
    pub hand_sizes: Vec<usize>,
    /// Active and not passed in the current trick (own seat: `true`).
    pub in_trick: Vec<bool>,
    pub ceilings: Vec<PassCeilings>,
    /// Every card some other seat holds right now.
    pub unseen: Vec<Card>,
    /// Holder and combo of the trick's current play; `None` when this seat leads.
    pub on_table: Option<(SeatId, Combo)>,
}

impl Observation {
    /// `None` when the context cannot be solved from (see the plan, Task 2):
    /// wrong pass rule, unrequested/inconsistent context parts.
    #[must_use]
    pub fn from_context(
        context: &TurnContext<'_>,
        duplicate_rule: DuplicateRule,
    ) -> Option<Self> {
        let table = context.table;
        let n = usize::from(table.player_count);
        if !(3..=6).contains(&n) || table.pass_rule != PassRule::Final {
            return None;
        }
        if context.opponents.len() + 1 != n {
            return None;
        }
        let opponents_total: usize = context.opponents.iter().map(|o| o.hand_size).sum();
        if context.unseen_cards.len() != opponents_total {
            return None;
        }
        if context.combo_holder.is_some() != context.current_combo.is_some() {
            return None;
        }
        let mut hand_sizes = vec![0usize; n];
        let mut in_trick = vec![false; n];
        let mut ceilings = vec![PassCeilings::default(); n];
        hand_sizes[usize::from(context.seat)] = context.hand.len();
        in_trick[usize::from(context.seat)] = true;
        for o in &context.opponents {
            let s = usize::from(o.seat);
            if s >= n || s == usize::from(context.seat) {
                return None;
            }
            hand_sizes[s] = o.hand_size;
            in_trick[s] = o.in_trick;
            ceilings[s] = o.pass_ceilings;
        }
        Some(Self {
            duplicate_rule,
            seat: context.seat,
            player_count: table.player_count,
            hand: context.hand.to_vec(),
            hand_sizes,
            in_trick,
            ceilings,
            unseen: context.unseen_cards.clone(),
            on_table: context.combo_holder.zip(context.current_combo.copied()),
        })
    }

    /// Seats that already emptied their hand.
    #[must_use]
    pub fn seats_out(&self) -> usize {
        self.hand_sizes.iter().filter(|&&size| size == 0).count()
    }

    /// The round with `deal` (hands by seat) in place, or `None` if the
    /// position is inconsistent (see [`Round::from_position`]).
    #[must_use]
    pub fn round_for(&self, deal: &[Vec<Card>]) -> Option<Round> {
        let passed: Vec<bool> = (0..usize::from(self.player_count))
            .map(|s| !self.in_trick[s] && self.hand_sizes[s] > 0)
            .collect();
        Round::from_position(
            deal.to_vec(),
            self.duplicate_rule,
            self.seat,
            self.on_table,
            &passed,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand_reading::PassTracker;
    use crate::match_runner::turn_context_for;
    use crate::strategy::ContextNeeds;
    use crate::test_support::for_each_state;
    use rand::{RngExt, SeedableRng};

    /// Plays both rounds on with the same random moves and returns the
    /// finishing places of the seats that were still in at the start.
    fn finish(round: &mut Round, seed: u64) -> Vec<(SeatId, usize)> {
        let still_in: Vec<SeatId> = (0..round.finishing_order().len().max(1))
            .map(|_| 0)
            .collect::<Vec<_>>()
            .iter()
            .map(|_| 0)
            .collect();
        drop(still_in);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        while !round.is_complete() {
            let seat = round.seat_to_move().unwrap();
            let moves = round.legal_moves();
            let mv = moves[rng.random_range(0..moves.len())];
            round.submit_move(seat, mv).unwrap();
        }
        round
            .finishing_order()
            .iter()
            .enumerate()
            .map(|(place, &seat)| (seat, place))
            .collect()
    }

    #[test]
    fn a_rebuilt_round_behaves_like_the_live_round() {
        let mut checked = 0;
        for_each_state(11, 2, |state, rng| {
            let round = state.round;
            if round.pass_rule() != PassRule::Final {
                return;
            }
            let seat = round.seat_to_move().unwrap();
            let mut tracker = PassTracker::new(usize::from(state.players), state.rule);
            let context = turn_context_for(
                round,
                seat,
                state.players,
                state.round_deck,
                &mut tracker,
                ContextNeeds::ALL,
            );
            let observation = Observation::from_context(&context, state.rule).unwrap();
            let hands: Vec<Vec<Card>> = (0..state.players).map(|s| round.hand(s).to_vec()).collect();
            let mut rebuilt = observation.round_for(&hands).unwrap();

            assert_eq!(rebuilt.seat_to_move(), round.seat_to_move());
            assert_eq!(rebuilt.legal_moves(), round.legal_moves());
            assert_eq!(rebuilt.current_combo(), round.current_combo());

            let seed = rng.random_range(0..u64::MAX);
            let mut live = round.clone();
            let live_order = finish(&mut live, seed);
            let rebuilt_order = finish(&mut rebuilt, seed);
            // The seats still in when the position was taken finish in the same
            // relative order; seats already out come first in the rebuilt round.
            let out_before = observation.seats_out();
            let relative = |order: &[(SeatId, usize)], skip: usize| -> Vec<SeatId> {
                order.iter().skip(skip).map(|&(s, _)| s).collect()
            };
            assert_eq!(
                relative(&live_order, round.finishing_order().len()),
                relative(&rebuilt_order, out_before)
            );
            checked += 1;
        });
        assert!(checked > 100, "only {checked} states were checked");
    }

    #[test]
    fn from_context_rejects_a_free_pass_rule_and_unrequested_context() {
        for_each_state(5, 1, |state, _| {
            let round = state.round;
            let seat = round.seat_to_move().unwrap();
            let mut tracker = PassTracker::new(usize::from(state.players), state.rule);
            let mut context = turn_context_for(
                round,
                seat,
                state.players,
                state.round_deck,
                &mut tracker,
                ContextNeeds::ALL,
            );
            context.table.pass_rule = engine::PassRule::Free;
            assert!(Observation::from_context(&context, state.rule).is_none());
            let bare = turn_context_for(
                round,
                seat,
                state.players,
                state.round_deck,
                &mut tracker,
                ContextNeeds::NONE,
            );
            assert!(Observation::from_context(&bare, state.rule).is_none());
        });
    }
}
```

Clean up while writing: the `finish` helper above contains a leftover no-op (`still_in`) from drafting; delete those lines so it only runs the round to the end and returns the finishing order.

- [ ] **Step 3: Run to verify failure**

Run: `cargo test -p sim solver::`
Expected: first FAIL (e.g. unresolved import or failing assertion); fix compile errors only, not assertions.

- [ ] **Step 4: Make it pass**

The code above is the implementation. If `a_rebuilt_round_behaves_like_the_live_round` fails on `legal_moves` equality, the cause is a rebuilt-state difference (turn, holder, passed): debug with `assert_eq!` on `has_passed(s)` for every seat. Ties between live and rebuilt finishing order cannot arise (same moves, same state).

Note for the `for_each_state` `State` fields: `round`, `round_deck`, `rule`, `players` (see `sim/src/test_support.rs`). `turn_context_for` is `pub(crate)` in `sim/src/match_runner.rs`; `PassTracker` is `pub` in `sim/src/hand_reading.rs`.

- [ ] **Step 5: Run everything and commit**

Run: `cargo test -p sim solver:: && cargo clippy -p sim --all-targets`
Expected: PASS, no warnings.

```bash
git add sim/src/lib.rs sim/src/solver
git commit -m "sim: solver Observation, card notation and live/rebuilt round equivalence test

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Deal enumeration

**Files:**
- Create: `sim/src/solver/deals.rs`
- Modify: `sim/src/solver/mod.rs` (add `pub mod deals;`)

**Interfaces:**
- Consumes: `Observation` (Task 2), `PassCeilings::ceiling(size, rule) -> Option<Card>`.
- Produces:
  - `pub fn count_deals(observation: &Observation) -> u128` — multinomial count of ignoring the ceilings (an upper bound); `u128::MAX` if it would overflow.
  - `pub fn for_each_deal(observation: &Observation, visit: impl FnMut(&[Vec<Card>]) -> bool) -> u64` — calls `visit` for every consistent deal (hands by seat; own seat gets `observation.hand`; seats out get `vec![]`); `visit` returns `false` to stop early; returns the number of deals visited. Deterministic order.
  - A deal is *consistent* when no opponent holds a rank group of at least `s` cards whose highest card beats that seat's ceiling at size `s` (`ceiling(s)` is "the lowest top card it is known unable to beat").

- [ ] **Step 1: Write failing tests** (in `sim/src/solver/deals.rs`)

```rust
//! Every way the unseen cards can be split among the other seats.

use std::cmp::Ordering;

use engine::{Card, Combo, Rank, Suit};

use super::observation::Observation;

fn count(observation: &Observation) -> u128 { unimplemented!() }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand_reading::{read_pass_ceilings, PassCeilings};
    use engine::DuplicateRule;

    fn card(rank: Rank, suit: Suit, index: u8) -> Card {
        Card::new(rank, suit, index)
    }

    /// 3 seats; seat 0 (us) holds `hand`; seats 1 and 2 hold `sizes`.
    fn observation(hand: Vec<Card>, sizes: [usize; 2], unseen: Vec<Card>) -> Observation {
        Observation {
            duplicate_rule: DuplicateRule::FirstDealtWins,
            seat: 0,
            player_count: 3,
            hand_sizes: vec![hand.len(), sizes[0], sizes[1]],
            hand,
            in_trick: vec![true; 3],
            ceilings: vec![PassCeilings::default(); 3],
            unseen,
            on_table: None,
        }
    }

    fn three_cards() -> Vec<Card> {
        vec![
            card(Rank::Four, Suit::Clubs, 0),
            card(Rank::Eight, Suit::Hearts, 1),
            card(Rank::King, Suit::Spades, 2),
        ]
    }

    #[test]
    fn splitting_three_distinct_cards_one_two_gives_three_deals() {
        let o = observation(vec![card(Rank::Two, Suit::Clubs, 3)], [1, 2], three_cards());
        assert_eq!(count_deals(&o), 3);
        let mut seen = Vec::new();
        let n = for_each_deal(&o, |hands| {
            assert_eq!(hands[0], o.hand);
            assert_eq!(hands[1].len(), 1);
            assert_eq!(hands[2].len(), 2);
            seen.push(hands.to_vec());
            true
        });
        assert_eq!(n, 3);
        seen.sort_by_key(|h| format!("{h:?}"));
        seen.dedup();
        assert_eq!(seen.len(), 3, "every deal is distinct");
    }

    #[test]
    fn a_seat_that_is_out_holds_nothing_and_the_count_matches_a_multinomial() {
        // sizes 2 and 0 (seat 2 is out): one way to give 2 cards to seat 1... choose which 2 of 3? No: unseen must equal the sum.
        let unseen = vec![card(Rank::Four, Suit::Clubs, 0), card(Rank::Eight, Suit::Hearts, 1)];
        let o = observation(vec![card(Rank::Two, Suit::Clubs, 3)], [2, 0], unseen);
        assert_eq!(count_deals(&o), 1);
        let n = for_each_deal(&o, |hands| {
            assert!(hands[2].is_empty());
            true
        });
        assert_eq!(n, 1);
        // 4 unseen, sizes 2/2 -> C(4,2) = 6
        let four = vec![
            card(Rank::Four, Suit::Clubs, 0),
            card(Rank::Eight, Suit::Hearts, 1),
            card(Rank::Nine, Suit::Hearts, 2),
            card(Rank::King, Suit::Spades, 4),
        ];
        assert_eq!(count_deals(&observation(vec![card(Rank::Two, Suit::Clubs, 3)], [2, 2], four)), 6);
    }

    #[test]
    fn a_pass_ceiling_removes_deals_that_hold_something_that_beats_it() {
        // Seat 1 passed on a single 9 of hearts: it holds nothing above it, so
        // the King of spades must be seat 2's.
        let nine = Combo::new(vec![card(Rank::Nine, Suit::Hearts, 9)]).unwrap();
        let mut o = observation(vec![card(Rank::Two, Suit::Clubs, 3)], [1, 2], three_cards());
        o.ceilings = read_pass_ceilings(3, &[], &[(1, nine, 0)], DuplicateRule::FirstDealtWins);
        let mut deals = Vec::new();
        let n = for_each_deal(&o, |hands| {
            deals.push(hands.to_vec());
            true
        });
        // Seat 1 gets exactly one of {4C, 8H}: 2 deals, the King goes to seat 2.
        assert_eq!(n, 2);
        assert!(deals.iter().all(|h| h[2].contains(&card(Rank::King, Suit::Spades, 2))));
    }

    #[test]
    fn contradictory_ceilings_leave_no_deal() {
        let three = Combo::new(vec![card(Rank::Three, Suit::Hearts, 9)]).unwrap();
        let mut o = observation(vec![card(Rank::Two, Suit::Clubs, 3)], [1, 2], three_cards());
        // Both opponents passed on a 3: neither holds anything higher, but
        // every unseen card is higher.
        o.ceilings = read_pass_ceilings(
            3,
            &[],
            &[(1, three, 0), (2, three, 0)],
            DuplicateRule::FirstDealtWins,
        );
        assert_eq!(for_each_deal(&o, |_| true), 0);
    }

    #[test]
    fn visiting_can_stop_early_and_the_order_is_deterministic() {
        let o = observation(vec![card(Rank::Two, Suit::Clubs, 3)], [1, 2], three_cards());
        let mut first = Vec::new();
        for_each_deal(&o, |h| {
            first.push(h.to_vec());
            true
        });
        let mut second = Vec::new();
        for_each_deal(&o, |h| {
            second.push(h.to_vec());
            true
        });
        assert_eq!(first, second);
        assert_eq!(for_each_deal(&o, |_| false), 1);
    }
}
```

(The `count` stub at the top is only there so the first run fails on behaviour; replace it with the real code in Step 3.)

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p sim solver::deals`
Expected: FAIL (`count_deals` / `for_each_deal` not defined).

- [ ] **Step 3: Implement** (replace the stub at the top of the file)

```rust
use std::cmp::Ordering;

use engine::Card;

use super::observation::Observation;

/// Number of ways to split the unseen cards among the other seats by hand
/// size, ignoring pass ceilings (an upper bound of the consistent deals).
/// `u128::MAX` on overflow.
#[must_use]
pub fn count_deals(observation: &Observation) -> u128 {
    let mut remaining = observation.unseen.len();
    let mut total: u128 = 1;
    for (seat, &size) in observation.hand_sizes.iter().enumerate() {
        if seat == usize::from(observation.seat) {
            continue;
        }
        total = match total.checked_mul(binomial(remaining, size)) {
            Some(value) => value,
            None => return u128::MAX,
        };
        remaining -= size;
    }
    total
}

fn binomial(n: usize, k: usize) -> u128 {
    let k = k.min(n - k.min(n));
    let mut result: u128 = 1;
    for i in 0..k {
        // result * (n - i) / (i + 1) stays integral at every step.
        result = match result.checked_mul((n - i) as u128) {
            Some(value) => value / (i as u128 + 1),
            None => return u128::MAX,
        };
    }
    result
}

/// Calls `visit` for every consistent deal; `visit` returns `false` to
/// stop. Returns how many deals were visited.
pub fn for_each_deal(observation: &Observation, mut visit: impl FnMut(&[Vec<Card>]) -> bool) -> u64 {
    let n = usize::from(observation.player_count);
    let mut hands: Vec<Vec<Card>> = (0..n).map(|_| Vec::new()).collect();
    hands[usize::from(observation.seat)].clone_from(&observation.hand);
    let capacity: Vec<usize> = (0..n)
        .map(|s| if s == usize::from(observation.seat) { 0 } else { observation.hand_sizes[s] })
        .collect();
    let mut walker = Walker {
        observation,
        capacity,
        hands,
        visited: 0,
        stopped: false,
    };
    walker.assign(0, &mut visit);
    walker.visited
}

struct Walker<'a> {
    observation: &'a Observation,
    capacity: Vec<usize>,
    hands: Vec<Vec<Card>>,
    visited: u64,
    stopped: bool,
}

impl Walker<'_> {
    /// Gives `unseen[index]` to each seat with room left, in seat order.
    fn assign(&mut self, index: usize, visit: &mut impl FnMut(&[Vec<Card>]) -> bool) {
        if self.stopped {
            return;
        }
        if index == self.observation.unseen.len() {
            if self.consistent() {
                self.visited += 1;
                if !visit(&self.hands) {
                    self.stopped = true;
                }
            }
            return;
        }
        let card = self.observation.unseen[index];
        for seat in 0..self.hands.len() {
            if self.capacity[seat] == 0 {
                continue;
            }
            self.capacity[seat] -= 1;
            self.hands[seat].push(card);
            self.assign(index + 1, visit);
            self.hands[seat].pop();
            self.capacity[seat] += 1;
            if self.stopped {
                return;
            }
        }
    }

    /// No opponent holds a combo that beats its pass ceiling: for every
    /// rank group of `g` cards and every size `s <= g`, the group's highest
    /// card must not beat the ceiling at size `s`.
    fn consistent(&self) -> bool {
        let rule = self.observation.duplicate_rule;
        for (seat, hand) in self.hands.iter().enumerate() {
            if seat == usize::from(self.observation.seat) || hand.is_empty() {
                continue;
            }
            let ceilings = &self.observation.ceilings[seat];
            for (i, card) in hand.iter().enumerate() {
                let same_rank: Vec<&Card> = hand.iter().filter(|c| c.rank == card.rank).collect();
                let top = same_rank
                    .iter()
                    .copied()
                    .max_by(|a, b| a.compare(b, rule))
                    .expect("the group holds at least the card itself");
                if top != card {
                    continue; // look at each group once, via its highest card
                }
                let _ = i;
                for size in 1..=same_rank.len() {
                    if let Some(ceiling) = ceilings.ceiling(size, rule) {
                        if top.compare(&ceiling, rule) == Ordering::Greater {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }
}
```

Tidy while compiling: remove the unused `i`/`let _ = i;` (iterate `for card in hand`), and the leftover unused imports from the stub (`Combo`, `Rank`, `Suit` are only used by tests; import them in the test module instead).

- [ ] **Step 4: Run tests**

Run: `cargo test -p sim solver::deals && cargo clippy -p sim --all-targets`
Expected: PASS, no warnings.

- [ ] **Step 5: Commit**

```bash
git add sim/src/solver
git commit -m "sim: enumerate the deals of the unseen cards that fit the pass ceilings

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Exact max^n search and `analyze`

**Files:**
- Create: `sim/src/solver/search.rs`, `sim/src/solver/report.rs`
- Modify: `sim/src/solver/mod.rs` (add `pub mod report; pub mod search;`, define `SolverConfig` and re-exports)

**Interfaces:**
- Consumes: `Observation`, `for_each_deal`, `count_deals`, `Round`, `Move`.
- Produces:
  - In `search.rs`: `pub type Places = [u8; 6]` (0-based finishing place by seat, among all seats incl. those already out); `pub struct BudgetExceeded;`; `pub struct Search { .. }` with `Search::new(node_budget: u64) -> Search`, `Search::root_values(&mut self, round: &Round) -> Result<Vec<(Move, Places)>, BudgetExceeded>` (one entry per legal move of the seat to move, in the engine's move order), `Search::nodes_used(&self) -> u64`. Rule: every seat to move chooses the move that gives *itself* the lowest place; ties go to the first move in engine order.
  - In `mod.rs`: `pub struct SolverConfig { pub max_unseen: usize, pub node_budget: u64 }` with `Default` (`max_unseen: 6`, `node_budget: 2_000_000`).
  - In `report.rs`: `pub struct MoveStats { pub mv: Move, pub mean_place: f64, pub best_place: u8, pub worst_place: u8, pub best_share: f64 }`; `pub struct MoveReport { pub moves: Vec<MoveStats>, pub deals: u64, pub nodes: u64, pub player_count: u8 }` with `MoveReport::best(&self) -> &MoveStats` (lowest `mean_place`, first on ties) and `MoveReport::score(&self, stats: &MoveStats) -> f64` (`1 - 2 * mean_place / (n - 1)`); `pub enum SolveError { TooManyUnseen, BudgetExceeded, NoConsistentDeal, InvalidPosition }`; `pub fn analyze(observation: &Observation, config: &SolverConfig) -> Result<MoveReport, SolveError>`. `mean_place` is a 0-based place (0 = first out) over all consistent deals, uniform weights.

- [ ] **Step 1: Write failing tests**

In `sim/src/solver/search.rs` (tests at the bottom; helper builds rounds with `Round::from_position`):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use engine::{Card, Combo, DuplicateRule, Rank, Suit};

    fn c(rank: Rank, suit: Suit, i: u8) -> Card { Card::new(rank, suit, i) }
    fn play(cards: &[Card]) -> Move { Move::Play(Combo::from_slice(cards).unwrap()) }
    const RULE: DuplicateRule = DuplicateRule::FirstDealtWins;

    #[test]
    fn the_leader_with_a_winning_last_card_finishes_first() {
        // Seat 0 holds a single Ace; seats 1 and 2 hold low cards. Seat 0 leads and goes out.
        let hands = vec![
            vec![c(Rank::Ace, Suit::Clubs, 0)],
            vec![c(Rank::Three, Suit::Clubs, 1), c(Rank::Four, Suit::Clubs, 2)],
            vec![c(Rank::Five, Suit::Clubs, 3), c(Rank::Six, Suit::Clubs, 4)],
        ];
        let round = Round::from_position(hands, RULE, 0, None, &[false; 3]).unwrap();
        let mut search = Search::new(1_000_000);
        let values = search.root_values(&round).unwrap();
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].1[0], 0, "seat 0 finishes first");
    }

    #[test]
    fn passing_to_keep_a_pair_beats_breaking_it() {
        // Seat 1 follows a lone 5 held by seat 0; seat 1 holds {9,9} and a 6.
        // Playing the 6 to win the trick lets seat 1 lead the pair next and go out
        // second... verify the solver prefers whichever gives the lower place and
        // that its choice is no worse than every alternative (property).
        let hands = vec![
            vec![c(Rank::Seven, Suit::Clubs, 0), c(Rank::Eight, Suit::Clubs, 1)],
            vec![c(Rank::Six, Suit::Clubs, 2), c(Rank::Nine, Suit::Clubs, 3), c(Rank::Nine, Suit::Hearts, 4)],
            vec![c(Rank::Four, Suit::Clubs, 5), c(Rank::Ten, Suit::Clubs, 6)],
        ];
        let on_table = Some((0u8, Combo::from_slice(&[c(Rank::Five, Suit::Hearts, 7)]).unwrap()));
        let round = Round::from_position(hands, RULE, 1, on_table, &[false; 3]).unwrap();
        let mut search = Search::new(1_000_000);
        let values = search.root_values(&round).unwrap();
        assert!(values.len() >= 2);
        // Every value is a permutation of places 0..3.
        for (_, places) in &values {
            let mut p = places[..3].to_vec();
            p.sort_unstable();
            assert_eq!(p, vec![0, 1, 2]);
        }
    }

    #[test]
    fn a_too_small_budget_is_reported_not_guessed() {
        let hands = vec![
            vec![c(Rank::Seven, Suit::Clubs, 0), c(Rank::Eight, Suit::Clubs, 1)],
            vec![c(Rank::Six, Suit::Clubs, 2), c(Rank::Nine, Suit::Clubs, 3)],
            vec![c(Rank::Four, Suit::Clubs, 5), c(Rank::Ten, Suit::Clubs, 6)],
        ];
        let round = Round::from_position(hands, RULE, 0, None, &[false; 3]).unwrap();
        assert!(Search::new(2).root_values(&round).is_err());
    }

    /// An independent, deliberately naive reference: plays every line out with
    /// recursion over `Round::legal_moves` and returns the places vector of
    /// the max^n line.
    fn reference(round: &Round) -> Places {
        if round.is_complete() {
            let mut p = [0u8; 6];
            for (place, &seat) in round.finishing_order().iter().enumerate() {
                p[usize::from(seat)] = u8::try_from(place).unwrap();
            }
            return p;
        }
        let seat = round.seat_to_move().unwrap();
        let mut best: Option<Places> = None;
        for mv in round.legal_moves() {
            let mut next = round.clone();
            next.submit_move(seat, mv).unwrap();
            let value = reference(&next);
            if best.is_none_or(|b| value[usize::from(seat)] < b[usize::from(seat)]) {
                best = Some(value);
            }
        }
        best.unwrap()
    }

    #[test]
    fn the_search_agrees_with_the_naive_reference_on_random_small_positions() {
        use crate::test_support::for_each_state;
        let mut checked = 0;
        for_each_state(21, 2, |state, _| {
            let round = state.round;
            let cards: usize = (0..state.players).map(|s| round.hand_size(s)).sum();
            if cards > 9 || round.pass_rule() != engine::PassRule::Final {
                return;
            }
            let seat = round.seat_to_move().unwrap();
            let mut search = Search::new(5_000_000);
            let values = search.root_values(round).unwrap();
            for (mv, places) in values {
                let mut after = round.clone();
                after.submit_move(seat, mv).unwrap();
                assert_eq!(places, reference(&after));
            }
            checked += 1;
        });
        assert!(checked > 20, "only {checked} positions were small enough");
    }
}
```

In `sim/src/solver/report.rs` (tests):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand_reading::{read_pass_ceilings, PassCeilings};
    use engine::{Card, Combo, DuplicateRule, Move, Rank, Suit};

    fn c(rank: Rank, suit: Suit, i: u8) -> Card { Card::new(rank, suit, i) }

    fn obs(hand: Vec<Card>, sizes: Vec<usize>, unseen: Vec<Card>, on_table: Option<(u8, Combo)>) -> Observation {
        let n = sizes.len();
        let mut hand_sizes = sizes;
        hand_sizes[0] = hand.len();
        Observation {
            duplicate_rule: DuplicateRule::FirstDealtWins,
            seat: 0,
            player_count: u8::try_from(n).unwrap(),
            hand,
            in_trick: vec![true; n],
            ceilings: vec![PassCeilings::default(); n],
            hand_sizes,
            unseen,
            on_table,
        }
    }

    #[test]
    fn with_no_hidden_choice_the_report_is_one_exact_line() {
        // Seat 0: {Ace}; leading; the others hold a single low card each (all unseen cards fixed).
        let o = obs(
            vec![c(Rank::Ace, Suit::Clubs, 0)],
            vec![1, 1, 1],
            vec![c(Rank::Three, Suit::Clubs, 1), c(Rank::Four, Suit::Clubs, 2)],
            None,
        );
        let r = analyze(&o, &SolverConfig::default()).unwrap();
        assert_eq!(r.deals, 2);
        assert_eq!(r.moves.len(), 1);
        assert!((r.moves[0].mean_place - 0.0).abs() < 1e-12);
        assert!((r.score(&r.moves[0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn too_many_unseen_cards_are_refused() {
        let unseen: Vec<Card> = [Rank::Three, Rank::Four, Rank::Five, Rank::Six, Rank::Seven, Rank::Eight, Rank::Nine]
            .iter()
            .enumerate()
            .map(|(i, &r)| c(r, Suit::Clubs, u8::try_from(i).unwrap() + 1))
            .collect();
        let o = obs(vec![c(Rank::Ace, Suit::Clubs, 0)], vec![1, 4, 3], unseen, None);
        let config = SolverConfig { max_unseen: 6, ..SolverConfig::default() };
        assert!(matches!(analyze(&o, &config), Err(SolveError::TooManyUnseen)));
    }

    #[test]
    fn contradictory_ceilings_are_reported() {
        let three = Combo::new(vec![c(Rank::Three, Suit::Hearts, 9)]).unwrap();
        let mut o = obs(
            vec![c(Rank::Two, Suit::Clubs, 0)],
            vec![1, 1, 1],
            vec![c(Rank::Four, Suit::Clubs, 1), c(Rank::Eight, Suit::Hearts, 2)],
            None,
        );
        o.ceilings = read_pass_ceilings(3, &[], &[(1, three, 0), (2, three, 0)], DuplicateRule::FirstDealtWins);
        assert!(matches!(analyze(&o, &SolverConfig::default()), Err(SolveError::NoConsistentDeal)));
    }

    #[test]
    fn a_seat_that_is_already_out_shifts_everyones_place() {
        // 4 seats; seat 3 is out (hand size 0). Seat 0 leads {Ace}: it finishes first among the
        // three remaining seats, i.e. place 1 overall (seat 3 was out before).
        let mut o = obs(
            vec![c(Rank::Ace, Suit::Clubs, 0)],
            vec![1, 1, 1, 0],
            vec![c(Rank::Three, Suit::Clubs, 1), c(Rank::Four, Suit::Clubs, 2)],
            None,
        );
        o.in_trick = vec![true, true, true, false];
        let r = analyze(&o, &SolverConfig::default()).unwrap();
        assert!((r.moves[0].mean_place - 1.0).abs() < 1e-12);
        assert_eq!(r.moves[0].best_place, 1);
    }

    #[test]
    fn the_best_move_is_never_worse_than_any_other_move() {
        use crate::hand_reading::PassTracker;
        use crate::match_runner::turn_context_for;
        use crate::strategy::ContextNeeds;
        use crate::test_support::for_each_state;
        let mut checked = 0;
        for_each_state(33, 2, |state, _| {
            let round = state.round;
            if round.pass_rule() != engine::PassRule::Final {
                return;
            }
            let seat = round.seat_to_move().unwrap();
            let mut tracker = PassTracker::new(usize::from(state.players), state.rule);
            let ctx = turn_context_for(round, seat, state.players, state.round_deck, &mut tracker, ContextNeeds::ALL);
            let Some(o) = Observation::from_context(&ctx, state.rule) else { return };
            let Ok(report) = analyze(&o, &SolverConfig { max_unseen: 5, node_budget: 300_000 }) else { return };
            let best = report.best();
            assert!(report.moves.iter().all(|m| best.mean_place <= m.mean_place + 1e-12));
            let legal = round.legal_moves();
            assert_eq!(
                report.moves.iter().map(|m| m.mv).collect::<Vec<Move>>(),
                legal,
                "the solver searches exactly the engine's legal moves"
            );
            checked += 1;
        });
        assert!(checked > 5, "only {checked} positions were solvable");
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p sim solver::`
Expected: FAIL (types not defined).

- [ ] **Step 3: Implement `search.rs`** (above the tests)

```rust
//! Exact max^n search of one fully known deal.
//!
//! Every seat to move picks the move that gives *itself* the best (lowest)
//! finishing place; ties go to the first move in the engine's move order.
//! This is the rational-opponents model (the spec's default); alpha-beta is
//! not sound for more than two players, so there is no pruning beyond the
//! node budget.

use engine::{Move, Round};

/// 0-based finishing place by seat (0 = first out), for all seats.
pub type Places = [u8; 6];

/// The search needed more nodes than allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetExceeded;

pub struct Search {
    remaining: u64,
    used: u64,
}

impl Search {
    #[must_use]
    pub fn new(node_budget: u64) -> Self {
        Self { remaining: node_budget, used: 0 }
    }

    #[must_use]
    pub fn nodes_used(&self) -> u64 {
        self.used
    }

    /// The outcome of each legal move of the seat to move (engine order).
    ///
    /// # Errors
    ///
    /// [`BudgetExceeded`] when the node budget runs out.
    ///
    /// # Panics
    ///
    /// Panics if the engine rejects one of its own legal moves.
    pub fn root_values(&mut self, round: &Round) -> Result<Vec<(Move, Places)>, BudgetExceeded> {
        let Some(seat) = round.seat_to_move() else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        for mv in round.legal_moves() {
            let mut next = round.clone();
            next.submit_move(seat, mv)
                .expect("moves from legal_moves are accepted");
            out.push((mv, self.value(&next)?));
        }
        Ok(out)
    }

    fn value(&mut self, round: &Round) -> Result<Places, BudgetExceeded> {
        if round.is_complete() {
            return Ok(places_of(round));
        }
        if self.remaining == 0 {
            return Err(BudgetExceeded);
        }
        self.remaining -= 1;
        self.used += 1;
        let seat = round.seat_to_move().expect("round is not complete");
        let mut best: Option<Places> = None;
        for mv in round.legal_moves() {
            let mut next = round.clone();
            next.submit_move(seat, mv)
                .expect("moves from legal_moves are accepted");
            let value = self.value(&next)?;
            if best.is_none_or(|b| value[usize::from(seat)] < b[usize::from(seat)]) {
                best = Some(value);
            }
        }
        Ok(best.expect("a seat to move has at least one legal move"))
    }
}

fn places_of(round: &Round) -> Places {
    let mut places = [0u8; 6];
    for (place, &seat) in round.finishing_order().iter().enumerate() {
        places[usize::from(seat)] = u8::try_from(place).expect("at most 6 seats");
    }
    places
}
```

- [ ] **Step 4: Implement `mod.rs` config and `report.rs`**

Append to `sim/src/solver/mod.rs`:

```rust
pub mod deals;
pub mod report;
pub mod search;

pub use report::{analyze, MoveReport, MoveStats, SolveError};

/// Limits of one solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolverConfig {
    /// Solve only when at most this many cards are unseen.
    pub max_unseen: usize,
    /// Search nodes allowed per decision, over all deals.
    pub node_budget: u64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self { max_unseen: 6, node_budget: 2_000_000 }
    }
}
```

`sim/src/solver/report.rs` (above the tests):

```rust
//! Per-move expected outcome over all consistent deals.

use engine::Move;

use super::deals::for_each_deal;
use super::observation::Observation;
use super::search::Search;
use super::SolverConfig;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveStats {
    pub mv: Move,
    /// Mean 0-based finishing place of the acting seat over all deals.
    pub mean_place: f64,
    pub best_place: u8,
    pub worst_place: u8,
    /// Share of the deals in which this move is (one of) the best.
    pub best_share: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoveReport {
    /// One entry per legal move, in the engine's move order.
    pub moves: Vec<MoveStats>,
    pub deals: u64,
    pub nodes: u64,
    pub player_count: u8,
}

impl MoveReport {
    /// The move with the lowest mean place (the first one on ties).
    ///
    /// # Panics
    ///
    /// Panics on an empty report (`analyze` never returns one).
    #[must_use]
    pub fn best(&self) -> &MoveStats {
        let mut best = &self.moves[0];
        for stats in &self.moves[1..] {
            if stats.mean_place < best.mean_place {
                best = stats;
            }
        }
        best
    }

    /// `1 - 2 * place / (n - 1)`: +1 first out, -1 last.
    #[must_use]
    pub fn score(&self, stats: &MoveStats) -> f64 {
        1.0 - 2.0 * stats.mean_place / f64::from(self.player_count - 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveError {
    TooManyUnseen,
    BudgetExceeded,
    NoConsistentDeal,
    InvalidPosition,
}

/// Solves the observed position exactly for every consistent deal.
///
/// # Errors
///
/// See [`SolveError`].
pub fn analyze(observation: &Observation, config: &SolverConfig) -> Result<MoveReport, SolveError> {
    if observation.unseen.len() > config.max_unseen {
        return Err(SolveError::TooManyUnseen);
    }
    let seat = usize::from(observation.seat);
    let offset = u8::try_from(observation.seats_out()).expect("at most 6 seats");
    let mut search = Search::new(config.node_budget);
    let mut sums: Vec<f64> = Vec::new();
    let mut moves: Vec<Move> = Vec::new();
    let mut best_places: Vec<u8> = Vec::new();
    let mut worst_places: Vec<u8> = Vec::new();
    let mut best_counts: Vec<u64> = Vec::new();
    let mut error = None;

    let deals = for_each_deal(observation, |deal| {
        let Some(round) = observation.round_for(deal) else {
            error = Some(SolveError::InvalidPosition);
            return false;
        };
        let values = match search.root_values(&round) {
            Ok(values) => values,
            Err(_) => {
                error = Some(SolveError::BudgetExceeded);
                return false;
            }
        };
        if moves.is_empty() {
            moves = values.iter().map(|(mv, _)| *mv).collect();
            sums = vec![0.0; moves.len()];
            best_places = vec![u8::MAX; moves.len()];
            worst_places = vec![0; moves.len()];
            best_counts = vec![0; moves.len()];
        }
        // The seat's own place among all seats: seats already out come first.
        let place_of = |places: &[u8; 6]| places[seat] + 0;
        let lowest = values.iter().map(|(_, p)| place_of(p)).min().unwrap_or(0);
        for (i, (_, places)) in values.iter().enumerate() {
            let place = place_of(places);
            sums[i] += f64::from(place);
            best_places[i] = best_places[i].min(place);
            worst_places[i] = worst_places[i].max(place);
            if place == lowest {
                best_counts[i] += 1;
            }
        }
        true
    });

    if let Some(error) = error {
        return Err(error);
    }
    if deals == 0 {
        return Err(SolveError::NoConsistentDeal);
    }
    let _ = offset;
    let denominator = deals as f64;
    Ok(MoveReport {
        moves: moves
            .iter()
            .enumerate()
            .map(|(i, &mv)| MoveStats {
                mv,
                mean_place: sums[i] / denominator,
                best_place: best_places[i],
                worst_place: worst_places[i],
                best_share: best_counts[i] as f64 / denominator,
            })
            .collect(),
        deals,
        nodes: search.nodes_used(),
        player_count: observation.player_count,
    })
}
```

Resolve the place-offset note while implementing: `Round::from_position` enters the seats already out first in `finishing_order`, so the places the search returns are already absolute (seats out occupy places `0..k`, the remaining seats `k..n`). So no offset arithmetic is needed: delete `offset`, `place_of` and the `let _ = offset;` line and use `places[seat]` directly. The test `a_seat_that_is_already_out_shifts_everyones_place` pins this (expects place 1).

- [ ] **Step 5: Run tests and clippy**

Run: `cargo test -p sim solver:: && cargo clippy -p sim --all-targets`
Expected: PASS. If `the_search_agrees_with_the_naive_reference...` is slow (> a few seconds), lower the card limit in the test from 9 to 8, not the production budget.

- [ ] **Step 6: Commit**

```bash
git add sim/src/solver
git commit -m "sim: exact max^n search of one deal and per-move analysis over all deals

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The `EndgameSolver` strategy

**Files:**
- Create: `sim/src/solver/config.rs` (strategy names and the strategy-level config), `sim/src/strategies/endgame_solver.rs`
- Modify: `sim/src/solver/mod.rs` (add `pub mod config;`), `sim/src/strategies/mod.rs` (add `mod endgame_solver;` and `pub use endgame_solver::{EndgameSolver, SolverStats};`), `sim/src/lib.rs` (re-export `EndgameSolver`, `SolverStats`, `SolverConfig`)

**Interfaces:**
- Consumes: `analyze`, `Observation`, `SolverConfig`, the fixed strategies (`LowestLegal`, `GreedyHighest`, `HoldBackPairs`, `CardCounter`, `EndgameDenial`, `Adaptive`).
- Produces:
  - `solver::config::StrategyName` enum (`LowestLegal`, `GreedyHighest`, `HoldBackPairs`, `CardCounter`, `EndgameDenial`, `Adaptive`) implementing `FromStr` (`lowest-legal`, `greedy-highest`, `hold-back-pairs`, `card-counter`, `endgame-denial`, `adaptive`) and `Display`, with `build(self) -> Arc<dyn Strategy>`.
  - `solver::config::SolverSpec { pub solver: SolverConfig, pub fallback: StrategyName }` implementing `FromStr` for comma-separated options `max-unseen=N`, `budget=N`, `fallback=NAME` (empty string = defaults) and `Default`.
  - `EndgameSolver::new(spec: SolverSpec) -> EndgameSolver`; `EndgameSolver::stats(&self) -> SolverStats`; `SolverStats { pub decisions: u64, pub eligible: u64, pub solved: u64, pub fell_back_budget: u64, pub fell_back_other: u64, pub nodes: u64 }` (all `u64`, `Debug, Clone, Copy, Default, PartialEq, Eq`).
  - Strategy name: `EndgameSolver(<fallback>,T=<max_unseen>)`, e.g. `EndgameSolver(adaptive,T=6)`.
  - Semantics: with one legal move return it (not counted as eligible); with `unseen > max_unseen` or `Observation::from_context` returning `None` (e.g. `Free`): fallback, counted only in `decisions`. Otherwise `eligible += 1`; success -> `solved += 1`, the best move **only if it is in `legal_moves`**; `BudgetExceeded` -> `fell_back_budget`; any other error or a best move not in `legal_moves` -> `fell_back_other`.

- [ ] **Step 1: Write failing tests** (in `sim/src/strategies/endgame_solver.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand_reading::PassTracker;
    use crate::match_runner::turn_context_for;
    use crate::solver::config::SolverSpec;
    use crate::strategy::ContextNeeds;
    use crate::test_support::for_each_state;
    use rand::SeedableRng;

    fn spec(max_unseen: usize) -> SolverSpec {
        SolverSpec {
            solver: crate::solver::SolverConfig { max_unseen, node_budget: 300_000 },
            ..SolverSpec::default()
        }
    }

    #[test]
    fn it_always_returns_a_legal_move_and_counts_what_it_did() {
        let solver = EndgameSolver::new(spec(5));
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        for_each_state(41, 3, |state, _| {
            let round = state.round;
            let seat = round.seat_to_move().unwrap();
            let mut tracker = PassTracker::new(usize::from(state.players), state.rule);
            let ctx = turn_context_for(round, seat, state.players, state.round_deck, &mut tracker, ContextNeeds::ALL);
            let legal = round.legal_moves();
            let mv = solver.choose_play(&legal, state.rule, &ctx, &mut rng);
            assert!(legal.contains(&mv));
        });
        let s = solver.stats();
        assert!(s.solved > 0, "the endgames of the random rounds are solved");
        assert_eq!(s.solved + s.fell_back_budget + s.fell_back_other, s.eligible);
        assert!(s.eligible <= s.decisions);
    }

    #[test]
    fn under_the_free_pass_rule_it_defers_to_the_fallback() {
        let solver = EndgameSolver::new(spec(40));
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        for_each_state(43, 1, |state, _| {
            let round = state.round;
            let seat = round.seat_to_move().unwrap();
            let mut tracker = PassTracker::new(usize::from(state.players), state.rule);
            let mut ctx = turn_context_for(round, seat, state.players, state.round_deck, &mut tracker, ContextNeeds::ALL);
            ctx.table.pass_rule = engine::PassRule::Free;
            let legal = round.legal_moves();
            let mv = solver.choose_play(&legal, state.rule, &ctx, &mut rng);
            assert!(legal.contains(&mv));
        });
        assert_eq!(solver.stats().eligible, 0);
        assert!(solver.stats().decisions > 0);
    }

    #[test]
    fn a_single_legal_move_is_returned_without_searching() {
        let solver = EndgameSolver::new(spec(40));
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let mut seen_single = false;
        for_each_state(47, 2, |state, _| {
            let round = state.round;
            let legal = round.legal_moves();
            if legal.len() != 1 {
                return;
            }
            seen_single = true;
            let seat = round.seat_to_move().unwrap();
            let mut tracker = PassTracker::new(usize::from(state.players), state.rule);
            let ctx = turn_context_for(round, seat, state.players, state.round_deck, &mut tracker, ContextNeeds::ALL);
            assert_eq!(solver.choose_play(&legal, state.rule, &ctx, &mut rng), legal[0]);
        });
        assert!(seen_single);
        assert_eq!(solver.stats().eligible, 0);
    }

    #[test]
    fn the_name_shows_the_configuration() {
        assert_eq!(EndgameSolver::new(spec(7)).name(), "EndgameSolver(adaptive,T=7)");
    }
}
```

and in `sim/src/solver/config.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_parse_and_unknown_ones_are_rejected() {
        let spec: SolverSpec = "max-unseen=8,budget=100,fallback=lowest-legal".parse().unwrap();
        assert_eq!(spec.solver.max_unseen, 8);
        assert_eq!(spec.solver.node_budget, 100);
        assert_eq!(spec.fallback, StrategyName::LowestLegal);
        assert_eq!("".parse::<SolverSpec>().unwrap(), SolverSpec::default());
        assert!("depth=3".parse::<SolverSpec>().is_err());
        assert!("max-unseen=x".parse::<SolverSpec>().is_err());
        assert!("fallback=nope".parse::<SolverSpec>().is_err());
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p sim solver::config strategies::endgame_solver`
Expected: FAIL (not defined).

- [ ] **Step 3: Implement `config.rs`**

```rust
//! Strategy names and the options of `endgame-solver[:OPTIONS]`.

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use crate::strategies::{
    Adaptive, AdaptiveConfig, CardCounter, EndgameDenial, GreedyHighest, HoldBackPairs, LowestLegal,
};
use crate::strategy::Strategy;

use super::SolverConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategyName {
    LowestLegal,
    GreedyHighest,
    HoldBackPairs,
    CardCounter,
    EndgameDenial,
    Adaptive,
}

impl StrategyName {
    #[must_use]
    pub fn build(self) -> Arc<dyn Strategy> {
        match self {
            Self::LowestLegal => Arc::new(LowestLegal),
            Self::GreedyHighest => Arc::new(GreedyHighest),
            Self::HoldBackPairs => Arc::new(HoldBackPairs),
            Self::CardCounter => Arc::new(CardCounter),
            Self::EndgameDenial => Arc::new(EndgameDenial),
            Self::Adaptive => Arc::new(Adaptive::new(AdaptiveConfig::default())),
        }
    }
}

impl fmt::Display for StrategyName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::LowestLegal => "lowest-legal",
            Self::GreedyHighest => "greedy-highest",
            Self::HoldBackPairs => "hold-back-pairs",
            Self::CardCounter => "card-counter",
            Self::EndgameDenial => "endgame-denial",
            Self::Adaptive => "adaptive",
        })
    }
}

impl FromStr for StrategyName {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        Ok(match text.trim() {
            "lowest-legal" => Self::LowestLegal,
            "greedy-highest" => Self::GreedyHighest,
            "hold-back-pairs" => Self::HoldBackPairs,
            "card-counter" => Self::CardCounter,
            "endgame-denial" => Self::EndgameDenial,
            "adaptive" => Self::Adaptive,
            other => {
                return Err(format!(
                    "unknown fallback `{other}`; expected lowest-legal, greedy-highest, \
                     hold-back-pairs, card-counter, endgame-denial or adaptive"
                ))
            }
        })
    }
}

/// The options of `endgame-solver[:OPTIONS]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolverSpec {
    pub solver: SolverConfig,
    pub fallback: StrategyName,
}

impl Default for SolverSpec {
    fn default() -> Self {
        Self { solver: SolverConfig::default(), fallback: StrategyName::Adaptive }
    }
}

impl FromStr for SolverSpec {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        let mut spec = Self::default();
        for option in text.split(',').map(str::trim).filter(|o| !o.is_empty()) {
            let (key, value) = option
                .split_once('=')
                .ok_or_else(|| format!("`{option}` is not KEY=VALUE"))?;
            match key {
                "max-unseen" => {
                    spec.solver.max_unseen = value
                        .parse()
                        .map_err(|_| format!("max-unseen: `{value}` is not a number"))?;
                }
                "budget" => {
                    spec.solver.node_budget = value
                        .parse()
                        .map_err(|_| format!("budget: `{value}` is not a number"))?;
                }
                "fallback" => spec.fallback = value.parse()?,
                other => {
                    return Err(format!(
                        "unknown solver option `{other}`; expected max-unseen, budget or fallback"
                    ))
                }
            }
        }
        Ok(spec)
    }
}
```

- [ ] **Step 4: Implement the strategy** (above the tests in `endgame_solver.rs`)

```rust
//! `EndgameSolver`: exact play once few cards are unseen, a fallback
//! strategy before that. See docs/superpowers/specs/2026-10-16-endgame-solver-design.md.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use engine::{Card, DuplicateRule, Move};

use crate::solver::config::SolverSpec;
use crate::solver::{analyze, Observation, SolveError, SolverConfig};
use crate::strategy::{ContextNeeds, Strategy, TurnContext};

/// What the solver did so far (all decisions of this instance).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SolverStats {
    /// Decisions with more than one legal move.
    pub decisions: u64,
    /// Decisions the solver tried to solve (supported position, few enough unseen cards).
    pub eligible: u64,
    pub solved: u64,
    pub fell_back_budget: u64,
    pub fell_back_other: u64,
    /// Search nodes spent in total.
    pub nodes: u64,
}

pub struct EndgameSolver {
    config: SolverConfig,
    fallback: Arc<dyn Strategy>,
    name: String,
    decisions: AtomicU64,
    eligible: AtomicU64,
    solved: AtomicU64,
    fell_back_budget: AtomicU64,
    fell_back_other: AtomicU64,
    nodes: AtomicU64,
}

impl EndgameSolver {
    #[must_use]
    pub fn new(spec: SolverSpec) -> Self {
        Self {
            config: spec.solver,
            fallback: spec.fallback.build(),
            name: format!("EndgameSolver({},T={})", spec.fallback, spec.solver.max_unseen),
            decisions: AtomicU64::new(0),
            eligible: AtomicU64::new(0),
            solved: AtomicU64::new(0),
            fell_back_budget: AtomicU64::new(0),
            fell_back_other: AtomicU64::new(0),
            nodes: AtomicU64::new(0),
        }
    }

    #[must_use]
    pub fn stats(&self) -> SolverStats {
        SolverStats {
            decisions: self.decisions.load(Ordering::Relaxed),
            eligible: self.eligible.load(Ordering::Relaxed),
            solved: self.solved.load(Ordering::Relaxed),
            fell_back_budget: self.fell_back_budget.load(Ordering::Relaxed),
            fell_back_other: self.fell_back_other.load(Ordering::Relaxed),
            nodes: self.nodes.load(Ordering::Relaxed),
        }
    }

    /// The solver's move, or `None` to defer to the fallback.
    fn solve(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        context: &TurnContext<'_>,
    ) -> Option<Move> {
        let observation = Observation::from_context(context, duplicate_rule)?;
        if observation.unseen.len() > self.config.max_unseen {
            return None;
        }
        self.eligible.fetch_add(1, Ordering::Relaxed);
        match analyze(&observation, &self.config) {
            Ok(report) => {
                self.nodes.fetch_add(report.nodes, Ordering::Relaxed);
                let best = report.best().mv;
                if legal_moves.contains(&best) {
                    self.solved.fetch_add(1, Ordering::Relaxed);
                    Some(best)
                } else {
                    self.fell_back_other.fetch_add(1, Ordering::Relaxed);
                    None
                }
            }
            Err(SolveError::BudgetExceeded) => {
                self.fell_back_budget.fetch_add(1, Ordering::Relaxed);
                None
            }
            Err(_) => {
                self.fell_back_other.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }
}

impl Strategy for EndgameSolver {
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
        if legal_moves.len() == 1 {
            return legal_moves[0];
        }
        self.decisions.fetch_add(1, Ordering::Relaxed);
        self.solve(legal_moves, duplicate_rule, context)
            .unwrap_or_else(|| self.fallback.choose_play(legal_moves, duplicate_rule, context, rng))
    }

    fn needs(&self) -> ContextNeeds {
        ContextNeeds::ALL
    }

    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        duplicate_rule: DuplicateRule,
        rng: &mut dyn rand::Rng,
    ) -> Vec<Card> {
        self.fallback
            .choose_exchange_cards(hand, count, duplicate_rule, rng)
    }
}
```

Wire-up: `pub mod config;` in `solver/mod.rs`; `mod endgame_solver;` + `pub use endgame_solver::{EndgameSolver, SolverStats};` in `strategies/mod.rs`; add `EndgameSolver, SolverStats` to the `pub use strategies::{...}` list in `sim/src/lib.rs` and `pub use solver::{SolverConfig, SolverSpec, StrategyName};`. Check `Strategy::choose_exchange_cards`'s exact signature in `sim/src/strategy.rs` and match it (the plan copies the shape seen at line ~157: `hand: &[Card], count: usize, duplicate_rule: DuplicateRule, rng`).

- [ ] **Step 5: Run tests and clippy**

Run: `cargo test -p sim && cargo clippy -p sim --all-targets`
Expected: PASS. `the first test` needs `solved > 0`: with `max_unseen = 5` the last hands of random rounds qualify; if it fails, raise `max_unseen` to 8 in the test, not the production default.

- [ ] **Step 6: Commit**

```bash
git add sim
git commit -m "sim: EndgameSolver strategy with fallback, counters and spec options

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: CLI registration and determinism under `run_batch`

**Files:**
- Modify: `cli/src/args.rs` (`StrategyArg` enum, `FromStr`, `build`, help text of `--strategy`)
- Test: `cli/src/args/tests.rs` (append), `sim/src/strategies/endgame_solver.rs` (a determinism test)

**Interfaces:**
- Consumes: `sim::SolverSpec`, `sim::EndgameSolver`.
- Produces: `StrategyArg::Solver(sim::SolverSpec)`; the spec `endgame-solver` or `endgame-solver:OPTIONS` (options as in Task 5) anywhere a strategy spec is accepted by `StrategyArg`.

- [ ] **Step 1: Write failing tests**

Append to `cli/src/args/tests.rs` (use the existing parsing helpers' style: `"card-counter".parse::<StrategyArg>().unwrap()` appears at line ~175):

```rust
#[test]
fn endgame_solver_specs_parse() {
    assert_eq!(
        "endgame-solver".parse::<StrategyArg>().unwrap(),
        StrategyArg::Solver(sim::SolverSpec::default())
    );
    let arg = "endgame-solver:max-unseen=8,fallback=lowest-legal"
        .parse::<StrategyArg>()
        .unwrap();
    match arg {
        StrategyArg::Solver(spec) => {
            assert_eq!(spec.solver.max_unseen, 8);
            assert_eq!(spec.fallback, sim::StrategyName::LowestLegal);
        }
        other => panic!("expected a solver spec, got {other:?}"),
    }
    assert!("endgame-solver:depth=3".parse::<StrategyArg>().is_err());
    assert_eq!(
        "endgame-solver".parse::<StrategyArg>().unwrap().build().name(),
        "EndgameSolver(adaptive,T=6)"
    );
}
```

In `sim/src/strategies/endgame_solver.rs` tests, a determinism check using the public batch API (read `sim/src/match_runner.rs` `run_batch` signature and `MatchConfig` fields first and adapt; the shape is: same `MatchConfig`, strategies `[solver, lowest-legal, lowest-legal, lowest-legal]`, 8 matches, run once with `threads = 1` and once with the default pool, compare the serialised `MatchResult`s):

```rust
#[test]
fn a_batch_with_the_solver_is_identical_on_one_and_many_threads() {
    // See sim/src/match_runner.rs: run_batch(config, strategies, matches, seed) and the
    // test helpers used by its own determinism tests; copy their setup. Compare
    // `format!("{results:?}")` of both runs.
}
```

(The body must be filled from the existing `run_batch` determinism test in `sim/src/match_runner.rs`: search for `threads` / `rayon::ThreadPoolBuilder` in its tests and reuse that exact setup with the solver in seat 0. The assertion is equality of the two result sets.)

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p cli endgame_solver_specs_parse`
Expected: FAIL (`StrategyArg::Solver` does not exist).

- [ ] **Step 3: Implement**

In `cli/src/args.rs`:

```rust
pub enum StrategyArg {
    Fixed(FixedStrategy),
    Adaptive(sim::AdaptiveConfig),
    Neat(NeatSpec),
    Solver(sim::SolverSpec),
}
```

(Keep the existing derives; `SolverSpec` is `Debug + Clone + Copy + PartialEq + Eq`, so `StrategyArg`'s derives still hold.) In `from_str`, before the `options.is_some()` check:

```rust
        if head == "endgame-solver" {
            return match options {
                None => Ok(Self::Solver(sim::SolverSpec::default())),
                Some(o) => o.parse().map(Self::Solver),
            };
        }
```

In `build`: `Self::Solver(spec) => Arc::new(sim::EndgameSolver::new(spec)),`. Update the `--strategy` help text and the "unknown strategy" error to mention `endgame-solver[:OPTIONS]` (options `max-unseen=N`, `budget=N`, `fallback=NAME`). Fix every exhaustive `match` over `StrategyArg` the compiler reports (`cargo build -p cli` lists them).

- [ ] **Step 4: Run tests**

Run: `cargo test --workspace && cargo clippy --workspace --all-targets`
Expected: PASS.

- [ ] **Step 5: Smoke run**

Run: `cargo run -q --release -p cli -- --player-count 4 --matches 40 --rounds 3 --seed 1 --output /tmp/claude-1000/-home-sebi-Documents-Code-arschloch/1f365010-e198-40c1-b9cc-fc4e8345a671/scratchpad/solver-smoke.json --strategy endgame-solver --strategy lowest-legal --strategy lowest-legal --strategy lowest-legal`
Expected: the summary prints a row named `EndgameSolver(adaptive,T=6)`; no panic.

- [ ] **Step 6: Commit**

```bash
git add cli sim
git commit -m "cli: endgame-solver strategy spec; determinism test with the solver in a batch

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `cli solve` — "why this move" for a real situation

**Files:**
- Create: `cli/src/solve.rs`
- Modify: `cli/src/main.rs` (add `mod solve;` and `Some("solve") => return solve::run(std::env::args().skip(2)),` next to the other subcommands)
- Test: unit tests in `cli/src/solve.rs`

**Interfaces:**
- Consumes: `sim::solver::{analyze, Observation, SolverConfig, MoveReport}`, `sim::solver::notation::{card_token, parse_card_token}`, `engine::{standard_deck, DeckVariant, DuplicateRule, Card, Combo, Move, Round}`.
- Produces: `cli solve FILE [--max-unseen N] [--budget N]` printing the ranked moves. Also `pub(crate) fn parse_position(text: &str) -> Result<Observation, String>` and `pub(crate) fn render(report: &MoveReport, observation: &Observation) -> String` (unit-tested).

Input format (text file, `#` starts a comment, keys case-insensitive, one per line):

```
players: 4
deck: single            # single | double
duplicate: first        # first | last (who wins between identical cards); default first
seat: 0                 # the seat to move
hand: 3C 9H KS          # our cards
sizes: 3 4 2 3          # cards held by every seat, in seat order (0 = already out); own entry must equal the hand size
table: pair 8H 8S by 2  # "lead" when we lead; otherwise the combo on the table and who holds it ("by SEAT")
passed: 1               # seats that passed in this trick (optional)
played: 2D 2H 2S ...    # every card played so far this round (the solver derives the unseen cards)
```

Cards of a double deck that appear twice get increasing `deal_index` in the order listed (first listed wins under `duplicate: first`). Cards are matched to the deck by rank and suit; asking for more copies than the deck has is an error. The unseen cards are the deck minus `hand` minus `played`; their count must equal the sum of the other seats' sizes.

- [ ] **Step 1: Write failing tests** (bottom of `cli/src/solve.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    const ENDGAME: &str = "\
players: 3
deck: single
seat: 0
hand: AC
sizes: 1 1 1
table: lead
played: 2D 2H 2S 2C 3D 3H 3S 3C 4D 4H 4S 4C 5D 5H 5S 5C 6D 6H 6S 6C 7D 7H 7S 7C 8D 8H 8S 8C 9D 9H 9S 9C 10D 10H 10S 10C JD JH JS JC QD QH QS QC KD KH KS KC AD AH AS
";

    #[test]
    fn a_position_file_parses_and_solves() {
        let observation = parse_position(ENDGAME).unwrap();
        assert_eq!(observation.unseen.len(), 2);
        let report = sim::solver::analyze(&observation, &sim::solver::SolverConfig::default()).unwrap();
        let text = render(&report, &observation);
        assert!(text.contains("AC"), "{text}");
        assert!(text.contains("deals considered: 2"), "{text}");
    }

    #[test]
    fn mistakes_are_reported_in_plain_words() {
        assert!(parse_position("players: 4\n").unwrap_err().contains("hand"));
        let bad_size = ENDGAME.replace("sizes: 1 1 1", "sizes: 2 1 1");
        assert!(parse_position(&bad_size).unwrap_err().contains("sizes"));
        let too_many = ENDGAME.replace("hand: AC", "hand: AC AC");
        assert!(parse_position(&too_many).is_err());
        let wrong_unseen = ENDGAME.replace("sizes: 1 1 1", "sizes: 1 1 2");
        assert!(parse_position(&wrong_unseen).unwrap_err().contains("unseen"));
    }
}
```

(The ENDGAME `played` list is the whole single deck except `AC` (ours) and two more cards left unseen: the deck has 52 cards; the list above omits `AC` and `AS`? Recount while implementing: the file must leave exactly 2 unseen cards plus our `AC`. The test asserts `unseen.len() == 2`, so adjust the `played` line (drop one more card from it) until the parser's own count says so.)

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p cli solve::`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

`cli/src/solve.rs`:

```rust
//! `cli solve FILE`: the exact analysis of one real situation
//! (docs/SOLVER.md). The position file format is described there.

use std::path::PathBuf;

use anyhow::Context;
use engine::{standard_deck, Card, Combo, DeckVariant, DuplicateRule, PassRule};
use sim::hand_reading::PassCeilings;
use sim::solver::notation::{card_token, parse_card_token};
use sim::solver::{analyze, MoveReport, Observation, SolverConfig};

pub fn run(args: impl Iterator<Item = String>) -> anyhow::Result<()> {
    let mut path: Option<PathBuf> = None;
    let mut config = SolverConfig { max_unseen: 12, node_budget: 50_000_000 };
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--max-unseen" => {
                config.max_unseen = args.next().context("--max-unseen needs a number")?.parse()?;
            }
            "--budget" => {
                config.node_budget = args.next().context("--budget needs a number")?.parse()?;
            }
            "-h" | "--help" => {
                println!("usage: cli solve FILE [--max-unseen N] [--budget N]   (see docs/SOLVER.md)");
                return Ok(());
            }
            other if path.is_none() && !other.starts_with("--") => path = Some(PathBuf::from(other)),
            other => anyhow::bail!("unexpected argument `{other}` (see docs/SOLVER.md)"),
        }
    }
    let path = path.context("usage: cli solve FILE [--max-unseen N] [--budget N]")?;
    let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let observation = parse_position(&text).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let report = analyze(&observation, &config)
        .map_err(|e| anyhow::anyhow!("cannot solve this position: {e:?} (try --max-unseen / --budget)"))?;
    print!("{}", render(&report, &observation));
    Ok(())
}
```

`parse_position` (same file): split lines, strip `#` comments, `key: value`; keys `players`, `deck`, `duplicate`, `seat`, `hand`, `sizes`, `table`, `passed`, `played`. Steps: build the deck with `standard_deck(variant)` and set `deal_index` to the position (`for (i, c) in deck.iter_mut().enumerate() { c.deal_index = u8::try_from(i)?; }`); a closure `take(token) -> Result<Card>` removes the first deck card with the token's rank/suit from a `pool: Vec<Card>` (error "no more copies of 9H in the deck" if none) — use it for `hand`, `played` and the `table` combo cards in that order so each listed card consumes one copy; the remaining `pool` is `unseen`. Validate: `players` 3..=6; `seat < players`; `sizes.len() == players`, `sizes[seat] == hand.len()` (error text must contain `sizes`); `unseen.len() == sum of other sizes` (error text must contain `unseen`, include both numbers); `table: lead` -> `on_table = None`, else `table: <size-word|cards...> by <seat>` parse every card token between `table:` and `by` as the combo cards (all one rank via `Combo::from_slice`, else error) and the holder after `by`; the combo's cards are removed from `pool` like played cards (they were played this trick). `passed` -> `in_trick[s] = false` for listed seats (and `in_trick[s] = sizes[s] > 0 || s == seat` otherwise... use: `in_trick[s] = !passed.contains(&s)`). Ceilings are `PassCeilings::default()` for every seat in v1. A missing `players`/`hand`/`sizes`/`seat`/`played` key gives an error that names the key (`"missing `hand`"`). Build:

```rust
Ok(Observation {
    duplicate_rule,
    seat,
    player_count: players,
    hand,
    hand_sizes: sizes,
    in_trick,
    ceilings: vec![PassCeilings::default(); usize::from(players)],
    unseen: pool,
    on_table,
})
```

`render`:

```rust
pub(crate) fn render(report: &MoveReport, observation: &Observation) -> String {
    use std::fmt::Write as _;
    let mut order: Vec<&sim::solver::MoveStats> = report.moves.iter().collect();
    order.sort_by(|a, b| a.mean_place.total_cmp(&b.mean_place));
    let mut out = String::new();
    let _ = writeln!(out, "seat {} to move, hand: {}", observation.seat,
        observation.hand.iter().map(card_token).collect::<Vec<_>>().join(" "));
    let _ = writeln!(out, "deals considered: {} ({} search nodes)", report.deals, report.nodes);
    let _ = writeln!(out, "{:<22} {:>9} {:>8} {:>10} {:>12}", "move", "exp.place", "score", "best/worst", "best in");
    for stats in order {
        let name = match stats.mv {
            engine::Move::Pass => "pass".to_owned(),
            engine::Move::Play(combo) => combo.cards().iter().map(card_token).collect::<Vec<_>>().join(" "),
        };
        let _ = writeln!(out, "{:<22} {:>9.2} {:>+8.2} {:>6}/{:<3} {:>11.0}%", name,
            stats.mean_place + 1.0, report.score(stats), stats.best_place + 1, stats.worst_place + 1,
            stats.best_share * 100.0);
    }
    out
}
```

`exp.place` is 1-based for humans (1 = first out). `best in` is the share of deals where the move is (one of) the best.

- [ ] **Step 4: Run tests and clippy**

Run: `cargo test -p cli solve:: && cargo clippy -p cli --all-targets`
Expected: PASS.

- [ ] **Step 5: Try it on a real file** (create `docs/examples/solve-endgame.txt` with a small, valid position, e.g. the test's) and run `cargo run -q --release -p cli -- solve docs/examples/solve-endgame.txt`
Expected: a ranked table.

- [ ] **Step 6: Commit**

```bash
git add cli docs/examples
git commit -m "cli: solve command analyses a written-down position move by move

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Disagreement benchmark `cli solver-gap`, measurement of `T`, head-to-head

**Files:**
- Create: `sim/src/solver/gap.rs`, `cli/src/solver_gap.rs`, `docs/baselines/solver/README.md`
- Modify: `sim/src/solver/mod.rs` (`pub mod gap;`), `cli/src/main.rs` (`mod solver_gap;` + `Some("solver-gap") => return solver_gap::run(std::env::args().skip(2)),`)

**Interfaces:**
- Consumes: `analyze`, `Observation`, `turn_context_for` (`pub(crate)`, same crate), `PassTracker`, `Strategy`.
- Produces: 
  - `pub struct GapConfig { pub players: u8, pub deck: DeckVariant, pub rule: DuplicateRule, pub games: usize, pub seed: u64, pub solver: SolverConfig }`.
  - `pub struct GapRow { pub strategy: String, pub decisions: u64, pub differing: u64, pub cost_sum: f64, pub worst: Vec<Disagreement> }` (cost in places; `worst` holds at most 5, largest cost first) and `pub struct Disagreement { pub cost: f64, pub situation: String }`.
  - `pub fn run_gap(strategies: &[Arc<dyn Strategy>], config: &GapConfig) -> (Vec<GapRow>, GapStats)` with `pub struct GapStats { pub decisions: u64, pub solvable: u64, pub budget_exceeded: u64, pub nodes_total: u64, pub nodes_max: u64 }`. Strategy `i` sits at seat `(i + game) % players` of game `game` (rotation); each game is one round dealt from `StdRng::seed_from_u64(seed + game)`, played out by the strategies. At every decision with more than one legal move, `Observation::from_context` and `analyze` run on the position; for solvable decisions the cost of the strategy's chosen move is `mean_place(chosen) - mean_place(best)` (>= 0). `differing` counts costs above `1e-9`.
  - CLI: `cli solver-gap --strategy SPEC ... [--player-count N] [--deck single|double] [--games N] [--seed N] [--max-unseen N] [--budget N]` using `StrategyArg` for specs; prints one row per strategy (decisions solved, share differing, mean cost in places per solved decision) then the worst disagreements with the situation (hand, table, the strategy's move vs. the solver's move) and the `GapStats` (`solvable`, `budget_exceeded`, mean/max nodes per decision).

- [ ] **Step 1: Write failing tests** in `sim/src/solver/gap.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategies::{LowestLegal, GreedyHighest};
    use crate::solver::SolverConfig;
    use engine::{DeckVariant, DuplicateRule};

    fn config(games: usize) -> GapConfig {
        GapConfig {
            players: 4,
            deck: DeckVariant::Single,
            rule: DuplicateRule::FirstDealtWins,
            games,
            seed: 5,
            solver: SolverConfig { max_unseen: 6, node_budget: 500_000 },
        }
    }

    #[test]
    fn the_benchmark_is_deterministic_and_costs_are_never_negative() {
        let table: Vec<Arc<dyn Strategy>> = vec![Arc::new(LowestLegal), Arc::new(GreedyHighest), Arc::new(LowestLegal), Arc::new(GreedyHighest)];
        let (rows, stats) = run_gap(&table, &config(30));
        let (rows_again, stats_again) = run_gap(&table, &config(30));
        assert_eq!(format!("{rows:?}"), format!("{rows_again:?}"));
        assert_eq!(stats.decisions, stats_again.decisions);
        assert!(stats.solvable > 0);
        for row in &rows {
            assert!(row.cost_sum >= -1e-9);
            assert!(row.differing <= row.decisions);
            assert!(row.worst.len() <= 5);
        }
    }

    #[test]
    fn a_strategy_that_is_the_solver_itself_never_disagrees() {
        use crate::strategies::EndgameSolver;
        let solver_spec = crate::solver::config::SolverSpec {
            solver: SolverConfig { max_unseen: 6, node_budget: 500_000 },
            ..Default::default()
        };
        let table: Vec<Arc<dyn Strategy>> = (0..4).map(|_| Arc::new(EndgameSolver::new(solver_spec)) as Arc<dyn Strategy>).collect();
        let (rows, _) = run_gap(&table, &config(20));
        assert!(rows.iter().all(|r| r.differing == 0), "{rows:?}");
    }
}
```

- [ ] **Step 2: Run to verify failure**, then implement `gap.rs`

Run: `cargo test -p sim solver::gap` -> FAIL.

Implementation notes (write the code from these exact steps):
1. Deck: `standard_deck(config.deck)`, shuffle with `StdRng::seed_from_u64(seed + game)` (`rand::seq::SliceRandom`), set `deal_index` by position, `engine::deal(deck, players)`, `round_deck = hands.iter().flatten().copied().collect()`, `leader = engine::lowest_card_holder(&hands, rule)`, `Round::new(hands, rule, leader)` (default pass rule is `Final`).
2. Loop `while !round.is_complete()`: `seat = round.seat_to_move()`; `strategy = &strategies[(usize::from(seat) + games_offset) % n]` where the rotation is `strategy_index = (usize::from(seat) + n - game % n) % n`; `legal = round.legal_moves()`; `ctx = turn_context_for(&round, seat, players, &round_deck, &mut tracker, ContextNeeds::ALL)`; `chosen = strategy.choose_play(&legal, rule, &ctx, &mut rng)` with `rng = StdRng::seed_from_u64(seed ^ game_seed_mix)`; when `legal.len() > 1`: `decisions += 1`, then `Observation::from_context` + `analyze`; `Ok(report)`: `solvable += 1`, `nodes_total += report.nodes`, `nodes_max = max`, find the `MoveStats` whose `mv == chosen` (always present) and compute the cost; `Err(BudgetExceeded)`: `budget_exceeded += 1`; other errors (`TooManyUnseen`, `NoConsistentDeal`) are skipped. Submit `chosen`.
3. A `Disagreement.situation` string: `format!("seat {seat}, hand [{}], {}: {} played, solver prefers {}", ...)` using `card_token`, `"leading"` or `"on {combo tokens} by seat {holder}"`, and move names (`pass` or the card tokens).
4. `worst` keeps the 5 largest costs per strategy (sort descending after each insertion, truncate to 5; ties keep the earlier one so it is deterministic).

Run: `cargo test -p sim solver::gap` -> PASS.

- [ ] **Step 3: CLI command** `cli/src/solver_gap.rs`

Parse the flags listed above by hand (same style as `cli/src/solve.rs`), build strategies with `StrategyArg::from_str(..).map(StrategyArg::build)`, call `sim::solver::gap::run_gap`, print. Add a unit test that `--strategy` count must equal `--player-count` (reuse the error message style of `args::validate`).

Run: `cargo test -p cli solver_gap && cargo clippy --workspace --all-targets` -> PASS.

- [ ] **Step 4: Measure `T`** (this is the data the open points of the spec asked for). Run each, release build, and record the printed `GapStats` and wall time with `time`:

```bash
for T in 4 6 8 10 12; do
  for P in 3 4 5; do
    time cargo run -q --release -p cli -- solver-gap --player-count $P --deck single --games 300 --seed 1 \
      --max-unseen $T --budget 20000000 \
      $(for i in $(seq 1 $P); do printf -- '--strategy adaptive '; done)
  done
done
```

Repeat for `--deck double` with `P = 4 5 6`. Write the table (T, players, deck, solvable share, budget_exceeded share, mean and max nodes, seconds per 300 games) into `docs/baselines/solver/README.md`, pick the largest `T` per table size/deck whose `budget_exceeded` share is below 1% and whose time per decision is acceptable (state the criterion in the README), and set `SolverConfig::default().max_unseen` to the best single value for the 4-player single-deck table. If the nodes show the search is too slow for useful `T` (budget exceeded often at `T <= 8`), add the transposition table here: key `(hands by seat sorted by card, current combo, holder, passed set, seat to move)` hashed with `std::collections::HashMap` (lookups only, never iterated, so determinism holds), per deal; keep it only if it cuts nodes by more than 2x on the 4-player `T = 8` run, and say so in the README.

- [ ] **Step 5: Head-to-head and disagreement benchmark** (these produce the lessons):

```bash
# luck-free comparison against plain adaptive and against the neat-v2 champion
cargo run -q --release -p cli -- --player-count 4 --matches 2000 --rounds 5 --seed 7 --skill-score duplicate \
  --output /tmp/claude-1000/-home-sebi-Documents-Code-arschloch/1f365010-e198-40c1-b9cc-fc4e8345a671/scratchpad/h2h.json \
  --strategy endgame-solver --strategy adaptive --strategy adaptive --strategy adaptive
cargo run -q --release -p cli -- --player-count 4 --matches 2000 --rounds 5 --seed 7 --skill-score duplicate \
  --output /tmp/claude-1000/-home-sebi-Documents-Code-arschloch/1f365010-e198-40c1-b9cc-fc4e8345a671/scratchpad/h2h-neat.json \
  --strategy endgame-solver --strategy neat:docs/baselines/neat-v2/champion.json --strategy adaptive --strategy lowest-legal
cargo run -q --release -p cli -- solver-gap --player-count 4 --games 2000 --seed 3 \
  --strategy adaptive --strategy endgame-denial --strategy neat:docs/baselines/neat-v2/champion.json --strategy lowest-legal
```

Paste the key results (skill score with standard error per strategy; per strategy: share of endgame decisions that differ from the solver and the mean cost in places; the three costliest disagreements) into `docs/baselines/solver/README.md` under "Findings", and write 3-5 sentences on what they teach (which situations the heuristics get wrong). Report numbers honestly: if the solver does not beat `adaptive` with significance, say so.

- [ ] **Step 6: Commit**

```bash
git add sim cli docs/baselines/solver
git commit -m "solver: disagreement benchmark, measured hand-off threshold, head-to-head results

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Documentation and roadmap

**Files:**
- Create: `docs/SOLVER.md`
- Modify: `docs/ROADMAP.md` (the "Phase 21 — Endgame solver and determinized search" entry already exists under "Planned"; update its status and results when phase A is done, and move it up into "Done" with the overview table row), `docs/ARCHITECTURE.md` (a `sim::solver` section), `README.md` (one line under the strategies list, if it lists them)

**Interfaces:**
- Consumes: everything above.
- Produces: user documentation of `endgame-solver[:OPTIONS]`, `cli solve` (the position file format and how to read the table), `cli solver-gap`, the model's assumptions and limits (expected-value over consistent deals, max^n rational opponents, `Final` rule only, no hidden-information play such as bluffing).

- [ ] **Step 1: Write `docs/SOLVER.md`** with these sections: what it computes (and what it does not: not a proof against a specific hidden hand), `endgame-solver` options table (`max-unseen`, `budget`, `fallback`) with the measured defaults from Task 8, `cli solve` with the full position-file format from Task 7 and a worked example read-through of the output (what `exp.place`, `score`, `best/worst`, `best in` mean), `cli solver-gap`, the measured findings (link `docs/baselines/solver/README.md`), limits and the phase B outline (link the spec).
- [ ] **Step 2: Update ROADMAP and ARCHITECTURE** (module map: `solver/{observation,deals,search,report,config,notation,gap}.rs`, the `Round::from_position` engine addition, the `Final`-only decision).
- [ ] **Step 3: Check docs against the code** — every flag and option named in the docs must exist: run `cargo run -q -p cli -- solve --help`, `cargo run -q -p cli -- solver-gap --help` and the `--strategy endgame-solver:max-unseen=8` example from the doc.
- [ ] **Step 4: Full verification before declaring done**

Run: `cargo fmt --check && cargo clippy --workspace --all-targets && cargo test --workspace`
Expected: all green; also run the repository's bench/golden checks named in `docs/CODING_GUIDELINES.md` (they must be unchanged: the solver is opt-in).

- [ ] **Step 5: Commit**

```bash
git add docs README.md
git commit -m "docs: endgame solver (SOLVER.md, roadmap Phase 21, architecture)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** enumeration with ceiling filter (Task 3), exact max^n (4), report/`MoveReport` (4), `EndgameSolver` with fallback and statistics counters (5), CLI strategy spec (6), `cli solve` (7), disagreement benchmark, head-to-head and measurement of `T` (8), docs/roadmap (9), engine addition and `Final`-only decision (1, 5 and the spec revisions). Spec items deliberately *not* built, with the reason in the spec revisions: separate make/unmake `SearchState` (clone-based search is simpler; measured in Task 8), `OpponentModel::Policy` (YAGNI for phase A: max^n is the approved default; the interface in `search.rs` keeps room for it and it gets its own task only if Task 8's findings call for the comparison), ceilings input to `cli solve`, tie-breaks by cards given up. Phase B is out of scope by the approved design.

**Placeholder scan:** two intentional "adapt from the existing code" instructions remain (the determinism test in Task 6 must copy the `run_batch` test setup of `sim/src/match_runner.rs`; the `solve` position parser is specified step by step but not written out) because the exact surrounding signatures must be read from the code at that moment; both state precisely what to copy and what to assert.

**Type consistency:** `Observation` fields (Task 2) are used unchanged in Tasks 3-8; `SolverConfig { max_unseen, node_budget }` (Task 4) is used in Tasks 5-8 and `SolverSpec { solver, fallback }` (Task 5) in Tasks 6 and 8; `MoveStats`/`MoveReport` names and fields (Task 4) match their uses in Tasks 7-8; `Places = [u8; 6]` is 0-based and already absolute because `Round::from_position` enters seats already out first in the finishing order (Task 1 doc comment, Task 4 test).
