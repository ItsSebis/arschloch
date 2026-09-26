# Phase 1 Round State Machine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single round's full trick-taking mechanics to `engine`:
dealing, the trick loop (lead/follow/pass/resolution/hand-exhaustion),
round-end role assignment, and the mandatory card exchange — plus
scripted integration tests proving the whole pipeline works together.

**Architecture:** Five new `engine` modules (`deal`, `role`'s new
`assign_roles`, `exchange`, a crate-private `trick`, and `round`), each
one concept per file per project convention, all still dependency-free.
`round.rs`'s `Round` type is a push-based state machine: a caller submits
one seat's move at a time; `Round` validates and applies it. Looping
multiple rounds into a full match stays out of scope (that's `sim`'s
Phase 2 job per `docs/ARCHITECTURE.md`).

**Tech Stack:** Rust (stable, 1.98.1), no new dependencies.

**Spec:** `/home/sebi/.claude/plans/plan-phase-1-kind-globe.md` (the
approved technical design this plan implements). Also see
`docs/RULES.md` for the rules being encoded, `docs/ARCHITECTURE.md` for
the crate layout, and `docs/CODING_GUIDELINES.md` for conventions.

## Global Constraints

- `engine` stays dependency-free — no new crates in any `Cargo.toml`.
- Suit order: `Diamonds < Hearts < Spades < Clubs`. Rank order: `Two ..
  Ace`. Card comparison: rank, then suit, then (true duplicates only)
  `DuplicateRule` via `deal_index` — all via the existing
  `Card::compare` (`engine/src/card.rs`). Do not add an `Ord` impl to
  `Card`.
- `Combo::beats` (existing, `engine/src/combo.rs`) is same-size AND
  strictly-higher top card — reuse it, don't reimplement combo legality.
- Role tables (`roles_for_player_count`) and exchange-count tables
  (`exchange_counts_for_player_count`), both existing in
  `engine/src/role.rs`, are the only source of truth for per-player-count
  behavior — 3p: `[President, Dorftrottel, Arschloch]` / counts `[1, 0]`;
  4p: `[President, Vize, ViceArschloch, Arschloch]` / counts `[2, 1]`; 5p:
  `[President, Vize, Dorftrottel, ViceArschloch, Arschloch]` / counts
  `[2, 1, 0]`; 6p: `[President, Vize, Offizier, Dummkopf, ViceArschloch,
  Arschloch]` / counts `[3, 2, 1]`.
- Only table sizes 3–6 are supported; every new fallible constructor
  rejects anything else via `Option`/`Result`, never a panic.
- Passing is always legal except when leading a fresh trick (no combo on
  the table yet) — never only "when you can't beat," since voluntary
  passing must be representable for later strategy work.
- **Confirmed house rule:** when a trick's winning play also empties the
  winner's hand (with 2+ opponents still holding cards), the next active
  seat in turn order leads the next trick.
- Illegal moves return a `Result`/`Option` error value; reserve
  `panic!`/`expect`/indexing panics for genuine internal invariants only,
  and give every `pub` function that can legitimately panic a `# Panics`
  doc section (clippy's `missing_panics_doc`, part of this workspace's
  `clippy::pedantic = "warn"`).
- Every phase must pass, in order: `cargo fmt --check`, `cargo clippy
  --workspace --all-targets -- -D warnings`, `cargo test --workspace`.

---

### Task 1: `SeatId` alias + `Combo::cards()` accessor

**Files:**
- Modify: `engine/src/lib.rs`
- Modify: `engine/src/combo.rs`

**Interfaces:**
- Produces: `pub type SeatId = u8;` (crate root), and
  `Combo::cards(&self) -> &[Card]`. Both consumed by every later task in
  this plan.

- [ ] **Step 1: Add `SeatId` to `engine/src/lib.rs`**

Replace the file with:

```rust
//! Pure Arschloch game rules. See docs/RULES.md for the authoritative
//! ruleset and docs/ARCHITECTURE.md for how this crate fits into the
//! workspace.

pub mod card;
pub mod combo;
pub mod role;

pub use card::{Card, DeckVariant, DuplicateRule, Rank, Suit};
pub use combo::Combo;
pub use role::{exchange_counts_for_player_count, roles_for_player_count, Role};

/// A seat's position at the table (0-indexed). Table sizes are 3-6, so
/// `u8` matches `player_count`'s type used throughout this crate.
pub type SeatId = u8;
```

(This step only adds `SeatId`; `deal`/`exchange`/`round` modules and
their re-exports are added by later tasks, one at a time, each showing
the full resulting file.)

- [ ] **Step 2: Add `cards()` to `engine/src/combo.rs`**

Add this method to the existing `impl Combo` block, directly after
`size`:

```rust
    /// The cards making up this combo, in the order given to `Combo::new`.
    #[must_use]
    pub fn cards(&self) -> &[Card] {
        &self.cards
    }
```

Add this test to the existing `#[cfg(test)] mod tests` block:

```rust
    #[test]
    fn cards_returns_the_cards_the_combo_was_built_from() {
        let cards = vec![
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Seven, Suit::Diamonds),
        ];
        let combo = Combo::new(cards.clone()).unwrap();
        assert_eq!(combo.cards(), &cards[..]);
    }
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p engine`
Expected: all 23 tests pass (22 existing + 1 new).

- [ ] **Step 4: Commit**

```bash
git add engine/src/lib.rs engine/src/combo.rs
git commit -m "engine: add SeatId alias and Combo::cards() accessor"
```

---

### Task 2: `deal.rs`

**Files:**
- Create: `engine/src/deal.rs`
- Modify: `engine/src/lib.rs`

**Interfaces:**
- Consumes: `Card`, `DuplicateRule` (`engine/src/card.rs`), `SeatId`
  (crate root, Task 1).
- Produces: `deal(cards: Vec<Card>, player_count: u8) -> Option<Vec<Vec<Card>>>`,
  `lowest_card_holder(hands: &[Vec<Card>], duplicate_rule: DuplicateRule) -> Option<SeatId>`.
  Used by Task 7's integration test; `deal`'s output shape (`Vec<Vec<Card>>`)
  is exactly `Round::new`'s (Task 6) `hands` parameter.

- [ ] **Step 1: Write `engine/src/deal.rs` with its own tests**

```rust
//! Dealing an already-ordered deck into hands, and finding the session's
//! very first leader. See docs/RULES.md, "Players & Deck" and "First
//! lead of a trick / round".

use crate::card::{Card, DuplicateRule};
use crate::SeatId;

/// Deals `cards` round-robin into `player_count` hands, in the order
/// given — card `i` goes to seat `i % player_count`, so the earliest
/// seats get the extra card(s) when `cards.len()` doesn't divide evenly,
/// matching docs/RULES.md's documented remainder rule. The caller is
/// responsible for shuffling `cards` beforehand; this function only
/// distributes them.
///
/// Returns `None` if `player_count` is 0, or if some seat would end up
/// with an empty hand (`cards.len() < player_count`).
#[must_use]
pub fn deal(cards: Vec<Card>, player_count: u8) -> Option<Vec<Vec<Card>>> {
    if player_count == 0 || cards.len() < usize::from(player_count) {
        return None;
    }
    let seat_count = usize::from(player_count);
    let mut hands = vec![Vec::new(); seat_count];
    for (i, card) in cards.into_iter().enumerate() {
        hands[i % seat_count].push(card);
    }
    Some(hands)
}

/// Finds which seat holds the single lowest card across all hands, by
/// `Card::compare` under `duplicate_rule`. Used only to seed the very
/// first leader of a session's first round (docs/RULES.md, "First
/// lead"); every later round's leader is that round's Arschloch instead.
///
/// Returns `None` if `hands` is empty or every hand is empty.
#[must_use]
pub fn lowest_card_holder(hands: &[Vec<Card>], duplicate_rule: DuplicateRule) -> Option<SeatId> {
    hands
        .iter()
        .enumerate()
        .flat_map(|(seat, hand)| hand.iter().map(move |card| (seat, card)))
        .min_by(|(_, a), (_, b)| a.compare(b, duplicate_rule))
        .map(|(seat, _)| SeatId::try_from(seat).expect("table sizes are capped at 6 seats"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{Rank, Suit};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit, 0)
    }

    #[test]
    fn deal_splits_cards_round_robin() {
        let cards = vec![
            card(Rank::Two, Suit::Clubs),
            card(Rank::Three, Suit::Clubs),
            card(Rank::Four, Suit::Clubs),
            card(Rank::Five, Suit::Clubs),
        ];
        let hands = deal(cards, 2).unwrap();
        assert_eq!(hands.len(), 2);
        assert_eq!(
            hands[0],
            vec![card(Rank::Two, Suit::Clubs), card(Rank::Four, Suit::Clubs)]
        );
        assert_eq!(
            hands[1],
            vec![card(Rank::Three, Suit::Clubs), card(Rank::Five, Suit::Clubs)]
        );
    }

    #[test]
    fn deal_gives_earliest_seats_the_remainder() {
        let cards = vec![
            card(Rank::Two, Suit::Clubs),
            card(Rank::Three, Suit::Clubs),
            card(Rank::Four, Suit::Clubs),
        ];
        let hands = deal(cards, 2).unwrap();
        assert_eq!(hands[0].len(), 2);
        assert_eq!(hands[1].len(), 1);
    }

    #[test]
    fn deal_rejects_zero_players() {
        assert!(deal(vec![card(Rank::Two, Suit::Clubs)], 0).is_none());
    }

    #[test]
    fn deal_rejects_too_few_cards_for_every_seat() {
        assert!(deal(vec![card(Rank::Two, Suit::Clubs)], 2).is_none());
    }

    #[test]
    fn lowest_card_holder_finds_the_minimum_across_hands() {
        let hands = vec![
            vec![card(Rank::King, Suit::Clubs)],
            vec![
                card(Rank::Two, Suit::Diamonds),
                card(Rank::Ace, Suit::Spades),
            ],
        ];
        assert_eq!(
            lowest_card_holder(&hands, DuplicateRule::FirstDealtWins),
            Some(1)
        );
    }

    #[test]
    fn lowest_card_holder_returns_none_for_empty_hands() {
        let hands: Vec<Vec<Card>> = vec![vec![], vec![]];
        assert_eq!(lowest_card_holder(&hands, DuplicateRule::FirstDealtWins), None);
    }
}
```

- [ ] **Step 2: Wire the module into `engine/src/lib.rs`**

```rust
//! Pure Arschloch game rules. See docs/RULES.md for the authoritative
//! ruleset and docs/ARCHITECTURE.md for how this crate fits into the
//! workspace.

pub mod card;
pub mod combo;
pub mod deal;
pub mod role;

pub use card::{Card, DeckVariant, DuplicateRule, Rank, Suit};
pub use combo::Combo;
pub use deal::{deal, lowest_card_holder};
pub use role::{exchange_counts_for_player_count, roles_for_player_count, Role};

/// A seat's position at the table (0-indexed). Table sizes are 3-6, so
/// `u8` matches `player_count`'s type used throughout this crate.
pub type SeatId = u8;
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p engine`
Expected: all 29 tests pass (23 from Task 1 + 6 new).

- [ ] **Step 4: Commit**

```bash
git add engine/src/deal.rs engine/src/lib.rs
git commit -m "engine: add deal() and lowest_card_holder()"
```

---

### Task 3: `role.rs` — `assign_roles`

**Files:**
- Modify: `engine/src/role.rs`

**Interfaces:**
- Consumes: `SeatId` (crate root, Task 1).
- Produces: `assign_roles(finishing_order: &[SeatId], player_count: u8) -> Option<Vec<Role>>`.
  Output is seat-indexed (`role_by_seat`) — exactly Task 4's `exchange`
  input shape, and exactly what Task 7's integration test feeds into
  `exchange` after a round completes.

- [ ] **Step 1: Add `assign_roles` to `engine/src/role.rs`**

Add `use crate::SeatId;` as the first line after the module doc comment,
then add this function after `exchange_counts_for_player_count`:

```rust
/// Maps a completed round's finishing order (the seat that emptied its
/// hand first, ..., the seat ranked last) to each seat's new `Role`, via
/// `roles_for_player_count`. Returns `None` if `player_count` is
/// unsupported, if `finishing_order`'s length doesn't match it, or if
/// `finishing_order` isn't a valid permutation of every seat exactly
/// once (e.g. a duplicate or out-of-range seat).
#[must_use]
pub fn assign_roles(finishing_order: &[SeatId], player_count: u8) -> Option<Vec<Role>> {
    let roles = roles_for_player_count(player_count)?;
    if finishing_order.len() != roles.len() {
        return None;
    }
    let seat_count = usize::from(player_count);
    let mut seen = vec![false; seat_count];
    for &seat in finishing_order {
        let seat = usize::from(seat);
        if seat >= seat_count || seen[seat] {
            return None;
        }
        seen[seat] = true;
    }
    let mut role_by_seat = vec![roles[roles.len() - 1]; seat_count];
    for (place, &seat) in finishing_order.iter().enumerate() {
        role_by_seat[usize::from(seat)] = roles[place];
    }
    Some(role_by_seat)
}
```

Add these tests to the existing `#[cfg(test)] mod tests` block, after
`every_supported_table_size_has_matching_role_and_exchange_lengths`:

```rust
    #[test]
    fn assign_roles_maps_finishing_order_to_roles_by_seat() {
        // Seat 2 finished first (President), seat 0 second (Dorftrottel),
        // seat 1 last (Arschloch).
        let role_by_seat = assign_roles(&[2, 0, 1], 3).unwrap();
        assert_eq!(role_by_seat[2], Role::President);
        assert_eq!(role_by_seat[0], Role::Dorftrottel);
        assert_eq!(role_by_seat[1], Role::Arschloch);
    }

    #[test]
    fn assign_roles_rejects_unsupported_player_count() {
        assert_eq!(assign_roles(&[0, 1], 2), None);
    }

    #[test]
    fn assign_roles_rejects_wrong_length_finishing_order() {
        assert_eq!(assign_roles(&[0, 1], 3), None);
    }

    #[test]
    fn assign_roles_rejects_duplicate_or_out_of_range_seats() {
        assert_eq!(assign_roles(&[0, 0, 1], 3), None);
        assert_eq!(assign_roles(&[0, 1, 5], 3), None);
    }
```

- [ ] **Step 2: Run the tests**

Run: `cargo test -p engine`
Expected: all 33 tests pass (29 from Task 2 + 4 new).

- [ ] **Step 3: Commit**

```bash
git add engine/src/role.rs
git commit -m "engine: add assign_roles"
```

---

### Task 4: `exchange.rs`

**Files:**
- Create: `engine/src/exchange.rs`
- Modify: `engine/src/lib.rs`

**Interfaces:**
- Consumes: `Card`, `DuplicateRule` (`engine/src/card.rs`),
  `Role`, `roles_for_player_count`, `exchange_counts_for_player_count`
  (`engine/src/role.rs`).
- Produces: `ExchangeError`, `exchange(hands: &mut [Vec<Card>], role_by_seat: &[Role], duplicate_rule: DuplicateRule) -> Result<(), ExchangeError>`.
  Used by Task 7's integration test.

**Design note (important — do not simplify away):** validate every
pair's card counts on *both* sides *before* mutating any hand, so a
failed exchange leaves every hand completely untouched. Do not take
cards from one side of a pair, then discover the other side is short and
return an error with the first side already mutated.

- [ ] **Step 1: Write `engine/src/exchange.rs` with its own tests**

```rust
//! The mandatory pre-round card exchange ("Drücken"). See docs/RULES.md,
//! "Card Exchange (\"Drücken\")".

use crate::card::{Card, DuplicateRule};
use crate::role::{exchange_counts_for_player_count, roles_for_player_count, Role};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExchangeError {
    /// `hands.len() != role_by_seat.len()`.
    SeatCountMismatch,
    /// `hands.len()` isn't a supported table size (3-6).
    UnsupportedPlayerCount,
    /// `role_by_seat` didn't contain every role for this player count
    /// exactly once.
    InvalidRoleMapping,
    /// A seat's hand had fewer cards than its required exchange count.
    NotEnoughCards,
}

/// Applies the mandatory "Drücken" exchange to freshly dealt hands for
/// the round about to start, using the role each seat held at the end
/// of the *previous* round (`role_by_seat[i]` = seat `i`'s previous
/// role). Naive tie-break: the lower role's highest N cards move to the
/// higher role, whose lowest N cards move back, both ordered by
/// `Card::compare` under `duplicate_rule` — no strategic selection
/// (that's a later phase). On any error, `hands` is left completely
/// unchanged.
pub fn exchange(
    hands: &mut [Vec<Card>],
    role_by_seat: &[Role],
    duplicate_rule: DuplicateRule,
) -> Result<(), ExchangeError> {
    if hands.len() != role_by_seat.len() {
        return Err(ExchangeError::SeatCountMismatch);
    }
    let player_count =
        u8::try_from(hands.len()).map_err(|_| ExchangeError::UnsupportedPlayerCount)?;
    let roles = roles_for_player_count(player_count).ok_or(ExchangeError::UnsupportedPlayerCount)?;
    let counts = exchange_counts_for_player_count(player_count)
        .ok_or(ExchangeError::UnsupportedPlayerCount)?;
    validate_role_mapping(role_by_seat, roles)?;

    let seat_of = |role: Role| -> usize {
        role_by_seat
            .iter()
            .position(|&r| r == role)
            .expect("validate_role_mapping already confirmed every role is present exactly once")
    };

    // Validate every pair has enough cards on both sides before mutating
    // anything, so a failed exchange leaves hands untouched.
    for (i, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let count = usize::from(count);
        let high_seat = seat_of(roles[i]);
        let low_seat = seat_of(roles[roles.len() - 1 - i]);
        if hands[low_seat].len() < count || hands[high_seat].len() < count {
            return Err(ExchangeError::NotEnoughCards);
        }
    }

    let mut incoming: Vec<Vec<Card>> = vec![Vec::new(); hands.len()];
    for (i, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let count = usize::from(count);
        let high_seat = seat_of(roles[i]);
        let low_seat = seat_of(roles[roles.len() - 1 - i]);

        let from_low = take_highest(&mut hands[low_seat], count, duplicate_rule);
        let from_high = take_lowest(&mut hands[high_seat], count, duplicate_rule);

        incoming[high_seat].extend(from_low);
        incoming[low_seat].extend(from_high);
    }

    for (hand, extra) in hands.iter_mut().zip(incoming) {
        hand.extend(extra);
    }
    Ok(())
}

fn validate_role_mapping(role_by_seat: &[Role], roles: &[Role]) -> Result<(), ExchangeError> {
    for &role in roles {
        if role_by_seat.iter().filter(|&&r| r == role).count() != 1 {
            return Err(ExchangeError::InvalidRoleMapping);
        }
    }
    Ok(())
}

fn take_highest(hand: &mut Vec<Card>, count: usize, duplicate_rule: DuplicateRule) -> Vec<Card> {
    hand.sort_by(|a, b| a.compare(b, duplicate_rule));
    hand.split_off(hand.len() - count)
}

fn take_lowest(hand: &mut Vec<Card>, count: usize, duplicate_rule: DuplicateRule) -> Vec<Card> {
    hand.sort_by(|a, b| a.compare(b, duplicate_rule));
    hand.drain(0..count).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{Rank, Suit};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit, 0)
    }

    #[test]
    fn four_player_exchange_swaps_highest_and_lowest() {
        let role_by_seat = vec![
            Role::President,
            Role::Vize,
            Role::ViceArschloch,
            Role::Arschloch,
        ];
        let mut hands = vec![
            vec![card(Rank::Two, Suit::Clubs), card(Rank::Three, Suit::Clubs)],
            vec![card(Rank::Four, Suit::Clubs), card(Rank::Five, Suit::Clubs)],
            vec![card(Rank::Six, Suit::Clubs), card(Rank::Seven, Suit::Clubs)],
            vec![card(Rank::King, Suit::Clubs), card(Rank::Ace, Suit::Clubs)],
        ];
        exchange(&mut hands, &role_by_seat, DuplicateRule::FirstDealtWins).unwrap();

        assert_eq!(
            hands[0],
            vec![card(Rank::King, Suit::Clubs), card(Rank::Ace, Suit::Clubs)]
        );
        assert_eq!(
            hands[1],
            vec![card(Rank::Five, Suit::Clubs), card(Rank::Seven, Suit::Clubs)]
        );
        assert_eq!(
            hands[2],
            vec![card(Rank::Six, Suit::Clubs), card(Rank::Four, Suit::Clubs)]
        );
        assert_eq!(
            hands[3],
            vec![card(Rank::Two, Suit::Clubs), card(Rank::Three, Suit::Clubs)]
        );
    }

    #[test]
    fn three_player_lone_middle_role_is_untouched() {
        let role_by_seat = vec![Role::President, Role::Dorftrottel, Role::Arschloch];
        let dorftrottel_hand = vec![card(Rank::Nine, Suit::Clubs), card(Rank::Ten, Suit::Clubs)];
        let mut hands = vec![
            vec![card(Rank::Two, Suit::Clubs)],
            dorftrottel_hand.clone(),
            vec![card(Rank::Ace, Suit::Clubs)],
        ];
        exchange(&mut hands, &role_by_seat, DuplicateRule::FirstDealtWins).unwrap();
        assert_eq!(hands[1], dorftrottel_hand);
    }

    #[test]
    fn exchange_rejects_seat_count_mismatch() {
        let mut hands = vec![vec![card(Rank::Two, Suit::Clubs)]; 3];
        let role_by_seat = vec![Role::President, Role::Arschloch];
        assert_eq!(
            exchange(&mut hands, &role_by_seat, DuplicateRule::FirstDealtWins),
            Err(ExchangeError::SeatCountMismatch)
        );
    }

    #[test]
    fn exchange_rejects_unsupported_player_count() {
        let mut hands = vec![vec![card(Rank::Two, Suit::Clubs)]; 2];
        let role_by_seat = vec![Role::President, Role::Arschloch];
        assert_eq!(
            exchange(&mut hands, &role_by_seat, DuplicateRule::FirstDealtWins),
            Err(ExchangeError::UnsupportedPlayerCount)
        );
    }

    #[test]
    fn exchange_rejects_invalid_role_mapping() {
        let mut hands = vec![vec![card(Rank::Two, Suit::Clubs)]; 3];
        // President repeated, Arschloch missing.
        let role_by_seat = vec![Role::President, Role::President, Role::Dorftrottel];
        assert_eq!(
            exchange(&mut hands, &role_by_seat, DuplicateRule::FirstDealtWins),
            Err(ExchangeError::InvalidRoleMapping)
        );
    }

    #[test]
    fn exchange_rejects_not_enough_cards() {
        let mut hands = vec![
            vec![],
            vec![card(Rank::Nine, Suit::Clubs), card(Rank::Ten, Suit::Clubs)],
            vec![card(Rank::Ace, Suit::Clubs), card(Rank::King, Suit::Clubs)],
        ];
        let role_by_seat = vec![Role::President, Role::Dorftrottel, Role::Arschloch];
        assert_eq!(
            exchange(&mut hands, &role_by_seat, DuplicateRule::FirstDealtWins),
            Err(ExchangeError::NotEnoughCards)
        );
    }

    #[test]
    fn exchange_leaves_hands_untouched_when_it_fails() {
        let original = vec![
            vec![],
            vec![card(Rank::Nine, Suit::Clubs), card(Rank::Ten, Suit::Clubs)],
            vec![card(Rank::Ace, Suit::Clubs), card(Rank::King, Suit::Clubs)],
        ];
        let mut hands = original.clone();
        let role_by_seat = vec![Role::President, Role::Dorftrottel, Role::Arschloch];
        assert!(exchange(&mut hands, &role_by_seat, DuplicateRule::FirstDealtWins).is_err());
        assert_eq!(hands, original);
    }
}
```

- [ ] **Step 2: Wire the module into `engine/src/lib.rs`**

```rust
//! Pure Arschloch game rules. See docs/RULES.md for the authoritative
//! ruleset and docs/ARCHITECTURE.md for how this crate fits into the
//! workspace.

pub mod card;
pub mod combo;
pub mod deal;
pub mod exchange;
pub mod role;

pub use card::{Card, DeckVariant, DuplicateRule, Rank, Suit};
pub use combo::Combo;
pub use deal::{deal, lowest_card_holder};
pub use exchange::{exchange, ExchangeError};
pub use role::{assign_roles, exchange_counts_for_player_count, roles_for_player_count, Role};

/// A seat's position at the table (0-indexed). Table sizes are 3-6, so
/// `u8` matches `player_count`'s type used throughout this crate.
pub type SeatId = u8;
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p engine`
Expected: all 40 tests pass (33 from Task 3 + 7 new).

- [ ] **Step 4: Commit**

```bash
git add engine/src/exchange.rs engine/src/lib.rs
git commit -m "engine: add exchange() for the mandatory pre-round card swap"
```

---

### Task 5: `trick.rs` (crate-private)

**Files:**
- Create: `engine/src/trick.rs`
- Modify: `engine/src/lib.rs`

**Interfaces:**
- Consumes: `SeatId` (crate root, Task 1). Nothing else — deliberately
  card/combo-agnostic.
- Produces (all `pub(crate)`, not part of the public API): `Trick`,
  `Trick::new(leader) -> Self`, `Trick::leader(&self) -> SeatId`,
  `Trick::turn(&self) -> SeatId`, `Trick::has_current_play(&self) -> bool`,
  `Trick::record_play(&mut self, seat: SeatId, active: &[bool])`,
  `Trick::record_pass(&mut self, seat: SeatId, active: &[bool]) -> Option<SeatId>`.
  Consumed by Task 6's `round.rs`.

- [ ] **Step 1: Write `engine/src/trick.rs` with its own tests**

```rust
//! Turn-order and pass-counting bookkeeping for a single trick, kept
//! deliberately unaware of cards or combos so it's simple to reason
//! about in isolation. `round.rs` owns hand contents and combo legality;
//! this type only tracks whose turn it is and when a trick ends. See
//! docs/RULES.md, "Playing a Round".

use crate::SeatId;

/// One trick's turn-taking state: who led it, whose turn it is now, who
/// currently holds the winning play (if anyone has played yet), and how
/// many consecutive passes are needed to resolve it.
#[derive(Debug)]
pub(crate) struct Trick {
    leader: SeatId,
    turn: SeatId,
    winner: Option<SeatId>,
    passes_since_last_play: u8,
    required_passes: u8,
}

impl Trick {
    pub(crate) fn new(leader: SeatId) -> Self {
        Self {
            leader,
            turn: leader,
            winner: None,
            passes_since_last_play: 0,
            required_passes: 0,
        }
    }

    pub(crate) fn leader(&self) -> SeatId {
        self.leader
    }

    pub(crate) fn turn(&self) -> SeatId {
        self.turn
    }

    /// Whether anyone has played a combo yet this trick (the leader must
    /// play; every later active seat may play or pass).
    pub(crate) fn has_current_play(&self) -> bool {
        self.winner.is_some()
    }

    /// Records that `seat` (the current `turn`) played a combo. `active`
    /// reflects hand-emptiness immediately after this play (so if `seat`
    /// just emptied their hand, `active[seat]` is already `false`).
    pub(crate) fn record_play(&mut self, seat: SeatId, active: &[bool]) {
        self.winner = Some(seat);
        self.passes_since_last_play = 0;
        let active_count = u8::try_from(active.iter().filter(|&&a| a).count())
            .expect("table sizes are capped at 6 seats");
        let winner_still_active = active[usize::from(seat)];
        self.required_passes = active_count - u8::from(winner_still_active);
        self.turn = next_active_seat_after(seat, active);
    }

    /// Records that `seat` (the current `turn`) passed. `active` is
    /// unaffected by a pass, so it's the same snapshot in effect since
    /// the last play. Returns `Some(new_leader)` if this pass resolves
    /// the trick (every other active seat has now passed since the last
    /// play), else `None` (and `turn` has advanced).
    pub(crate) fn record_pass(&mut self, seat: SeatId, active: &[bool]) -> Option<SeatId> {
        self.passes_since_last_play += 1;
        if self.passes_since_last_play < self.required_passes {
            self.turn = next_active_seat_after(seat, active);
            return None;
        }
        let winner = self
            .winner
            .expect("record_pass is only reachable after a play established required_passes > 0");
        let new_leader = if active[usize::from(winner)] {
            winner
        } else {
            next_active_seat_after(winner, active)
        };
        Some(new_leader)
    }
}

fn next_active_seat_after(seat: SeatId, active: &[bool]) -> SeatId {
    let n = active.len();
    let start = usize::from(seat);
    for offset in 1..=n {
        let candidate = (start + offset) % n;
        if active[candidate] {
            return SeatId::try_from(candidate)
                .expect("candidate is a valid index into active, which is at most 6 seats");
        }
    }
    unreachable!(
        "next_active_seat_after requires at least one active seat; \
         callers must ensure the round isn't already complete"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_trick_starts_with_leader_to_move_and_no_current_play() {
        let trick = Trick::new(2);
        assert_eq!(trick.leader(), 2);
        assert_eq!(trick.turn(), 2);
        assert!(!trick.has_current_play());
    }

    #[test]
    fn record_play_sets_winner_and_required_passes_for_remaining_active_seats() {
        let mut trick = Trick::new(0);
        let active = [true, true, true, true];
        trick.record_play(0, &active);
        assert!(trick.has_current_play());
        assert_eq!(trick.turn(), 1);
        assert_eq!(trick.record_pass(1, &active), None);
        assert_eq!(trick.record_pass(2, &active), None);
        assert_eq!(trick.record_pass(3, &active), Some(0));
    }

    #[test]
    fn a_beating_play_mid_trick_resets_required_passes() {
        let mut trick = Trick::new(0);
        let active = [true, true, true, true];
        trick.record_play(0, &active);
        assert_eq!(trick.record_pass(1, &active), None);
        trick.record_play(2, &active); // seat 2 beats seat 0's combo
        assert_eq!(trick.turn(), 3);
        // All 3 other active seats (3, 0, 1) must pass again to resolve,
        // even though seat 1 already passed on the previous play.
        assert_eq!(trick.record_pass(3, &active), None);
        assert_eq!(trick.record_pass(0, &active), None);
        assert_eq!(trick.record_pass(1, &active), Some(2));
    }

    #[test]
    fn everyone_but_the_leader_passing_returns_the_leader_as_new_leader() {
        let mut trick = Trick::new(1);
        let active = [true, true, true];
        trick.record_play(1, &active);
        assert_eq!(trick.record_pass(2, &active), None);
        assert_eq!(trick.record_pass(0, &active), Some(1));
    }

    #[test]
    fn winner_emptying_their_hand_hands_the_lead_to_the_next_active_seat() {
        let mut trick = Trick::new(0);
        // Seat 0 plays and empties their hand.
        let active_after_play = [false, true, true];
        trick.record_play(0, &active_after_play);
        // Only 2 active seats remain (1 and 2); both must pass to resolve.
        assert_eq!(trick.record_pass(1, &active_after_play), None);
        assert_eq!(trick.record_pass(2, &active_after_play), Some(1));
    }
}
```

- [ ] **Step 2: Wire the module into `engine/src/lib.rs`**

Add `mod trick;` (no `pub` — this module is crate-internal only) to the
`pub mod`/`mod` list. Do not add any `pub use` for it. Full resulting
file:

```rust
//! Pure Arschloch game rules. See docs/RULES.md for the authoritative
//! ruleset and docs/ARCHITECTURE.md for how this crate fits into the
//! workspace.

pub mod card;
pub mod combo;
pub mod deal;
pub mod exchange;
pub mod role;
mod trick;

pub use card::{Card, DeckVariant, DuplicateRule, Rank, Suit};
pub use combo::Combo;
pub use deal::{deal, lowest_card_holder};
pub use exchange::{exchange, ExchangeError};
pub use role::{assign_roles, exchange_counts_for_player_count, roles_for_player_count, Role};

/// A seat's position at the table (0-indexed). Table sizes are 3-6, so
/// `u8` matches `player_count`'s type used throughout this crate.
pub type SeatId = u8;
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p engine`
Expected: all 45 tests pass (40 from Task 4 + 5 new). (Clippy note for
later: `unreachable!`/`expect` inside `pub(crate)` items don't require
`# Panics` docs — that lint targets the crate's public API surface,
which `trick` isn't part of.)

- [ ] **Step 4: Commit**

```bash
git add engine/src/trick.rs engine/src/lib.rs
git commit -m "engine: add crate-private Trick turn-order bookkeeping"
```

---

### Task 6: `round.rs`

**Files:**
- Create: `engine/src/round.rs`
- Modify: `engine/src/lib.rs`

**Interfaces:**
- Consumes: `Card`, `DuplicateRule` (`engine/src/card.rs`), `Combo`
  (`engine/src/combo.rs`, incl. `cards()` from Task 1), `Trick`
  (`engine/src/trick.rs`, Task 5), `SeatId` (crate root, Task 1).
- Produces: `Move`, `MoveError`, `Round`, `Round::new`, `seat_to_move`,
  `hand`, `is_active`, `is_complete`, `finishing_order`, `current_combo`,
  `current_trick_leader`, `submit_move`. This is the last Phase 1 type;
  Task 7's integration test drives it end-to-end.

**Design note (important — do not simplify away):** `validate_play` must
consume a *cloned* copy of the hand while checking each of the combo's
cards, removing each match before checking the next — not a naive
independent `.contains()` check per card. Otherwise a combo that lists
the same physical card twice would pass validation (each lookup
independently finds it in the *original* hand) and then panic in
`apply_play` when the second removal can't find a card that's already
gone.

- [ ] **Step 1: Write `engine/src/round.rs` with its own tests**

```rust
//! The single-round trick-taking state machine: hands, combo legality,
//! and finishing order. See docs/RULES.md, "Playing a Round".

use crate::card::{Card, DuplicateRule};
use crate::combo::Combo;
use crate::trick::Trick;
use crate::SeatId;

/// A move a seat can submit on its turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Move {
    Play(Combo),
    Pass,
}

/// Why a submitted move was rejected. The round's state is unchanged
/// when this is returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoveError {
    /// The round has already finished; no more moves can be submitted.
    RoundAlreadyComplete,
    /// It isn't `seat`'s turn.
    NotYourTurn { expected: SeatId },
    /// No combo is currently on the table, so the acting seat must play,
    /// not pass.
    CannotPassOnLead,
    /// The submitted combo contains a card the seat doesn't hold (or
    /// lists a card it holds only once, twice).
    CardNotInHand(Card),
    /// The submitted combo doesn't beat the current combo on the table
    /// (wrong size, or not strictly higher).
    ComboDoesNotBeat,
}

/// A single round's trick-taking state, from a freshly dealt (and, if
/// applicable, already-exchanged) set of hands through to every seat's
/// finishing order.
#[derive(Debug)]
pub struct Round {
    hands: Vec<Vec<Card>>,
    duplicate_rule: DuplicateRule,
    finishing_order: Vec<SeatId>,
    current_combo: Option<Combo>,
    trick: Trick,
}

impl Round {
    /// Starts a new round from `hands` (one per seat, already dealt and,
    /// for rounds after the first, already exchanged), with
    /// `first_leader` leading the first trick. Returns `None` if
    /// `hands.len()` isn't a supported table size (3-6), `first_leader`
    /// is out of range, or any hand starts empty (a round can't begin
    /// with a seat already out).
    #[must_use]
    pub fn new(hands: Vec<Vec<Card>>, duplicate_rule: DuplicateRule, first_leader: SeatId) -> Option<Self> {
        if !(3..=6).contains(&hands.len()) {
            return None;
        }
        if usize::from(first_leader) >= hands.len() {
            return None;
        }
        if hands.iter().any(Vec::is_empty) {
            return None;
        }
        Some(Self {
            hands,
            duplicate_rule,
            finishing_order: Vec::new(),
            current_combo: None,
            trick: Trick::new(first_leader),
        })
    }

    /// The seat that must act next, or `None` if the round is complete.
    #[must_use]
    pub fn seat_to_move(&self) -> Option<SeatId> {
        if self.is_complete() {
            None
        } else {
            Some(self.trick.turn())
        }
    }

    /// `seat`'s current hand.
    ///
    /// # Panics
    ///
    /// Panics if `seat` is not a valid seat for this round
    /// (`usize::from(seat) >= player_count`).
    #[must_use]
    pub fn hand(&self, seat: SeatId) -> &[Card] {
        &self.hands[usize::from(seat)]
    }

    /// Whether `seat` still holds cards (hasn't finished the round yet).
    ///
    /// # Panics
    ///
    /// Panics if `seat` is not a valid seat for this round.
    #[must_use]
    pub fn is_active(&self, seat: SeatId) -> bool {
        !self.hands[usize::from(seat)].is_empty()
    }

    /// Whether every seat but one has emptied its hand.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.finishing_order.len() == self.hands.len()
    }

    /// Seats in the order they emptied their hand (first out first),
    /// including the auto-appended last seat once the round completes.
    #[must_use]
    pub fn finishing_order(&self) -> &[SeatId] {
        &self.finishing_order
    }

    /// The combo currently on the table for the active trick, or `None`
    /// if the active seat must lead a fresh trick.
    #[must_use]
    pub fn current_combo(&self) -> Option<&Combo> {
        self.current_combo.as_ref()
    }

    /// The seat that led the trick currently in progress.
    #[must_use]
    pub fn current_trick_leader(&self) -> SeatId {
        self.trick.leader()
    }

    /// Submits `seat`'s move. On success, the round's state has already
    /// advanced (hand updated, trick/finishing-order progressed as
    /// needed). On failure, the round's state is unchanged.
    pub fn submit_move(&mut self, seat: SeatId, mv: Move) -> Result<(), MoveError> {
        if self.is_complete() {
            return Err(MoveError::RoundAlreadyComplete);
        }
        let expected = self.trick.turn();
        if seat != expected {
            return Err(MoveError::NotYourTurn { expected });
        }

        match mv {
            Move::Pass => {
                if !self.trick.has_current_play() {
                    return Err(MoveError::CannotPassOnLead);
                }
                self.apply_pass(seat);
                Ok(())
            }
            Move::Play(combo) => {
                self.validate_play(seat, &combo)?;
                self.apply_play(seat, combo);
                Ok(())
            }
        }
    }

    fn validate_play(&self, seat: SeatId, combo: &Combo) -> Result<(), MoveError> {
        let mut remaining_hand = self.hands[usize::from(seat)].clone();
        for card in combo.cards() {
            match remaining_hand.iter().position(|c| c == card) {
                Some(index) => {
                    remaining_hand.remove(index);
                }
                None => return Err(MoveError::CardNotInHand(*card)),
            }
        }
        if let Some(current) = &self.current_combo {
            if !combo.beats(current, self.duplicate_rule) {
                return Err(MoveError::ComboDoesNotBeat);
            }
        }
        Ok(())
    }

    fn apply_play(&mut self, seat: SeatId, combo: Combo) {
        let hand = &mut self.hands[usize::from(seat)];
        for card in combo.cards() {
            let position = hand
                .iter()
                .position(|c| c == card)
                .expect("validate_play already confirmed every card in the combo has a matching card in hand");
            hand.remove(position);
        }
        let just_emptied = hand.is_empty();
        self.current_combo = Some(combo);

        if just_emptied {
            self.finishing_order.push(seat);
            if self.check_for_single_seat_remaining() {
                return;
            }
        }

        let active = self.active_mask();
        self.trick.record_play(seat, &active);
    }

    fn apply_pass(&mut self, seat: SeatId) {
        let active = self.active_mask();
        if let Some(new_leader) = self.trick.record_pass(seat, &active) {
            self.current_combo = None;
            self.trick = Trick::new(new_leader);
        }
    }

    /// If exactly one seat still holds cards, appends it to the
    /// finishing order and completes the round. Returns whether this
    /// happened, so `apply_play` can skip further trick bookkeeping.
    fn check_for_single_seat_remaining(&mut self) -> bool {
        let remaining_active_count = self.hands.iter().filter(|h| !h.is_empty()).count();
        if remaining_active_count == 1 {
            let last_seat = SeatId::try_from(
                self.hands
                    .iter()
                    .position(|h| !h.is_empty())
                    .expect("remaining_active_count == 1"),
            )
            .expect("table sizes are capped at 6 seats");
            self.finishing_order.push(last_seat);
            true
        } else {
            false
        }
    }

    fn active_mask(&self) -> Vec<bool> {
        self.hands.iter().map(|h| !h.is_empty()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{Rank, Suit};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit, 0)
    }

    fn combo(cards: Vec<Card>) -> Combo {
        Combo::new(cards).unwrap()
    }

    #[test]
    fn new_round_rejects_unsupported_player_counts() {
        assert!(Round::new(
            vec![vec![card(Rank::Two, Suit::Clubs)]; 2],
            DuplicateRule::FirstDealtWins,
            0
        )
        .is_none());
        assert!(Round::new(
            vec![vec![card(Rank::Two, Suit::Clubs)]; 7],
            DuplicateRule::FirstDealtWins,
            0
        )
        .is_none());
    }

    #[test]
    fn new_round_rejects_out_of_range_leader() {
        let hands = vec![vec![card(Rank::Two, Suit::Clubs)]; 3];
        assert!(Round::new(hands, DuplicateRule::FirstDealtWins, 3).is_none());
    }

    #[test]
    fn new_round_rejects_an_empty_starting_hand() {
        let hands = vec![
            vec![card(Rank::Two, Suit::Clubs)],
            vec![],
            vec![card(Rank::Three, Suit::Clubs)],
        ];
        assert!(Round::new(hands, DuplicateRule::FirstDealtWins, 0).is_none());
    }

    #[test]
    fn leader_must_play_and_cannot_pass() {
        let hands = vec![
            vec![card(Rank::Two, Suit::Clubs)],
            vec![card(Rank::Three, Suit::Clubs)],
            vec![card(Rank::Four, Suit::Clubs)],
        ];
        let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
        assert_eq!(round.submit_move(0, Move::Pass), Err(MoveError::CannotPassOnLead));
    }

    #[test]
    fn rejects_a_move_from_the_wrong_seat() {
        let hands = vec![
            vec![card(Rank::Two, Suit::Clubs)],
            vec![card(Rank::Three, Suit::Clubs)],
            vec![card(Rank::Four, Suit::Clubs)],
        ];
        let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
        let play = Move::Play(combo(vec![card(Rank::Three, Suit::Clubs)]));
        assert_eq!(
            round.submit_move(1, play),
            Err(MoveError::NotYourTurn { expected: 0 })
        );
    }

    #[test]
    fn rejects_a_combo_with_a_card_not_in_hand() {
        let hands = vec![
            vec![card(Rank::Two, Suit::Clubs)],
            vec![card(Rank::Three, Suit::Clubs)],
            vec![card(Rank::Four, Suit::Clubs)],
        ];
        let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
        let play = Move::Play(combo(vec![card(Rank::Nine, Suit::Clubs)]));
        assert_eq!(
            round.submit_move(0, play),
            Err(MoveError::CardNotInHand(card(Rank::Nine, Suit::Clubs)))
        );
    }

    #[test]
    fn rejects_a_combo_that_does_not_beat_the_current_combo() {
        let hands = vec![
            vec![card(Rank::Eight, Suit::Clubs)],
            vec![card(Rank::Seven, Suit::Clubs)],
            vec![card(Rank::Nine, Suit::Clubs)],
        ];
        let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
        round
            .submit_move(0, Move::Play(combo(vec![card(Rank::Eight, Suit::Clubs)])))
            .unwrap();
        let play = Move::Play(combo(vec![card(Rank::Seven, Suit::Clubs)]));
        assert_eq!(round.submit_move(1, play), Err(MoveError::ComboDoesNotBeat));
    }

    #[test]
    fn everyone_passing_returns_the_lead_to_the_original_leader() {
        let hands = vec![
            vec![card(Rank::Eight, Suit::Clubs), card(Rank::Ten, Suit::Clubs)],
            vec![card(Rank::Seven, Suit::Clubs), card(Rank::Six, Suit::Clubs)],
            vec![card(Rank::Nine, Suit::Clubs), card(Rank::Five, Suit::Clubs)],
        ];
        let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
        round
            .submit_move(0, Move::Play(combo(vec![card(Rank::Eight, Suit::Clubs)])))
            .unwrap();
        round.submit_move(1, Move::Pass).unwrap();
        round.submit_move(2, Move::Pass).unwrap();
        assert_eq!(round.seat_to_move(), Some(0));
        assert_eq!(round.current_combo(), None);
        assert_eq!(round.current_trick_leader(), 0);
    }

    #[test]
    fn a_seat_emptying_its_hand_mid_round_is_recorded_and_skipped() {
        // 3 players, each starts with 1 card so seat 0 empties
        // immediately on its lead.
        let hands = vec![
            vec![card(Rank::Two, Suit::Clubs)],
            vec![card(Rank::Three, Suit::Clubs)],
            vec![card(Rank::Four, Suit::Clubs)],
        ];
        let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
        round
            .submit_move(0, Move::Play(combo(vec![card(Rank::Two, Suit::Clubs)])))
            .unwrap();
        assert_eq!(round.finishing_order(), &[0]);
        assert!(!round.is_active(0));
        assert_eq!(round.seat_to_move(), Some(1));

        round
            .submit_move(1, Move::Play(combo(vec![card(Rank::Three, Suit::Clubs)])))
            .unwrap();
        // Only seat 2 remains active: the round auto-completes.
        assert!(round.is_complete());
        assert_eq!(round.finishing_order(), &[0, 1, 2]);
        assert_eq!(round.seat_to_move(), None);
    }
}
```

- [ ] **Step 2: Wire the module into `engine/src/lib.rs`**

```rust
//! Pure Arschloch game rules. See docs/RULES.md for the authoritative
//! ruleset and docs/ARCHITECTURE.md for how this crate fits into the
//! workspace.

pub mod card;
pub mod combo;
pub mod deal;
pub mod exchange;
pub mod role;
pub mod round;
mod trick;

pub use card::{Card, DeckVariant, DuplicateRule, Rank, Suit};
pub use combo::Combo;
pub use deal::{deal, lowest_card_holder};
pub use exchange::{exchange, ExchangeError};
pub use role::{assign_roles, exchange_counts_for_player_count, roles_for_player_count, Role};
pub use round::{Move, MoveError, Round};

/// A seat's position at the table (0-indexed). Table sizes are 3-6, so
/// `u8` matches `player_count`'s type used throughout this crate.
pub type SeatId = u8;
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p engine`
Expected: all 54 tests pass (45 from Task 5 + 9 new).

- [ ] **Step 4: Commit**

```bash
git add engine/src/round.rs engine/src/lib.rs
git commit -m "engine: add Round trick-taking state machine"
```

---

### Task 7: `engine/tests/full_round.rs`

**Files:**
- Create: `engine/tests/full_round.rs`

**Interfaces:**
- Consumes: the crate's full public API as re-exported from
  `engine/src/lib.rs` (`deal`, `lowest_card_holder`, `assign_roles`,
  `exchange`, `Round`, `Move`, `Combo`, `Card`, `Rank`, `Suit`,
  `DuplicateRule`, `Role`) — this is a separate integration-test crate,
  so it can only see `pub` items, which is exactly why `trick` stays
  crate-private and untested from here (it's fully covered by its own
  inline unit tests in Task 5).

This is the plan's only task without a "wire into lib.rs" step — nothing
in `src/` changes.

- [ ] **Step 1: Write `engine/tests/full_round.rs`**

```rust
//! Scripted, fully deterministic integration tests exercising the whole
//! Phase 1 pipeline together: deal -> round play -> finishing order ->
//! role assignment -> exchange for the next round. See docs/RULES.md.

use engine::{
    assign_roles, deal, exchange, lowest_card_holder, Card, Combo, DuplicateRule, Move, Rank,
    Role, Round, Suit,
};

fn clubs(rank: Rank) -> Card {
    Card::new(rank, Suit::Clubs, 0)
}

fn diamonds(rank: Rank) -> Card {
    Card::new(rank, Suit::Diamonds, 0)
}

fn single(card: Card) -> Combo {
    Combo::new(vec![card]).unwrap()
}

#[test]
fn four_player_round_deal_through_exchange() {
    // --- Deal ---
    let deck = vec![
        clubs(Rank::Two),
        clubs(Rank::Three),
        clubs(Rank::Four),
        clubs(Rank::Five),
        clubs(Rank::Six),
        clubs(Rank::Seven),
        clubs(Rank::Eight),
        clubs(Rank::Nine),
    ];
    let hands = deal(deck, 4).expect("8 cards deal evenly into 4 hands of 2");
    assert_eq!(hands[0], vec![clubs(Rank::Two), clubs(Rank::Six)]);
    assert_eq!(hands[1], vec![clubs(Rank::Three), clubs(Rank::Seven)]);
    assert_eq!(hands[2], vec![clubs(Rank::Four), clubs(Rank::Eight)]);
    assert_eq!(hands[3], vec![clubs(Rank::Five), clubs(Rank::Nine)]);

    let leader =
        lowest_card_holder(&hands, DuplicateRule::FirstDealtWins).expect("every hand has cards");
    assert_eq!(leader, 0, "seat 0 holds the globally lowest card (Two of Clubs)");

    let mut round =
        Round::new(hands, DuplicateRule::FirstDealtWins, leader).expect("4 hands, valid leader, no empty hands");

    // --- Trick 1: an escalating single-card war, then everyone passes
    //     even though seat 0 could beat seat 3's Five with its own Six
    //     (passing is always legal, not just a last resort) ---
    round.submit_move(0, Move::Play(single(clubs(Rank::Two)))).unwrap();
    round.submit_move(1, Move::Play(single(clubs(Rank::Three)))).unwrap();
    round.submit_move(2, Move::Play(single(clubs(Rank::Four)))).unwrap();
    round.submit_move(3, Move::Play(single(clubs(Rank::Five)))).unwrap();
    round.submit_move(0, Move::Pass).unwrap(); // voluntary pass, holds a beating Six
    round.submit_move(1, Move::Pass).unwrap();
    round.submit_move(2, Move::Pass).unwrap();
    assert_eq!(round.seat_to_move(), Some(3), "seat 3 won trick 1 and leads trick 2");

    // --- Trick 2: seat 3 leads its last card and empties its hand while
    //     leading. Per the confirmed house rule, leadership then passes
    //     to the next active seat in turn order (seat 0), not staying
    //     with the now-finished seat 3. ---
    round.submit_move(3, Move::Play(single(clubs(Rank::Nine)))).unwrap();
    round.submit_move(0, Move::Pass).unwrap();
    round.submit_move(1, Move::Pass).unwrap();
    round.submit_move(2, Move::Pass).unwrap();
    assert_eq!(round.finishing_order(), &[3]);
    assert_eq!(
        round.seat_to_move(),
        Some(0),
        "seat 3 emptied its hand leading trick 2; seat 0 (next active) leads trick 3"
    );

    // --- Trick 3: seat 0 leads its last card and empties its hand ---
    round.submit_move(0, Move::Play(single(clubs(Rank::Six)))).unwrap();
    round.submit_move(1, Move::Pass).unwrap();
    round.submit_move(2, Move::Pass).unwrap();
    assert_eq!(round.finishing_order(), &[3, 0]);
    assert_eq!(round.seat_to_move(), Some(1));

    // --- Trick 4: seat 1 leads its last card; only seat 2 remains, so
    //     the round completes immediately without needing seat 2 to
    //     explicitly pass on an unbeatable card. ---
    round.submit_move(1, Move::Play(single(clubs(Rank::Seven)))).unwrap();
    assert!(round.is_complete());
    assert_eq!(round.finishing_order(), &[3, 0, 1, 2]);
    assert_eq!(round.seat_to_move(), None);

    // --- Role assignment ---
    let role_by_seat = assign_roles(round.finishing_order(), 4).expect("a complete 4-player finishing order");
    assert_eq!(role_by_seat[3], Role::President);
    assert_eq!(role_by_seat[0], Role::Vize);
    assert_eq!(role_by_seat[1], Role::ViceArschloch);
    assert_eq!(role_by_seat[2], Role::Arschloch);

    // --- Exchange for the next round, on a fresh (unrelated) deal ---
    let mut next_hands = vec![
        vec![diamonds(Rank::Ten), diamonds(Rank::Jack), diamonds(Rank::Queen)], // seat 0, Vize
        vec![diamonds(Rank::Two), diamonds(Rank::Three), diamonds(Rank::Four)], // seat 1, ViceArschloch
        vec![diamonds(Rank::Five), diamonds(Rank::Six), diamonds(Rank::Seven)], // seat 2, Arschloch
        vec![diamonds(Rank::Eight), diamonds(Rank::Nine), diamonds(Rank::King)], // seat 3, President
    ];
    exchange(&mut next_hands, &role_by_seat, DuplicateRule::FirstDealtWins).expect("valid 4-player exchange");

    // President (seat 3) <-> Arschloch (seat 2) swap 2 cards; Vize (seat
    // 0) <-> ViceArschloch (seat 1) swap 1 card.
    assert_eq!(
        next_hands[0],
        vec![diamonds(Rank::Jack), diamonds(Rank::Queen), diamonds(Rank::Four)]
    );
    assert_eq!(
        next_hands[1],
        vec![diamonds(Rank::Two), diamonds(Rank::Three), diamonds(Rank::Ten)]
    );
    assert_eq!(
        next_hands[2],
        vec![diamonds(Rank::Five), diamonds(Rank::Eight), diamonds(Rank::Nine)]
    );
    assert_eq!(
        next_hands[3],
        vec![diamonds(Rank::King), diamonds(Rank::Six), diamonds(Rank::Seven)]
    );
}
```

- [ ] **Step 2: Run the tests**

Run: `cargo test -p engine`
Expected: all 55 tests pass (54 unit tests from Task 6 + 1 new
integration test — `cargo test` reports the integration test as a
separate binary, e.g. `Running tests/full_round.rs`).

- [ ] **Step 3: Commit**

```bash
git add engine/tests/full_round.rs
git commit -m "engine: add scripted full-round integration test"
```

---

### Task 8: Phase-done verification gate

**Files:** none (verification only).

**Interfaces:** none — runs the gate from `docs/CODING_GUIDELINES.md`,
"Before finishing a phase", across the whole workspace, and fixes
anything it finds.

- [ ] **Step 1: Check formatting**

Run: `cargo fmt --check`
Expected: no output. If it reports diffs, run `cargo fmt` and re-check.

- [ ] **Step 2: Run clippy with pedantic-as-warn promoted to deny**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: no warnings or errors. If clippy reports something, fix the
underlying code; only add a targeted `#[allow(clippy::lint_name)]` with a
one-line reason comment if the lint genuinely doesn't apply.

- [ ] **Step 3: Run the full test suite**

Run: `cargo test --workspace`
Expected: 55 tests pass, 0 failed (across `engine`; `sim`/`cli`/`web`
have none yet).

- [ ] **Step 4: Commit any fixes from Steps 1-3**

Only if Steps 1-3 required changes:

```bash
git add -A
git commit -m "engine: fix fmt/clippy issues from Phase 1 verification gate"
```

If no changes were needed, skip this step — there's nothing to commit.
