# Phase 5 — Smart Exchange Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `engine::exchange`'s naive top-N card selection with
strategy-aware selection for the giving-away-your-best-cards half of the
mandatory exchange, per `docs/ROADMAP.md`'s Phase 5 entry.

**Architecture:** Add a lower-level `engine::exchange_with_selection`
that takes a per-seat card-selection callback instead of always sorting
by `Card::compare`; reimplement the existing `engine::exchange` as a
thin wrapper supplying the old naive callback (so its behavior and
tests are unchanged). Add `Strategy::choose_exchange_cards` in `sim`,
implement it for all four existing strategies (three replicate the
naive behavior, `HoldBackPairs` gets genuine pair/triple-aware logic),
and wire `sim::match_runner::run_match` to call strategies through the
new `engine` entry point.

**Tech Stack:** Rust workspace (`engine`, `sim`, `cli`), `rand` (already
a dependency), `serde`/`serde_json` (unaffected — no schema changes this
phase).

**Spec:** `docs/ROADMAP.md`'s Phase 5 entry ("Smart exchange"); design
rationale in the approved native-plan-mode doc this plan transcribes
(not saved as a separate spec file — this plan *is* the spec for this
phase, same as Phase 4's).

## Global Constraints

- No new Cargo dependencies — everything needed (`rand::seq::SliceRandom`,
  `std::collections`) is already in use elsewhere in these crates.
- `engine` stays dependency-free of `sim`/`cli` — the new
  `exchange_with_selection` takes a generic `impl FnMut(...)` callback,
  never a `sim::Strategy` reference.
- `engine::exchange`'s existing public behavior and every one of its
  current tests must be unchanged byte-for-byte — it becomes a thin
  wrapper around `exchange_with_selection`, not a rewrite.
- Tasks 1-2 are verified with `cargo test -p engine` / `cargo test -p
  sim` respectively (plus `cargo clippy -p <crate> --all-targets -- -D
  warnings`); Tasks 3-4 touch cross-crate call sites and use the full
  workspace gate (`cargo fmt --check`, `cargo clippy --workspace
  --all-targets -- -D warnings`, `cargo test --workspace`).
- Exact field/type names in code blocks below (`ExchangeError`,
  `Strategy`, `BatchConfig`, etc.) must match what's actually in the
  crate at the time each task runs — an implementer hits a real
  mismatch by reading the current file first, not by guessing from this
  plan.

## Review Focus

- **Multi-pair table sizes (5-6 players).** The plan/apply split in
  `exchange_with_selection` must correctly validate and apply *every*
  exchanging low seat's selection, not just the single pair a 3- or
  4-player table exercises. Task 1's tests must include a 5- or
  6-player scenario.
- **Partial-mutation safety.** An invalid selection from one low seat
  must leave *all* hands untouched, including seats whose own selection
  was valid and already validated earlier in the plan pass. Task 1's
  `..._leaves_hands_untouched_when_selection_invalid` test must
  specifically arrange a valid-then-invalid ordering to catch a
  validate-as-you-go regression that only invalidates lazily.
- **`HoldBackPairs` forced to break more than one group.** If `count`
  exceeds the number of isolated singles in the hand, the strategy must
  still return exactly `count` cards, even if that means partially
  breaking two different same-rank groups (not just one). Task 2's
  tests only exercise breaking a single pair — add a case with two
  small groups and no isolated singles at all if the hand-construction
  cost is low, otherwise ledger this as a known gap rather than silently
  skip it.
- **`RandomLegal` at the boundary `count == hand.len()`.** The whole
  hand must come back with no duplicates and no panic when every card
  is given away. Task 2's test should include this boundary case.
- **RNG threading through the exchange closure.** The closure captured
  by `match_runner::run_match` borrows `&mut rng` — verify this doesn't
  fight the borrow checker against `rng`'s other uses in the same
  function, and that results stay deterministic for a fixed seed
  (`docs/BUILDING.md`'s reproducibility claim). Task 3 must not weaken
  or change the seeding/determinism behavior `run_match`'s existing
  tests already pin.

---

### Task 1: `engine` — expose `rank_groups`, add `exchange_with_selection`

**Files:**
- Modify: `engine/src/card.rs`
- Modify: `engine/src/legal_moves.rs`
- Modify: `engine/src/lib.rs`
- Modify: `engine/src/exchange.rs`
- Test: inline `#[cfg(test)]` modules in `card.rs` and `exchange.rs`

**Interfaces:**
- Produces: `pub fn engine::rank_groups(hand: &[Card]) -> Vec<Vec<Card>>`
  (re-exported at crate root), `pub fn
  engine::exchange_with_selection(hands: &mut [Vec<Card>], role_by_seat:
  &[Role], duplicate_rule: DuplicateRule, choose_cards_to_give: impl
  FnMut(usize, &[Card], usize, DuplicateRule) -> Vec<Card>) ->
  Result<(), ExchangeError>`, and a new `ExchangeError::InvalidSelection`
  variant. `engine::exchange`'s existing signature and behavior are
  unchanged.
- Consumes: nothing new — reuses `Card::compare`, `Role`,
  `roles_for_player_count`, `exchange_counts_for_player_count`, all
  already in `engine`.

- [ ] **Step 1: Read the current files before editing**

Read `engine/src/card.rs`, `engine/src/legal_moves.rs`,
`engine/src/lib.rs`, and `engine/src/exchange.rs` in full first. The
code blocks below assume `legal_moves.rs` currently has a private
`rank_groups` helper and `exchange.rs` currently has `exchange`,
`take_highest`, `take_lowest`, `ExchangeError`, and
`validate_role_mapping` — confirm the exact current shape before
changing anything, since a mismatch here (e.g. a different existing
helper name) means adapting the steps below to what's actually present
rather than guessing.

- [ ] **Step 2: Move `rank_groups` to `card.rs` and make it public**

In `engine/src/card.rs`, add:

```rust
/// Groups `hand` by rank, in ascending rank order. Each inner `Vec` is
/// every card of one rank (a "reserve" a strategy might want to keep
/// together rather than split up — see `docs/ROADMAP.md`, Phase 5).
#[must_use]
pub fn rank_groups(hand: &[Card]) -> Vec<Vec<Card>> {
    let mut sorted = hand.to_vec();
    sorted.sort_by_key(|card| card.rank);
    let mut groups: Vec<Vec<Card>> = Vec::new();
    for card in sorted {
        match groups.last_mut() {
            Some(group) if group[0].rank == card.rank => group.push(card),
            _ => groups.push(vec![card]),
        }
    }
    groups
}

#[cfg(test)]
mod rank_groups_tests {
    use super::*;

    #[test]
    fn empty_hand_has_no_groups() {
        assert_eq!(rank_groups(&[]), Vec::<Vec<Card>>::new());
    }

    #[test]
    fn groups_same_rank_cards_together_in_ascending_rank_order() {
        let low_a = Card::new(Rank::Two, Suit::Clubs, 0);
        let low_b = Card::new(Rank::Two, Suit::Hearts, 1);
        let mid = Card::new(Rank::Five, Suit::Clubs, 2);
        let high = Card::new(Rank::King, Suit::Clubs, 3);
        let hand = vec![high.clone(), low_a.clone(), mid.clone(), low_b.clone()];

        let groups = rank_groups(&hand);

        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].len(), 2);
        assert!(groups[0].contains(&low_a));
        assert!(groups[0].contains(&low_b));
        assert_eq!(groups[1], vec![mid]);
        assert_eq!(groups[2], vec![high]);
    }
}
```

Adjust the test module's `use super::*;` and constructor calls
(`Card::new`, `Rank::`, `Suit::`) to match whatever `card.rs` already
uses in its own existing test module — copy that module's exact
construction idiom rather than inventing a new one, and place these
tests in that same existing `#[cfg(test)]` block instead of a second
one if `card.rs` already has one (avoid two competing test modules in
one file).

Remove `rank_groups` from `engine/src/legal_moves.rs` and replace its
call site with `crate::card::rank_groups(hand)` (or `use
crate::card::rank_groups;` at the top and call it unqualified — match
this file's existing import style). Do not change any of
`legal_moves.rs`'s own test assertions; they exercise behavior, not the
helper directly, so they should keep passing unchanged.

In `engine/src/lib.rs`, find the existing `pub use card::{...};`
re-export line and add `rank_groups` to it.

- [ ] **Step 3: Run engine tests to confirm the move compiles and passes**

Run: `cargo test -p engine`
Expected: all existing tests pass, plus the two new `rank_groups` tests.

- [ ] **Step 4: Add `ExchangeError::InvalidSelection`**

In `engine/src/exchange.rs`, add a variant to the existing
`ExchangeError` enum:

```rust
    /// A `choose_cards_to_give` callback (see `exchange_with_selection`)
    /// returned something other than exactly `count` distinct cards
    /// drawn from the hand it was given.
    InvalidSelection,
```

- [ ] **Step 5: Write the new `exchange_with_selection` tests first**

Add to `engine/src/exchange.rs`'s existing `#[cfg(test)] mod tests`
(reuse whatever hand-construction helper — e.g. a `card(rank, suit)`
closure/function — that module already defines; do not redefine one):

```rust
#[test]
fn exchange_with_selection_uses_the_provided_cards() {
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

    // Opposite of naive: always give the LOWEST `count` cards instead
    // of the highest.
    exchange_with_selection(
        &mut hands,
        &role_by_seat,
        DuplicateRule::FirstDealtWins,
        |_seat, hand, count, duplicate_rule| {
            let mut sorted = hand.to_vec();
            sorted.sort_by(|a, b| a.compare(b, duplicate_rule));
            sorted.truncate(count);
            sorted
        },
    )
    .unwrap();

    // Seat 3 (Arschloch, giving 1 card up to President) gave its LOWEST
    // card (Two) instead of the naive highest (Ace).
    assert!(hands[0].contains(&card(Rank::Two, Suit::Clubs)));
    assert!(hands[3].contains(&card(Rank::Ace, Suit::Clubs)) || hands[3].contains(&card(Rank::King, Suit::Clubs)));
}

#[test]
fn exchange_with_selection_rejects_wrong_count() {
    let role_by_seat = vec![Role::President, Role::Dorftrottel, Role::Arschloch];
    let mut hands = vec![
        vec![card(Rank::Two, Suit::Clubs)],
        vec![card(Rank::Five, Suit::Clubs)],
        vec![card(Rank::King, Suit::Clubs)],
    ];
    let result = exchange_with_selection(
        &mut hands,
        &role_by_seat,
        DuplicateRule::FirstDealtWins,
        |_seat, _hand, _count, _duplicate_rule| Vec::new(),
    );
    assert_eq!(result, Err(ExchangeError::InvalidSelection));
}

#[test]
fn exchange_with_selection_rejects_card_not_in_hand() {
    let role_by_seat = vec![Role::President, Role::Dorftrottel, Role::Arschloch];
    let mut hands = vec![
        vec![card(Rank::Two, Suit::Clubs)],
        vec![card(Rank::Five, Suit::Clubs)],
        vec![card(Rank::King, Suit::Clubs)],
    ];
    let result = exchange_with_selection(
        &mut hands,
        &role_by_seat,
        DuplicateRule::FirstDealtWins,
        |_seat, _hand, _count, _duplicate_rule| vec![card(Rank::Nine, Suit::Diamonds)],
    );
    assert_eq!(result, Err(ExchangeError::InvalidSelection));
}

#[test]
fn exchange_with_selection_rejects_duplicate_selection() {
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
    let result = exchange_with_selection(
        &mut hands,
        &role_by_seat,
        DuplicateRule::FirstDealtWins,
        |_seat, hand, count, _duplicate_rule| {
            // Return the same card `count` times instead of `count`
            // distinct cards.
            vec![hand[0].clone(); count]
        },
    );
    assert_eq!(result, Err(ExchangeError::InvalidSelection));
}

#[test]
fn exchange_with_selection_leaves_hands_untouched_when_selection_invalid() {
    // 6 players -> 3 exchanging pairs. Seat order matches
    // roles_for_player_count(6): President, Vize, Offizier, Dummkopf,
    // ViceArschloch, Arschloch.
    let role_by_seat = vec![
        Role::President,
        Role::Vize,
        Role::Offizier,
        Role::Dummkopf,
        Role::ViceArschloch,
        Role::Arschloch,
    ];
    let original_hands = vec![
        vec![card(Rank::Two, Suit::Clubs), card(Rank::Three, Suit::Clubs), card(Rank::Four, Suit::Clubs)],
        vec![card(Rank::Five, Suit::Clubs), card(Rank::Six, Suit::Clubs)],
        vec![card(Rank::Seven, Suit::Clubs), card(Rank::Eight, Suit::Clubs)],
        vec![card(Rank::Nine, Suit::Clubs), card(Rank::Ten, Suit::Clubs)],
        vec![card(Rank::Jack, Suit::Clubs), card(Rank::Queen, Suit::Clubs)],
        vec![card(Rank::King, Suit::Clubs), card(Rank::Ace, Suit::Clubs), card(Rank::Two, Suit::Hearts)],
    ];
    let mut hands = original_hands.clone();

    // Valid for the first (outermost) pair's low seat (seat 5,
    // Arschloch, giving 3 cards to President), invalid for the second
    // pair's low seat (seat 4, ViceArschloch, giving to Vize) — proves
    // an earlier valid selection doesn't get applied before a later
    // one is found invalid.
    let result = exchange_with_selection(
        &mut hands,
        &role_by_seat,
        DuplicateRule::FirstDealtWins,
        |seat, hand, count, duplicate_rule| {
            if seat == 4 {
                Vec::new() // wrong count -> InvalidSelection
            } else {
                let mut sorted = hand.to_vec();
                sorted.sort_by(|a, b| a.compare(b, duplicate_rule));
                sorted.split_off(sorted.len() - count)
            }
        },
    );

    assert_eq!(result, Err(ExchangeError::InvalidSelection));
    assert_eq!(hands, original_hands);
}
```

Double-check the exact exchange counts for 6 players
(`exchange_counts_for_player_count(6)`) and adjust hand sizes above if
the real table gives different counts than assumed here — read
`engine/src/role.rs` to confirm before relying on the numbers in this
step.

- [ ] **Step 6: Run tests to verify they fail to compile**

Run: `cargo test -p engine`
Expected: compile error — `exchange_with_selection` and
`ExchangeError::InvalidSelection` (as a variant, if not yet added in
Step 4) don't exist yet.

- [ ] **Step 7: Implement `exchange_with_selection` and rewrite `exchange`**

Replace `exchange.rs`'s existing `exchange` function and `take_highest`
helper with:

```rust
/// Like `exchange`, but the lower role's mandatory best-`count` cards
/// are chosen by `choose_cards_to_give` instead of a naive highest-N
/// sort (`docs/ROADMAP.md`, Phase 5, "Smart exchange"). The higher
/// role's worst-`count` cards it gives back are still chosen naively —
/// Phase 5 only makes the giving-away-your-best-cards side
/// strategy-aware.
///
/// Validated before anything is mutated (same failure atomicity as
/// `exchange`): `choose_cards_to_give(seat, hand, count,
/// duplicate_rule)` is called once per exchanging low seat with that
/// seat's untouched hand, and must return exactly `count` distinct
/// cards each present in `hand` — anything else is
/// `ExchangeError::InvalidSelection` and no hand is changed.
pub fn exchange_with_selection(
    hands: &mut [Vec<Card>],
    role_by_seat: &[Role],
    duplicate_rule: DuplicateRule,
    mut choose_cards_to_give: impl FnMut(usize, &[Card], usize, DuplicateRule) -> Vec<Card>,
) -> Result<(), ExchangeError> {
    if hands.len() != role_by_seat.len() {
        return Err(ExchangeError::SeatCountMismatch);
    }
    let player_count =
        u8::try_from(hands.len()).map_err(|_| ExchangeError::UnsupportedPlayerCount)?;
    let roles =
        roles_for_player_count(player_count).ok_or(ExchangeError::UnsupportedPlayerCount)?;
    let counts = exchange_counts_for_player_count(player_count)
        .ok_or(ExchangeError::UnsupportedPlayerCount)?;
    validate_role_mapping(role_by_seat, roles)?;

    let seat_of = |role: Role| -> usize {
        role_by_seat
            .iter()
            .position(|&r| r == role)
            .expect("validate_role_mapping already confirmed every role is present exactly once")
    };

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

    // Plan pass: ask each low seat what it's giving and validate the
    // answer against its still-untouched hand, before mutating
    // anything.
    let mut planned_gives: Vec<Option<Vec<Card>>> = vec![None; hands.len()];
    for (i, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let count = usize::from(count);
        let low_seat = seat_of(roles[roles.len() - 1 - i]);
        let selected = choose_cards_to_give(low_seat, &hands[low_seat], count, duplicate_rule);
        if selected.len() != count || !is_sub_multiset(&selected, &hands[low_seat]) {
            return Err(ExchangeError::InvalidSelection);
        }
        planned_gives[low_seat] = Some(selected);
    }

    // Apply pass: identical shape to the old `exchange`, except the low
    // seat's outgoing cards come from `planned_gives` instead of
    // `take_highest`.
    let mut incoming: Vec<Vec<Card>> = vec![Vec::new(); hands.len()];
    for (i, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let count = usize::from(count);
        let high_seat = seat_of(roles[i]);
        let low_seat = seat_of(roles[roles.len() - 1 - i]);

        let selected = planned_gives[low_seat]
            .take()
            .expect("every exchanging low seat was planned above");
        let from_low = remove_selected(&mut hands[low_seat], &selected);
        let from_high = take_lowest(&mut hands[high_seat], count, duplicate_rule);

        incoming[high_seat].extend(from_low);
        incoming[low_seat].extend(from_high);
    }

    for (hand, extra) in hands.iter_mut().zip(incoming) {
        hand.extend(extra);
    }
    Ok(())
}

/// The naive top/bottom-N exchange from Phase 1, now implemented as
/// `exchange_with_selection` with a selection callback that reproduces
/// the old `take_highest` behavior exactly.
pub fn exchange(
    hands: &mut [Vec<Card>],
    role_by_seat: &[Role],
    duplicate_rule: DuplicateRule,
) -> Result<(), ExchangeError> {
    exchange_with_selection(
        hands,
        role_by_seat,
        duplicate_rule,
        |_seat, hand, count, duplicate_rule| {
            let mut sorted = hand.to_vec();
            sorted.sort_by(|a, b| a.compare(b, duplicate_rule));
            sorted.split_off(sorted.len() - count)
        },
    )
}

fn is_sub_multiset(selected: &[Card], hand: &[Card]) -> bool {
    let mut remaining = hand.to_vec();
    for card in selected {
        match remaining.iter().position(|c| c == card) {
            Some(pos) => {
                remaining.remove(pos);
            }
            None => return false,
        }
    }
    true
}

fn remove_selected(hand: &mut Vec<Card>, selected: &[Card]) -> Vec<Card> {
    selected
        .iter()
        .map(|card| {
            let pos = hand
                .iter()
                .position(|c| c == card)
                .expect("is_sub_multiset already validated this selection");
            hand.remove(pos)
        })
        .collect()
}
```

Keep `take_lowest` exactly as it currently is. Add `rank_groups` is NOT
needed in this file (that's `sim`'s job in Task 2) — `exchange.rs`
itself never calls it.

- [ ] **Step 8: Run tests to verify they pass**

Run: `cargo test -p engine`
Expected: PASS — every existing `exchange(...)` test unchanged and
still green (proving the naive path is behavior-preserving), plus all
five new `exchange_with_selection` tests green.

- [ ] **Step 9: Lint and format**

Run: `cargo fmt -p engine` then `cargo clippy -p engine --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 10: Commit**

```bash
git add engine/src/card.rs engine/src/legal_moves.rs engine/src/lib.rs engine/src/exchange.rs
git commit -m "engine: add exchange_with_selection and public rank_groups"
```

---

### Task 2: `sim::Strategy` — add `choose_exchange_cards` for all four strategies

**Files:**
- Modify: `sim/src/strategy.rs`
- Modify: `sim/src/strategies/mod.rs`
- Modify: `sim/src/strategies/lowest_legal.rs`
- Modify: `sim/src/strategies/greedy_highest.rs`
- Modify: `sim/src/strategies/random_legal.rs`
- Modify: `sim/src/strategies/hold_back_pairs.rs`

**Interfaces:**
- Consumes: `engine::{Card, DuplicateRule, rank_groups}` (Task 1).
- Produces: `Strategy::choose_exchange_cards(&self, hand: &[Card],
  count: usize, duplicate_rule: DuplicateRule, rng: &mut dyn
  rand::Rng) -> Vec<Card>` — a new required trait method every
  `Strategy` implementor must have. Task 3 (`match_runner`) calls this
  per seat.

- [ ] **Step 1: Read the current files before editing**

Read all six files listed above in full. Confirm each strategy file's
current `impl Strategy for ...` block and its existing imports, so the
additions below merge cleanly with what's actually there rather than
assuming an exact byte-for-byte match to this plan's quoted history.

- [ ] **Step 2: Add the trait method (this will break the build — expected)**

In `sim/src/strategy.rs`, add `Card` to the `use engine::{...};` import
and add the new method to the `Strategy` trait:

```rust
    /// Chooses which `count` cards to give up when this seat holds a
    /// role required to hand over its best cards during the exchange
    /// (`docs/ROADMAP.md`, Phase 5, "Smart exchange"). Must return
    /// exactly `count` distinct cards, each present in `hand`;
    /// `engine::exchange_with_selection` treats anything else as a bug
    /// (`ExchangeError::InvalidSelection`).
    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        duplicate_rule: DuplicateRule,
        rng: &mut dyn rand::Rng,
    ) -> Vec<Card>;
```

- [ ] **Step 3: Run tests to confirm the expected build failure**

Run: `cargo test -p sim`
Expected: compile errors in every `strategies/*.rs` file — each
`impl Strategy for ...` is now missing a required method. This
confirms the trait change took effect; proceed to implement each one.

- [ ] **Step 4: Add the shared naive-selection helper**

In `sim/src/strategies/mod.rs`, add `use engine::{Card, DuplicateRule};`
to its imports (merge with whatever's already imported there) and add:

```rust
/// The naive "give up your highest `count` cards" behavior from Phase 1
/// (`engine::exchange`'s old default), reused by strategies that have no
/// stronger opinion about which cards to give up.
pub(crate) fn take_highest_naive(
    hand: &[Card],
    count: usize,
    duplicate_rule: DuplicateRule,
) -> Vec<Card> {
    let mut sorted = hand.to_vec();
    sorted.sort_by(|a, b| a.compare(b, duplicate_rule));
    sorted.split_off(sorted.len() - count)
}
```

- [ ] **Step 5: Implement `choose_exchange_cards` for `LowestLegal` and `GreedyHighest`**

Add this method to both `sim/src/strategies/lowest_legal.rs` and
`sim/src/strategies/greedy_highest.rs`'s `impl Strategy for ...` block
(identical in both files):

```rust
    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        duplicate_rule: DuplicateRule,
        _rng: &mut dyn rand::Rng,
    ) -> Vec<Card> {
        crate::strategies::take_highest_naive(hand, count, duplicate_rule)
    }
```

Add `Card` to each file's `use engine::{...};` import if not already
present.

Add one test to each file (same fixture and assertion in both,
consistent with how both files already share the same test style for
`choose_play`):

```rust
#[test]
fn choose_exchange_cards_gives_up_the_highest_cards() {
    let strategy = LowestLegal; // or GreedyHighest, matching the file
    let hand = vec![
        card(Rank::Two, Suit::Clubs),
        card(Rank::Five, Suit::Clubs),
        card(Rank::Seven, Suit::Clubs),
        card(Rank::Jack, Suit::Clubs),
        card(Rank::Ace, Suit::Clubs),
    ];
    let mut rng = rand::rngs::mock::StepRng::new(0, 1);
    let given = strategy.choose_exchange_cards(&hand, 2, DuplicateRule::FirstDealtWins, &mut rng);
    assert_eq!(given.len(), 2);
    assert!(given.contains(&card(Rank::Ace, Suit::Clubs)));
    assert!(given.contains(&card(Rank::Jack, Suit::Clubs)));
}
```

Reuse whatever `card(rank, suit)` helper and RNG-construction idiom
(`StepRng`, or a real `rand::rngs::StdRng::seed_from_u64`) this file's
existing `choose_play` tests already use — do not invent a second
helper if one already exists in the file. If `rand::rngs::mock::StepRng`
isn't available (check `Cargo.toml`'s `rand` feature flags), use
`rand::rngs::StdRng::seed_from_u64(0)` instead, matching whatever
`random_legal.rs`'s own existing tests already do for `choose_play`.

- [ ] **Step 6: Implement `choose_exchange_cards` for `RandomLegal`**

Add to `sim/src/strategies/random_legal.rs`'s `impl Strategy for
RandomLegal` (this file already imports `rand::seq::SliceRandom` for
`choose_play`'s `.choose(rng)` — reuse that import):

```rust
    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        _duplicate_rule: DuplicateRule,
        rng: &mut dyn rand::Rng,
    ) -> Vec<Card> {
        let mut indices: Vec<usize> = (0..hand.len()).collect();
        indices.shuffle(rng);
        indices.truncate(count);
        indices.into_iter().map(|i| hand[i].clone()).collect()
    }
```

Add tests:

```rust
#[test]
fn choose_exchange_cards_returns_exactly_count_distinct_cards_from_hand() {
    let strategy = RandomLegal;
    let hand = vec![
        card(Rank::Two, Suit::Clubs),
        card(Rank::Five, Suit::Clubs),
        card(Rank::Seven, Suit::Clubs),
        card(Rank::Jack, Suit::Clubs),
        card(Rank::Ace, Suit::Clubs),
    ];
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let given = strategy.choose_exchange_cards(&hand, 2, DuplicateRule::FirstDealtWins, &mut rng);
    assert_eq!(given.len(), 2);
    assert_ne!(given[0], given[1]);
    for card in &given {
        assert!(hand.contains(card));
    }
}

#[test]
fn choose_exchange_cards_can_give_up_the_entire_hand() {
    let strategy = RandomLegal;
    let hand = vec![card(Rank::Two, Suit::Clubs), card(Rank::Five, Suit::Clubs)];
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    let given = strategy.choose_exchange_cards(&hand, hand.len(), DuplicateRule::FirstDealtWins, &mut rng);
    assert_eq!(given.len(), hand.len());
    assert_ne!(given[0], given[1]);
}
```

Match this file's existing `use rand::...;` and RNG-construction idiom
exactly — add `use rand::SeedableRng;` only if not already imported.

- [ ] **Step 7: Implement `choose_exchange_cards` for `HoldBackPairs`**

Add to `sim/src/strategies/hold_back_pairs.rs`'s `impl Strategy for
HoldBackPairs`:

```rust
    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        duplicate_rule: DuplicateRule,
        _rng: &mut dyn rand::Rng,
    ) -> Vec<Card> {
        // Give up whole isolated cards before touching any same-rank
        // group of 2+, smallest groups first; within a length class,
        // prefer the highest-ranked group (still shed value
        // preferentially, just never break a reserve while an isolated
        // card remains).
        let mut groups = engine::rank_groups(hand);
        groups.sort_by(|a, b| {
            a.len()
                .cmp(&b.len())
                .then_with(|| b[0].compare(&a[0], duplicate_rule))
        });

        let mut selected = Vec::with_capacity(count);
        for mut group in groups {
            if selected.len() >= count {
                break;
            }
            let take = (count - selected.len()).min(group.len());
            group.sort_by(|a, b| b.compare(a, duplicate_rule));
            selected.extend(group.into_iter().take(take));
        }
        selected
    }
```

Add tests:

```rust
#[test]
fn choose_exchange_cards_prefers_isolated_cards_over_a_pair() {
    let strategy = HoldBackPairs;
    let hand = vec![
        card(Rank::Five, Suit::Clubs),
        card(Rank::Five, Suit::Hearts), // pair of 5s
        card(Rank::Nine, Suit::Clubs), // isolated
        card(Rank::Jack, Suit::Clubs), // isolated
    ];
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);

    let one = strategy.choose_exchange_cards(&hand, 1, DuplicateRule::FirstDealtWins, &mut rng);
    assert_eq!(one, vec![card(Rank::Jack, Suit::Clubs)]);

    let two = strategy.choose_exchange_cards(&hand, 2, DuplicateRule::FirstDealtWins, &mut rng);
    assert_eq!(two.len(), 2);
    assert!(two.contains(&card(Rank::Jack, Suit::Clubs)));
    assert!(two.contains(&card(Rank::Nine, Suit::Clubs)));
}

#[test]
fn choose_exchange_cards_breaks_the_pair_only_when_forced() {
    let strategy = HoldBackPairs;
    let hand = vec![
        card(Rank::Five, Suit::Clubs),
        card(Rank::Five, Suit::Hearts),
        card(Rank::Nine, Suit::Clubs),
        card(Rank::Jack, Suit::Clubs),
    ];
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);

    let three = strategy.choose_exchange_cards(&hand, 3, DuplicateRule::FirstDealtWins, &mut rng);
    assert_eq!(three.len(), 3);
    assert!(three.contains(&card(Rank::Jack, Suit::Clubs)));
    assert!(three.contains(&card(Rank::Nine, Suit::Clubs)));
    let fives_included = three
        .iter()
        .filter(|c| c.rank == Rank::Five)
        .count();
    assert_eq!(fives_included, 1);
}

#[test]
fn choose_exchange_cards_handles_no_isolated_cards_at_all() {
    // Two pairs, no singles: forced to break at least one pair even for
    // a small count.
    let strategy = HoldBackPairs;
    let hand = vec![
        card(Rank::Five, Suit::Clubs),
        card(Rank::Five, Suit::Hearts),
        card(Rank::Nine, Suit::Clubs),
        card(Rank::Nine, Suit::Hearts),
    ];
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let given = strategy.choose_exchange_cards(&hand, 3, DuplicateRule::FirstDealtWins, &mut rng);
    assert_eq!(given.len(), 3);
    for card in &given {
        assert!(hand.contains(card));
    }
}
```

Add `use engine::rank_groups;` or call it as `engine::rank_groups(...)`
matching this file's existing import style (it already imports
`engine::{Card, DuplicateRule, Move, Rank}` per Task 1's exploration
findings — extend that line rather than adding a separate `use`).

- [ ] **Step 8: Run tests to verify everything passes**

Run: `cargo test -p sim`
Expected: PASS — all pre-existing `sim` tests plus every new test added
in Steps 5-7.

- [ ] **Step 9: Lint and format**

Run: `cargo fmt -p sim` then `cargo clippy -p sim --all-targets -- -D warnings`
Expected: no warnings. Pay particular attention to any
`clippy::needless_clone` or similar on the `.clone()` calls introduced
in `random_legal.rs`'s `hand[i].clone()` — if `Card` is `Copy` (check
`card.rs`'s derive list from Task 1), use `hand[i]` without `.clone()`
instead.

- [ ] **Step 10: Commit**

```bash
git add sim/src/strategy.rs sim/src/strategies/mod.rs sim/src/strategies/lowest_legal.rs sim/src/strategies/greedy_highest.rs sim/src/strategies/random_legal.rs sim/src/strategies/hold_back_pairs.rs
git commit -m "sim: add Strategy::choose_exchange_cards for all four strategies"
```

---

### Task 3: `sim::match_runner` — wire strategies into the exchange call

**Files:**
- Modify: `sim/src/match_runner.rs`

**Interfaces:**
- Consumes: `engine::exchange_with_selection` (Task 1),
  `Strategy::choose_exchange_cards` (Task 2), and whatever
  `strategies`/`rng` bindings `run_match` already has in scope at the
  existing `exchange(...)` call site.
- Produces: nothing new for later tasks — this is the integration
  point, not a new public interface.

- [ ] **Step 1: Read the current file before editing**

Read `sim/src/match_runner.rs` in full, specifically the
`(Some(roles), Some(arschloch)) => { exchange(&mut hands, roles,
config.duplicate_rule).expect(...); arschloch }` branch inside
`run_match`, and confirm the exact names in scope there
(`strategies`, `rng`, `roles`, `hands`) match what's assumed below —
these were read during Phase 5's exploration but re-verify before
editing, since Phase 4 work may have touched this function too.

- [ ] **Step 2: Replace the `exchange` call with `exchange_with_selection`**

Change the `use engine::{..., exchange, ...};` import to `exchange_with_selection` (keep every other name in that import list unchanged), and replace the call:

```rust
            exchange_with_selection(
                &mut hands,
                roles,
                config.duplicate_rule,
                |seat, hand, count, duplicate_rule| {
                    strategies[seat].choose_exchange_cards(hand, count, duplicate_rule, &mut rng)
                },
            )
            .expect(
                "previous_roles always came from assign_roles for this player_count, and every \
                 Strategy::choose_exchange_cards returns exactly `count` cards from its own hand",
            );
```

If `cargo build` reports the plain `exchange` name is still used
elsewhere in this file, keep it in the import list; otherwise remove it
to avoid an unused-import warning.

If the closure's `&mut rng` capture conflicts with another borrow of
`rng` in the same scope (a borrow-checker error), narrow the closure's
capture or restructure the surrounding code minimally to resolve it —
report back with the exact compiler error and your fix in the task
report rather than guessing silently, since this is exactly the risk
flagged in this plan's Review Focus section.

- [ ] **Step 3: Run the full `sim` test suite**

Run: `cargo test -p sim`
Expected: PASS — `run_match`'s existing determinism and
statistical-invariant tests (`identical_config_and_strategies_are_fully_deterministic`,
etc.) all still pass, now exercising the new call path. If a
determinism test fails, the RNG threading changed observable behavior —
stop and report rather than adjusting the test's expected values.

- [ ] **Step 4: Lint and format**

Run: `cargo fmt -p sim` then `cargo clippy -p sim --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 5: Commit**

```bash
git add sim/src/match_runner.rs
git commit -m "sim: route the exchange through each seat's Strategy::choose_exchange_cards"
```

---

### Task 4: Integration coverage + docs

**Files:**
- Create: `sim/tests/exchange_smoke.rs`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `README.md` (only if it needs a small wording update — see Step 4)

**Interfaces:**
- Consumes: `sim::{run_batch, GreedyHighest, HoldBackPairs, LowestLegal,
  RandomLegal, Strategy}` and whatever the batch-configuration type is
  actually called (verify its name in Step 1 — do not assume
  `BatchConfig` without checking).

- [ ] **Step 1: Confirm the real batch-config type and `run_batch` signature**

`sim/tests/multi_config.rs` (already read while writing this plan)
confirms the real shapes: the config struct is `sim::MatchConfig` with
fields `player_count: u8`, `deck_variant: DeckVariant`, `duplicate_rule:
DuplicateRule`, `rounds: u32` (or whatever integer type `rounds` is —
confirm the exact type when reading the struct definition in
`sim/src/match_runner.rs` or wherever it lives), `seed: u64`; and
`run_batch(&configs, &strategies) -> Vec<MatchResult>` takes `&[MatchConfig]`
and `&[Arc<dyn Strategy>]` as two separate arguments (strategies are
shared across every config/match in one call, one config per match).
`multi_config.rs` also already has a private `baseline_strategies(player_count)`
helper with the rotating `.cycle().skip(...).take(...)` pool-selection
formula — re-read it before writing Step 2 in case it changed since
this plan was written, and copy its exact formula rather than
reintroducing the plain-`.cycle().take(n)` bug Phase 4 fixed.

- [ ] **Step 2: Write the integration test**

Create `sim/tests/exchange_smoke.rs`:

```rust
//! Exercises `exchange_with_selection` through real strategies across
//! every supported table size, guarding against
//! `ExchangeError::InvalidSelection` ever firing in practice (Phase 5,
//! docs/ROADMAP.md).

use std::sync::Arc;

use engine::{DeckVariant, DuplicateRule};
use sim::{run_batch, GreedyHighest, HoldBackPairs, LowestLegal, MatchConfig, RandomLegal, Strategy};

fn baseline_strategies(player_count: u8) -> Vec<Arc<dyn Strategy>> {
    let pool: [Arc<dyn Strategy>; 4] = [
        Arc::new(LowestLegal),
        Arc::new(RandomLegal),
        Arc::new(GreedyHighest),
        Arc::new(HoldBackPairs),
    ];
    pool.iter()
        .cycle()
        .skip(usize::from(player_count) % pool.len())
        .take(usize::from(player_count))
        .cloned()
        .collect()
}

#[test]
fn every_table_size_runs_many_rounds_without_panicking() {
    for player_count in [3u8, 4, 5, 6] {
        let configs: Vec<MatchConfig> = (0..20u64)
            .map(|seed| MatchConfig {
                player_count,
                deck_variant: DeckVariant::Single,
                duplicate_rule: DuplicateRule::FirstDealtWins,
                rounds: 6,
                seed,
            })
            .collect();
        let results = run_batch(&configs, &baseline_strategies(player_count));
        assert_eq!(results.len(), 20, "player_count {player_count}");
    }
}
```

Adjust `MatchConfig`'s field list/types above to match whatever the
real struct definition turns out to be if it differs from
`multi_config.rs`'s usage (e.g. an additional field with a sensible
default) — `multi_config.rs`'s own construction (quoted in Step 1) is
the ground truth, not this code block.

- [ ] **Step 3: Run the new test**

Run: `cargo test -p sim --test exchange_smoke`
Expected: PASS for all four player counts, proving no real strategy
combination ever produces an `ExchangeError::InvalidSelection` panic
in `run_match`.

- [ ] **Step 4: Update `docs/ARCHITECTURE.md`**

Read the current `sim` section of `docs/ARCHITECTURE.md` (it currently
says `choose_exchange_cards` doesn't exist yet, per Phase 4's edit).
Replace that sentence with a short description of what Phase 5 built:
`Strategy::choose_exchange_cards` now exists and is implemented by all
four strategies; `engine::exchange`'s naive top-N tie-break has been
replaced by `engine::exchange_with_selection`, which lets each low
seat's strategy choose which cards it gives up (validated by `engine`,
not merely tie-broken) — `HoldBackPairs` avoids splitting a same-rank
reserve here the same way it does while playing, the other three
strategies reproduce the old naive behavior (two by replicating the
highest-N sort, `RandomLegal` by picking uniformly at random). Also
note `engine::rank_groups` is now public.

Check whether `README.md`'s description of `hold-back-pairs` (in its
"What you get" section) still reads accurately once this lands — if it
only described play-time behavior, extend its one line to mention it
also avoids splitting reserves during the exchange. Skip this file
entirely if the existing wording is generic enough to already cover it
(e.g. it just says "strategy diversification" without describing the
mechanism) — do not force an edit that isn't needed.

- [ ] **Step 5: Full workspace verification**

Run, in order:
1. `cargo fmt --check`
2. `cargo clippy --workspace --all-targets -- -D warnings`
3. `cargo test --workspace`

All three must be clean. Then run the manual sanity check:
`cargo run -p cli -- --player-count 4 --matches 40 --rounds 5 --strategy
lowest-legal --strategy greedy-highest --strategy random-legal
--strategy hold-back-pairs --seed 1 --output /tmp/results.json` and
confirm it exits successfully (Phase 5 adds no new observable CLI
output — this is a regression check).

- [ ] **Step 6: Commit**

```bash
git add sim/tests/exchange_smoke.rs docs/ARCHITECTURE.md README.md
git commit -m "sim: add exchange smoke test; docs: describe Phase 5 smart exchange"
```
