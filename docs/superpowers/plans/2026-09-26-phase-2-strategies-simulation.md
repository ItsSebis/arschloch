# Phase 2 — Strategies & Simulation Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `docs/BUILDING.md` (native build + a real, verified Windows
cross-compile via `cargo-xwin`), then build `sim`: a `Strategy` trait,
three baseline strategies, a legal-move enumerator in `engine`
(`Round::legal_moves`), and a multi-round match runner
(`run_match`/`run_batch`) with `MatchResult`/`Statistics` serialized via
`serde_json`.

**Architecture:** `engine` gains two small, dependency-free-except-serde
additions (a static deck builder, a legal-move enumerator) that are pure
rules knowledge, matching its existing role. `sim` is new content that
loops `engine`'s single-round primitives into a full match: fresh
shuffle+deal each round, mandatory exchange using the previous round's
roles, and each seat's `Strategy` picking from `engine`-reported legal
moves. `run_batch` parallelizes independent matches via `rayon` with no
shared mutable state.

**Tech Stack:** Rust (workspace: `engine`, `sim`, `cli`, `web`), `rand`,
`rayon`, `serde`/`serde_json`, `cargo-xwin` for Windows cross-compilation.

**Spec:** `/home/sebi/.claude/plans/plan-phase-1-kind-globe.md` (the
approved native-plan-mode design for this phase — despite the filename,
its content is Phase 2's design, not Phase 1's), plus `docs/RULES.md`,
`docs/ARCHITECTURE.md`, `docs/ROADMAP.md` (Phase 2 entry), and
`docs/CODING_GUIDELINES.md`.

## Global Constraints

- `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D
  warnings`, `cargo test --workspace` must all be clean before this phase
  is done (`docs/CODING_GUIDELINES.md`).
- Workspace lints: `unsafe_code = "forbid"`, `clippy::pedantic = "warn"`,
  `module_name_repetitions = "allow"`.
- `engine` stays dependency-free except the documented `serde` exception
  for round-tripping game state (`CODING_GUIDELINES.md`). No other
  dependency may be added to `engine`.
- `sim` may add exactly: `rand`, `rayon`, `serde` (with `derive`),
  `serde_json`. No other dependency.
- **Never hand-type a dependency version.** Add dependencies with `cargo
  add <crate> [--features derive] [--dev] -p <crate>` from the repo root,
  so Cargo resolves the current compatible release, not a guessed
  version string.
- One concept per module; a module nearing ~300-400 lines is a signal to
  split (`CODING_GUIDELINES.md`).
- Tests live next to the code they test (`#[cfg(test)] mod tests`);
  cross-module scripted behavior gets a file under `tests/`.
- `#[must_use]` on pure query functions/methods; `# Panics`/`# Errors` doc
  sections wherever `clippy::pedantic` requires them.

---

### Task 1: `docs/BUILDING.md` + verified Windows cross-compile

**Files:**
- Create: `docs/BUILDING.md`
- Modify: none (no code touched)

**Interfaces:**
- Produces: nothing later tasks depend on — this task is independent and
  can run first or in parallel with the others in spirit, though SDD
  dispatches it first since it's quickest to verify.

- [ ] **Step 1: Write the doc**

Create `docs/BUILDING.md` with exactly this content:

````markdown
# Building & Packaging

## Prerequisites

- Rust via [rustup](https://rustup.rs) (this repo pins no toolchain file;
  any current stable toolchain works).

## Native build & test

From the workspace root:

```bash
cargo build --release
cargo test --workspace
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
```

All four are the phase-done gate from `docs/CODING_GUIDELINES.md` and
should be clean before any phase is considered finished.

The `cli` crate is a binary; once it does something (Phase 3 onward),
`cargo run -p cli -- <args>` runs it, and `cargo build --release -p cli`
produces `target/release/cli` (or `cli.exe` on Windows, natively).

## Cross-compiling a Windows executable (from Linux or macOS)

This workspace has zero C dependencies — `rayon`, `rand`, and `serde` are
all pure Rust — so cross-compiling to Windows only needs Rust's own
toolchain plus a Windows CRT/SDK, which
[`cargo-xwin`](https://github.com/rust-cross/cargo-xwin) provides without
installing MinGW or needing `sudo`/root access:

```bash
rustup target add x86_64-pc-windows-msvc
cargo install cargo-xwin
cargo xwin build --release --target x86_64-pc-windows-msvc -p cli
```

The `.exe` lands at `target/x86_64-pc-windows-msvc/release/cli.exe`.

The first `cargo xwin build` downloads and caches the Windows SDK and CRT
headers (a few hundred MB, under `~/.cache/cargo-xwin`); later builds
reuse the cache and are as fast as a normal `cargo build`.

To cross-compile the whole workspace instead of just `cli`, drop `-p cli`.

### Why MSVC instead of the GNU target

The traditional route (`rustup target add x86_64-pc-windows-gnu` +
`apt install mingw-w64`) needs a system package manager and (on most
setups) `sudo`. `cargo-xwin` against `x86_64-pc-windows-msvc` needs
neither — everything it uses is fetched by `cargo`/`rustup` under your own
user account, in keeping with this project's preference for
self-installable tooling.
````

- [ ] **Step 2: Verify it for real**

```bash
rustup target add x86_64-pc-windows-msvc
cargo install cargo-xwin
cargo xwin build --release --target x86_64-pc-windows-msvc -p cli
ls target/x86_64-pc-windows-msvc/release/cli.exe
```

Expected: all four commands succeed and the last one shows the `.exe`
exists. If `cargo install cargo-xwin` or the build fails, capture the
exact error in the task report — don't edit the doc to paper over a
build failure.

- [ ] **Step 3: Commit**

```bash
git add docs/BUILDING.md
git commit -m "docs: add build and Windows cross-compile instructions"
```

---

### Task 2: `engine`: `standard_deck`

**Files:**
- Create: `engine/src/deck.rs`
- Modify: `engine/src/lib.rs`

**Interfaces:**
- Consumes: `crate::card::{Card, DeckVariant, Rank, Suit}` (all already
  `pub` in `engine/src/card.rs`).
- Produces: `pub fn standard_deck(variant: DeckVariant) -> Vec<Card>`,
  re-exported from `lib.rs` as `engine::standard_deck` — Task 6's match
  runner calls this every round.

- [ ] **Step 1: Write `engine/src/deck.rs`**

```rust
//! Building a fresh, unshuffled standard deck. Shuffling is the caller's
//! job (see docs/RULES.md, "Duplicate cards") — this only knows which
//! cards exist.

use crate::card::{Card, DeckVariant, Rank, Suit};

/// All cards for a `variant` deck, in a fixed canonical order (unshuffled,
/// every `deal_index` set to `0`). In the `Double` variant, each
/// (rank, suit) pair appears twice consecutively. Reassign `deal_index` by
/// shuffle position before dealing, so `DuplicateRule` can disambiguate
/// true duplicates (see docs/RULES.md, "Duplicate cards").
#[must_use]
pub fn standard_deck(variant: DeckVariant) -> Vec<Card> {
    let suits = [Suit::Diamonds, Suit::Hearts, Suit::Spades, Suit::Clubs];
    let ranks = [
        Rank::Two,
        Rank::Three,
        Rank::Four,
        Rank::Five,
        Rank::Six,
        Rank::Seven,
        Rank::Eight,
        Rank::Nine,
        Rank::Ten,
        Rank::Jack,
        Rank::Queen,
        Rank::King,
        Rank::Ace,
    ];
    let capacity = if variant == DeckVariant::Double { 104 } else { 52 };
    let mut cards = Vec::with_capacity(capacity);
    for suit in suits {
        for rank in ranks {
            cards.push(Card::new(rank, suit, 0));
            if variant == DeckVariant::Double {
                cards.push(Card::new(rank, suit, 0));
            }
        }
    }
    cards
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn single_deck_has_52_unique_cards() {
        let cards = standard_deck(DeckVariant::Single);
        assert_eq!(cards.len(), 52);
        let mut counts: HashMap<(Rank, Suit), u32> = HashMap::new();
        for card in &cards {
            *counts.entry((card.rank, card.suit)).or_insert(0) += 1;
        }
        assert_eq!(counts.len(), 52);
        assert!(counts.values().all(|&count| count == 1));
    }

    #[test]
    fn double_deck_has_104_cards_each_pair_appearing_twice() {
        let cards = standard_deck(DeckVariant::Double);
        assert_eq!(cards.len(), 104);
        let mut counts: HashMap<(Rank, Suit), u32> = HashMap::new();
        for card in &cards {
            *counts.entry((card.rank, card.suit)).or_insert(0) += 1;
        }
        assert_eq!(counts.len(), 52);
        assert!(counts.values().all(|&count| count == 2));
    }
}
```

- [ ] **Step 2: Wire it into `lib.rs`**

In `engine/src/lib.rs`, add `pub mod deck;` alongside the other `pub mod`
lines, and add `standard_deck` to the `pub use` list:

```rust
pub mod deck;
```
```rust
pub use deck::standard_deck;
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p engine deck::
```
Expected: 2 new tests pass.

- [ ] **Step 4: Commit**

```bash
git add engine/src/deck.rs engine/src/lib.rs
git commit -m "engine: add standard_deck"
```

---

### Task 3: `engine`: legal-move enumerator + public `Combo::top_card`

**Files:**
- Create: `engine/src/legal_moves.rs` (crate-private — no `pub`)
- Modify: `engine/src/round.rs` (add `Round::legal_moves`)
- Modify: `engine/src/combo.rs` (`top_card` becomes `pub`, returns owned
  `Card`)
- Modify: `engine/src/lib.rs` (`mod legal_moves;`)

**Interfaces:**
- Consumes: `crate::card::{Card, DuplicateRule}`, `crate::combo::Combo`,
  `crate::round::Move`.
- Produces: `Round::legal_moves(&self) -> Vec<Move>` (public, used by
  Task 6's match runner and by strategies via the moves it returns);
  `Combo::top_card(&self, DuplicateRule) -> Card` (now `pub`, used by
  Task 5's strategies to rank combos).

- [ ] **Step 1: Make `Combo::top_card` public and owned**

In `engine/src/combo.rs`, change:

```rust
    fn top_card(&self, duplicate_rule: DuplicateRule) -> &Card {
        self.cards
            .iter()
            .max_by(|a, b| a.compare(b, duplicate_rule))
            .expect("Combo is always constructed with at least one card")
    }
```

to:

```rust
    /// The representative card used for comparison: since every card in
    /// a combo shares a rank, the highest card (by suit/duplicate
    /// tiebreak) stands in for the whole combo.
    #[must_use]
    pub fn top_card(&self, duplicate_rule: DuplicateRule) -> Card {
        *self
            .cards
            .iter()
            .max_by(|a, b| a.compare(b, duplicate_rule))
            .expect("Combo is always constructed with at least one card")
    }
```

and update `beats` (same file) since `top_card` now returns an owned
`Card` instead of `&Card`:

```rust
    #[must_use]
    pub fn beats(&self, previous: &Combo, duplicate_rule: DuplicateRule) -> bool {
        self.size() == previous.size()
            && self
                .top_card(duplicate_rule)
                .compare(&previous.top_card(duplicate_rule), duplicate_rule)
                == Ordering::Greater
    }
```

Add this test to `combo.rs`'s existing `mod tests`:

```rust
    #[test]
    fn top_card_is_public_and_returns_the_highest_card_in_the_combo() {
        let combo = Combo::new(vec![
            card(Rank::Seven, Suit::Diamonds),
            card(Rank::Seven, Suit::Clubs),
        ])
        .unwrap();
        assert_eq!(
            combo.top_card(DuplicateRule::FirstDealtWins),
            card(Rank::Seven, Suit::Clubs)
        );
    }
```

- [ ] **Step 2: Run existing combo tests**

```bash
cargo test -p engine combo::
```
Expected: all existing combo tests plus the new one pass unchanged
(`beats`'s behavior didn't change, only `top_card`'s signature).

- [ ] **Step 3: Write `engine/src/legal_moves.rs`**

```rust
//! Enumerates the legal moves for a hand against an optional current
//! combo. Crate-private: `Round::legal_moves` is the public entry point.
//!
//! **Decision:** rather than every suit-subset of a rank (a full power
//! set, which blows up combinatorially in the double-deck variant — up
//! to 2^8-1 = 255 subsets for one rank held 8 times), this produces at
//! most two candidates per (rank, size): the `size` lowest-ranked and
//! `size` highest-ranked physical cards of that rank (deduped when equal).
//! `sim`'s baseline strategies only ever need "the weakest" or "the
//! strongest" combo of a given size, so this is enough without solving a
//! combinatorial optimization problem the roadmap doesn't call for.

use crate::card::{Card, DuplicateRule};
use crate::combo::Combo;
use crate::round::Move;

pub(crate) fn legal_moves(
    hand: &[Card],
    current_combo: Option<&Combo>,
    duplicate_rule: DuplicateRule,
) -> Vec<Move> {
    let mut moves = Vec::new();
    if current_combo.is_some() {
        moves.push(Move::Pass);
    }
    for group in rank_groups(hand) {
        let sizes: Vec<usize> = match current_combo {
            Some(current) if current.size() <= group.len() => vec![current.size()],
            Some(_) => Vec::new(),
            None => (1..=group.len()).collect(),
        };
        for size in sizes {
            for subset in canonical_subsets(&group, size, duplicate_rule) {
                let candidate = Combo::new(subset)
                    .expect("a non-empty subset of one rank group is always a valid Combo");
                let legal = match current_combo {
                    Some(current) => candidate.beats(current, duplicate_rule),
                    None => true,
                };
                if legal {
                    moves.push(Move::Play(candidate));
                }
            }
        }
    }
    moves
}

/// Groups `hand` by rank.
fn rank_groups(hand: &[Card]) -> Vec<Vec<Card>> {
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

/// The `size` lowest-ranked and `size` highest-ranked cards of `group`
/// (already-confirmed single rank), deduped when they're the same set.
/// Returns nothing if `size` is `0` or larger than `group`.
fn canonical_subsets(
    group: &[Card],
    size: usize,
    duplicate_rule: DuplicateRule,
) -> Vec<Vec<Card>> {
    if size == 0 || size > group.len() {
        return Vec::new();
    }
    let mut sorted = group.to_vec();
    sorted.sort_by(|a, b| a.compare(b, duplicate_rule));
    let lowest = sorted[..size].to_vec();
    let highest = sorted[sorted.len() - size..].to_vec();
    if lowest == highest {
        vec![lowest]
    } else {
        vec![lowest, highest]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{Rank, Suit};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit, 0)
    }

    #[test]
    fn leading_enumerates_every_size_up_to_the_rank_groups_count() {
        let hand = vec![
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Seven, Suit::Diamonds),
            card(Rank::Nine, Suit::Hearts),
        ];
        let moves = legal_moves(&hand, None, DuplicateRule::FirstDealtWins);
        // Seven group (2 cards): size 1 -> 2 candidates (distinct
        // suits), size 2 -> 1 candidate (whole group, deduped). Nine
        // group (1 card): size 1 -> 1 candidate. Total 4. No Pass while
        // leading.
        assert_eq!(moves.len(), 4);
        assert!(!moves.contains(&Move::Pass));
    }

    #[test]
    fn following_only_considers_the_current_combos_size_and_filters_by_beats() {
        let hand = vec![
            card(Rank::Six, Suit::Clubs),
            card(Rank::Eight, Suit::Clubs),
            card(Rank::Eight, Suit::Diamonds),
        ];
        let current = Combo::new(vec![card(Rank::Seven, Suit::Clubs)]).unwrap();
        let moves = legal_moves(&hand, Some(&current), DuplicateRule::FirstDealtWins);
        // Pass, plus both size-1 Eights (both beat Seven regardless of
        // suit, since rank alone decides once it's strictly higher); the
        // size-1 Six doesn't beat, and there's no size-2 group to match a
        // size-1 current combo anyway.
        assert_eq!(moves.len(), 3);
        assert!(moves.contains(&Move::Pass));
        let eight_plays = moves
            .iter()
            .filter(|m| {
                matches!(m, Move::Play(c) if c.size() == 1 && c.cards()[0].rank == Rank::Eight)
            })
            .count();
        assert_eq!(eight_plays, 2);
        assert!(!moves
            .iter()
            .any(|m| matches!(m, Move::Play(c) if c.cards()[0].rank == Rank::Six)));
    }

    #[test]
    fn a_rank_group_smaller_than_the_current_combos_size_contributes_nothing() {
        let hand = vec![card(Rank::Nine, Suit::Clubs)];
        let current = Combo::new(vec![
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Seven, Suit::Diamonds),
        ])
        .unwrap();
        let moves = legal_moves(&hand, Some(&current), DuplicateRule::FirstDealtWins);
        assert_eq!(moves, vec![Move::Pass]);
    }

    #[test]
    fn a_single_card_of_a_rank_produces_one_deduped_candidate() {
        let hand = vec![card(Rank::Ten, Suit::Clubs)];
        let moves = legal_moves(&hand, None, DuplicateRule::FirstDealtWins);
        assert_eq!(moves.len(), 1);
    }

    #[test]
    fn double_deck_duplicate_cards_are_disambiguated_by_deal_index() {
        let hand = vec![
            Card::new(Rank::Ten, Suit::Clubs, 0),
            Card::new(Rank::Ten, Suit::Clubs, 1),
        ];
        let moves = legal_moves(&hand, None, DuplicateRule::FirstDealtWins);
        // size 1 -> 2 candidates (the two physical cards are distinct by
        // deal_index), size 2 -> 1 candidate (whole group). Total 3.
        assert_eq!(moves.len(), 3);
    }
}
```

- [ ] **Step 4: Wire `legal_moves` into `Round`**

In `engine/src/round.rs`, add this method to `impl Round` (near
`current_combo`/`current_trick_leader`):

```rust
    /// All moves currently legal for `self.seat_to_move()`. Empty if the
    /// round is already complete.
    #[must_use]
    pub fn legal_moves(&self) -> Vec<Move> {
        let Some(seat) = self.seat_to_move() else {
            return Vec::new();
        };
        crate::legal_moves::legal_moves(
            &self.hands[usize::from(seat)],
            self.current_combo.as_ref(),
            self.duplicate_rule,
        )
    }
```

Add this test to `round.rs`'s existing `mod tests`:

```rust
    #[test]
    fn legal_moves_is_empty_once_the_round_is_complete() {
        let hands = vec![
            vec![card(Rank::Two, Suit::Clubs)],
            vec![card(Rank::Three, Suit::Clubs)],
            vec![card(Rank::Four, Suit::Clubs)],
        ];
        let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
        round
            .submit_move(0, Move::Play(combo(vec![card(Rank::Two, Suit::Clubs)])))
            .unwrap();
        round
            .submit_move(1, Move::Play(combo(vec![card(Rank::Three, Suit::Clubs)])))
            .unwrap();
        assert!(round.is_complete());
        assert_eq!(round.legal_moves(), Vec::new());
    }

    #[test]
    fn legal_moves_matches_the_leaders_hand_when_no_combo_is_on_the_table() {
        let hands = vec![
            vec![card(Rank::Two, Suit::Clubs), card(Rank::Two, Suit::Diamonds)],
            vec![card(Rank::Three, Suit::Clubs)],
            vec![card(Rank::Four, Suit::Clubs)],
        ];
        let round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
        let moves = round.legal_moves();
        assert!(!moves.is_empty());
        assert!(!moves.contains(&Move::Pass));
    }
```

- [ ] **Step 5: Wire the module into `lib.rs`**

In `engine/src/lib.rs`, add `mod legal_moves;` (no `pub` — crate-private,
mirroring `mod trick;`):

```rust
mod legal_moves;
```

- [ ] **Step 6: Run tests**

```bash
cargo test -p engine
```
Expected: all prior engine tests plus the new `legal_moves`/`round`/
`combo` tests pass.

- [ ] **Step 7: Commit**

```bash
git add engine/src/legal_moves.rs engine/src/round.rs engine/src/combo.rs engine/src/lib.rs
git commit -m "engine: add Round::legal_moves and make Combo::top_card public"
```

---

### Task 4: `engine`: `Role` gains `serde::Serialize`/`Deserialize`

**Files:**
- Modify: `engine/Cargo.toml` (add `serde`, `serde_json` as dev-dep)
- Modify: `engine/src/role.rs`

**Interfaces:**
- Produces: `Role: serde::Serialize + serde::Deserialize` — Task 6's
  `MatchResult` embeds `Vec<Vec<Role>>` and needs this to serialize.

- [ ] **Step 1: Add dependencies**

```bash
cargo add serde --features derive -p engine
cargo add serde_json --dev -p engine
```

`serde_json` is dev-only (test-only), so it doesn't affect `engine`'s
production dependency footprint — the `serde` (+derive) exception is
already documented in `CODING_GUIDELINES.md`.

- [ ] **Step 2: Add the derive**

In `engine/src/role.rs`, change:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
```

to:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Role {
```

Add this test to `role.rs`'s existing `mod tests`:

```rust
    #[test]
    fn role_round_trips_through_json() {
        let json = serde_json::to_string(&Role::President).unwrap();
        let role: Role = serde_json::from_str(&json).unwrap();
        assert_eq!(role, Role::President);
    }
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p engine role::
```
Expected: all existing role tests plus the new one pass.

- [ ] **Step 4: Commit**

```bash
git add engine/Cargo.toml engine/src/role.rs
git commit -m "engine: derive Serialize/Deserialize on Role"
```

---

### Task 5: `sim`: `Strategy` trait + three baseline strategies

**Files:**
- Modify: `sim/Cargo.toml` (add `rand`)
- Modify: `sim/src/lib.rs`
- Create: `sim/src/strategy.rs`
- Create: `sim/src/strategies/mod.rs`
- Create: `sim/src/strategies/lowest_legal.rs`
- Create: `sim/src/strategies/greedy_highest.rs`
- Create: `sim/src/strategies/random_legal.rs`

**Interfaces:**
- Consumes: `engine::{DuplicateRule, Move}` (public in `engine`, confirmed
  in `engine/src/lib.rs`); `Combo::size()`/`Combo::top_card()` (public,
  Task 3).
- Produces: `pub trait Strategy` and `LowestLegal`/`GreedyHighest`/
  `RandomLegal` — Task 6's match runner holds `Arc<dyn Strategy>` per seat
  and calls `choose_play`.

- [ ] **Step 1: Add the dependency**

```bash
cargo add rand -p sim
```

- [ ] **Step 2: Write `sim/src/strategy.rs`**

```rust
//! The `Strategy` trait: how a simulated seat picks among legal moves.

use engine::{DuplicateRule, Move};

/// Chooses a move from the moves `engine` reports as legal. Implementors
/// must be `Send + Sync` so a single instance (behind `Arc`) can be
/// shared read-only across many parallel matches; per-match randomness is
/// threaded through via `rng` rather than owned by the strategy, so every
/// match's outcome depends only on its own seed, not on thread
/// scheduling.
pub trait Strategy: Send + Sync {
    /// A short, stable name used to group results by strategy (see
    /// `crate::statistics::aggregate`).
    fn name(&self) -> &'static str;

    /// Picks one entry from `legal_moves` (never empty when a seat is
    /// actually to move — see `engine::Round::legal_moves`).
    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        rng: &mut dyn rand::RngCore,
    ) -> Move;
}
```

- [ ] **Step 3: Write `sim/src/strategies/lowest_legal.rs`**

```rust
//! The most conservative legal strategy: always plays the smallest, then
//! lowest-ranked, legal combo; passes only when no `Play` is legal.

use engine::{DuplicateRule, Move};

use crate::strategy::Strategy;

#[derive(Debug, Clone, Copy, Default)]
pub struct LowestLegal;

impl Strategy for LowestLegal {
    fn name(&self) -> &'static str {
        "LowestLegal"
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        _rng: &mut dyn rand::RngCore,
    ) -> Move {
        legal_moves
            .iter()
            .filter_map(|mv| match mv {
                Move::Play(combo) => Some((combo.size(), combo.top_card(duplicate_rule), mv)),
                Move::Pass => None,
            })
            .min_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.compare(&b.1, duplicate_rule)))
            .map_or(Move::Pass, |(_, _, mv)| mv.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::{Card, Combo, Rank, Suit};
    use rand::SeedableRng;

    fn test_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(0)
    }

    fn single(rank: Rank, suit: Suit) -> Move {
        Move::Play(Combo::new(vec![Card::new(rank, suit, 0)]).unwrap())
    }

    #[test]
    fn prefers_the_smallest_combo_size() {
        let pair = Move::Play(
            Combo::new(vec![
                Card::new(Rank::Two, Suit::Clubs, 0),
                Card::new(Rank::Two, Suit::Diamonds, 0),
            ])
            .unwrap(),
        );
        let legal = vec![pair, single(Rank::Nine, Suit::Clubs)];
        let strategy = LowestLegal;
        let chosen = strategy.choose_play(&legal, DuplicateRule::FirstDealtWins, &mut test_rng());
        assert_eq!(chosen, single(Rank::Nine, Suit::Clubs));
    }

    #[test]
    fn among_equal_sizes_prefers_the_lowest_top_card() {
        let legal = vec![
            single(Rank::Nine, Suit::Clubs),
            single(Rank::Six, Suit::Diamonds),
        ];
        let strategy = LowestLegal;
        let chosen = strategy.choose_play(&legal, DuplicateRule::FirstDealtWins, &mut test_rng());
        assert_eq!(chosen, single(Rank::Six, Suit::Diamonds));
    }

    #[test]
    fn passes_when_no_play_is_legal() {
        let legal = vec![Move::Pass];
        let strategy = LowestLegal;
        let chosen = strategy.choose_play(&legal, DuplicateRule::FirstDealtWins, &mut test_rng());
        assert_eq!(chosen, Move::Pass);
    }
}
```

- [ ] **Step 4: Write `sim/src/strategies/greedy_highest.rs`**

```rust
//! The most aggressive legal strategy: prefers the largest combo (shed
//! the most cards per lead — following is already size-locked by the
//! current combo, so this only matters while leading), then the
//! highest-ranked, legal combo; passes only when no `Play` is legal.

use engine::{DuplicateRule, Move};

use crate::strategy::Strategy;

#[derive(Debug, Clone, Copy, Default)]
pub struct GreedyHighest;

impl Strategy for GreedyHighest {
    fn name(&self) -> &'static str {
        "GreedyHighest"
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        _rng: &mut dyn rand::RngCore,
    ) -> Move {
        legal_moves
            .iter()
            .filter_map(|mv| match mv {
                Move::Play(combo) => Some((combo.size(), combo.top_card(duplicate_rule), mv)),
                Move::Pass => None,
            })
            .max_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.compare(&b.1, duplicate_rule)))
            .map_or(Move::Pass, |(_, _, mv)| mv.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::{Card, Combo, Rank, Suit};
    use rand::SeedableRng;

    fn test_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(0)
    }

    fn single(rank: Rank, suit: Suit) -> Move {
        Move::Play(Combo::new(vec![Card::new(rank, suit, 0)]).unwrap())
    }

    #[test]
    fn prefers_the_largest_combo_size() {
        let pair = Move::Play(
            Combo::new(vec![
                Card::new(Rank::Two, Suit::Clubs, 0),
                Card::new(Rank::Two, Suit::Diamonds, 0),
            ])
            .unwrap(),
        );
        let legal = vec![pair.clone(), single(Rank::Ace, Suit::Clubs)];
        let strategy = GreedyHighest;
        let chosen = strategy.choose_play(&legal, DuplicateRule::FirstDealtWins, &mut test_rng());
        assert_eq!(chosen, pair);
    }

    #[test]
    fn among_equal_sizes_prefers_the_highest_top_card() {
        let legal = vec![
            single(Rank::Nine, Suit::Clubs),
            single(Rank::Ace, Suit::Diamonds),
        ];
        let strategy = GreedyHighest;
        let chosen = strategy.choose_play(&legal, DuplicateRule::FirstDealtWins, &mut test_rng());
        assert_eq!(chosen, single(Rank::Ace, Suit::Diamonds));
    }

    #[test]
    fn passes_when_no_play_is_legal() {
        let legal = vec![Move::Pass];
        let strategy = GreedyHighest;
        let chosen = strategy.choose_play(&legal, DuplicateRule::FirstDealtWins, &mut test_rng());
        assert_eq!(chosen, Move::Pass);
    }
}
```

- [ ] **Step 5: Write `sim/src/strategies/random_legal.rs`**

```rust
//! Uniformly samples among every legal move, including `Pass` — this is
//! the strategy that produces the "voluntary pass" signal (see
//! docs/ROADMAP.md, Phase 4).

use engine::{DuplicateRule, Move};
use rand::seq::SliceRandom;

use crate::strategy::Strategy;

#[derive(Debug, Clone, Copy, Default)]
pub struct RandomLegal;

impl Strategy for RandomLegal {
    fn name(&self) -> &'static str {
        "RandomLegal"
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        _duplicate_rule: DuplicateRule,
        rng: &mut dyn rand::RngCore,
    ) -> Move {
        legal_moves
            .choose(rng)
            .cloned()
            .expect("legal_moves is never empty when a seat is actually to move")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::{Card, Combo, Rank, Suit};
    use rand::SeedableRng;

    fn test_rng(seed: u64) -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(seed)
    }

    #[test]
    fn always_returns_one_of_the_legal_moves() {
        let legal = vec![
            Move::Pass,
            Move::Play(Combo::new(vec![Card::new(Rank::Nine, Suit::Clubs, 0)]).unwrap()),
        ];
        let strategy = RandomLegal;
        for seed in 0..20 {
            let chosen =
                strategy.choose_play(&legal, DuplicateRule::FirstDealtWins, &mut test_rng(seed));
            assert!(legal.contains(&chosen));
        }
    }

    #[test]
    fn eventually_picks_every_option_across_many_seeds() {
        let legal = vec![
            Move::Pass,
            Move::Play(Combo::new(vec![Card::new(Rank::Nine, Suit::Clubs, 0)]).unwrap()),
        ];
        let strategy = RandomLegal;
        let mut saw_pass = false;
        let mut saw_play = false;
        for seed in 0..50 {
            match strategy.choose_play(&legal, DuplicateRule::FirstDealtWins, &mut test_rng(seed))
            {
                Move::Pass => saw_pass = true,
                Move::Play(_) => saw_play = true,
            }
        }
        assert!(saw_pass && saw_play);
    }
}
```

- [ ] **Step 6: Write `sim/src/strategies/mod.rs`**

```rust
mod greedy_highest;
mod lowest_legal;
mod random_legal;

pub use greedy_highest::GreedyHighest;
pub use lowest_legal::LowestLegal;
pub use random_legal::RandomLegal;
```

- [ ] **Step 7: Replace `sim/src/lib.rs`**

```rust
//! Strategies, the multi-round match runner, and result/statistics types
//! built on top of `engine`'s single-round primitives. See
//! docs/ARCHITECTURE.md, "sim".

pub mod strategies;
pub mod strategy;

pub use strategies::{GreedyHighest, LowestLegal, RandomLegal};
pub use strategy::Strategy;
```

- [ ] **Step 8: Run tests**

```bash
cargo test -p sim
```
Expected: 9 new tests pass (3 per strategy).

- [ ] **Step 9: Commit**

```bash
git add sim/Cargo.toml sim/src/lib.rs sim/src/strategy.rs sim/src/strategies
git commit -m "sim: add Strategy trait and three baseline strategies"
```

---

### Task 6: `sim`: `MatchConfig`, `MatchResult`, `run_match`

**Files:**
- Modify: `sim/Cargo.toml` (add `serde` with `derive`, `serde_json`)
- Modify: `sim/src/lib.rs`
- Create: `sim/src/match_config.rs`
- Create: `sim/src/match_result.rs`
- Create: `sim/src/match_runner.rs`

**Interfaces:**
- Consumes: `engine::{assign_roles, deal, exchange, lowest_card_holder,
  standard_deck, DeckVariant, DuplicateRule, Move, Role, Round, SeatId}`
  (all confirmed `pub` re-exports in `engine/src/lib.rs`, plus Task 2's
  `standard_deck` and Task 3's `Round::legal_moves`); `crate::strategy::
  Strategy` (Task 5).
- Produces: `MatchConfig` (plain struct), `MatchResult` (plain struct,
  `serde::Serialize`), `run_match(&MatchConfig, &[Arc<dyn Strategy>]) ->
  MatchResult` — Task 7's `run_batch` and `Statistics::aggregate` consume
  `MatchResult`.

- [ ] **Step 1: Add dependencies**

```bash
cargo add serde --features derive -p sim
cargo add serde_json -p sim
```

- [ ] **Step 2: Write `sim/src/match_config.rs`**

```rust
//! Configuration for one simulated match (multiple rounds with role
//! carry-over). See `crate::match_runner::run_match`.

use engine::{DeckVariant, DuplicateRule};

/// One match's setup: how many seats, which deck/duplicate rules, how
/// many rounds to play, and the seed its shuffles derive from. Two
/// `MatchConfig`s with the same fields always produce the same
/// `MatchResult` from `run_match` given the same strategies (see
/// `sim/tests/small_batch.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatchConfig {
    pub player_count: u8,
    pub deck_variant: DeckVariant,
    pub duplicate_rule: DuplicateRule,
    pub rounds: usize,
    pub seed: u64,
}
```

- [ ] **Step 3: Write `sim/src/match_result.rs`**

```rust
//! The outcome of one simulated match: role history and move-shape
//! counters consumed by `crate::statistics::aggregate`.

use engine::Role;

#[derive(Debug, Clone, serde::Serialize)]
pub struct MatchResult {
    pub player_count: u8,
    /// One entry per seat (index = `SeatId`).
    pub strategy_names: Vec<String>,
    /// One entry per round played, each seat-indexed.
    pub role_history: Vec<Vec<Role>>,
    pub trick_count: u32,
    pub pass_count: u32,
    /// Passes submitted while at least one `Move::Play` was also legal
    /// (see docs/ROADMAP.md, Phase 4, "Strategy diversification").
    pub voluntary_pass_count: u32,
}
```

- [ ] **Step 4: Write `sim/src/match_runner.rs`**

```rust
//! Drives `engine`'s single-round primitives into a full multi-round
//! match: a fresh shuffle and deal every round (cards are discarded
//! within a round, not carried over — see docs/RULES.md, "Playing a
//! Round"), the mandatory exchange using the previous round's roles, and
//! each seat's `Strategy` choosing among `engine`'s reported legal moves.

use std::sync::Arc;

use engine::{assign_roles, deal, exchange, lowest_card_holder, standard_deck, Move, Round, SeatId};
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::match_config::MatchConfig;
use crate::match_result::MatchResult;
use crate::strategy::Strategy;

/// Simulates one full match (`config.rounds` rounds, role carry-over
/// between them) using `strategies` (one per seat).
///
/// # Panics
///
/// Panics if `strategies.len() != usize::from(config.player_count)`, or
/// if `config.rounds == 0`, or if `config.player_count` isn't a table
/// size `engine` supports (3-6) — all are programming errors in how the
/// caller built `MatchConfig`/`strategies`, not user input this phase
/// exposes to anyone yet (see docs/CODING_GUIDELINES.md, "Errors"; a CLI
/// boundary with proper `Result`-based validation arrives in Phase 3).
#[must_use]
pub fn run_match(config: &MatchConfig, strategies: &[Arc<dyn Strategy>]) -> MatchResult {
    assert_eq!(
        strategies.len(),
        usize::from(config.player_count),
        "one strategy is required per seat"
    );
    assert!(config.rounds > 0, "a match needs at least one round");

    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    let mut previous_roles: Option<Vec<engine::Role>> = None;
    let mut previous_arschloch: Option<SeatId> = None;
    let mut role_history = Vec::with_capacity(config.rounds);
    let mut trick_count = 0u32;
    let mut pass_count = 0u32;
    let mut voluntary_pass_count = 0u32;

    for _ in 0..config.rounds {
        let mut deck = standard_deck(config.deck_variant);
        deck.shuffle(&mut rng);
        for (index, card) in deck.iter_mut().enumerate() {
            card.deal_index = u8::try_from(index).expect("deck sizes (52/104) fit in u8");
        }
        let mut hands = deal(deck, config.player_count)
            .expect("standard_deck always yields enough cards for a supported player count");

        let leader = match (&previous_roles, previous_arschloch) {
            (Some(roles), Some(arschloch)) => {
                exchange(&mut hands, roles, config.duplicate_rule)
                    .expect("previous_roles always came from assign_roles for this player_count");
                arschloch
            }
            _ => lowest_card_holder(&hands, config.duplicate_rule)
                .expect("a freshly dealt hand set is never empty"),
        };

        let mut round = Round::new(hands, config.duplicate_rule, leader)
            .expect("player_count/leader are always valid for a supported table size");

        while !round.is_complete() {
            let seat = round.seat_to_move().expect("round is not complete");
            let legal_moves = round.legal_moves();
            let chosen = strategies[usize::from(seat)].choose_play(
                &legal_moves,
                config.duplicate_rule,
                &mut rng,
            );

            if chosen == Move::Pass {
                pass_count += 1;
                if legal_moves.iter().any(|mv| matches!(mv, Move::Play(_))) {
                    voluntary_pass_count += 1;
                }
            }
            let combo_was_on_table = round.current_combo().is_some();
            round
                .submit_move(seat, chosen)
                .expect("strategies only choose from the moves engine just reported as legal");
            if combo_was_on_table && round.current_combo().is_none() {
                trick_count += 1;
            }
        }

        let finishing_order = round.finishing_order().to_vec();
        let roles = assign_roles(&finishing_order, config.player_count)
            .expect("finishing_order is always a valid permutation for a supported player count");
        previous_arschloch = finishing_order.last().copied();
        role_history.push(roles.clone());
        previous_roles = Some(roles);
    }

    MatchResult {
        player_count: config.player_count,
        strategy_names: strategies.iter().map(|s| s.name().to_string()).collect(),
        role_history,
        trick_count,
        pass_count,
        voluntary_pass_count,
    }
}
```

- [ ] **Step 5: Replace `sim/src/lib.rs`**

```rust
//! Strategies, the multi-round match runner, and result/statistics types
//! built on top of `engine`'s single-round primitives. See
//! docs/ARCHITECTURE.md, "sim".

pub mod match_config;
pub mod match_result;
pub mod match_runner;
pub mod strategies;
pub mod strategy;

pub use match_config::MatchConfig;
pub use match_result::MatchResult;
pub use match_runner::run_match;
pub use strategies::{GreedyHighest, LowestLegal, RandomLegal};
pub use strategy::Strategy;
```

- [ ] **Step 6: Write a quick sanity test**

Add to `sim/src/match_runner.rs`'s bottom, a `#[cfg(test)] mod tests`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use engine::{DeckVariant, DuplicateRule};

    fn four_lowest_legal() -> Vec<Arc<dyn Strategy>> {
        vec![
            Arc::new(crate::strategies::LowestLegal),
            Arc::new(crate::strategies::LowestLegal),
            Arc::new(crate::strategies::LowestLegal),
            Arc::new(crate::strategies::LowestLegal),
        ]
    }

    #[test]
    fn run_match_produces_one_role_history_entry_per_round() {
        let config = MatchConfig {
            player_count: 4,
            deck_variant: DeckVariant::Single,
            duplicate_rule: DuplicateRule::FirstDealtWins,
            rounds: 3,
            seed: 1,
        };
        let result = run_match(&config, &four_lowest_legal());
        assert_eq!(result.role_history.len(), 3);
        for round_roles in &result.role_history {
            assert_eq!(round_roles.len(), 4);
        }
    }

    #[test]
    fn identical_config_and_strategies_are_fully_deterministic() {
        let config = MatchConfig {
            player_count: 4,
            deck_variant: DeckVariant::Single,
            duplicate_rule: DuplicateRule::FirstDealtWins,
            rounds: 3,
            seed: 42,
        };
        let strategies = four_lowest_legal();
        let first = run_match(&config, &strategies);
        let second = run_match(&config, &strategies);
        assert_eq!(first.role_history, second.role_history);
        assert_eq!(first.trick_count, second.trick_count);
        assert_eq!(first.pass_count, second.pass_count);
    }
}
```

- [ ] **Step 7: Run tests**

```bash
cargo test -p sim
```
Expected: all prior `sim` tests plus these 2 new ones pass.

- [ ] **Step 8: Commit**

```bash
git add sim/Cargo.toml sim/src/lib.rs sim/src/match_config.rs sim/src/match_result.rs sim/src/match_runner.rs
git commit -m "sim: add MatchConfig, MatchResult, and run_match"
```

---

### Task 7: `sim`: `Statistics`, `run_batch`, and the small-batch integration test

**Files:**
- Modify: `sim/Cargo.toml` (add `rayon`)
- Modify: `sim/src/lib.rs`
- Modify: `sim/src/match_runner.rs` (add `run_batch`)
- Create: `sim/src/statistics.rs`
- Create: `sim/tests/small_batch.rs`

**Interfaces:**
- Consumes: `crate::match_result::MatchResult`, `engine::Role`.
- Produces: `Statistics`, `aggregate(&[MatchResult]) -> Statistics`,
  `run_batch(&[MatchConfig], &[Arc<dyn Strategy>]) -> Vec<MatchResult>` —
  this is the phase's final public surface; nothing later in this plan
  depends on it.

- [ ] **Step 1: Add the dependency**

```bash
cargo add rayon -p sim
```

- [ ] **Step 2: Add `run_batch` to `sim/src/match_runner.rs`**

Add near the top:
```rust
use rayon::prelude::*;
```

Add after `run_match`:
```rust
/// Simulates every config in `configs` in parallel — independent
/// matches, no shared mutable state (see docs/ARCHITECTURE.md,
/// "Threading model").
#[must_use]
pub fn run_batch(configs: &[MatchConfig], strategies: &[Arc<dyn Strategy>]) -> Vec<MatchResult> {
    configs
        .par_iter()
        .map(|config| run_match(config, strategies))
        .collect()
}
```

- [ ] **Step 3: Write `sim/src/statistics.rs`**

```rust
//! Aggregating a batch of `MatchResult`s into per-strategy role counts
//! and a diversification signal. See docs/ROADMAP.md, Phase 2 and 4.

use std::collections::HashMap;

use engine::Role;

use crate::match_result::MatchResult;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Statistics {
    pub matches_played: usize,
    pub role_counts_by_strategy: HashMap<String, HashMap<Role, u32>>,
    /// `sum(voluntary_pass_count) / sum(pass_count)` across every match
    /// (`0.0` if no passes occurred at all).
    pub voluntary_pass_rate: f64,
}

/// Aggregates `results` into `Statistics`. Every round in every match
/// contributes one role count per seat, credited to that seat's
/// strategy (by name — seats using the same strategy type share a
/// bucket).
#[must_use]
pub fn aggregate(results: &[MatchResult]) -> Statistics {
    let mut role_counts_by_strategy: HashMap<String, HashMap<Role, u32>> = HashMap::new();
    let mut total_passes = 0u64;
    let mut total_voluntary_passes = 0u64;

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
        total_passes += u64::from(result.pass_count);
        total_voluntary_passes += u64::from(result.voluntary_pass_count);
    }

    let voluntary_pass_rate = if total_passes == 0 {
        0.0
    } else {
        // Precision loss only matters above 2^53 passes — far beyond any
        // simulated batch this project runs.
        #[allow(clippy::cast_precision_loss)]
        {
            total_voluntary_passes as f64 / total_passes as f64
        }
    };

    Statistics {
        matches_played: results.len(),
        role_counts_by_strategy,
        voluntary_pass_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(strategy_names: Vec<&str>, roles: Vec<Role>, pass_count: u32, voluntary: u32) -> MatchResult {
        MatchResult {
            player_count: u8::try_from(strategy_names.len()).unwrap(),
            strategy_names: strategy_names.into_iter().map(String::from).collect(),
            role_history: vec![roles],
            trick_count: 0,
            pass_count,
            voluntary_pass_count: voluntary,
        }
    }

    #[test]
    fn aggregate_counts_matches_and_roles_by_strategy() {
        let results = vec![
            result(
                vec!["LowestLegal", "GreedyHighest"],
                vec![Role::President, Role::Arschloch],
                10,
                2,
            ),
            result(
                vec!["LowestLegal", "GreedyHighest"],
                vec![Role::Arschloch, Role::President],
                10,
                8,
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
        let results = vec![result(vec!["LowestLegal"], vec![Role::President], 0, 0)];
        let stats = aggregate(&results);
        assert_eq!(stats.voluntary_pass_rate, 0.0);
    }
}
```

- [ ] **Step 4: Replace `sim/src/lib.rs`**

```rust
//! Strategies, the multi-round match runner, and result/statistics types
//! built on top of `engine`'s single-round primitives. See
//! docs/ARCHITECTURE.md, "sim".

pub mod match_config;
pub mod match_result;
pub mod match_runner;
pub mod statistics;
pub mod strategies;
pub mod strategy;

pub use match_config::MatchConfig;
pub use match_result::MatchResult;
pub use match_runner::{run_batch, run_match};
pub use statistics::{aggregate, Statistics};
pub use strategies::{GreedyHighest, LowestLegal, RandomLegal};
pub use strategy::Strategy;
```

- [ ] **Step 5: Write `sim/tests/small_batch.rs`**

```rust
//! A small batch of matches across all three baseline strategies,
//! asserting the statistical invariants docs/ARCHITECTURE.md calls for:
//! well-shaped role history, sane aggregate counts, and a JSON round-trip.

use std::sync::Arc;

use engine::{DeckVariant, DuplicateRule};
use sim::{aggregate, run_batch, run_match, GreedyHighest, LowestLegal, MatchConfig, RandomLegal, Strategy};

fn baseline_strategies() -> Vec<Arc<dyn Strategy>> {
    vec![
        Arc::new(LowestLegal),
        Arc::new(RandomLegal),
        Arc::new(GreedyHighest),
        Arc::new(RandomLegal),
    ]
}

#[test]
fn a_small_batch_produces_well_shaped_results() {
    let strategies = baseline_strategies();
    let configs: Vec<MatchConfig> = (0..50)
        .map(|seed| MatchConfig {
            player_count: 4,
            deck_variant: DeckVariant::Single,
            duplicate_rule: DuplicateRule::FirstDealtWins,
            rounds: 5,
            seed,
        })
        .collect();

    let results = run_batch(&configs, &strategies);
    assert_eq!(results.len(), 50);

    for result in &results {
        assert_eq!(result.role_history.len(), 5);
        for round_roles in &result.role_history {
            assert_eq!(round_roles.len(), 4);
        }
        assert_eq!(result.strategy_names.len(), 4);
    }

    let stats = aggregate(&results);
    assert_eq!(stats.matches_played, 50);
    let total_role_assignments: u32 = stats
        .role_counts_by_strategy
        .values()
        .flat_map(|counts| counts.values())
        .sum();
    assert_eq!(total_role_assignments, 50 * 5 * 4);
    assert!((0.0..=1.0).contains(&stats.voluntary_pass_rate));

    let json = serde_json::to_string(&stats).expect("Statistics serializes to JSON");
    assert!(json.contains("matches_played"));
}

#[test]
fn identical_seeds_produce_identical_results() {
    let strategies = baseline_strategies();
    let config = MatchConfig {
        player_count: 4,
        deck_variant: DeckVariant::Single,
        duplicate_rule: DuplicateRule::FirstDealtWins,
        rounds: 3,
        seed: 42,
    };
    let first = run_match(&config, &strategies);
    let second = run_match(&config, &strategies);
    assert_eq!(first.role_history, second.role_history);
    assert_eq!(first.trick_count, second.trick_count);
    assert_eq!(first.pass_count, second.pass_count);
}
```

- [ ] **Step 6: Run tests**

```bash
cargo test -p sim
```
Expected: all `sim` unit tests plus these 2 new integration tests pass.

- [ ] **Step 7: Run the full workspace gate**

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```
Expected: all clean — this is the phase-done gate from
`docs/CODING_GUIDELINES.md`.

- [ ] **Step 8: Commit**

```bash
git add sim/Cargo.toml sim/src/lib.rs sim/src/match_runner.rs sim/src/statistics.rs sim/tests/small_batch.rs
git commit -m "sim: add Statistics, run_batch, and small-batch integration tests"
```

---

### Task 8: Docs accuracy pass

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/ROADMAP.md`

**Interfaces:** none — documentation only, no code.

- [ ] **Step 1: Fix `docs/ROADMAP.md`'s stale phase marker**

It currently reads `## Phase 0 — Foundations (current)` even though
Phases 0 and 1 are both merged to `main`. Change the Phase 0 heading to
drop `(current)`, and add `(current)` to the Phase 2 heading instead:

```markdown
## Phase 0 — Foundations
```
```markdown
## Phase 2 — Strategies & simulation runner (current)
```

- [ ] **Step 2: Fix `docs/ARCHITECTURE.md`'s `sim` section**

Its `Strategy` trait sketch (`choose_play(&self, hand, legal_moves) ->
Move` and `choose_exchange_cards(&self, hand, count) -> Vec<Card>`) no
longer matches what this phase built. Replace that paragraph with:

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
```

Also confirm the paragraph above it (`A match runner that drives
engine's state machine...`) still reads accurately against
`match_runner::run_match`/`run_batch` as built; adjust wording only if it
has drifted, without expanding scope beyond this phase's actual shape.

- [ ] **Step 3: Commit**

```bash
git add docs/ARCHITECTURE.md docs/ROADMAP.md
git commit -m "docs: correct ROADMAP phase marker and ARCHITECTURE's sim section"
```

---

## Notes for the final whole-branch review

- Confirm Task 8's doc edits still match the final shipped shape of
  `sim` (in case a fix round in an earlier task changed a signature).
- Confirm `docs/ROADMAP.md`'s Phase 3 entry still matches what `cli` will
  need from `sim` given what actually got built.
