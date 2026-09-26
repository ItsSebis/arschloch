# Phase 0 Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold the Cargo workspace and implement the `engine` crate's
foundational domain types (cards, suits, ranks, roles, exchange-count
tables, and combo legality), fully unit-tested, with nothing else built
yet.

**Architecture:** A 4-crate Cargo workspace (`engine`, `sim`, `cli`,
`web`) with a one-way dependency direction (`cli`/`web` → `sim` → `engine`).
This phase only implements `engine`; `sim`, `cli`, `web` are empty
placeholder crates so the workspace builds as a whole from day one.

**Tech Stack:** Rust (stable, 1.98.1 installed), Cargo workspaces, no
external dependencies in `engine` for this phase.

**Spec:** `docs/superpowers/specs/2026-09-26-arschloch-simulator-design.md`
(also see `docs/RULES.md` for the exact rules being encoded and
`docs/ARCHITECTURE.md` for the crate layout).

## Global Constraints

- `engine` has no project-internal or external dependencies (see
  `docs/ARCHITECTURE.md`, "`engine` stays dependency-free").
- Suit order (house rule): `Diamonds < Hearts < Spades < Clubs` (see
  `docs/RULES.md`, "Card Ranking").
- Rank order: `Two < Three < ... < King < Ace` (standard poker order).
- Card comparison order: rank, then suit, then (only for true duplicates
  in the double-deck variant) the configurable `DuplicateRule` using each
  card's `deal_index` (see `docs/RULES.md`, "Duplicate cards").
- Role tables and exchange-count tables must match `docs/RULES.md`
  exactly for player counts 3, 4, 5, 6 (including the explicitly-flagged
  inferred 3-player row: President ↔ Arschloch exchange 1 card,
  Dorftrottel exchanges 0).
- A `Combo` is one or more cards of equal rank; a combo only beats another
  of the **same size** with a **strictly higher** top card. No
  straights/bombs (out of scope per `docs/RULES.md`, "Deferred /
  Out-of-scope Rules").
- Every phase must pass, in order: `cargo fmt --check`, `cargo clippy
  --workspace --all-targets -- -D warnings`, `cargo test --workspace`
  (see `docs/CODING_GUIDELINES.md`, "Before finishing a phase").
- Workspace-level lints: `unsafe_code = "forbid"`,
  `clippy::pedantic = "warn"`, with `clippy::module_name_repetitions`
  allowed workspace-wide (this project's convention is one primary type
  per module, named after the module — see `docs/ARCHITECTURE.md`).

---

### Task 1: Workspace scaffold

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `engine/Cargo.toml`
- Create: `engine/src/lib.rs`
- Create: `sim/Cargo.toml`
- Create: `sim/src/lib.rs`
- Create: `cli/Cargo.toml`
- Create: `cli/src/main.rs`
- Create: `web/Cargo.toml`
- Create: `web/src/lib.rs`

**Interfaces:**
- Produces: a workspace that builds via `cargo build --workspace`, with
  `engine` as an empty-but-valid crate that Task 2+ will fill in.

- [ ] **Step 1: Create the workspace root `Cargo.toml`**

```toml
[workspace]
resolver = "2"
members = ["engine", "sim", "cli", "web"]

[workspace.package]
edition = "2021"
version = "0.1.0"
license = "MIT"

[workspace.lints.rust]
unsafe_code = "forbid"

[workspace.lints.clippy]
pedantic = { level = "warn", priority = -1 }
# One primary type per module, named after the module, is this
# project's convention (see docs/ARCHITECTURE.md) — this lint would
# otherwise fire on every module.
module_name_repetitions = "allow"
```

- [ ] **Step 2: Create `engine/Cargo.toml`**

```toml
[package]
name = "engine"
version.workspace = true
edition.workspace = true
license.workspace = true

[lints]
workspace = true

[dependencies]
```

- [ ] **Step 3: Create a placeholder `engine/src/lib.rs`**

```rust
//! Pure Arschloch game rules. See docs/RULES.md for the authoritative
//! ruleset and docs/ARCHITECTURE.md for how this crate fits into the
//! workspace. Filled in by Tasks 2-4 of this plan.
```

- [ ] **Step 4: Create `sim/Cargo.toml`, `cli/Cargo.toml`, `web/Cargo.toml`**

`sim/Cargo.toml`:

```toml
[package]
name = "sim"
version.workspace = true
edition.workspace = true
license.workspace = true

[lints]
workspace = true

[dependencies]
```

`cli/Cargo.toml`:

```toml
[package]
name = "cli"
version.workspace = true
edition.workspace = true
license.workspace = true

[lints]
workspace = true

[dependencies]
```

`web/Cargo.toml`:

```toml
[package]
name = "web"
version.workspace = true
edition.workspace = true
license.workspace = true

[lints]
workspace = true

[dependencies]
```

- [ ] **Step 5: Create placeholder crate bodies**

`sim/src/lib.rs`:

```rust
//! Placeholder crate. The simulation runner, strategies, and statistics
//! begin in Phase 2 (see docs/ROADMAP.md).
```

`cli/src/main.rs`:

```rust
//! Placeholder binary. CLI wiring begins in Phase 3 (see docs/ROADMAP.md).

fn main() {
    println!("arschloch CLI: not implemented yet (see docs/ROADMAP.md, Phase 3)");
}
```

`web/src/lib.rs`:

```rust
//! Placeholder crate. The web interface begins in Phase 6 (see docs/ROADMAP.md).
```

- [ ] **Step 6: Verify the workspace builds**

Run: `cargo build --workspace`
Expected: builds successfully, no errors (warnings about empty/unused
crates are not expected since each placeholder has no dead code, just doc
comments).

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock engine/Cargo.toml engine/src/lib.rs \
  sim/Cargo.toml sim/src/lib.rs cli/Cargo.toml cli/src/main.rs \
  web/Cargo.toml web/src/lib.rs
git commit -m "Scaffold Cargo workspace: engine, sim, cli, web crates"
```

---

### Task 2: `engine::card` — Suit, Rank, Card, DeckVariant, DuplicateRule

**Files:**
- Create: `engine/src/card.rs`
- Modify: `engine/src/lib.rs`

**Interfaces:**
- Consumes: nothing (first real module).
- Produces: `Suit`, `Rank`, `Card { rank, suit, deal_index }`,
  `Card::new(rank, suit, deal_index) -> Card`,
  `Card::compare(&self, other: &Card, duplicate_rule: DuplicateRule) -> std::cmp::Ordering`,
  `DeckVariant` (`Single` | `Double`), `DuplicateRule`
  (`FirstDealtWins` | `LastDealtWins`). Used by Task 4 (`combo.rs`).

- [ ] **Step 1: Write `engine/src/card.rs` with its own tests**

```rust
//! Card, suit, and rank types, including the project's house-rule suit
//! ordering and double-deck duplicate-card tiebreak. See docs/RULES.md,
//! "Card Ranking" and "Duplicate cards".

use std::cmp::Ordering;

/// Suit ranking is a house rule for this project (see docs/RULES.md):
/// Diamonds < Hearts < Spades < Clubs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Suit {
    Diamonds,
    Hearts,
    Spades,
    Clubs,
}

/// Standard poker rank order, low to high.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Rank {
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Jack,
    Queen,
    King,
    Ace,
}

/// Which physical deck(s) a match is played with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeckVariant {
    Single,
    Double,
}

/// Resolves ties between two cards that are identical in rank and suit,
/// which can only happen in the `Double` deck variant. See docs/RULES.md,
/// "Duplicate cards".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuplicateRule {
    FirstDealtWins,
    LastDealtWins,
}

/// A single playing card. `deal_index` disambiguates true duplicate
/// rank+suit pairs in the double-deck variant (the copy's position in
/// deal order) and is otherwise ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Card {
    pub rank: Rank,
    pub suit: Suit,
    pub deal_index: u8,
}

impl Card {
    #[must_use]
    pub fn new(rank: Rank, suit: Suit, deal_index: u8) -> Self {
        Self {
            rank,
            suit,
            deal_index,
        }
    }

    /// Total order over cards for a given match's duplicate-tiebreak
    /// rule. Rank is compared first, then suit, then (only when rank and
    /// suit are both equal) deal order.
    #[must_use]
    pub fn compare(&self, other: &Card, duplicate_rule: DuplicateRule) -> Ordering {
        self.rank
            .cmp(&other.rank)
            .then_with(|| self.suit.cmp(&other.suit))
            .then_with(|| match duplicate_rule {
                DuplicateRule::FirstDealtWins => other.deal_index.cmp(&self.deal_index),
                DuplicateRule::LastDealtWins => self.deal_index.cmp(&other.deal_index),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_orders_low_to_high() {
        assert!(Rank::Two < Rank::Three);
        assert!(Rank::King < Rank::Ace);
    }

    #[test]
    fn suit_orders_per_house_rule() {
        assert!(Suit::Diamonds < Suit::Hearts);
        assert!(Suit::Hearts < Suit::Spades);
        assert!(Suit::Spades < Suit::Clubs);
    }

    #[test]
    fn higher_rank_beats_lower_rank_regardless_of_suit() {
        let low = Card::new(Rank::Seven, Suit::Clubs, 0);
        let high = Card::new(Rank::Eight, Suit::Diamonds, 0);
        assert_eq!(
            low.compare(&high, DuplicateRule::FirstDealtWins),
            Ordering::Less
        );
    }

    #[test]
    fn equal_rank_breaks_tie_by_suit() {
        let diamonds = Card::new(Rank::Nine, Suit::Diamonds, 0);
        let clubs = Card::new(Rank::Nine, Suit::Clubs, 0);
        assert_eq!(
            diamonds.compare(&clubs, DuplicateRule::FirstDealtWins),
            Ordering::Less
        );
    }

    #[test]
    fn true_duplicate_first_dealt_wins() {
        let first = Card::new(Rank::Ten, Suit::Hearts, 0);
        let second = Card::new(Rank::Ten, Suit::Hearts, 1);
        assert_eq!(
            first.compare(&second, DuplicateRule::FirstDealtWins),
            Ordering::Greater
        );
        assert_eq!(
            second.compare(&first, DuplicateRule::FirstDealtWins),
            Ordering::Less
        );
    }

    #[test]
    fn true_duplicate_last_dealt_wins() {
        let first = Card::new(Rank::Ten, Suit::Hearts, 0);
        let second = Card::new(Rank::Ten, Suit::Hearts, 1);
        assert_eq!(
            first.compare(&second, DuplicateRule::LastDealtWins),
            Ordering::Less
        );
        assert_eq!(
            second.compare(&first, DuplicateRule::LastDealtWins),
            Ordering::Greater
        );
    }
}
```

- [ ] **Step 2: Wire the module into `engine/src/lib.rs`**

```rust
//! Pure Arschloch game rules. See docs/RULES.md for the authoritative
//! ruleset and docs/ARCHITECTURE.md for how this crate fits into the
//! workspace.

pub mod card;

pub use card::{Card, DeckVariant, DuplicateRule, Rank, Suit};
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p engine`
Expected: all `card` tests pass (6 tests).

- [ ] **Step 4: Commit**

```bash
git add engine/src/card.rs engine/src/lib.rs
git commit -m "engine: add Card/Suit/Rank/DeckVariant/DuplicateRule types"
```

---

### Task 3: `engine::role` — Role, role table, exchange-count table

**Files:**
- Create: `engine/src/role.rs`
- Modify: `engine/src/lib.rs`

**Interfaces:**
- Consumes: nothing (independent of `card.rs`).
- Produces: `Role` enum, `roles_for_player_count(player_count: u8) -> Option<&'static [Role]>`,
  `exchange_counts_for_player_count(player_count: u8) -> Option<&'static [u8]>`.
  Not consumed by any other Phase 0 task, but is the basis for Phase 1's
  round-end role assignment and card exchange.

- [ ] **Step 1: Write `engine/src/role.rs` with its own tests**

```rust
//! Roles and the per-player-count role/exchange-count tables from
//! docs/RULES.md, "Roles" and "Card Exchange (\"Drücken\")".

/// A player's role at the end of a round. Variants are listed here from
/// highest to lowest across all table sizes; which subset applies to a
/// given table is determined by `roles_for_player_count`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    President,
    Vize,
    Offizier,
    Dorftrottel,
    Dummkopf,
    ViceArschloch,
    Arschloch,
}

/// The ordered list of roles for a table of `player_count` seats, highest
/// role first, as defined in docs/RULES.md, "Roles". Returns `None` for
/// unsupported table sizes (anything other than 3, 4, 5, or 6).
#[must_use]
pub fn roles_for_player_count(player_count: u8) -> Option<&'static [Role]> {
    match player_count {
        3 => Some(&[Role::President, Role::Dorftrottel, Role::Arschloch]),
        4 => Some(&[
            Role::President,
            Role::Vize,
            Role::ViceArschloch,
            Role::Arschloch,
        ]),
        5 => Some(&[
            Role::President,
            Role::Vize,
            Role::Dorftrottel,
            Role::ViceArschloch,
            Role::Arschloch,
        ]),
        6 => Some(&[
            Role::President,
            Role::Vize,
            Role::Offizier,
            Role::Dummkopf,
            Role::ViceArschloch,
            Role::Arschloch,
        ]),
        _ => None,
    }
}

/// How many cards each role-pair exchanges before a new round, indexed
/// from the outermost pair (President/Arschloch) inward, as defined in
/// docs/RULES.md, "Card Exchange". A lone unpaired middle role
/// (Dorftrottel at 3 or 5 players) exchanges 0 cards and appears as a
/// trailing `0` in the table with no partner.
#[must_use]
pub fn exchange_counts_for_player_count(player_count: u8) -> Option<&'static [u8]> {
    match player_count {
        3 => Some(&[1, 0]),
        4 => Some(&[2, 1]),
        5 => Some(&[2, 1, 0]),
        6 => Some(&[3, 2, 1]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_player_roles_match_rules_doc() {
        assert_eq!(
            roles_for_player_count(3),
            Some(&[Role::President, Role::Dorftrottel, Role::Arschloch][..])
        );
    }

    #[test]
    fn four_player_roles_match_rules_doc() {
        assert_eq!(
            roles_for_player_count(4),
            Some(&[Role::President, Role::Vize, Role::ViceArschloch, Role::Arschloch][..])
        );
    }

    #[test]
    fn five_player_roles_match_rules_doc() {
        assert_eq!(
            roles_for_player_count(5),
            Some(
                &[
                    Role::President,
                    Role::Vize,
                    Role::Dorftrottel,
                    Role::ViceArschloch,
                    Role::Arschloch,
                ][..]
            )
        );
    }

    #[test]
    fn six_player_roles_match_rules_doc() {
        assert_eq!(
            roles_for_player_count(6),
            Some(
                &[
                    Role::President,
                    Role::Vize,
                    Role::Offizier,
                    Role::Dummkopf,
                    Role::ViceArschloch,
                    Role::Arschloch,
                ][..]
            )
        );
    }

    #[test]
    fn unsupported_player_count_returns_none() {
        assert_eq!(roles_for_player_count(2), None);
        assert_eq!(roles_for_player_count(7), None);
    }

    #[test]
    fn exchange_counts_match_rules_doc() {
        assert_eq!(exchange_counts_for_player_count(3), Some(&[1, 0][..]));
        assert_eq!(exchange_counts_for_player_count(4), Some(&[2, 1][..]));
        assert_eq!(exchange_counts_for_player_count(5), Some(&[2, 1, 0][..]));
        assert_eq!(exchange_counts_for_player_count(6), Some(&[3, 2, 1][..]));
    }

    #[test]
    fn every_supported_table_size_has_matching_role_and_exchange_lengths() {
        for player_count in [3u8, 4, 5, 6] {
            let roles = roles_for_player_count(player_count).unwrap();
            let exchanges = exchange_counts_for_player_count(player_count).unwrap();
            // Exchange table has one entry per role-pair, plus a
            // trailing 0 for a lone middle role at odd-shaped tables
            // (3 and 5 players); role count is always exchanges.len()
            // entries * 2, +/-1 for a lone middle role.
            let has_lone_middle = roles.len() % 2 == 1;
            let expected_exchange_len = roles.len() / 2 + usize::from(has_lone_middle);
            assert_eq!(exchanges.len(), expected_exchange_len);
        }
    }
}
```

- [ ] **Step 2: Wire the module into `engine/src/lib.rs`**

```rust
//! Pure Arschloch game rules. See docs/RULES.md for the authoritative
//! ruleset and docs/ARCHITECTURE.md for how this crate fits into the
//! workspace.

pub mod card;
pub mod role;

pub use card::{Card, DeckVariant, DuplicateRule, Rank, Suit};
pub use role::{exchange_counts_for_player_count, roles_for_player_count, Role};
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p engine`
Expected: all `card` and `role` tests pass (6 + 7 = 13 tests).

- [ ] **Step 4: Commit**

```bash
git add engine/src/role.rs engine/src/lib.rs
git commit -m "engine: add Role type and per-player-count role/exchange tables"
```

---

### Task 4: `engine::combo` — Combo and legal-play checker

**Files:**
- Create: `engine/src/combo.rs`
- Modify: `engine/src/lib.rs`

**Interfaces:**
- Consumes: `Card`, `DuplicateRule` from `engine::card` (Task 2).
- Produces: `Combo`, `Combo::new(cards: Vec<Card>) -> Option<Combo>`,
  `Combo::size(&self) -> usize`,
  `Combo::beats(&self, previous: &Combo, duplicate_rule: DuplicateRule) -> bool`.
  This is the last Phase 0 type; Phase 1's trick loop will consume it.

- [ ] **Step 1: Write `engine/src/combo.rs` with its own tests**

```rust
//! A `Combo` is a set of cards of equal rank played together (a single,
//! pair, triple, ...). Straights and bombs are out of scope for this
//! phase (see docs/RULES.md, "Deferred / Out-of-scope Rules").

use std::cmp::Ordering;

use crate::card::{Card, DuplicateRule};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Combo {
    cards: Vec<Card>,
}

impl Combo {
    /// Builds a combo from cards that must all share the same rank and
    /// must be non-empty. Returns `None` otherwise.
    #[must_use]
    pub fn new(cards: Vec<Card>) -> Option<Self> {
        let first = cards.first()?;
        if cards.iter().all(|c| c.rank == first.rank) {
            Some(Self { cards })
        } else {
            None
        }
    }

    #[must_use]
    pub fn size(&self) -> usize {
        self.cards.len()
    }

    /// The representative card used for comparison: since every card in
    /// a combo shares a rank, the highest card (by suit/duplicate
    /// tiebreak) stands in for the whole combo.
    fn top_card(&self, duplicate_rule: DuplicateRule) -> &Card {
        self.cards
            .iter()
            .max_by(|a, b| a.compare(b, duplicate_rule))
            .expect("Combo is always constructed with at least one card")
    }

    /// Whether `self`, played on top of `previous`, is a legal follow:
    /// same size, and strictly higher by the match's duplicate-tiebreak
    /// rule.
    #[must_use]
    pub fn beats(&self, previous: &Combo, duplicate_rule: DuplicateRule) -> bool {
        self.size() == previous.size()
            && self
                .top_card(duplicate_rule)
                .compare(previous.top_card(duplicate_rule), duplicate_rule)
                == Ordering::Greater
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
    fn empty_combo_is_rejected() {
        assert!(Combo::new(vec![]).is_none());
    }

    #[test]
    fn mixed_rank_combo_is_rejected() {
        let cards = vec![
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Eight, Suit::Clubs),
        ];
        assert!(Combo::new(cards).is_none());
    }

    #[test]
    fn same_rank_combo_is_accepted() {
        let cards = vec![
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Seven, Suit::Diamonds),
        ];
        assert!(Combo::new(cards).is_some());
    }

    #[test]
    fn higher_rank_combo_beats_lower_rank_combo_of_same_size() {
        let low = Combo::new(vec![card(Rank::Seven, Suit::Clubs)]).unwrap();
        let high = Combo::new(vec![card(Rank::Eight, Suit::Diamonds)]).unwrap();
        assert!(high.beats(&low, DuplicateRule::FirstDealtWins));
        assert!(!low.beats(&high, DuplicateRule::FirstDealtWins));
    }

    #[test]
    fn different_size_combo_never_beats() {
        let single = Combo::new(vec![card(Rank::Nine, Suit::Clubs)]).unwrap();
        let pair = Combo::new(vec![
            card(Rank::Ten, Suit::Clubs),
            card(Rank::Ten, Suit::Diamonds),
        ])
        .unwrap();
        assert!(!pair.beats(&single, DuplicateRule::FirstDealtWins));
        assert!(!single.beats(&pair, DuplicateRule::FirstDealtWins));
    }

    #[test]
    fn equal_rank_combo_broken_by_suit() {
        let low = Combo::new(vec![card(Rank::Nine, Suit::Diamonds)]).unwrap();
        let high = Combo::new(vec![card(Rank::Nine, Suit::Clubs)]).unwrap();
        assert!(high.beats(&low, DuplicateRule::FirstDealtWins));
    }

    #[test]
    fn equal_rank_and_suit_combo_never_beats_itself() {
        let a = Combo::new(vec![card(Rank::Nine, Suit::Clubs)]).unwrap();
        let b = Combo::new(vec![card(Rank::Nine, Suit::Clubs)]).unwrap();
        assert!(!a.beats(&b, DuplicateRule::FirstDealtWins));
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
pub mod role;

pub use card::{Card, DeckVariant, DuplicateRule, Rank, Suit};
pub use combo::Combo;
pub use role::{exchange_counts_for_player_count, roles_for_player_count, Role};
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p engine`
Expected: all `card`, `role`, and `combo` tests pass (6 + 7 + 7 = 20 tests).

- [ ] **Step 4: Commit**

```bash
git add engine/src/combo.rs engine/src/lib.rs
git commit -m "engine: add Combo type and same-rank legal-play checker"
```

---

### Task 5: Phase-done verification gate

**Files:** none (verification only).

**Interfaces:** none — this task only runs the gate from
`docs/CODING_GUIDELINES.md`, "Before finishing a phase", across the whole
workspace, and fixes anything it finds.

- [ ] **Step 1: Check formatting**

Run: `cargo fmt --check`
Expected: no output (already formatted). If it reports diffs, run
`cargo fmt` and re-check.

- [ ] **Step 2: Run clippy with pedantic-as-warn promoted to deny**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: no warnings or errors. If clippy reports something, fix the
underlying code; only add a targeted `#[allow(clippy::lint_name)]` with a
one-line reason comment if the lint genuinely doesn't apply (per
`docs/CODING_GUIDELINES.md`).

- [ ] **Step 3: Run the full test suite**

Run: `cargo test --workspace`
Expected: 20 tests pass, 0 failed (across `engine`; `sim`/`cli`/`web`
have none yet).

- [ ] **Step 4: Commit any fixes from Steps 1-3**

Only if Steps 1-3 required changes:

```bash
git add -A
git commit -m "engine: fix fmt/clippy issues from Phase 0 verification gate"
```

If no changes were needed, skip this step — there's nothing to commit.
