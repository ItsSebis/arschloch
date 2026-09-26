//! Pure Arschloch game rules. See docs/RULES.md for the authoritative
//! ruleset and docs/ARCHITECTURE.md for how this crate fits into the
//! workspace.

pub mod card;
pub mod role;

pub use card::{Card, DeckVariant, DuplicateRule, Rank, Suit};
pub use role::{exchange_counts_for_player_count, roles_for_player_count, Role};
