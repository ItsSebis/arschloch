//! Pure Arschloch game rules. See docs/RULES.md for the authoritative
//! ruleset and docs/ARCHITECTURE.md for how this crate fits into the
//! workspace.

pub mod card;

pub use card::{Card, DeckVariant, DuplicateRule, Rank, Suit};
