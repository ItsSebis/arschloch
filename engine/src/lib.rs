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
