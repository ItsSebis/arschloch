//! Pure Arschloch game rules. See docs/RULES.md for the authoritative
//! ruleset and docs/ARCHITECTURE.md for how this crate fits into the
//! workspace.

pub mod card;
pub mod combo;
pub mod deal;
pub mod deck;
pub mod exchange;
mod legal_moves;
pub mod role;
pub mod round;
mod trick;

pub use card::{rank_groups, Card, DeckVariant, DuplicateRule, Rank, Suit};
pub use combo::Combo;
pub use deal::{deal, lowest_card_holder};
pub use deck::standard_deck;
pub use exchange::{
    exchange, exchange_with_rule, exchange_with_selection, ExchangeError, ExchangeRule,
};
pub use role::{assign_roles, exchange_counts_for_player_count, roles_for_player_count, Role};
pub use round::{Move, MoveError, Round};
pub use trick::PassRule;

/// A seat's position at the table (0-indexed). Table sizes are 3-6, so
/// `u8` matches `player_count`'s type used throughout this crate.
pub type SeatId = u8;
