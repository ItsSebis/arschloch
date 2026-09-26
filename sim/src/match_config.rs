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
