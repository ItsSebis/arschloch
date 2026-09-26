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
        rng: &mut dyn rand::Rng,
    ) -> Move;
}
