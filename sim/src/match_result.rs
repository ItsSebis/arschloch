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
