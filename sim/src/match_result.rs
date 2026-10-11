//! The outcome of one simulated match: role history and per-seat
//! move-shape counters consumed by `crate::statistics::aggregate`.

use engine::Role;

#[derive(Debug, Clone, serde::Serialize)]
pub struct MatchResult {
    pub player_count: u8,
    /// One entry per seat (index = `SeatId`).
    pub strategy_names: Vec<String>,
    /// One entry per round played, each seat-indexed.
    pub role_history: Vec<Vec<Role>>,
    pub trick_count: u32,
    /// One entry per seat: how many passes that seat submitted across the
    /// whole match.
    pub pass_counts: Vec<u32>,
    /// One entry per seat: how many of that seat's passes were voluntary
    /// (a `Move::Play` was also legal — see "Strategy diversification" in
    /// docs/ROADMAP.md).
    pub voluntary_pass_counts: Vec<u32>,
    /// Round-1 hand features per seat, after the deal and before the
    /// exchange; only present when `RunOptions::record_deal_features` was set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_hand_features: Option<Vec<crate::hand_features::HandFeatures>>,
}
