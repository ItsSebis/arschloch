//! Exact endgame search (docs/superpowers/specs/2026-10-16-endgame-solver-design.md).
//!
//! From one seat's point of view: enumerate every distribution of the
//! unseen cards the other seats could hold (`deals`), solve each one exactly
//! (`search`, max^n) and average per move (`report`).

pub mod config;
pub mod deals;
pub mod notation;
pub mod observation;

pub mod report;
pub mod search;

pub use observation::Observation;
pub use report::{analyze, MoveReport, MoveStats, SolveError};

/// Limits of one solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolverConfig {
    /// Solve only when at most this many cards are unseen.
    pub max_unseen: usize,
    /// Search nodes allowed per decision, over all deals.
    pub node_budget: u64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_unseen: 6,
            node_budget: 2_000_000,
        }
    }
}
