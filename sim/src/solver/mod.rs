//! Exact endgame search (docs/superpowers/specs/2026-10-16-endgame-solver-design.md).
//!
//! From one seat's point of view: enumerate every distribution of the
//! unseen cards the other seats could hold (`deals`), solve each one exactly
//! (`search`, max^n) and average per move (`report`).

pub mod config;
pub mod deals;
pub mod gap;
pub mod notation;
pub mod observation;

pub mod leaf;
pub mod report;
pub mod sample;
pub mod sampler;
pub mod search;

pub use observation::Observation;
pub use report::{analyze, MoveReport, MoveStats, SolveError};
pub use sampler::{sample_analyze, LeafSpec, SamplerConfig};

/// Limits of one solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolverConfig {
    /// Solve only when at most this many cards are unseen.
    pub max_unseen: usize,
    /// Search nodes allowed per decision, over all deals.
    pub node_budget: u64,
    /// How the opponents' pass ceilings restrict the deals (exact and sampled).
    pub ceilings: CeilingMode,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_unseen: 6,
            node_budget: 2_000_000,
            ceilings: CeilingMode::Hard,
        }
    }
}

/// What a pass ceiling ("seat 2 passed on a 9: it holds nothing above") does to
/// the deals the solver considers. A seat that passes on purpose while holding a
/// beating card makes its true hand look impossible under `Hard`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CeilingMode {
    /// A deal must respect every ceiling; none left means the fallback plays.
    #[default]
    Hard,
    /// Respect them, but when no deal does, consider the deals without them.
    Soft,
    /// Ignore them: every split of the unseen cards counts.
    Off,
}

impl CeilingMode {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hard => "hard",
            Self::Soft => "soft",
            Self::Off => "off",
        }
    }
}

impl std::str::FromStr for CeilingMode {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        match text.trim() {
            "hard" => Ok(Self::Hard),
            "soft" => Ok(Self::Soft),
            "off" => Ok(Self::Off),
            other => Err(format!(
                "ceilings: `{other}` is not one of hard, soft or off"
            )),
        }
    }
}
