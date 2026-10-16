//! Exact endgame search (docs/superpowers/specs/2026-10-16-endgame-solver-design.md).
//!
//! From one seat's point of view: enumerate every distribution of the
//! unseen cards the other seats could hold (`deals`), solve each one exactly
//! (`search`, max^n) and average per move (`report`).

pub mod notation;
pub mod observation;

pub use observation::Observation;
