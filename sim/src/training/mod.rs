//! Training evolved players: fitness evaluation, the run's files and
//! event stream, and the generational loop. See
//! docs/superpowers/specs/2026-10-08-neat-engine-design.md, sections 6-8.

pub mod evaluate;

pub use evaluate::{evaluate, match_seed, role_score, Opponents, Score, TableSpec};
