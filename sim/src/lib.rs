//! Strategies, the multi-round match runner, and result/statistics types
//! built on top of `engine`'s single-round primitives. See
//! docs/ARCHITECTURE.md, "sim".

pub mod match_config;
pub mod match_result;
pub mod match_runner;
pub mod statistics;
pub mod strategies;
pub mod strategy;

pub use match_config::MatchConfig;
pub use match_result::MatchResult;
pub use match_runner::{run_batch, run_match};
pub use statistics::{aggregate, RoleRetention, Statistics};
pub use strategies::{GreedyHighest, LowestLegal, RandomLegal};
pub use strategy::Strategy;
