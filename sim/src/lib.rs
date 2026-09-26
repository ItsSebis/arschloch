//! Strategies, the multi-round match runner, and result/statistics types
//! built on top of `engine`'s single-round primitives. See
//! docs/ARCHITECTURE.md, "sim".

pub mod strategies;
pub mod strategy;

pub use strategies::{GreedyHighest, LowestLegal, RandomLegal};
pub use strategy::Strategy;
