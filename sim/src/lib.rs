//! Strategies, the multi-round match runner, and result/statistics types
//! built on top of `engine`'s single-round primitives. See
//! docs/ARCHITECTURE.md, "sim".

pub mod hand_reading;
pub mod match_config;
pub mod match_result;
pub mod match_runner;
pub mod statistics;
pub mod strategies;
pub mod strategy;
pub mod training;

pub use hand_reading::{read_pass_ceilings, PassCeilings};
pub use match_config::MatchConfig;
pub use match_result::MatchResult;
pub use match_runner::{run_batch, run_match};
pub use statistics::{aggregate, RoleRetention, Statistics};
pub use strategies::{
    Adaptive, AdaptiveConfig, CardCounter, DenialMode, EndgameDenial, GenomeFile, GenomeFileError,
    GreedyHighest, HoldBackPairs, LowestLegal, NeatStrategy, RandomLegal, ScoredCandidate,
    TurnSummary, FEATURE_COUNT, FEATURE_NAMES, FEATURE_SET_VERSION,
};
pub use strategy::{OpponentHand, Strategy, TurnContext};
