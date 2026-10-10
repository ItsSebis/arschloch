//! Strategies, the multi-round match runner, and result/statistics types
//! built on top of `engine`'s single-round primitives. See
//! docs/ARCHITECTURE.md, "sim".

pub mod duplicate;
pub mod extended_stats;
pub mod hand_features;
pub mod hand_reading;
pub mod linalg;
pub mod match_config;
pub mod match_result;
pub mod match_runner;
pub mod session;
pub mod skill;
pub mod statistics;
pub mod stats_catalog;
pub mod strategies;
pub mod strategy;
pub mod training;

pub use engine::{
    roles_for_player_count, DeckVariant, DuplicateRule, ExchangeRule, PassRule, Role,
};
pub use hand_features::HandFeatures;
pub use hand_reading::{read_pass_ceilings, PassCeilings};
pub use match_config::MatchConfig;
pub use match_result::MatchResult;
pub use match_runner::{
    deal_round_seed, play_out, run_batch, run_match, run_match_with, PlayCounters, RunOptions,
};
pub use statistics::{aggregate, RoleRetention, Statistics};
pub use strategies::{
    Adaptive, AdaptiveConfig, CardCounter, DenialMode, EndgameDenial, GenomeFile, GenomeFileError,
    GreedyHighest, HoldBackPairs, LowestLegal, NeatStrategy, RandomLegal, ScoredCandidate,
    TurnSummary, FEATURE_COUNT, FEATURE_NAMES, FEATURE_SET_VERSION,
};
pub use strategy::{OpponentHand, Strategy, TurnContext};
