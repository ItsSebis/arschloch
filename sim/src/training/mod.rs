//! Training evolved players: fitness evaluation, the run's files and
//! event stream, and the generational loop. See
//! docs/superpowers/specs/2026-10-08-neat-engine-design.md, sections 6-8.

pub mod config;
pub mod decisions;
pub mod evaluate;
pub mod events;
pub mod run_dir;
pub mod trainer;

pub use config::{DeckChoice, DuplicateChoice, TrainConfig};
pub use decisions::{record_decisions, DecisionFile, DecisionRecord};
pub use evaluate::{evaluate, match_seed, role_score, Opponents, Score, TableSpec};
pub use events::{Event, GenerationEvent, RunEnd, RunStart, ScoreStat, SCHEMA_VERSION};
pub use run_dir::{load_config, HallMember, RunDir, TrainError};
pub use trainer::{Opponent, TrainObserver, Trainer};
