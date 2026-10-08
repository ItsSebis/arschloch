//! `NeatStrategy`: plays by scoring every legal move with an evolved
//! neural network and choosing the highest score. See
//! docs/superpowers/specs/2026-10-08-neat-engine-design.md, section 4.

mod features;
mod genome_file;

pub use features::{TurnSummary, FEATURE_COUNT, FEATURE_NAMES};
pub use genome_file::{GenomeFile, GenomeFileError, FORMAT_VERSION};
