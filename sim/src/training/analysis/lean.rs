//! The part of an event the analysis reads. Event files from older versions
//! lack fields the full `Event` types require (and may have extra ones), so
//! the loader deserialises into these lean structs instead: only what the
//! analysis uses, everything but the champion's re-evaluation defaulted.

use serde::de::IgnoredAny;
use serde::Deserialize;

use crate::training::events::StageTimings;

/// A score and its standard error.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub struct LeanScore {
    pub mean: f64,
    #[serde(default)]
    pub std_error: f64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LeanChampion {
    /// `None` when the event does not carry it.
    #[serde(default)]
    pub train_fitness: Option<f64>,
    pub reeval: LeanScore,
    #[serde(default)]
    pub heldout: Option<LeanScore>,
    #[serde(default)]
    pub hidden_nodes: usize,
    #[serde(default)]
    pub enabled_connections: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LeanOpponent {
    pub name: String,
    pub score: LeanScore,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct LeanFitness {
    #[serde(default)]
    pub histogram: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub struct LeanRacing {
    #[serde(default)]
    pub raced: usize,
    #[serde(default)]
    pub extra_matches: usize,
    #[serde(default)]
    pub mean_shift: f64,
    #[serde(default)]
    pub raced_final: usize,
}

/// One generation event, reduced to what the analysis reads.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LeanGeneration {
    pub generation: u32,
    pub champion: LeanChampion,
    #[serde(default)]
    pub opponents: Vec<LeanOpponent>,
    #[serde(default)]
    pub fitness: LeanFitness,
    /// Only the number of species is used.
    #[serde(default)]
    pub species: Vec<IgnoredAny>,
    #[serde(default)]
    pub compatibility_threshold: f64,
    #[serde(default)]
    pub racing: Option<LeanRacing>,
    #[serde(default)]
    pub timings: Option<StageTimings>,
}

/// The `run_end` event.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LeanEnd {
    #[serde(default)]
    pub best_generation: Option<u32>,
    #[serde(default)]
    pub best_reeval: Option<LeanScore>,
    #[serde(default)]
    pub best_heldout: Option<LeanScore>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
struct LeanNeat {
    #[serde(default)]
    population_size: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
struct LeanConfig {
    #[serde(default)]
    neat: LeanNeat,
    #[serde(default)]
    matches_per_genome: usize,
}

/// The `run_start` event, reduced to the configuration numbers the racing
/// totals need and the resume marker.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct RunStartInfo {
    #[serde(default)]
    config: LeanConfig,
    /// `Some(generation)` when this start is a resume.
    #[serde(default)]
    pub resumed_from_generation: Option<u32>,
}

impl RunStartInfo {
    /// Genomes per generation (`None` when the event did not say).
    #[must_use]
    pub fn population_size(&self) -> Option<usize> {
        Some(self.config.neat.population_size).filter(|n| *n > 0)
    }

    /// Training matches per genome (`None` when the event did not say).
    #[must_use]
    pub fn matches_per_genome(&self) -> Option<usize> {
        Some(self.config.matches_per_genome).filter(|n| *n > 0)
    }
}

/// One line of the events file.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(clippy::large_enum_variant)]
pub enum LeanEvent {
    RunStart(Box<RunStartInfo>),
    Generation(Box<LeanGeneration>),
    RunEnd(LeanEnd),
}
