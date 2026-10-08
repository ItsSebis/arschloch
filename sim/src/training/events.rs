//! The training event stream: one JSON object per line in
//! `events.jsonl`. Everything the terminal, the dashboard and later
//! analysis show comes from these events, so they cannot disagree.
//! Changes must be additive, with `SCHEMA_VERSION` bumped for any
//! removal or rename.

use neat::SpeciesStats;
use serde::{Deserialize, Serialize};

use super::config::TrainConfig;
use super::evaluate::Score;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    RunStart(Box<RunStart>),
    Generation(Box<GenerationEvent>),
    RunEnd(RunEnd),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunStart {
    pub schema_version: u32,
    pub config: TrainConfig,
    /// Display names of the opponent pool, in pool order.
    pub opponents: Vec<String>,
    pub feature_names: Vec<String>,
    /// `Some(generation)` when this start is a resume.
    pub resumed_from_generation: Option<u32>,
}

/// A candidate's score with the statistics around it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreStat {
    pub mean: f64,
    pub std_error: f64,
    pub matches: usize,
    /// Finishing places, best first, over every round played.
    pub placements: Vec<u64>,
}

impl From<Score> for ScoreStat {
    fn from(score: Score) -> Self {
        Self {
            mean: score.mean,
            std_error: score.std_error,
            matches: score.matches,
            placements: score.placements,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitnessStats {
    pub best: f64,
    pub mean: f64,
    pub median: f64,
    pub min: f64,
    pub std_dev: f64,
    /// Ten equal-width buckets from `min` to `best`.
    pub histogram: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChampionStats {
    /// The champion's fitness in the generation that selected it.
    pub train_fitness: f64,
    /// Its score against the mixed pool on a fixed set of matches that
    /// training never plays and that is the same every generation, so
    /// champions of different generations are compared on the same deals
    /// (the training fitness is inflated by selection).
    pub reeval: ScoreStat,
    /// Present only for a new best champion: its score on a second fixed,
    /// held-out set of matches that was *not* used to pick it. This is
    /// the number to believe: choosing the maximum over many
    /// generations inflates `reeval`.
    #[serde(default)]
    pub heldout: Option<ScoreStat>,
    pub hidden_nodes: usize,
    pub enabled_connections: usize,
    /// File name (inside the run directory) of this champion's genome.
    pub genome_file: String,
    pub is_new_best: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpponentStat {
    pub name: String,
    /// The champion against tables made only of this opponent.
    pub score: ScoreStat,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Complexity {
    pub mean_hidden_nodes: f64,
    pub mean_enabled_connections: f64,
    pub innovation_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationEvent {
    pub generation: u32,
    /// Wall time since the run was first started (summed across resumes).
    pub elapsed_secs: f64,
    pub generation_secs: f64,
    /// Rounds played this generation, evaluation plus re-evaluation.
    pub rounds_evaluated: u64,
    pub total_rounds: u64,
    pub rounds_per_sec: f64,
    pub fitness: FitnessStats,
    pub champion: ChampionStats,
    pub opponents: Vec<OpponentStat>,
    pub species: Vec<SpeciesStats>,
    pub compatibility_threshold: f64,
    pub complexity: Complexity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunEnd {
    pub generations_completed: u32,
    pub best_generation: Option<u32>,
    /// The best champion's score on the fixed set it was selected on.
    pub best_reeval: Option<ScoreStat>,
    /// The best champion's score on held-out matches (see
    /// `ChampionStats::heldout`).
    #[serde(default)]
    pub best_heldout: Option<ScoreStat>,
    pub elapsed_secs: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stat() -> ScoreStat {
        ScoreStat {
            mean: 0.5,
            std_error: 0.02,
            matches: 10,
            placements: vec![5, 3, 1, 1],
        }
    }

    #[test]
    fn events_are_tagged_by_type_and_round_trip() {
        let end = Event::RunEnd(RunEnd {
            generations_completed: 3,
            best_generation: Some(2),
            best_reeval: Some(stat()),
            best_heldout: Some(stat()),
            elapsed_secs: 1.5,
        });
        let line = serde_json::to_string(&end).unwrap();
        assert!(line.contains(r#""type":"run_end""#), "{line}");
        assert!(!line.contains('\n'), "one event is one line");
        assert_eq!(serde_json::from_str::<Event>(&line).unwrap(), end);
    }

    #[test]
    fn a_score_converts_to_a_stat_without_losing_anything() {
        let score = Score {
            mean: -0.25,
            std_error: 0.1,
            matches: 7,
            placements: vec![1, 2, 3, 4],
        };
        let stat = ScoreStat::from(score.clone());
        assert_eq!(
            (stat.mean, stat.std_error, stat.matches),
            (score.mean, score.std_error, score.matches)
        );
        assert_eq!(stat.placements, score.placements);
    }
}
