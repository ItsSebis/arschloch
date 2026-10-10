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
    /// The run directory whose final population this run started from.
    #[serde(default)]
    pub warm_started_from: Option<String>,
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
    /// Only with a non-zero `skill_weight`: the generation's luck-adjusted
    /// terms (the fitness above is the blend).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skill_term: Option<SkillTermStats>,
}

/// Mean and spread of the luck-adjusted round-1 term over a generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillTermStats {
    pub mean: f64,
    pub std_dev: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChampionStats {
    /// The champion's fitness in the generation that selected it.
    pub train_fitness: f64,
    /// Only with a non-zero `skill_weight`: the champion's luck-adjusted
    /// term in that generation (`train_fitness` is the blend). Champion
    /// choice, re-evaluation and held-out scores use the plain mean role
    /// score either way.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skill_term: Option<f64>,
    /// Its score against the mixed pool on a fixed set of matches that
    /// training never plays, that is the same every generation (so
    /// champions of different generations are compared on the same deals)
    /// and that played no part in choosing it (candidates are ranked on
    /// separate matches). The training fitness, by contrast, is inflated
    /// by selection.
    pub reeval: ScoreStat,
    /// Present only for a new best champion: its score on a second fixed,
    /// held-out set of matches that was *not* used to pick it. This is
    /// the number to believe: choosing the maximum over many
    /// generations inflates `reeval`.
    #[serde(default)]
    pub heldout: Option<ScoreStat>,
    /// Where the champion ranked by *training* fitness among this
    /// generation's genomes (0 = it was the training best). Non-zero means
    /// the fixed-match re-evaluation overruled a lucky training score.
    #[serde(default)]
    pub training_rank: usize,
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

/// Wall-clock seconds spent in each stage of one generation (measured with
/// `std::time::Instant`, so they differ from run to run and are not part
/// of any determinism comparison, like `elapsed_secs`). The mixed,
/// per-opponent and hall-of-fame re-evaluations run at the same time, so
/// their times overlap; the other stages run one after the other.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct StageTimings {
    /// Every genome played its training matches.
    pub training_evaluation: f64,
    /// The champion candidates played the selection matches.
    pub champion_selection: f64,
    /// Speciation, crossover, mutation: `Population::advance`.
    pub speciation_and_reproduction: f64,
    /// The champion against the mixed fixed pool on the fixed matches.
    pub reevaluation_mixed: f64,
    /// The champion against each fixed opponent alone.
    pub reevaluation_per_opponent: f64,
    /// The champion against the hall of fame, plus updating the hall.
    pub hall_of_fame: f64,
    /// Held-out confirmation of a new best (0 otherwise).
    pub confirmation: f64,
    /// Recording the decision sample of a new best (0 otherwise).
    pub decision_sample: f64,
    /// Writing the champion, best and decision files. The event line and
    /// the checkpoint are written after the event is built, so they are in
    /// no stage (nor in `generation_secs`); the gap between two events'
    /// `elapsed_secs` deltas and `generation_secs` is their cost.
    pub checkpoint_and_files: f64,
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
    /// Generations of the champions that were in the hall of fame (extra
    /// training opponents) while this generation was evaluated.
    #[serde(default)]
    pub hall_of_fame: Vec<u32>,
    /// The champion against tables of hall-of-fame members only (`None`
    /// while the hall is empty).
    #[serde(default)]
    pub hall_score: Option<ScoreStat>,
    pub species: Vec<SpeciesStats>,
    pub compatibility_threshold: f64,
    pub complexity: Complexity,
    /// Seconds per stage; absent in events written before it existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timings: Option<StageTimings>,
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
