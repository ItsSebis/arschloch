//! The training event stream: one JSON object per line in
//! `events.jsonl`. Everything the terminal, the dashboard and later
//! analysis show comes from these events, so they cannot disagree.
//! Changes must be additive, with `SCHEMA_VERSION` bumped for any
//! removal or rename.

use neat::SpeciesStats;
use serde::{Deserialize, Serialize};

use super::config::TrainConfig;
use super::evaluate::Score;
#[cfg(test)]
use super::fitness::{FitnessKind, Moments};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(clippy::large_enum_variant)] // `RunEnd` is written once per run
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
    /// Standard deviation of the per-round role score. Written only when
    /// the run's fitness is not the plain mean (a default run's events are
    /// byte for byte what they were before the fitness variants existed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spread: Option<f64>,
    /// The configured fitness of this evaluation; written only with a
    /// non-mean fitness (otherwise it is `mean`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fitness: Option<f64>,
    /// One column set per setting; written only when more than one
    /// setting (table size, deck) is trained on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub by_setting: Option<Vec<SettingStat>>,
}

impl ScoreStat {
    /// The number runs are ranked on: the configured fitness, which for a
    /// score written by a mean run (or an older run) is its `mean`.
    #[must_use]
    pub fn fitness_value(&self) -> f64 {
        self.fitness.unwrap_or(self.mean)
    }
}

/// One setting's columns of a multi-setting evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SettingStat {
    /// Such as `4p-single`.
    pub label: String,
    pub matches: usize,
    pub mean: f64,
    /// Standard deviation of the per-round role score in this setting.
    pub spread: f64,
    pub fitness: f64,
    /// Finishing places at this table size, best first.
    pub placements: Vec<u64>,
}

impl From<Score> for ScoreStat {
    fn from(score: Score) -> Self {
        let report_fitness = !score.kind.is_mean();
        Self {
            mean: score.mean,
            std_error: score.std_error,
            matches: score.matches,
            placements: score.placements,
            spread: report_fitness.then_some(score.spread),
            fitness: report_fitness.then_some(score.fitness),
            by_setting: (!score.by_setting.is_empty()).then(|| {
                score
                    .by_setting
                    .into_iter()
                    .map(|s| SettingStat {
                        label: s.label,
                        matches: s.matches,
                        mean: s.mean,
                        spread: s.spread,
                        fitness: s.fitness,
                        placements: s.placements,
                    })
                    .collect()
            }),
        }
    }
}

/// What racing did in one generation (present only with racing on): how
/// many genomes were re-evaluated on extra matches and how far their
/// fitness moved, which shows the winner's curse being corrected (the
/// shift is typically negative: the genomes that looked best were lucky).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RacingStats {
    pub raced: usize,
    /// Extra matches each raced genome played.
    pub extra_matches: usize,
    /// Mean fitness of the raced genomes before and after pooling.
    pub mean_before: f64,
    pub mean_after: f64,
    /// `mean_after - mean_before`.
    pub mean_shift: f64,
    /// Rounds of expansion after the first race of the top `raced` genomes
    /// (0 = no un-raced genome outranked a raced one). Absent in events
    /// written before the race expanded.
    #[serde(default)]
    pub expansion_rounds: u32,
    /// How many genomes were raced in the end (`raced` plus the expansion).
    #[serde(default)]
    pub raced_final: usize,
    /// Whether the cap on the raced set stopped the expansion while an
    /// un-raced genome still outranked a raced one.
    #[serde(default)]
    pub cap_hit: bool,
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
    /// Racing's effect this generation; absent when racing is off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub racing: Option<RacingStats>,
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
            spread: None,
            fitness: None,
            by_setting: None,
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
            spread: 0.5,
            fitness: -0.25,
            kind: FitnessKind::Mean,
            moments: Moments::default(),
            by_setting: Vec::new(),
        };
        let stat = ScoreStat::from(score.clone());
        assert_eq!(
            (stat.mean, stat.std_error, stat.matches),
            (score.mean, score.std_error, score.matches)
        );
        assert_eq!(stat.placements, score.placements);
        // A mean run's stat carries nothing new, so its events keep their bytes.
        let line = serde_json::to_string(&stat).unwrap();
        assert!(
            !line.contains("spread") && !line.contains("fitness") && !line.contains("setting"),
            "{line}"
        );
        assert!((stat.fitness_value() + 0.25).abs() < 1e-15);
    }

    #[test]
    fn a_non_mean_or_multi_setting_score_reports_spread_fitness_and_settings() {
        let score = Score {
            mean: 0.2,
            std_error: 0.1,
            matches: 7,
            placements: vec![1, 2, 3, 4],
            spread: 0.6,
            fitness: -0.1,
            kind: FitnessKind::Risk { lambda: 0.5 },
            moments: Moments::default(),
            by_setting: vec![super::super::evaluate::SettingScore {
                label: "4p-single".into(),
                matches: 7,
                mean: 0.2,
                spread: 0.6,
                fitness: -0.1,
                placements: vec![1, 2, 3, 4],
            }],
        };
        let stat = ScoreStat::from(score);
        assert_eq!(stat.spread, Some(0.6));
        assert_eq!(stat.fitness, Some(-0.1));
        assert!((stat.fitness_value() + 0.1).abs() < 1e-15);
        assert_eq!(stat.by_setting.as_ref().unwrap()[0].label, "4p-single");
        let again: ScoreStat =
            serde_json::from_str(&serde_json::to_string(&stat).unwrap()).unwrap();
        assert_eq!(again, stat);
    }
}
