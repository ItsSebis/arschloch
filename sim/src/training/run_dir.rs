//! The files of a training run, all inside one directory:
//!
//! - `config.json`: the run's settings (informational; the checkpoint is
//!   authoritative for resuming);
//! - `events.jsonl`: the event stream;
//! - `checkpoint.json`: everything needed to resume, replaced atomically
//!   after every generation;
//! - `gen-NNNN.json`: each generation's champion, as a `GenomeFile`
//!   playable with `--strategy neat:PATH`;
//! - `best.json`: the champion with the best fresh-seed score so far.

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use neat::{Genome, PopulationState};
use serde::{Deserialize, Serialize};

use super::config::TrainConfig;
use super::decisions::DecisionFile;
use super::events::{Event, ScoreStat, SCHEMA_VERSION};
use crate::GenomeFile;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainError {
    Io(String),
    /// A setting is invalid.
    Config(String),
    /// A run directory's files are unusable.
    Checkpoint(String),
    /// A resume does not match the run it continues.
    Mismatch(String),
}

impl fmt::Display for TrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(reason) => write!(f, "file error: {reason}"),
            Self::Config(reason) => write!(f, "invalid training setup: {reason}"),
            Self::Checkpoint(reason) => write!(f, "unusable run directory: {reason}"),
            Self::Mismatch(reason) => write!(f, "cannot resume: {reason}"),
        }
    }
}

impl std::error::Error for TrainError {}

fn io_error(path: &Path, error: &std::io::Error) -> TrainError {
    TrainError::Io(format!("{}: {error}", path.display()))
}

/// The best champion so far, judged on fresh matches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BestRecord {
    pub generation: u32,
    pub reeval: ScoreStat,
    pub heldout: ScoreStat,
}

/// A frozen past champion serving as an extra training opponent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallMember {
    pub generation: u32,
    pub genome: Genome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub schema_version: u32,
    /// The feature set the genomes were trained against: a checkpoint from
    /// a build with different features cannot be continued (its genomes
    /// would be evaluated as something else).
    pub feature_count: usize,
    pub feature_set_version: u32,
    pub config: TrainConfig,
    pub opponent_names: Vec<String>,
    pub population: PopulationState,
    pub best: Option<BestRecord>,
    #[serde(default)]
    pub hall_of_fame: Vec<HallMember>,
    pub total_rounds: u64,
    pub elapsed_secs: f64,
}

/// Reads the settings of the run in `dir` from its checkpoint (what a
/// resume needs before it can rebuild the opponent pool).
///
/// # Errors
///
/// `TrainError::Checkpoint` if `dir` holds no usable checkpoint.
pub fn load_config(dir: &Path) -> Result<TrainConfig, TrainError> {
    Ok(RunDir::open_existing(dir)?.read_checkpoint()?.config)
}

pub struct RunDir {
    root: PathBuf,
}

/// Writes `root/name` so a reader (or a crash) sees either the old or the
/// whole new contents: temp file, fsync, rename.
pub(crate) fn write_atomically_in(
    root: &Path,
    name: &str,
    contents: &str,
) -> Result<(), TrainError> {
    let target = root.join(name);
    let temporary = root.join(format!("{name}.tmp"));
    let mut file = File::create(&temporary).map_err(|e| io_error(&temporary, &e))?;
    file.write_all(contents.as_bytes())
        .map_err(|e| io_error(&temporary, &e))?;
    file.sync_all().map_err(|e| io_error(&temporary, &e))?;
    fs::rename(&temporary, &target).map_err(|e| io_error(&target, &e))
}

impl RunDir {
    /// Starts a new run directory, creating it if needed.
    ///
    /// # Errors
    ///
    /// Refuses a directory that already holds a run (use resume instead,
    /// or pick another directory), so a run is never overwritten.
    pub fn create_new(root: &Path) -> Result<Self, TrainError> {
        fs::create_dir_all(root).map_err(|e| io_error(root, &e))?;
        let dir = Self {
            root: root.to_owned(),
        };
        for existing in ["checkpoint.json", "events.jsonl"] {
            if dir.path(existing).exists() {
                return Err(TrainError::Config(format!(
                    "{} already holds a run; resume it or choose another directory",
                    root.display()
                )));
            }
        }
        Ok(dir)
    }

    /// Opens a directory that holds a run.
    ///
    /// # Errors
    ///
    /// Returns `TrainError::Checkpoint` if there is no checkpoint.
    pub fn open_existing(root: &Path) -> Result<Self, TrainError> {
        let dir = Self {
            root: root.to_owned(),
        };
        if !dir.path("checkpoint.json").exists() {
            return Err(TrainError::Checkpoint(format!(
                "{} has no checkpoint.json",
                root.display()
            )));
        }
        Ok(dir)
    }

    #[must_use]
    pub fn path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    fn write_atomically(&self, name: &str, contents: &str) -> Result<(), TrainError> {
        write_atomically_in(&self.root, name, contents)
    }

    /// # Errors
    ///
    /// `TrainError::Io` if the file cannot be written.
    pub fn write_config(&self, config: &TrainConfig) -> Result<(), TrainError> {
        let text =
            serde_json::to_string_pretty(config).map_err(|e| TrainError::Config(e.to_string()))?;
        self.write_atomically("config.json", &text)
    }

    /// Appends one event as a line and flushes it, so a watcher sees it
    /// immediately.
    ///
    /// # Errors
    ///
    /// `TrainError::Io` if the log cannot be written.
    pub fn append_event(&self, event: &Event) -> Result<(), TrainError> {
        let path = self.path("events.jsonl");
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| io_error(&path, &e))?;
        let mut line =
            serde_json::to_string(event).map_err(|e| TrainError::Checkpoint(e.to_string()))?;
        line.push('\n');
        file.write_all(line.as_bytes())
            .map_err(|e| io_error(&path, &e))
    }

    /// Drops generation events at or after `generation` (and any run-end
    /// marker), keeping the log consistent with a checkpoint when a run
    /// died between writing an event and its checkpoint.
    ///
    /// # Errors
    ///
    /// `TrainError::Io` if the log cannot be read or rewritten.
    pub fn truncate_events_from(&self, generation: u32) -> Result<(), TrainError> {
        let path = self.path("events.jsonl");
        let Ok(text) = fs::read_to_string(&path) else {
            return Ok(());
        };
        let mut kept = String::new();
        for line in text.lines() {
            let keep = match serde_json::from_str::<Event>(line) {
                Ok(Event::Generation(e)) => e.generation < generation,
                Ok(Event::RunEnd(_)) => false,
                _ => true,
            };
            if keep {
                kept.push_str(line);
                kept.push('\n');
            }
        }
        self.write_atomically("events.jsonl", &kept)
    }

    /// # Errors
    ///
    /// `TrainError::Io` if the checkpoint cannot be written.
    pub fn write_checkpoint(&self, checkpoint: &Checkpoint) -> Result<(), TrainError> {
        let text =
            serde_json::to_string(checkpoint).map_err(|e| TrainError::Checkpoint(e.to_string()))?;
        self.write_atomically("checkpoint.json", &text)
    }

    /// # Errors
    ///
    /// `TrainError::Checkpoint` if the file is unreadable or malformed.
    pub fn read_checkpoint(&self) -> Result<Checkpoint, TrainError> {
        let path = self.path("checkpoint.json");
        let text = fs::read_to_string(&path).map_err(|e| io_error(&path, &e))?;
        let checkpoint: Checkpoint = serde_json::from_str(&text)
            .map_err(|e| TrainError::Checkpoint(format!("{}: {e}", path.display())))?;
        if checkpoint.schema_version != SCHEMA_VERSION {
            return Err(TrainError::Checkpoint(format!(
                "schema version {} (this build reads {SCHEMA_VERSION})",
                checkpoint.schema_version
            )));
        }
        Ok(checkpoint)
    }

    /// Writes `gen-NNNN.json` and returns its file name.
    ///
    /// # Errors
    ///
    /// `TrainError::Io` on write failure, `TrainError::Config` if the
    /// genome does not fit this build's features.
    pub fn write_champion(&self, generation: u32, genome: &Genome) -> Result<String, TrainError> {
        let name = format!("gen-{generation:04}.json");
        self.write_genome(&name, genome)?;
        Ok(name)
    }

    /// Writes `decisions/gen-NNNN.json`.
    ///
    /// # Errors
    ///
    /// `TrainError::Io` on write failure.
    pub fn write_decisions(&self, decisions: &DecisionFile) -> Result<(), TrainError> {
        let directory = self.path("decisions");
        fs::create_dir_all(&directory).map_err(|e| io_error(&directory, &e))?;
        let text =
            serde_json::to_string(decisions).map_err(|e| TrainError::Config(e.to_string()))?;
        self.write_atomically(
            &format!("decisions/gen-{:04}.json", decisions.generation),
            &text,
        )
    }

    /// # Errors
    ///
    /// As `write_champion`.
    pub fn write_best(&self, genome: &Genome) -> Result<(), TrainError> {
        self.write_genome("best.json", genome)
    }

    fn write_genome(&self, name: &str, genome: &Genome) -> Result<(), TrainError> {
        let file =
            GenomeFile::new(genome.clone()).map_err(|e| TrainError::Config(e.to_string()))?;
        let text =
            serde_json::to_string_pretty(&file).map_err(|e| TrainError::Config(e.to_string()))?;
        self.write_atomically(name, &text)
    }
}

#[cfg(test)]
mod tests {
    use neat::{NeatConfig, Population};

    use super::super::config::test_support::sample;
    use super::super::events::{RunEnd, RunStart};
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("arschloch-rundir-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn checkpoint() -> Checkpoint {
        let population = Population::new(
            crate::FEATURE_COUNT,
            NeatConfig {
                population_size: 4,
                ..NeatConfig::default()
            },
            1,
        )
        .unwrap();
        Checkpoint {
            schema_version: SCHEMA_VERSION,
            feature_count: crate::FEATURE_COUNT,
            feature_set_version: crate::FEATURE_SET_VERSION,
            config: sample(),
            opponent_names: vec!["LowestLegal".into()],
            population: population.snapshot(),
            best: None,
            hall_of_fame: Vec::new(),
            total_rounds: 10,
            elapsed_secs: 1.0,
        }
    }

    fn generation_event(generation: u32) -> Event {
        let stat = ScoreStat {
            mean: 0.0,
            std_error: 0.0,
            matches: 1,
            placements: vec![1],
        };
        Event::Generation(Box::new(super::super::events::GenerationEvent {
            generation,
            elapsed_secs: 0.0,
            generation_secs: 0.0,
            rounds_evaluated: 0,
            total_rounds: 0,
            rounds_per_sec: 0.0,
            fitness: super::super::events::FitnessStats {
                best: 0.0,
                mean: 0.0,
                median: 0.0,
                min: 0.0,
                std_dev: 0.0,
                histogram: vec![],
                skill_term: None,
            },
            champion: super::super::events::ChampionStats {
                train_fitness: 0.0,
                skill_term: None,
                reeval: stat,
                heldout: None,
                training_rank: 0,
                hidden_nodes: 0,
                enabled_connections: 0,
                genome_file: String::new(),
                is_new_best: false,
            },
            opponents: vec![],
            hall_of_fame: vec![],
            hall_score: None,
            species: vec![],
            compatibility_threshold: 0.5,
            complexity: super::super::events::Complexity {
                mean_hidden_nodes: 0.0,
                mean_enabled_connections: 0.0,
                innovation_count: 0,
            },
            timings: None,
        }))
    }

    #[test]
    fn a_new_run_refuses_a_directory_that_already_holds_one() {
        let dir = temp_dir("refuse");
        let run = RunDir::create_new(&dir).unwrap();
        run.write_checkpoint(&checkpoint()).unwrap();
        let error = RunDir::create_new(&dir).err().unwrap();
        assert!(matches!(error, TrainError::Config(_)), "{error}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn opening_a_directory_without_a_checkpoint_fails_clearly() {
        let dir = temp_dir("nocheckpoint");
        fs::create_dir_all(&dir).unwrap();
        let error = RunDir::open_existing(&dir).err().unwrap();
        assert!(error.to_string().contains("no checkpoint.json"), "{error}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_checkpoint_survives_a_round_trip_and_leaves_no_temporary_file() {
        let dir = temp_dir("roundtrip");
        let run = RunDir::create_new(&dir).unwrap();
        let original = checkpoint();
        run.write_checkpoint(&original).unwrap();
        let loaded = run.read_checkpoint().unwrap();
        assert_eq!(loaded.config, original.config);
        assert_eq!(
            serde_json::to_string(&loaded.population).unwrap(),
            serde_json::to_string(&original.population).unwrap()
        );
        assert!(!run.path("checkpoint.json.tmp").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn load_config_reads_the_runs_settings_back() {
        let dir = temp_dir("loadconfig");
        let run = RunDir::create_new(&dir).unwrap();
        run.write_checkpoint(&checkpoint()).unwrap();
        assert_eq!(load_config(&dir).unwrap(), sample());
        assert!(load_config(&dir.join("missing")).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_corrupt_checkpoint_is_an_error_not_a_panic() {
        let dir = temp_dir("corrupt");
        let run = RunDir::create_new(&dir).unwrap();
        fs::write(run.path("checkpoint.json"), "{ truncated").unwrap();
        assert!(matches!(
            run.read_checkpoint(),
            Err(TrainError::Checkpoint(_))
        ));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn champions_are_playable_genome_files() {
        let dir = temp_dir("champions");
        let run = RunDir::create_new(&dir).unwrap();
        let genome = neat::Population::restore(checkpoint().population)
            .unwrap()
            .genomes()[0]
            .clone();
        let name = run.write_champion(7, &genome).unwrap();
        assert_eq!(name, "gen-0007.json");
        run.write_best(&genome).unwrap();
        assert!(crate::NeatStrategy::from_file(&run.path(&name)).is_ok());
        assert!(crate::NeatStrategy::from_file(&run.path("best.json")).is_ok());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn events_append_as_lines_and_truncation_drops_the_tail() {
        let dir = temp_dir("events");
        let run = RunDir::create_new(&dir).unwrap();
        run.append_event(&Event::RunStart(Box::new(RunStart {
            schema_version: SCHEMA_VERSION,
            config: sample(),
            opponents: vec![],
            feature_names: vec![],
            resumed_from_generation: None,
            warm_started_from: None,
        })))
        .unwrap();
        for generation in 0..4 {
            run.append_event(&generation_event(generation)).unwrap();
        }
        run.append_event(&Event::RunEnd(RunEnd {
            generations_completed: 4,
            best_generation: None,
            best_reeval: None,
            best_heldout: None,
            elapsed_secs: 0.0,
        }))
        .unwrap();
        run.truncate_events_from(2).unwrap();
        let text = fs::read_to_string(run.path("events.jsonl")).unwrap();
        let events: Vec<Event> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(events.len(), 3, "start + generations 0 and 1");
        assert!(matches!(events[0], Event::RunStart(_)));
        assert!(matches!(&events[2], Event::Generation(e) if e.generation == 1));
        fs::remove_dir_all(&dir).unwrap();
    }
}
