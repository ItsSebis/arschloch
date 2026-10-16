//! Reading `events.jsonl` files, run directories and set directories.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use super::lean::{LeanEnd, LeanEvent, LeanGeneration, RunStartInfo};

/// Why a path could not be loaded for analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnalysisError {
    /// A file or directory could not be read.
    Io(String),
    /// The path holds no run with a generation event.
    NoEvents(String),
}

impl fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(reason) => write!(f, "file error: {reason}"),
            Self::NoEvents(path) => write!(f, "{path}: no generation events to analyse"),
        }
    }
}

impl std::error::Error for AnalysisError {}

/// One run's events, deduplicated by generation.
#[derive(Debug, Clone)]
pub struct RunData {
    /// Directory name (or file stem) the run is shown under.
    pub label: String,
    /// Name of the directory that holds the run (the set directory, or the
    /// folder of loose files), used to tell equal labels apart.
    pub parent: Option<String>,
    /// The last `run_start` (the one of the latest resume); absent when the
    /// file has none. Its `resumed_from_generation` says whether the run was
    /// resumed.
    pub start: Option<RunStartInfo>,
    /// How many `run_start` events the file holds (more than one: resumed).
    pub starts: usize,
    /// Sorted by generation; a later event for a generation (after a
    /// resume) replaces the earlier one.
    pub generations: Vec<LeanGeneration>,
    /// The last `run_end`, when the run finished. A `run_end` followed by a
    /// later `run_start` (the run was resumed) is dropped: it describes a
    /// stage that is no longer the end.
    pub end: Option<LeanEnd>,
    /// Lines that were not valid events (old format, truncated tail).
    pub skipped_lines: usize,
}

/// The runs found at a path and the ones that had to be left out.
#[derive(Debug, Clone)]
pub struct Loaded {
    pub runs: Vec<RunData>,
    /// One line per run of a set directory that could not be read
    /// (`name: reason`); the other runs are still analysed.
    pub skipped_runs: Vec<String>,
}

/// Parses the text of an `events.jsonl`; lines that are not events are
/// counted in `skipped_lines`.
#[must_use]
pub fn parse_run(label: String, text: &str) -> RunData {
    let mut start = None;
    let mut starts = 0;
    let mut end = None;
    let mut skipped_lines = 0;
    let mut by_generation = BTreeMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        match serde_json::from_str::<LeanEvent>(line) {
            Ok(LeanEvent::RunStart(s)) => {
                start = Some(*s);
                starts += 1;
                end = None;
            }
            Ok(LeanEvent::Generation(g)) => {
                by_generation.insert(g.generation, *g);
            }
            Ok(LeanEvent::RunEnd(e)) => end = Some(e),
            Err(_) => skipped_lines += 1,
        }
    }
    RunData {
        label,
        parent: None,
        start,
        starts,
        generations: by_generation.into_values().collect(),
        end,
        skipped_lines,
    }
}

/// Gives runs unambiguous labels. When `force` is set or two labels are
/// equal, every label gets the name of its parent directory in front
/// (`A0/seed1`); labels still equal after that get ` (2)`, ` (3)`.
pub fn qualify_labels(runs: &mut [RunData], force: bool) {
    let collide = runs
        .iter()
        .enumerate()
        .any(|(i, r)| runs[..i].iter().any(|o| o.label == r.label));
    if force || collide {
        for run in runs.iter_mut() {
            if let Some(parent) = &run.parent {
                run.label = format!("{parent}/{}", run.label);
            }
        }
    }
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for run in runs.iter_mut() {
        let count = seen.entry(run.label.clone()).or_insert(0);
        *count += 1;
        if *count > 1 {
            run.label = format!("{} ({count})", run.label);
        }
    }
}

/// How paths are checked and shown: with a root, nothing outside it is read
/// and error text names paths relative to it.
struct Scope {
    root: Option<PathBuf>,
}

impl Scope {
    /// The path as it may appear in a message.
    fn shown(&self, path: &Path) -> String {
        match &self.root {
            None => path.display().to_string(),
            Some(root) => {
                let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
                match canonical.strip_prefix(root) {
                    Ok(rel) if rel.as_os_str().is_empty() => "the served directory".to_owned(),
                    Ok(rel) => rel.display().to_string(),
                    Err(_) => name_of(path),
                }
            }
        }
    }

    /// Whether `path` (symlinks followed) is inside the root; always true
    /// without one.
    fn allows(&self, path: &Path) -> bool {
        self.root
            .as_ref()
            .is_none_or(|root| path.canonicalize().is_ok_and(|c| c.starts_with(root)))
    }

    fn io_error(&self, path: &Path, error: &std::io::Error) -> AnalysisError {
        AnalysisError::Io(format!("{}: {error}", self.shown(path)))
    }

    fn read_run(
        &self,
        file: &Path,
        label: String,
        parent: Option<String>,
    ) -> Result<RunData, AnalysisError> {
        if !self.allows(file) {
            return Err(AnalysisError::Io(format!(
                "{}: outside the served directory",
                self.shown(file)
            )));
        }
        let bytes = fs::read(file).map_err(|e| self.io_error(file, &e))?;
        let mut run = parse_run(label, &String::from_utf8_lossy(&bytes));
        run.parent = parent;
        Ok(run)
    }
}

fn name_of(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

fn parent_name(path: &Path) -> Option<String> {
    path.parent()
        .filter(|p| p.file_name().is_some())
        .map(name_of)
}

fn stem_of(path: &Path) -> String {
    path.file_stem()
        .map_or_else(String::new, |s| s.to_string_lossy().into_owned())
}

fn is_jsonl_file(path: &Path) -> bool {
    path.is_file() && path.extension().is_some_and(|e| e == "jsonl")
}

fn single(run: RunData, shown: String) -> Result<Loaded, AnalysisError> {
    if run.generations.is_empty() {
        Err(AnalysisError::NoEvents(shown))
    } else {
        Ok(Loaded {
            runs: vec![run],
            skipped_runs: Vec::new(),
        })
    }
}

/// Loads the runs at `path`: an events file (one run, labelled by its stem,
/// or by its directory when it is called `events.jsonl`), a run directory
/// (holds `events.jsonl`), or a set directory (one run per subdirectory
/// that holds `events.jsonl`, or per loose `*.jsonl` file, sorted by name;
/// subdirectories whose run has no generation event yet are left out, and
/// ones that cannot be read are listed in `skipped_runs`).
///
/// # Errors
///
/// `AnalysisError::Io` when the path itself cannot be read (or no run of a
/// set could), `NoEvents` when no run with a generation event is found.
pub fn load_runs(path: &Path) -> Result<Loaded, AnalysisError> {
    load_runs_within(path, None)
}

/// `load_runs`, but nothing outside `root` is read: every run directory and
/// events file of a set is canonicalised (symlinks followed) and dropped when
/// it lies outside, and error text names paths relative to `root`, never
/// absolute ones.
///
/// # Errors
///
/// As `load_runs`.
pub fn load_runs_within(path: &Path, root: Option<&Path>) -> Result<Loaded, AnalysisError> {
    let scope =
        Scope {
            root: match root {
                None => None,
                Some(r) => Some(r.canonicalize().map_err(|_| {
                    AnalysisError::Io("the served directory is not readable".into())
                })?),
            },
        };
    let shown = scope.shown(path);
    let meta = fs::metadata(path).map_err(|e| scope.io_error(path, &e))?;
    if meta.is_file() {
        let stem = stem_of(path);
        let canonical = path.canonicalize().ok();
        let (label, parent) = match (&canonical, stem == "events") {
            (Some(c), true) => (
                c.parent().map_or_else(|| stem.clone(), name_of),
                c.parent().and_then(parent_name),
            ),
            (Some(c), false) => (stem, c.parent().map(name_of)),
            (None, _) => (stem, None),
        };
        return single(scope.read_run(path, label, parent)?, shown);
    }
    let own = path.join("events.jsonl");
    if own.is_file() {
        let canonical = path.canonicalize().map_err(|e| scope.io_error(path, &e))?;
        let run = scope.read_run(&own, name_of(&canonical), parent_name(&canonical))?;
        return single(run, shown);
    }
    // A set directory (run subdirectories) or a folder of loose `*.jsonl` event files.
    let set_name = path.canonicalize().ok().map(|c| name_of(&c));
    let mut children: Vec<PathBuf> = fs::read_dir(path)
        .map_err(|e| scope.io_error(path, &e))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.join("events.jsonl").is_file() || is_jsonl_file(p))
        .filter(|p| scope.allows(p))
        .collect();
    children.sort();
    let mut runs = Vec::new();
    let mut skipped_runs = Vec::new();
    for child in children {
        let (file, label) = if child.is_dir() {
            (child.join("events.jsonl"), name_of(&child))
        } else {
            (child.clone(), stem_of(&child))
        };
        match scope.read_run(&file, label.clone(), set_name.clone()) {
            Ok(run) if !run.generations.is_empty() => runs.push(run),
            Ok(_) => {}
            Err(AnalysisError::Io(reason)) => skipped_runs.push(reason),
            Err(other) => skipped_runs.push(other.to_string()),
        }
    }
    if runs.is_empty() {
        return Err(if skipped_runs.is_empty() {
            AnalysisError::NoEvents(shown)
        } else {
            AnalysisError::Io(format!(
                "{shown}: no run could be read ({})",
                skipped_runs.join("; ")
            ))
        });
    }
    Ok(Loaded { runs, skipped_runs })
}
