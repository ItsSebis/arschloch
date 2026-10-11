//! Reading `events.jsonl` files, run directories and set directories.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use crate::training::events::{Event, GenerationEvent, RunEnd, RunStart};

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
    /// The first `run_start`; absent when it did not parse (old runs).
    pub start: Option<RunStart>,
    /// Sorted by generation; a later event for a generation (after a
    /// resume) replaces the earlier one.
    pub generations: Vec<GenerationEvent>,
    /// The last `run_end`, when the run finished.
    pub end: Option<RunEnd>,
    /// Lines that were not valid events (old format, truncated tail).
    pub skipped_lines: usize,
}

fn io_error(path: &Path, error: &std::io::Error) -> AnalysisError {
    AnalysisError::Io(format!("{}: {error}", path.display()))
}

/// Parses the text of an `events.jsonl`; lines that are not events are
/// counted in `skipped_lines`.
#[must_use]
pub fn parse_run(label: String, text: &str) -> RunData {
    let mut start = None;
    let mut end = None;
    let mut skipped_lines = 0;
    let mut by_generation = BTreeMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        match serde_json::from_str::<Event>(line) {
            Ok(Event::RunStart(s)) => {
                start.get_or_insert(*s);
            }
            Ok(Event::Generation(g)) => {
                by_generation.insert(g.generation, *g);
            }
            Ok(Event::RunEnd(e)) => end = Some(e),
            Err(_) => skipped_lines += 1,
        }
    }
    RunData {
        label,
        start,
        generations: by_generation.into_values().collect(),
        end,
        skipped_lines,
    }
}

fn read_run(file: &Path, label: String) -> Result<RunData, AnalysisError> {
    let bytes = fs::read(file).map_err(|e| io_error(file, &e))?;
    Ok(parse_run(label, &String::from_utf8_lossy(&bytes)))
}

fn name_of(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

fn is_jsonl_file(path: &Path) -> bool {
    path.is_file() && path.extension().is_some_and(|e| e == "jsonl")
}

/// Loads the runs at `path`: an events file (one run, labelled by its stem,
/// or by its directory when it is called `events.jsonl`), a run directory
/// (holds `events.jsonl`), or a set directory (one run per subdirectory
/// that holds `events.jsonl`, or per loose `*.jsonl` file, sorted by name; subdirectories whose run has
/// no generation event yet are left out).
///
/// # Errors
///
/// `AnalysisError::Io` when something cannot be read, `NoEvents` when no
/// run with a generation event is found.
pub fn load_runs(path: &Path) -> Result<Vec<RunData>, AnalysisError> {
    let shown = path.display().to_string();
    let meta = fs::metadata(path).map_err(|e| io_error(path, &e))?;
    if meta.is_file() {
        let stem = path
            .file_stem()
            .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
        let label = if stem == "events" {
            path.canonicalize()
                .ok()
                .and_then(|p| p.parent().map(name_of))
                .unwrap_or(stem)
        } else {
            stem
        };
        let run = read_run(path, label)?;
        return if run.generations.is_empty() {
            Err(AnalysisError::NoEvents(shown))
        } else {
            Ok(vec![run])
        };
    }
    let own = path.join("events.jsonl");
    if own.is_file() {
        let label = name_of(&path.canonicalize().map_err(|e| io_error(path, &e))?);
        let run = read_run(&own, label)?;
        return if run.generations.is_empty() {
            Err(AnalysisError::NoEvents(shown))
        } else {
            Ok(vec![run])
        };
    }
    // A set directory (run subdirectories) or a folder of loose `*.jsonl` event files.
    let mut children: Vec<PathBuf> = fs::read_dir(path)
        .map_err(|e| io_error(path, &e))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.join("events.jsonl").is_file() || is_jsonl_file(p))
        .collect();
    children.sort();
    let mut runs = Vec::new();
    for child in children {
        let run = if child.is_dir() {
            read_run(&child.join("events.jsonl"), name_of(&child))?
        } else {
            let stem = child
                .file_stem()
                .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
            read_run(&child, stem)?
        };
        if !run.generations.is_empty() {
            runs.push(run);
        }
    }
    if runs.is_empty() {
        return Err(AnalysisError::NoEvents(shown));
    }
    Ok(runs)
}
