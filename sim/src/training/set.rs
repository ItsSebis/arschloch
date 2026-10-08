//! A *set* of runs: `cli train --runs N` trains N runs one after another
//! in `OUT/run-01 ... OUT/run-NN`, and `OUT/set.json` records which runs
//! are finished so the time left for the whole set can be estimated and a
//! killed set can be resumed.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::run_dir::{write_atomically_in, TrainError};

const FILE: &str = "set.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SetFile {
    pub total_runs: u32,
    /// The run in progress (1-based); `finished_secs.len() + 1` while a
    /// run is going, `total_runs` once the last one has finished.
    pub current_run: u32,
    /// Wall-clock seconds of each finished run, in order.
    pub finished_secs: Vec<f64>,
}

/// The directory name of run `index` (1-based): `run-01`, `run-02`, ...
#[must_use]
pub fn run_dir_name(index: u32) -> String {
    format!("run-{index:02}")
}

impl SetFile {
    /// # Errors
    ///
    /// `TrainError::Io` if the file cannot be written.
    pub fn write(&self, dir: &Path) -> Result<(), TrainError> {
        let text = serde_json::to_string_pretty(self)
            .map_err(|e| TrainError::Checkpoint(e.to_string()))?;
        write_atomically_in(dir, FILE, &text)
    }

    /// `Ok(None)` when `dir` holds no set (a plain run directory).
    ///
    /// # Errors
    ///
    /// `TrainError::Checkpoint` if the file exists but is unreadable or
    /// malformed.
    pub fn read(dir: &Path) -> Result<Option<Self>, TrainError> {
        let path = dir.join(FILE);
        if !path.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(&path)
            .map_err(|e| TrainError::Checkpoint(format!("{}: {e}", path.display())))?;
        serde_json::from_str(&text)
            .map(Some)
            .map_err(|e| TrainError::Checkpoint(format!("{}: {e}", path.display())))
    }
}

/// Seconds left for the whole set: the current run's own estimate, plus
/// the runs after it at the average time of the finished runs (or, before
/// any has finished, at this run's projected total time). `None` while the
/// current run has no estimate yet.
#[must_use]
pub fn set_eta(
    finished_secs: &[f64],
    total_runs: u32,
    current_run_eta: Option<f64>,
    current_run_elapsed: f64,
) -> Option<f64> {
    let eta = current_run_eta?;
    let finished = u32::try_from(finished_secs.len()).unwrap_or(u32::MAX);
    let remaining = total_runs.saturating_sub(finished).saturating_sub(1);
    #[allow(clippy::cast_precision_loss)] // a handful of runs
    let per_run = if finished_secs.is_empty() {
        current_run_elapsed + eta
    } else {
        finished_secs.iter().sum::<f64>() / finished_secs.len() as f64
    };
    Some(eta + f64::from(remaining) * per_run)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("an estimate");
        assert!((actual - expected).abs() < 1e-9, "{actual} vs {expected}");
    }

    #[test]
    fn before_any_run_finishes_the_current_run_is_projected_forward() {
        // 100 s elapsed, 100 s to go: a run takes 200 s, two more follow.
        approx(set_eta(&[], 3, Some(100.0), 100.0), 100.0 + 2.0 * 200.0);
    }

    #[test]
    fn finished_runs_set_the_pace_for_the_rest() {
        approx(
            set_eta(&[60.0, 80.0], 5, Some(30.0), 10.0),
            30.0 + 2.0 * 70.0,
        );
    }

    #[test]
    fn the_last_run_only_has_its_own_time_left() {
        approx(set_eta(&[60.0, 80.0], 3, Some(30.0), 10.0), 30.0);
    }

    #[test]
    fn no_estimate_for_the_current_run_means_none() {
        assert_eq!(set_eta(&[], 3, None, 0.0), None);
        assert_eq!(set_eta(&[10.0], 3, None, 5.0), None);
    }

    #[test]
    fn more_finished_runs_than_the_total_never_goes_negative() {
        approx(set_eta(&[10.0, 10.0, 10.0], 2, Some(4.0), 1.0), 4.0);
        approx(set_eta(&[], 0, Some(4.0), 1.0), 4.0);
    }

    fn temp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("arschloch-set-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn the_set_file_round_trips_and_an_absent_one_is_none() {
        let dir = temp("roundtrip");
        assert_eq!(SetFile::read(&dir).unwrap(), None);
        let file = SetFile {
            total_runs: 4,
            current_run: 2,
            finished_secs: vec![12.5],
        };
        file.write(&dir).unwrap();
        assert_eq!(SetFile::read(&dir).unwrap(), Some(file));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_corrupt_set_file_is_an_error_not_a_panic() {
        let dir = temp("corrupt");
        fs::write(dir.join("set.json"), "{ nope").unwrap();
        assert!(matches!(
            SetFile::read(&dir),
            Err(TrainError::Checkpoint(_))
        ));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn run_directories_sort_in_run_order() {
        assert_eq!(run_dir_name(1), "run-01");
        assert_eq!(run_dir_name(12), "run-12");
    }
}
