//! `cli train --when-flat advise|stop`: looks at the run's own `events.jsonl` with the Phase 17
//! analysis (`cli analyse-run`) while it trains, and says so, or stops, when the curve is flat.

use std::fmt::Write as _;
use std::path::PathBuf;

use clap::ValueEnum;
use sim::training::analysis::{analyse, load_runs, Verdict, VerdictCode};

/// Nothing is judged before this many generations: the last third must hold enough points.
pub const MIN_GENERATIONS: u32 = 30;
/// The analysis runs after every this many generations.
const CHECK_EVERY: u32 = 5;
/// A resumed run is judged again only after this many generations of its own since the resume,
/// so a run stopped as flat is not stopped again at once on the same old history.
pub const MIN_NEW_GENERATIONS: u32 = 5;

/// How a stopped run is continued (the one hint, used by the advice text and by `cli train`).
#[must_use]
pub fn resume_hint(dir: &std::path::Path) -> String {
    format!(
        "`cli train --out {} --resume` continues it (add --generations N to change the total)",
        dir.display()
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum WhenFlat {
    /// Print what the analysis says once it finds the curve flat, and keep training.
    Advise,
    /// Print it and end the run when the last 40% of it added nothing visible
    /// (`--resume --generations N` continues it).
    Stop,
}

/// The codes that mean "the last-third trend is flat and the run is not still improving".
fn is_flat(code: VerdictCode) -> bool {
    matches!(
        code,
        VerdictCode::StopEarly
            | VerdictCode::NoiseLimited
            | VerdictCode::PoolLimited
            | VerdictCode::ModelLimited
    )
}

/// What to do with the verdicts of one analysis: the flat ones, and whether to stop.
pub fn decide(verdicts: &[Verdict], mode: WhenFlat) -> Option<(bool, Vec<&Verdict>)> {
    let flat: Vec<&Verdict> = verdicts.iter().filter(|v| is_flat(v.code)).collect();
    if flat.is_empty() {
        return None;
    }
    let stop = mode == WhenFlat::Stop && flat.iter().any(|v| v.code == VerdictCode::StopEarly);
    Some((stop, flat))
}

pub struct FlatWatch {
    dir: PathBuf,
    mode: WhenFlat,
    /// The codes last advised, so the same advice is not repeated every five generations.
    advised: Vec<VerdictCode>,
    stopping: bool,
    /// The first generation this process trained (differs from 0 on a resume).
    first_generation: Option<u32>,
}

impl FlatWatch {
    #[must_use]
    pub fn new(dir: PathBuf, mode: WhenFlat) -> Self {
        Self {
            dir,
            mode,
            advised: Vec::new(),
            stopping: false,
            first_generation: None,
        }
    }

    #[must_use]
    pub fn stopping(&self) -> bool {
        self.stopping
    }

    /// Called after generation number `generation` is written; the text to print, if any.
    pub fn after_generation(&mut self, generation: u32) -> Option<String> {
        let completed = generation + 1;
        let first = *self.first_generation.get_or_insert(generation);
        if completed < MIN_GENERATIONS
            || !completed.is_multiple_of(CHECK_EVERY)
            || completed - first < MIN_NEW_GENERATIONS
        {
            return None;
        }
        // An unreadable or too short file is not a reason to disturb the run.
        let loaded = load_runs(&self.dir).ok()?;
        let analysis = analyse(loaded.runs.first()?);
        let (stop, flat) = decide(&analysis.verdict, self.mode)?;
        let codes: Vec<VerdictCode> = flat.iter().map(|v| v.code).collect();
        if stop {
            self.stopping = true;
        } else if codes == self.advised {
            return None;
        }
        self.advised = codes;
        let mut text = format!("\nadvice after {completed} generations (the Phase 17 analysis, see `cli analyse-run`):");
        for verdict in flat {
            let _ = write!(text, "\n  - {}", verdict.message);
        }
        if stop {
            let _ = write!(
                text,
                "\n  stopping here (--when-flat stop); {}.",
                resume_hint(&self.dir)
            );
        }
        Some(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verdict(code: VerdictCode) -> Verdict {
        Verdict {
            code,
            message: format!("{code:?}"),
        }
    }

    #[test]
    fn only_a_flat_curve_triggers_and_only_stop_early_stops() {
        assert!(decide(&[verdict(VerdictCode::StillImproving)], WhenFlat::Stop).is_none());
        assert!(decide(&[verdict(VerdictCode::Converged)], WhenFlat::Stop).is_none());
        let (stop, _) = decide(&[verdict(VerdictCode::ModelLimited)], WhenFlat::Stop).unwrap();
        assert!(
            !stop,
            "flat but the last 40% did not add nothing: advise only"
        );
        let both = [
            verdict(VerdictCode::ModelLimited),
            verdict(VerdictCode::StopEarly),
        ];
        let (stop, flat) = decide(&both, WhenFlat::Stop).unwrap();
        assert!(stop && flat.len() == 2);
        let (stop, _) = decide(&both, WhenFlat::Advise).unwrap();
        assert!(!stop, "advise never stops");
    }

    #[test]
    fn nothing_is_judged_early_or_between_checks() {
        let mut watch = FlatWatch::new(PathBuf::from("/nonexistent"), WhenFlat::Stop);
        assert!(watch.after_generation(3).is_none());
        assert!(
            watch.after_generation(31).is_none(),
            "32 is not a multiple of 5"
        );
        assert!(
            watch.after_generation(34).is_none(),
            "35 generations, but the folder holds no run"
        );
        assert!(!watch.stopping());
    }

    fn flat_run_dir(name: &str, generations: u32) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("arschloch-flat-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut text = String::new();
        for g in 0..generations {
            let _ = writeln!(
                text,
                r#"{{"type":"generation","generation":{g},"champion":{{"reeval":{{"mean":0.5,"std_error":0.05}}}}}}"#
            );
        }
        std::fs::write(dir.join("events.jsonl"), text).unwrap();
        dir
    }

    #[test]
    fn a_resumed_run_is_not_judged_before_it_has_new_generations_of_its_own() {
        let dir = flat_run_dir("resume", 60);
        // A run that trained generations 55..=59 itself is judged at 60 and stops.
        let mut live = FlatWatch::new(dir.clone(), WhenFlat::Stop);
        for g in 55..59 {
            assert!(live.after_generation(g).is_none());
        }
        assert!(live.after_generation(59).is_some() && live.stopping());
        // The same history, resumed at generation 59: one new generation is not enough.
        let mut resumed = FlatWatch::new(dir.clone(), WhenFlat::Stop);
        assert!(resumed.after_generation(59).is_none() && !resumed.stopping());
        let _ = std::fs::remove_dir_all(dir);
    }
}
