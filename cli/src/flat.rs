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
}

impl FlatWatch {
    #[must_use]
    pub fn new(dir: PathBuf, mode: WhenFlat) -> Self {
        Self {
            dir,
            mode,
            advised: Vec::new(),
            stopping: false,
        }
    }

    #[must_use]
    pub fn stopping(&self) -> bool {
        self.stopping
    }

    /// Called after generation number `generation` is written; the text to print, if any.
    pub fn after_generation(&mut self, generation: u32) -> Option<String> {
        let completed = generation + 1;
        if completed < MIN_GENERATIONS || !completed.is_multiple_of(CHECK_EVERY) {
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
                "\n  stopping here (--when-flat stop); `cli train --out {} --resume --generations N` continues the run.",
                self.dir.display()
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
}
