//! The observer that prints a run: rows, the refreshed progress line, the
//! `ARSCHLOCH_PROGRESS` lines for the desktop window, and the summary.

use std::fmt::Write as _;
use std::io::{IsTerminal, Write as _};
use std::path::PathBuf;

use sim::training::{GenerationEvent, RunEnd, RunStart, TrainObserver};

use super::render::{
    format_duration, render_banner, render_header, render_row, render_summary, SetContext,
};

/// How often (in generations) the column header is repeated.
pub(super) const HEADER_EVERY: u32 = 20;

/// With this environment variable set, `cli train` prints `progress: FRACTION TEXT` lines on
/// stderr (the desktop window's progress bar reads them; the output is unchanged without it).
pub const PROGRESS_ENV: &str = "ARSCHLOCH_PROGRESS";
/// A progress line is printed when the fraction grew by at least this much.
pub(super) const PROGRESS_STEP: f64 = 0.005;

/// How far the run is: finished generations plus the share of this one's genomes evaluated.
pub(super) fn progress_fraction(
    generation: u32,
    done: usize,
    total: usize,
    generations: u32,
) -> f64 {
    if generations == 0 || total == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let within = done as f64 / total as f64;
    ((f64::from(generation) + within) / f64::from(generations)).clamp(0.0, 1.0)
}

/// Where a run's fraction lands in a set of `total_runs` runs, `finished` of them done.
pub(super) fn set_fraction(finished: usize, total_runs: u32, run_fraction: f64) -> f64 {
    if total_runs == 0 {
        return run_fraction;
    }
    #[allow(clippy::cast_precision_loss)]
    let before = finished as f64;
    ((before + run_fraction) / f64::from(total_runs)).clamp(0.0, 1.0)
}

/// `position` is `(k, N)` when this is run `k` of a set of `N`: the text then says so
/// (the fraction passed in is already the set's).
pub(super) fn progress_line(
    fraction: f64,
    generation: u32,
    done: usize,
    total: usize,
    generations: u32,
    position: Option<(usize, u32)>,
) -> String {
    let run = position.map_or_else(String::new, |(k, n)| format!("run {k}/{n} "));
    format!(
        "progress: {fraction:.4} {run}generation {}/{generations}, genomes {done}/{total}",
        generation + 1
    )
}

#[allow(clippy::struct_excessive_bools)] // independent display switches
pub struct TerminalObserver {
    set: Option<SetContext>,
    quiet: bool,
    out: PathBuf,
    live_progress: bool,
    progress_visible: bool,
    /// `ARSCHLOCH_PROGRESS` is set: print machine-readable `progress:` lines (the desktop window's bar).
    progress_lines: bool,
    last_progress: f64,
    opponent_count: usize,
    show_hall: bool,
    total_generations: u32,
    first_generation: Option<u32>,
    generation_secs: Vec<f64>,
    /// `--when-flat`: looks at the run's own events while it trains.
    flat: Option<crate::flat::FlatWatch>,
}

impl TerminalObserver {
    #[must_use]
    pub fn new(out: PathBuf, quiet: bool) -> Self {
        Self {
            set: None,
            quiet,
            out,
            live_progress: std::io::stderr().is_terminal(),
            progress_visible: false,
            progress_lines: std::env::var_os(PROGRESS_ENV).is_some(),
            last_progress: -1.0,
            opponent_count: 0,
            show_hall: false,
            total_generations: 0,
            first_generation: None,
            generation_secs: Vec::new(),
            flat: None,
        }
    }

    /// Advises or stops when the run's analysis finds the curve flat (`--when-flat`).
    #[must_use]
    pub fn with_flat(mut self, mode: crate::flat::WhenFlat) -> Self {
        self.flat = Some(crate::flat::FlatWatch::new(self.out.clone(), mode));
        self
    }

    /// Marks this run as part of a set (its rows then end with the set's ETA).
    #[must_use]
    pub fn with_set(mut self, set: SetContext) -> Self {
        self.set = Some(set);
        self
    }

    fn clear_progress(&mut self) {
        if self.progress_visible {
            eprint!("\r{:60}\r", "");
            self.progress_visible = false;
        }
    }

    fn header(&self) -> String {
        let mut header = render_header(self.opponent_count, self.show_hall);
        if self.set.is_some() {
            header.push_str("  set ETA");
        }
        header
    }

    fn eta(&self, event: &GenerationEvent) -> Option<f64> {
        if self.generation_secs.is_empty() {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let average = self.generation_secs.iter().sum::<f64>() / self.generation_secs.len() as f64;
        let remaining = self.total_generations.saturating_sub(event.generation + 1);
        Some(average * f64::from(remaining))
    }
}

impl TrainObserver for TerminalObserver {
    fn on_start(&mut self, start: &RunStart) {
        self.opponent_count = start.opponents.len();
        self.show_hall = start.config.hall_of_fame_size > 0;
        self.total_generations = start.config.generations;
        if !self.quiet {
            println!("{}\n\n{}", render_banner(start, &self.out), self.header());
        }
    }

    fn on_eval_progress(&mut self, generation: u32, done: usize, total: usize) {
        if self.progress_lines {
            let mut fraction = progress_fraction(generation, done, total, self.total_generations);
            let position = self
                .set
                .as_ref()
                .map(|set| (set.finished_secs.len() + 1, set.total_runs));
            if let Some(set) = &self.set {
                fraction = set_fraction(set.finished_secs.len(), set.total_runs, fraction);
            }
            if fraction - self.last_progress >= PROGRESS_STEP || done == total {
                self.last_progress = fraction;
                eprintln!(
                    "{}",
                    progress_line(
                        fraction,
                        generation,
                        done,
                        total,
                        self.total_generations,
                        position
                    )
                );
            }
        }
        if self.quiet || !self.live_progress {
            return;
        }
        eprint!("\r  generation {generation}: evaluated {done}/{total} genomes ");
        let _ = std::io::stderr().flush();
        self.progress_visible = true;
    }

    fn on_generation(&mut self, event: &GenerationEvent) {
        self.clear_progress();
        self.generation_secs.push(event.generation_secs);
        let first = *self.first_generation.get_or_insert(event.generation);
        let advice = self
            .flat
            .as_mut()
            .and_then(|flat| flat.after_generation(event.generation));
        if self.quiet {
            if let Some(advice) = advice {
                println!("{advice}");
            }
            return;
        }
        if (event.generation - first).is_multiple_of(HEADER_EVERY) && event.generation != first {
            println!("{}", self.header());
        }
        let mut row = render_row(event, self.eta(event), self.show_hall);
        if let Some(set) = &self.set {
            let eta = sim::training::set_eta(
                &set.finished_secs,
                set.total_runs,
                self.eta(event),
                event.elapsed_secs,
            );
            let _ = write!(
                row,
                "  set {}",
                eta.map_or_else(|| "--:--:--".to_owned(), format_duration)
            );
        }
        println!("{row}");
        if let Some(advice) = advice {
            println!("{advice}");
        }
    }

    fn should_stop(&mut self) -> bool {
        self.flat
            .as_ref()
            .is_some_and(crate::flat::FlatWatch::stopping)
    }

    fn on_finish(&mut self, end: &RunEnd) {
        self.clear_progress();
        println!("\n{}", render_summary(end, &self.out));
    }
}
