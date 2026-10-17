//! Run analysis: where a training run stopped improving and why. Pure
//! functions over the events a run wrote (`events.jsonl`); the CLI and the
//! web page both show `RunAnalysis`. All numbers are on the champion's
//! fixed-set re-evaluation score, the one `docs/baselines/phase-13/
//! analyse.py` uses. See docs/superpowers/specs/2026-10-11-phase-17-run-analysis-design.md.

mod lean;
mod load;
mod stats;
#[cfg(test)]
mod tests;
mod verdict;

use serde::Serialize;

use super::events::StageTimings;

pub use lean::{LeanEnd, LeanGeneration, LeanScore, RunStartInfo};
pub use load::{
    load_runs, load_runs_within, parse_run, qualify_labels, AnalysisError, Loaded, RunData,
};
pub use stats::{slope_with_interval, t_quantile, Slope};
pub use verdict::{verdict_for, Verdict, VerdictCode};

use stats::{argmax, mean, smooth, stall_index, std_dev};

/// Trailing-mean window of the smoothed champion curve.
const SMOOTHING_WINDOW: usize = 10;
/// Generations are reported per this many.
const PER: f64 = 100.0;

/// A score and its standard error.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ScoreSummary {
    pub mean: f64,
    pub std_error: f64,
}

impl From<&LeanScore> for ScoreSummary {
    fn from(stat: &LeanScore) -> Self {
        Self {
            mean: stat.mean,
            std_error: stat.std_error,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Noise {
    /// Mean standard error of one champion re-evaluation.
    pub single_se: f64,
    /// Standard deviation of the generation-to-generation change of the
    /// re-evaluation mean (`None` below three generations: it needs two
    /// changes).
    pub change_sd: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Trend {
    /// Slope per 100 generations over the whole run (`None` below three
    /// generations).
    pub overall: Option<Slope>,
    /// The same over the last third of the generations.
    pub last_third: Option<Slope>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct HeldoutGap {
    /// `best_reeval.mean - best_heldout.mean`.
    pub value: f64,
    /// In held-out standard errors (`None` when that error is zero).
    pub in_heldout_se: Option<f64>,
    /// In `sqrt(se_reeval^2 + se_heldout^2)` units, the error of the
    /// difference of the two measurements (`None` when that is zero).
    pub in_combined_se: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Curse {
    /// The winner's-curse gap; `None` without `run_end` data.
    pub reeval_minus_heldout: Option<HeldoutGap>,
    /// Mean over generations of `train_fitness - reeval.mean`.
    pub train_minus_reeval: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrendClass {
    Rising,
    Flat,
    Falling,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OpponentAnalysis {
    pub name: String,
    /// Mean of the champion's score against this opponent over the first
    /// and the last third of the run's generations (the same windows as the
    /// champion's trend); `None` when no generation of the window reports it.
    pub first_third_mean: Option<f64>,
    pub last_third_mean: Option<f64>,
    /// Slope per 100 generations over the whole run.
    pub overall_slope: Option<Slope>,
    /// Slope per 100 generations over the last third: the window of the
    /// champion's last-third trend, and the one `class` is taken from.
    pub last_third_slope: Option<Slope>,
    pub class: Option<TrendClass>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Growth {
    pub first_hidden_nodes: usize,
    pub last_hidden_nodes: usize,
    pub first_connections: usize,
    pub last_connections: usize,
    /// Mean hidden-node count of the first `third_window` generations of
    /// the last third, and of its last `third_window` generations.
    pub third_start_nodes: f64,
    pub third_end_nodes: f64,
    pub third_window: usize,
    /// The node count grew within the last third (`third_end_nodes >
    /// third_start_nodes`) while the last-third trend interval contains zero.
    pub without_gain: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PopulationShape {
    /// Mean over the last third of the share of genomes in the top
    /// fitness-histogram bin (`None` without histograms).
    pub top_bin_share: Option<f64>,
    pub species_first: usize,
    pub species_last: usize,
    pub threshold_first: f64,
    pub threshold_last: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct RacingTotals {
    pub generations: usize,
    /// Extra matches played by raced genomes over the run.
    pub extra_matches: u64,
    /// Share of all matches (training plus extra) that were extra; needs
    /// the run's configuration.
    pub extra_share: Option<f64>,
    /// Mean over generations of the racing fitness shift.
    pub mean_shift: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RunAnalysis {
    pub generations: usize,
    pub best_generation: Option<u32>,
    pub final_reeval: Option<ScoreSummary>,
    pub final_heldout: Option<ScoreSummary>,
    /// Number of the last generation (`generations` counts events, and a
    /// run need not start at 0 or be gap free).
    pub last_generation: u32,
    /// Peak of the smoothed re-evaluation curve.
    pub peak_smoothed: f64,
    /// Where the peak is: an index into the generations, and the generation
    /// number there.
    pub peak_index: usize,
    pub peak_generation: u32,
    /// The smoothed value at the last generation.
    pub final_smoothed: f64,
    /// Generation NUMBER from which the smoothed curve stays within one
    /// mean standard error of its peak (`stall_index` is the position in
    /// the list of generations).
    pub stall_generation: u32,
    pub stall_index: usize,
    /// The final smoothed value is itself below `peak - single_se`: no
    /// stall exists and `stall_index` is the last index (the fallback of
    /// `analyse.py`); the run ended below its peak.
    pub stall_fallback: bool,
    pub noise: Noise,
    pub trend: Trend,
    pub curse: Curse,
    pub opponents: Vec<OpponentAnalysis>,
    pub growth: Growth,
    pub population: PopulationShape,
    pub racing: Option<RacingTotals>,
    /// Seconds per stage summed over the run, when it was recorded.
    pub timings: Option<StageTimings>,
    /// Lines of the file that were not events.
    pub skipped_lines: usize,
    pub verdict: Vec<Verdict>,
}

/// One opponent's per-generation scores (`None` where not reported).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OpponentSeries {
    pub name: String,
    pub means: Vec<Option<f64>>,
}

/// Per-generation arrays for the charts.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RunSeries {
    pub generation: Vec<u32>,
    pub reeval_mean: Vec<f64>,
    pub reeval_se: Vec<f64>,
    pub smoothed: Vec<f64>,
    /// `None` where the event does not carry the training fitness.
    pub train_fitness: Vec<Option<f64>>,
    /// Held-out mean of the generations that set a new best.
    pub heldout_mean: Vec<Option<f64>>,
    pub opponents: Vec<OpponentSeries>,
}

/// Opponent names in order of first appearance.
fn opponent_names(run: &RunData) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for g in &run.generations {
        for o in &g.opponents {
            if !names.contains(&o.name) {
                names.push(o.name.clone());
            }
        }
    }
    names
}

fn opponent_score(g: &LeanGeneration, name: &str) -> Option<f64> {
    g.opponents
        .iter()
        .find(|o| o.name == name)
        .map(|o| o.score.mean)
}

/// The chart arrays of a run.
#[must_use]
pub fn series(run: &RunData) -> RunSeries {
    let gens = &run.generations;
    let reeval_mean: Vec<f64> = gens.iter().map(|g| g.champion.reeval.mean).collect();
    RunSeries {
        generation: gens.iter().map(|g| g.generation).collect(),
        smoothed: smooth(&reeval_mean, SMOOTHING_WINDOW),
        reeval_mean,
        reeval_se: gens.iter().map(|g| g.champion.reeval.std_error).collect(),
        train_fitness: gens.iter().map(|g| g.champion.train_fitness).collect(),
        heldout_mean: gens
            .iter()
            .map(|g| g.champion.heldout.as_ref().map(|h| h.mean))
            .collect(),
        opponents: opponent_names(run)
            .into_iter()
            .map(|name| OpponentSeries {
                means: gens.iter().map(|g| opponent_score(g, &name)).collect(),
                name,
            })
            .collect(),
    }
}

/// First and last third of `0..n` (each at least one element).
fn thirds(n: usize) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
    let third = (n / 3).max(1).min(n);
    (0..third, n - third..n)
}

fn class_of(slope: &Slope) -> TrendClass {
    if slope.is_positive() {
        TrendClass::Rising
    } else if slope.is_negative() {
        TrendClass::Falling
    } else {
        TrendClass::Flat
    }
}

/// The `(generation, score)` points of an opponent within a window of
/// generation indices.
fn opponent_points(
    run: &RunData,
    name: &str,
    window: std::ops::Range<usize>,
) -> (Vec<f64>, Vec<f64>) {
    run.generations[window]
        .iter()
        .filter_map(|g| opponent_score(g, name).map(|m| (f64::from(g.generation), m)))
        .unzip()
}

fn opponent_analysis(
    run: &RunData,
    name: String,
    first: &std::ops::Range<usize>,
    last: &std::ops::Range<usize>,
) -> OpponentAnalysis {
    let slope = |window: std::ops::Range<usize>| {
        let (xs, ys) = opponent_points(run, &name, window);
        slope_with_interval(&xs, &ys).map(|s| s.scaled(PER))
    };
    let window_mean = |window: &std::ops::Range<usize>| {
        let (_, ys) = opponent_points(run, &name, window.clone());
        (!ys.is_empty()).then(|| mean(&ys))
    };
    let last_third_slope = slope(last.clone());
    OpponentAnalysis {
        first_third_mean: window_mean(first),
        last_third_mean: window_mean(last),
        overall_slope: slope(0..run.generations.len()),
        class: last_third_slope.as_ref().map(class_of),
        last_third_slope,
        name,
    }
}

#[allow(clippy::cast_precision_loss)]
fn top_bin_share(g: &LeanGeneration) -> Option<f64> {
    let total: u32 = g.fitness.histogram.iter().sum();
    let top = g.fitness.histogram.last()?;
    (total > 0).then(|| f64::from(*top) / f64::from(total))
}

#[allow(clippy::cast_precision_loss)]
fn racing_totals(run: &RunData) -> Option<RacingTotals> {
    let racing: Vec<_> = run
        .generations
        .iter()
        .filter_map(|g| g.racing.as_ref())
        .collect();
    if racing.is_empty() {
        return None;
    }
    let extra_matches: u64 = racing
        .iter()
        .map(|r| (r.raced_final.max(r.raced) * r.extra_matches) as u64)
        .sum();
    let extra_share = run.start.as_ref().and_then(|s| {
        let base = (run.generations.len() * s.population_size()? * s.matches_per_genome()?) as f64;
        Some(extra_matches as f64 / (extra_matches as f64 + base))
    });
    Some(RacingTotals {
        generations: racing.len(),
        extra_matches,
        extra_share,
        mean_shift: mean(&racing.iter().map(|r| r.mean_shift).collect::<Vec<_>>()),
    })
}

fn timing_totals(run: &RunData) -> Option<StageTimings> {
    let mut any = false;
    let mut total = StageTimings::default();
    for t in run.generations.iter().filter_map(|g| g.timings.as_ref()) {
        any = true;
        total.training_evaluation += t.training_evaluation;
        total.champion_selection += t.champion_selection;
        total.speciation_and_reproduction += t.speciation_and_reproduction;
        total.reevaluation_mixed += t.reevaluation_mixed;
        total.reevaluation_per_opponent += t.reevaluation_per_opponent;
        total.hall_of_fame += t.hall_of_fame;
        total.confirmation += t.confirmation;
        total.decision_sample += t.decision_sample;
        total.checkpoint_and_files += t.checkpoint_and_files;
    }
    any.then_some(total)
}

fn heldout_gap(run: &RunData) -> Option<HeldoutGap> {
    let end = run.end.as_ref()?;
    let reeval = end.best_reeval.as_ref()?;
    let heldout = end.best_heldout.as_ref()?;
    let value = reeval.mean - heldout.mean;
    let combined = reeval.std_error.hypot(heldout.std_error);
    Some(HeldoutGap {
        value,
        in_heldout_se: (heldout.std_error > 0.0).then(|| value / heldout.std_error),
        in_combined_se: (combined > 0.0).then(|| value / combined),
    })
}

/// Mean hidden-node count over `gens[range]`.
#[allow(clippy::cast_precision_loss)]
fn mean_nodes(gens: &[LeanGeneration]) -> f64 {
    mean(
        &gens
            .iter()
            .map(|g| g.champion.hidden_nodes as f64)
            .collect::<Vec<_>>(),
    )
}

/// Analyses a run.
///
/// # Panics
///
/// Panics if `run` has no generation events (`load_runs` never returns one).
#[must_use]
pub fn analyse(run: &RunData) -> RunAnalysis {
    let gens = &run.generations;
    assert!(!gens.is_empty(), "a run to analyse has generation events");
    let n = gens.len();
    let (first, last) = thirds(n);

    let xs: Vec<f64> = gens.iter().map(|g| f64::from(g.generation)).collect();
    let means: Vec<f64> = gens.iter().map(|g| g.champion.reeval.mean).collect();
    let ses: Vec<f64> = gens.iter().map(|g| g.champion.reeval.std_error).collect();
    let smoothed = smooth(&means, SMOOTHING_WINDOW);
    let single_se = mean(&ses);
    let changes: Vec<f64> = means.windows(2).map(|w| w[1] - w[0]).collect();
    let peak_index = argmax(&smoothed).unwrap_or(0);
    let stall = stall_index(&smoothed, single_se).expect("a non-empty curve has a stall");

    let trend = Trend {
        overall: slope_with_interval(&xs, &means).map(|s| s.scaled(PER)),
        last_third: slope_with_interval(&xs[last.clone()], &means[last.clone()])
            .map(|s| s.scaled(PER)),
    };
    let train_gap: Vec<f64> = gens
        .iter()
        .filter_map(|g| g.champion.train_fitness.map(|t| t - g.champion.reeval.mean))
        .collect();

    let (first_gen, last_gen) = (&gens[0], &gens[n - 1]);
    let window = (last.len() / 2).clamp(1, 3);
    let third_start_nodes = mean_nodes(&gens[last.start..last.start + window]);
    let third_end_nodes = mean_nodes(&gens[n - window..]);
    let growth = Growth {
        first_hidden_nodes: first_gen.champion.hidden_nodes,
        last_hidden_nodes: last_gen.champion.hidden_nodes,
        first_connections: first_gen.champion.enabled_connections,
        last_connections: last_gen.champion.enabled_connections,
        third_start_nodes,
        third_end_nodes,
        third_window: window,
        without_gain: third_end_nodes > third_start_nodes
            && trend.last_third.is_some_and(|s| s.contains_zero()),
    };
    let shares: Vec<f64> = gens[last.clone()]
        .iter()
        .filter_map(top_bin_share)
        .collect();
    let population = PopulationShape {
        top_bin_share: (!shares.is_empty()).then(|| mean(&shares)),
        species_first: first_gen.species.len(),
        species_last: last_gen.species.len(),
        threshold_first: first_gen.compatibility_threshold,
        threshold_last: last_gen.compatibility_threshold,
    };

    let mut analysis = RunAnalysis {
        generations: n,
        last_generation: last_gen.generation,
        best_generation: run.end.as_ref().and_then(|e| e.best_generation),
        final_reeval: run
            .end
            .as_ref()
            .and_then(|e| e.best_reeval.as_ref())
            .map(ScoreSummary::from),
        final_heldout: run
            .end
            .as_ref()
            .and_then(|e| e.best_heldout.as_ref())
            .map(ScoreSummary::from),
        peak_smoothed: smoothed[peak_index],
        peak_index,
        peak_generation: gens[peak_index].generation,
        final_smoothed: smoothed[n - 1],
        stall_generation: gens[stall.index].generation,
        stall_index: stall.index,
        stall_fallback: stall.fallback,
        noise: Noise {
            single_se,
            change_sd: (changes.len() >= 2).then(|| std_dev(&changes)),
        },
        trend,
        curse: Curse {
            reeval_minus_heldout: heldout_gap(run),
            train_minus_reeval: mean(&train_gap),
        },
        opponents: opponent_names(run)
            .into_iter()
            .map(|name| opponent_analysis(run, name, &first, &last))
            .collect(),
        growth,
        population,
        racing: racing_totals(run),
        timings: timing_totals(run),
        skipped_lines: run.skipped_lines,
        verdict: Vec::new(),
    };
    // Without a run_end (a run still training, or stopped) the best generation is the best
    // re-evaluation mean in the events, so the stop-early guard works the same mid-run as on
    // the finished run; the reported `best_generation` stays null.
    let mut judged = analysis.clone();
    if judged.best_generation.is_none() {
        judged.best_generation = argmax(&means).map(|i| gens[i].generation);
    }
    analysis.verdict = verdict::verdict_for(&judged);
    analysis
}
