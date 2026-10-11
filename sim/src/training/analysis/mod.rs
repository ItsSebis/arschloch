//! Run analysis: where a training run stopped improving and why. Pure
//! functions over the events a run wrote (`events.jsonl`); the CLI and the
//! web page both show `RunAnalysis`. All numbers are on the champion's
//! fixed-set re-evaluation score, the one `docs/baselines/phase-13/
//! analyse.py` uses. See docs/superpowers/specs/2026-10-11-phase-17-run-analysis-design.md.

mod load;
mod stats;
#[cfg(test)]
mod tests;
mod verdict;

use serde::Serialize;

use super::events::{GenerationEvent, ScoreStat, StageTimings};

pub use load::{load_runs, parse_run, AnalysisError, RunData};
pub use stats::{slope_with_interval, Slope};
pub use verdict::{verdict_for, Verdict, VerdictCode};

use stats::{mean, smooth, stall_index, std_dev};

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

impl From<&ScoreStat> for ScoreSummary {
    fn from(stat: &ScoreStat) -> Self {
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
    /// re-evaluation mean (`None` below two generations).
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
    /// and the last third of the generations that report it.
    pub first_third_mean: f64,
    pub last_third_mean: f64,
    /// Per 100 generations.
    pub slope: Option<Slope>,
    pub class: Option<TrendClass>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Growth {
    pub first_hidden_nodes: usize,
    pub last_hidden_nodes: usize,
    pub first_connections: usize,
    pub last_connections: usize,
    /// Nodes grew in the last third while the last-third trend interval
    /// contains zero.
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
    /// Peak of the smoothed re-evaluation curve.
    pub peak_smoothed: f64,
    /// Index (into the generations) from which the smoothed curve stays
    /// within one mean standard error of its peak.
    pub stall_generation: usize,
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
    pub train_fitness: Vec<f64>,
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

fn opponent_score(g: &GenerationEvent, name: &str) -> Option<f64> {
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

fn opponent_analysis(run: &RunData, name: String) -> OpponentAnalysis {
    // Only generations that report the opponent take part, so thirds are
    // taken over those.
    let points: Vec<(f64, f64)> = run
        .generations
        .iter()
        .filter_map(|g| opponent_score(g, &name).map(|m| (f64::from(g.generation), m)))
        .collect();
    let (first, last) = thirds(points.len());
    let ys: Vec<f64> = points.iter().map(|p| p.1).collect();
    let xs: Vec<f64> = points.iter().map(|p| p.0).collect();
    let slope = slope_with_interval(&xs, &ys).map(|s| s.scaled(PER));
    OpponentAnalysis {
        first_third_mean: mean(&ys[first]),
        last_third_mean: mean(&ys[last]),
        class: slope.as_ref().map(class_of),
        slope,
        name,
    }
}

#[allow(clippy::cast_precision_loss)]
fn top_bin_share(g: &GenerationEvent) -> Option<f64> {
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
    let extra_share = run.start.as_ref().map(|s| {
        let base = (run.generations.len()
            * s.config.neat.population_size
            * s.config.matches_per_genome) as f64;
        extra_matches as f64 / (extra_matches as f64 + base)
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
    Some(HeldoutGap {
        value,
        in_heldout_se: (heldout.std_error > 0.0).then(|| value / heldout.std_error),
    })
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
    let (_, last) = thirds(n);

    let xs: Vec<f64> = gens.iter().map(|g| f64::from(g.generation)).collect();
    let means: Vec<f64> = gens.iter().map(|g| g.champion.reeval.mean).collect();
    let ses: Vec<f64> = gens.iter().map(|g| g.champion.reeval.std_error).collect();
    let smoothed = smooth(&means, SMOOTHING_WINDOW);
    let single_se = mean(&ses);
    let changes: Vec<f64> = means.windows(2).map(|w| w[1] - w[0]).collect();

    let trend = Trend {
        overall: slope_with_interval(&xs, &means).map(|s| s.scaled(PER)),
        last_third: slope_with_interval(&xs[last.clone()], &means[last.clone()])
            .map(|s| s.scaled(PER)),
    };
    let train_gap: Vec<f64> = gens
        .iter()
        .map(|g| g.champion.train_fitness - g.champion.reeval.mean)
        .collect();

    let (first_gen, last_gen) = (&gens[0], &gens[n - 1]);
    let nodes_at_last_third_start = gens[last.start].champion.hidden_nodes;
    let growth = Growth {
        first_hidden_nodes: first_gen.champion.hidden_nodes,
        last_hidden_nodes: last_gen.champion.hidden_nodes,
        first_connections: first_gen.champion.enabled_connections,
        last_connections: last_gen.champion.enabled_connections,
        without_gain: last_gen.champion.hidden_nodes > nodes_at_last_third_start
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
        peak_smoothed: smoothed.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        stall_generation: stall_index(&smoothed, single_se).unwrap_or(0),
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
            .map(|name| opponent_analysis(run, name))
            .collect(),
        growth,
        population,
        racing: racing_totals(run),
        timings: timing_totals(run),
        skipped_lines: run.skipped_lines,
        verdict: Vec::new(),
    };
    analysis.verdict = verdict::verdict_for(&analysis);
    analysis
}
