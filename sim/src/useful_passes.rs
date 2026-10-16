//! Counterfactual measurement of whether voluntary passes are *useful*
//! (docs/STATISTICS.md ids `useful_pass_share` and
//! `useful_pass_gain`).
//!
//! # Definition
//!
//! A *voluntary* pass is a pass chosen when at least one `Move::Play` was
//! legal. For a sampled voluntary pass of seat `s` the round is cloned at
//! that exact state and finished `K` times for each of three candidate
//! moves: the pass itself, the *weakest* legal play (lowest top card) and
//! the *strongest* legal play (highest top card); these are the two
//! extremes of the engine's canonical candidate set. Every rollout applies
//! the move and finishes the round with [`play_out`] using the same
//! strategies the match uses. Rollout `k` of one decision uses the same
//! deterministic rng seed for all three candidates (derived from the
//! analysis seed, match, round, decision and `k`), so the candidates are
//! compared on common random numbers.
//!
//! `mean_place(m)` is the mean of `s`'s finishing place (1 = best .. n =
//! worst, its index in `Round::finishing_order` plus one) over the `K`
//! rollouts of candidate `m`. The best alternative is the better (lower)
//! of the weakest and strongest play. Then:
//!
//! * `gain = mean_place(best alternative) - mean_place(pass)` (positive:
//!   passing was better); the mean over all sampled voluntary passes,
//!   useful or not, is `useful_pass_gain`, so it can be negative;
//! * the pass is *useful* iff `mean_place(pass) < mean_place(best
//!   alternative) - margin` (`margin` defaults to 0: strictly better);
//!   the share of useful passes among the sampled ones is
//!   `useful_pass_share`.
//!
//! # Replay
//!
//! The analysis never alters the original matches. [`replay_match`]
//! re-simulates a match with `run_match_observed`, the driver of
//! `run_match_with` with a hook on voluntary passes (same seeds, same rng
//! stream: rollouts use their own generators and never touch the match
//! stream), so its [`MatchResult`]
//! equals the original and the voluntary passes occur at the same states.
//! A voluntary pass is sampled when a hash of (analysis seed, match seed,
//! match index, round, decision index) falls below `sample_fraction`; the
//! outcome therefore depends only on the inputs, not on thread count.

mod rollout;

pub use self::rollout::{evaluate_pass, replay_match};

use std::collections::BTreeMap;
use std::sync::Arc;

use rayon::prelude::*;

use crate::extended_stats::{standard_error, wilson_interval, Estimate, Interval};
use crate::match_config::MatchConfig;
use crate::match_runner::RunOptions;
use crate::strategy::Strategy;

/// Settings of the useful-pass analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UsefulPassOptions {
    /// Rollouts `K` per candidate move (default 16).
    pub rollouts: usize,
    /// Fraction of voluntary passes analysed, `0..=1` (default 0.05).
    pub sample_fraction: f64,
    /// A pass is useful only if it beats the best alternative by more than
    /// this many places (default 0: strictly better).
    pub margin: f64,
    /// Seed of the sampling hash and the rollout generators.
    pub seed: u64,
}

impl Default for UsefulPassOptions {
    fn default() -> Self {
        Self {
            rollouts: 16,
            sample_fraction: 0.05,
            margin: 0.0,
            seed: 0,
        }
    }
}

/// The verdict on one voluntary pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PassVerdict {
    pub mean_place_pass: f64,
    pub mean_place_weakest: f64,
    pub mean_place_strongest: f64,
    /// `min(weakest, strongest)` mean place.
    pub mean_place_best_alternative: f64,
    /// `mean_place_best_alternative - mean_place_pass`.
    pub gain: f64,
    pub useful: bool,
}

/// A share with its binomial standard error and Wilson 95% interval.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct ShareEstimate {
    pub value: f64,
    pub std_error: f64,
    pub n: usize,
    pub interval: Interval,
}

/// The useful-pass numbers of one strategy name.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct StrategyUsefulPasses {
    /// All voluntary passes of the strategy in the analysed matches.
    pub voluntary_passes_total: u64,
    /// How many of them the sampling selected and rolled out.
    pub sampled_voluntary_passes: u64,
    /// How many sampled passes were useful.
    pub useful: u64,
    /// Useful share of the sampled passes; `None` without samples.
    pub useful_pass_share: Option<ShareEstimate>,
    /// Mean gain in places over the sampled passes (standard error over
    /// the sampled passes, treated as independent); `None` without samples.
    pub useful_pass_gain: Option<Estimate>,
}

/// Useful-pass statistics per strategy name.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct UsefulPassStats {
    pub by_strategy: BTreeMap<String, StrategyUsefulPasses>,
}

/// Raw per-strategy samples; merge in a fixed order for determinism.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UsefulPassTally {
    voluntary_total: u64,
    useful: u64,
    gains: Vec<f64>,
}

/// Per-match analysis output, keyed by strategy name.
pub type MatchUsefulPasses = BTreeMap<String, UsefulPassTally>;

/// Folds per-match analyses (in the given order) into statistics.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn summarise(matches: &[MatchUsefulPasses]) -> UsefulPassStats {
    let mut merged: BTreeMap<String, UsefulPassTally> = BTreeMap::new();
    for one in matches {
        for (name, tally) in one {
            let into = merged.entry(name.clone()).or_default();
            into.voluntary_total += tally.voluntary_total;
            into.useful += tally.useful;
            into.gains.extend_from_slice(&tally.gains);
        }
    }
    let by_strategy = merged
        .into_iter()
        .map(|(name, tally)| {
            let sampled = tally.gains.len() as u64;
            let share = wilson_interval(tally.useful, sampled).map(|(low, high)| {
                let n = sampled as f64;
                let p = tally.useful as f64 / n;
                ShareEstimate {
                    value: p,
                    std_error: (p * (1.0 - p) / n).sqrt(),
                    n: tally.gains.len(),
                    interval: Interval { low, high },
                }
            });
            let gain = (!tally.gains.is_empty()).then(|| Estimate {
                value: tally.gains.iter().sum::<f64>() / tally.gains.len() as f64,
                std_error: standard_error(&tally.gains),
                n: tally.gains.len(),
            });
            (
                name,
                StrategyUsefulPasses {
                    voluntary_passes_total: tally.voluntary_total,
                    sampled_voluntary_passes: sampled,
                    useful: tally.useful,
                    useful_pass_share: share,
                    useful_pass_gain: gain,
                },
            )
        })
        .collect();
    UsefulPassStats { by_strategy }
}

/// Replays every config (seats rotated by the config's position, exactly
/// as `run_batch` does, with `strategies` the unrotated list) and analyses
/// the sampled voluntary passes. Parallel over matches; the result does
/// not depend on the thread count. The original matches are not altered.
///
/// # Panics
///
/// As `replay_match`.
#[must_use]
pub fn analyse_useful_passes(
    configs: &[MatchConfig],
    strategies: &[Arc<dyn Strategy>],
    options: &UsefulPassOptions,
) -> UsefulPassStats {
    let per_match: Vec<MatchUsefulPasses> = configs
        .par_iter()
        .enumerate()
        .map(|(index, config)| {
            let rotation = index % strategies.len();
            let rotated: Vec<Arc<dyn Strategy>> = strategies
                .iter()
                .cycle()
                .skip(rotation)
                .take(strategies.len())
                .cloned()
                .collect();
            replay_match(config, &rotated, &RunOptions::default(), index, options).1
        })
        .collect();
    summarise(&per_match)
}
