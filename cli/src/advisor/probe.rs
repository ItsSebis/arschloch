//! The probe: plays every candidate opponent against the reference players and
//! measures its strength, spread, speed and luck share.

use std::time::Instant;

use rayon::prelude::*;
use sim::training::evaluate::{evaluate_with_round1, Round1Obs};
use sim::training::skill_term::{adjusted_score, fit_deal_model};
use sim::training::{match_seed, Opponents, Score, TableSpec};

use super::plan::{judge, pooled_sd, Verdict};
use super::references::Reference;
use super::AdvisorArgs;
use crate::args::StrategyArg;

/// What the probe measured about one candidate opponent.
pub(super) struct Row {
    pub(super) spec: String,
    /// The reference players' mean score against it (+1 best .. -1 worst) and its standard error.
    pub(super) mean: f64,
    pub(super) error: f64,
    pub(super) verdict: Verdict,
    /// The spread of one match's score against it.
    pub(super) sd: f64,
    /// Rounds played and seconds spent probing this candidate.
    pub(super) rounds: usize,
    pub(super) seconds: f64,
    /// Variance of one match's round-1 score, and of the same score after the regression on
    /// the six hand features (`skill_term`): what the luck adjustment would leave of it.
    pub(super) var_round1: f64,
    pub(super) var_adjusted: f64,
    /// How many round-1 observations those two variances rest on.
    pub(super) observations: usize,
}

/// What the probe measured.
pub(super) struct Probe {
    pub(super) rows: Vec<Row>,
    /// Rounds per second over the opponents of the recommended pool only (set once the pool
    /// is chosen): a dropped candidate that plays slower or faster must not set the speed.
    pub(super) rounds_per_sec: f64,
    /// Standard deviation of one match's score.
    pub(super) sd_match: f64,
    /// Share of the round-1 variance that the hand explains (pool opponents only); `None` when
    /// fewer than `MIN_LUCK_OBSERVATIONS` observations stand behind it.
    pub(super) luck_share: Option<f64>,
    /// The round-1 observations of the pool opponents (the luck share's sample size).
    pub(super) luck_observations: usize,
}

/// A fit of six features on fewer round-1 observations than this says nothing about luck.
pub(super) const MIN_LUCK_OBSERVATIONS: usize = 30;

/// Probe matches against each reference player: the candidate's matches split over them,
/// at least 2 against each (so a candidate plays `per_reference * references` in all).
pub(super) fn matches_per_reference(probe_matches: usize, references: usize) -> usize {
    probe_matches.div_ceil(references).max(2)
}

/// Rounds per second over the given probe rows.
pub(super) fn speed_of(rows: &[&Row]) -> f64 {
    let seconds: f64 = rows.iter().map(|r| r.seconds).sum();
    let rounds: usize = rows.iter().map(|r| r.rounds).sum();
    rounds as f64 / seconds.max(1e-9)
}

pub(super) fn variance(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

/// Variances of the round-1 scores and of the luck-adjusted ones, from one fit over all of them.
pub(super) fn round1_variances(observations: &[Round1Obs]) -> (f64, f64) {
    let Some(model) = fit_deal_model(std::slice::from_ref(&observations.to_vec())) else {
        return (0.0, 0.0);
    };
    let plain: Vec<f64> = observations.iter().map(|o| o.score).collect();
    let adjusted: Vec<f64> = observations
        .iter()
        .map(|o| adjusted_score(&model, std::slice::from_ref(o)))
        .collect();
    (variance(&plain), variance(&adjusted))
}

pub(super) fn run_probe(
    args: &AdvisorArgs,
    candidates: &[String],
    references: &[Reference],
    table: &TableSpec,
) -> anyhow::Result<Probe> {
    let mut rows = Vec::new();
    // The probe matches of a candidate are split over the references.
    let per_reference = matches_per_reference(args.probe_matches, references.len());
    for (index, spec) in candidates.iter().enumerate() {
        let opponent = spec
            .parse::<StrategyArg>()
            .map_err(anyhow::Error::msg)?
            .build();
        let (mut means, mut variances, mut sds) = (Vec::new(), 0.0, Vec::new());
        let (mut rounds, mut observations) = (0usize, Vec::new());
        let started = Instant::now();
        // All reference x seed matches of this candidate run in one parallel batch (the
        // references side by side, their matches inside), so the clock does not include
        // the idle threads of several small batches one after the other.
        let results: Vec<(Score, Vec<Round1Obs>)> = references
            .par_iter()
            .enumerate()
            .map(|(r, reference)| {
                let seeds: Vec<u64> = (0..per_reference as u64)
                    .map(|i| match_seed(args.seed, 0xAD + index as u64 + 0x1000 * r as u64, i))
                    .collect();
                evaluate_with_round1(
                    &reference.strategy,
                    table,
                    Opponents::Only(&opponent),
                    &seeds,
                )
            })
            .collect();
        for (score, obs) in results {
            rounds += score.matches * table.rounds;
            means.push(score.mean);
            variances += score.std_error * score.std_error;
            sds.push(score.std_error * (score.matches as f64).sqrt());
            observations.extend(obs);
        }
        let seconds = started.elapsed().as_secs_f64();
        let k = references.len() as f64;
        let mean = means.iter().sum::<f64>() / k;
        let (var_round1, var_adjusted) = round1_variances(&observations);
        rows.push(Row {
            spec: spec.clone(),
            mean,
            error: variances.sqrt() / k,
            verdict: judge(mean),
            sd: pooled_sd(&sds),
            rounds,
            seconds,
            var_round1,
            var_adjusted,
            observations: observations.len(),
        });
    }
    Ok(Probe {
        rows,
        rounds_per_sec: 0.0,
        sd_match: 0.0,
        luck_share: None,
        luck_observations: 0,
    })
}
