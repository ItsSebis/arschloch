//! The cheap luck estimator: regress each seat's round-1 role score on
//! its hand features plus one fixed effect per strategy name, then
//! subtract the part of the score the deal explains.

use super::{estimate, luck_share, mean, reduction, variance, Estimate};
use crate::hand_features::HandFeatures;
use crate::linalg::ols;
use crate::match_result::MatchResult;
use crate::training::evaluate::role_score;

const FEATURE_NAMES: [&str; 7] = [
    "high_cards",
    "pairs",
    "triples",
    "quads",
    "lowest_strength",
    "mean_strength",
    "hand_size",
];

/// Estimator result for one strategy name.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct EstimatorStrategy {
    pub name: String,
    /// Mean over the strategy's matches of its round-1 score after
    /// subtracting `beta . (features - mean features)`; `std_error` is the
    /// sd of the per-match values / sqrt(matches), `n` = matches. (Seats
    /// with the same name inside one match are averaged first.)
    pub skill_score_estimate: Estimate,
    /// The unadjusted round-1 mean over the same matches.
    pub plain_mean_role_score_round1: Estimate,
    /// Var(raw per-match round-1 score) / Var(adjusted), capped at 1e6.
    pub variance_reduction: f64,
    /// `1 - 1/variance_reduction` clamped to `[0, 1]`.
    pub luck_share: f64,
    /// Share of this strategy's round-1 seat-score variance explained by
    /// the feature part of the regression, `Var(beta . x) / Var(y)`,
    /// clamped to `[0, 1]`.
    pub variance_explained: f64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct EstimatorReport {
    /// Matches that carried hand features and entered the regression.
    pub match_count: usize,
    pub feature_names: Vec<&'static str>,
    /// One OLS coefficient per feature (strategy fixed effects omitted).
    /// Note `hand_size` is constant (collinear with the fixed effects, so
    /// its coefficient is arbitrary but harmless) when all hands have the
    /// same size.
    pub feature_coefficients: Vec<f64>,
    /// Partial R^2 of the features over the fixed-effects-only model:
    /// `1 - SSR(full) / SSR(fixed effects only)`.
    pub feature_r2: f64,
    /// One entry per strategy name, in first-seat order.
    pub strategies: Vec<EstimatorStrategy>,
}

struct Observation {
    match_index: usize,
    name: usize,
    features: [f64; 7],
    score: f64,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn empty_report() -> EstimatorReport {
    EstimatorReport {
        match_count: 0,
        feature_names: FEATURE_NAMES.to_vec(),
        feature_coefficients: vec![0.0; FEATURE_NAMES.len()],
        feature_r2: 0.0,
        strategies: Vec::new(),
    }
}

/// Fits the estimator on `results` (matches without
/// `first_hand_features` are ignored; no usable match gives an empty
/// report). When every seat uses the same strategy name there is no
/// per-strategy result: `strategies` is empty (the CLI prints "n/a: a
/// single strategy name"). Deterministic.
///
/// # Panics
///
/// Never: the design rows are rectangular by construction.
#[must_use]
#[allow(clippy::too_many_lines)] // one linear pipeline: design, fit, adjust, summarise
pub fn estimator_report(results: &[MatchResult]) -> EstimatorReport {
    let mut names: Vec<&str> = Vec::new();
    let mut obs: Vec<Observation> = Vec::new();
    let mut match_count = 0;
    for m in results {
        let Some(features) = &m.first_hand_features else {
            continue;
        };
        let Some(round1) = m.role_history.first() else {
            continue;
        };
        for seat in 0..m.strategy_names.len() {
            let name = &m.strategy_names[seat];
            let idx = names.iter().position(|n| *n == name).unwrap_or_else(|| {
                names.push(name);
                names.len() - 1
            });
            obs.push(Observation {
                match_index: match_count,
                name: idx,
                features: HandFeatures::design_vector(&features[seat]),
                score: role_score(round1[seat], m.player_count),
            });
        }
        match_count += 1;
    }
    if obs.is_empty() {
        return empty_report();
    }
    // One strategy name in every seat: nothing to separate from the
    // table, and the fixed effect would absorb the mean (spurious SE).
    if names.len() < 2 {
        return EstimatorReport {
            match_count,
            ..empty_report()
        };
    }

    let p = FEATURE_NAMES.len();
    let rows: Vec<Vec<f64>> = obs
        .iter()
        .map(|o| {
            let mut row = o.features.to_vec();
            row.extend((0..names.len()).map(|n| f64::from(u8::from(n == o.name))));
            row
        })
        .collect();
    let y: Vec<f64> = obs.iter().map(|o| o.score).collect();
    let beta = ols(&rows, &y).expect("rows are rectangular and non-empty");
    let beta_f = &beta[..p];

    #[allow(clippy::cast_precision_loss)] // observation counts are far below 2^52
    let n_obs = obs.len() as f64;
    let mut feature_mean = [0.0; 7];
    for o in &obs {
        for (acc, v) in feature_mean.iter_mut().zip(o.features) {
            *acc += v / n_obs;
        }
    }
    let part = |o: &Observation| dot(beta_f, &o.features);
    let centered = |o: &Observation| part(o) - dot(beta_f, &feature_mean);

    // Partial R^2: residual sums of squares with and without the features.
    let ssr_full: f64 = obs
        .iter()
        .zip(&rows)
        .map(|(o, row)| (o.score - dot(&beta, row)).powi(2))
        .sum();
    let fixed_means: Vec<f64> = (0..names.len())
        .map(|n| {
            mean(
                &obs.iter()
                    .filter(|o| o.name == n)
                    .map(|o| o.score)
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let ssr_fixed: f64 = obs
        .iter()
        .map(|o| (o.score - fixed_means[o.name]).powi(2))
        .sum();
    let feature_r2 = if ssr_fixed > 0.0 {
        (1.0 - ssr_full / ssr_fixed).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let strategies = names
        .iter()
        .enumerate()
        .map(|(n, name)| {
            let mine: Vec<&Observation> = obs.iter().filter(|o| o.name == n).collect();
            // Per-match units: average the strategy's seats within a match.
            let mut raw = Vec::new();
            let mut adjusted = Vec::new();
            let mut start = 0;
            while start < mine.len() {
                let id = mine[start].match_index;
                let end = mine[start..]
                    .iter()
                    .position(|o| o.match_index != id)
                    .map_or(mine.len(), |off| start + off);
                let group = &mine[start..end];
                raw.push(mean(&group.iter().map(|o| o.score).collect::<Vec<_>>()));
                adjusted.push(mean(
                    &group
                        .iter()
                        .map(|o| o.score - centered(o))
                        .collect::<Vec<_>>(),
                ));
                start = end;
            }
            let scores: Vec<f64> = mine.iter().map(|o| o.score).collect();
            let parts: Vec<f64> = mine.iter().map(|o| part(o)).collect();
            let var_y = variance(&scores);
            let m = reduction(variance(&raw), variance(&adjusted));
            EstimatorStrategy {
                name: (*name).to_string(),
                skill_score_estimate: estimate(&adjusted),
                plain_mean_role_score_round1: estimate(&raw),
                variance_reduction: m,
                luck_share: luck_share(m),
                variance_explained: if var_y > 0.0 {
                    (variance(&parts) / var_y).clamp(0.0, 1.0)
                } else {
                    0.0
                },
            }
        })
        .collect();

    EstimatorReport {
        match_count,
        feature_names: FEATURE_NAMES.to_vec(),
        feature_coefficients: beta_f.to_vec(),
        feature_r2,
        strategies,
    }
}
