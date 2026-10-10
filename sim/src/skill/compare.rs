//! Side-by-side comparison of the duplicate skill score and the cheap
//! estimator.

use super::{duplicate_report::SkillReport, estimator::EstimatorReport, Estimate};

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct StrategyComparison {
    pub name: String,
    /// The duplicate round-1 skill (same round the estimator describes).
    pub skill_score_duplicate_round1: Estimate,
    pub skill_score_estimate: Estimate,
    /// `(estimate - duplicate) / sqrt(se_est^2 + se_dup^2)`; 0 if both
    /// standard errors are 0.
    pub difference_in_se: f64,
    /// Estimator luck share divided by duplicate round-1 luck share (the
    /// fraction of the exactly removable variance the estimator removes);
    /// 0 if the duplicate luck share is 0.
    pub variance_removed_ratio: f64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ComparisonReport {
    /// Strategies present in both reports, in the duplicate report's order.
    pub strategies: Vec<StrategyComparison>,
    /// Kendall tau between the two skill orderings: +1 identical, -1
    /// reversed, ties count as 0; 1 for fewer than two strategies.
    pub rank_agreement: f64,
    /// Mean of `variance_removed_ratio` over the strategies.
    pub mean_variance_removed_ratio: f64,
    pub verdict: String,
}

fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn kendall_tau(pairs: &[(f64, f64)]) -> f64 {
    let mut total = 0.0;
    let mut sum = 0.0;
    for (i, a) in pairs.iter().enumerate() {
        for b in &pairs[i + 1..] {
            total += 1.0;
            sum += sign(a.0 - b.0) * sign(a.1 - b.1);
        }
    }
    if total == 0.0 {
        1.0
    } else {
        sum / total
    }
}

/// Compares the two reports by strategy name. Note the estimator is only
/// an independent check when it was fitted on matches that do not share
/// deals; on the matches of a duplicate batch its standard errors are
/// optimistic.
#[must_use]
pub fn compare(dup: &SkillReport, est: &EstimatorReport) -> ComparisonReport {
    let strategies: Vec<StrategyComparison> = dup
        .strategies
        .iter()
        .filter_map(|d| {
            let e = est.strategies.iter().find(|e| e.name == d.name)?;
            let se = d
                .skill_score_duplicate_round1
                .std_error
                .hypot(e.skill_score_estimate.std_error);
            let diff = e.skill_score_estimate.value - d.skill_score_duplicate_round1.value;
            Some(StrategyComparison {
                name: d.name.clone(),
                skill_score_duplicate_round1: d.skill_score_duplicate_round1,
                skill_score_estimate: e.skill_score_estimate,
                difference_in_se: if se > 0.0 { diff / se } else { 0.0 },
                variance_removed_ratio: if d.luck_share_round1 > 0.0 {
                    e.luck_share / d.luck_share_round1
                } else {
                    0.0
                },
            })
        })
        .collect();
    let rank_agreement = kendall_tau(
        &strategies
            .iter()
            .map(|s| {
                (
                    s.skill_score_duplicate_round1.value,
                    s.skill_score_estimate.value,
                )
            })
            .collect::<Vec<_>>(),
    );
    #[allow(clippy::cast_precision_loss)] // a handful of strategies
    let mean_ratio = if strategies.is_empty() {
        0.0
    } else {
        strategies
            .iter()
            .map(|s| s.variance_removed_ratio)
            .sum::<f64>()
            / strategies.len() as f64
    };
    let worst = strategies
        .iter()
        .map(|s| s.difference_in_se.abs())
        .fold(0.0, f64::max);
    let verdict = format!(
        "estimator removes {:.0}% of the luck variance the duplicate deals remove (round 1), \
         rank agreement tau {:.2}, largest gap {:.1} SE",
        mean_ratio * 100.0,
        rank_agreement,
        worst
    );
    ComparisonReport {
        strategies,
        rank_agreement,
        mean_variance_removed_ratio: mean_ratio,
        verdict,
    }
}
