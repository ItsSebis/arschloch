//! Luck/skill measurement. Two independent ways to separate the skill of
//! a strategy from the luck of the deal:
//!
//! - `skill_report`: exact, from duplicate groups (`crate::duplicate`);
//! - `estimator_report`: cheap, a regression on round-1 hand features
//!   that works on any ordinary run recorded with `record_deal_features`;
//! - `compare`: how well the two agree.
//!
//! All scores are `role_score`s: +1 for the best role, -1 for the worst.

mod compare;
mod duplicate_report;
mod estimator;

pub use compare::{compare, ComparisonReport, StrategyComparison};
pub use duplicate_report::{skill_report, SkillReport, StrategySkill};
pub use estimator::{estimator_report, EstimatorReport, EstimatorStrategy};

pub use crate::extended_stats::Estimate;

#[allow(clippy::cast_precision_loss)] // sample counts are far below 2^52
fn count(xs: &[f64]) -> f64 {
    xs.len() as f64
}

pub(crate) fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / count(xs)
    }
}

/// Sample variance (n - 1 denominator); 0 for fewer than two values.
pub(crate) fn variance(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (count(xs) - 1.0)
}

/// Mean and standard error `sd / sqrt(n)` (0 for a single value).
pub(crate) fn estimate(xs: &[f64]) -> Estimate {
    Estimate {
        value: mean(xs),
        std_error: if xs.is_empty() {
            0.0
        } else {
            (variance(xs) / count(xs)).sqrt()
        },
        n: xs.len(),
    }
}

/// Largest variance-reduction factor reported, so a (nearly) zero
/// variance of the group means cannot produce infinity.
pub(crate) const MAX_REDUCTION: f64 = 1e6;

/// `var_before / var_after` as an "equivalent number of ordinary games
/// per game": 1 when there is nothing to compare (`var_before == 0`),
/// capped at `MAX_REDUCTION`.
pub(crate) fn reduction(var_before: f64, var_after: f64) -> f64 {
    if var_before <= 0.0 {
        return 1.0;
    }
    (var_before / var_after.max(var_before / MAX_REDUCTION)).max(0.0)
}

/// `1 - 1/m` clamped to `[0, 1]`: the share of single-game variance that
/// is luck.
pub(crate) fn luck_share(m: f64) -> f64 {
    (1.0 - 1.0 / m).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_matches_hand_computation() {
        let e = estimate(&[1.0, 2.0, 3.0, 4.0]);
        assert!((e.value - 2.5).abs() < 1e-12);
        // sample variance 5/3, se = sqrt(5/12)
        assert!((e.std_error - (5.0_f64 / 12.0).sqrt()).abs() < 1e-12);
        assert_eq!(e.n, 4);
    }

    #[test]
    fn reduction_guards_zero_variances() {
        assert!((reduction(0.0, 0.0) - 1.0).abs() < 1e-12);
        assert!((reduction(2.0, 0.0) - MAX_REDUCTION).abs() < 1e-6);
        assert!((reduction(4.0, 1.0) - 4.0).abs() < 1e-12);
        assert!((luck_share(4.0) - 0.75).abs() < 1e-12);
        assert!(luck_share(0.5).abs() < 1e-12);
    }
}
