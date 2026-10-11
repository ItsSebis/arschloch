//! Small numeric helpers for the run analysis: trailing mean, standard
//! deviation, and an OLS slope with a 95% interval.

use serde::Serialize;

/// Two-sided 95% Student t quantiles for 1..=28 degrees of freedom.
const T_975: [f64; 28] = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
    2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
    2.052, 2.048,
];

/// A fitted slope with its 95% interval.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Slope {
    pub value: f64,
    pub low: f64,
    pub high: f64,
}

impl Slope {
    /// The same slope in other units (`per_100` = 100 generations).
    #[must_use]
    pub fn scaled(self, factor: f64) -> Self {
        Self {
            value: self.value * factor,
            low: self.low * factor,
            high: self.high * factor,
        }
    }

    /// The interval lies strictly above zero.
    #[must_use]
    pub fn is_positive(&self) -> bool {
        self.low > 0.0
    }

    /// The interval lies strictly below zero.
    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.high < 0.0
    }

    /// The interval contains zero.
    #[must_use]
    pub fn contains_zero(&self) -> bool {
        !self.is_positive() && !self.is_negative()
    }
}

/// OLS slope of `ys` on `xs` with a 95% interval: the t quantile for
/// `n - 2` degrees of freedom when `n < 30`, the normal 1.96 otherwise.
/// `None` for fewer than three points, mismatched lengths or constant `xs`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn slope_with_interval(xs: &[f64], ys: &[f64]) -> Option<Slope> {
    let n = xs.len();
    if n < 3 || ys.len() != n {
        return None;
    }
    let nf = n as f64;
    let mean_x = xs.iter().sum::<f64>() / nf;
    let mean_y = ys.iter().sum::<f64>() / nf;
    let sxx: f64 = xs.iter().map(|x| (x - mean_x).powi(2)).sum();
    if sxx <= 0.0 {
        return None;
    }
    let sxy: f64 = xs
        .iter()
        .zip(ys)
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let value = sxy / sxx;
    let intercept = mean_y - value * mean_x;
    let rss: f64 = xs
        .iter()
        .zip(ys)
        .map(|(x, y)| (y - intercept - value * x).powi(2))
        .sum();
    let std_error = (rss / (nf - 2.0) / sxx).sqrt();
    let quantile = if n < 30 { T_975[n - 3] } else { 1.96 };
    Some(Slope {
        value,
        low: value - quantile * std_error,
        high: value + quantile * std_error,
    })
}

/// Mean; `0.0` for an empty slice.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Sample standard deviation (n - 1 divisor); `0.0` below two values.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

/// Trailing mean over the last `window` values (shorter at the start).
#[must_use]
pub fn smooth(values: &[f64], window: usize) -> Vec<f64> {
    (0..values.len())
        .map(|i| mean(&values[(i + 1).saturating_sub(window)..=i]))
        .collect()
}

/// First index from which every smoothed value is at least `peak - se`
/// (the rule of `docs/baselines/phase-13/analyse.py`); `None` when empty.
#[must_use]
pub fn stall_index(smoothed: &[f64], se: f64) -> Option<usize> {
    let peak = smoothed.iter().copied().reduce(f64::max)?;
    Some(
        (0..smoothed.len())
            .find(|&i| smoothed[i..].iter().all(|&v| v >= peak - se))
            .unwrap_or(smoothed.len() - 1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoothing_is_a_trailing_mean_shorter_at_the_start() {
        assert_eq!(smooth(&[1.0, 2.0, 3.0, 4.0], 2), vec![1.0, 1.5, 2.5, 3.5]);
        assert_eq!(smooth(&[], 10), Vec::<f64>::new());
    }

    #[test]
    fn stall_index_is_the_first_index_that_stays_within_se_of_the_peak() {
        // peak 3, threshold 2.4: index 2 (value 2) fails, 3.. all pass.
        let smoothed = [0.0, 1.0, 2.0, 3.0, 3.0, 2.5, 3.0];
        assert_eq!(stall_index(&smoothed, 0.6), Some(3));
        // A huge se admits everything from the start.
        assert_eq!(stall_index(&smoothed, 10.0), Some(0));
        // The last value drops out of the band: python's fallback len - 1.
        assert_eq!(stall_index(&[3.0, 3.0, 0.0], 0.6), Some(2));
        assert_eq!(stall_index(&[], 0.6), None);
    }

    #[test]
    fn slope_matches_a_hand_computed_fit() {
        // x 1..5, y 2,4,5,4,5: slope 0.6, residual sum of squares 2.4,
        // se = sqrt(2.4 / 3 / 10), t(3 df) = 3.182.
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [2.0, 4.0, 5.0, 4.0, 5.0];
        let slope = slope_with_interval(&xs, &ys).unwrap();
        let half = 3.182 * (0.08_f64).sqrt();
        assert!((slope.value - 0.6).abs() < 1e-12);
        assert!((slope.low - (0.6 - half)).abs() < 1e-9, "{slope:?}");
        assert!((slope.high - (0.6 + half)).abs() < 1e-9);
        assert!(slope.contains_zero() && !slope.is_positive() && !slope.is_negative());
        let scaled = slope.scaled(100.0);
        assert!((scaled.value - 60.0).abs() < 1e-9);
    }

    #[test]
    fn a_perfect_line_has_a_zero_width_interval_and_a_sign() {
        let xs: Vec<f64> = (0..10).map(f64::from).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 3.0 - 0.5 * x).collect();
        let slope = slope_with_interval(&xs, &ys).unwrap();
        assert!((slope.value + 0.5).abs() < 1e-12);
        assert!((slope.high - slope.low).abs() < 1e-9);
        assert!(slope.is_negative());
    }

    #[test]
    fn thirty_points_use_the_normal_quantile() {
        // y = x + (-1)^x for x in 0..30: sxx = 2247.5, slope = 1 - 15 / 2247.5.
        let xs: Vec<f64> = (0..30).map(f64::from).collect();
        let ys: Vec<f64> = (0..30)
            .map(|x| f64::from(x) + if x % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let slope = slope_with_interval(&xs, &ys).unwrap();
        assert!((slope.value - (1.0 - 15.0 / 2247.5)).abs() < 1e-12);
        let intercept = mean(&ys) - slope.value * mean(&xs);
        let rss: f64 = xs
            .iter()
            .zip(&ys)
            .map(|(x, y)| (y - intercept - slope.value * x).powi(2))
            .sum();
        let se = (rss / 28.0 / 2247.5).sqrt();
        assert!((slope.high - slope.value - 1.96 * se).abs() < 1e-12);
        // 29 points still use the t table (27 df).
        let t = slope_with_interval(&xs[..29], &ys[..29]).unwrap();
        let m = mean(&xs[..29]);
        let sxx: f64 = xs[..29].iter().map(|x| (x - m).powi(2)).sum();
        let icpt = mean(&ys[..29]) - t.value * m;
        let rss: f64 = (0..29)
            .map(|i| (ys[i] - icpt - t.value * xs[i]).powi(2))
            .sum();
        assert!(((t.high - t.value) - 2.052 * (rss / 27.0 / sxx).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn degenerate_inputs_have_no_slope() {
        assert_eq!(slope_with_interval(&[1.0, 2.0], &[1.0, 2.0]), None);
        assert_eq!(
            slope_with_interval(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]),
            None
        );
        assert_eq!(slope_with_interval(&[1.0, 2.0, 3.0], &[1.0, 2.0]), None);
    }

    #[test]
    fn std_dev_uses_the_sample_divisor() {
        // 1,2,3,4: variance 5/3.
        assert!((std_dev(&[1.0, 2.0, 3.0, 4.0]) - (5.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert!(std_dev(&[1.0]).abs() < f64::EPSILON);
    }
}
