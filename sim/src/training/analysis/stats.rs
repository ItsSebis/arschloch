//! Small numeric helpers for the run analysis: trailing mean, standard
//! deviation, and an OLS slope with a 95% interval that is widened for the
//! lag-1 autocorrelation of the residuals.

use serde::Serialize;

/// Two-sided 95% Student t quantiles for 1..=30 degrees of freedom.
const T_975: [f64; 30] = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
    2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
    2.052, 2.048, 2.045, 2.042,
];

/// Further tabulated quantiles `(degrees of freedom, t)`, ending in the
/// normal quantile at infinite degrees of freedom.
const T_975_TAIL: [(f64, f64); 5] = [
    (30.0, 2.042),
    (40.0, 2.021),
    (60.0, 2.000),
    (120.0, 1.980),
    (f64::INFINITY, 1.960),
];

/// Largest factor the autocorrelation correction may widen an interval by.
const MAX_AUTOCORRELATION_FACTOR: f64 = 5.0;

/// Two-sided 95% Student t quantile for `df` degrees of freedom: the table
/// up to 30, then linear interpolation in `1 / df` between 30, 40, 60, 120
/// and the normal 1.96 (which is what `1 / df = 0` gives).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn t_quantile(df: usize) -> f64 {
    if df == 0 {
        return f64::INFINITY;
    }
    if df <= T_975.len() {
        return T_975[df - 1];
    }
    let df = df as f64;
    for pair in T_975_TAIL.windows(2) {
        let ((d0, t0), (d1, t1)) = (pair[0], pair[1]);
        if df <= d1 {
            let (a, b, x) = (1.0 / d0, 1.0 / d1, 1.0 / df);
            return t1 + (t0 - t1) * (x - b) / (a - b);
        }
    }
    1.96
}

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

/// OLS slope of `ys` on `xs` with a 95% interval: `slope +- t * se`, the t
/// quantile for `n - 2` degrees of freedom (`t_quantile`). The standard
/// error is multiplied by `sqrt((1 + r) / (1 - r))`, capped at 5, when the
/// lag-1 autocorrelation `r` of the residuals is positive (consecutive
/// generations are not independent). `None` for fewer than three points,
/// mismatched lengths or constant `xs`.
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
    let std_error =
        (rss / (nf - 2.0) / sxx).sqrt() * autocorrelation_factor(xs, ys, value, intercept, rss);
    let quantile = t_quantile(n - 2);
    Some(Slope {
        value,
        low: value - quantile * std_error,
        high: value + quantile * std_error,
    })
}

/// `sqrt((1 + r) / (1 - r))` for the lag-1 autocorrelation `r` of the
/// residuals of the fit when it is positive, capped; 1 otherwise.
fn autocorrelation_factor(xs: &[f64], ys: &[f64], slope: f64, intercept: f64, rss: f64) -> f64 {
    if rss <= 0.0 {
        return 1.0;
    }
    let residuals: Vec<f64> = xs
        .iter()
        .zip(ys)
        .map(|(x, y)| y - intercept - slope * x)
        .collect();
    let lagged: f64 = residuals.windows(2).map(|w| w[0] * w[1]).sum();
    let r = lagged / rss;
    if r <= 0.0 {
        return 1.0;
    }
    ((1.0 + r) / (1.0 - r).max(1e-12))
        .sqrt()
        .min(MAX_AUTOCORRELATION_FACTOR)
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

/// Where a smoothed curve stalls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stall {
    /// First index from which every smoothed value is at least `peak - se`.
    pub index: usize,
    /// The last value itself is below `peak - se`, so no such index exists
    /// and `index` is the last one (the fallback of `analyse.py`).
    pub fallback: bool,
}

/// The stall of a smoothed curve (the rule of
/// `docs/baselines/phase-13/analyse.py`); `None` when empty.
#[must_use]
pub fn stall_index(smoothed: &[f64], se: f64) -> Option<Stall> {
    let peak = smoothed.iter().copied().reduce(f64::max)?;
    let found = (0..smoothed.len()).find(|&i| smoothed[i..].iter().all(|&v| v >= peak - se));
    Some(Stall {
        index: found.unwrap_or(smoothed.len() - 1),
        fallback: found.is_none(),
    })
}

/// Index of the first maximum; `None` when empty.
#[must_use]
pub fn argmax(values: &[f64]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, v) in values.iter().enumerate() {
        if best.is_none_or(|b| *v > values[b]) {
            best = Some(i);
        }
    }
    best
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
        let stall = stall_index(&smoothed, 0.6).unwrap();
        assert_eq!((stall.index, stall.fallback), (3, false));
        // A huge se admits everything from the start.
        assert_eq!(stall_index(&smoothed, 10.0).unwrap().index, 0);
        // The last value drops out of the band: python's fallback len - 1,
        // and the stall says so.
        let fell = stall_index(&[3.0, 3.0, 0.0], 0.6).unwrap();
        assert_eq!((fell.index, fell.fallback), (2, true));
        assert_eq!(stall_index(&[], 0.6), None);
    }

    #[test]
    fn argmax_is_the_first_maximum() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0, 3.0]), Some(1));
        assert_eq!(argmax(&[-2.0]), Some(0));
        assert_eq!(argmax(&[]), None);
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

    /// Plain (uncorrected) standard error of a fit, for the tests.
    #[allow(clippy::cast_precision_loss)]
    fn plain_se(xs: &[f64], ys: &[f64], slope: f64) -> f64 {
        let (mx, my) = (mean(xs), mean(ys));
        let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
        let intercept = my - slope * mx;
        let rss: f64 = xs
            .iter()
            .zip(ys)
            .map(|(x, y)| (y - intercept - slope * x).powi(2))
            .sum();
        (rss / (xs.len() as f64 - 2.0) / sxx).sqrt()
    }

    #[test]
    fn t_quantiles_follow_the_table_then_interpolate_in_one_over_df() {
        assert!((t_quantile(1) - 12.706).abs() < 1e-12);
        assert!((t_quantile(28) - 2.048).abs() < 1e-12);
        assert!((t_quantile(30) - 2.042).abs() < 1e-12);
        for (df, t) in [(40, 2.021), (60, 2.000), (120, 1.980)] {
            assert!((t_quantile(df) - t).abs() < 1e-12, "{df}");
        }
        // Halfway in 1/df between 60 and 120 (df = 80): 1.98 + 0.02 * 0.5.
        assert!((t_quantile(80) - 1.99).abs() < 1e-12);
        // Between 30 and 40.
        let t35 = t_quantile(35);
        assert!(t35 < 2.042 && t35 > 2.021, "{t35}");
        // Monotone, and the normal quantile far out.
        let mut last = f64::INFINITY;
        for df in 1..400 {
            let t = t_quantile(df);
            assert!(t <= last + 1e-12 && t >= 1.96, "{df}");
            last = t;
        }
        assert!((t_quantile(100_000) - 1.96).abs() < 1e-3);
    }

    #[test]
    fn thirty_points_still_use_the_t_table_not_the_normal_quantile() {
        // y = x + (-1)^x for x in 0..30: sxx = 2247.5, slope = 1 - 15 / 2247.5.
        let xs: Vec<f64> = (0..30).map(f64::from).collect();
        let ys: Vec<f64> = (0..30)
            .map(|x| f64::from(x) + if x % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let slope = slope_with_interval(&xs, &ys).unwrap();
        assert!((slope.value - (1.0 - 15.0 / 2247.5)).abs() < 1e-12);
        // The alternating residuals have negative lag-1 autocorrelation:
        // no widening, and t for 28 degrees of freedom (2.048, not 1.96).
        let se = plain_se(&xs, &ys, slope.value);
        assert!((slope.high - slope.value - 2.048 * se).abs() < 1e-12);
        // 29 points: 27 df.
        let t = slope_with_interval(&xs[..29], &ys[..29]).unwrap();
        let se = plain_se(&xs[..29], &ys[..29], t.value);
        assert!(((t.high - t.value) - 2.052 * se).abs() < 1e-12);
        // 62 points: df 60 gives exactly 2.000.
        let xs: Vec<f64> = (0..62).map(f64::from).collect();
        let ys: Vec<f64> = (0..62)
            .map(|x| f64::from(x) + if x % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let t = slope_with_interval(&xs, &ys).unwrap();
        let se = plain_se(&xs, &ys, t.value);
        assert!(((t.high - t.value) - 2.000 * se).abs() < 1e-12);
    }

    #[test]
    fn positively_autocorrelated_residuals_widen_the_interval() {
        // x 0..7, y = [2, 1, -1, -4, -1, 1, 2]: symmetric and centred, so the
        // fit is the horizontal line 0 and the residuals are y itself.
        // rss = 28, lag-1 sum = 10: r = 10/28, factor sqrt(38/18).
        let xs: Vec<f64> = (0..7).map(f64::from).collect();
        let ys = [2.0, 1.0, -1.0, -4.0, -1.0, 1.0, 2.0];
        let slope = slope_with_interval(&xs, &ys).unwrap();
        assert!(slope.value.abs() < 1e-12);
        let plain = (28.0_f64 / 5.0 / 28.0).sqrt();
        let factor = (38.0_f64 / 18.0).sqrt();
        let half = 2.571 * plain * factor;
        assert!((slope.high - half).abs() < 1e-9, "{slope:?} {half}");
        assert!((slope.low + half).abs() < 1e-9);
        // The same shape with alternating signs (r < 0) is not narrowed.
        let alt = [1.0, -1.0, 0.0, -1.0, 1.0];
        let xs5: Vec<f64> = (0..5).map(f64::from).collect();
        let s = slope_with_interval(&xs5, &alt).unwrap();
        assert!((s.high - 3.182 * plain_se(&xs5, &alt, 0.0)).abs() < 1e-9);
    }

    #[test]
    fn the_autocorrelation_factor_is_capped() {
        // A smooth bowl leaves residuals with r close to 1; whatever it is,
        // the interval is at most five times the uncorrected one.
        let xs: Vec<f64> = (0..40).map(f64::from).collect();
        let ys: Vec<f64> = xs.iter().map(|x| (x - 19.5).powi(2)).collect();
        let slope = slope_with_interval(&xs, &ys).unwrap();
        let plain = t_quantile(38) * plain_se(&xs, &ys, slope.value);
        let half = slope.high - slope.value;
        assert!(
            half > 1.5 * plain && half <= 5.0 * plain + 1e-9,
            "{half} {plain}"
        );
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
