//! Confidence intervals and standard errors for the extended statistics.

/// A closed interval `[low, high]`.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Interval {
    pub low: f64,
    pub high: f64,
}

/// Two-sided 95% normal quantile.
const Z_95: f64 = 1.959_963_984_540_054;

/// Wilson score interval (95%) for `successes` out of `trials`; `None`
/// when `trials == 0`. Unlike the plain normal interval it stays inside
/// `[0, 1]` and behaves sensibly at 0% and 100%.
///
/// # Panics
///
/// Panics if `successes > trials`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn wilson_interval(successes: u64, trials: u64) -> Option<(f64, f64)> {
    assert!(successes <= trials, "successes cannot exceed trials");
    if trials == 0 {
        return None;
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = Z_95 * Z_95;
    let denom = 1.0 + z2 / n;
    let centre = (p + z2 / (2.0 * n)) / denom;
    let half = Z_95 * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denom;
    Some(((centre - half).max(0.0), (centre + half).min(1.0)))
}

/// Standard error of the mean: sample standard deviation (n - 1 divisor)
/// over `sqrt(n)`; `0.0` for fewer than two values.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn standard_error(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (variance / n).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilson_matches_known_values() {
        let (low, high) = wilson_interval(8, 10).unwrap();
        assert!((low - 0.490).abs() < 1e-3, "{low}");
        assert!((high - 0.943).abs() < 1e-3, "{high}");
        let (low, high) = wilson_interval(0, 10).unwrap();
        assert!(low.abs() < 1e-12);
        assert!((high - 0.2775).abs() < 1e-3, "{high}");
        assert_eq!(wilson_interval(0, 0), None);
    }

    #[test]
    fn standard_error_matches_hand_computation() {
        // values 1,2,3,4: mean 2.5, sample variance 5/3, se = sqrt(5/12).
        let se = standard_error(&[1.0, 2.0, 3.0, 4.0]);
        assert!((se - (5.0_f64 / 12.0).sqrt()).abs() < 1e-12);
        assert!(standard_error(&[3.0]).abs() < f64::EPSILON);
        assert!(standard_error(&[]).abs() < f64::EPSILON);
    }
}
