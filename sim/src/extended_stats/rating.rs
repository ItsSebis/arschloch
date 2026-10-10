//! Bradley-Terry strength ratings fitted with the MM algorithm.
//!
//! Every round contributes pairwise outcomes: for each pair of seats the
//! better place "beats" the other. Counts are pooled by strategy name
//! and pairs of the same name are skipped.
//!
//! Regularisation: each strategy additionally plays one virtual
//! opponent of fixed strength 1 (the geometric-mean field average, as
//! strengths are renormalised to geometric mean 1 every iteration),
//! winning 0.5 and losing 0.5 games. This keeps the fit finite when a
//! strategy never wins or never loses.

#![allow(clippy::cast_precision_loss)] // counts are far below 2^53

use super::intervals::standard_error;

/// Elo points per natural-log unit of strength: `400 / ln 10`.
const ELO_PER_LN: f64 = 400.0 / std::f64::consts::LN_10;
const MAX_ITERATIONS: usize = 2000;
const TOLERANCE: f64 = 1e-10;

/// Pairwise win counts for one match over `k` strategies:
/// `wins[i * k + j]` = times `i` beat `j`.
#[derive(Debug, Clone)]
pub struct PairCounts {
    k: usize,
    wins: Vec<f64>,
}

impl PairCounts {
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            k,
            wins: vec![0.0; k * k],
        }
    }

    /// Records that strategy `winner` beat strategy `loser` once.
    pub fn record(&mut self, winner: usize, loser: usize) {
        if winner != loser {
            self.wins[winner * self.k + loser] += 1.0;
        }
    }

    fn add(&mut self, other: &Self) {
        for (a, b) in self.wins.iter_mut().zip(&other.wins) {
            *a += b;
        }
    }
}

/// Fits centred Elo-like ratings (mean 0) to the summed `counts`.
#[must_use]
fn fit(total: &PairCounts) -> Vec<f64> {
    let k = total.k;
    let games = |i: usize, j: usize| total.wins[i * k + j] + total.wins[j * k + i];
    let mut strength = vec![1.0_f64; k];
    for _ in 0..MAX_ITERATIONS {
        let mut next = vec![0.0; k];
        for i in 0..k {
            let won: f64 = (0..k).map(|j| total.wins[i * k + j]).sum::<f64>() + 0.5;
            let mut denom = 1.0 / (strength[i] + 1.0); // virtual opponent
            for j in (0..k).filter(|&j| j != i) {
                denom += games(i, j) / (strength[i] + strength[j]);
            }
            next[i] = won / denom;
        }
        let mean_log = next.iter().map(|s| s.ln()).sum::<f64>() / k as f64;
        let mut change = 0.0_f64;
        for (old, new) in strength.iter_mut().zip(&next) {
            let updated = (new.ln() - mean_log).exp();
            change = change.max((updated.ln() - old.ln()).abs());
            *old = updated;
        }
        if change < TOLERANCE {
            break;
        }
    }
    let logs: Vec<f64> = strength.iter().map(|s| ELO_PER_LN * s.ln()).collect();
    let mean = logs.iter().sum::<f64>() / k as f64;
    logs.into_iter().map(|l| l - mean).collect()
}

/// Splitmix64: tiny deterministic generator for bootstrap resampling.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Ratings and bootstrap standard errors, one entry per strategy.
/// `per_match` holds one `PairCounts` per match. Without resampling
/// (`bootstrap_resamples < 2`) every standard error is `None`.
#[must_use]
pub fn rate(
    per_match: &[PairCounts],
    k: usize,
    bootstrap_resamples: usize,
    bootstrap_seed: u64,
) -> Vec<(f64, Option<f64>)> {
    let mut total = PairCounts::new(k);
    for m in per_match {
        total.add(m);
    }
    let point = fit(&total);
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); k];
    if !per_match.is_empty() {
        let mut rng = SplitMix64(bootstrap_seed ^ (per_match.len() as u64).rotate_left(32));
        for _ in 0..bootstrap_resamples {
            let mut resampled = PairCounts::new(k);
            for _ in 0..per_match.len() {
                #[allow(clippy::cast_possible_truncation)]
                let idx = (rng.next() % per_match.len() as u64) as usize;
                resampled.add(&per_match[idx]);
            }
            for (s, r) in samples.iter_mut().zip(fit(&resampled)) {
                s.push(r);
            }
        }
    }
    point
        .into_iter()
        .zip(&samples)
        .map(|(p, s)| (p, (s.len() >= 2).then(|| std_dev(s))))
        .collect()
}

fn std_dev(values: &[f64]) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let n = values.len() as f64;
    standard_error(values) * n.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_counts_give_zero_ratings() {
        let mut m = PairCounts::new(2);
        m.record(0, 1);
        m.record(1, 0);
        let r = rate(&[m], 2, 0, 1);
        assert!(r[0].0.abs() < 1e-6 && r[1].0.abs() < 1e-6);
    }

    #[test]
    fn separation_stays_finite_and_ordered() {
        let mut m = PairCounts::new(2);
        for _ in 0..50 {
            m.record(0, 1);
        }
        let r = rate(&[m], 2, 0, 1);
        assert!(r[0].0.is_finite() && r[1].0.is_finite());
        assert!(r[0].0 > 100.0 && (r[0].0 + r[1].0).abs() < 1e-9);
    }
}
