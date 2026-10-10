//! The optional luck-adjusted fitness term (`TrainConfig::skill_weight`).
//!
//! ```text
//! fitness_total = (1 - w) * mean_role_score + w * luck_adjusted_score
//! ```
//!
//! `mean_role_score` is the usual mean over all rounds of all matches.
//! `luck_adjusted_score` of a genome uses round 1 only: in each match
//! `round1_score - beta . (round1_features - mean_features)`, averaged
//! over the matches. `beta` comes from one ordinary least-squares fit of
//! round-1 score on the round-1 hand features (plus an intercept) over the
//! pooled observations of *every* genome of the generation, and
//! `mean_features` is the mean over that same pool. Fitting once for the
//! whole generation (not per genome) keeps a genome from explaining away
//! its own result.
//!
//! Every genome of a generation plays the same matches (same deals, seats
//! and opponents), so every genome has the same feature rows. The mean of
//! `beta . (x - mean_x)` over those matches is then exactly zero, and the
//! adjusted score of a genome equals its plain round-1 mean score (up to
//! rounding): within one generation the term can only change the ranking
//! by weighting round 1 more heavily, never by cancelling deal luck, which
//! common random numbers already cancel between genomes. What the
//! adjustment does reduce is the variance of the per-match values.

use super::evaluate::Round1Obs;
use crate::linalg::ols;

/// Number of hand features (see `HandFeatures::as_vector`).
const P: usize = 6;

/// The regression of one generation's pooled round-1 observations.
#[derive(Debug, Clone, PartialEq)]
pub struct DealModel {
    /// Effect of each hand feature on the round-1 role score.
    pub beta: [f64; P],
    /// Mean of each feature over the pooled observations.
    pub mean_features: [f64; P],
}

fn dot(a: &[f64; P], b: &[f64; P]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Fits `DealModel` on the observations of all genomes (in order; the
/// result is deterministic for a given input). `None` when there are no
/// observations.
///
/// # Panics
///
/// Never: the design rows are rectangular by construction.
#[must_use]
pub fn fit_deal_model(per_genome: &[Vec<Round1Obs>]) -> Option<DealModel> {
    let pooled: Vec<&Round1Obs> = per_genome.iter().flatten().collect();
    if pooled.is_empty() {
        return None;
    }
    let rows: Vec<Vec<f64>> = pooled
        .iter()
        .map(|o| {
            let mut row = o.features.to_vec();
            row.push(1.0);
            row
        })
        .collect();
    let y: Vec<f64> = pooled.iter().map(|o| o.score).collect();
    let coefficients = ols(&rows, &y).expect("rows are rectangular and non-empty");
    let mut beta = [0.0; P];
    beta.copy_from_slice(&coefficients[..P]);
    #[allow(clippy::cast_precision_loss)] // observation counts are far below 2^52
    let n = pooled.len() as f64;
    let mut mean_features = [0.0; P];
    for o in &pooled {
        for (acc, v) in mean_features.iter_mut().zip(o.features) {
            *acc += v / n;
        }
    }
    Some(DealModel {
        beta,
        mean_features,
    })
}

/// A genome's luck-adjusted round-1 score: the mean over its matches of
/// `score - beta . (features - mean_features)`.
///
/// # Panics
///
/// Panics if `observations` is empty.
#[must_use]
pub fn adjusted_score(model: &DealModel, observations: &[Round1Obs]) -> f64 {
    assert!(!observations.is_empty(), "no matches to adjust");
    let sum: f64 = observations
        .iter()
        .map(|o| {
            let mut centered = o.features;
            for (c, m) in centered.iter_mut().zip(model.mean_features) {
                *c -= m;
            }
            o.score - dot(&model.beta, &centered)
        })
        .sum();
    #[allow(clippy::cast_precision_loss)]
    let n = observations.len() as f64;
    sum / n
}

/// `(1 - w) * role_mean + w * skill`.
#[must_use]
pub fn blend(weight: f64, role_mean: f64, skill: f64) -> f64 {
    (1.0 - weight) * role_mean + weight * skill
}

/// The generation's fitness vector and each genome's luck-adjusted term,
/// from every genome's mean role score and round-1 observations.
///
/// # Panics
///
/// Panics if the two slices differ in length or a genome has no matches.
#[must_use]
pub fn combine(
    weight: f64,
    role_means: &[f64],
    per_genome: &[Vec<Round1Obs>],
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(role_means.len(), per_genome.len());
    let model = fit_deal_model(per_genome).expect("at least one observation");
    let skill: Vec<f64> = per_genome
        .iter()
        .map(|obs| adjusted_score(&model, obs))
        .collect();
    let fitness = role_means
        .iter()
        .zip(&skill)
        .map(|(&r, &s)| blend(weight, r, s))
        .collect();
    (fitness, skill)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(score: f64, first_feature: f64) -> Round1Obs {
        let mut features = [0.0; P];
        features[0] = first_feature;
        // A second, independent feature so the design is full rank.
        features[1] = (first_feature * 1.7).sin();
        Round1Obs { score, features }
    }

    #[test]
    fn ols_recovers_a_known_feature_effect() {
        // score = 0.1 * f0 - 0.3 * f1 + 0.05 exactly.
        let data: Vec<Round1Obs> = (0..20)
            .map(|i| {
                let f0 = f64::from(i);
                let mut o = obs(0.0, f0);
                o.score = 0.1 * o.features[0] - 0.3 * o.features[1] + 0.05;
                o
            })
            .collect();
        let model = fit_deal_model(&[data]).unwrap();
        assert!((model.beta[0] - 0.1).abs() < 1e-6, "{:?}", model.beta);
        assert!((model.beta[1] + 0.3).abs() < 1e-6, "{:?}", model.beta);
        assert!((model.mean_features[0] - 9.5).abs() < 1e-12);
    }

    #[test]
    fn the_adjustment_removes_the_part_of_the_score_the_deal_explains() {
        // A genome that gets +0.1 per unit of f0 and is otherwise exactly
        // average: its two matches have different deals, so the plain
        // scores differ, but the adjusted per-match values agree.
        let mean = fit_deal_model(&[vec![obs(0.0, 0.0), obs(0.0, 10.0)]]).unwrap();
        let model = DealModel {
            beta: [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            mean_features: mean.mean_features,
        };
        let weak = obs(0.1 * 0.0 + 0.2, 0.0);
        let strong = obs(0.1 * 10.0 + 0.2, 10.0);
        let a = adjusted_score(&model, &[weak]);
        let b = adjusted_score(&model, &[strong]);
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        // Hand-computed: mean f0 = 5, so adjusted = 0.2 + 0.1 * 5 = 0.7.
        assert!((a - 0.7).abs() < 1e-12, "{a}");
    }

    #[test]
    fn on_shared_matches_the_adjusted_mean_equals_the_plain_round1_mean() {
        // Same deals for every genome (as in training), different results.
        let deals = [0.0, 3.0, 7.0, 1.0, 9.0, 4.0];
        let genomes: Vec<Vec<Round1Obs>> = [0.2, -0.1, 0.5]
            .iter()
            .map(|skill| {
                deals
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| obs(skill + 0.07 * d + if i % 2 == 0 { 0.3 } else { -0.3 }, d))
                    .collect()
            })
            .collect();
        let role_means = vec![0.0; 3];
        let (_, skill) = combine(0.5, &role_means, &genomes);
        for (g, s) in genomes.iter().zip(&skill) {
            #[allow(clippy::cast_precision_loss)]
            let plain = g.iter().map(|o| o.score).sum::<f64>() / g.len() as f64;
            assert!((plain - s).abs() < 1e-9, "{plain} vs {s}");
        }
    }

    #[test]
    fn blend_follows_the_documented_formula() {
        assert!((blend(0.0, 0.4, -0.2) - 0.4).abs() < 1e-15);
        assert!((blend(1.0, 0.4, -0.2) + 0.2).abs() < 1e-15);
        assert!((blend(0.25, 0.4, -0.2) - (0.75 * 0.4 + 0.25 * -0.2)).abs() < 1e-15);
        // Zero weight returns the role mean bit for bit.
        assert_eq!(
            blend(0.0, 0.123_456_789, 9.0).to_bits(),
            0.123_456_789_f64.to_bits()
        );
    }

    #[test]
    fn combine_ranks_by_the_blend_and_is_deterministic() {
        let deals = [0.0, 3.0, 7.0, 1.0];
        let genomes: Vec<Vec<Round1Obs>> = [0.5, -0.5]
            .iter()
            .map(|s| deals.iter().map(|&d| obs(*s, d)).collect())
            .collect();
        // Genome 0 is better on round 1, genome 1 on all rounds.
        let role_means = [0.0, 0.2];
        let (f0, _) = combine(0.0, &role_means, &genomes);
        assert!(f0[1] > f0[0]);
        let (f1, skill) = combine(1.0, &role_means, &genomes);
        assert!(f1[0] > f1[1]);
        assert!((f1[0] - skill[0]).abs() < 1e-15);
        assert_eq!(
            combine(0.5, &role_means, &genomes),
            combine(0.5, &role_means, &genomes)
        );
    }

    #[test]
    fn no_observations_means_no_model() {
        assert_eq!(fit_deal_model(&[]), None);
    }
}
