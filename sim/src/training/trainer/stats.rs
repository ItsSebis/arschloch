//! Ranking and summary statistics of a generation's fitness values.

use super::super::events::{FitnessStats, SkillTermStats};

/// The indices of the `k` highest values, best first (ties: lower index).
pub(super) fn top_indices(values: &[f64], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    order.truncate(k);
    order
}

#[allow(clippy::cast_precision_loss)] // counts are far below 2^52
pub(super) fn fitness_stats(fitness: &[f64], skill: Option<&[f64]>) -> FitnessStats {
    let count = fitness.len() as f64;
    let mean = fitness.iter().sum::<f64>() / count;
    let mut sorted = fitness.to_vec();
    sorted.sort_by(f64::total_cmp);
    let (min, best) = (sorted[0], sorted[sorted.len() - 1]);
    let median = if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        f64::midpoint(sorted[sorted.len() / 2 - 1], sorted[sorted.len() / 2])
    };
    let std_dev = (fitness.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / count).sqrt();
    let mut histogram = vec![0u32; 10];
    for &value in fitness {
        let bucket = if best > min {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let bucket = (((value - min) / (best - min)) * 10.0) as usize;
            bucket.min(9)
        } else {
            0
        };
        histogram[bucket] += 1;
    }
    FitnessStats {
        best,
        mean,
        median,
        min,
        std_dev,
        histogram,
        skill_term: skill.map(|terms| {
            let n = terms.len() as f64;
            let mean = terms.iter().sum::<f64>() / n;
            SkillTermStats {
                mean,
                std_dev: (terms.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n).sqrt(),
            }
        }),
    }
}
