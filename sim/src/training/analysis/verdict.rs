//! The verdict rules: why a run stopped improving. Every message quotes
//! the numbers that made the rule fire. Rules are independent, so several
//! can fire; the order is the order of importance.

use serde::Serialize;

use super::{RunAnalysis, TrendClass};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VerdictCode {
    StillImproving,
    NoiseLimited,
    PoolLimited,
    ModelLimited,
    GrowthWithoutGain,
    Converged,
    StopEarly,
    /// No rule fired.
    NoClearLimit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Verdict {
    pub code: VerdictCode,
    pub message: String,
}

/// A stall this close to the last generation (or later) counts as still
/// improving; one this early (or earlier) as a stop that came too late.
const STILL_IMPROVING_FRACTION: f64 = 0.95;
const STOP_EARLY_FRACTION: f64 = 0.60;
const NOISE_CHANGE_FACTOR: f64 = 2.0;
const CONVERGED_SHARE: f64 = 0.5;

fn verdict(code: VerdictCode, message: String) -> Verdict {
    Verdict { code, message }
}

/// Applies the seven rules to an analysis (its own `verdict` is ignored).
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
pub fn verdict_for(a: &RunAnalysis) -> Vec<Verdict> {
    let mut out = Vec::new();
    let last_index = a.generations.saturating_sub(1) as f64;
    let stall = a.stall_generation as f64;
    let last_third = a.trend.last_third;
    let last_flat = last_third.is_some_and(|s| s.contains_zero());
    let interval = |s: Option<super::Slope>| {
        s.map_or_else(
            || "n/a".to_string(),
            |s| {
                format!(
                    "{:+.4} [{:+.4}, {:+.4}] per 100 generations",
                    s.value, s.low, s.high
                )
            },
        )
    };

    // 1. still improving
    let rising_end = last_third.is_some_and(|s| s.is_positive());
    let near_end = last_index > 0.0 && stall >= STILL_IMPROVING_FRACTION * last_index;
    let still_improving = near_end || rising_end;
    if still_improving {
        out.push(verdict(
            VerdictCode::StillImproving,
            format!(
                "still improving at the end (stall generation {} of {}, last-third trend {}): \
                 train longer (--resume/--from)",
                a.stall_generation,
                a.generations.saturating_sub(1),
                interval(last_third),
            ),
        ));
    }

    // 2. noise limited
    let change_sd = a.noise.change_sd;
    let gap_in_se = a.curse.reeval_minus_heldout.and_then(|g| g.in_heldout_se);
    let noisy_changes = change_sd.is_some_and(|sd| sd >= NOISE_CHANGE_FACTOR * a.noise.single_se);
    let big_gap = gap_in_se.is_some_and(|g| g >= 1.0);
    let noise_limited = last_flat && (noisy_changes || big_gap);
    if noise_limited {
        out.push(verdict(
            VerdictCode::NoiseLimited,
            format!(
                "progress is below the noise (last-third trend {}, change sd {} vs single se {:.4}, \
                 re-evaluation minus held-out {} held-out se): more --matches-per-genome, the \
                 duplicate-deal skill fitness, several seeds and choose the winner on fresh matches",
                interval(last_third),
                change_sd.map_or_else(|| "n/a".to_string(), |sd| format!("{sd:.4}")),
                a.noise.single_se,
                gap_in_se.map_or_else(|| "n/a".to_string(), |g| format!("{g:.2}")),
            ),
        ));
    }

    // 3. pool limited
    let rising: Vec<&str> = a
        .opponents
        .iter()
        .filter(|o| o.class == Some(TrendClass::Rising))
        .map(|o| o.name.as_str())
        .collect();
    let flat: Vec<&str> = a
        .opponents
        .iter()
        .filter(|o| o.class == Some(TrendClass::Flat))
        .map(|o| o.name.as_str())
        .collect();
    if last_flat && !rising.is_empty() && !flat.is_empty() {
        out.push(verdict(
            VerdictCode::PoolLimited,
            format!(
                "the pool no longer teaches against the flat opponents (rising: {}; flat: {}): \
                 harder or additional opponents",
                rising.join(", "),
                flat.join(", "),
            ),
        ));
    }

    // 4. model limited
    if last_flat && rising.is_empty() && !noise_limited {
        out.push(verdict(
            VerdictCode::ModelLimited,
            format!(
                "flat everywhere above the noise (last-third trend {}, no opponent rising): \
                 new inputs (--features hidden) or a different fitness",
                interval(last_third),
            ),
        ));
    }

    // 5. growth without gain
    if a.growth.without_gain {
        out.push(verdict(
            VerdictCode::GrowthWithoutGain,
            format!(
                "the network keeps growing without gain (hidden nodes {} -> {}, connections {} -> {}, \
                 last-third trend {}): stagnation limit / complexity penalty",
                a.growth.first_hidden_nodes,
                a.growth.last_hidden_nodes,
                a.growth.first_connections,
                a.growth.last_connections,
                interval(last_third),
            ),
        ));
    }

    // 6. converged
    if let Some(share) = a.population.top_bin_share.filter(|s| *s >= CONVERGED_SHARE) {
        out.push(verdict(
            VerdictCode::Converged,
            format!(
                "the population has converged ({:.0}% of the genomes in the top fitness bin over \
                 the last third): raise mutation rates / species target",
                share * 100.0
            ),
        ));
    }

    // 7. stop early
    if !still_improving && last_index > 0.0 && stall <= STOP_EARLY_FRACTION * last_index {
        out.push(verdict(
            VerdictCode::StopEarly,
            format!(
                "the last {} generations added nothing (stall generation {} of {}): stop earlier",
                a.generations - 1 - a.stall_generation,
                a.stall_generation,
                a.generations - 1,
            ),
        ));
    }

    if out.is_empty() {
        out.push(verdict(
            VerdictCode::NoClearLimit,
            "no clear limit found".into(),
        ));
    }
    out
}
