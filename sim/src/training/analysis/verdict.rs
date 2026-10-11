//! The verdict rules: why a run stopped improving. Every message quotes
//! the numbers that made the rule fire. Rules are independent, so several
//! can fire; the order is the order of importance.

use serde::Serialize;

use super::{RunAnalysis, TrendClass};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VerdictCode {
    StillImproving,
    Declining,
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

/// A smoothed peak this close to the last generation (or later) counts as
/// still improving; a stall this early (or earlier) as a stop that came too
/// late. Both are fractions of the last generation index.
const STILL_IMPROVING_FRACTION: f64 = 0.95;
const STOP_EARLY_FRACTION: f64 = 0.60;
const NOISE_CHANGE_FACTOR: f64 = 2.0;
/// The winner's-curse gap in standard errors of the difference of the two
/// measurements that counts as noise.
const NOISE_CURSE_COMBINED_SE: f64 = 2.0;
const CONVERGED_SHARE: f64 = 0.5;

fn verdict(code: VerdictCode, message: String) -> Verdict {
    Verdict { code, message }
}

fn interval(s: Option<super::Slope>) -> String {
    s.map_or_else(
        || "n/a".to_string(),
        |s| {
            format!(
                "{:+.4} [{:+.4}, {:+.4}] per 100 generations",
                s.value, s.low, s.high
            )
        },
    )
}

/// Applies the rules to an analysis (its own `verdict` is ignored). Rules
/// are independent except that `still_improving` shuts out `noise_limited`,
/// `pool_limited`, `model_limited`, `growth_without_gain` and `stop_early`
/// (a run that is still going has not reached a limit), and a declining
/// run is never still improving.
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
pub fn verdict_for(a: &RunAnalysis) -> Vec<Verdict> {
    let mut out = Vec::new();
    let last_index = a.generations.saturating_sub(1) as f64;
    let last_third = a.trend.last_third;
    let last_flat = last_third.is_some_and(|s| s.contains_zero());
    let last_falling = last_third.is_some_and(|s| s.is_negative());

    // 1. still improving: the smoothed peak is at the very end, or the last
    // third clearly rises; never when the last third clearly falls or the
    // run ended below its peak.
    let declining = last_falling || a.stall_fallback;
    let rising_end = last_third.is_some_and(|s| s.is_positive());
    let peak_at_end =
        last_index > 0.0 && a.peak_index as f64 >= STILL_IMPROVING_FRACTION * last_index;
    let still_improving = (peak_at_end || rising_end) && !declining;
    if still_improving {
        out.push(verdict(
            VerdictCode::StillImproving,
            format!(
                "still improving at the end (smoothed peak at generation {} of {}, last-third \
                 trend {}): train longer (--resume/--from)",
                a.peak_generation,
                a.last_generation,
                interval(last_third),
            ),
        ));
    }

    // 1b. declining: the end is worse than the peak.
    if declining {
        out.push(verdict(
            VerdictCode::Declining,
            format!(
                "the smoothed score peaked at generation {} ({:.3}), now {:.3} \
                 (last-third trend {}): the final champion is not the best one; use the best \
                 generation's genome / stop earlier",
                a.peak_generation,
                a.peak_smoothed,
                a.final_smoothed,
                interval(last_third),
            ),
        ));
    }

    // 2. noise limited
    let change_sd = a.noise.change_sd;
    let gap_in_se = a.curse.reeval_minus_heldout.and_then(|g| g.in_combined_se);
    let noisy_changes = a.noise.single_se > 0.0
        && change_sd.is_some_and(|sd| sd > NOISE_CHANGE_FACTOR * a.noise.single_se);
    let big_gap = gap_in_se.is_some_and(|g| g >= NOISE_CURSE_COMBINED_SE);
    let noise_limited = !still_improving && last_flat && (noisy_changes || big_gap);
    if noise_limited {
        out.push(verdict(
            VerdictCode::NoiseLimited,
            format!(
                "the last-third trend ({}) is not distinguishable from zero given the noise \
                 (change sd {} vs single se {:.4}, re-evaluation minus held-out {} combined se): \
                 more --matches-per-genome, the duplicate-deal skill fitness, several seeds and \
                 choose the winner on fresh matches",
                interval(last_third),
                change_sd.map_or_else(|| "n/a".to_string(), |sd| format!("{sd:.4}")),
                a.noise.single_se,
                gap_in_se.map_or_else(|| "n/a".to_string(), |g| format!("{g:.2}")),
            ),
        ));
    }

    // 3. pool limited (opponent classes are those of the last third)
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
    if !still_improving && last_flat && !rising.is_empty() && !flat.is_empty() {
        out.push(verdict(
            VerdictCode::PoolLimited,
            format!(
                "over the last third the pool no longer teaches against the flat opponents \
                 (rising: {}; flat: {}): harder or additional opponents",
                rising.join(", "),
                flat.join(", "),
            ),
        ));
    }

    // 4. model limited
    if !still_improving && last_flat && rising.is_empty() && !noise_limited {
        out.push(verdict(
            VerdictCode::ModelLimited,
            format!(
                "no progress left in the last third that the noise does not explain (last-third trend {}, no opponent rising): \
                 new inputs (--features hidden) or a different fitness",
                interval(last_third),
            ),
        ));
    }

    // 5. growth without gain
    if !still_improving && a.growth.without_gain {
        let w = a.growth.third_window;
        out.push(verdict(
            VerdictCode::GrowthWithoutGain,
            format!(
                "the network keeps growing without gain (mean hidden nodes {:.1} over the first {w} \
                 generations of the last third -> {:.1} over its last {w}; last-third trend {}): \
                 stagnation limit / complexity penalty",
                a.growth.third_start_nodes,
                a.growth.third_end_nodes,
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

    // 7. stop early: not while still improving, not when the end is rising
    // or the best generation came after the stall.
    let rising_estimate = last_third.is_some_and(|s| s.value > 0.0);
    let best_after_stall = a.best_generation.is_some_and(|b| b > a.stall_generation);
    if !still_improving
        && last_index > 0.0
        && a.stall_index as f64 <= STOP_EARLY_FRACTION * last_index
        && !rising_estimate
        && !best_after_stall
    {
        out.push(verdict(
            VerdictCode::StopEarly,
            format!(
                "after generation {} the smoothed score stayed within one noise unit ({:.3}) of \
                 its peak {:.3}: the last {} generations (up to generation {}) added nothing \
                 visible; stop earlier",
                a.stall_generation,
                a.noise.single_se,
                a.peak_smoothed,
                a.last_generation - a.stall_generation,
                a.last_generation,
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
