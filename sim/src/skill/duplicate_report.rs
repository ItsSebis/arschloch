//! The exact skill score from duplicate groups.

use super::{estimate, luck_share, mean, reduction, variance, Estimate};
use crate::duplicate::DuplicateGroup;
use crate::match_result::MatchResult;
use crate::training::evaluate::role_score;

/// Skill of one strategy (all seats sharing its name pooled).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct StrategySkill {
    pub name: String,
    /// Mean over groups of x(s,g), the mean over the group's `k` seatings
    /// of the strategy's per-match mean role score (all rounds). Later
    /// rounds start from carried-over roles, so the deal luck is only
    /// reduced there, not removed. `std_error` = sample sd of x(s,g)
    /// across groups / sqrt(groups); `n` = groups.
    pub skill_score_duplicate: Estimate,
    /// The same using round 1 only, where the deal luck cancels exactly.
    pub skill_score_duplicate_round1: Estimate,
    /// Mean of the strategy's per-match mean role score over all its
    /// matches, as an ordinary (non-duplicate) run would report it.
    /// `n` = matches; `std_error` = sd / sqrt(matches).
    pub plain_mean_role_score: Estimate,
    /// Round-1 plain mean (side by side with `skill_score_duplicate_round1`).
    pub plain_mean_role_score_round1: Estimate,
    /// `M = Var(single-match score) / (k * Var(x over groups))`: how many
    /// ordinary matches one duplicate match is worth for this strategy
    /// (all rounds). Variances are sample variances over the strategy's
    /// matches / groups; 1 when the single-match variance is 0; capped at
    /// 1e6 when the group means do not vary.
    pub variance_reduction: f64,
    /// `1 - 1/M` clamped to `[0, 1]`: the share of single-match variance
    /// that is deal luck.
    pub luck_share: f64,
    /// `variance_reduction` using round-1 scores only.
    pub variance_reduction_round1: f64,
    pub luck_share_round1: f64,
}

/// Duplicate-mode report.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SkillReport {
    pub group_count: usize,
    /// Matches per group = strategies per table.
    pub k: usize,
    pub rounds_per_match: usize,
    /// One entry per distinct strategy name, in first-seat order.
    pub strategies: Vec<StrategySkill>,
}

/// A match's score for `name`: mean over the seats named `name` and over
/// the rounds counted by `rounds` (`None` = all).
fn match_score(m: &MatchResult, name: &str, first_round_only: bool) -> Option<f64> {
    let seats: Vec<usize> = (0..m.strategy_names.len())
        .filter(|&s| m.strategy_names[s] == name)
        .collect();
    if seats.is_empty() {
        return None;
    }
    let rounds = if first_round_only {
        1
    } else {
        m.role_history.len()
    };
    let scores: Vec<f64> = m
        .role_history
        .iter()
        .take(rounds)
        .flat_map(|roles| seats.iter().map(|&s| role_score(roles[s], m.player_count)))
        .collect();
    Some(mean(&scores))
}

struct Series {
    per_match: Vec<f64>,
    per_group: Vec<f64>,
}

fn series(groups: &[DuplicateGroup], name: &str, first_round_only: bool) -> Series {
    let mut per_match = Vec::new();
    let mut per_group = Vec::new();
    for group in groups {
        let scores: Vec<f64> = group
            .matches
            .iter()
            .filter_map(|m| match_score(m, name, first_round_only))
            .collect();
        if !scores.is_empty() {
            per_group.push(mean(&scores));
            per_match.extend(scores);
        }
    }
    Series {
        per_match,
        per_group,
    }
}

fn reduction_of(s: &Series, k: usize) -> f64 {
    #[allow(clippy::cast_precision_loss)] // k is a table size
    reduction(variance(&s.per_match), k as f64 * variance(&s.per_group))
}

/// Builds the duplicate-mode report. Strategies are identified by name
/// (seats with the same name are pooled); a group lacking a name is
/// skipped for that strategy. An empty `groups` yields an empty report.
#[must_use]
pub fn skill_report(groups: &[DuplicateGroup]) -> SkillReport {
    let Some(first) = groups.first().and_then(|g| g.matches.first()) else {
        return SkillReport {
            group_count: 0,
            k: 0,
            rounds_per_match: 0,
            strategies: Vec::new(),
        };
    };
    let mut names: Vec<&str> = Vec::new();
    for name in &first.strategy_names {
        if !names.contains(&name.as_str()) {
            names.push(name);
        }
    }
    let k = first.strategy_names.len();
    let strategies = names
        .into_iter()
        .map(|name| {
            let all = series(groups, name, false);
            let one = series(groups, name, true);
            let (m_all, m_one) = (reduction_of(&all, k), reduction_of(&one, k));
            StrategySkill {
                name: name.to_string(),
                skill_score_duplicate: estimate(&all.per_group),
                skill_score_duplicate_round1: estimate(&one.per_group),
                plain_mean_role_score: estimate(&all.per_match),
                plain_mean_role_score_round1: estimate(&one.per_match),
                variance_reduction: m_all,
                luck_share: luck_share(m_all),
                variance_reduction_round1: m_one,
                luck_share_round1: luck_share(m_one),
            }
        })
        .collect();
    SkillReport {
        group_count: groups.len(),
        k,
        rounds_per_match: first.role_history.len(),
        strategies,
    }
}
