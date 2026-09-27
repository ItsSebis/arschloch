//! Aggregating a batch of `MatchResult`s into per-strategy role counts
//! and a diversification signal. See docs/ROADMAP.md, Phase 2 and 4.

use std::collections::BTreeMap;

use engine::Role;

use crate::match_result::MatchResult;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Statistics {
    pub matches_played: usize,
    pub role_counts_by_strategy: BTreeMap<String, BTreeMap<Role, u32>>,
    /// `sum(voluntary_pass_count) / sum(pass_count)` across every match
    /// (`0.0` if no passes occurred at all).
    pub voluntary_pass_rate: f64,
}

/// Aggregates `results` into `Statistics`. Every round in every match
/// contributes one role count per seat, credited to that seat's
/// strategy (by name — seats using the same strategy type share a
/// bucket).
#[must_use]
pub fn aggregate(results: &[MatchResult]) -> Statistics {
    let mut role_counts_by_strategy: BTreeMap<String, BTreeMap<Role, u32>> = BTreeMap::new();
    let mut total_passes = 0u64;
    let mut total_voluntary_passes = 0u64;

    for result in results {
        for round_roles in &result.role_history {
            for (seat, &role) in round_roles.iter().enumerate() {
                let strategy_name = result.strategy_names[seat].clone();
                *role_counts_by_strategy
                    .entry(strategy_name)
                    .or_default()
                    .entry(role)
                    .or_insert(0) += 1;
            }
        }
        for seat in 0..result.pass_counts.len() {
            total_passes += u64::from(result.pass_counts[seat]);
            total_voluntary_passes += u64::from(result.voluntary_pass_counts[seat]);
        }
    }

    let voluntary_pass_rate = if total_passes == 0 {
        0.0
    } else {
        // Precision loss only matters above 2^53 passes — far beyond any
        // simulated batch this project runs.
        #[allow(clippy::cast_precision_loss)]
        {
            total_voluntary_passes as f64 / total_passes as f64
        }
    };

    Statistics {
        matches_played: results.len(),
        role_counts_by_strategy,
        voluntary_pass_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(
        strategy_names: Vec<&str>,
        roles: Vec<Role>,
        pass_counts: Vec<u32>,
        voluntary_pass_counts: Vec<u32>,
    ) -> MatchResult {
        MatchResult {
            player_count: u8::try_from(strategy_names.len()).unwrap(),
            strategy_names: strategy_names.into_iter().map(String::from).collect(),
            role_history: vec![roles],
            trick_count: 0,
            pass_counts,
            voluntary_pass_counts,
        }
    }

    #[test]
    fn aggregate_counts_matches_and_roles_by_strategy() {
        let results = vec![
            result(
                vec!["LowestLegal", "GreedyHighest"],
                vec![Role::President, Role::Arschloch],
                vec![5, 5],
                vec![1, 1],
            ),
            result(
                vec!["LowestLegal", "GreedyHighest"],
                vec![Role::Arschloch, Role::President],
                vec![5, 5],
                vec![4, 4],
            ),
        ];
        let stats = aggregate(&results);
        assert_eq!(stats.matches_played, 2);
        assert_eq!(
            stats.role_counts_by_strategy["LowestLegal"][&Role::President],
            1
        );
        assert_eq!(
            stats.role_counts_by_strategy["LowestLegal"][&Role::Arschloch],
            1
        );
        assert!((stats.voluntary_pass_rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn aggregate_of_no_passes_reports_zero_rate() {
        let results = vec![result(
            vec!["LowestLegal"],
            vec![Role::President],
            vec![0],
            vec![0],
        )];
        let stats = aggregate(&results);
        // Exact equality is guaranteed, not approximate: `aggregate` returns
        // the hardcoded literal `0.0` on this path (no passes occurred),
        // never a computed ratio, so `float_cmp` is a false positive here.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(stats.voluntary_pass_rate, 0.0);
        }
    }
}
