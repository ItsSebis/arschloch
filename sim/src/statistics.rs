//! Aggregating a batch of `MatchResult`s into per-strategy role counts,
//! diversification, role-sustainment, and luck-vs-skill signals. See
//! docs/ROADMAP.md, Phase 2 and 4.

use std::collections::BTreeMap;

use engine::{roles_for_player_count, Role};

use crate::match_result::MatchResult;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct RoleRetention {
    /// Round-to-next-round transitions where a strategy held this role
    /// going in (and a next round existed to check).
    pub held: u32,
    /// Of `held`, how many kept the same role the very next round.
    pub retained_next_round: u32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Statistics {
    pub matches_played: usize,
    pub role_counts_by_strategy: BTreeMap<String, BTreeMap<Role, u32>>,
    /// `sum(voluntary_pass_counts) / sum(pass_counts)` across every match
    /// and seat (`0.0` if no passes occurred at all).
    pub voluntary_pass_rate: f64,
    /// Same ratio, broken out per strategy name (docs/ROADMAP.md, Phase 4,
    /// "Strategy diversification").
    pub voluntary_pass_rate_by_strategy: BTreeMap<String, f64>,
    /// Role-sustainment: of the times a strategy held a role with at
    /// least one more round left in the match, how often it held the
    /// *same* role the next round too (docs/ROADMAP.md, Phase 4,
    /// "Role-sustainment tracking"). A strategy that only ever appeared
    /// in single-round matches has no entry here at all (there is no
    /// "next round" to check).
    pub role_retention_by_strategy: BTreeMap<String, BTreeMap<Role, RoleRetention>>,
    /// Luck-vs-skill: variance of a strategy's first-round finishing
    /// placement (0 = the table's best role, per
    /// `engine::roles_for_player_count`'s order) across matches that
    /// seated it identically (the same `MatchResult::strategy_names`)
    /// but necessarily shuffled differently (docs/ROADMAP.md, Phase 4,
    /// "Luck-vs-skill signal"). `None` if every seating this strategy
    /// appeared in had fewer than 2 matches to compare (nothing to take a
    /// variance of) — e.g. any batch smaller than roughly
    /// `2 * strategies.len()` matches.
    pub first_round_placement_variance_by_strategy: BTreeMap<String, Option<f64>>,
}

/// Aggregates `results` into `Statistics`. Every round in every match
/// contributes one role count per seat, credited to that seat's
/// strategy (by name — seats using the same strategy type share a
/// bucket, for every field below, not just `role_counts_by_strategy`).
///
/// # Panics
///
/// Panics if any `MatchResult` in `results` has a `player_count` outside
/// the range 3-6 (all `MatchResult` instances must come from the `cli`
/// which validates `--player-count` at the boundary), or if a role
/// assigned by `assign_roles` is not in the list returned by
/// `engine::roles_for_player_count` (an internal invariant violation).
#[must_use]
pub fn aggregate(results: &[MatchResult]) -> Statistics {
    let mut role_counts_by_strategy: BTreeMap<String, BTreeMap<Role, u32>> = BTreeMap::new();
    let mut total_passes = 0u64;
    let mut total_voluntary_passes = 0u64;
    let mut passes_by_strategy: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let mut retention_by_strategy: BTreeMap<String, BTreeMap<Role, RoleRetention>> =
        BTreeMap::new();
    // Keyed by (seating, seat index) rather than (seating, strategy name):
    // a strategy occupying two seats in one match must contribute two
    // separately-tracked series, never two entries from a single match
    // into one series — each series is "the same seating, repeated
    // across matches," so within one series only the shuffle varies.
    let mut placements_by_seat: BTreeMap<(Vec<String>, usize), Vec<f64>> = BTreeMap::new();

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

        for (seat, strategy_name) in result.strategy_names.iter().enumerate() {
            let voluntary = u64::from(result.voluntary_pass_counts[seat]);
            let total = u64::from(result.pass_counts[seat]);
            total_passes += total;
            total_voluntary_passes += voluntary;
            let entry = passes_by_strategy
                .entry(strategy_name.clone())
                .or_insert((0, 0));
            entry.0 += voluntary;
            entry.1 += total;
        }

        for window in result.role_history.windows(2) {
            let (current, next) = (&window[0], &window[1]);
            for (seat, &role) in current.iter().enumerate() {
                let retention = retention_by_strategy
                    .entry(result.strategy_names[seat].clone())
                    .or_default()
                    .entry(role)
                    .or_default();
                retention.held += 1;
                if next[seat] == role {
                    retention.retained_next_round += 1;
                }
            }
        }

        if let Some(first_round) = result.role_history.first() {
            let roles = roles_for_player_count(result.player_count)
                .expect("MatchResult always comes from a supported player_count");
            for (seat, &role) in first_round.iter().enumerate() {
                let placement_index = roles
                    .iter()
                    .position(|&r| r == role)
                    .expect("assign_roles only assigns roles from this table's own role list");
                #[allow(clippy::cast_precision_loss)]
                let placement = placement_index as f64;
                placements_by_seat
                    .entry((result.strategy_names.clone(), seat))
                    .or_default()
                    .push(placement);
            }
        }
    }

    let voluntary_pass_rate = ratio(total_voluntary_passes, total_passes);
    let voluntary_pass_rate_by_strategy = passes_by_strategy
        .into_iter()
        .map(|(name, (voluntary, total))| (name, ratio(voluntary, total)))
        .collect();

    // Pooled variance across every seating a strategy appeared in
    // (weighted by degrees of freedom), skipping any seating with fewer
    // than 2 matches to compare.
    let mut variance_weighted: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    for ((seating, seat), placements) in &placements_by_seat {
        if placements.len() < 2 {
            continue;
        }
        let strategy_name = &seating[*seat];
        #[allow(clippy::cast_precision_loss)]
        let weight = (placements.len() - 1) as f64;
        let entry = variance_weighted
            .entry(strategy_name.clone())
            .or_insert((0.0, 0.0));
        entry.0 += sample_variance(placements) * weight;
        entry.1 += weight;
    }
    let mut first_round_placement_variance_by_strategy: BTreeMap<String, Option<f64>> =
        role_counts_by_strategy
            .keys()
            .map(|name| (name.clone(), None))
            .collect();
    for (name, (weighted_sum, weight)) in variance_weighted {
        first_round_placement_variance_by_strategy.insert(name, Some(weighted_sum / weight));
    }

    Statistics {
        matches_played: results.len(),
        role_counts_by_strategy,
        voluntary_pass_rate,
        voluntary_pass_rate_by_strategy,
        role_retention_by_strategy: retention_by_strategy,
        first_round_placement_variance_by_strategy,
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        // Precision loss only matters above 2^53 passes — far beyond any
        // simulated batch this project runs.
        #[allow(clippy::cast_precision_loss)]
        {
            numerator as f64 / denominator as f64
        }
    }
}

/// Sample variance (Bessel's correction, `n - 1` divisor). Requires
/// `values.len() >= 2` — with fewer values this returns `NaN` rather than
/// panicking; every caller already guards this length before calling.
fn sample_variance(values: &[f64]) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(
        player_count: u8,
        strategy_names: Vec<&str>,
        role_history: Vec<Vec<Role>>,
        pass_counts: Vec<u32>,
        voluntary_pass_counts: Vec<u32>,
    ) -> MatchResult {
        MatchResult {
            player_count,
            strategy_names: strategy_names.into_iter().map(String::from).collect(),
            role_history,
            trick_count: 0,
            pass_counts,
            voluntary_pass_counts,
        }
    }

    #[test]
    fn aggregate_counts_matches_and_roles_by_strategy() {
        let results = vec![
            result(
                3,
                vec!["LowestLegal", "GreedyHighest"],
                vec![vec![Role::President, Role::Arschloch]],
                vec![5, 5],
                vec![1, 1],
            ),
            result(
                3,
                vec!["LowestLegal", "GreedyHighest"],
                vec![vec![Role::Arschloch, Role::President]],
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
            3,
            vec!["LowestLegal"],
            vec![vec![Role::President]],
            vec![0],
            vec![0],
        )];
        let stats = aggregate(&results);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(stats.voluntary_pass_rate, 0.0);
        }
    }

    #[test]
    fn voluntary_pass_rate_by_strategy_distinguishes_strategies() {
        let results = vec![
            result(
                3,
                vec!["LowestLegal", "GreedyHighest"],
                vec![vec![Role::President, Role::Arschloch]],
                vec![10, 10],
                vec![0, 10],
            ),
            result(
                3,
                vec!["LowestLegal", "GreedyHighest"],
                vec![vec![Role::Arschloch, Role::President]],
                vec![10, 10],
                vec![0, 10],
            ),
        ];
        let stats = aggregate(&results);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(stats.voluntary_pass_rate_by_strategy["LowestLegal"], 0.0);
            assert_eq!(stats.voluntary_pass_rate_by_strategy["GreedyHighest"], 1.0);
        }
    }

    #[test]
    fn role_retention_counts_round_to_round_transitions() {
        let results = vec![result(
            3,
            vec!["LowestLegal", "RandomLegal"],
            vec![
                vec![Role::President, Role::Arschloch],
                vec![Role::President, Role::Arschloch],
                vec![Role::Arschloch, Role::President],
            ],
            vec![0, 0],
            vec![0, 0],
        )];
        let stats = aggregate(&results);
        assert_eq!(
            stats.role_retention_by_strategy["LowestLegal"][&Role::President],
            RoleRetention {
                held: 2,
                retained_next_round: 1
            }
        );
        assert_eq!(
            stats.role_retention_by_strategy["RandomLegal"][&Role::Arschloch],
            RoleRetention {
                held: 2,
                retained_next_round: 1
            }
        );
    }

    #[test]
    fn single_round_matches_produce_no_retention_entries() {
        let results = vec![result(
            3,
            vec!["LowestLegal"],
            vec![vec![Role::President]],
            vec![0],
            vec![0],
        )];
        let stats = aggregate(&results);
        assert!(stats.role_retention_by_strategy.is_empty());
    }

    #[test]
    fn first_round_placement_variance_is_computed_across_repeated_identical_seatings() {
        let results = vec![
            result(
                3,
                vec!["LowestLegal", "GreedyHighest", "RandomLegal"],
                vec![vec![Role::President, Role::Dorftrottel, Role::Arschloch]],
                vec![0, 0, 0],
                vec![0, 0, 0],
            ),
            result(
                3,
                vec!["LowestLegal", "GreedyHighest", "RandomLegal"],
                vec![vec![Role::Arschloch, Role::Dorftrottel, Role::President]],
                vec![0, 0, 0],
                vec![0, 0, 0],
            ),
        ];
        let stats = aggregate(&results);
        // Placements (0=President, 1=Dorftrottel, 2=Arschloch): LowestLegal
        // saw [0, 2], GreedyHighest saw [1, 1], RandomLegal saw [2, 0].
        let lowest_legal_variance = stats.first_round_placement_variance_by_strategy["LowestLegal"]
            .expect("seating repeated twice");
        assert!((lowest_legal_variance - 2.0).abs() < 1e-9);
        let greedy_highest_variance = stats.first_round_placement_variance_by_strategy
            ["GreedyHighest"]
            .expect("seating repeated twice");
        assert!((greedy_highest_variance - 0.0).abs() < 1e-9);
        let random_legal_variance = stats.first_round_placement_variance_by_strategy["RandomLegal"]
            .expect("seating repeated twice");
        assert!((random_legal_variance - 2.0).abs() < 1e-9);
    }

    #[test]
    fn first_round_placement_variance_is_none_for_a_seating_seen_only_once() {
        let results = vec![result(
            3,
            vec!["LowestLegal", "GreedyHighest", "RandomLegal"],
            vec![vec![Role::President, Role::Dorftrottel, Role::Arschloch]],
            vec![0, 0, 0],
            vec![0, 0, 0],
        )];
        let stats = aggregate(&results);
        assert_eq!(
            stats.first_round_placement_variance_by_strategy["LowestLegal"],
            None
        );
        assert_eq!(
            stats.first_round_placement_variance_by_strategy["GreedyHighest"],
            None
        );
        assert_eq!(
            stats.first_round_placement_variance_by_strategy["RandomLegal"],
            None
        );
    }

    #[test]
    fn a_single_match_with_a_repeated_strategy_reports_no_variance() {
        // "LowestLegal" occupies two seats in the same (and only) match.
        // Each seat's own placement series has just 1 sample (this match),
        // so neither reaches the 2-sample minimum — even though 2 seats'
        // worth of placements exist, they must not be pooled together as
        // if they were 2 repeats of one series.
        let results = vec![result(
            3,
            vec!["LowestLegal", "LowestLegal", "GreedyHighest"],
            vec![vec![Role::President, Role::Dorftrottel, Role::Arschloch]],
            vec![0, 0, 0],
            vec![0, 0, 0],
        )];
        let stats = aggregate(&results);
        assert_eq!(
            stats.first_round_placement_variance_by_strategy["LowestLegal"],
            None
        );
    }

    #[test]
    fn first_round_placement_variance_pools_across_multiple_seatings_of_the_same_strategy() {
        let results = vec![
            // Seating A: LowestLegal in seat 0, repeated twice, placements
            // [0, 2] (0=President, 1=Dorftrottel, 2=Arschloch) -> sample
            // variance 2.0, weight (n - 1) = 1.
            result(
                3,
                vec!["LowestLegal", "GreedyHighest", "RandomLegal"],
                vec![vec![Role::President, Role::Dorftrottel, Role::Arschloch]],
                vec![0, 0, 0],
                vec![0, 0, 0],
            ),
            result(
                3,
                vec!["LowestLegal", "GreedyHighest", "RandomLegal"],
                vec![vec![Role::Arschloch, Role::Dorftrottel, Role::President]],
                vec![0, 0, 0],
                vec![0, 0, 0],
            ),
            // Seating B: a different seating (seats 1 and 2 swapped
            // strategies) that still seats LowestLegal in seat 0, repeated
            // twice, placements [1, 1] -> sample variance 0.0, weight 1.
            result(
                3,
                vec!["LowestLegal", "RandomLegal", "GreedyHighest"],
                vec![vec![Role::Dorftrottel, Role::Arschloch, Role::President]],
                vec![0, 0, 0],
                vec![0, 0, 0],
            ),
            result(
                3,
                vec!["LowestLegal", "RandomLegal", "GreedyHighest"],
                vec![vec![Role::Dorftrottel, Role::President, Role::Arschloch]],
                vec![0, 0, 0],
                vec![0, 0, 0],
            ),
        ];
        let stats = aggregate(&results);
        // Pooled by degrees of freedom: (2.0 * 1 + 0.0 * 1) / (1 + 1) = 1.0.
        let pooled = stats.first_round_placement_variance_by_strategy["LowestLegal"]
            .expect("two distinct seatings, each repeated twice");
        assert!((pooled - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_strategy_used_in_two_seats_merges_retention_transitions_too() {
        let results = vec![result(
            3,
            vec!["LowestLegal", "LowestLegal"],
            vec![
                vec![Role::President, Role::Arschloch],
                vec![Role::President, Role::Arschloch],
                vec![Role::Arschloch, Role::President],
            ],
            vec![0, 0],
            vec![0, 0],
        )];
        let stats = aggregate(&results);
        // Seat 0 (President -> President -> Arschloch) and seat 1
        // (Arschloch -> Arschloch -> President) both count as "LowestLegal",
        // so their transitions merge into one bucket per role: each role
        // is held twice (once per round-to-round window), retained once.
        assert_eq!(
            stats.role_retention_by_strategy["LowestLegal"][&Role::President],
            RoleRetention {
                held: 2,
                retained_next_round: 1
            }
        );
        assert_eq!(
            stats.role_retention_by_strategy["LowestLegal"][&Role::Arschloch],
            RoleRetention {
                held: 2,
                retained_next_round: 1
            }
        );
    }

    #[test]
    fn a_strategy_used_in_two_seats_shares_one_bucket_across_every_field() {
        let results = vec![result(
            3,
            vec!["LowestLegal", "LowestLegal"],
            vec![vec![Role::President, Role::Arschloch]],
            vec![3, 5],
            vec![1, 2],
        )];
        let stats = aggregate(&results);
        // Both seats' role assignments land in the same "LowestLegal" bucket.
        assert_eq!(
            stats.role_counts_by_strategy["LowestLegal"][&Role::President],
            1
        );
        assert_eq!(
            stats.role_counts_by_strategy["LowestLegal"][&Role::Arschloch],
            1
        );
        // Both seats' pass counts pool into one rate: (1+2)/(3+5) = 0.375.
        assert!(
            (stats.voluntary_pass_rate_by_strategy["LowestLegal"] - 0.375).abs() < f64::EPSILON
        );
    }
}
