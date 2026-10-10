//! Duplicate deals: groups of `k` matches (k = number of strategies)
//! that all deal the same decks round by round while the strategies
//! rotate through every seat, so each strategy plays every seat's hands
//! exactly once. Comparing strategies inside a group removes the luck of
//! the deal (see `crate::skill`).

use std::sync::Arc;

use rayon::prelude::*;

use crate::match_config::MatchConfig;
use crate::match_result::MatchResult;
use crate::match_runner::{run_match_with, RunOptions};
use crate::strategy::Strategy;
use crate::training::evaluate::mix;

/// Why a duplicate batch cannot be run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DuplicateError {
    NoStrategies,
    /// The number of matches is not a multiple of the number of strategies.
    NotAMultiple {
        matches: usize,
        k: usize,
    },
    /// A config's table size differs from the number of strategies.
    TableSizeMismatch {
        player_count: u8,
        k: usize,
    },
    /// Configs of one group must share deck, rounds and rules.
    InconsistentGroup {
        group: usize,
    },
}

impl std::fmt::Display for DuplicateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoStrategies => write!(f, "duplicate mode needs at least one strategy"),
            Self::NotAMultiple { matches, k } => write!(
                f,
                "duplicate mode plays groups of {k} matches (one per rotation of the {k} \
                 strategies), so the match count must be a multiple of {k}, got {matches}"
            ),
            Self::TableSizeMismatch { player_count, k } => write!(
                f,
                "duplicate mode needs one strategy per seat: {k} strategies for a \
                 {player_count}-player table"
            ),
            Self::InconsistentGroup { group } => write!(
                f,
                "the configs of duplicate group {group} differ in more than their seed"
            ),
        }
    }
}

impl std::error::Error for DuplicateError {}

/// One duplicate group: `k` matches with identical deals.
///
/// Match `r` (rotation `r`) seats strategy `(seat + r) % k` of the input
/// table at `seat`, so strategy `i` sits at seat `(i + k - r) % k` in
/// rotation `r` (`seat_of`). Strategies sharing a name (for example two
/// `RandomLegal` in one table) are treated by `crate::skill` as one
/// pooled strategy: all seats with that name are averaged.
#[derive(Debug, Clone)]
pub struct DuplicateGroup {
    /// The seed every match of the group deals from.
    pub deal_seed: u64,
    /// Rotation `r` in order.
    pub matches: Vec<MatchResult>,
}

impl DuplicateGroup {
    /// Seats of the table: `k`.
    #[must_use]
    pub fn k(&self) -> usize {
        self.matches.len()
    }

    /// Seat of input-table strategy `strategy_index` in rotation `rotation`.
    #[must_use]
    pub fn seat_of(&self, strategy_index: usize, rotation: usize) -> usize {
        let k = self.k();
        (strategy_index + k - rotation % k) % k
    }
}

/// The deal seed of the group whose first config has `first_seed`.
#[must_use]
pub fn group_deal_seed(first_seed: u64) -> u64 {
    mix(first_seed ^ 0xD0B1_E5EE_D5EE_D001)
}

/// Runs `configs` as consecutive duplicate groups of `strategies.len()`
/// matches, in parallel across all matches. Every group's matches deal
/// from `group_deal_seed(first config's seed)`; their play seeds remain
/// each config's own. With `record_deal_features` each match records its
/// round-1 `HandFeatures`.
///
/// # Errors
///
/// `DuplicateError` if the batch shape is wrong (see variants).
pub fn try_run_duplicate_batch(
    configs: &[MatchConfig],
    strategies: &[Arc<dyn Strategy>],
    record_deal_features: bool,
) -> Result<Vec<DuplicateGroup>, DuplicateError> {
    let k = strategies.len();
    if k == 0 {
        return Err(DuplicateError::NoStrategies);
    }
    if !configs.len().is_multiple_of(k) {
        return Err(DuplicateError::NotAMultiple {
            matches: configs.len(),
            k,
        });
    }
    for (group, chunk) in configs.chunks(k).enumerate() {
        if chunk.iter().any(|c| usize::from(c.player_count) != k) {
            return Err(DuplicateError::TableSizeMismatch {
                player_count: chunk[0].player_count,
                k,
            });
        }
        let same = |a: &MatchConfig, b: &MatchConfig| {
            MatchConfig { seed: 0, ..*a } == MatchConfig { seed: 0, ..*b }
        };
        if chunk.iter().any(|c| !same(c, &chunk[0])) {
            return Err(DuplicateError::InconsistentGroup { group });
        }
    }

    let matches: Vec<MatchResult> = configs
        .par_iter()
        .enumerate()
        .map(|(index, config)| {
            let group_start = index - index % k;
            let rotation = index % k;
            let rotated: Vec<Arc<dyn Strategy>> = strategies
                .iter()
                .cycle()
                .skip(rotation)
                .take(k)
                .cloned()
                .collect();
            let options = RunOptions {
                deal_seed: Some(group_deal_seed(configs[group_start].seed)),
                record_deal_features,
            };
            run_match_with(config, &rotated, &options)
        })
        .collect();

    Ok(matches
        .chunks(k)
        .zip(configs.chunks(k))
        .map(|(chunk, cfgs)| DuplicateGroup {
            deal_seed: group_deal_seed(cfgs[0].seed),
            matches: chunk.to_vec(),
        })
        .collect())
}

/// `try_run_duplicate_batch` without deal features.
///
/// # Panics
///
/// Panics on a malformed batch; call `try_run_duplicate_batch` to get the
/// error instead (the CLI does).
#[must_use]
pub fn run_duplicate_batch(
    configs: &[MatchConfig],
    strategies: &[Arc<dyn Strategy>],
) -> Vec<DuplicateGroup> {
    try_run_duplicate_batch(configs, strategies, false).expect("malformed duplicate batch")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategies::{LowestLegal, RandomLegal};
    use engine::{DeckVariant, DuplicateRule};

    pub(crate) fn configs(n: usize, rounds: usize) -> Vec<MatchConfig> {
        (0..n as u64)
            .map(|seed| MatchConfig {
                player_count: 4,
                deck_variant: DeckVariant::Single,
                duplicate_rule: DuplicateRule::FirstDealtWins,
                rounds,
                seed,
                pass_rule: engine::PassRule::default(),
                exchange_rule: engine::ExchangeRule::default(),
            })
            .collect()
    }

    fn table() -> Vec<Arc<dyn Strategy>> {
        vec![
            Arc::new(LowestLegal),
            Arc::new(RandomLegal),
            Arc::new(LowestLegal),
            Arc::new(RandomLegal),
        ]
    }

    #[test]
    fn rejects_non_multiples_with_a_clear_message() {
        let err = try_run_duplicate_batch(&configs(6, 1), &table(), false).unwrap_err();
        assert_eq!(err, DuplicateError::NotAMultiple { matches: 6, k: 4 });
        assert!(err.to_string().contains("multiple of 4"));
    }

    #[test]
    fn rejects_table_size_mismatch() {
        let three: Vec<Arc<dyn Strategy>> = (0..3).map(|_| Arc::new(LowestLegal) as _).collect();
        let err = try_run_duplicate_batch(&configs(3, 1), &three, false).unwrap_err();
        assert!(matches!(err, DuplicateError::TableSizeMismatch { .. }));
    }

    #[test]
    fn every_strategy_sits_in_every_seat_exactly_once_per_group() {
        let strategies: Vec<Arc<dyn Strategy>> = vec![
            Arc::new(LowestLegal),
            Arc::new(RandomLegal),
            Arc::new(crate::strategies::GreedyHighest),
            Arc::new(crate::strategies::HoldBackPairs),
        ];
        let groups = run_duplicate_batch(&configs(8, 2), &strategies);
        assert_eq!(groups.len(), 2);
        for group in &groups {
            for (i, strategy) in strategies.iter().enumerate() {
                let mut seats: Vec<usize> = (0..4)
                    .map(|r| {
                        let seat = group.seat_of(i, r);
                        assert_eq!(group.matches[r].strategy_names[seat], strategy.name());
                        seat
                    })
                    .collect();
                seats.sort_unstable();
                assert_eq!(seats, vec![0, 1, 2, 3]);
            }
        }
    }

    #[test]
    fn round_one_scores_cancel_over_a_group() {
        use crate::training::evaluate::role_score;
        let groups = run_duplicate_batch(&configs(8, 3), &table());
        for group in &groups {
            let total: f64 = group
                .matches
                .iter()
                .flat_map(|m| m.role_history[0].iter().map(|&r| role_score(r, 4)))
                .sum();
            assert!(total.abs() < 1e-9);
        }
    }

    #[test]
    fn features_are_recorded_only_on_request() {
        let with = try_run_duplicate_batch(&configs(4, 1), &table(), true).unwrap();
        assert!(with[0]
            .matches
            .iter()
            .all(|m| m.first_hand_features.is_some()));
        let without = run_duplicate_batch(&configs(4, 1), &table());
        assert!(without[0]
            .matches
            .iter()
            .all(|m| m.first_hand_features.is_none()));
    }

    #[test]
    fn a_group_deals_the_same_hands_in_every_rotation() {
        let groups = try_run_duplicate_batch(&configs(4, 1), &table(), true).unwrap();
        let g = &groups[0];
        for r in 1..4 {
            // Rotation r shifts strategies, not hands: seat s holds the same cards.
            assert_eq!(
                g.matches[0].first_hand_features,
                g.matches[r].first_hand_features
            );
        }
    }
}
