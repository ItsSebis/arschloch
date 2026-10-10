//! Extended batch statistics: average finishing place with standard
//! errors, rank distributions, per-seat bias, Bradley-Terry strength
//! ratings and Wilson intervals (docs/superpowers/specs/
//! 2026-10-09-phase-12-statistics-skill-score-design.md, section B).
//!
//! Pure and deterministic: the only randomness is the bootstrap, which
//! uses a fixed seed from [`ExtendedOptions`].

mod intervals;
mod rating;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;

use engine::{roles_for_player_count, Role};

pub use intervals::{standard_error, wilson_interval, Interval};

use crate::match_result::MatchResult;
use crate::training::evaluate::role_score;
use rating::PairCounts;

/// A point estimate with its standard error. `n` is the number of
/// matches (series) that went into the standard error.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Estimate {
    pub value: f64,
    pub std_error: f64,
    pub n: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct StrategyStats {
    /// Finishing place, 1 = best .. players = worst.
    pub avg_rank: Estimate,
    /// Mean of `role_score` (+1 best .. -1 worst).
    pub mean_role_score: Estimate,
    /// Percent of rounds per place, best first; sums to 100.
    pub rank_distribution: Vec<f64>,
    pub rounds: u64,
    /// Bradley-Terry rating in Elo-like points, mean 0 over strategies;
    /// `None` when the batch has a single strategy name.
    pub strength_rating: Option<Estimate>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SeatStats {
    pub avg_rank: Estimate,
    pub mean_role_score: Estimate,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ExtendedStatistics {
    pub by_strategy: BTreeMap<String, StrategyStats>,
    /// Index = seat. Only meaningful when the batch rotates seats.
    pub by_seat: Vec<SeatStats>,
    pub role_retention_intervals: BTreeMap<String, BTreeMap<Role, Interval>>,
    pub voluntary_pass_rate_intervals: BTreeMap<String, Interval>,
}

#[derive(Debug, Clone, Copy)]
pub struct ExtendedOptions {
    /// Bootstrap resamples for rating standard errors; 0 disables.
    pub bootstrap_resamples: usize,
    pub bootstrap_seed: u64,
}

impl Default for ExtendedOptions {
    fn default() -> Self {
        Self {
            bootstrap_resamples: 200,
            bootstrap_seed: 0x5EED_B007,
        }
    }
}

/// Running sums for one (strategy or seat) group.
#[derive(Default)]
struct Acc {
    rank_sum: f64,
    score_sum: f64,
    rounds: u64,
    places: Vec<u64>,
    series_rank: Vec<f64>,
    series_score: Vec<f64>,
}

impl Acc {
    fn add_series(&mut self, places: &[usize], player_count: u8) {
        if places.is_empty() {
            return;
        }
        let roles = roles_for_player_count(player_count).expect("supported table size");
        let (mut rank, mut score) = (0.0, 0.0);
        if self.places.len() < roles.len() {
            self.places.resize(roles.len(), 0);
        }
        for &p in places {
            #[allow(clippy::cast_precision_loss)]
            let place_rank = (p + 1) as f64;
            let s = role_score(roles[p], player_count);
            rank += place_rank;
            score += s;
            self.places[p] += 1;
        }
        #[allow(clippy::cast_precision_loss)]
        let len = places.len() as f64;
        self.rank_sum += rank;
        self.score_sum += score;
        self.rounds += places.len() as u64;
        self.series_rank.push(rank / len);
        self.series_score.push(score / len);
    }

    #[allow(clippy::cast_precision_loss)]
    fn estimates(&self) -> (Estimate, Estimate) {
        let rounds = self.rounds.max(1) as f64;
        let n = self.series_rank.len();
        (
            Estimate {
                value: self.rank_sum / rounds,
                std_error: standard_error(&self.series_rank),
                n,
            },
            Estimate {
                value: self.score_sum / rounds,
                std_error: standard_error(&self.series_score),
                n,
            },
        )
    }
}

/// `aggregate_extended_with` using [`ExtendedOptions::default`].
#[must_use]
pub fn aggregate_extended(results: &[MatchResult]) -> ExtendedStatistics {
    aggregate_extended_with(results, &ExtendedOptions::default())
}

/// Aggregates `results` into [`ExtendedStatistics`].
///
/// # Panics
///
/// Panics if a result has an unsupported `player_count` (outside 3-6) or
/// a role that does not belong to that table.
#[must_use]
#[allow(clippy::too_many_lines)] // one linear pass over the batch
pub fn aggregate_extended_with(
    results: &[MatchResult],
    options: &ExtendedOptions,
) -> ExtendedStatistics {
    let names: Vec<String> = {
        let mut all: Vec<String> = results
            .iter()
            .flat_map(|r| r.strategy_names.iter().cloned())
            .collect();
        all.sort();
        all.dedup();
        all
    };
    let index_of = |name: &str| names.binary_search_by(|n| n.as_str().cmp(name)).unwrap();

    let mut strategy_acc: BTreeMap<String, Acc> = BTreeMap::new();
    let mut seat_acc: Vec<Acc> = Vec::new();
    let mut pair_counts: Vec<PairCounts> = Vec::with_capacity(results.len());
    let mut retention: BTreeMap<String, BTreeMap<Role, (u64, u64)>> = BTreeMap::new();
    let mut passes: BTreeMap<String, (u64, u64)> = BTreeMap::new();

    for result in results {
        let roles = roles_for_player_count(result.player_count).expect("supported table size");
        let places: Vec<Vec<usize>> = result
            .role_history
            .iter()
            .map(|round| {
                round
                    .iter()
                    .map(|r| roles.iter().position(|x| x == r).expect("role in table"))
                    .collect()
            })
            .collect();
        let seats = result.strategy_names.len();
        if seat_acc.len() < seats {
            seat_acc.resize_with(seats, Acc::default);
        }
        for seat in 0..seats {
            let series: Vec<usize> = places.iter().map(|round| round[seat]).collect();
            let name = &result.strategy_names[seat];
            strategy_acc
                .entry(name.clone())
                .or_default()
                .add_series(&series, result.player_count);
            seat_acc[seat].add_series(&series, result.player_count);

            let entry = passes.entry(name.clone()).or_insert((0, 0));
            entry.0 += u64::from(result.voluntary_pass_counts[seat]);
            entry.1 += u64::from(result.pass_counts[seat]);
        }
        for window in result.role_history.windows(2) {
            for (seat, &role) in window[0].iter().enumerate() {
                let entry = retention
                    .entry(result.strategy_names[seat].clone())
                    .or_default()
                    .entry(role)
                    .or_insert((0, 0));
                entry.0 += 1;
                entry.1 += u64::from(window[1][seat] == role);
            }
        }
        let mut counts = PairCounts::new(names.len());
        for round in &places {
            for a in 0..seats {
                for b in a + 1..seats {
                    let (ia, ib) = (
                        index_of(&result.strategy_names[a]),
                        index_of(&result.strategy_names[b]),
                    );
                    if round[a] < round[b] {
                        counts.record(ia, ib);
                    } else if round[b] < round[a] {
                        counts.record(ib, ia);
                    }
                }
            }
        }
        pair_counts.push(counts);
    }

    let ratings = if names.len() < 2 {
        None
    } else {
        Some(rating::rate(
            &pair_counts,
            names.len(),
            options.bootstrap_resamples,
            options.bootstrap_seed,
        ))
    };

    let by_strategy = strategy_acc
        .iter()
        .map(|(name, acc)| {
            let (avg_rank, mean_role_score) = acc.estimates();
            #[allow(clippy::cast_precision_loss)]
            let rank_distribution = acc
                .places
                .iter()
                .map(|&c| 100.0 * c as f64 / acc.rounds.max(1) as f64)
                .collect();
            let strength_rating = ratings.as_ref().map(|r| {
                let (value, std_error) = r[index_of(name)];
                Estimate {
                    value,
                    std_error,
                    n: results.len(),
                }
            });
            (
                name.clone(),
                StrategyStats {
                    avg_rank,
                    mean_role_score,
                    rank_distribution,
                    rounds: acc.rounds,
                    strength_rating,
                },
            )
        })
        .collect();
    let by_seat = seat_acc
        .iter()
        .map(|acc| {
            let (avg_rank, mean_role_score) = acc.estimates();
            SeatStats {
                avg_rank,
                mean_role_score,
            }
        })
        .collect();
    let role_retention_intervals = retention
        .into_iter()
        .map(|(name, roles)| {
            let map = roles
                .into_iter()
                .filter_map(|(role, (held, kept))| {
                    wilson_interval(kept, held).map(|(low, high)| (role, Interval { low, high }))
                })
                .collect();
            (name, map)
        })
        .collect();
    let voluntary_pass_rate_intervals = passes
        .into_iter()
        .filter_map(|(name, (vol, total))| {
            wilson_interval(vol, total).map(|(low, high)| (name, Interval { low, high }))
        })
        .collect();

    ExtendedStatistics {
        by_strategy,
        by_seat,
        role_retention_intervals,
        voluntary_pass_rate_intervals,
    }
}
