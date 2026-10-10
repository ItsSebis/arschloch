//! Fitness evaluation: how well one candidate plays against opponents.
//!
//! A candidate's score is its mean finishing-role score over many
//! matches, from +1 (always President) to -1 (always last). The seat it
//! sits in and the opponents it faces are a pure function of the match
//! seed, so every genome in a generation can be evaluated on exactly the
//! same deals, seats and opponents ("common random numbers"): score
//! differences then reflect the genome, not the luck of the shuffle.

use std::sync::Arc;

use engine::{roles_for_player_count, DeckVariant, DuplicateRule, ExchangeRule, PassRule, Role};
use rand::rngs::Xoshiro256PlusPlus;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

use crate::hand_features::HandFeatures;
use crate::{run_match_with, MatchConfig, RunOptions, Strategy};

/// What a candidate plays: the table and the length of each match.
#[derive(Debug, Clone, Copy)]
pub struct TableSpec {
    pub player_count: u8,
    pub deck_variant: DeckVariant,
    pub duplicate_rule: DuplicateRule,
    pub pass_rule: PassRule,
    pub exchange_rule: ExchangeRule,
    pub rounds: usize,
}

/// Who fills the other seats.
#[derive(Clone, Copy)]
pub enum Opponents<'a> {
    /// Each other seat independently draws from the pool, per match.
    Mixed(&'a [Arc<dyn Strategy>]),
    /// Every other seat plays this one strategy.
    Only(&'a Arc<dyn Strategy>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    pub mean: f64,
    /// Standard error of `mean` across matches (0 for a single match).
    pub std_error: f64,
    pub matches: usize,
    /// How often the candidate finished in each place, best first
    /// (`placements[0]` is President), over every round of every match.
    pub placements: Vec<u64>,
}

/// The `SplitMix64` finalizer: a cheap, well-mixed 64-bit hash.
#[must_use]
pub fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// The seed of match `index` in `stream` (for example generation `g`'s
/// training matches, or its champion re-evaluation) of a run.
#[must_use]
pub fn match_seed(run_seed: u64, stream: u64, index: u64) -> u64 {
    mix(mix(run_seed ^ mix(stream)) ^ index)
}

/// `+1` for the best role of a table of `player_count`, `-1` for the
/// worst, linear in between.
///
/// # Panics
///
/// Panics if `player_count` is not a supported table size (3-6).
#[must_use]
pub fn role_score(role: Role, player_count: u8) -> f64 {
    let roles = roles_for_player_count(player_count).expect("supported table size");
    let rank = roles
        .iter()
        .position(|&r| r == role)
        .expect("role belongs to this table");
    #[allow(clippy::cast_precision_loss)] // table sizes are 3-6
    let (rank, last) = (rank as f64, (roles.len() - 1) as f64);
    1.0 - 2.0 * rank / last
}

/// What one match says about the candidate's first round: the role
/// score it finished with and the features of the hand it was dealt
/// (before the exchange). Used by the optional luck-adjusted fitness
/// term (`super::skill_term`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Round1Obs {
    pub score: f64,
    pub features: [f64; 6],
}

/// Mean role score of `candidate` over one match per seed in `seeds`.
///
/// The matches run in parallel on the current rayon pool (so it nests
/// inside a parallel loop over candidates: rayon work-steals, no pool is
/// created); the result is independent of the thread count.
///
/// # Panics
///
/// Panics if `seeds` is empty, the pool is empty, or the table size is
/// unsupported.
pub fn evaluate(
    candidate: &Arc<dyn Strategy>,
    table: &TableSpec,
    opponents: Opponents<'_>,
    seeds: &[u64],
) -> Score {
    evaluate_inner(candidate, table, opponents, seeds, false).0
}

/// `evaluate`, also returning one `Round1Obs` per match. The matches
/// played, and so the `Score`, are exactly those of `evaluate`.
///
/// # Panics
///
/// As `evaluate`.
pub fn evaluate_with_round1(
    candidate: &Arc<dyn Strategy>,
    table: &TableSpec,
    opponents: Opponents<'_>,
    seeds: &[u64],
) -> (Score, Vec<Round1Obs>) {
    evaluate_inner(candidate, table, opponents, seeds, true)
}

/// What one match says about the candidate, before the cross-match sums.
struct MatchOutcome {
    /// Mean role score over the match's rounds.
    mean: f64,
    /// The candidate's place (index into the table's roles) in each round.
    places: Vec<usize>,
    round1: Option<Round1Obs>,
}

fn evaluate_inner(
    candidate: &Arc<dyn Strategy>,
    table: &TableSpec,
    opponents: Opponents<'_>,
    seeds: &[u64],
    record_round1: bool,
) -> (Score, Vec<Round1Obs>) {
    assert!(!seeds.is_empty(), "evaluation needs at least one match");
    let players = usize::from(table.player_count);
    let roles = roles_for_player_count(table.player_count).expect("supported table size");
    let options = RunOptions {
        record_deal_features: record_round1,
        ..RunOptions::default()
    };
    // Matches are independent (everything about one is a function of its
    // seed), so they run in parallel; the order-preserving collect and the
    // sequential sums below keep the result bit-identical to a serial loop.
    let outcomes: Vec<MatchOutcome> = seeds
        .par_iter()
        .enumerate()
        .map(|(index, &seed)| {
            let seat = index % players;
            let mut pick = Xoshiro256PlusPlus::seed_from_u64(mix(seed ^ 0x5EA7));
            let strategies: Vec<Arc<dyn Strategy>> = (0..players)
                .map(|s| {
                    if s == seat {
                        return candidate.clone();
                    }
                    match opponents {
                        Opponents::Only(only) => only.clone(),
                        Opponents::Mixed(pool) => pool[pick.random_range(0..pool.len())].clone(),
                    }
                })
                .collect();
            let result = run_match_with(
                &MatchConfig {
                    player_count: table.player_count,
                    deck_variant: table.deck_variant,
                    duplicate_rule: table.duplicate_rule,
                    rounds: table.rounds,
                    seed,
                    pass_rule: table.pass_rule,
                    exchange_rule: table.exchange_rule,
                },
                &strategies,
                &options,
            );
            let round1 = result
                .first_hand_features
                .as_ref()
                .map(|features| Round1Obs {
                    score: role_score(result.role_history[0][seat], table.player_count),
                    features: HandFeatures::as_vector(&features[seat]),
                });
            let mut total = 0.0;
            let mut places = Vec::with_capacity(result.role_history.len());
            for round in &result.role_history {
                total += role_score(round[seat], table.player_count);
                places.push(
                    roles
                        .iter()
                        .position(|&r| r == round[seat])
                        .expect("role belongs to this table"),
                );
            }
            #[allow(clippy::cast_precision_loss)]
            let mean = total / result.role_history.len() as f64;
            MatchOutcome {
                mean,
                places,
                round1,
            }
        })
        .collect();
    let mut placements = vec![0u64; players];
    let mut per_match = Vec::with_capacity(outcomes.len());
    let mut round1 = Vec::new();
    for outcome in outcomes {
        for place in outcome.places {
            placements[place] += 1;
        }
        per_match.push(outcome.mean);
        round1.extend(outcome.round1);
    }
    #[allow(clippy::cast_precision_loss)]
    let count = per_match.len() as f64;
    let mean = per_match.iter().sum::<f64>() / count;
    let std_error = if per_match.len() < 2 {
        0.0
    } else {
        let variance = per_match.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / (count - 1.0);
        (variance / count).sqrt()
    };
    let score = Score {
        mean,
        std_error,
        matches: per_match.len(),
        placements,
    };
    (score, round1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LowestLegal, RandomLegal};

    const TABLE: TableSpec = TableSpec {
        player_count: 4,
        deck_variant: DeckVariant::Single,
        duplicate_rule: DuplicateRule::FirstDealtWins,
        rounds: 4,
        pass_rule: PassRule::Final,
        exchange_rule: ExchangeRule::Forced,
    };

    fn seeds(count: u64) -> Vec<u64> {
        (0..count).map(|i| match_seed(7, 0, i)).collect()
    }

    #[test]
    fn role_scores_run_from_plus_one_to_minus_one_at_every_table_size() {
        for players in 3..=6 {
            let roles = roles_for_player_count(players).unwrap();
            let scores: Vec<f64> = roles.iter().map(|&r| role_score(r, players)).collect();
            assert!((scores[0] - 1.0).abs() < 1e-12, "{players}: {scores:?}");
            assert!(
                (scores[roles.len() - 1] + 1.0).abs() < 1e-12,
                "{players}: {scores:?}"
            );
            assert!(
                scores.windows(2).all(|w| w[0] > w[1]),
                "strictly decreasing: {scores:?}"
            );
        }
    }

    #[test]
    fn match_seeds_are_stable_and_differ_by_stream_and_index() {
        assert_eq!(match_seed(1, 2, 3), match_seed(1, 2, 3));
        let all = [
            match_seed(1, 2, 3),
            match_seed(1, 2, 4),
            match_seed(1, 3, 3),
            match_seed(2, 2, 3),
        ];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn evaluation_is_deterministic() {
        let candidate: Arc<dyn Strategy> = Arc::new(RandomLegal);
        let pool: Vec<Arc<dyn Strategy>> = vec![Arc::new(LowestLegal), Arc::new(RandomLegal)];
        let run = || evaluate(&candidate, &TABLE, Opponents::Mixed(&pool), &seeds(20));
        assert_eq!(run(), run());
    }

    #[test]
    fn placements_account_for_every_round_and_the_mean_matches_them() {
        let candidate: Arc<dyn Strategy> = Arc::new(LowestLegal);
        let opponent: Arc<dyn Strategy> = Arc::new(RandomLegal);
        let score = evaluate(&candidate, &TABLE, Opponents::Only(&opponent), &seeds(30));
        assert_eq!(score.matches, 30);
        assert_eq!(score.placements.len(), 4);
        assert_eq!(score.placements.iter().sum::<u64>(), 30 * 4);
        // Rebuild the mean from the placement counts.
        let total: f64 = score
            .placements
            .iter()
            .enumerate()
            .map(|(place, &n)| {
                f64::from(u32::try_from(n).unwrap())
                    * (1.0 - 2.0 * f64::from(u32::try_from(place).unwrap()) / 3.0)
            })
            .sum();
        assert!(
            (total / 120.0 - score.mean).abs() < 1e-9,
            "{} vs {total}",
            score.mean
        );
        assert!(score.std_error > 0.0);
    }

    #[test]
    fn a_sensible_player_beats_random_and_mirrors_itself() {
        let lowest: Arc<dyn Strategy> = Arc::new(LowestLegal);
        let random: Arc<dyn Strategy> = Arc::new(RandomLegal);
        let many = seeds(150);
        let versus_random = evaluate(&lowest, &TABLE, Opponents::Only(&random), &many);
        assert!(versus_random.mean > 0.4, "{versus_random:?}");
        let mirror = evaluate(&lowest, &TABLE, Opponents::Only(&lowest), &many);
        assert!(
            mirror.mean.abs() < 0.2,
            "a player against copies of itself is about even: {mirror:?}"
        );
        let reverse = evaluate(&random, &TABLE, Opponents::Only(&lowest), &many);
        assert!(reverse.mean < -0.4, "{reverse:?}");
    }

    #[test]
    #[allow(clippy::float_cmp)] // the same deals give bit-identical features
    fn recording_round_one_changes_neither_the_matches_nor_the_score() {
        let candidate: Arc<dyn Strategy> = Arc::new(RandomLegal);
        let pool: Vec<Arc<dyn Strategy>> = vec![Arc::new(LowestLegal), Arc::new(RandomLegal)];
        let many = seeds(25);
        let plain = evaluate(&candidate, &TABLE, Opponents::Mixed(&pool), &many);
        let (score, round1) =
            evaluate_with_round1(&candidate, &TABLE, Opponents::Mixed(&pool), &many);
        assert_eq!(plain, score);
        assert_eq!(round1.len(), 25);
        assert!(round1.iter().all(|o| o.score.abs() <= 1.0));
        // The deal does not depend on who plays: another candidate sees the
        // same round-1 features in every match.
        let other: Arc<dyn Strategy> = Arc::new(LowestLegal);
        let (_, again) = evaluate_with_round1(&other, &TABLE, Opponents::Mixed(&pool), &many);
        assert!(round1
            .iter()
            .zip(&again)
            .all(|(a, b)| a.features == b.features));
    }

    #[test]
    fn a_single_match_has_no_standard_error() {
        let candidate: Arc<dyn Strategy> = Arc::new(LowestLegal);
        let score = evaluate(&candidate, &TABLE, Opponents::Only(&candidate), &seeds(1));
        assert!(score.std_error.abs() < f64::EPSILON);
    }

    /// The original serial implementation, kept as the oracle for the
    /// parallel one.
    #[allow(clippy::cast_precision_loss)]
    fn evaluate_serially(
        candidate: &Arc<dyn Strategy>,
        table: &TableSpec,
        opponents: Opponents<'_>,
        seeds: &[u64],
    ) -> Score {
        let players = usize::from(table.player_count);
        let roles = roles_for_player_count(table.player_count).unwrap();
        let mut placements = vec![0u64; players];
        let mut per_match = Vec::new();
        for (index, &seed) in seeds.iter().enumerate() {
            let seat = index % players;
            let mut pick = Xoshiro256PlusPlus::seed_from_u64(mix(seed ^ 0x5EA7));
            let strategies: Vec<Arc<dyn Strategy>> = (0..players)
                .map(|s| {
                    if s == seat {
                        return candidate.clone();
                    }
                    match opponents {
                        Opponents::Only(only) => only.clone(),
                        Opponents::Mixed(pool) => pool[pick.random_range(0..pool.len())].clone(),
                    }
                })
                .collect();
            let result = run_match_with(
                &MatchConfig {
                    player_count: table.player_count,
                    deck_variant: table.deck_variant,
                    duplicate_rule: table.duplicate_rule,
                    rounds: table.rounds,
                    seed,
                    pass_rule: table.pass_rule,
                    exchange_rule: table.exchange_rule,
                },
                &strategies,
                &RunOptions::default(),
            );
            let mut total = 0.0;
            for round in &result.role_history {
                total += role_score(round[seat], table.player_count);
                placements[roles.iter().position(|&r| r == round[seat]).unwrap()] += 1;
            }
            per_match.push(total / result.role_history.len() as f64);
        }
        let count = per_match.len() as f64;
        let mean = per_match.iter().sum::<f64>() / count;
        let std_error = if per_match.len() < 2 {
            0.0
        } else {
            let variance =
                per_match.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / (count - 1.0);
            (variance / count).sqrt()
        };
        Score {
            mean,
            std_error,
            matches: per_match.len(),
            placements,
        }
    }

    #[test]
    fn parallel_evaluation_equals_the_serial_oracle_on_any_thread_count() {
        let candidate: Arc<dyn Strategy> = Arc::new(RandomLegal);
        let pool: Vec<Arc<dyn Strategy>> = vec![Arc::new(LowestLegal), Arc::new(RandomLegal)];
        let many = seeds(37);
        let expected_mixed = evaluate_serially(&candidate, &TABLE, Opponents::Mixed(&pool), &many);
        let expected_only = evaluate_serially(&candidate, &TABLE, Opponents::Only(&pool[0]), &many);
        for threads in [1, 3, 8] {
            let rayon_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            rayon_pool.install(|| {
                let mixed = evaluate(&candidate, &TABLE, Opponents::Mixed(&pool), &many);
                let only = evaluate(&candidate, &TABLE, Opponents::Only(&pool[0]), &many);
                // Full structs, f64 fields bit for bit (`Score: PartialEq`).
                assert_eq!(mixed, expected_mixed, "{threads} threads");
                assert_eq!(only, expected_only, "{threads} threads");
                assert_eq!(mixed.mean.to_bits(), expected_mixed.mean.to_bits());
                assert_eq!(
                    mixed.std_error.to_bits(),
                    expected_mixed.std_error.to_bits()
                );
            });
        }
    }

    #[test]
    #[should_panic(expected = "at least one match")]
    fn evaluating_zero_matches_panics() {
        let candidate: Arc<dyn Strategy> = Arc::new(LowestLegal);
        let _ = evaluate(&candidate, &TABLE, Opponents::Only(&candidate), &[]);
    }
}
