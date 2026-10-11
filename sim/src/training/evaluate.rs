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

use super::fitness::{FitnessKind, Moments};
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

impl TableSpec {
    /// A short name of the setting, such as `4p-single`.
    #[must_use]
    pub fn label(&self) -> String {
        let deck = match self.deck_variant {
            DeckVariant::Single => "single",
            DeckVariant::Double => "double",
        };
        format!("{}p-{deck}", self.player_count)
    }
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
    /// Over several table sizes the counts of the same place index are
    /// added (the vector is as long as the largest table); `by_setting`
    /// has the per-table numbers.
    pub placements: Vec<u64>,
    /// The standard deviation of the per-round role score over every round
    /// of every match (not of the match means: that is `std_error`).
    pub spread: f64,
    /// The configured fitness (`kind`) of this evaluation. For
    /// `FitnessKind::Mean` it is `mean`, bit for bit.
    pub fitness: f64,
    pub kind: FitnessKind,
    /// The sums behind `spread` and `fitness`, so evaluations of one
    /// genome can be pooled (`Moments::pooled`).
    pub moments: Moments,
    /// One entry per setting, in setting order; empty when the evaluation
    /// used a single table.
    pub by_setting: Vec<SettingScore>,
}

/// One setting's share of an evaluation over several tables.
#[derive(Debug, Clone, PartialEq)]
pub struct SettingScore {
    /// `TableSpec::label`, such as `4p-single`.
    pub label: String,
    pub matches: usize,
    /// Mean of the setting's match means.
    pub mean: f64,
    pub spread: f64,
    pub fitness: f64,
    /// Places as in `Score::placements`, for this table only.
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
    evaluate_with(
        candidate,
        std::slice::from_ref(table),
        opponents,
        seeds,
        FitnessKind::Mean,
        0,
    )
}

/// `evaluate` over several settings and with a fitness variant. Match `i`
/// of `seeds` is played at `tables[(first_index + i) % tables.len()]`, in
/// seat `((first_index + i) / tables.len()) % players` of that table (each
/// setting rotates through its own seats; with one table this is
/// `(first_index + i) % players`), so every candidate
/// evaluated on the same seeds meets the same table, seat and opponents
/// (common random numbers). `first_index` lets a second batch of matches
/// continue the cycle of the first. With one table, `FitnessKind::Mean` and
/// `first_index` 0 this is exactly `evaluate`.
///
/// # Panics
///
/// As `evaluate`, and if `tables` is empty.
pub fn evaluate_with(
    candidate: &Arc<dyn Strategy>,
    tables: &[TableSpec],
    opponents: Opponents<'_>,
    seeds: &[u64],
    kind: FitnessKind,
    first_index: usize,
) -> Score {
    evaluate_inner(
        candidate,
        tables,
        opponents,
        seeds,
        kind,
        first_index,
        false,
    )
    .0
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
    evaluate_inner(
        candidate,
        std::slice::from_ref(table),
        opponents,
        seeds,
        FitnessKind::Mean,
        0,
        true,
    )
}

/// What one match says about the candidate, before the cross-match sums.
struct MatchOutcome {
    /// Which table of the cycle the match was played at (set by the caller).
    setting: usize,
    /// Mean role score over the match's rounds.
    mean: f64,
    /// The match's own sums of the per-round role scores.
    moments: Moments,
    /// The candidate's place (index into the table's roles) in each round.
    places: Vec<usize>,
    round1: Option<Round1Obs>,
}

/// The cross-match sums of one setting (or of all of them).
struct Totals {
    matches: usize,
    /// Sum of the match means.
    sum_of_means: f64,
    moments: Moments,
    placements: Vec<u64>,
}

impl Totals {
    fn new(players: usize) -> Self {
        Self {
            matches: 0,
            sum_of_means: 0.0,
            moments: Moments::default(),
            placements: vec![0; players],
        }
    }

    fn add(&mut self, outcome: &MatchOutcome) {
        self.matches += 1;
        self.sum_of_means += outcome.mean;
        self.moments = self.moments.pooled(&outcome.moments);
        for &place in &outcome.places {
            self.placements[place] += 1;
        }
    }
}

fn evaluate_inner(
    candidate: &Arc<dyn Strategy>,
    tables: &[TableSpec],
    opponents: Opponents<'_>,
    seeds: &[u64],
    kind: FitnessKind,
    first_index: usize,
    record_round1: bool,
) -> (Score, Vec<Round1Obs>) {
    assert!(!seeds.is_empty(), "evaluation needs at least one match");
    assert!(!tables.is_empty(), "evaluation needs at least one table");
    let options = RunOptions {
        record_deal_features: record_round1,
        ..RunOptions::default()
    };
    // Matches are independent (everything about one is a function of its
    // seed), so they run in parallel; the order-preserving collect and the
    // sequential sums in `summarise` keep the result bit-identical to a
    // serial loop.
    let outcomes: Vec<MatchOutcome> = seeds
        .par_iter()
        .enumerate()
        .map(|(index, &seed)| {
            let index = first_index + index;
            let setting = index % tables.len();
            let mut outcome = play_match(
                candidate,
                &tables[setting],
                opponents,
                kind,
                &options,
                // The setting cycles fastest, so the seat rotates once per
                // pass over the settings: every setting visits all its seats.
                index / tables.len(),
                seed,
            );
            outcome.setting = setting;
            outcome
        })
        .collect();
    summarise(outcomes, tables, kind)
}

/// Plays a match at `table` with the candidate in seat `rotation mod
/// players` and the deals of `seed`.
fn play_match(
    candidate: &Arc<dyn Strategy>,
    table: &TableSpec,
    opponents: Opponents<'_>,
    kind: FitnessKind,
    options: &RunOptions,
    rotation: usize,
    seed: u64,
) -> MatchOutcome {
    let players = usize::from(table.player_count);
    let roles = roles_for_player_count(table.player_count).expect("supported table size");
    let seat = rotation % players;
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
        options,
    );
    let round1 = result
        .first_hand_features
        .as_ref()
        .map(|features| Round1Obs {
            score: role_score(result.role_history[0][seat], table.player_count),
            features: HandFeatures::as_vector(&features[seat]),
        });
    let mut total = 0.0;
    let mut moments = Moments::default();
    let mut places = Vec::with_capacity(result.role_history.len());
    for round in &result.role_history {
        let s = role_score(round[seat], table.player_count);
        total += s;
        moments.add_round(s, kind.squash(s));
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
        setting: 0,
        mean,
        moments,
        places,
        round1,
    }
}

/// The cross-match statistics of the outcomes, summed in match order.
fn summarise(
    outcomes: Vec<MatchOutcome>,
    tables: &[TableSpec],
    kind: FitnessKind,
) -> (Score, Vec<Round1Obs>) {
    let largest = tables
        .iter()
        .map(|t| usize::from(t.player_count))
        .max()
        .expect("at least one table");
    let mut all = Totals::new(largest);
    let mut per_setting: Vec<Totals> = tables
        .iter()
        .map(|t| Totals::new(usize::from(t.player_count)))
        .collect();
    let mut per_match = Vec::with_capacity(outcomes.len());
    let mut round1 = Vec::new();
    for outcome in outcomes {
        all.add(&outcome);
        per_setting[outcome.setting].add(&outcome);
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
    let fitness = if kind.is_mean() {
        mean
    } else {
        all.moments.fitness(&kind)
    };
    let by_setting = if tables.len() > 1 {
        tables
            .iter()
            .zip(per_setting)
            .filter(|(_, totals)| totals.matches > 0)
            .map(|(table, totals)| {
                #[allow(clippy::cast_precision_loss)]
                let matches = totals.matches as f64;
                SettingScore {
                    label: table.label(),
                    matches: totals.matches,
                    mean: totals.sum_of_means / matches,
                    spread: totals.moments.spread(),
                    fitness: totals.moments.fitness(&kind),
                    placements: totals.placements,
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    let score = Score {
        mean,
        std_error,
        matches: per_match.len(),
        placements: all.placements,
        spread: all.moments.spread(),
        fitness,
        kind,
        moments: all.moments,
        by_setting,
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
        let mut moments = Moments::default();
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
            let mut match_moments = Moments::default();
            for round in &result.role_history {
                let s = role_score(round[seat], table.player_count);
                total += s;
                match_moments.add_round(s, 0.0);
                placements[roles.iter().position(|&r| r == round[seat]).unwrap()] += 1;
            }
            moments = moments.pooled(&match_moments);
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
            spread: moments.spread(),
            fitness: mean,
            kind: FitnessKind::Mean,
            moments,
            by_setting: Vec::new(),
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

    /// The per-round role scores of an evaluation, rebuilt from its
    /// placements: `(score, how many rounds)` per place of a table of
    /// `players`.
    #[allow(clippy::cast_precision_loss)]
    fn scores_from_placements(placements: &[u64], players: u8) -> Vec<(f64, f64)> {
        let roles = roles_for_player_count(players).unwrap();
        placements
            .iter()
            .enumerate()
            .map(|(place, &n)| (role_score(roles[place], players), n as f64))
            .collect()
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn each_fitness_variant_equals_its_formula_on_the_rebuilt_round_scores() {
        let candidate: Arc<dyn Strategy> = Arc::new(RandomLegal);
        let pool: Vec<Arc<dyn Strategy>> = vec![Arc::new(LowestLegal), Arc::new(RandomLegal)];
        let many = seeds(40);
        let run = |kind| {
            evaluate_with(
                &candidate,
                std::slice::from_ref(&TABLE),
                Opponents::Mixed(&pool),
                &many,
                kind,
                0,
            )
        };
        let plain = run(FitnessKind::Mean);
        assert_eq!(
            plain,
            evaluate(&candidate, &TABLE, Opponents::Mixed(&pool), &many)
        );
        assert_eq!(plain.fitness.to_bits(), plain.mean.to_bits());
        let rounds = scores_from_placements(&plain.placements, 4);
        let n: f64 = rounds.iter().map(|r| r.1).sum();
        assert!((n - 160.0).abs() < 1e-9);
        let mean: f64 = rounds.iter().map(|&(s, c)| s * c).sum::<f64>() / n;
        let variance: f64 = rounds
            .iter()
            .map(|&(s, c)| (s - mean).powi(2) * c)
            .sum::<f64>()
            / n;
        assert!((plain.mean - mean).abs() < 1e-12);
        assert!((plain.spread - variance.sqrt()).abs() < 1e-12);
        assert!(plain.spread > 0.1, "a random player's rounds differ");

        let risk = run(FitnessKind::Risk { lambda: 0.7 });
        assert!((risk.fitness - (mean - 0.7 * variance.sqrt())).abs() < 1e-12);
        let kappa = 3.0f64;
        let tanh = run(FitnessKind::Tanh { kappa });
        let expected: f64 = rounds
            .iter()
            .map(|&(s, c)| (kappa * s).tanh() / kappa.tanh() * c)
            .sum::<f64>()
            / n;
        assert!((tanh.fitness - expected).abs() < 1e-12);
        // The variant changes neither the matches nor the reported mean,
        // standard error, placements or spread, bit for bit.
        for other in [&risk, &tanh] {
            assert_eq!(other.mean.to_bits(), plain.mean.to_bits());
            assert_eq!(other.std_error.to_bits(), plain.std_error.to_bits());
            assert_eq!(other.placements, plain.placements);
            assert_eq!(other.spread.to_bits(), plain.spread.to_bits());
            assert_eq!(other.matches, plain.matches);
        }
        assert_eq!(risk.kind, FitnessKind::Risk { lambda: 0.7 });
    }

    #[test]
    fn a_lambda_of_zero_is_the_mean_up_to_rounding_and_the_pooled_moments_match_the_parts() {
        let candidate: Arc<dyn Strategy> = Arc::new(RandomLegal);
        let opponent: Arc<dyn Strategy> = Arc::new(LowestLegal);
        let a = evaluate_with(
            &candidate,
            std::slice::from_ref(&TABLE),
            Opponents::Only(&opponent),
            &seeds(12),
            FitnessKind::Risk { lambda: 0.0 },
            0,
        );
        assert!((a.fitness - a.mean).abs() < 1e-12);
        assert_eq!(a.moments.rounds, 48);
        // Evaluating 12 matches in two halves and pooling the moments
        // equals the moments of the whole: the first half is matches 0..6,
        // the second continues the seat cycle at index 6.
        let all = seeds(12);
        let half = |range: std::ops::Range<usize>| {
            evaluate_with(
                &candidate,
                std::slice::from_ref(&TABLE),
                Opponents::Only(&opponent),
                &all[range.clone()],
                FitnessKind::Mean,
                range.start,
            )
        };
        let pooled = half(0..6).moments.pooled(&half(6..12).moments);
        assert_eq!(pooled.rounds, a.moments.rounds);
        assert!((pooled.sum - a.moments.sum).abs() < 1e-12);
        assert!((pooled.sum_sq - a.moments.sum_sq).abs() < 1e-12);
    }

    fn mixed_tables() -> Vec<TableSpec> {
        [
            (3, DeckVariant::Single),
            (4, DeckVariant::Double),
            (6, DeckVariant::Single),
        ]
        .into_iter()
        .map(|(player_count, deck_variant)| TableSpec {
            player_count,
            deck_variant,
            ..TABLE
        })
        .collect()
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn several_tables_cycle_by_match_index_and_the_columns_add_up() {
        let tables = mixed_tables();
        let candidate: Arc<dyn Strategy> = Arc::new(RandomLegal);
        let pool: Vec<Arc<dyn Strategy>> = vec![Arc::new(LowestLegal), Arc::new(RandomLegal)];
        let many = seeds(20);
        let run = |first_index| {
            evaluate_with(
                &candidate,
                &tables,
                Opponents::Mixed(&pool),
                &many,
                FitnessKind::Tanh { kappa: 2.0 },
                first_index,
            )
        };
        let score = run(0);
        let labels: Vec<&str> = score.by_setting.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(labels, ["3p-single", "4p-double", "6p-single"]);
        // Matches 0..20 go to settings 0,1,2,0,1,2,...
        let counts: Vec<usize> = score.by_setting.iter().map(|s| s.matches).collect();
        assert_eq!(counts, [7, 7, 6]);
        assert_eq!(counts.iter().sum::<usize>(), score.matches);
        // Each setting's placements are as long as its table and cover its rounds.
        for (setting, players) in score.by_setting.iter().zip([3usize, 4, 6]) {
            assert_eq!(setting.placements.len(), players);
            assert_eq!(
                setting.placements.iter().sum::<u64>(),
                (setting.matches * 4) as u64
            );
        }
        // The overall placements are the per-place sums, as long as the largest table.
        assert_eq!(score.placements.len(), 6);
        for place in 0..6 {
            let total: u64 = score
                .by_setting
                .iter()
                .filter_map(|s| s.placements.get(place))
                .sum();
            assert_eq!(score.placements[place], total, "place {place}");
        }
        // The mean is the match-weighted mean of the settings' means.
        let weighted: f64 = score
            .by_setting
            .iter()
            .map(|s| s.mean * s.matches as f64)
            .sum::<f64>()
            / score.matches as f64;
        assert!((score.mean - weighted).abs() < 1e-12);
        // So is the tanh fitness (every match has the same number of rounds).
        let weighted: f64 = score
            .by_setting
            .iter()
            .map(|s| s.fitness * s.matches as f64)
            .sum::<f64>()
            / score.matches as f64;
        assert!((score.fitness - weighted).abs() < 1e-12);
        assert_eq!(score, run(0), "deterministic");
        // Continuing the cycle at an offset moves the match counts along.
        let shifted = run(1);
        let counts: Vec<usize> = shifted.by_setting.iter().map(|s| s.matches).collect();
        assert_eq!(counts, [6, 7, 7]);
    }

    #[test]
    fn every_candidate_meets_the_same_settings_and_one_table_is_the_plain_evaluation() {
        let tables = mixed_tables();
        let pool: Vec<Arc<dyn Strategy>> = vec![Arc::new(LowestLegal)];
        let many = seeds(9);
        let a: Arc<dyn Strategy> = Arc::new(RandomLegal);
        let b: Arc<dyn Strategy> = Arc::new(LowestLegal);
        let shape = |candidate: &Arc<dyn Strategy>| {
            evaluate_with(
                candidate,
                &tables,
                Opponents::Mixed(&pool),
                &many,
                FitnessKind::Mean,
                0,
            )
            .by_setting
            .iter()
            .map(|s| (s.label.clone(), s.matches, s.placements.len()))
            .collect::<Vec<_>>()
        };
        assert_eq!(shape(&a), shape(&b));
        let one = evaluate_with(
            &a,
            &tables[..1],
            Opponents::Mixed(&pool),
            &many,
            FitnessKind::Mean,
            0,
        );
        assert!(one.by_setting.is_empty(), "no columns for a single setting");
        assert_eq!(
            one,
            evaluate(&a, &tables[0], Opponents::Mixed(&pool), &many)
        );
    }

    #[test]
    fn every_setting_rotates_through_all_its_seats() {
        // With L settings the candidate's seat is (index / L) % players, so a
        // setting's own matches (every L-th) take seats 0, 1, 2, ... in turn;
        // by_setting[j] is therefore exactly the evaluation of that table
        // alone on every L-th seed with the plain rotation.
        let tables = mixed_tables();
        let candidate: Arc<dyn Strategy> = Arc::new(RandomLegal);
        let pool: Vec<Arc<dyn Strategy>> = vec![Arc::new(LowestLegal), Arc::new(RandomLegal)];
        let many = seeds(36);
        let multi = evaluate_with(
            &candidate,
            &tables,
            Opponents::Mixed(&pool),
            &many,
            FitnessKind::Mean,
            0,
        );
        for (j, table) in tables.iter().enumerate() {
            let own: Vec<u64> = many.iter().copied().skip(j).step_by(tables.len()).collect();
            let alone = evaluate(&candidate, table, Opponents::Mixed(&pool), &own);
            let column = &multi.by_setting[j];
            assert_eq!(column.matches, own.len());
            assert_eq!(column.placements, alone.placements, "{}", column.label);
            assert!((column.mean - alone.mean).abs() < 1e-12);
        }
    }

    #[test]
    fn several_tables_do_not_depend_on_the_thread_count() {
        let tables = mixed_tables();
        let candidate: Arc<dyn Strategy> = Arc::new(RandomLegal);
        let pool: Vec<Arc<dyn Strategy>> = vec![Arc::new(LowestLegal), Arc::new(RandomLegal)];
        let many = seeds(14);
        let run = || {
            evaluate_with(
                &candidate,
                &tables,
                Opponents::Mixed(&pool),
                &many,
                FitnessKind::Risk { lambda: 0.5 },
                3,
            )
        };
        let expected = run();
        for threads in [1, 4] {
            let rayon_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            assert_eq!(rayon_pool.install(run), expected, "{threads} threads");
        }
    }
}
