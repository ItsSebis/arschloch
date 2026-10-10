#![allow(clippy::cast_precision_loss)] // test counts are tiny
use super::*;
use crate::{run_batch, GreedyHighest, LowestLegal, MatchConfig, RandomLegal, Strategy};
use std::sync::Arc;

fn result(names: &[&str], rounds: Vec<Vec<Role>>) -> MatchResult {
    let n = names.len();
    MatchResult {
        player_count: u8::try_from(n).unwrap(),
        strategy_names: names.iter().map(|s| (*s).to_string()).collect(),
        role_history: rounds,
        trick_count: 0,
        pass_counts: vec![10; n],
        voluntary_pass_counts: vec![5; n],
        first_hand_features: None,
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

/// Deterministic uniform generator for synthetic data.
struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        #[allow(clippy::cast_precision_loss)]
        let v = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        v
    }
}

#[test]
fn three_player_average_rank_and_distribution() {
    use Role::{Arschloch, Dorftrottel, President};
    let results = vec![
        result(
            &["A", "B", "C"],
            vec![
                vec![President, Dorftrottel, Arschloch],
                vec![President, Arschloch, Dorftrottel],
            ],
        ),
        result(
            &["A", "B", "C"],
            vec![vec![Arschloch, President, Dorftrottel]],
        ),
    ];
    let stats = aggregate_extended(&results);
    let a = &stats.by_strategy["A"];
    // Places 1,1,3 over 3 rounds: mean 5/3; per-match means 1 and 3.
    assert!(close(a.avg_rank.value, 5.0 / 3.0));
    assert_eq!(a.rounds, 3);
    assert_eq!(a.avg_rank.n, 2);
    assert!(close(a.avg_rank.std_error, standard_error(&[1.0, 3.0])));
    assert!(close(a.rank_distribution[0], 200.0 / 3.0));
    assert!(close(a.rank_distribution[1], 0.0));
    assert!(close(a.rank_distribution[2], 100.0 / 3.0));
    // Scores +1,+1,-1 -> 1/3.
    assert!(close(a.mean_role_score.value, 1.0 / 3.0));
    assert_eq!(stats.by_seat.len(), 3);
    assert!(close(stats.by_seat[0].avg_rank.value, 5.0 / 3.0));
}

#[test]
fn four_player_invariants() {
    use Role::{Arschloch, President, ViceArschloch, Vize};
    let roles = roles_for_player_count(4).unwrap();
    assert_eq!(roles[0], President);
    assert_eq!(roles[3], Arschloch);
    let r = vec![
        result(
            &["A", "B", "A", "C"],
            vec![
                vec![President, Vize, ViceArschloch, Arschloch],
                vec![Arschloch, President, Vize, ViceArschloch],
            ],
        ),
        result(
            &["B", "A", "C", "A"],
            vec![vec![Vize, President, Arschloch, ViceArschloch]],
        ),
    ];
    let stats = aggregate_extended(&r);
    for s in stats.by_strategy.values() {
        let total: f64 = s.rank_distribution.iter().sum();
        assert!(close(total, 100.0));
        let n = s.rank_distribution.len();
        let from_dist: f64 = s
            .rank_distribution
            .iter()
            .enumerate()
            .map(|(p, pct)| pct / 100.0 * (1.0 - 2.0 * p as f64 / (n - 1) as f64))
            .sum();
        assert!(close(from_dist, s.mean_role_score.value));
        let ratings = s.strength_rating.unwrap();
        assert!(ratings.value.is_finite());
    }
    let mean: f64 = stats
        .by_strategy
        .values()
        .map(|s| s.strength_rating.unwrap().value)
        .sum::<f64>()
        / 3.0;
    assert!(mean.abs() < 1e-9);
    // A holds two seats per match: 3 series per... 2 matches x 2 seats.
    assert_eq!(stats.by_strategy["A"].avg_rank.n, 4);
}

#[test]
fn single_strategy_has_no_rating() {
    use Role::{Arschloch, Dorftrottel, President};
    let r = vec![result(
        &["A", "A", "A"],
        vec![vec![President, Dorftrottel, Arschloch]],
    )];
    assert!(aggregate_extended(&r).by_strategy["A"]
        .strength_rating
        .is_none());
}

#[test]
fn bradley_terry_recovers_known_strengths() {
    let strengths = [8.0, 4.0, 2.0, 1.0, 0.5];
    let names = ["A", "B", "C", "D", "E"];
    let mut rng = Lcg(42);
    let mut results = Vec::new();
    for m in 0..20_000 {
        let size = 3 + m % 3;
        // random subset of `size` strategies, seat order shuffled
        let mut pool: Vec<usize> = (0..5).collect();
        let mut seats = Vec::new();
        while seats.len() < size {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let i = (rng.unit() * pool.len() as f64) as usize;
            seats.push(pool.swap_remove(i));
        }
        // Plackett-Luce order
        let mut remaining = seats.clone();
        let mut order = Vec::new();
        while !remaining.is_empty() {
            let total: f64 = remaining.iter().map(|&s| strengths[s]).sum();
            let mut x = rng.unit() * total;
            let mut pick = remaining.len() - 1;
            for (i, &s) in remaining.iter().enumerate() {
                if x < strengths[s] {
                    pick = i;
                    break;
                }
                x -= strengths[s];
            }
            order.push(remaining.remove(pick));
        }
        let table = roles_for_player_count(u8::try_from(size).unwrap()).unwrap();
        let round: Vec<Role> = seats
            .iter()
            .map(|s| table[order.iter().position(|o| o == s).unwrap()])
            .collect();
        let seat_names: Vec<&str> = seats.iter().map(|&s| names[s]).collect();
        results.push(result(&seat_names, vec![round]));
    }
    let stats = aggregate_extended_with(
        &results,
        &ExtendedOptions {
            bootstrap_resamples: 0,
            bootstrap_seed: 0,
        },
    );
    let elo = |n: &str| stats.by_strategy[n].strength_rating.unwrap().value;
    let expected = |s: f64| 400.0 / std::f64::consts::LN_10 * s.ln();
    let mean_expected: f64 = strengths.iter().map(|s| expected(*s)).sum::<f64>() / 5.0;
    for (n, s) in names.iter().zip(strengths) {
        assert!(
            (elo(n) - (expected(s) - mean_expected)).abs() < 25.0,
            "{n}: {} vs {}",
            elo(n),
            expected(s) - mean_expected
        );
        assert!(close(
            stats.by_strategy[*n].strength_rating.unwrap().std_error,
            0.0
        ));
    }
    assert!(
        elo("A") > elo("B") && elo("B") > elo("C") && elo("C") > elo("D") && elo("D") > elo("E")
    );
}

#[test]
fn separation_does_not_blow_up() {
    use Role::{Arschloch, Dorftrottel, President};
    let r: Vec<MatchResult> = (0..30)
        .map(|_| {
            result(
                &["A", "B", "C"],
                vec![vec![President, Dorftrottel, Arschloch]],
            )
        })
        .collect();
    let stats = aggregate_extended(&r);
    let v: Vec<f64> = ["A", "B", "C"]
        .iter()
        .map(|n| stats.by_strategy[*n].strength_rating.unwrap().value)
        .collect();
    assert!(v.iter().all(|x| x.is_finite()));
    assert!(v[0] > v[1] && v[1] > v[2]);
}

#[test]
fn bootstrap_is_deterministic_and_positive() {
    use Role::{Arschloch, Dorftrottel, President};
    let r: Vec<MatchResult> = (0..40)
        .map(|i| {
            if i % 3 == 0 {
                result(
                    &["A", "B", "C"],
                    vec![vec![Dorftrottel, President, Arschloch]],
                )
            } else {
                result(
                    &["A", "B", "C"],
                    vec![vec![President, Dorftrottel, Arschloch]],
                )
            }
        })
        .collect();
    let a = aggregate_extended(&r);
    let b = aggregate_extended(&r);
    let se = |s: &ExtendedStatistics| s.by_strategy["A"].strength_rating.unwrap().std_error;
    assert!(se(&a) > 0.0);
    assert!(close(se(&a), se(&b)));
}

#[test]
fn intervals_are_reported() {
    use Role::{Arschloch, Dorftrottel, President};
    let r = vec![result(
        &["A", "B", "C"],
        vec![
            vec![President, Dorftrottel, Arschloch],
            vec![President, Arschloch, Dorftrottel],
        ],
    )];
    let stats = aggregate_extended(&r);
    let i = stats.role_retention_intervals["A"][&President];
    let (low, high) = wilson_interval(1, 1).unwrap();
    assert!(close(i.low, low) && close(i.high, high));
    let p = stats.voluntary_pass_rate_intervals["A"];
    let (low, high) = wilson_interval(5, 10).unwrap();
    assert!(close(p.low, low) && close(p.high, high));
}

#[test]
fn real_batch_puts_random_last() {
    let strategies: Vec<Arc<dyn Strategy>> = vec![
        Arc::new(LowestLegal),
        Arc::new(GreedyHighest),
        Arc::new(RandomLegal),
    ];
    let configs: Vec<MatchConfig> = (0..60)
        .map(|seed| MatchConfig {
            player_count: 3,
            deck_variant: engine::DeckVariant::Single,
            duplicate_rule: engine::DuplicateRule::FirstDealtWins,
            rounds: 5,
            seed,
            pass_rule: engine::PassRule::default(),
            exchange_rule: engine::ExchangeRule::default(),
        })
        .collect();
    let results = run_batch(&configs, &strategies);
    let stats = aggregate_extended(&results);
    let rank = |n: &str| stats.by_strategy[n].avg_rank.value;
    assert!(rank("RandomLegal") > rank("LowestLegal"));
    assert!(rank("RandomLegal") > rank("GreedyHighest"));
    let elo = |n: &str| stats.by_strategy[n].strength_rating.unwrap().value;
    assert!(elo("RandomLegal") < elo("LowestLegal"));
    assert_eq!(stats.by_seat.len(), 3);
}
