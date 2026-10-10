//! Duplicate-deal skill score and the cheap estimator, end to end.

use std::sync::Arc;

use engine::{DeckVariant, DuplicateRule, Role};
use sim::duplicate::{run_duplicate_batch, try_run_duplicate_batch};
use sim::skill::{compare, estimator_report, skill_report};
use sim::{
    CardCounter, GreedyHighest, HandFeatures, HoldBackPairs, LowestLegal, MatchConfig, MatchResult,
    RandomLegal, Strategy,
};

fn configs(n: u64, rounds: usize) -> Vec<MatchConfig> {
    (0..n)
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

/// `LowestLegal` under another name, to get distinct but equal players.
struct Twin(&'static str);

impl Strategy for Twin {
    fn name(&self) -> &'static str {
        self.0
    }
    fn choose_play(
        &self,
        legal_moves: &[engine::Move],
        duplicate_rule: DuplicateRule,
        context: &sim::TurnContext<'_>,
        rng: &mut dyn rand::Rng,
    ) -> engine::Move {
        LowestLegal.choose_play(legal_moves, duplicate_rule, context, rng)
    }
    fn choose_exchange_cards(
        &self,
        hand: &[engine::Card],
        count: usize,
        duplicate_rule: DuplicateRule,
        rng: &mut dyn rand::Rng,
    ) -> Vec<engine::Card> {
        LowestLegal.choose_exchange_cards(hand, count, duplicate_rule, rng)
    }
}

#[test]
fn identical_strategies_have_zero_skill() {
    let table: Vec<Arc<dyn Strategy>> = vec![
        Arc::new(Twin("A")),
        Arc::new(Twin("B")),
        Arc::new(Twin("C")),
        Arc::new(Twin("D")),
    ];
    let report = skill_report(&run_duplicate_batch(&configs(400, 3), &table));
    assert_eq!(report.group_count, 100);
    assert_eq!(report.k, 4);
    let sum: f64 = report
        .strategies
        .iter()
        .map(|s| s.skill_score_duplicate_round1.value)
        .sum();
    assert!(sum.abs() < 1e-9, "round-1 skill sums to zero: {sum}");
    for s in &report.strategies {
        let e = s.skill_score_duplicate_round1;
        assert!(
            e.value.abs() < 4.0 * e.std_error + 1e-9,
            "{}: {e:?}",
            s.name
        );
    }

    // The same strategy four times under one name pools to exactly zero.
    let pooled: Vec<Arc<dyn Strategy>> = (0..4).map(|_| Arc::new(LowestLegal) as _).collect();
    let report = skill_report(&run_duplicate_batch(&configs(40, 3), &pooled));
    assert_eq!(report.strategies.len(), 1);
    assert!(report.strategies[0].skill_score_duplicate.value.abs() < 1e-12);
}

/// Measured (2000 groups, 1 round): the deal explains up to about half of
/// a strategy's single-match variance in this game, so M tops out near 2;
/// with duplicated names (two seats pooled) rotations repeat and M is
/// about 1 or below. The asserts keep clear margins around those values.
#[test]
fn strong_beats_weak_with_smaller_error_than_plain() {
    let table: Vec<Arc<dyn Strategy>> = vec![
        Arc::new(LowestLegal),
        Arc::new(GreedyHighest),
        Arc::new(HoldBackPairs),
        Arc::new(CardCounter),
    ];
    let report = skill_report(&run_duplicate_batch(&configs(1200, 1), &table));
    let by = |name: &str| report.strategies.iter().find(|s| s.name == name).unwrap();
    let (strong, weak) = (by("LowestLegal"), by("GreedyHighest"));
    assert!(strong.skill_score_duplicate.value > weak.skill_score_duplicate.value);
    assert!(
        strong.skill_score_duplicate_round1.std_error
            < strong.plain_mean_role_score_round1.std_error,
        "{strong:?}"
    );
    assert!(strong.variance_reduction_round1 > 1.3, "{strong:?}");
    assert!(strong.luck_share_round1 > 0.2);
    let cc = by("CardCounter");
    assert!(cc.variance_reduction_round1 > 1.3, "{cc:?}");
}

fn synthetic(n: usize, skill: &[f64]) -> Vec<MatchResult> {
    // 4 seats, strategy i has true skill skill[i]. A seat's round-1 score is
    // skill + 0.3 * luck, where luck is carried by mean_strength; the rank
    // order of the scores gives the roles.
    let mut state = 12345_u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        f64::from(u32::try_from(state >> 40).unwrap()) / f64::from(1u32 << 24)
    };
    (0..n)
        .map(|_| {
            let luck: Vec<f64> = (0..4).map(|_| next() * 2.0 - 1.0).collect();
            let mut scored: Vec<(usize, f64)> =
                (0..4).map(|s| (s, skill[s] + 1.2 * luck[s])).collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1));
            let mut roles = vec![Role::President; 4];
            let order = engine::roles_for_player_count(4).unwrap();
            for (place, (seat, _)) in scored.iter().enumerate() {
                roles[*seat] = order[place];
            }
            MatchResult {
                player_count: 4,
                strategy_names: ["s0", "s1", "s2", "s3"].map(String::from).to_vec(),
                role_history: vec![roles],
                trick_count: 0,
                pass_counts: vec![0; 4],
                voluntary_pass_counts: vec![0; 4],
                first_hand_features: Some(
                    luck.iter()
                        .map(|&l| HandFeatures {
                            high_cards: 0,
                            pairs: 0,
                            triples: 0,
                            quads: 0,
                            lowest_strength: 0.0,
                            mean_strength: l,
                            hand_size: 13,
                        })
                        .collect(),
                ),
            }
        })
        .collect()
}

#[test]
fn estimator_recovers_injected_skill_and_removes_luck() {
    let skill = [0.6, 0.2, -0.2, -0.6];
    let report = estimator_report(&synthetic(2000, &skill));
    assert_eq!(report.match_count, 2000);
    assert!(report.feature_r2 > 0.3, "r2 {}", report.feature_r2);
    let values: Vec<f64> = report
        .strategies
        .iter()
        .map(|s| s.skill_score_estimate.value)
        .collect();
    assert!(values.windows(2).all(|w| w[0] > w[1]), "{values:?}");
    for s in &report.strategies {
        assert!(s.variance_reduction > 1.3, "{s:?}");
        assert!(s.skill_score_estimate.std_error < s.plain_mean_role_score_round1.std_error);
    }
}

#[test]
fn estimator_and_duplicate_agree_on_real_matches() {
    let table: Vec<Arc<dyn Strategy>> = vec![
        Arc::new(LowestLegal),
        Arc::new(RandomLegal),
        Arc::new(LowestLegal),
        Arc::new(RandomLegal),
    ];
    let groups = try_run_duplicate_batch(&configs(800, 1), &table, true).unwrap();
    let flat: Vec<MatchResult> = groups.iter().flat_map(|g| g.matches.clone()).collect();
    let dup = skill_report(&groups);
    let est = estimator_report(&flat);
    let cmp = compare(&dup, &est);
    assert_eq!(cmp.strategies.len(), 2);
    assert!((cmp.rank_agreement - 1.0).abs() < 1e-12);
    assert_ne!(cmp.verdict, "");
    assert!(serde_json::to_string(&cmp).is_ok());
}

#[test]
fn estimator_skips_a_table_with_a_single_strategy_name() {
    let mut matches = synthetic(50, &[0.0; 4]);
    for m in &mut matches {
        m.strategy_names = vec!["same".to_owned(); 4];
    }
    let report = estimator_report(&matches);
    assert_eq!(report.match_count, 50);
    assert_eq!(report.strategies, vec![]);
}

#[test]
fn hand_size_is_a_design_column() {
    let hand = [engine::Card::new(engine::Rank::Two, engine::Suit::Clubs, 0); 5];
    let f = HandFeatures::from_hand(&hand);
    assert_eq!(f.hand_size, 5);
    assert!((f.design_vector()[6] - 5.0).abs() < 1e-12);
    assert_eq!(f.as_vector().len(), 6);
}
