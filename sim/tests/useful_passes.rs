//! The useful-pass analysis (Phase 12): hand-built scenarios, replay
//! fidelity, determinism and the sampling edge cases.

use std::sync::Arc;

use engine::{
    Card, Combo, DeckVariant, DuplicateRule, ExchangeRule, Move, PassRule, Rank, Round, Suit,
};
use sim::useful_passes::{
    analyse_useful_passes, evaluate_pass, replay_match, PassVerdict, UsefulPassOptions,
};
use sim::{
    run_match_with, Adaptive, AdaptiveConfig, CardCounter, HoldBackPairs, LowestLegal, MatchConfig,
    RunOptions, Strategy,
};

fn card(rank: Rank, suit: Suit, deal_index: u8) -> Card {
    Card::new(rank, suit, deal_index)
}

fn config(player_count: u8, seed: u64, rounds: usize) -> MatchConfig {
    MatchConfig {
        player_count,
        deck_variant: DeckVariant::Single,
        duplicate_rule: DuplicateRule::FirstDealtWins,
        rounds,
        seed,
        pass_rule: PassRule::default(),
        exchange_rule: ExchangeRule::default(),
    }
}

/// Seat 2 leads its lowest card from `hands[2]`; seat 0 is then to move
/// facing it. Every seat plays the deterministic `LowestLegal`, so the
/// rollouts have a single outcome and the verdict is exact.
fn verdict_with_margin(hands: Vec<Vec<Card>>, margin: f64) -> PassVerdict {
    let strategies: Vec<Arc<dyn Strategy>> = vec![
        Arc::new(LowestLegal),
        Arc::new(LowestLegal),
        Arc::new(LowestLegal),
    ];
    let deck: Vec<Card> = hands.iter().flatten().copied().collect();
    let lead = *hands[2]
        .iter()
        .min_by(|a, b| a.compare(b, DuplicateRule::FirstDealtWins))
        .expect("seat 2 holds cards");
    let mut round =
        Round::with_pass_rule(hands, DuplicateRule::FirstDealtWins, PassRule::default(), 2)
            .expect("valid round");
    round
        .submit_move(2, Move::Play(Combo::new(vec![lead]).expect("one card")))
        .expect("the lead is legal");
    assert_eq!(round.seat_to_move(), Some(0));
    let options = UsefulPassOptions {
        rollouts: 4,
        margin,
        ..UsefulPassOptions::default()
    };
    evaluate_pass(&round, 0, &strategies, &config(3, 0, 1), &deck, 1, &options)
        .expect("a play is legal, so the pass is voluntary")
}

fn verdict(hands: Vec<Vec<Card>>) -> PassVerdict {
    verdict_with_margin(hands, 0.0)
}

#[test]
fn a_pass_that_keeps_the_seat_ahead_is_useful() {
    use Rank::*;
    use Suit::*;
    // Seat 2 leads the Six of hearts (keeping Seven of hearts, Six of
    // spades). Seat 0 holds Seven of spades and Three of hearts, so its only
    // play is the Seven. With LowestLegal everywhere that play leaves it
    // 2nd in every rollout, while passing (seat 1's Eight takes the trick,
    // and seat 0 later sheds its two cards first) ends 1st: the pass gains
    // exactly one place.
    let v = verdict(vec![
        vec![card(Seven, Spades, 0), card(Three, Hearts, 1)],
        vec![
            card(Four, Hearts, 2),
            card(Three, Diamonds, 3),
            card(Eight, Hearts, 4),
        ],
        vec![
            card(Six, Hearts, 5),
            card(Seven, Hearts, 6),
            card(Six, Spades, 7),
        ],
    ]);
    assert!(v.useful, "{v:?}");
    assert!((v.gain - 1.0).abs() < 1e-9, "{v:?}");
    assert!(v.mean_place_pass < v.mean_place_best_alternative);
}

#[test]
fn a_pass_that_gives_up_the_win_is_harmful() {
    use Rank::*;
    use Suit::*;
    // Seat 0 holds Three of hearts and Four of spades; seat 2 leads the
    // Three of diamonds (keeping Eight of hearts, Three of clubs). Playing
    // the weakest beating card wins seat 0 first place in every rollout;
    // the pass drops it to last, a loss of two places.
    let v = verdict(vec![
        vec![card(Three, Hearts, 0), card(Four, Spades, 1)],
        vec![card(Three, Spades, 2), card(Five, Hearts, 3)],
        vec![
            card(Eight, Hearts, 4),
            card(Three, Diamonds, 5),
            card(Three, Clubs, 6),
        ],
    ]);
    assert!(!v.useful, "{v:?}");
    assert!(v.gain < 0.0, "{v:?}");
    assert!((v.gain + 2.0).abs() < 1e-9, "{v:?}");
}

#[test]
fn a_margin_can_veto_a_marginal_pass() {
    use Rank::*;
    use Suit::*;
    let hands = vec![
        vec![card(Seven, Spades, 0), card(Three, Hearts, 1)],
        vec![
            card(Four, Hearts, 2),
            card(Three, Diamonds, 3),
            card(Eight, Hearts, 4),
        ],
        vec![
            card(Six, Hearts, 5),
            card(Seven, Hearts, 6),
            card(Six, Spades, 7),
        ],
    ];
    // The gain is exactly one place: strictly better with margin 0 and
    // 0.5, not better once the margin reaches the gain.
    assert!(verdict_with_margin(hands.clone(), 0.0).useful);
    assert!(verdict_with_margin(hands.clone(), 0.5).useful);
    assert!(!verdict_with_margin(hands, 1.0).useful);
}

fn batch_configs(count: u64, rounds: usize) -> Vec<MatchConfig> {
    (0..count).map(|seed| config(4, seed, rounds)).collect()
}

fn mixed_strategies() -> Vec<Arc<dyn Strategy>> {
    vec![
        Arc::new(HoldBackPairs),
        Arc::new(LowestLegal),
        Arc::new(CardCounter),
        Arc::new(Adaptive::new(
            "reading,tempo,bully"
                .parse::<AdaptiveConfig>()
                .expect("valid spec"),
        )),
    ]
}

fn rotated(strategies: &[Arc<dyn Strategy>], index: usize) -> Vec<Arc<dyn Strategy>> {
    let rotation = index % strategies.len();
    strategies
        .iter()
        .cycle()
        .skip(rotation)
        .take(strategies.len())
        .cloned()
        .collect()
}

#[test]
fn replay_reproduces_the_original_matches() {
    let strategies = mixed_strategies();
    let options = UsefulPassOptions {
        rollouts: 2,
        sample_fraction: 0.3,
        ..UsefulPassOptions::default()
    };
    for (index, cfg) in batch_configs(24, 4).iter().enumerate() {
        let seats = rotated(&strategies, index);
        let original = run_match_with(cfg, &seats, &RunOptions::default());
        let (replayed, _) = replay_match(cfg, &seats, &RunOptions::default(), index, &options);
        assert_eq!(
            original.role_history, replayed.role_history,
            "match {index}"
        );
        assert_eq!(original.trick_count, replayed.trick_count);
        assert_eq!(original.pass_counts, replayed.pass_counts);
        assert_eq!(
            original.voluntary_pass_counts,
            replayed.voluntary_pass_counts
        );
        assert_eq!(original.strategy_names, replayed.strategy_names);
    }
}

#[test]
fn sampling_fraction_zero_samples_nothing_and_one_samples_everything() {
    let strategies = mixed_strategies();
    let configs = batch_configs(12, 3);
    let none = analyse_useful_passes(
        &configs,
        &strategies,
        &UsefulPassOptions {
            sample_fraction: 0.0,
            ..UsefulPassOptions::default()
        },
    );
    assert!(none
        .by_strategy
        .values()
        .all(|s| s.sampled_voluntary_passes == 0
            && s.useful_pass_share.is_none()
            && s.useful_pass_gain.is_none()));
    let all = analyse_useful_passes(
        &configs,
        &strategies,
        &UsefulPassOptions {
            rollouts: 1,
            sample_fraction: 1.0,
            ..UsefulPassOptions::default()
        },
    );
    for stats in all.by_strategy.values() {
        assert_eq!(stats.sampled_voluntary_passes, stats.voluntary_passes_total);
    }
    // Totals do not depend on the fraction.
    for (name, stats) in &none.by_strategy {
        assert_eq!(
            stats.voluntary_passes_total,
            all.by_strategy[name].voluntary_passes_total
        );
    }
}

#[test]
fn lowest_legal_never_passes_voluntarily_but_hold_back_pairs_does() {
    let stats = analyse_useful_passes(
        &batch_configs(40, 4),
        &mixed_strategies(),
        &UsefulPassOptions {
            rollouts: 2,
            sample_fraction: 0.5,
            ..UsefulPassOptions::default()
        },
    );
    let lowest = stats.by_strategy.get("LowestLegal");
    assert!(lowest.is_none_or(|s| s.sampled_voluntary_passes == 0 && s.voluntary_passes_total == 0));
    let hold = &stats.by_strategy["HoldBackPairs"];
    assert!(hold.sampled_voluntary_passes > 0, "{hold:?}");
    let share = hold.useful_pass_share.expect("has samples");
    assert!((0.0..=1.0).contains(&share.value));
    assert!(share.interval.low <= share.value && share.value <= share.interval.high);
    assert!(hold.useful_pass_gain.is_some());
}

#[test]
fn the_analysis_is_deterministic_and_independent_of_the_thread_count() {
    let strategies = mixed_strategies();
    let configs = batch_configs(30, 3);
    let options = UsefulPassOptions {
        rollouts: 3,
        sample_fraction: 0.4,
        seed: 7,
        ..UsefulPassOptions::default()
    };
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool")
            .install(|| analyse_useful_passes(&configs, &strategies, &options))
    };
    let one = run(1);
    assert_eq!(one, run(1));
    assert_eq!(one, run(4));
    let other_seed = analyse_useful_passes(
        &configs,
        &strategies,
        &UsefulPassOptions { seed: 8, ..options },
    );
    assert_ne!(one, other_seed);
    let json = serde_json::to_string(&one).expect("serialises");
    assert!(json.contains("useful_pass_share") && json.contains("useful_pass_gain"));
}

/// Real-batch timing check, run with `--ignored --nocapture`.
#[test]
#[ignore = "timing check: 400 matches x 6 rounds"]
fn real_batch_numbers() {
    let started = std::time::Instant::now();
    let stats = analyse_useful_passes(
        &batch_configs(400, 6),
        &mixed_strategies(),
        &UsefulPassOptions {
            rollouts: 16,
            sample_fraction: 0.2,
            ..UsefulPassOptions::default()
        },
    );
    for (name, s) in &stats.by_strategy {
        println!("{name}: {s:#?}");
    }
    println!("wall time {:?}", started.elapsed());
}
