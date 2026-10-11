use super::*;
use crate::hand_reading::PassCeilings;
use engine::{Card, DeckVariant, DuplicateRule};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;

fn four_lowest_legal() -> Vec<Arc<dyn Strategy>> {
    vec![
        Arc::new(crate::strategies::LowestLegal),
        Arc::new(crate::strategies::LowestLegal),
        Arc::new(crate::strategies::LowestLegal),
        Arc::new(crate::strategies::LowestLegal),
    ]
}

#[test]
fn deal_seed_fixes_every_rounds_deck_regardless_of_play_stream() {
    let config = MatchConfig {
        player_count: 4,
        deck_variant: DeckVariant::Single,
        duplicate_rule: DuplicateRule::FirstDealtWins,
        rounds: 3,
        seed: 1,
        pass_rule: engine::PassRule::default(),
        exchange_rule: engine::ExchangeRule::default(),
    };
    let fixed = RunOptions {
        deal_seed: Some(77),
        record_deal_features: false,
    };
    let other = RunOptions {
        deal_seed: Some(78),
        record_deal_features: false,
    };
    let key = |deck: &[Card]| -> Vec<(u8, u8, u8)> {
        deck.iter()
            .map(|c| (c.rank as u8, c.suit as u8, c.deal_index))
            .collect()
    };
    for round in 0..3 {
        // Play streams with different positions and seeds, as differently
        // consuming strategies would leave them.
        let mut a = rand::rngs::StdRng::seed_from_u64(1);
        let mut b = rand::rngs::StdRng::seed_from_u64(2);
        let _: u64 = rand::RngExt::random(&mut b);
        assert_eq!(
            key(&shuffled_deck(&config, &fixed, round, &mut a)),
            key(&shuffled_deck(&config, &fixed, round, &mut b))
        );
        assert_ne!(
            key(&shuffled_deck(&config, &fixed, round, &mut a)),
            key(&shuffled_deck(&config, &other, round, &mut a))
        );
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    assert_ne!(
        key(&shuffled_deck(&config, &fixed, 0, &mut rng)),
        key(&shuffled_deck(&config, &fixed, 1, &mut rng))
    );
}

#[test]
fn default_options_equal_run_match_and_deal_seed_changes_the_deal() {
    let config = MatchConfig {
        player_count: 4,
        deck_variant: DeckVariant::Single,
        duplicate_rule: DuplicateRule::FirstDealtWins,
        rounds: 3,
        seed: 5,
        pass_rule: engine::PassRule::default(),
        exchange_rule: engine::ExchangeRule::default(),
    };
    let strategies = four_lowest_legal();
    let plain = run_match(&config, &strategies);
    let with = run_match_with(&config, &strategies, &RunOptions::default());
    assert_eq!(plain.role_history, with.role_history);
    assert!(with.first_hand_features.is_none());
    let features = |deal_seed| {
        run_match_with(
            &config,
            &strategies,
            &RunOptions {
                deal_seed: Some(deal_seed),
                record_deal_features: true,
            },
        )
        .first_hand_features
        .unwrap()
    };
    assert_eq!(features(1), features(1));
    assert_ne!(features(1), features(2));
}

#[test]
fn run_match_produces_one_role_history_entry_per_round() {
    let config = MatchConfig {
        player_count: 4,
        deck_variant: DeckVariant::Single,
        duplicate_rule: DuplicateRule::FirstDealtWins,
        rounds: 3,
        seed: 1,
        pass_rule: engine::PassRule::default(),
        exchange_rule: engine::ExchangeRule::default(),
    };
    let result = run_match(&config, &four_lowest_legal());
    assert_eq!(result.role_history.len(), 3);
    for round_roles in &result.role_history {
        assert_eq!(round_roles.len(), 4);
    }
}

#[test]
fn identical_config_and_strategies_are_fully_deterministic() {
    let config = MatchConfig {
        player_count: 4,
        deck_variant: DeckVariant::Single,
        duplicate_rule: DuplicateRule::FirstDealtWins,
        rounds: 3,
        seed: 42,
        pass_rule: engine::PassRule::default(),
        exchange_rule: engine::ExchangeRule::default(),
    };
    let strategies = four_lowest_legal();
    let first = run_match(&config, &strategies);
    let second = run_match(&config, &strategies);
    assert_eq!(first.role_history, second.role_history);
    assert_eq!(first.trick_count, second.trick_count);
    assert_eq!(first.pass_counts, second.pass_counts);
}

/// Wraps `LowestLegal`, independently counting every call where
/// `legal_moves` has no `Pass` — which is exactly when the seat is
/// leading (`engine`'s legal-move enumeration only offers `Pass`
/// while a combo is on the table), i.e. once per trick actually
/// played.
struct LeadCounter {
    leads: AtomicU32,
}

impl Strategy for LeadCounter {
    fn name(&self) -> &'static str {
        "LeadCounter"
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        context: &TurnContext<'_>,
        rng: &mut dyn rand::Rng,
    ) -> Move {
        if !legal_moves.contains(&Move::Pass) {
            self.leads.fetch_add(1, Ordering::Relaxed);
        }
        crate::strategies::LowestLegal.choose_play(legal_moves, duplicate_rule, context, rng)
    }

    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        duplicate_rule: DuplicateRule,
        rng: &mut dyn rand::Rng,
    ) -> Vec<Card> {
        crate::strategies::LowestLegal.choose_exchange_cards(hand, count, duplicate_rule, rng)
    }
}

#[test]
fn trick_count_counts_every_trick_led_including_each_rounds_final_one() {
    for seed in 0..20 {
        let counter = Arc::new(LeadCounter {
            leads: AtomicU32::new(0),
        });
        let strategies: Vec<Arc<dyn Strategy>> = vec![
            counter.clone(),
            counter.clone(),
            counter.clone(),
            counter.clone(),
        ];
        let config = MatchConfig {
            player_count: 4,
            deck_variant: DeckVariant::Single,
            duplicate_rule: DuplicateRule::FirstDealtWins,
            rounds: 3,
            seed,
            pass_rule: engine::PassRule::default(),
            exchange_rule: engine::ExchangeRule::default(),
        };
        let result = run_match(&config, &strategies);
        // Every round's last trick ends on a `Play` (the round ends
        // the moment only one seat still holds cards), never on a
        // pass-around, so counting only pass-resolved tricks would
        // come up at least `rounds` short of the true lead count.
        assert_eq!(
            result.trick_count,
            counter.leads.load(Ordering::Relaxed),
            "seed {seed}"
        );
    }
}

/// Wraps `LowestLegal`, independently counting every call to
/// `choose_exchange_cards` — used to confirm `run_match` actually
/// routes the exchange through each seat's `Strategy` rather than
/// falling back to some naive selection that never calls it.
struct ExchangeCallCounter {
    calls: Arc<AtomicU32>,
}

impl Strategy for ExchangeCallCounter {
    fn name(&self) -> &'static str {
        "ExchangeCallCounter"
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        context: &TurnContext<'_>,
        rng: &mut dyn rand::Rng,
    ) -> Move {
        crate::strategies::LowestLegal.choose_play(legal_moves, duplicate_rule, context, rng)
    }

    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        duplicate_rule: DuplicateRule,
        rng: &mut dyn rand::Rng,
    ) -> Vec<Card> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        crate::strategies::LowestLegal.choose_exchange_cards(hand, count, duplicate_rule, rng)
    }
}

#[test]
fn run_match_calls_choose_exchange_cards_during_the_exchange() {
    let calls = Arc::new(AtomicU32::new(0));
    let counter: Arc<dyn Strategy> = Arc::new(ExchangeCallCounter {
        calls: calls.clone(),
    });
    let strategies: Vec<Arc<dyn Strategy>> = vec![
        counter.clone(),
        counter.clone(),
        counter.clone(),
        counter.clone(),
    ];
    let config = MatchConfig {
        player_count: 4,
        deck_variant: DeckVariant::Single,
        duplicate_rule: DuplicateRule::FirstDealtWins,
        rounds: 3,
        seed: 7,
        pass_rule: engine::PassRule::default(),
        exchange_rule: engine::ExchangeRule::Free,
    };
    // The first round never exchanges (no prior roles yet), so at
    // least one of the two later rounds must trigger the exchange
    // for every seat that ends up in a low-ranked role.
    let _ = run_match(&config, &strategies);
    assert!(calls.load(Ordering::Relaxed) > 0);
    // Under the forced rule nobody chooses: the strategies are never asked.
    calls.store(0, Ordering::Relaxed);
    let forced = MatchConfig {
        exchange_rule: engine::ExchangeRule::Forced,
        ..config
    };
    let _ = run_match(&forced, &strategies);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn run_batch_rotates_which_seat_each_strategy_occupies() {
    let strategies: Vec<Arc<dyn Strategy>> = vec![
        Arc::new(crate::strategies::LowestLegal),
        Arc::new(crate::strategies::RandomLegal),
        Arc::new(crate::strategies::GreedyHighest),
    ];
    let configs: Vec<MatchConfig> = (0..4)
        .map(|seed| MatchConfig {
            player_count: 3,
            deck_variant: DeckVariant::Single,
            duplicate_rule: DuplicateRule::FirstDealtWins,
            rounds: 1,
            seed,
            pass_rule: engine::PassRule::default(),
            exchange_rule: engine::ExchangeRule::default(),
        })
        .collect();
    let results = run_batch(&configs, &strategies);
    let names: Vec<Vec<&str>> = results
        .iter()
        .map(|r| r.strategy_names.iter().map(String::as_str).collect())
        .collect();
    assert_eq!(names[0], ["LowestLegal", "RandomLegal", "GreedyHighest"]);
    assert_eq!(names[1], ["RandomLegal", "GreedyHighest", "LowestLegal"]);
    assert_eq!(names[2], ["GreedyHighest", "LowestLegal", "RandomLegal"]);
    assert_eq!(names[3], names[0]);
}

#[test]
fn run_batch_with_defaults_is_run_batch_and_features_only_add_data() {
    let strategies = four_lowest_legal();
    let configs: Vec<MatchConfig> = (0..4)
        .map(|seed| MatchConfig {
            player_count: 4,
            deck_variant: DeckVariant::Single,
            duplicate_rule: DuplicateRule::FirstDealtWins,
            rounds: 2,
            seed,
            pass_rule: engine::PassRule::default(),
            exchange_rule: engine::ExchangeRule::default(),
        })
        .collect();
    let plain = run_batch(&configs, &strategies);
    let same = run_batch_with(&configs, &strategies, &RunOptions::default());
    assert_eq!(format!("{plain:?}"), format!("{same:?}"));
    let with = run_batch_with(
        &configs,
        &strategies,
        &RunOptions {
            deal_seed: None,
            record_deal_features: true,
        },
    );
    for (a, b) in plain.iter().zip(&with) {
        assert_eq!(a.role_history, b.role_history);
        assert!(a.first_hand_features.is_none() && b.first_hand_features.is_some());
    }
}

/// Records whether any seat's `TurnContext.opponents` ever included
/// that seat itself, or had the wrong length — checked once after the
/// match completes.
struct TurnContextChecker {
    player_count: u8,
    violation: Arc<AtomicBool>,
}

impl Strategy for TurnContextChecker {
    fn name(&self) -> &'static str {
        "TurnContextChecker"
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        context: &TurnContext<'_>,
        rng: &mut dyn rand::Rng,
    ) -> Move {
        let wrong_length = context.opponents.len() != usize::from(self.player_count) - 1;
        let includes_self = context.opponents.iter().any(|o| o.seat == context.seat);
        if wrong_length || includes_self {
            self.violation.store(true, Ordering::Relaxed);
        }
        crate::strategies::LowestLegal.choose_play(legal_moves, duplicate_rule, context, rng)
    }

    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        duplicate_rule: DuplicateRule,
        rng: &mut dyn rand::Rng,
    ) -> Vec<Card> {
        crate::strategies::LowestLegal.choose_exchange_cards(hand, count, duplicate_rule, rng)
    }
}

#[test]
fn turn_context_excludes_the_acting_seat_from_opponents() {
    let config = MatchConfig {
        player_count: 4,
        deck_variant: DeckVariant::Single,
        duplicate_rule: DuplicateRule::FirstDealtWins,
        rounds: 3,
        seed: 7,
        pass_rule: engine::PassRule::default(),
        exchange_rule: engine::ExchangeRule::default(),
    };
    let violation = Arc::new(AtomicBool::new(false));
    let strategies: Vec<Arc<dyn Strategy>> = (0..4)
        .map(|_| {
            Arc::new(TurnContextChecker {
                player_count: config.player_count,
                violation: violation.clone(),
            }) as Arc<dyn Strategy>
        })
        .collect();
    let _ = run_match(&config, &strategies);
    assert!(!violation.load(Ordering::Relaxed));
}

/// Wraps `LowestLegal`, recording every `OpponentHand.pass_ceilings`
/// it's ever shown in `TurnContext.opponents` — used to confirm
/// `turn_context_for` actually wires the real
/// `hand_reading::read_pass_ceilings` output into each turn's
/// `TurnContext`, not just that the field exists on the type
/// (regression guard for the *wiring*).
struct PassCeilingRecorder {
    recorded: Mutex<Vec<PassCeilings>>,
}

impl Strategy for PassCeilingRecorder {
    fn name(&self) -> &'static str {
        "PassCeilingRecorder"
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        context: &TurnContext<'_>,
        rng: &mut dyn rand::Rng,
    ) -> Move {
        self.recorded
            .lock()
            .expect("test-only mutex is never poisoned")
            .extend(context.opponents.iter().map(|o| o.pass_ceilings));
        crate::strategies::LowestLegal.choose_play(legal_moves, duplicate_rule, context, rng)
    }

    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        duplicate_rule: DuplicateRule,
        rng: &mut dyn rand::Rng,
    ) -> Vec<Card> {
        crate::strategies::LowestLegal.choose_exchange_cards(hand, count, duplicate_rule, rng)
    }
}

#[test]
fn turn_context_wires_real_pass_ceilings_into_opponent_hands() {
    // Seat 0 (`RandomLegal`) sometimes voluntarily passes on a
    // combo it could have beaten; once it does, a later turn's
    // `TurnContext` (built by the production `turn_context_for`,
    // not a stub) must expose a non-default `PassCeilings` for seat
    // 0 to whichever other seat is acting. Scanning several seeds
    // keeps this deterministic without depending on exactly which
    // seed happens to produce an unrefuted pass.
    let mut saw_non_default = false;
    for seed in 0..30 {
        let recorder = Arc::new(PassCeilingRecorder {
            recorded: Mutex::new(Vec::new()),
        });
        let strategies: Vec<Arc<dyn Strategy>> = vec![
            Arc::new(crate::strategies::RandomLegal),
            recorder.clone(),
            recorder.clone(),
            recorder.clone(),
        ];
        let config = MatchConfig {
            player_count: 4,
            deck_variant: DeckVariant::Single,
            duplicate_rule: DuplicateRule::FirstDealtWins,
            rounds: 3,
            seed,
            pass_rule: engine::PassRule::default(),
            exchange_rule: engine::ExchangeRule::default(),
        };
        let _ = run_match(&config, &strategies);
        let seen_ceilings = recorder
            .recorded
            .lock()
            .expect("test-only mutex is never poisoned");
        if seen_ceilings.iter().any(|c| *c != PassCeilings::default()) {
            saw_non_default = true;
            break;
        }
    }
    assert!(
        saw_non_default,
        "expected at least one seed's match to produce a non-default \
         PassCeilings somewhere in the recorded OpponentHand.pass_ceilings"
    );
}

/// The unseen cards are exactly the round's deck minus the actor's
/// hand minus every played card, in deck order.
#[test]
fn unseen_cards_are_the_deck_minus_hand_and_played_cards() {
    use crate::test_support::for_each_state;

    let mut tracker = None;
    let mut checked = 0;
    for_each_state(23, 6, |state, _| {
        let round = state.round;
        if round.play_history().is_empty() && round.pass_history().is_empty() {
            tracker = Some(PassTracker::new(usize::from(state.players), state.rule));
        }
        let seat = round.seat_to_move().unwrap();
        let context = turn_context_for(
            round,
            seat,
            state.players,
            state.round_deck,
            tracker.as_mut().unwrap(),
            ContextNeeds::UNSEEN,
        );
        let played: Vec<Card> = round
            .play_history()
            .iter()
            .flat_map(|(_, combo)| combo.cards().iter().copied())
            .collect();
        let expected: Vec<Card> = state
            .round_deck
            .iter()
            .copied()
            .filter(|c| !round.hand(seat).contains(c) && !played.contains(c))
            .collect();
        assert_eq!(context.unseen_cards, expected);
        checked += 1;
    });
    assert!(checked > 3000);
}
