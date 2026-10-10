//! Drives `engine`'s single-round primitives into a full multi-round
//! match: a fresh shuffle and deal every round (cards are discarded
//! within a round, not carried over — see docs/RULES.md, "Playing a
//! Round"), the mandatory exchange using the previous round's roles, and
//! each seat's `Strategy` choosing among `engine`'s reported legal moves.

use std::sync::Arc;

use engine::{
    assign_roles, deal, exchange_with_rule, lowest_card_holder, standard_deck, Card, Move, Round,
    SeatId,
};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::hand_features::HandFeatures;
use crate::hand_reading::PassTracker;
use crate::match_config::MatchConfig;
use crate::match_result::MatchResult;
use crate::strategy::{OpponentHand, Strategy, TurnContext};
use crate::training::evaluate::mix;

/// Builds the `TurnContext` for `seat`'s upcoming `choose_play` call:
/// every other seat's current hand size/activity/pass ceilings, the
/// exact multiset of cards neither in `seat`'s hand nor played by
/// anyone yet this round, this seat's own pass ceilings, and the
/// combo currently on the table (if any). `round_deck` is the full set
/// of cards dealt this round (see `run_match`'s `round_deck`), used as
/// the starting point before subtracting what's now visible.
pub(crate) fn turn_context_for<'a>(
    round: &'a Round,
    seat: SeatId,
    player_count: u8,
    round_deck: &[Card],
    tracker: &mut PassTracker,
) -> TurnContext<'a> {
    tracker.update(round.play_history(), round.pass_history());
    let all_ceilings = tracker.ceilings();

    let opponents: Vec<OpponentHand> = (0..player_count)
        .filter(|&s| s != seat)
        .map(|s| OpponentHand {
            seat: s,
            hand_size: round.hand_size(s),
            active: round.is_active(s),
            pass_ceilings: all_ceilings[usize::from(s)],
        })
        .collect();

    // Every card of the round has its own `deal_index` (run_match numbers
    // the deck), so "seen" is a bit per index; the unseen cards keep the
    // deck's order.
    let mut seen = [false; 256];
    for card in round.hand(seat).iter().chain(
        round
            .play_history()
            .iter()
            .flat_map(|(_, combo)| combo.cards()),
    ) {
        seen[usize::from(card.deal_index)] = true;
    }
    let unseen_cards: Vec<Card> = round_deck
        .iter()
        .filter(|card| !seen[usize::from(card.deal_index)])
        .copied()
        .collect();

    TurnContext {
        seat,
        hand: round.hand(seat),
        opponents,
        unseen_cards,
        own_pass_ceilings: all_ceilings[usize::from(seat)],
        current_combo: round.current_combo(),
    }
}

/// Optional behaviour of `run_match_with`. The default reproduces
/// `run_match` exactly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunOptions {
    /// When set, round `r`'s deck is shuffled from its own generator
    /// seeded by `deal_round_seed(deal_seed, r)` instead of the shared
    /// play stream, so two matches with the same `deal_seed` deal the
    /// same decks whatever randomness their strategies consume.
    pub deal_seed: Option<u64>,
    /// Record each seat's round-1 hand features (after the deal, before
    /// any exchange) in `MatchResult::first_hand_features`.
    pub record_deal_features: bool,
}

/// The seed of round `round_index`'s shuffle under `deal_seed`.
#[must_use]
pub fn deal_round_seed(deal_seed: u64, round_index: usize) -> u64 {
    mix(mix(deal_seed) ^ round_index as u64)
}

/// Move-shape counters `play_out` adds to; a rollout passes a scratch
/// value it throws away.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlayCounters {
    /// Tricks led.
    pub trick_count: u32,
    /// Passes submitted, per seat.
    pub pass_counts: Vec<u32>,
    /// Passes submitted while a play was also legal, per seat.
    pub voluntary_pass_counts: Vec<u32>,
}

impl PlayCounters {
    #[must_use]
    pub fn new(player_count: u8) -> Self {
        Self {
            trick_count: 0,
            pass_counts: vec![0; usize::from(player_count)],
            voluntary_pass_counts: vec![0; usize::from(player_count)],
        }
    }
}

/// Plays `round` to completion from its current state: each seat to move
/// asks its strategy (drawing randomness from `rng`) and the move is
/// submitted. `round_deck` is every card dealt this round (see
/// `turn_context_for`); `counters` accumulates tricks and passes. The
/// pass tracker restarts from the round's history, so it works for a
/// round cloned mid-way. Resulting placings are in
/// `round.finishing_order()`.
///
/// # Panics
///
/// Panics if `strategies` or `counters` do not match
/// `config.player_count` seats, or a strategy returns an illegal move.
pub fn play_out(
    round: &mut Round,
    strategies: &[Arc<dyn Strategy>],
    config: &MatchConfig,
    round_deck: &[Card],
    rng: &mut dyn rand::Rng,
    counters: &mut PlayCounters,
) {
    let mut tracker = PassTracker::new(usize::from(config.player_count), config.duplicate_rule);
    while !round.is_complete() {
        let seat = round.seat_to_move().expect("round is not complete");
        if round.current_combo().is_none() {
            counters.trick_count += 1;
        }
        let legal_moves = round.legal_moves();

        let context = turn_context_for(round, seat, config.player_count, round_deck, &mut tracker);

        let chosen = strategies[usize::from(seat)].choose_play(
            &legal_moves,
            config.duplicate_rule,
            &context,
            rng,
        );

        if chosen == Move::Pass {
            counters.pass_counts[usize::from(seat)] += 1;
            if legal_moves.iter().any(|mv| matches!(mv, Move::Play(_))) {
                counters.voluntary_pass_counts[usize::from(seat)] += 1;
            }
        }
        round
            .submit_move(seat, chosen)
            .expect("strategies only choose from the moves engine just reported as legal");
    }
}

/// Round `round_index`'s numbered, shuffled deck: from the play stream
/// `rng`, or from the round's own generator when `options.deal_seed` is set.
fn shuffled_deck(
    config: &MatchConfig,
    options: &RunOptions,
    round_index: usize,
    rng: &mut rand::rngs::StdRng,
) -> Vec<Card> {
    let mut deck = standard_deck(config.deck_variant);
    match options.deal_seed {
        Some(deal_seed) => {
            let mut deal_rng =
                rand::rngs::StdRng::seed_from_u64(deal_round_seed(deal_seed, round_index));
            deck.shuffle(&mut deal_rng);
        }
        None => deck.shuffle(rng),
    }
    for (index, card) in deck.iter_mut().enumerate() {
        card.deal_index = u8::try_from(index).expect("deck sizes (52/104) fit in u8");
    }
    deck
}

/// Simulates one full match (`config.rounds` rounds, role carry-over
/// between them) using `strategies` (one per seat). Same as
/// `run_match_with` with default options.
///
/// # Panics
///
/// Panics if `strategies.len() != usize::from(config.player_count)`, or
/// if `config.rounds == 0`, or if `config.player_count` isn't a table
/// size `engine` supports (3-6) — all are programming errors in how the
/// caller built `MatchConfig`/`strategies`, not user input this phase
/// exposes to anyone yet (see `docs/CODING_GUIDELINES.md`, "Errors"; a CLI
/// boundary with proper `Result`-based validation arrives in Phase 3).
#[must_use]
pub fn run_match(config: &MatchConfig, strategies: &[Arc<dyn Strategy>]) -> MatchResult {
    run_match_with(config, strategies, &RunOptions::default())
}

/// `run_match` with `RunOptions` (fixed deal seed, deal-feature
/// recording).
///
/// # Panics
///
/// As `run_match`.
#[must_use]
pub fn run_match_with(
    config: &MatchConfig,
    strategies: &[Arc<dyn Strategy>],
    options: &RunOptions,
) -> MatchResult {
    assert_eq!(
        strategies.len(),
        usize::from(config.player_count),
        "one strategy is required per seat"
    );
    assert!(config.rounds > 0, "a match needs at least one round");

    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    let mut previous_roles: Option<Vec<engine::Role>> = None;
    let mut previous_arschloch: Option<SeatId> = None;
    let mut role_history = Vec::with_capacity(config.rounds);
    let mut counters = PlayCounters::new(config.player_count);
    let mut first_hand_features = None;

    for round_index in 0..config.rounds {
        let deck = shuffled_deck(config, options, round_index, &mut rng);
        let mut hands = deal(deck, config.player_count)
            .expect("standard_deck always yields enough cards for a supported player count");
        if round_index == 0 && options.record_deal_features {
            first_hand_features = Some(hands.iter().map(|h| HandFeatures::from_hand(h)).collect());
        }

        let leader = match (&previous_roles, previous_arschloch) {
            (Some(roles), Some(arschloch)) => {
                exchange_with_rule(
                    &mut hands,
                    roles,
                    config.duplicate_rule,
                    config.exchange_rule,
                    |seat, hand, count, duplicate_rule| {
                        strategies[seat].choose_exchange_cards(hand, count, duplicate_rule, &mut rng)
                    },
                )
                .expect(
                    "previous_roles always came from assign_roles for this player_count, and every \
                     Strategy::choose_exchange_cards returns exactly `count` cards from its own hand",
                );
                arschloch
            }
            _ => lowest_card_holder(&hands, config.duplicate_rule)
                .expect("a freshly dealt hand set is never empty"),
        };

        let round_deck: Vec<Card> = hands.iter().flatten().copied().collect();

        let mut round =
            Round::with_pass_rule(hands, config.duplicate_rule, config.pass_rule, leader)
                .expect("player_count/leader are always valid for a supported table size");

        play_out(
            &mut round,
            strategies,
            config,
            &round_deck,
            &mut rng,
            &mut counters,
        );

        let finishing_order = round.finishing_order().to_vec();
        let roles = assign_roles(&finishing_order, config.player_count)
            .expect("finishing_order is always a valid permutation for a supported player count");
        previous_arschloch = finishing_order.last().copied();
        role_history.push(roles.clone());
        previous_roles = Some(roles);
    }

    MatchResult {
        player_count: config.player_count,
        strategy_names: strategies.iter().map(|s| s.name().to_string()).collect(),
        role_history,
        trick_count: counters.trick_count,
        pass_counts: counters.pass_counts,
        voluntary_pass_counts: counters.voluntary_pass_counts,
        first_hand_features,
    }
}

/// Simulates every config in `configs` in parallel — independent
/// matches, no shared mutable state (see docs/ARCHITECTURE.md,
/// "Threading model"). Rotates which physical seat each strategy
/// occupies by each config's position in `configs` (cyclically, by
/// `strategies.len()`), so `deal`'s documented uneven-remainder rule
/// (docs/RULES.md, "Players & Deck") doesn't bias aggregate
/// role-by-strategy statistics toward whichever strategies happen to sit
/// in the earliest seats — the bias cancels out across the batch instead.
#[must_use]
pub fn run_batch(configs: &[MatchConfig], strategies: &[Arc<dyn Strategy>]) -> Vec<MatchResult> {
    configs
        .par_iter()
        .enumerate()
        .map(|(index, config)| {
            let rotation = index % strategies.len();
            let rotated: Vec<Arc<dyn Strategy>> = strategies
                .iter()
                .cycle()
                .skip(rotation)
                .take(strategies.len())
                .cloned()
                .collect();
            run_match(config, &rotated)
        })
        .collect()
}

#[cfg(test)]
mod tests {
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

    /// Records whether any seat's `TurnContext.opponents` ever included
    /// that seat itself, or had the wrong length — checked once after the
    /// match completes (docs/ROADMAP.md, Phase 6, Review Focus).
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
    /// (regression guard for the *wiring*, per Phase 6's final review).
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
}
