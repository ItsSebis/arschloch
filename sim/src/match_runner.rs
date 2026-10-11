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
use crate::hand_reading::{PassCeilings, PassTracker};
use crate::match_config::MatchConfig;
use crate::match_result::MatchResult;
use crate::strategy::{ContextNeeds, OpponentHand, Strategy, TurnContext};
use crate::training::evaluate::mix;

/// Builds the `TurnContext` for `seat`'s upcoming `choose_play` call:
/// every other seat's current hand size/activity/pass ceilings, the
/// exact multiset of cards neither in `seat`'s hand nor played by
/// anyone yet this round, this seat's own pass ceilings, and the
/// combo currently on the table (if any). `round_deck` is the full set
/// of cards dealt this round (see `run_match`'s `round_deck`), used as
/// the starting point before subtracting what's now visible.
///
/// Only the parts in `needs` (the seat's `Strategy::needs`) are built;
/// the rest hold empty/default values. The pass tracker is only advanced
/// when some pass ceiling is needed: it is incremental over the round's
/// complete histories, so a later call that needs ceilings catches up on
/// everything skipped and yields the same ceilings.
pub(crate) fn turn_context_for<'a>(
    round: &'a Round,
    seat: SeatId,
    player_count: u8,
    round_deck: &[Card],
    tracker: &mut PassTracker,
    needs: ContextNeeds,
) -> TurnContext<'a> {
    let ceilings_needed = needs.pass_ceilings();
    if ceilings_needed {
        tracker.update(round.play_history(), round.pass_history());
    }
    let all_ceilings = tracker.ceilings();

    let opponents: Vec<OpponentHand> = if needs.contains(ContextNeeds::OPPONENTS) {
        let with_ceilings = needs.contains(ContextNeeds::OPPONENT_PASS_CEILINGS);
        (0..player_count)
            .filter(|&s| s != seat)
            .map(|s| OpponentHand {
                seat: s,
                hand_size: round.hand_size(s),
                active: round.is_active(s),
                pass_ceilings: if with_ceilings {
                    all_ceilings[usize::from(s)]
                } else {
                    PassCeilings::default()
                },
            })
            .collect()
    } else {
        Vec::new()
    };

    // Every card of the round has its own `deal_index` (run_match numbers
    // the deck; decks hold at most 104 cards), so "seen" is a bit per
    // index; the unseen cards keep the deck's order.
    let unseen_cards: Vec<Card> = if needs.contains(ContextNeeds::UNSEEN) {
        let bit = |card: &Card| {
            1u128
                .checked_shl(u32::from(card.deal_index))
                .expect("deal indices of a round's deck are below 128")
        };
        let mut seen = 0u128;
        for card in round.hand(seat).iter().chain(
            round
                .play_history()
                .iter()
                .flat_map(|(_, combo)| combo.cards()),
        ) {
            seen |= bit(card);
        }
        let mut unseen = Vec::with_capacity(round_deck.len());
        unseen.extend(round_deck.iter().filter(|card| seen & bit(card) == 0));
        unseen
    } else {
        Vec::new()
    };

    TurnContext {
        seat,
        hand: round.hand(seat),
        opponents,
        unseen_cards,
        own_pass_ceilings: if needs.contains(ContextNeeds::OWN_PASS_CEILINGS) {
            all_ceilings[usize::from(seat)]
        } else {
            PassCeilings::default()
        },
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
    play_out_observed(
        round,
        strategies,
        config,
        round_deck,
        rng,
        counters,
        |_, _, _| {},
    );
}

/// `play_out` that also calls `on_voluntary_pass(round, seat, decision)`
/// for every voluntary pass (a pass although a play was legal), with the
/// round in the state before the pass and `decision` the number of moves
/// already submitted in this round. The observer must not touch `rng`'s
/// stream, or the play changes.
pub(crate) fn play_out_observed(
    round: &mut Round,
    strategies: &[Arc<dyn Strategy>],
    config: &MatchConfig,
    round_deck: &[Card],
    rng: &mut dyn rand::Rng,
    counters: &mut PlayCounters,
    mut on_voluntary_pass: impl FnMut(&Round, SeatId, usize),
) {
    let mut decision = 0usize;
    let mut tracker = PassTracker::new(usize::from(config.player_count), config.duplicate_rule);
    let mut legal_moves = Vec::new();
    while !round.is_complete() {
        let seat = round.seat_to_move().expect("round is not complete");
        if round.current_combo().is_none() {
            counters.trick_count += 1;
        }
        round.legal_moves_into(&mut legal_moves);

        let strategy = &strategies[usize::from(seat)];
        let context = turn_context_for(
            round,
            seat,
            config.player_count,
            round_deck,
            &mut tracker,
            strategy.needs(),
        );

        let chosen = strategy.choose_play(&legal_moves, config.duplicate_rule, &context, rng);

        if chosen == Move::Pass {
            counters.pass_counts[usize::from(seat)] += 1;
            if legal_moves.iter().any(|mv| matches!(mv, Move::Play(_))) {
                counters.voluntary_pass_counts[usize::from(seat)] += 1;
                on_voluntary_pass(round, seat, decision);
            }
        }
        round
            .submit_move(seat, chosen)
            .expect("strategies only choose from the moves engine just reported as legal");
        decision += 1;
    }
}

/// Round `round_index`'s numbered, shuffled deck: from the play stream
/// `rng`, or from the round's own generator when `options.deal_seed` is set.
pub(crate) fn shuffled_deck(
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
/// exposes to anyone yet (see `docs/CODING_GUIDELINES.md`, "Errors"; the CLI
/// boundary validates before calling).
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
    run_match_observed(config, strategies, options, |_| {})
}

/// A voluntary pass seen by `run_match_observed`'s observer, at the state
/// before the pass.
pub(crate) struct VoluntaryPass<'a> {
    pub round_index: usize,
    /// Moves already submitted in this round.
    pub decision: usize,
    pub seat: SeatId,
    pub round: &'a Round,
    /// Every card dealt this round (see `turn_context_for`).
    pub round_deck: &'a [Card],
}

/// `run_match_with` that also calls `observer` on every voluntary pass.
/// The observer must not touch the match's random stream, so the result
/// equals `run_match_with`'s.
///
/// # Panics
///
/// As `run_match`.
pub(crate) fn run_match_observed(
    config: &MatchConfig,
    strategies: &[Arc<dyn Strategy>],
    options: &RunOptions,
    mut observer: impl FnMut(&VoluntaryPass<'_>),
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

        play_out_observed(
            &mut round,
            strategies,
            config,
            &round_deck,
            &mut rng,
            &mut counters,
            |round, seat, decision| {
                observer(&VoluntaryPass {
                    round_index,
                    decision,
                    seat,
                    round,
                    round_deck: &round_deck,
                });
            },
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
    run_batch_with(configs, strategies, &RunOptions::default())
}

/// `run_batch` with `RunOptions` applied to every match (same seat
/// rotation). With the default options it is exactly `run_batch`.
#[must_use]
pub fn run_batch_with(
    configs: &[MatchConfig],
    strategies: &[Arc<dyn Strategy>],
    options: &RunOptions,
) -> Vec<MatchResult> {
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
            run_match_with(config, &rotated, options)
        })
        .collect()
}

#[cfg(test)]
mod tests;
