//! Drives `engine`'s single-round primitives into a full multi-round
//! match: a fresh shuffle and deal every round (cards are discarded
//! within a round, not carried over — see docs/RULES.md, "Playing a
//! Round"), the mandatory exchange using the previous round's roles, and
//! each seat's `Strategy` choosing among `engine`'s reported legal moves.

use std::sync::Arc;

use engine::{
    assign_roles, deal, exchange, lowest_card_holder, standard_deck, Move, Round, SeatId,
};
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::match_config::MatchConfig;
use crate::match_result::MatchResult;
use crate::strategy::Strategy;

/// Simulates one full match (`config.rounds` rounds, role carry-over
/// between them) using `strategies` (one per seat).
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
    let mut trick_count = 0u32;
    let mut pass_count = 0u32;
    let mut voluntary_pass_count = 0u32;

    for _ in 0..config.rounds {
        let mut deck = standard_deck(config.deck_variant);
        deck.shuffle(&mut rng);
        for (index, card) in deck.iter_mut().enumerate() {
            card.deal_index = u8::try_from(index).expect("deck sizes (52/104) fit in u8");
        }
        let mut hands = deal(deck, config.player_count)
            .expect("standard_deck always yields enough cards for a supported player count");

        let leader = match (&previous_roles, previous_arschloch) {
            (Some(roles), Some(arschloch)) => {
                exchange(&mut hands, roles, config.duplicate_rule)
                    .expect("previous_roles always came from assign_roles for this player_count");
                arschloch
            }
            _ => lowest_card_holder(&hands, config.duplicate_rule)
                .expect("a freshly dealt hand set is never empty"),
        };

        let mut round = Round::new(hands, config.duplicate_rule, leader)
            .expect("player_count/leader are always valid for a supported table size");

        while !round.is_complete() {
            let seat = round.seat_to_move().expect("round is not complete");
            let legal_moves = round.legal_moves();
            let chosen = strategies[usize::from(seat)].choose_play(
                &legal_moves,
                config.duplicate_rule,
                &mut rng,
            );

            if chosen == Move::Pass {
                pass_count += 1;
                if legal_moves.iter().any(|mv| matches!(mv, Move::Play(_))) {
                    voluntary_pass_count += 1;
                }
            }
            let combo_was_on_table = round.current_combo().is_some();
            round
                .submit_move(seat, chosen)
                .expect("strategies only choose from the moves engine just reported as legal");
            if combo_was_on_table && round.current_combo().is_none() {
                trick_count += 1;
            }
        }

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
        trick_count,
        pass_count,
        voluntary_pass_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::{DeckVariant, DuplicateRule};

    fn four_lowest_legal() -> Vec<Arc<dyn Strategy>> {
        vec![
            Arc::new(crate::strategies::LowestLegal),
            Arc::new(crate::strategies::LowestLegal),
            Arc::new(crate::strategies::LowestLegal),
            Arc::new(crate::strategies::LowestLegal),
        ]
    }

    #[test]
    fn run_match_produces_one_role_history_entry_per_round() {
        let config = MatchConfig {
            player_count: 4,
            deck_variant: DeckVariant::Single,
            duplicate_rule: DuplicateRule::FirstDealtWins,
            rounds: 3,
            seed: 1,
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
        };
        let strategies = four_lowest_legal();
        let first = run_match(&config, &strategies);
        let second = run_match(&config, &strategies);
        assert_eq!(first.role_history, second.role_history);
        assert_eq!(first.trick_count, second.trick_count);
        assert_eq!(first.pass_count, second.pass_count);
    }
}
