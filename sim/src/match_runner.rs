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
use rayon::prelude::*;

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
            if round.current_combo().is_none() {
                trick_count += 1;
            }
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
            round
                .submit_move(seat, chosen)
                .expect("strategies only choose from the moves engine just reported as legal");
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
    use engine::{DeckVariant, DuplicateRule};
    use std::sync::atomic::{AtomicU32, Ordering};

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
            rng: &mut dyn rand::Rng,
        ) -> Move {
            if !legal_moves.contains(&Move::Pass) {
                self.leads.fetch_add(1, Ordering::Relaxed);
            }
            crate::strategies::LowestLegal.choose_play(legal_moves, duplicate_rule, rng)
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
}
