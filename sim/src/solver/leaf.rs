//! Leaf evaluators of the sampler: what a position at the depth limit is
//! worth to every seat, as expected 0-based places.

use std::cmp::Ordering;
use std::sync::Arc;

use engine::{Card, DuplicateRule, Round, SeatId};
use rand::SeedableRng;

use crate::hand_reading::PassTracker;
use crate::match_runner::turn_context_for;
use crate::strategy::{ContextNeeds, Strategy};

/// Expected 0-based place by seat (0 = first out), for all seats.
pub type Values = [f32; 6];

/// Values a position that is not finished. `cards` is every card that was in
/// a hand when the round was rebuilt (`Observation::cards_in_play`): the rest
/// of the round is played with exactly these. `seed` makes it reproducible.
pub trait LeafEvaluator: Send + Sync {
    fn values(&self, round: &Round, cards: &[Card], rule: DuplicateRule, seed: u64) -> Values;
}

/// The places of a finished round.
///
/// # Panics
///
/// Panics on more than 255 seats (a round has at most 6).
#[must_use]
pub fn finished_values(round: &Round) -> Values {
    let mut values = [0.0; 6];
    for (place, &seat) in round.finishing_order().iter().enumerate() {
        values[usize::from(seat)] = f32::from(u8::try_from(place).expect("at most 6 seats"));
    }
    values
}

/// Plays the round to its end with one strategy in every seat and takes the
/// finishing places (one rollout, so the value is a single outcome).
pub struct Rollout {
    policy: Arc<dyn Strategy>,
}

impl Rollout {
    #[must_use]
    pub fn new(policy: Arc<dyn Strategy>) -> Self {
        Self { policy }
    }
}

impl LeafEvaluator for Rollout {
    fn values(&self, round: &Round, cards: &[Card], rule: DuplicateRule, seed: u64) -> Values {
        let mut round = round.clone();
        let players = round.player_count();
        let player_count = u8::try_from(players).expect("at most 6 seats");
        let needs = self.policy.needs();
        let mut tracker = PassTracker::new(players, rule);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut moves = Vec::new();
        while let Some(seat) = round.seat_to_move() {
            round.legal_moves_into(&mut moves);
            let mv = if moves.len() == 1 {
                moves[0]
            } else {
                let mut context =
                    turn_context_for(&round, seat, player_count, cards, &mut tracker, needs);
                // The rebuilt round's history starts at the rebuild, so the
                // holder of a combo that was already on the table is not in it.
                if needs.contains(ContextNeeds::TRICK) {
                    context.combo_holder = round.current_combo().and(round.combo_holder());
                }
                self.policy.choose_play(&moves, rule, &context, &mut rng)
            };
            round
                .submit_move(seat, mv)
                .expect("the rollout policy picks one of the legal moves");
        }
        finished_values(&round)
    }
}

/// A static guess without playing on: seats still in are ranked by cards left,
/// less half a card for each single no other hand can beat (a sure trick), the
/// fewest first; equal scores share their places. Cheap and crude.
pub struct CardCount;

impl LeafEvaluator for CardCount {
    fn values(&self, round: &Round, _cards: &[Card], rule: DuplicateRule, _seed: u64) -> Values {
        let mut values = finished_values(round);
        let players = round.player_count();
        let out = round.finishing_order().len();
        let seats: Vec<SeatId> = (0..players)
            .filter_map(|s| SeatId::try_from(s).ok())
            .filter(|&s| round.hand_size(s) > 0)
            .collect();
        let best_of = |seat: SeatId| {
            round
                .hand(seat)
                .iter()
                .copied()
                .max_by(|a, b| a.compare(b, rule))
        };
        #[allow(clippy::cast_precision_loss)] // at most 54 cards
        let scores: Vec<f32> = seats
            .iter()
            .map(|&seat| {
                let others_best = seats
                    .iter()
                    .filter(|&&other| other != seat)
                    .filter_map(|&other| best_of(other))
                    .max_by(|a, b| a.compare(b, rule));
                let sure = round
                    .hand(seat)
                    .iter()
                    .filter(|card| {
                        others_best
                            .is_none_or(|best| card.compare(&best, rule) == Ordering::Greater)
                    })
                    .count();
                round.hand_size(seat) as f32 - 0.5 * sure as f32
            })
            .collect();
        #[allow(clippy::cast_precision_loss)] // at most 6 seats
        for (i, &seat) in seats.iter().enumerate() {
            let better = scores.iter().filter(|&&s| s < scores[i]).count();
            let equal = scores
                .iter()
                .filter(|s| s.total_cmp(&scores[i]).is_eq())
                .count();
            // Mean of the places `better .. better + equal`, after the seats already out.
            values[usize::from(seat)] = out as f32 + better as f32 + (equal as f32 - 1.0) / 2.0;
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::Observation;
    use crate::strategies::{Adaptive, AdaptiveConfig, LowestLegal};
    use crate::test_support::for_each_state;
    use engine::{Combo, Rank, Suit};

    const RULE: DuplicateRule = DuplicateRule::FirstDealtWins;

    fn c(rank: Rank, suit: Suit, i: u8) -> Card {
        Card::new(rank, suit, i)
    }

    fn sorted(values: Values, n: usize) -> Vec<f32> {
        let mut v = values[..n].to_vec();
        v.sort_by(f32::total_cmp);
        v
    }

    #[test]
    fn a_rollout_finishes_rebuilt_positions_with_a_full_set_of_places() {
        let rollout = Rollout::new(Arc::new(Adaptive::new(AdaptiveConfig::default())));
        let mut checked = 0;
        for_each_state(51, 2, |state, _| {
            let round = state.round;
            if round.pass_rule() != engine::PassRule::Final {
                return;
            }
            let hands: Vec<Vec<Card>> =
                (0..state.players).map(|s| round.hand(s).to_vec()).collect();
            let seat = round.seat_to_move().unwrap();
            let passed: Vec<bool> = (0..state.players)
                .map(|s| round.is_active(s) && round.has_passed(s))
                .collect();
            let on_table = round.combo_holder().zip(round.current_combo().copied());
            let Some(rebuilt) = Round::from_position(hands.clone(), RULE, seat, on_table, &passed)
            else {
                return;
            };
            let cards: Vec<Card> = hands.into_iter().flatten().collect();
            let values = rollout.values(&rebuilt, &cards, state.rule, 7);
            let n = usize::from(state.players);
            #[allow(clippy::cast_precision_loss)]
            let expected: Vec<f32> = (0..n).map(|p| p as f32).collect();
            assert_eq!(sorted(values, n), expected);
            assert_eq!(values, rollout.values(&rebuilt, &cards, state.rule, 7));
            checked += 1;
        });
        assert!(checked > 50, "only {checked} positions");
    }

    #[test]
    fn a_rollout_sees_the_holder_of_a_combo_already_on_the_table() {
        // Seat 1 holds a 9 on the table; seat 2 to move. The context of the first
        // rollout decision must name seat 1 as the holder (Observation needs it).
        struct Check;
        impl Strategy for Check {
            fn name(&self) -> &'static str {
                "check"
            }
            fn needs(&self) -> ContextNeeds {
                ContextNeeds::ALL
            }
            fn choose_play(
                &self,
                legal: &[engine::Move],
                rule: DuplicateRule,
                context: &crate::strategy::TurnContext<'_>,
                rng: &mut dyn rand::Rng,
            ) -> engine::Move {
                assert_eq!(
                    context.combo_holder.is_some(),
                    context.current_combo.is_some()
                );
                assert!(Observation::from_context(context, rule).is_some());
                LowestLegal.choose_play(legal, rule, context, rng)
            }
            fn choose_exchange_cards(
                &self,
                hand: &[Card],
                count: usize,
                rule: DuplicateRule,
                rng: &mut dyn rand::Rng,
            ) -> Vec<Card> {
                LowestLegal.choose_exchange_cards(hand, count, rule, rng)
            }
        }
        let hands = vec![
            vec![c(Rank::Three, Suit::Clubs, 0), c(Rank::Ace, Suit::Clubs, 1)],
            vec![c(Rank::Four, Suit::Clubs, 2), c(Rank::Five, Suit::Clubs, 3)],
            vec![c(Rank::Ten, Suit::Clubs, 4), c(Rank::Jack, Suit::Clubs, 5)],
        ];
        let nine = Combo::from_slice(&[c(Rank::Nine, Suit::Hearts, 6)]).unwrap();
        let round =
            Round::from_position(hands.clone(), RULE, 2, Some((1, nine)), &[false; 3]).unwrap();
        let cards: Vec<Card> = hands.into_iter().flatten().collect();
        let values = Rollout::new(Arc::new(Check)).values(&round, &cards, RULE, 1);
        assert_eq!(sorted(values, 3), vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn card_count_puts_the_shortest_hand_first_and_shares_ties() {
        let hands = vec![
            vec![c(Rank::Three, Suit::Clubs, 0)],
            vec![c(Rank::Four, Suit::Clubs, 1), c(Rank::Five, Suit::Clubs, 2)],
            vec![c(Rank::Six, Suit::Clubs, 3), c(Rank::Seven, Suit::Clubs, 4)],
            vec![],
        ];
        let round = Round::from_position(hands, RULE, 0, None, &[false; 4]).unwrap();
        let values = CardCount.values(&round, &[], RULE, 0);
        // Seat 3 is out (place 0). Seat 0: one card, score 1. Seat 2: two cards,
        // both above every other hand's best (5C), 2 - 2 x 0.5 = 1: it shares
        // places 1 and 2 with seat 0. Seat 1: 2, last.
        assert!(values[3].abs() < 1e-6);
        assert!((values[0] - 1.5).abs() < 1e-6);
        assert!((values[2] - 1.5).abs() < 1e-6);
        assert!((values[1] - 3.0).abs() < 1e-6);
    }
}
