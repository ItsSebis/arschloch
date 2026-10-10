//! A pure endgame-denial strategy: every decision is driven by
//! opponents' hand sizes, never by card counting (docs/ROADMAP.md,
//! Phase 6).
//!
//! `CLOSE_TO_FINISHING` (2 cards) is the "about to finish" threshold —
//! see docs/RULES.md's per-player-count deal sizes (as low as ~8-9
//! cards/seat at 6 players with a single deck, up to the mid-30s at 3
//! players with a double deck): 2 cards is deep into the final
//! stretch at every supported table size and deck variant, and larger
//! deals only make it a smaller, still-urgent fraction. 2 rather than
//! 1 gives this seat one extra turn of lead-time to act before the
//! low-hand opponent is already gone.
//!
//! Whenever at least one still-active opponent is at or below that
//! threshold, this seat switches to "deny control": never pass when a
//! legal beating play exists, and prefer the highest legal play — a
//! higher combo is harder for a near-empty hand to match, and winning
//! the trick means leading (and choosing the shape of) the next one,
//! which can lock out an opponent who can't field that shape. That
//! combination is exactly `GreedyHighest`'s existing behavior, so
//! danger mode delegates to it rather than re-deriving the same rule
//! — a cheap proxy for "retain control," not a simulation of future
//! turns. Otherwise (no opponent close), there's nothing yet to deny,
//! so it conserves instead by delegating to `LowestLegal`.

use engine::{Card, DuplicateRule, Move};

use crate::strategies::{GreedyHighest, LowestLegal};
use crate::strategy::{ContextNeeds, Strategy, TurnContext};

/// A still-active opponent at or below this many cards is "close to
/// finishing" — see the module doc comment for the threshold
/// rationale.
const CLOSE_TO_FINISHING: usize = 2;

#[derive(Debug, Clone, Copy, Default)]
pub struct EndgameDenial;

impl Strategy for EndgameDenial {
    fn name(&self) -> &'static str {
        "EndgameDenial"
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        context: &TurnContext<'_>,
        rng: &mut dyn rand::Rng,
    ) -> Move {
        let danger = context
            .opponents
            .iter()
            .any(|o| o.active && o.hand_size <= CLOSE_TO_FINISHING);
        if danger {
            GreedyHighest.choose_play(legal_moves, duplicate_rule, context, rng)
        } else {
            LowestLegal.choose_play(legal_moves, duplicate_rule, context, rng)
        }
    }

    fn needs(&self) -> ContextNeeds {
        ContextNeeds::OPPONENTS
    }

    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        duplicate_rule: DuplicateRule,
        _rng: &mut dyn rand::Rng,
    ) -> Vec<Card> {
        crate::strategies::take_highest_naive(hand, count, duplicate_rule)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::OpponentHand;
    use engine::{Combo, Rank, Suit};
    use rand::SeedableRng;

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit, 0)
    }

    fn context_with_opponent(hand_size: usize, active: bool) -> TurnContext<'static> {
        TurnContext {
            seat: 0,
            hand: &[],
            opponents: vec![OpponentHand {
                seat: 1,
                hand_size,
                active,
                pass_ceilings: crate::hand_reading::PassCeilings::default(),
            }],
            unseen_cards: vec![],
            own_pass_ceilings: crate::hand_reading::PassCeilings::default(),
            current_combo: None,
        }
    }

    #[test]
    fn no_close_opponent_leads_low_like_lowest_legal() {
        let legal_moves = vec![
            Move::Play(Combo::new(vec![card(Rank::Six, Suit::Clubs)]).unwrap()),
            Move::Play(Combo::new(vec![card(Rank::Nine, Suit::Clubs)]).unwrap()),
        ];
        let context = context_with_opponent(10, true);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let chosen = EndgameDenial.choose_play(
            &legal_moves,
            DuplicateRule::FirstDealtWins,
            &context,
            &mut rng,
        );
        assert_eq!(
            chosen,
            Move::Play(Combo::new(vec![card(Rank::Six, Suit::Clubs)]).unwrap())
        );
    }

    #[test]
    fn close_opponent_leads_high_to_retain_control() {
        let legal_moves = vec![
            Move::Play(Combo::new(vec![card(Rank::Six, Suit::Clubs)]).unwrap()),
            Move::Play(Combo::new(vec![card(Rank::Nine, Suit::Clubs)]).unwrap()),
        ];
        let context = context_with_opponent(2, true);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let chosen = EndgameDenial.choose_play(
            &legal_moves,
            DuplicateRule::FirstDealtWins,
            &context,
            &mut rng,
        );
        assert_eq!(
            chosen,
            Move::Play(Combo::new(vec![card(Rank::Nine, Suit::Clubs)]).unwrap())
        );
    }

    #[test]
    fn close_opponent_never_passes_when_a_beating_play_is_legal() {
        let legal_moves = vec![
            Move::Pass,
            Move::Play(Combo::new(vec![card(Rank::Eight, Suit::Clubs)]).unwrap()),
        ];
        let context = context_with_opponent(1, true);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let chosen = EndgameDenial.choose_play(
            &legal_moves,
            DuplicateRule::FirstDealtWins,
            &context,
            &mut rng,
        );
        assert_eq!(
            chosen,
            Move::Play(Combo::new(vec![card(Rank::Eight, Suit::Clubs)]).unwrap())
        );
    }

    #[test]
    fn a_finished_opponents_low_hand_size_does_not_count_as_close() {
        let legal_moves = vec![
            Move::Play(Combo::new(vec![card(Rank::Six, Suit::Clubs)]).unwrap()),
            Move::Play(Combo::new(vec![card(Rank::Nine, Suit::Clubs)]).unwrap()),
        ];
        let context = context_with_opponent(0, false);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let chosen = EndgameDenial.choose_play(
            &legal_moves,
            DuplicateRule::FirstDealtWins,
            &context,
            &mut rng,
        );
        assert_eq!(
            chosen,
            Move::Play(Combo::new(vec![card(Rank::Six, Suit::Clubs)]).unwrap())
        );
    }
}
