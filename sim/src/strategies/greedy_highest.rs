//! The most aggressive legal strategy: prefers the largest combo (shed
//! the most cards per lead — following is already size-locked by the
//! current combo, so this only matters while leading), then the
//! highest-ranked, legal combo; passes only when no `Play` is legal.

use engine::{Card, DuplicateRule, Move};

use crate::strategy::{Strategy, TurnContext};

#[derive(Debug, Clone, Copy, Default)]
pub struct GreedyHighest;

impl Strategy for GreedyHighest {
    fn name(&self) -> &'static str {
        "GreedyHighest"
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        _context: &TurnContext<'_>,
        _rng: &mut dyn rand::Rng,
    ) -> Move {
        legal_moves
            .iter()
            .filter_map(|mv| match mv {
                Move::Play(combo) => Some((combo.size(), combo.top_card(duplicate_rule), mv)),
                Move::Pass => None,
            })
            .max_by(|a, b| {
                a.0.cmp(&b.0)
                    .then_with(|| a.1.compare(&b.1, duplicate_rule))
            })
            .map_or(Move::Pass, |(_, _, mv)| mv.clone())
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
    use engine::{Card, Combo, Rank, Suit};
    use rand::SeedableRng;

    fn test_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(0)
    }

    fn empty_context() -> TurnContext<'static> {
        TurnContext {
            seat: 0,
            hand: &[],
            opponents: Vec::new(),
            unseen_cards: Vec::new(),
            own_pass_ceilings: crate::hand_reading::PassCeilings::default(),
            current_combo: None,
        }
    }

    fn single(rank: Rank, suit: Suit) -> Move {
        Move::Play(Combo::new(vec![Card::new(rank, suit, 0)]).unwrap())
    }

    #[test]
    fn prefers_the_largest_combo_size() {
        let pair = Move::Play(
            Combo::new(vec![
                Card::new(Rank::Two, Suit::Clubs, 0),
                Card::new(Rank::Two, Suit::Diamonds, 0),
            ])
            .unwrap(),
        );
        let legal = vec![pair.clone(), single(Rank::Ace, Suit::Clubs)];
        let strategy = GreedyHighest;
        let chosen = strategy.choose_play(
            &legal,
            DuplicateRule::FirstDealtWins,
            &empty_context(),
            &mut test_rng(),
        );
        assert_eq!(chosen, pair);
    }

    #[test]
    fn among_equal_sizes_prefers_the_highest_top_card() {
        let legal = vec![
            single(Rank::Nine, Suit::Clubs),
            single(Rank::Ace, Suit::Diamonds),
        ];
        let strategy = GreedyHighest;
        let chosen = strategy.choose_play(
            &legal,
            DuplicateRule::FirstDealtWins,
            &empty_context(),
            &mut test_rng(),
        );
        assert_eq!(chosen, single(Rank::Ace, Suit::Diamonds));
    }

    #[test]
    fn passes_when_no_play_is_legal() {
        let legal = vec![Move::Pass];
        let strategy = GreedyHighest;
        let chosen = strategy.choose_play(
            &legal,
            DuplicateRule::FirstDealtWins,
            &empty_context(),
            &mut test_rng(),
        );
        assert_eq!(chosen, Move::Pass);
    }

    #[test]
    fn choose_exchange_cards_gives_up_the_highest_cards() {
        let strategy = GreedyHighest;
        let hand = vec![
            Card::new(Rank::Two, Suit::Clubs, 0),
            Card::new(Rank::Five, Suit::Clubs, 0),
            Card::new(Rank::Seven, Suit::Clubs, 0),
            Card::new(Rank::Jack, Suit::Clubs, 0),
            Card::new(Rank::Ace, Suit::Clubs, 0),
        ];
        let given = strategy.choose_exchange_cards(
            &hand,
            2,
            DuplicateRule::FirstDealtWins,
            &mut test_rng(),
        );
        assert_eq!(given.len(), 2);
        assert!(given.contains(&Card::new(Rank::Ace, Suit::Clubs, 0)));
        assert!(given.contains(&Card::new(Rank::Jack, Suit::Clubs, 0)));
    }
}
