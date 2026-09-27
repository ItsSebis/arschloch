//! The most conservative legal strategy: always plays the smallest, then
//! lowest-ranked, legal combo; passes only when no `Play` is legal.

use engine::{Card, DuplicateRule, Move};

use crate::strategy::{Strategy, TurnContext};

#[derive(Debug, Clone, Copy, Default)]
pub struct LowestLegal;

impl Strategy for LowestLegal {
    fn name(&self) -> &'static str {
        "LowestLegal"
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
            .min_by(|a, b| {
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
        }
    }

    fn single(rank: Rank, suit: Suit) -> Move {
        Move::Play(Combo::new(vec![Card::new(rank, suit, 0)]).unwrap())
    }

    #[test]
    fn prefers_the_smallest_combo_size() {
        let pair = Move::Play(
            Combo::new(vec![
                Card::new(Rank::Two, Suit::Clubs, 0),
                Card::new(Rank::Two, Suit::Diamonds, 0),
            ])
            .unwrap(),
        );
        let legal = vec![pair, single(Rank::Nine, Suit::Clubs)];
        let strategy = LowestLegal;
        let chosen = strategy.choose_play(
            &legal,
            DuplicateRule::FirstDealtWins,
            &empty_context(),
            &mut test_rng(),
        );
        assert_eq!(chosen, single(Rank::Nine, Suit::Clubs));
    }

    #[test]
    fn among_equal_sizes_prefers_the_lowest_top_card() {
        let legal = vec![
            single(Rank::Nine, Suit::Clubs),
            single(Rank::Six, Suit::Diamonds),
        ];
        let strategy = LowestLegal;
        let chosen = strategy.choose_play(
            &legal,
            DuplicateRule::FirstDealtWins,
            &empty_context(),
            &mut test_rng(),
        );
        assert_eq!(chosen, single(Rank::Six, Suit::Diamonds));
    }

    #[test]
    fn passes_when_no_play_is_legal() {
        let legal = vec![Move::Pass];
        let strategy = LowestLegal;
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
        let strategy = LowestLegal;
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
