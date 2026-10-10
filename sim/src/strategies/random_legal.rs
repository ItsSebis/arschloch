//! Uniformly samples among every legal move, including `Pass` — this is
//! the strategy that produces the "voluntary pass" signal (see
//! docs/ROADMAP.md, Phase 4).

use engine::{Card, DuplicateRule, Move};
use rand::seq::{IndexedRandom, SliceRandom};

use crate::strategy::{ContextNeeds, Strategy, TurnContext};

#[derive(Debug, Clone, Copy, Default)]
pub struct RandomLegal;

impl Strategy for RandomLegal {
    fn name(&self) -> &'static str {
        "RandomLegal"
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        _duplicate_rule: DuplicateRule,
        _context: &TurnContext<'_>,
        rng: &mut dyn rand::Rng,
    ) -> Move {
        legal_moves
            .choose(rng)
            .copied()
            .expect("legal_moves is never empty when a seat is actually to move")
    }

    fn needs(&self) -> ContextNeeds {
        ContextNeeds::NONE
    }

    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        _duplicate_rule: DuplicateRule,
        rng: &mut dyn rand::Rng,
    ) -> Vec<Card> {
        let mut indices: Vec<usize> = (0..hand.len()).collect();
        indices.shuffle(rng);
        indices.truncate(count);
        indices.into_iter().map(|i| hand[i]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::{Card, Combo, Rank, Suit};
    use rand::SeedableRng;

    fn test_rng(seed: u64) -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(seed)
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

    #[test]
    fn always_returns_one_of_the_legal_moves() {
        let legal = vec![
            Move::Pass,
            Move::Play(Combo::new(vec![Card::new(Rank::Nine, Suit::Clubs, 0)]).unwrap()),
        ];
        let strategy = RandomLegal;
        for seed in 0..20 {
            let chosen = strategy.choose_play(
                &legal,
                DuplicateRule::FirstDealtWins,
                &empty_context(),
                &mut test_rng(seed),
            );
            assert!(legal.contains(&chosen));
        }
    }

    #[test]
    fn eventually_picks_every_option_across_many_seeds() {
        let legal = vec![
            Move::Pass,
            Move::Play(Combo::new(vec![Card::new(Rank::Nine, Suit::Clubs, 0)]).unwrap()),
        ];
        let strategy = RandomLegal;
        let mut saw_pass = false;
        let mut saw_play = false;
        for seed in 0..50 {
            match strategy.choose_play(
                &legal,
                DuplicateRule::FirstDealtWins,
                &empty_context(),
                &mut test_rng(seed),
            ) {
                Move::Pass => saw_pass = true,
                Move::Play(_) => saw_play = true,
            }
        }
        assert!(saw_pass && saw_play);
    }

    #[test]
    fn choose_exchange_cards_returns_exactly_count_distinct_cards_from_hand() {
        let strategy = RandomLegal;
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
            &mut test_rng(7),
        );
        assert_eq!(given.len(), 2);
        assert_ne!(given[0], given[1]);
        for card in &given {
            assert!(hand.contains(card));
        }
    }

    #[test]
    fn choose_exchange_cards_can_give_up_the_entire_hand() {
        let strategy = RandomLegal;
        let hand = vec![
            Card::new(Rank::Two, Suit::Clubs, 0),
            Card::new(Rank::Five, Suit::Clubs, 0),
        ];
        let given = strategy.choose_exchange_cards(
            &hand,
            hand.len(),
            DuplicateRule::FirstDealtWins,
            &mut test_rng(3),
        );
        assert_eq!(given.len(), hand.len());
        assert_ne!(given[0], given[1]);
    }
}
