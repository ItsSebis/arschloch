//! Uniformly samples among every legal move, including `Pass` — this is
//! the strategy that produces the "voluntary pass" signal (see
//! docs/ROADMAP.md, Phase 4).

use engine::{DuplicateRule, Move};
use rand::seq::IndexedRandom;

use crate::strategy::Strategy;

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
        rng: &mut dyn rand::Rng,
    ) -> Move {
        legal_moves
            .choose(rng)
            .cloned()
            .expect("legal_moves is never empty when a seat is actually to move")
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

    #[test]
    fn always_returns_one_of_the_legal_moves() {
        let legal = vec![
            Move::Pass,
            Move::Play(Combo::new(vec![Card::new(Rank::Nine, Suit::Clubs, 0)]).unwrap()),
        ];
        let strategy = RandomLegal;
        for seed in 0..20 {
            let chosen =
                strategy.choose_play(&legal, DuplicateRule::FirstDealtWins, &mut test_rng(seed));
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
            match strategy.choose_play(&legal, DuplicateRule::FirstDealtWins, &mut test_rng(seed)) {
                Move::Pass => saw_pass = true,
                Move::Play(_) => saw_play = true,
            }
        }
        assert!(saw_pass && saw_play);
    }
}
