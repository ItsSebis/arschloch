//! Follows without splitting a same-rank reserve when a non-splitting
//! beating play is available; passes rather than split one when every
//! beating play would. Leading is unaffected (see docs/ROADMAP.md, Phase
//! 4, "Strategy diversification") — this strategy differs from
//! `LowestLegal` only in how it follows.

use std::collections::HashMap;

use engine::{rank_groups, Card, DuplicateRule, Move, Rank};

use crate::strategies::LowestLegal;
use crate::strategy::Strategy;

#[derive(Debug, Clone, Copy, Default)]
pub struct HoldBackPairs;

impl Strategy for HoldBackPairs {
    fn name(&self) -> &'static str {
        "HoldBackPairs"
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        rng: &mut dyn rand::Rng,
    ) -> Move {
        if !legal_moves.contains(&Move::Pass) {
            // Leading never offers Pass (engine::Round::legal_moves), so
            // there's no split-vs-preserve choice to make here.
            return LowestLegal.choose_play(legal_moves, duplicate_rule, rng);
        }

        // Following: every candidate combo already shares the current
        // combo's size, so two candidates at the same rank mean the hand
        // holds more of that rank than this play uses — playing either
        // strands the rest.
        let plays: Vec<(Rank, Card)> = legal_moves
            .iter()
            .filter_map(|mv| match mv {
                Move::Play(combo) => {
                    let top = combo.top_card(duplicate_rule);
                    Some((top.rank, top))
                }
                Move::Pass => None,
            })
            .collect();
        let mut candidates_per_rank: HashMap<Rank, u8> = HashMap::new();
        for &(rank, _) in &plays {
            *candidates_per_rank.entry(rank).or_insert(0) += 1;
        }

        let safe_top_card = plays
            .iter()
            .filter(|(rank, _)| candidates_per_rank[rank] == 1)
            .map(|&(_, card)| card)
            .min_by(|a, b| a.compare(b, duplicate_rule));

        match safe_top_card {
            Some(card) => legal_moves
                .iter()
                .find(
                    |mv| matches!(mv, Move::Play(combo) if combo.top_card(duplicate_rule) == card),
                )
                .expect("safe_top_card was derived from a Play move in legal_moves")
                .clone(),
            None => Move::Pass,
        }
    }

    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        duplicate_rule: DuplicateRule,
        _rng: &mut dyn rand::Rng,
    ) -> Vec<Card> {
        // Give up whole isolated cards before touching any same-rank
        // group of 2+, smallest groups first; within a length class,
        // prefer the highest-ranked group (still shed value
        // preferentially, just never break a reserve while an isolated
        // card remains).
        let mut groups = rank_groups(hand);
        groups.sort_by(|a, b| {
            a.len()
                .cmp(&b.len())
                .then_with(|| b[0].compare(&a[0], duplicate_rule))
        });

        let mut selected = Vec::with_capacity(count);
        for mut group in groups {
            if selected.len() >= count {
                break;
            }
            let take = (count - selected.len()).min(group.len());
            group.sort_by(|a, b| b.compare(a, duplicate_rule));
            selected.extend(group.into_iter().take(take));
        }
        selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::{Combo, Suit};
    use rand::SeedableRng;

    fn test_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(0)
    }

    fn single(rank: Rank, suit: Suit) -> Move {
        Move::Play(Combo::new(vec![Card::new(rank, suit, 0)]).unwrap())
    }

    fn pair(rank: Rank, suits: [Suit; 2]) -> Move {
        Move::Play(
            Combo::new(vec![
                Card::new(rank, suits[0], 0),
                Card::new(rank, suits[1], 0),
            ])
            .unwrap(),
        )
    }

    #[test]
    fn leading_defers_to_lowest_legal() {
        let legal = vec![
            single(Rank::Nine, Suit::Clubs),
            single(Rank::Six, Suit::Diamonds),
        ];
        let chosen =
            HoldBackPairs.choose_play(&legal, DuplicateRule::FirstDealtWins, &mut test_rng());
        assert_eq!(chosen, single(Rank::Six, Suit::Diamonds));
    }

    #[test]
    fn following_prefers_a_rank_with_no_spare_over_a_lower_rank_that_has_one() {
        // Six is offered twice (a spare Six exists in hand), Nine once
        // (no spare) — HoldBackPairs must pick Nine even though Six is
        // lower, to avoid splitting the pair.
        let legal = vec![
            Move::Pass,
            single(Rank::Six, Suit::Diamonds),
            single(Rank::Six, Suit::Spades),
            single(Rank::Nine, Suit::Clubs),
        ];
        let chosen =
            HoldBackPairs.choose_play(&legal, DuplicateRule::FirstDealtWins, &mut test_rng());
        assert_eq!(chosen, single(Rank::Nine, Suit::Clubs));
    }

    #[test]
    fn following_passes_rather_than_split_the_only_beating_rank() {
        let legal = vec![
            Move::Pass,
            single(Rank::Six, Suit::Diamonds),
            single(Rank::Six, Suit::Spades),
        ];
        let chosen =
            HoldBackPairs.choose_play(&legal, DuplicateRule::FirstDealtWins, &mut test_rng());
        assert_eq!(chosen, Move::Pass);
    }

    #[test]
    fn following_with_no_spares_anywhere_plays_the_lowest_like_lowest_legal() {
        let legal = vec![
            Move::Pass,
            single(Rank::Nine, Suit::Clubs),
            single(Rank::Six, Suit::Diamonds),
        ];
        let chosen =
            HoldBackPairs.choose_play(&legal, DuplicateRule::FirstDealtWins, &mut test_rng());
        assert_eq!(chosen, single(Rank::Six, Suit::Diamonds));
    }

    #[test]
    fn following_with_no_legal_play_at_all_passes() {
        let legal = vec![Move::Pass];
        let chosen =
            HoldBackPairs.choose_play(&legal, DuplicateRule::FirstDealtWins, &mut test_rng());
        assert_eq!(chosen, Move::Pass);
    }

    #[test]
    fn following_a_pair_prefers_a_rank_with_no_spare_over_a_lower_rank_that_has_one() {
        // Six is offered as two distinct pairs (a spare pair of Sixes
        // exists in hand — e.g. 3 Sixes total), Nine as only one pair
        // (no spare) — HoldBackPairs must pick Nine even though Six is
        // lower, to avoid splitting the pair.
        let legal = vec![
            Move::Pass,
            pair(Rank::Six, [Suit::Diamonds, Suit::Hearts]),
            pair(Rank::Six, [Suit::Spades, Suit::Clubs]),
            pair(Rank::Nine, [Suit::Diamonds, Suit::Hearts]),
        ];
        let chosen =
            HoldBackPairs.choose_play(&legal, DuplicateRule::FirstDealtWins, &mut test_rng());
        assert_eq!(chosen, pair(Rank::Nine, [Suit::Diamonds, Suit::Hearts]));
    }

    #[test]
    fn choose_exchange_cards_prefers_isolated_cards_over_a_pair() {
        let strategy = HoldBackPairs;
        let hand = vec![
            Card::new(Rank::Five, Suit::Clubs, 0),
            Card::new(Rank::Five, Suit::Hearts, 0), // pair of 5s
            Card::new(Rank::Nine, Suit::Clubs, 0),  // isolated
            Card::new(Rank::Jack, Suit::Clubs, 0),  // isolated
        ];

        let one = strategy.choose_exchange_cards(
            &hand,
            1,
            DuplicateRule::FirstDealtWins,
            &mut test_rng(),
        );
        assert_eq!(one, vec![Card::new(Rank::Jack, Suit::Clubs, 0)]);

        let two = strategy.choose_exchange_cards(
            &hand,
            2,
            DuplicateRule::FirstDealtWins,
            &mut test_rng(),
        );
        assert_eq!(two.len(), 2);
        assert!(two.contains(&Card::new(Rank::Jack, Suit::Clubs, 0)));
        assert!(two.contains(&Card::new(Rank::Nine, Suit::Clubs, 0)));
    }

    #[test]
    fn choose_exchange_cards_breaks_the_pair_only_when_forced() {
        let strategy = HoldBackPairs;
        let hand = vec![
            Card::new(Rank::Five, Suit::Clubs, 0),
            Card::new(Rank::Five, Suit::Hearts, 0),
            Card::new(Rank::Nine, Suit::Clubs, 0),
            Card::new(Rank::Jack, Suit::Clubs, 0),
        ];

        let three = strategy.choose_exchange_cards(
            &hand,
            3,
            DuplicateRule::FirstDealtWins,
            &mut test_rng(),
        );
        assert_eq!(three.len(), 3);
        assert!(three.contains(&Card::new(Rank::Jack, Suit::Clubs, 0)));
        assert!(three.contains(&Card::new(Rank::Nine, Suit::Clubs, 0)));
        let fives_included = three.iter().filter(|c| c.rank == Rank::Five).count();
        assert_eq!(fives_included, 1);
    }

    #[test]
    fn choose_exchange_cards_handles_no_isolated_cards_at_all() {
        // Two pairs, no singles: forced to break at least one pair even
        // for a small count.
        let strategy = HoldBackPairs;
        let hand = vec![
            Card::new(Rank::Five, Suit::Clubs, 0),
            Card::new(Rank::Five, Suit::Hearts, 0),
            Card::new(Rank::Nine, Suit::Clubs, 0),
            Card::new(Rank::Nine, Suit::Hearts, 0),
        ];
        let given = strategy.choose_exchange_cards(
            &hand,
            3,
            DuplicateRule::FirstDealtWins,
            &mut test_rng(),
        );
        assert_eq!(given.len(), 3);
        for card in &given {
            assert!(hand.contains(card));
        }
    }
}
