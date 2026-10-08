//! A pure card-counting strategy: every decision is driven by exact
//! knowledge of which cards remain unseen (unplayed and not in this
//! seat's own hand) — this is a closed deck with no draw pile, so
//! "unseen" is exact, not an estimate (docs/ROADMAP.md, Phase 6).
//!
//! **Precious combos.** A legal combo's top card is "precious" when no
//! unseen card beats it under `Card::compare` (the same total order —
//! rank, then suit, then the duplicate-deal tiebreak — that
//! `Combo::beats` uses), i.e. the top card is greater than or equal to
//! every unseen card. This is a deliberately cheap single-card check,
//! not a full analysis of whether a beating combo is actually
//! *formable*: a combo of size 2+ can only be beaten by another
//! same-size combo, which needs enough unseen copies of some
//! comparably-ranked card to exist — a single unseen card that would
//! beat this combo's top card is enough to mark it non-precious here
//! even if no one could ever actually assemble a same-size beater from
//! it. That means this rule is deliberately conservative for combos of
//! size 2+ (it may treat some truly-unbeatable combos as "spendable"),
//! never the reverse for those combos. This project's existing
//! strategies are cheap heuristics, not search, so this conservatism
//! is accepted rather than counting per-rank unseen quantities to
//! determine formability.
//!
//! Preciousness is upward-closed in the `Card::compare` order (if a
//! lower card is precious, every higher card is too), so this signal
//! only ever changes the outcome relative to `LowestLegal` while
//! *leading*, where combo size can differ across candidates
//! (`LowestLegal` breaks ties by size first, so it would spend a small
//! precious card immediately). While *following*, size is fixed by the
//! table, so this strategy deliberately degenerates to `LowestLegal`'s
//! choice — leading always requires a play (never `Pass`), so holding
//! a precious combo back only changes *which* combo leads, never
//! *whether* one does, and precious combos get forced out once every
//! legal lead is precious (typically late in the round) — this never
//! stalls emptying the hand.

use std::cmp::Ordering;

use engine::{Card, DuplicateRule, Move};

use crate::strategy::{Strategy, TurnContext};

#[derive(Debug, Clone, Copy, Default)]
pub struct CardCounter;

impl Strategy for CardCounter {
    fn name(&self) -> &'static str {
        "CardCounter"
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        context: &TurnContext<'_>,
        _rng: &mut dyn rand::Rng,
    ) -> Move {
        let highest_unseen = context
            .unseen_cards
            .iter()
            .copied()
            .max_by(|a, b| a.compare(b, duplicate_rule));
        let is_precious = |top: Card| {
            highest_unseen
                .is_none_or(|highest| top.compare(&highest, duplicate_rule) != Ordering::Less)
        };

        let plays: Vec<(usize, Card, bool, &Move)> = legal_moves
            .iter()
            .filter_map(|mv| match mv {
                Move::Play(combo) => {
                    let top = combo.top_card(duplicate_rule);
                    Some((combo.size(), top, is_precious(top), mv))
                }
                Move::Pass => None,
            })
            .collect();

        let lowest = |pool: &[(usize, Card, bool, &Move)]| {
            pool.iter()
                .min_by(|a, b| {
                    a.0.cmp(&b.0)
                        .then_with(|| a.1.compare(&b.1, duplicate_rule))
                })
                .map(|&(_, _, _, mv)| mv.clone())
        };

        let non_precious: Vec<_> = plays.iter().copied().filter(|&(_, _, p, _)| !p).collect();
        lowest(&non_precious)
            .or_else(|| lowest(&plays))
            .unwrap_or(Move::Pass)
    }

    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        duplicate_rule: DuplicateRule,
        _rng: &mut dyn rand::Rng,
    ) -> Vec<Card> {
        // Card-counting-aware exchange selection is out of this
        // phase's scope (Global Constraints) — reuse the naive
        // highest-N give-up.
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

    fn context_with_unseen(unseen_cards: Vec<Card>) -> TurnContext<'static> {
        TurnContext {
            seat: 0,
            hand: &[],
            opponents: vec![OpponentHand {
                seat: 1,
                hand_size: 5,
                active: true,
                pass_ceilings: crate::hand_reading::PassCeilings::default(),
            }],
            unseen_cards,
            own_pass_ceilings: crate::hand_reading::PassCeilings::default(),
            current_combo: None,
        }
    }

    #[test]
    fn leading_saves_a_precious_single_ace_for_a_non_precious_pair_of_threes() {
        let legal_moves = vec![
            Move::Play(Combo::new(vec![card(Rank::Ace, Suit::Clubs)]).unwrap()),
            Move::Play(
                Combo::new(vec![
                    card(Rank::Three, Suit::Clubs),
                    card(Rank::Three, Suit::Diamonds),
                ])
                .unwrap(),
            ),
        ];
        let context = context_with_unseen(vec![card(Rank::Nine, Suit::Hearts)]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let chosen = CardCounter.choose_play(
            &legal_moves,
            DuplicateRule::FirstDealtWins,
            &context,
            &mut rng,
        );
        assert_eq!(
            chosen,
            Move::Play(
                Combo::new(vec![
                    card(Rank::Three, Suit::Clubs),
                    card(Rank::Three, Suit::Diamonds),
                ])
                .unwrap()
            )
        );
    }

    #[test]
    fn leading_treats_a_same_rank_lower_suit_king_as_beatable_not_precious() {
        // Suit order (docs/RULES.md): Diamonds < Hearts < Spades < Clubs, so
        // a King of Clubs beats a King of Hearts even though both share the
        // same rank. A rank-only precious check would wrongly treat the
        // King of Hearts as unbeatable and hold it back; the fixed
        // `Card::compare`-based check must recognize it is genuinely
        // beatable and spend it instead of the pair of threes.
        let legal_moves = vec![
            Move::Play(Combo::new(vec![card(Rank::King, Suit::Hearts)]).unwrap()),
            Move::Play(
                Combo::new(vec![
                    card(Rank::Three, Suit::Clubs),
                    card(Rank::Three, Suit::Diamonds),
                ])
                .unwrap(),
            ),
        ];
        let context = context_with_unseen(vec![card(Rank::King, Suit::Clubs)]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let chosen = CardCounter.choose_play(
            &legal_moves,
            DuplicateRule::FirstDealtWins,
            &context,
            &mut rng,
        );
        assert_eq!(
            chosen,
            Move::Play(Combo::new(vec![card(Rank::King, Suit::Hearts)]).unwrap())
        );
    }

    #[test]
    fn leading_with_no_precious_option_picks_lowest_like_lowest_legal() {
        let legal_moves = vec![
            Move::Play(Combo::new(vec![card(Rank::Six, Suit::Clubs)]).unwrap()),
            Move::Play(Combo::new(vec![card(Rank::Nine, Suit::Clubs)]).unwrap()),
        ];
        let context = context_with_unseen(vec![card(Rank::King, Suit::Hearts)]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let chosen = CardCounter.choose_play(
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
    fn leading_when_everything_is_precious_still_plays_the_lowest() {
        let legal_moves = vec![
            Move::Play(Combo::new(vec![card(Rank::King, Suit::Clubs)]).unwrap()),
            Move::Play(Combo::new(vec![card(Rank::Ace, Suit::Clubs)]).unwrap()),
        ];
        let context = context_with_unseen(vec![]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let chosen = CardCounter.choose_play(
            &legal_moves,
            DuplicateRule::FirstDealtWins,
            &context,
            &mut rng,
        );
        assert_eq!(
            chosen,
            Move::Play(Combo::new(vec![card(Rank::King, Suit::Clubs)]).unwrap())
        );
    }

    #[test]
    fn following_plays_the_only_legal_beater_even_when_it_is_precious() {
        let legal_moves = vec![
            Move::Pass,
            Move::Play(Combo::new(vec![card(Rank::King, Suit::Clubs)]).unwrap()),
        ];
        let context = context_with_unseen(vec![]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let chosen = CardCounter.choose_play(
            &legal_moves,
            DuplicateRule::FirstDealtWins,
            &context,
            &mut rng,
        );
        assert_eq!(
            chosen,
            Move::Play(Combo::new(vec![card(Rank::King, Suit::Clubs)]).unwrap())
        );
    }
}
