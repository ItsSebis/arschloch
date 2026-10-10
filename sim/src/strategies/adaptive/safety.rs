//! The shared proof both `denial` and `tempo` build on: which legal
//! play, if any, is provably unbeatable by every still-active opponent
//! right now (docs/ROADMAP.md, Phase 7 and Phase 8).
//!
//! The two modifiers ask this for different reasons — `denial` to deny
//! an opponent close to finishing, `tempo` to seize the next trick's
//! lead while this seat itself is close to finishing — but the proof
//! itself doesn't care which seat prompted the question, only which
//! legal plays are safe against the whole active field: an opponent's
//! unrefuted pass ceilings (`PassCeilings::cannot_beat`), the fact that
//! a combo bigger than an opponent's whole hand can never be matched by
//! them, and the fact that no unseen card beats this play's top card at
//! all. Checking only *some* opponents (e.g. only the one being denied,
//! or only the one judged "close") is unsound in practice: any other
//! active opponent left unchecked could hold an unproven beater, raise
//! over the "safe" play, and hand the original target a fresh combo
//! none of this reasoning covered — see `denial`'s own doc comment for
//! the fuller account of why this bit `denial` once already.

use std::cmp::Ordering;

use engine::{Card, DuplicateRule, Move};

use crate::strategy::TurnContext;

/// The cheapest legal play that's provably unbeatable by every
/// still-active opponent, or `None` if no legal play clears that bar.
/// "Cheapest" is `LowestLegal`'s own ordering (smallest size, then
/// lowest top card) — among several safe plays, callers want to spend
/// the least.
pub(super) fn cheapest_universally_safe_play(
    legal_moves: &[Move],
    duplicate_rule: DuplicateRule,
    context: &TurnContext<'_>,
) -> Option<Move> {
    let active_opponents: Vec<_> = context.opponents.iter().filter(|o| o.active).collect();
    let highest_unseen = context
        .unseen_cards
        .iter()
        .copied()
        .max_by(|a, b| a.compare(b, duplicate_rule));
    let locks_out_every_opponent = |size: usize, top: Card| {
        highest_unseen.is_none_or(|h| top.compare(&h, duplicate_rule) != Ordering::Less)
            || active_opponents.iter().all(|o| {
                size > o.hand_size || o.pass_ceilings.cannot_beat(size, top, duplicate_rule)
            })
    };

    legal_moves
        .iter()
        .filter_map(|mv| match mv {
            Move::Play(combo) => Some((combo.size(), combo.top_card(duplicate_rule), mv)),
            Move::Pass => None,
        })
        .filter(|&(size, top, _)| locks_out_every_opponent(size, top))
        .min_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.compare(&b.1, duplicate_rule))
        })
        .map(|(_, _, mv)| *mv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand_reading::{read_pass_ceilings, PassCeilings};
    use crate::strategy::OpponentHand;
    use engine::{Combo, Rank, Suit};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit, 0)
    }

    fn play(cards: Vec<Card>) -> Move {
        Move::Play(Combo::new(cards).unwrap())
    }

    fn ceilings_from(passes: &[(usize, Card)]) -> PassCeilings {
        let pass_history: Vec<_> = passes
            .iter()
            .map(|&(size, top)| {
                let cards = vec![top; size];
                (0u8, Combo::new(cards).unwrap(), 0usize)
            })
            .collect();
        read_pass_ceilings(2, &[], &pass_history, DuplicateRule::FirstDealtWins)[0]
    }

    #[test]
    fn finds_the_cheapest_play_safe_against_a_single_opponent() {
        let ceilings = ceilings_from(&[(1, card(Rank::Jack, Suit::Spades))]);
        let context = TurnContext {
            seat: 0,
            hand: &[],
            opponents: vec![OpponentHand {
                seat: 1,
                hand_size: 2,
                active: true,
                pass_ceilings: ceilings,
            }],
            unseen_cards: vec![
                card(Rank::Queen, Suit::Diamonds),
                card(Rank::Ace, Suit::Diamonds),
            ],
            own_pass_ceilings: PassCeilings::default(),
            current_combo: None,
        };
        let legal = vec![
            Move::Pass,
            play(vec![card(Rank::King, Suit::Hearts)]),
            play(vec![card(Rank::Ace, Suit::Clubs)]),
        ];
        assert_eq!(
            cheapest_universally_safe_play(&legal, DuplicateRule::FirstDealtWins, &context),
            Some(play(vec![card(Rank::King, Suit::Hearts)]))
        );
    }

    #[test]
    fn a_play_safe_against_only_some_opponents_does_not_qualify() {
        // King of Hearts is safe against the pass-read opponent (seat 1)
        // but seat 2 has no recorded ceiling and could hold the unseen
        // Ace of Diamonds — so only Ace of Clubs (beats every unseen
        // card outright) qualifies.
        let ceilings = ceilings_from(&[(1, card(Rank::Jack, Suit::Spades))]);
        let context = TurnContext {
            seat: 0,
            hand: &[],
            opponents: vec![
                OpponentHand {
                    seat: 1,
                    hand_size: 2,
                    active: true,
                    pass_ceilings: ceilings,
                },
                OpponentHand {
                    seat: 2,
                    hand_size: 6,
                    active: true,
                    pass_ceilings: PassCeilings::default(),
                },
            ],
            unseen_cards: vec![
                card(Rank::Queen, Suit::Diamonds),
                card(Rank::Ace, Suit::Diamonds),
            ],
            own_pass_ceilings: PassCeilings::default(),
            current_combo: None,
        };
        let legal = vec![
            Move::Pass,
            play(vec![card(Rank::King, Suit::Hearts)]),
            play(vec![card(Rank::Ace, Suit::Clubs)]),
        ];
        assert_eq!(
            cheapest_universally_safe_play(&legal, DuplicateRule::FirstDealtWins, &context),
            Some(play(vec![card(Rank::Ace, Suit::Clubs)]))
        );
    }

    #[test]
    fn none_when_nothing_is_provably_safe() {
        let context = TurnContext {
            seat: 0,
            hand: &[],
            opponents: vec![OpponentHand {
                seat: 1,
                hand_size: 6,
                active: true,
                pass_ceilings: PassCeilings::default(),
            }],
            unseen_cards: vec![card(Rank::Ace, Suit::Diamonds)],
            own_pass_ceilings: PassCeilings::default(),
            current_combo: None,
        };
        let legal = vec![Move::Pass, play(vec![card(Rank::King, Suit::Hearts)])];
        assert_eq!(
            cheapest_universally_safe_play(&legal, DuplicateRule::FirstDealtWins, &context),
            None
        );
    }

    #[test]
    fn inactive_opponents_are_never_checked() {
        let context = TurnContext {
            seat: 0,
            hand: &[],
            opponents: vec![OpponentHand {
                seat: 1,
                hand_size: 0,
                active: false,
                pass_ceilings: PassCeilings::default(),
            }],
            unseen_cards: vec![],
            own_pass_ceilings: PassCeilings::default(),
            current_combo: None,
        };
        let legal = vec![play(vec![card(Rank::Four, Suit::Spades)])];
        assert_eq!(
            cheapest_universally_safe_play(&legal, DuplicateRule::FirstDealtWins, &context),
            Some(play(vec![card(Rank::Four, Suit::Spades)]))
        );
    }
}
