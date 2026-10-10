//! Trick-lead tempo.
//!
//! Finishing is a race to be the one who *leads* a trick while holding
//! exactly one card: leading your last card finishes you unconditionally
//! (nothing needs to fail to beat it in time), while being stuck
//! *responding* with your last card means someone else already claimed
//! that position. A strategy that always spends the cheapest legal
//! beater (`LowestLegal`'s own instinct) can lose a trick's lead to an
//! opponent who was willing to spend a bigger-than-strictly-necessary
//! card to cut off every possible re-escalation this trick — see the
//! worked example in the session that motivated this (two legal beaters
//! of a led card; spending the higher one left nothing for anyone else
//! to escalate past, while spending the lower one let two opponents
//! chain-escalate through it, one of them finishing on that chain).
//!
//! This modifier reuses `safety::cheapest_universally_safe_play` — the
//! same proof `denial` uses — but triggers on this seat's *own* hand
//! size instead of an opponent's: once this seat is down to
//! `TEMPO_CLOSE` cards or fewer, prefer the cheapest legal play that's
//! provably safe against the whole active field (nobody can re-escalate
//! past it), instead of whatever the base strategy's own cheapest-legal
//! instinct would pick.
//!
//! **Empirically validated** (see the tempo entry in docs/ROADMAP.md): `TEMPO_CLOSE =
//! 2` is a sharp sweet spot, not an arbitrary round number. Across four
//! independent seed bases and all four supported table sizes, `close=2`
//! gave a consistent ~4-6% relative edge over plain `LowestLegal` in a
//! clean head-to-head, with monotonic dominance across *every* role
//! (higher President/Vize/Offizier rates, lower Dummkopf/ViceArschloch/
//! Arschloch rates) — the first modifier tested in this project to beat
//! `LowestLegal` outright rather than trade placement for safety.
//! `close=1` was statistically a wash (too narrow to matter); `close=3`
//! and up actively lost ground (triggering the override too early meant
//! spending more than the base strategy would with no proof to show for
//! it). This asymmetric, narrow sweet spot is why `TEMPO_CLOSE` is a
//! fixed constant rather than a configurable `tempo=<n>` option: unlike
//! `denial`'s `close` (which degrades gracefully as it's tuned), this
//! threshold's good region doesn't extend safely in either direction,
//! so exposing it as a user-tunable number would be a footgun rather
//! than a knob.
//!
//! Also empirically validated: when no play is provably safe, deferring
//! to the base strategy's own choice beats guessing with a
//! `GreedyHighest`-style push (the fallback `denial` uses) — pushing
//! blind without proof costs more than it earns here, so `respond`
//! simply returns `None` in that case and lets the caller's existing
//! base-selection step handle it; this also means `respond` never needs
//! an `rng` parameter, since neither branch ever draws from one.

use engine::{DuplicateRule, Move};

use crate::strategy::TurnContext;

use super::safety::cheapest_universally_safe_play;

/// Below this many remaining cards in *this seat's own* hand, seizing
/// the next trick's lead is worth spending more than the base
/// strategy's own cheapest-legal instinct would. See the module doc
/// comment for why this is a fixed constant, not a configurable option.
const TEMPO_CLOSE: usize = 2;

/// `Adaptive`'s trick-lead-tempo modifier. Returns `None` when
/// `enabled` is false, this seat's own hand is still above
/// `TEMPO_CLOSE` cards, or no legal play is provably safe against the
/// whole active field — in every such case the caller falls through to
/// its own base selection unchanged.
pub(super) fn respond(
    enabled: bool,
    legal_moves: &[Move],
    duplicate_rule: DuplicateRule,
    context: &TurnContext<'_>,
) -> Option<Move> {
    if !enabled || context.hand.len() > TEMPO_CLOSE {
        return None;
    }
    cheapest_universally_safe_play(legal_moves, duplicate_rule, context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand_reading::{read_pass_ceilings, PassCeilings};
    use crate::strategy::OpponentHand;
    use engine::{Combo, Rank, Suit};

    fn card(rank: Rank, suit: Suit) -> engine::Card {
        engine::Card::new(rank, suit, 0)
    }

    fn play(cards: Vec<engine::Card>) -> Move {
        Move::Play(Combo::new(cards).unwrap())
    }

    fn ceilings_from(passes: &[(usize, engine::Card)]) -> PassCeilings {
        let pass_history: Vec<_> = passes
            .iter()
            .map(|&(size, top)| {
                let cards = vec![top; size];
                (0u8, Combo::new(cards).unwrap(), 0usize)
            })
            .collect();
        read_pass_ceilings(2, &[], &pass_history, DuplicateRule::FirstDealtWins)[0]
    }

    fn context_with_hand_and_opponent(
        hand: &[engine::Card],
        opponent_hand_size: usize,
        opponent_ceilings: PassCeilings,
        unseen: Vec<engine::Card>,
    ) -> TurnContext<'_> {
        TurnContext {
            seat: 0,
            hand,
            opponents: vec![OpponentHand {
                seat: 1,
                hand_size: opponent_hand_size,
                active: true,
                pass_ceilings: opponent_ceilings,
            }],
            unseen_cards: unseen,
            own_pass_ceilings: PassCeilings::default(),
            current_combo: None,
        }
    }

    #[test]
    fn disabled_never_fires() {
        let context = context_with_hand_and_opponent(
            &[],
            2,
            PassCeilings::default(),
            vec![card(Rank::Ace, Suit::Diamonds)],
        );
        let legal = vec![play(vec![card(Rank::King, Suit::Hearts)])];
        assert_eq!(
            respond(false, &legal, DuplicateRule::FirstDealtWins, &context),
            None
        );
    }

    #[test]
    fn never_fires_above_the_threshold() {
        // Three cards in hand is above TEMPO_CLOSE (2).
        let hand = vec![
            card(Rank::Four, Suit::Spades),
            card(Rank::Ten, Suit::Spades),
            card(Rank::King, Suit::Hearts),
        ];
        let context = context_with_hand_and_opponent(&hand, 2, PassCeilings::default(), vec![]);
        let legal = vec![play(vec![card(Rank::King, Suit::Hearts)])];
        assert_eq!(
            respond(true, &legal, DuplicateRule::FirstDealtWins, &context),
            None
        );
    }

    #[test]
    fn fires_at_exactly_the_threshold_and_picks_the_cheapest_safe_play() {
        let hand = vec![
            card(Rank::Four, Suit::Spades),
            card(Rank::Ten, Suit::Spades),
        ];
        let ceilings = ceilings_from(&[(1, card(Rank::Jack, Suit::Spades))]);
        let context = context_with_hand_and_opponent(
            &hand,
            2,
            ceilings,
            vec![
                card(Rank::Queen, Suit::Diamonds),
                card(Rank::Ace, Suit::Diamonds),
            ],
        );
        let legal = vec![
            Move::Pass,
            play(vec![card(Rank::King, Suit::Hearts)]),
            play(vec![card(Rank::Ace, Suit::Clubs)]),
        ];
        assert_eq!(
            respond(true, &legal, DuplicateRule::FirstDealtWins, &context),
            Some(play(vec![card(Rank::King, Suit::Hearts)]))
        );
    }

    #[test]
    fn fires_at_one_card_too() {
        let hand = vec![card(Rank::Ten, Suit::Spades)];
        let context = context_with_hand_and_opponent(
            &hand,
            6,
            PassCeilings::default(),
            vec![card(Rank::Four, Suit::Diamonds)],
        );
        let legal = vec![play(vec![card(Rank::Ten, Suit::Spades)])];
        assert_eq!(
            respond(true, &legal, DuplicateRule::FirstDealtWins, &context),
            Some(play(vec![card(Rank::Ten, Suit::Spades)]))
        );
    }

    #[test]
    fn defers_to_base_when_nothing_is_provably_safe() {
        let hand = vec![
            card(Rank::Four, Suit::Spades),
            card(Rank::King, Suit::Hearts),
        ];
        let context = context_with_hand_and_opponent(
            &hand,
            6,
            PassCeilings::default(),
            vec![card(Rank::Ace, Suit::Diamonds)],
        );
        let legal = vec![Move::Pass, play(vec![card(Rank::King, Suit::Hearts)])];
        assert_eq!(
            respond(true, &legal, DuplicateRule::FirstDealtWins, &context),
            None,
            "no legal play beats the unseen Ace, so this must defer rather than guess"
        );
    }
}
