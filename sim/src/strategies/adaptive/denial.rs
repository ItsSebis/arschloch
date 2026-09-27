//! Endgame denial, in both of `Adaptive`'s modes.
//!
//! `HandSize` reproduces `EndgameDenial`'s exact trigger and response
//! (any active opponent's `hand_size` <= `close` -> full `GreedyHighest`
//! push, else caller falls through to base selection) by direct
//! delegation — no logic duplicated, no behavior drift, so
//! `Adaptive(denial)` stays an honest baseline comparison against
//! `EndgameDenial` itself.
//!
//! `HandReading` asks a narrower question than a blanket push: which
//! legal plays can a close-to-finishing opponent *provably* not beat,
//! using their unrefuted pass ceilings plus two other sound public
//! facts (the combo is bigger than their whole hand; no unseen card
//! beats it at all)? Taking the *lowest* such play denies the threat as
//! surely as a highest-card push while spending less — the working
//! theory (docs/ROADMAP.md, Phase 7) is that `EndgameDenial` loses to
//! `LowestLegal` precisely because it spends high cards it doesn't
//! need to. When following and the table combo is already provably
//! safe against every threat, every legal beater qualifies, so this
//! reduces to `LowestLegal`'s own choice: aggression relaxes exactly
//! when it stops being necessary. Only falls back to a full push
//! (`GreedyHighest`) when nothing is provably safe.

// `respond` and its private helpers below aren't called from anywhere
// in the crate yet — only from this module's own tests — until a
// later task (`Adaptive::choose_play`, docs/ROADMAP.md Phase 7) wires
// this modifier in alongside card-counting and deception. Remove this
// allow once that call site lands.
#![allow(dead_code)]

use std::cmp::Ordering;

use engine::{Card, DuplicateRule, Move};

use crate::strategies::GreedyHighest;
use crate::strategy::{Strategy, TurnContext};

use super::config::DenialMode;

/// Whether any still-active opponent is at or below `close` cards —
/// `EndgameDenial`'s own trigger, reused verbatim by `HandSize`.
fn any_threat(context: &TurnContext<'_>, close: usize) -> bool {
    context
        .opponents
        .iter()
        .any(|o| o.active && o.hand_size <= close)
}

/// The result of asking `targeted_denial` whether some legal play can
/// provably deny every close-to-finishing opponent.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Denial {
    /// No opponent is close enough to finishing to be worth denying.
    NoThreat,
    /// This play is provably safe against every threat — the cheapest
    /// such play, specifically, since callers want to spend the least.
    Lock(Move),
    /// At least one opponent is close to finishing, but no legal play
    /// is provably safe against all of them; caller should fall back to
    /// a full push.
    Unproven,
}

/// Finds the cheapest legal play that provably denies every
/// close-to-finishing opponent, using only sound public facts: an
/// opponent's unrefuted pass ceilings (`PassCeilings::cannot_beat`), the
/// fact that a combo bigger than an opponent's whole hand can never be
/// matched by them, and the fact that no unseen card beats this play's
/// top card at all (so literally nobody, threat or not, can beat it).
/// Returns `Denial::Unproven` when no legal play clears that bar, even
/// though at least one threat exists.
fn targeted_denial(
    legal_moves: &[Move],
    duplicate_rule: DuplicateRule,
    context: &TurnContext<'_>,
    close: usize,
) -> Denial {
    let threats: Vec<_> = context
        .opponents
        .iter()
        .filter(|o| o.active && o.hand_size <= close)
        .collect();
    if threats.is_empty() {
        return Denial::NoThreat;
    }

    let highest_unseen = context
        .unseen_cards
        .iter()
        .copied()
        .max_by(|a, b| a.compare(b, duplicate_rule));
    let locks_out_every_threat = |size: usize, top: Card| {
        highest_unseen.is_none_or(|h| top.compare(&h, duplicate_rule) != Ordering::Less)
            || threats.iter().all(|t| {
                size > t.hand_size || t.pass_ceilings.cannot_beat(size, top, duplicate_rule)
            })
    };

    legal_moves
        .iter()
        .filter_map(|mv| match mv {
            Move::Play(combo) => Some((combo.size(), combo.top_card(duplicate_rule), mv)),
            Move::Pass => None,
        })
        .filter(|&(size, top, _)| locks_out_every_threat(size, top))
        .min_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.compare(&b.1, duplicate_rule))
        })
        .map_or(Denial::Unproven, |(_, _, mv)| Denial::Lock(mv.clone()))
}

/// `Adaptive`'s endgame-denial modifier, for either `DenialMode`.
/// Returns `None` when denial doesn't apply (mode is `Off`, or no
/// opponent is close enough to trigger it) so the caller falls through
/// to its own base selection (`LowestLegal`/`CardCounter`) — denial
/// never itself decides to pass or conserve, it only decides when to
/// override that base choice.
pub(super) fn respond(
    mode: DenialMode,
    legal_moves: &[Move],
    duplicate_rule: DuplicateRule,
    context: &TurnContext<'_>,
    rng: &mut dyn rand::Rng,
) -> Option<Move> {
    match mode {
        DenialMode::Off => None,
        DenialMode::HandSize { close } => {
            if any_threat(context, close) {
                Some(GreedyHighest.choose_play(legal_moves, duplicate_rule, context, rng))
            } else {
                None
            }
        }
        DenialMode::HandReading { close } => {
            match targeted_denial(legal_moves, duplicate_rule, context, close) {
                Denial::NoThreat => None,
                Denial::Lock(mv) => Some(mv),
                Denial::Unproven => {
                    Some(GreedyHighest.choose_play(legal_moves, duplicate_rule, context, rng))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::{Combo, Rank, Suit};
    use rand::SeedableRng;

    use crate::hand_reading::PassCeilings;
    use crate::strategy::OpponentHand;

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
                let cards = vec![top; size]; // same-rank filler for size >1; size-1 tests only need one card
                (0u8, Combo::new(cards).unwrap(), 0usize)
            })
            .collect();
        crate::hand_reading::read_pass_ceilings(
            2,
            &[],
            &pass_history,
            DuplicateRule::FirstDealtWins,
        )[0]
    }

    fn context_with_threat(
        hand_size: usize,
        pass_ceilings: PassCeilings,
        unseen: Vec<engine::Card>,
    ) -> TurnContext<'static> {
        TurnContext {
            seat: 0,
            hand: &[],
            opponents: vec![OpponentHand {
                seat: 1,
                hand_size,
                active: true,
                pass_ceilings,
            }],
            unseen_cards: unseen,
            own_pass_ceilings: PassCeilings::default(),
            current_combo: None,
        }
    }

    #[test]
    fn a_reads_the_ceiling_and_locks_with_the_cheaper_card() {
        let ceilings = ceilings_from(&[(1, card(Rank::Jack, Suit::Spades))]);
        let context = context_with_threat(
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
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let reading = respond(
            DenialMode::HandReading { close: 2 },
            &legal,
            DuplicateRule::FirstDealtWins,
            &context,
            &mut rng,
        );
        assert_eq!(reading, Some(play(vec![card(Rank::King, Suit::Hearts)])));

        let hand_size_mode = respond(
            DenialMode::HandSize { close: 2 },
            &legal,
            DuplicateRule::FirstDealtWins,
            &context,
            &mut rng,
        );
        assert_eq!(
            hand_size_mode,
            Some(play(vec![card(Rank::Ace, Suit::Clubs)])),
            "HandSize must reproduce EndgameDenial's own GreedyHighest push, diverging from HandReading here"
        );
    }

    #[test]
    fn b_no_pass_history_falls_back_to_the_only_unbeatable_play() {
        let context = context_with_threat(
            2,
            PassCeilings::default(),
            vec![card(Rank::Ace, Suit::Diamonds)],
        );
        let legal = vec![
            Move::Pass,
            play(vec![card(Rank::King, Suit::Hearts)]),
            play(vec![card(Rank::Ace, Suit::Clubs)]),
        ];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let result = respond(
            DenialMode::HandReading { close: 2 },
            &legal,
            DuplicateRule::FirstDealtWins,
            &context,
            &mut rng,
        );
        assert_eq!(result, Some(play(vec![card(Rank::Ace, Suit::Clubs)])));
    }

    #[test]
    fn c_suit_precision_in_the_lock_check() {
        let ceilings = ceilings_from(&[(1, card(Rank::King, Suit::Hearts))]);
        let context = context_with_threat(1, ceilings, vec![card(Rank::Ace, Suit::Diamonds)]);
        let legal = vec![
            play(vec![card(Rank::Five, Suit::Diamonds)]),
            play(vec![card(Rank::King, Suit::Diamonds)]),
            play(vec![card(Rank::King, Suit::Spades)]),
        ];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let result = respond(
            DenialMode::HandReading { close: 2 },
            &legal,
            DuplicateRule::FirstDealtWins,
            &context,
            &mut rng,
        );
        assert_eq!(
            result,
            Some(play(vec![card(Rank::King, Suit::Spades)])),
            "King of Diamonds must NOT be treated as locking-out (ceiling is King of Hearts, \
             and Diamonds < Hearts in this game's suit order, so King-Diamonds does not beat it)"
        );
    }

    #[test]
    fn d_size_lockout_with_no_pass_history() {
        let context = context_with_threat(
            1,
            PassCeilings::default(),
            vec![card(Rank::Ace, Suit::Diamonds)],
        );
        let legal = vec![
            play(vec![card(Rank::Four, Suit::Diamonds)]),
            play(vec![
                card(Rank::Six, Suit::Clubs),
                card(Rank::Six, Suit::Diamonds),
            ]),
            play(vec![
                card(Rank::Jack, Suit::Clubs),
                card(Rank::Jack, Suit::Diamonds),
            ]),
        ];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let result = respond(
            DenialMode::HandReading { close: 2 },
            &legal,
            DuplicateRule::FirstDealtWins,
            &context,
            &mut rng,
        );
        assert_eq!(
            result,
            Some(play(vec![
                card(Rank::Six, Suit::Clubs),
                card(Rank::Six, Suit::Diamonds)
            ])),
            "threat has only 1 card, so it can't field any size-2 combo at all — \
             the lowest size-2 play already locks it out"
        );
    }

    #[test]
    fn no_active_threat_falls_through_to_the_caller() {
        let context = context_with_threat(10, PassCeilings::default(), vec![]);
        let legal = vec![play(vec![card(Rank::Six, Suit::Clubs)])];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        assert_eq!(
            respond(
                DenialMode::HandReading { close: 2 },
                &legal,
                DuplicateRule::FirstDealtWins,
                &context,
                &mut rng
            ),
            None
        );
        assert_eq!(
            respond(
                DenialMode::HandSize { close: 2 },
                &legal,
                DuplicateRule::FirstDealtWins,
                &context,
                &mut rng
            ),
            None
        );
    }

    #[test]
    fn denial_off_always_returns_none() {
        let context = context_with_threat(1, PassCeilings::default(), vec![]);
        let legal = vec![play(vec![card(Rank::Six, Suit::Clubs)])];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        assert_eq!(
            respond(
                DenialMode::Off,
                &legal,
                DuplicateRule::FirstDealtWins,
                &context,
                &mut rng
            ),
            None
        );
    }
}
