//! Deception: occasionally passes while holding a beater, to plant a
//! false pass ceiling in an opponent's hand reading (`crate::hand_reading`).
//!
//! Bluffs only when *exactly one* rank in hand beats the table combo.
//! The lie ("nothing above this at this size") then survives until
//! that rank is finally played — which a LowestLegal-ordered hand does
//! last, exactly when a hand-reading opponent might otherwise choose a
//! `Lock` against this seat (see `denial::targeted_denial`). A
//! low-rank bluff would be refuted by the very next higher single this
//! seat plays and is worthless; this is why the trigger requires the
//! bluff to be about this seat's *only* beating rank, not any rank.
//!
//! Never bluffs while any active opponent is itself close to finishing
//! (denying is more valuable than deceiving), with a short hand (the
//! lie needs a future round-portion to pay off in), or when the public
//! ceiling already covers this exact combo (a redundant bluff costs a
//! tempo for no new misinformation).
//!
//! **Honest trade-off, stated plainly**: every bluff gives up a trick
//! this seat could have won — the same "hoard for safety, pay with
//! speed" trade this project's empirical batch run already showed
//! losing for `HoldBackPairs`/`CardCounter`. The payoff exists only
//! against opponents actually doing pass-based hand-reading, later in
//! the same match. Expect neutral-to-negative net placement for the
//! deceiver in tables without hand-reading opponents; the batch stats
//! decide whether it's worth it in mixed tables.

// `should_bluff_pass` isn't called from anywhere in the crate yet —
// only from this module's own tests — until a later task
// (`Adaptive::choose_play`, docs/ROADMAP.md Phase 7) wires this
// modifier in alongside card-counting and denial. Remove this allow
// once that call site lands.
#![allow(dead_code)]

use rand::RngExt;

use engine::{DuplicateRule, Move};

use crate::strategy::TurnContext;

/// No bluffing below this many cards: the lie needs a future to pay off in.
const MIN_HAND_TO_BLUFF: usize = 5;

pub(super) fn should_bluff_pass(
    rate: f64,
    legal_moves: &[Move],
    duplicate_rule: DuplicateRule,
    context: &TurnContext<'_>,
    rng: &mut dyn rand::Rng,
) -> bool {
    if rate <= 0.0 {
        return false;
    }
    let Some(current) = context.current_combo else {
        return false; // leading: Pass isn't legal
    };
    let pressing = context
        .opponents
        .iter()
        .any(|o| o.active && o.hand_size <= super::config::DEFAULT_CLOSE);
    if pressing || context.hand.len() < MIN_HAND_TO_BLUFF {
        return false;
    }

    let mut beating_ranks = legal_moves.iter().filter_map(|mv| match mv {
        Move::Play(combo) => Some(combo.top_card(duplicate_rule).rank),
        Move::Pass => None,
    });
    let Some(only_rank) = beating_ranks.next() else {
        return false; // forced pass already: nothing to lie about
    };
    if beating_ranks.any(|rank| rank != only_rank) {
        return false; // more than one beating rank: a lie here is cheaply refuted
    }

    let passed_top = current.top_card(duplicate_rule);
    if context
        .own_pass_ceilings
        .cannot_beat(current.size(), passed_top, duplicate_rule)
    {
        return false; // public read is already at or below this: redundant
    }
    rng.random_bool(rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::{Combo, Rank, Suit};
    use rand::SeedableRng;

    use crate::hand_reading::{read_pass_ceilings, PassCeilings};
    use crate::strategy::OpponentHand;

    fn card(rank: Rank, suit: Suit) -> engine::Card {
        engine::Card::new(rank, suit, 0)
    }
    fn play(cards: Vec<engine::Card>) -> Move {
        Move::Play(Combo::new(cards).unwrap())
    }
    fn ten_diamonds() -> Combo {
        Combo::new(vec![card(Rank::Ten, Suit::Diamonds)]).unwrap()
    }

    /// A 6-card hand with exactly one card (the King) that beats a
    /// Ten — shared by every test below via `qualifying_context`.
    fn six_card_hand() -> [engine::Card; 6] {
        [
            card(Rank::Two, Suit::Clubs),
            card(Rank::Three, Suit::Clubs),
            card(Rank::Four, Suit::Clubs),
            card(Rank::Five, Suit::Clubs),
            card(Rank::King, Suit::Clubs),
            card(Rank::Six, Suit::Clubs),
        ]
    }

    /// `combo`: the combo currently on the table (this seat is
    /// following it). `hand`: this seat's hand (see `six_card_hand`),
    /// threaded through by reference (rather than built inline here)
    /// because `engine::Card::new` isn't `const fn` — an inline array
    /// literal of non-constant values can't be promoted to the
    /// `'static` lifetime `TurnContext<'_>` would otherwise need.
    /// Takes both by reference so callers can vary
    /// `current_combo`/`own_pass_ceilings` independently per test.
    fn qualifying_context<'a>(
        combo: &'a Combo,
        hand: &'a [engine::Card],
        own_pass_ceilings: PassCeilings,
    ) -> TurnContext<'a> {
        TurnContext {
            seat: 0,
            hand,
            opponents: vec![OpponentHand {
                seat: 1,
                hand_size: 10,
                active: true,
                pass_ceilings: PassCeilings::default(),
            }],
            unseen_cards: vec![],
            own_pass_ceilings,
            current_combo: Some(combo),
        }
    }

    #[test]
    fn one_never_bluffs_while_pressing() {
        let table = ten_diamonds();
        let hand = six_card_hand();
        let mut context = qualifying_context(&table, &hand, PassCeilings::default());
        context.opponents = vec![OpponentHand {
            seat: 1,
            hand_size: 2,
            active: true,
            pass_ceilings: PassCeilings::default(),
        }];
        let legal = vec![Move::Pass, play(vec![card(Rank::King, Suit::Clubs)])];
        for seed in 0..500u64 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            assert!(!should_bluff_pass(
                0.9,
                &legal,
                DuplicateRule::FirstDealtWins,
                &context,
                &mut rng
            ));
        }
    }

    #[test]
    fn two_never_bluffs_when_more_than_one_beating_rank_exists() {
        let table = ten_diamonds();
        let hand = six_card_hand();
        let context = qualifying_context(&table, &hand, PassCeilings::default());
        let legal = vec![
            Move::Pass,
            play(vec![card(Rank::Jack, Suit::Clubs)]),
            play(vec![card(Rank::King, Suit::Clubs)]),
        ];
        for seed in 0..500u64 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            assert!(!should_bluff_pass(
                0.9,
                &legal,
                DuplicateRule::FirstDealtWins,
                &context,
                &mut rng
            ));
        }
    }

    #[test]
    fn three_never_bluffs_when_forced_or_leading_or_redundant() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let table = ten_diamonds();
        let hand = six_card_hand();

        // forced: legal_moves is Pass only
        let forced_ctx = qualifying_context(&table, &hand, PassCeilings::default());
        assert!(!should_bluff_pass(
            0.9,
            &[Move::Pass],
            DuplicateRule::FirstDealtWins,
            &forced_ctx,
            &mut rng
        ));

        // leading: no current combo
        let mut leading_ctx = qualifying_context(&table, &hand, PassCeilings::default());
        leading_ctx.current_combo = None;
        let legal = vec![play(vec![card(Rank::King, Suit::Clubs)])];
        assert!(!should_bluff_pass(
            0.9,
            &legal,
            DuplicateRule::FirstDealtWins,
            &leading_ctx,
            &mut rng
        ));

        // redundant: own_pass_ceilings already covers this exact combo
        // — build it the same way `read_pass_ceilings` would from a
        // real prior pass against a Ten at size 1.
        let prior_pass = vec![(0u8, ten_diamonds(), 0usize)];
        let own_ceilings =
            read_pass_ceilings(2, &[], &prior_pass, DuplicateRule::FirstDealtWins)[0];
        let redundant_ctx = qualifying_context(&table, &hand, own_ceilings);
        let legal = vec![Move::Pass, play(vec![card(Rank::King, Suit::Clubs)])];
        assert!(!should_bluff_pass(
            0.9,
            &legal,
            DuplicateRule::FirstDealtWins,
            &redundant_ctx,
            &mut rng
        ));
    }

    #[test]
    fn four_fires_at_roughly_the_configured_rate() {
        let table = ten_diamonds();
        let hand = six_card_hand();
        let context = qualifying_context(&table, &hand, PassCeilings::default());
        let legal = vec![Move::Pass, play(vec![card(Rank::King, Suit::Clubs)])];
        let fires = (0..1000u64)
            .filter(|&seed| {
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                should_bluff_pass(
                    0.25,
                    &legal,
                    DuplicateRule::FirstDealtWins,
                    &context,
                    &mut rng,
                )
            })
            .count();
        // Statistical, not exact: rate=0.25 over 1000 draws lands in
        // this band with overwhelming probability. If this ever
        // flakes, widen the band — don't chase exact determinism on a
        // Bernoulli draw.
        assert!(
            (150..=350).contains(&fires),
            "expected roughly 250 fires out of 1000, got {fires}"
        );
    }

    // No standalone "rate <= 0.0 never draws from rng" unit test here
    // — writing a custom panic-on-any-use `Rng` impl against rand
    // 0.10.3's redesigned trait hierarchy (`Rng: TryRng<Error =
    // Infallible>`, quite different from the simpler `RngCore` of
    // older rand versions) isn't worth the complexity. The real,
    // sufficient proof is Task 7's
    // `adaptive_none_matches_lowest_legal_exactly` equivalence test: a
    // single spurious draw here would advance the shared match-wide
    // `rng`'s state and desynchronize every later decision in that
    // match, which that test's byte-for-byte `MatchResult` comparison
    // would catch immediately.
}
