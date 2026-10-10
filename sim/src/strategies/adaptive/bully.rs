//! Lead-order bullying (docs/ROADMAP.md, Phase 9): while leading, with
//! this hand shaped mostly as same-rank groups, lead the cheapest
//! *whole* same-rank group before ever leading a single — banking every
//! leftover single for the very end of the hand.
//!
//! This is a different mechanism from `tempo`: `tempo` only engages once
//! *this seat's own* hand is down to `TEMPO_CLOSE` cards or fewer, and
//! picks the cheapest play *provably* safe against the whole active
//! field. Bullying can fire much earlier (a 7-card hand is a typical
//! trigger), and is a heuristic bet, not a proof — this seat can never
//! see an opponent's rank-level hand contents (`TurnContext` only ever
//! exposes `OpponentHand::hand_size`/`active`/`pass_ceilings`), so there
//! is no way to confirm any opponent actually lacks a matching combo at
//! the led rank. The bet is just that *some* opponent is likely to lack
//! one regardless of hand size (a same-rank set of 2+ is comparatively
//! rare to hold at any rank in a well-shuffled hand), so leading groups
//! low-to-high before ever touching a single forces repeated passes and
//! keeps the lead cycling back to this seat.
//!
//! The payoff is the same unconditional-finish mechanic `tempo` relies
//! on: playing your very last card locks in your placement immediately,
//! regardless of whether a later play could have beaten it. A hand
//! shaped "mostly same-rank groups, plus one or a few leftover singles"
//! can turn that mechanic into a free win *if* the groups are led first
//! and the singles are saved for last — leading a vulnerable single
//! early, while groups remain unplayed, risks losing the lead (and with
//! it, every one of those groups' usefulness as a "forced pass" lever)
//! to an opponent who happens to beat it, exactly as validated in this
//! project's worked example (see this module's tests).
//!
//! **Empirical correction from the original design** (docs/ROADMAP.md,
//! Phase 9): this modifier was originally designed with a third gate —
//! "some active opponent's `hand_size` is at or below a fixed
//! `BULLY_CLOSE` threshold" — reasoning by analogy to `denial`/`tempo`,
//! which both gate on an opponent (or this seat) being close to
//! finishing. A sweep of that threshold (1, 2, 3, 4, 6, 10, and finally
//! removing the gate outright) in a clean head-to-head against plain
//! `LowestLegal`, across all four table sizes and multiple independent
//! seed bases, showed the edge growing *monotonically* as the threshold
//! was relaxed, plateauing (not reversing) once the threshold exceeded
//! every realistic hand size — i.e. the gate was never protecting
//! against a real downside, only needlessly limiting how often a
//! costless benefit got realized. Unlike `tempo::TEMPO_CLOSE` (a sharp,
//! non-generalizing sweet spot), there is no opponent-proximity sweet
//! spot here at all: the hand-shape gate alone fully captures this
//! modifier's benefit. The opponent-proximity gate has been removed
//! entirely rather than shipped as a misleadingly-named no-op constant.
//!
//! Two gates must both pass before this modifier overrides the base
//! strategy's own lead choice; `None` means "defer," matching
//! `tempo::respond`'s convention:
//!
//! 1. **Leading, not following** — `Move::Pass` only ever appears in
//!    `legal_moves()` while following (`engine::legal_moves`), so its
//!    absence is this codebase's established way to detect a lead
//!    decision (see `HoldBackPairs::choose_play`). While following, the
//!    combo size is already fixed by the table, so there's no
//!    group-vs-single ordering choice left to make — defer entirely.
//! 2. **This hand is shaped mostly as same-rank groups** — using
//!    `engine::rank_groups`, multi-card groups (size >= 2) must be at
//!    least as numerous as singleton groups. This generalizes past the
//!    motivating example's "exactly one leftover single": it fires
//!    trivially when there are zero leftover singles, fires on the
//!    worked example's three-groups-one-single shape, and backs off once
//!    scattered singles dominate the hand, where lead-ordering has
//!    little left to add.
//!
//! Once triggered, the candidate pool is restricted to **whole-group**
//! leads only. `engine::legal_moves` enumerates every size `1..=N` for a
//! held N-card rank group while leading (not just 1 and N), so a naive
//! "smallest legal combo of size >= 2" filter would, for a held triple,
//! prefer leading just 2 of the 3 cards — splitting the group and
//! stranding the third card as a brand-new single, exactly the failure
//! mode this modifier exists to prevent. Restricting to combos whose
//! size equals the *full* count of that rank in hand avoids this.
//! Among whole-group leads, the tie-break is `LowestLegal`'s own
//! ordering (smallest size, then lowest top card) restricted to that
//! pool — the least-cost way to start chaining forced passes.
//!
//! **Interaction with `denial`/`reading`**: in the same head-to-head
//! testing, `bully`'s edge over plain `LowestLegal` was large and
//! monotonic across every role at every table size when `denial`/
//! `reading` was off, and stacked cleanly alongside `tempo` (their
//! trigger windows don't overlap: `tempo` only fires once this seat's
//! own hand is critically small, `bully` fires on larger, group-shaped
//! hands). But with `denial`'s `reading` mode *also* enabled, `bully`'s
//! marginal contribution vanished into noise — `denial` runs first in
//! the pipeline, fires on both leading and following (a strictly broader
//! trigger than `bully`'s leading-only one), and already claims most of
//! the situations `bully` would have. This isn't a bug to fix; it's
//! simply that `denial`'s own lead choice already captures most of the
//! same benefit once it's active. `bully` is most valuable in
//! configurations where `denial`/`reading` is off.

use engine::{rank_groups, Card, Combo, DuplicateRule, Move};

use crate::strategy::TurnContext;

/// Whether `hand`'s same-rank groups are at least as often multi-card
/// (size >= 2) as singleton — see this module's doc comment for the
/// reasoning behind this specific cutoff.
fn hand_is_mostly_grouped(hand: &[Card]) -> bool {
    let groups = rank_groups(hand);
    let multis = groups.iter().filter(|g| g.len() >= 2).count();
    let singles = groups.len() - multis;
    multis > 0 && multis >= singles
}

/// Whether `combo` uses *every* card of its rank that `hand` holds (as
/// opposed to a partial subset of a larger group) — see this module's
/// doc comment for why this distinction matters.
fn is_whole_rank_group(combo: &Combo, hand: &[Card]) -> bool {
    let rank = combo.cards()[0].rank;
    let group_size = hand.iter().filter(|c| c.rank == rank).count();
    combo.size() == group_size
}

pub(super) fn respond(
    enabled: bool,
    legal_moves: &[Move],
    duplicate_rule: DuplicateRule,
    context: &TurnContext<'_>,
) -> Option<Move> {
    if !enabled || legal_moves.contains(&Move::Pass) {
        return None;
    }
    if !hand_is_mostly_grouped(context.hand) {
        return None;
    }

    legal_moves
        .iter()
        .filter_map(|mv| match mv {
            Move::Play(combo) if combo.size() >= 2 && is_whole_rank_group(combo, context.hand) => {
                Some((combo.size(), combo.top_card(duplicate_rule), mv))
            }
            _ => None,
        })
        .min_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.compare(&b.1, duplicate_rule))
        })
        .map(|(_, _, mv)| *mv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand_reading::PassCeilings;
    use engine::{Rank, Suit};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit, 0)
    }

    fn play(cards: Vec<Card>) -> Move {
        Move::Play(Combo::new(cards).unwrap())
    }

    fn context_with(hand: &[Card]) -> TurnContext<'_> {
        TurnContext {
            seat: 0,
            hand,
            opponents: Vec::new(),
            unseen_cards: Vec::new(),
            own_pass_ceilings: PassCeilings::default(),
            current_combo: None,
        }
    }

    // The worked example: three pairs (2s, 4s, 8s) plus one leftover
    // single (3H).
    fn motivating_hand() -> Vec<Card> {
        vec![
            card(Rank::Two, Suit::Diamonds),
            card(Rank::Two, Suit::Hearts),
            card(Rank::Three, Suit::Hearts),
            card(Rank::Four, Suit::Diamonds),
            card(Rank::Four, Suit::Hearts),
            card(Rank::Eight, Suit::Diamonds),
            card(Rank::Eight, Suit::Spades),
        ]
    }

    fn leading_moves_for(hand: &[Card]) -> Vec<Move> {
        // Mirrors engine::legal_moves's own enumeration shape closely
        // enough for these unit tests: every size 1..=group.len() per
        // rank group, both the lowest and highest subset when they
        // differ (irrelevant here since every rank only ever appears in
        // one suit per test fixture, so lowest == highest always).
        rank_groups(hand)
            .into_iter()
            .flat_map(|group| (1..=group.len()).map(move |size| play(group[..size].to_vec())))
            .collect()
    }

    #[test]
    fn disabled_never_fires() {
        let hand = motivating_hand();
        let legal = leading_moves_for(&hand);
        let context = context_with(&hand);
        assert_eq!(
            respond(false, &legal, DuplicateRule::FirstDealtWins, &context),
            None
        );
    }

    #[test]
    fn following_never_fires() {
        let hand = motivating_hand();
        let mut legal = leading_moves_for(&hand);
        legal.push(Move::Pass);
        let context = context_with(&hand);
        assert_eq!(
            respond(true, &legal, DuplicateRule::FirstDealtWins, &context),
            None
        );
    }

    #[test]
    fn fires_with_no_opponents_present_at_all() {
        // The opponent-proximity gate was removed after empirical
        // validation showed it only limited the benefit without
        // protecting against any downside (see this module's doc
        // comment) -- confirm the hand-shape gate alone is sufficient,
        // with zero opponent information available.
        let hand = motivating_hand();
        let legal = leading_moves_for(&hand);
        let context = context_with(&hand);
        assert_eq!(
            respond(true, &legal, DuplicateRule::FirstDealtWins, &context),
            Some(play(vec![
                card(Rank::Two, Suit::Diamonds),
                card(Rank::Two, Suit::Hearts)
            ]))
        );
    }

    #[test]
    fn hand_with_no_multi_card_groups_never_fires() {
        let hand = vec![
            card(Rank::Two, Suit::Diamonds),
            card(Rank::Four, Suit::Hearts),
            card(Rank::Six, Suit::Spades),
        ];
        let legal = leading_moves_for(&hand);
        let context = context_with(&hand);
        assert_eq!(
            respond(true, &legal, DuplicateRule::FirstDealtWins, &context),
            None
        );
    }

    #[test]
    fn hand_with_singles_in_the_majority_never_fires() {
        // One pair (2s) plus three leftover singles: multis=1 < singles=3.
        let hand = vec![
            card(Rank::Two, Suit::Diamonds),
            card(Rank::Two, Suit::Hearts),
            card(Rank::Five, Suit::Hearts),
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Nine, Suit::Spades),
        ];
        let legal = leading_moves_for(&hand);
        let context = context_with(&hand);
        assert_eq!(
            respond(true, &legal, DuplicateRule::FirstDealtWins, &context),
            None
        );
    }

    #[test]
    fn fires_with_exactly_one_leftover_single() {
        let hand = motivating_hand();
        let legal = leading_moves_for(&hand);
        let context = context_with(&hand);
        assert_eq!(
            respond(true, &legal, DuplicateRule::FirstDealtWins, &context),
            Some(play(vec![
                card(Rank::Two, Suit::Diamonds),
                card(Rank::Two, Suit::Hearts)
            ]))
        );
    }

    #[test]
    fn fires_with_zero_leftover_singles() {
        let hand = vec![
            card(Rank::Five, Suit::Diamonds),
            card(Rank::Five, Suit::Hearts),
            card(Rank::Nine, Suit::Clubs),
            card(Rank::Nine, Suit::Spades),
        ];
        let legal = leading_moves_for(&hand);
        let context = context_with(&hand);
        assert_eq!(
            respond(true, &legal, DuplicateRule::FirstDealtWins, &context),
            Some(play(vec![
                card(Rank::Five, Suit::Diamonds),
                card(Rank::Five, Suit::Hearts)
            ]))
        );
    }

    #[test]
    fn picks_the_lowest_rank_among_equal_sized_groups() {
        let hand = vec![
            card(Rank::Nine, Suit::Clubs),
            card(Rank::Nine, Suit::Spades),
            card(Rank::Five, Suit::Diamonds),
            card(Rank::Five, Suit::Hearts),
        ];
        let legal = leading_moves_for(&hand);
        let context = context_with(&hand);
        assert_eq!(
            respond(true, &legal, DuplicateRule::FirstDealtWins, &context),
            Some(play(vec![
                card(Rank::Five, Suit::Diamonds),
                card(Rank::Five, Suit::Hearts)
            ]))
        );
    }

    #[test]
    fn never_splits_a_larger_group_into_a_smaller_lead() {
        let hand = vec![
            card(Rank::Seven, Suit::Diamonds),
            card(Rank::Seven, Suit::Hearts),
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Nine, Suit::Spades),
        ];
        let legal = leading_moves_for(&hand);
        let context = context_with(&hand);
        // Only the whole triple should ever be returned -- never a
        // 2-card subset of it, even though engine::legal_moves (and this
        // test's own leading_moves_for helper) also offers that subset
        // as a legal lead.
        assert_eq!(
            respond(true, &legal, DuplicateRule::FirstDealtWins, &context),
            Some(play(vec![
                card(Rank::Seven, Suit::Diamonds),
                card(Rank::Seven, Suit::Hearts),
                card(Rank::Seven, Suit::Clubs)
            ]))
        );
    }

    #[test]
    fn a_full_pair_is_chosen_over_a_full_triple_when_both_are_legal() {
        let hand = vec![
            card(Rank::Nine, Suit::Diamonds),
            card(Rank::Nine, Suit::Hearts),
            card(Rank::Nine, Suit::Clubs),
            card(Rank::Two, Suit::Diamonds),
            card(Rank::Two, Suit::Hearts),
        ];
        let legal = leading_moves_for(&hand);
        let context = context_with(&hand);
        assert_eq!(
            respond(true, &legal, DuplicateRule::FirstDealtWins, &context),
            Some(play(vec![
                card(Rank::Two, Suit::Diamonds),
                card(Rank::Two, Suit::Hearts)
            ]))
        );
    }

    #[test]
    fn defers_to_base_once_no_multi_card_lead_remains() {
        // Hand down to a single lone card -- the shape gate itself
        // already blocks this (no multi-card group exists), so the
        // caller's base strategy picks the lone single instead.
        let hand = vec![card(Rank::Three, Suit::Hearts)];
        let legal = leading_moves_for(&hand);
        let context = context_with(&hand);
        assert_eq!(
            respond(true, &legal, DuplicateRule::FirstDealtWins, &context),
            None
        );
    }
}
