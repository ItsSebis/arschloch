//! Enumerates the legal moves for a hand against an optional current
//! combo. Crate-private: `Round::legal_moves` is the public entry point.
//!
//! **Decision:** rather than every suit-subset of a rank (a full power
//! set, which blows up combinatorially in the double-deck variant — up
//! to 2^8-1 = 255 subsets for one rank held 8 times), this produces at
//! most two candidates per (rank, size): the weakest and the strongest
//! `size` physical cards of that rank (deduped when equal) — when
//! following, "weakest" means the weakest subset that still beats the
//! current combo (see `canonical_subsets`).
//! `sim`'s baseline strategies only ever need "the weakest" or "the
//! strongest" combo of a given size, so this is enough without solving a
//! combinatorial optimization problem the roadmap doesn't call for.

use std::cmp::Ordering;

use crate::card::{Card, DuplicateRule, Rank, Suit};
use crate::combo::{Combo, MAX_COMBO_SIZE};
use crate::round::Move;

/// Hands up to this size are sorted in a stack array. The largest hand of a
/// supported table is the double deck at three players: 104 / 3 -> 35.
const STACK_HAND: usize = 40;

/// Appends the legal moves of `hand` against `current_combo` to `out`
/// (clearing it first), in the order documented above: `Pass` first (only
/// when following), then ascending rank, ascending size, lowest before
/// highest. Allocation-free once `out` has capacity.
///
/// One pass over a sorted copy of the hand: the cards of a rank are a
/// contiguous slice, the lowest and highest `size`-subsets are windows of
/// it, and the lowest subset that beats the table is found by index.
pub(crate) fn legal_moves_into(
    hand: &[Card],
    current_combo: Option<&Combo>,
    duplicate_rule: DuplicateRule,
    out: &mut Vec<Move>,
) {
    out.clear();
    let filler = Card::new(Rank::Two, Suit::Diamonds, 0);
    let mut stack = [filler; STACK_HAND];
    let mut heap: Vec<Card>;
    let sorted: &mut [Card] = if hand.len() <= STACK_HAND {
        let slice = &mut stack[..hand.len()];
        slice.copy_from_slice(hand);
        slice
    } else {
        heap = hand.to_vec();
        &mut heap
    };
    // Two cards compare equal only when they are identical, so an unstable
    // sort gives the same sequence as a stable one.
    sorted.sort_unstable_by(|a, b| a.compare(b, duplicate_rule));

    let threshold = current_combo.map(|current| current.top_card(duplicate_rule));
    if current_combo.is_some() {
        out.push(Move::Pass);
    }
    let mut start = 0;
    while start < sorted.len() {
        let rank = sorted[start].rank;
        let mut end = start + 1;
        while end < sorted.len() && sorted[end].rank == rank {
            end += 1;
        }
        let group = &sorted[start..end];
        start = end;
        let n = group.len();
        match (current_combo, threshold) {
            (Some(current), Some(threshold)) => {
                let size = current.size();
                if size > n {
                    continue;
                }
                // The smallest index whose card beats the table; with none,
                // no `size`-subset of this rank does.
                let Some(first_beating) = group
                    .iter()
                    .position(|card| card.compare(&threshold, duplicate_rule) == Ordering::Greater)
                else {
                    continue;
                };
                let top_index = first_beating.max(size - 1);
                let lowest = &group[top_index + 1 - size..=top_index];
                let highest = &group[n - size..];
                // Equal by value, not by position: identical duplicate
                // cards can make two different windows the same set.
                if lowest != highest {
                    out.push(Move::Play(Combo::from_same_rank(lowest)));
                }
                out.push(Move::Play(Combo::from_same_rank(highest)));
            }
            _ => {
                for size in 1..=n.min(MAX_COMBO_SIZE) {
                    let lowest = &group[..size];
                    let highest = &group[n - size..];
                    if lowest != highest {
                        out.push(Move::Play(Combo::from_same_rank(lowest)));
                    }
                    out.push(Move::Play(Combo::from_same_rank(highest)));
                }
            }
        }
    }
}

pub(crate) fn legal_moves(
    hand: &[Card],
    current_combo: Option<&Combo>,
    duplicate_rule: DuplicateRule,
) -> Vec<Move> {
    let mut moves = Vec::new();
    legal_moves_into(hand, current_combo, duplicate_rule, &mut moves);
    moves
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{rank_groups, Rank, Suit};

    // The original per-rank, per-size `Vec` implementation: the oracle the
    // allocation-free one is tested against.
    fn legal_moves_reference(
        hand: &[Card],
        current_combo: Option<&Combo>,
        duplicate_rule: DuplicateRule,
    ) -> Vec<Move> {
        let mut moves = Vec::new();
        if current_combo.is_some() {
            moves.push(Move::Pass);
        }
        for group in rank_groups(hand) {
            let sizes: Vec<usize> = match current_combo {
                Some(current) if current.size() <= group.len() => vec![current.size()],
                Some(_) => Vec::new(),
                None => (1..=group.len()).collect(),
            };
            for size in sizes {
                for subset in canonical_subsets(&group, size, current_combo, duplicate_rule) {
                    let candidate = Combo::new(subset)
                        .expect("a non-empty subset of one rank group is always a valid Combo");
                    let legal = match current_combo {
                        Some(current) => candidate.beats(current, duplicate_rule),
                        None => true,
                    };
                    if legal {
                        moves.push(Move::Play(candidate));
                    }
                }
            }
        }
        moves
    }

    /// The lowest-top-card and highest-top-card `size`-subsets of `group`
    /// (already-confirmed single rank), deduped when they're the same set.
    /// When `current_combo` is `Some`, the "lowest" candidate is the
    /// minimal-top-card subset that actually beats it (not just the group's
    /// global lowest, which can fail to beat a same-rank current combo while
    /// a weaker-but-still-beating subset exists) — see
    /// `lowest_beating_subset`. Returns nothing if `size` is `0` or larger
    /// than `group`.
    fn canonical_subsets(
        group: &[Card],
        size: usize,
        current_combo: Option<&Combo>,
        duplicate_rule: DuplicateRule,
    ) -> Vec<Vec<Card>> {
        if size == 0 || size > group.len() {
            return Vec::new();
        }
        let mut sorted = group.to_vec();
        sorted.sort_by(|a, b| a.compare(b, duplicate_rule));
        let highest = sorted[sorted.len() - size..].to_vec();

        let lowest = match current_combo {
            Some(current) => lowest_beating_subset(
                &sorted,
                size,
                current.top_card(duplicate_rule),
                duplicate_rule,
            ),
            None => Some(sorted[..size].to_vec()),
        };

        match lowest {
            Some(lowest) if lowest == highest => vec![lowest],
            Some(lowest) => vec![lowest, highest],
            None => vec![highest],
        }
    }

    /// The `size`-card contiguous window (in `sorted`, ascending) with the
    /// smallest possible top card that still beats `threshold`, or `None` if
    /// no `size`-subset of `sorted` beats it. Since same-rank cards only
    /// differ by suit/duplicate-tiebreak, a subset's "top card" is just its
    /// maximum element, and minimizing that maximum while requiring it to
    /// exceed `threshold` is exactly: find the smallest index whose card
    /// beats `threshold`, then take the `size`-length window ending there
    /// (extending downward, or upward if too few cards sit below it).
    fn lowest_beating_subset(
        sorted: &[Card],
        size: usize,
        threshold: Card,
        duplicate_rule: DuplicateRule,
    ) -> Option<Vec<Card>> {
        let first_beating = sorted.iter().position(|card| {
            card.compare(&threshold, duplicate_rule) == std::cmp::Ordering::Greater
        })?;
        let top_index = first_beating.max(size - 1);
        if top_index >= sorted.len() {
            return None;
        }
        // `top_index + 1 - size`, not `top_index - size + 1`: `top_index` can
        // equal `size - 1`, where the latter underflows `usize` before the
        // `+ 1` is applied.
        Some(sorted[top_index + 1 - size..=top_index].to_vec())
    }

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit, 0)
    }

    #[test]
    fn leading_enumerates_every_size_up_to_the_rank_groups_count() {
        let hand = vec![
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Seven, Suit::Diamonds),
            card(Rank::Nine, Suit::Hearts),
        ];
        let moves = legal_moves(&hand, None, DuplicateRule::FirstDealtWins);
        // Seven group (2 cards): size 1 -> 2 candidates (distinct
        // suits), size 2 -> 1 candidate (whole group, deduped). Nine
        // group (1 card): size 1 -> 1 candidate. Total 4. No Pass while
        // leading.
        assert_eq!(moves.len(), 4);
        assert!(!moves.contains(&Move::Pass));
    }

    #[test]
    fn following_only_considers_the_current_combos_size_and_filters_by_beats() {
        let hand = vec![
            card(Rank::Six, Suit::Clubs),
            card(Rank::Eight, Suit::Clubs),
            card(Rank::Eight, Suit::Diamonds),
        ];
        let current = Combo::new(vec![card(Rank::Seven, Suit::Clubs)]).unwrap();
        let moves = legal_moves(&hand, Some(&current), DuplicateRule::FirstDealtWins);
        // Pass, plus both size-1 Eights (both beat Seven regardless of
        // suit, since rank alone decides once it's strictly higher); the
        // size-1 Six doesn't beat, and there's no size-2 group to match a
        // size-1 current combo anyway.
        assert_eq!(moves.len(), 3);
        assert!(moves.contains(&Move::Pass));
        let eight_plays = moves
            .iter()
            .filter(
                |m| matches!(m, Move::Play(c) if c.size() == 1 && c.cards()[0].rank == Rank::Eight),
            )
            .count();
        assert_eq!(eight_plays, 2);
        assert!(!moves
            .iter()
            .any(|m| matches!(m, Move::Play(c) if c.cards()[0].rank == Rank::Six)));
    }

    #[test]
    fn following_a_same_rank_combo_offers_the_weakest_card_that_still_beats() {
        // Suit order is Diamonds < Hearts < Spades < Clubs. Against a
        // single Seven of Hearts, the Seven of Diamonds doesn't beat, but
        // both the Seven of Spades and the Seven of Clubs do — and the
        // Spades one is the weaker of the two, so it must be offered as
        // the "lowest" candidate rather than being skipped because the
        // group's global lowest (Diamonds) fails to beat.
        let hand = vec![
            card(Rank::Seven, Suit::Diamonds),
            card(Rank::Seven, Suit::Spades),
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Three, Suit::Clubs),
        ];
        let current = Combo::new(vec![card(Rank::Seven, Suit::Hearts)]).unwrap();
        let moves = legal_moves(&hand, Some(&current), DuplicateRule::FirstDealtWins);
        let single = |rank, suit| Move::Play(Combo::new(vec![card(rank, suit)]).unwrap());
        assert_eq!(moves.len(), 3);
        assert!(moves.contains(&Move::Pass));
        assert!(moves.contains(&single(Rank::Seven, Suit::Spades)));
        assert!(moves.contains(&single(Rank::Seven, Suit::Clubs)));
        assert!(!moves.contains(&single(Rank::Seven, Suit::Diamonds)));
        assert!(!moves.contains(&single(Rank::Three, Suit::Clubs)));
    }

    #[test]
    fn following_a_same_rank_pair_offers_the_weakest_pair_that_still_beats() {
        // Against a pair topped by the Seven of Hearts, the weakest
        // beating pair from Diamonds/Spades/Clubs Sevens is
        // Diamonds+Spades (top card Spades) — the window extends downward
        // from the first beating card — alongside the strongest,
        // Spades+Clubs. (Double deck: the table's Seven of Diamonds is
        // the other physical copy, distinguished by `deal_index`.)
        let hand = vec![
            card(Rank::Seven, Suit::Diamonds),
            card(Rank::Seven, Suit::Spades),
            card(Rank::Seven, Suit::Clubs),
        ];
        let current = Combo::new(vec![
            Card::new(Rank::Seven, Suit::Hearts, 1),
            Card::new(Rank::Seven, Suit::Diamonds, 1),
        ])
        .unwrap();
        let moves = legal_moves(&hand, Some(&current), DuplicateRule::FirstDealtWins);
        let pair = |a, b| {
            Move::Play(Combo::new(vec![card(Rank::Seven, a), card(Rank::Seven, b)]).unwrap())
        };
        assert_eq!(
            moves,
            vec![
                Move::Pass,
                pair(Suit::Diamonds, Suit::Spades),
                pair(Suit::Spades, Suit::Clubs),
            ]
        );
    }

    #[test]
    fn a_rank_group_smaller_than_the_current_combos_size_contributes_nothing() {
        let hand = vec![card(Rank::Nine, Suit::Clubs)];
        let current = Combo::new(vec![
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Seven, Suit::Diamonds),
        ])
        .unwrap();
        let moves = legal_moves(&hand, Some(&current), DuplicateRule::FirstDealtWins);
        assert_eq!(moves, vec![Move::Pass]);
    }

    #[test]
    fn a_single_card_of_a_rank_produces_one_deduped_candidate() {
        let hand = vec![card(Rank::Ten, Suit::Clubs)];
        let moves = legal_moves(&hand, None, DuplicateRule::FirstDealtWins);
        assert_eq!(moves.len(), 1);
    }

    #[test]
    fn double_deck_duplicate_cards_are_disambiguated_by_deal_index() {
        let hand = vec![
            Card::new(Rank::Ten, Suit::Clubs, 0),
            Card::new(Rank::Ten, Suit::Clubs, 1),
        ];
        let moves = legal_moves(&hand, None, DuplicateRule::FirstDealtWins);
        // size 1 -> 2 candidates (the two physical cards are distinct by
        // deal_index), size 2 -> 1 candidate (whole group). Total 3.
        assert_eq!(moves.len(), 3);
    }

    /// xorshift64*: a fixed, dependency-free source of reproducible shuffles.
    fn next(state: &mut u64) -> u64 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(state: &mut u64, n: usize) -> usize {
        usize::try_from(next(state) % n as u64).unwrap()
    }

    #[test]
    fn allocation_free_enumeration_equals_the_reference_in_content_and_order() {
        use crate::card::DeckVariant;
        use crate::deal::deal;
        use crate::deck::standard_deck;

        let mut state = 0x1234_5678_9ABC_DEF1_u64;
        let mut cases = 0_u64;
        let mut buffer = vec![Move::Pass; 3]; // stale content must be cleared
        for variant in [DeckVariant::Single, DeckVariant::Double] {
            for players in 3..=6_u8 {
                for rule in [DuplicateRule::FirstDealtWins, DuplicateRule::LastDealtWins] {
                    for _ in 0..40 {
                        let mut deck = standard_deck(variant);
                        for i in (1..deck.len()).rev() {
                            let j = below(&mut state, i + 1);
                            deck.swap(i, j);
                        }
                        let all_cards = standard_deck(variant);
                        let hands = deal(deck, players).unwrap();
                        for full in hands.iter().take(2) {
                            // The dealt hand and a random part of it.
                            let keep = 1 + below(&mut state, full.len());
                            let part: Vec<Card> = full[..keep].to_vec();
                            for hand in [full, &part] {
                                check(hand, None, rule, &mut buffer);
                                cases += 1;
                                // Current combos: a random sample of the
                                // same-rank subsets of the whole deck.
                                for _ in 0..60 {
                                    let rank_cards: Vec<Card> = {
                                        let pick = all_cards[below(&mut state, all_cards.len())];
                                        all_cards
                                            .iter()
                                            .copied()
                                            .filter(|c| c.rank == pick.rank)
                                            .collect()
                                    };
                                    let mask = 1 + below(&mut state, (1 << rank_cards.len()) - 1);
                                    let subset: Vec<Card> = rank_cards
                                        .iter()
                                        .enumerate()
                                        .filter(|(i, _)| mask >> i & 1 == 1)
                                        .map(|(_, c)| *c)
                                        .collect();
                                    let current = Combo::new(subset).unwrap();
                                    check(hand, Some(&current), rule, &mut buffer);
                                    cases += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(cases >= 20_000, "only {cases} cases");
    }

    fn check(hand: &[Card], current: Option<&Combo>, rule: DuplicateRule, buffer: &mut Vec<Move>) {
        let expected = legal_moves_reference(hand, current, rule);
        legal_moves_into(hand, current, rule, buffer);
        assert_eq!(*buffer, expected, "hand {hand:?} current {current:?}");
        assert_eq!(legal_moves(hand, current, rule), expected);
    }
}
