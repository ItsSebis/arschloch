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

use crate::card::{Card, DuplicateRule};
use crate::combo::Combo;
use crate::round::Move;

pub(crate) fn legal_moves(
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

/// Groups `hand` by rank.
fn rank_groups(hand: &[Card]) -> Vec<Vec<Card>> {
    let mut sorted = hand.to_vec();
    sorted.sort_by_key(|card| card.rank);
    let mut groups: Vec<Vec<Card>> = Vec::new();
    for card in sorted {
        match groups.last_mut() {
            Some(group) if group[0].rank == card.rank => group.push(card),
            _ => groups.push(vec![card]),
        }
    }
    groups
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
    let first_beating = sorted
        .iter()
        .position(|card| card.compare(&threshold, duplicate_rule) == std::cmp::Ordering::Greater)?;
    let top_index = first_beating.max(size - 1);
    if top_index >= sorted.len() {
        return None;
    }
    // `top_index + 1 - size`, not `top_index - size + 1`: `top_index` can
    // equal `size - 1`, where the latter underflows `usize` before the
    // `+ 1` is applied.
    Some(sorted[top_index + 1 - size..=top_index].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{Rank, Suit};

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
}
