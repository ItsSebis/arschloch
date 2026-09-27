//! Pass-based hand reading (docs/ROADMAP.md, Phase 7).
//!
//! A seat that passed against a size-`s` combo topped by `c` held no
//! size-`s` combo topped above `c` at that moment; hands only shrink
//! (no draw pile), so that stays true for the rest of the round. By
//! upward closure it also holds no *larger* combo topped above `c` (any
//! such combo contains a size-`s` sub-combo with the same top card), so
//! a seat's ceiling at size `s` is the lowest top card it has passed
//! against at any size `<= s`. Ceilings are `Card`s, not `Rank`s:
//! passing on 9♦ says nothing about 9♣ — the same lesson `CardCounter`
//! learned the hard way (see its own doc comment).
//!
//! **Refutation.** Not every pass is honest — `HoldBackPairs` passes on
//! a single when every beating play would split a pair, `RandomLegal`
//! passes at random, and `Adaptive`'s deception modifier passes on
//! purpose. A pass is dropped (never recorded, or overwritten) once the
//! same seat later plays a combo of the same or larger size topped
//! above the passed-on card — proof it held a beater at the time. An
//! honest pass can never be refuted, since hands only shrink; this only
//! ever removes false information, never true information.

use std::cmp::Ordering;

use engine::{Card, Combo, DuplicateRule, SeatId};

/// The largest combo any seat can field: 8 copies of one rank in the
/// double deck.
pub const MAX_COMBO_SIZE: usize = 8;

/// Per-size "can't beat" facts about one seat, derived from its
/// unrefuted passes. `by_size[s - 1]` is the lowest top card passed
/// against at exactly size `s`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PassCeilings {
    by_size: [Option<Card>; MAX_COMBO_SIZE],
}

impl PassCeilings {
    /// The lowest top card this seat is known unable to beat at
    /// `size`, applying upward closure. `None`: nothing is known at
    /// this size.
    #[must_use]
    pub fn ceiling(&self, size: usize, duplicate_rule: DuplicateRule) -> Option<Card> {
        self.by_size
            .iter()
            .take(size)
            .flatten()
            .copied()
            .min_by(|a, b| a.compare(b, duplicate_rule))
    }

    /// Whether this seat is known unable to beat a size-`size` combo
    /// topped by `top`.
    ///
    /// A strictly higher *rank* is always covered, regardless of suit
    /// (rank dominates suit in `Card::compare`, so a lower-rank ceiling
    /// bounds every higher rank). Within the *same* rank, only an exact
    /// suit match counts: `Card::compare`'s suit tiebreak gives every
    /// pair of same-rank cards a definite order (e.g. King-of-Clubs >
    /// King-of-Hearts), but that order reflects an arbitrary house rule
    /// for resolving legality, not a real difference in how hard the
    /// two cards are to beat — a seat that passed on the King of Hearts
    /// has told us nothing about the King of Clubs (see the module
    /// doc's 9♦-vs-9♣ example). Deal index (double-deck duplicates) is
    /// likewise ignored here: two copies of the same rank/suit are the
    /// same card for this purpose.
    #[must_use]
    pub fn cannot_beat(&self, size: usize, top: Card, duplicate_rule: DuplicateRule) -> bool {
        self.ceiling(size, duplicate_rule)
            .is_some_and(|c| top.rank > c.rank || (top.rank == c.rank && top.suit == c.suit))
    }
}

/// Reduces a round's play/pass history to one `PassCeilings` per seat,
/// in a single reverse sweep: walking backward, `highest_later[seat]`
/// at index `s - 1` is the highest top card `seat` played at size
/// `>= s` *after* the pass currently being examined — exactly the
/// refutation test.
///
/// `pass_history` entries are `(seat, combo, plays_before)` exactly as
/// `engine::Round::pass_history()` returns them (`plays_before` is
/// `play_history.len()` at the moment of that pass).
#[must_use]
pub fn read_pass_ceilings(
    player_count: usize,
    play_history: &[(SeatId, Combo)],
    pass_history: &[(SeatId, Combo, usize)],
    duplicate_rule: DuplicateRule,
) -> Vec<PassCeilings> {
    let mut ceilings = vec![PassCeilings::default(); player_count];
    let mut highest_later = vec![[None::<Card>; MAX_COMBO_SIZE]; player_count];
    let mut play_index = play_history.len();

    for (seat, combo, plays_before) in pass_history.iter().rev() {
        while play_index > *plays_before {
            play_index -= 1;
            let (play_seat, play_combo) = &play_history[play_index];
            let top = play_combo.top_card(duplicate_rule);
            for slot in &mut highest_later[usize::from(*play_seat)][..play_combo.size()] {
                if slot.is_none_or(|h| top.compare(&h, duplicate_rule) == Ordering::Greater) {
                    *slot = Some(top);
                }
            }
        }

        let seat_idx = usize::from(*seat);
        let size = combo.size();
        let passed_top = combo.top_card(duplicate_rule);
        let refuted = highest_later[seat_idx][size - 1]
            .is_some_and(|h| h.compare(&passed_top, duplicate_rule) == Ordering::Greater);
        if !refuted {
            let slot = &mut ceilings[seat_idx].by_size[size - 1];
            if slot.is_none_or(|c| passed_top.compare(&c, duplicate_rule) == Ordering::Less) {
                *slot = Some(passed_top);
            }
        }
    }
    ceilings
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::{Rank, Suit};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit, 0)
    }

    fn combo(cards: Vec<Card>) -> Combo {
        Combo::new(cards).unwrap()
    }

    #[test]
    fn a_pass_at_size_one_bounds_size_two_as_well() {
        let pass_history = vec![(0u8, combo(vec![card(Rank::Nine, Suit::Diamonds)]), 0)];
        let ceilings = read_pass_ceilings(2, &[], &pass_history, DuplicateRule::FirstDealtWins);
        let nine = card(Rank::Nine, Suit::Diamonds);
        assert_eq!(
            ceilings[0].ceiling(1, DuplicateRule::FirstDealtWins),
            Some(nine)
        );
        assert_eq!(
            ceilings[0].ceiling(2, DuplicateRule::FirstDealtWins),
            Some(nine)
        );
    }

    #[test]
    fn a_pass_at_size_two_says_nothing_about_size_one() {
        let pair = combo(vec![
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Seven, Suit::Diamonds),
        ]);
        let pass_history = vec![(0u8, pair, 0)];
        let ceilings = read_pass_ceilings(2, &[], &pass_history, DuplicateRule::FirstDealtWins);
        assert_eq!(ceilings[0].ceiling(1, DuplicateRule::FirstDealtWins), None);
    }

    #[test]
    fn suit_precision_a_pass_against_one_suit_does_not_cover_a_higher_suit_same_rank() {
        // engine's Suit order: Diamonds < Hearts < Spades < Clubs.
        let pass_history = vec![(0u8, combo(vec![card(Rank::King, Suit::Hearts)]), 0)];
        let ceilings = read_pass_ceilings(2, &[], &pass_history, DuplicateRule::FirstDealtWins);
        let king_hearts = card(Rank::King, Suit::Hearts);
        let king_clubs = card(Rank::King, Suit::Clubs);
        assert!(ceilings[0].cannot_beat(1, king_hearts, DuplicateRule::FirstDealtWins));
        assert!(
            !ceilings[0].cannot_beat(1, king_clubs, DuplicateRule::FirstDealtWins),
            "a same-rank higher-suit card must NOT be considered covered by this pass \
             (a Rank-keyed ceiling would wrongly say it is)"
        );
    }

    #[test]
    fn a_later_play_above_the_passed_card_refutes_the_pass() {
        let nine = combo(vec![card(Rank::Nine, Suit::Diamonds)]);
        let king = combo(vec![card(Rank::King, Suit::Clubs)]);
        let pass_history = vec![(0u8, nine, 0)]; // plays_before: 0
        let play_history_refuting = vec![(0u8, king.clone())]; // this play is index 0, so plays_before(0) <= 0 means it happened AFTER
        let ceilings = read_pass_ceilings(
            2,
            &play_history_refuting,
            &pass_history,
            DuplicateRule::FirstDealtWins,
        );
        assert_eq!(
            ceilings[0].ceiling(1, DuplicateRule::FirstDealtWins),
            None,
            "seat 0 later played a King, proving the earlier pass was not honest"
        );
    }

    #[test]
    fn a_play_recorded_before_the_pass_does_not_refute_it() {
        let king = combo(vec![card(Rank::King, Suit::Clubs)]);
        let nine = combo(vec![card(Rank::Nine, Suit::Diamonds)]);
        let play_history = vec![(0u8, king)]; // index 0, happened before
        let pass_history = vec![(0u8, nine.clone(), 1)]; // plays_before: 1 (i.e. after that one play)
        let ceilings = read_pass_ceilings(
            2,
            &play_history,
            &pass_history,
            DuplicateRule::FirstDealtWins,
        );
        assert_eq!(
            ceilings[0].ceiling(1, DuplicateRule::FirstDealtWins),
            Some(card(Rank::Nine, Suit::Diamonds)),
            "the King was played BEFORE this pass, so it doesn't refute it"
        );
    }

    #[test]
    fn ceilings_are_tracked_independently_per_seat() {
        let pass_history = vec![
            (0u8, combo(vec![card(Rank::Nine, Suit::Diamonds)]), 0),
            (1u8, combo(vec![card(Rank::Six, Suit::Diamonds)]), 0),
        ];
        let ceilings = read_pass_ceilings(2, &[], &pass_history, DuplicateRule::FirstDealtWins);
        assert_eq!(
            ceilings[0].ceiling(1, DuplicateRule::FirstDealtWins),
            Some(card(Rank::Nine, Suit::Diamonds))
        );
        assert_eq!(
            ceilings[1].ceiling(1, DuplicateRule::FirstDealtWins),
            Some(card(Rank::Six, Suit::Diamonds))
        );
    }
}
