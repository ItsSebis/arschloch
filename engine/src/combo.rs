//! A `Combo` is a set of cards of equal rank played together (a single,
//! pair, triple, ...). Straights and bombs are out of scope for this
//! phase (see docs/RULES.md, "Deferred / Out-of-scope Rules").

use std::cmp::Ordering;
use std::fmt;

use crate::card::{Card, DuplicateRule, Rank, Suit};

/// The most cards one combo can hold: the double deck has eight cards of
/// each rank (two decks x four suits), the most any supported table has.
pub const MAX_COMBO_SIZE: usize = 8;

/// Unused slots of the inline array hold this card, so a `Combo` is always
/// the same bytes for the same cards.
const FILLER: Card = Card {
    rank: Rank::Two,
    suit: Suit::Diamonds,
    deal_index: 0,
};

/// Stored inline (no heap allocation) and therefore `Copy`.
#[derive(Clone, Copy)]
pub struct Combo {
    cards: [Card; MAX_COMBO_SIZE],
    len: u8,
}

impl PartialEq for Combo {
    fn eq(&self, other: &Self) -> bool {
        self.cards() == other.cards()
    }
}

impl Eq for Combo {}

impl fmt::Debug for Combo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Combo")
            .field("cards", &self.cards())
            .finish()
    }
}

impl Combo {
    /// Builds a combo from cards that must all share the same rank and
    /// must be non-empty. Returns `None` otherwise, and also for more than
    /// [`MAX_COMBO_SIZE`] cards (more cards of one rank than any supported
    /// deck holds, so never a legal play).
    #[must_use]
    #[allow(clippy::needless_pass_by_value)] // public signature kept; see `from_slice`
    pub fn new(cards: Vec<Card>) -> Option<Self> {
        Self::from_slice(&cards)
    }

    /// Like [`Combo::new`], from a slice.
    #[must_use]
    pub fn from_slice(cards: &[Card]) -> Option<Self> {
        let first = cards.first()?;
        if cards.len() > MAX_COMBO_SIZE || !cards.iter().all(|c| c.rank == first.rank) {
            return None;
        }
        Some(Self::from_same_rank(cards))
    }

    /// `cards` must be non-empty, of one rank, and at most `MAX_COMBO_SIZE`.
    pub(crate) fn from_same_rank(cards: &[Card]) -> Self {
        debug_assert!(!cards.is_empty() && cards.len() <= MAX_COMBO_SIZE);
        let mut array = [FILLER; MAX_COMBO_SIZE];
        array[..cards.len()].copy_from_slice(cards);
        Self {
            cards: array,
            #[allow(clippy::cast_possible_truncation)] // at most 8
            len: cards.len() as u8,
        }
    }

    #[must_use]
    pub fn size(&self) -> usize {
        usize::from(self.len)
    }

    /// The cards making up this combo, in the order given to `Combo::new`.
    #[must_use]
    pub fn cards(&self) -> &[Card] {
        &self.cards[..self.size()]
    }

    /// The representative card used for comparison: since every card in
    /// a combo shares a rank, the highest card (by suit/duplicate
    /// tiebreak) stands in for the whole combo.
    ///
    /// # Panics
    ///
    /// Never in practice: `Combo::new` only ever constructs a `Combo`
    /// from a non-empty card list, so there's always at least one card
    /// to compare.
    #[must_use]
    pub fn top_card(&self, duplicate_rule: DuplicateRule) -> Card {
        *self
            .cards()
            .iter()
            .max_by(|a, b| a.compare(b, duplicate_rule))
            .expect("Combo is always constructed with at least one card")
    }

    /// Whether `self`, played on top of `previous`, is a legal follow:
    /// same size, and strictly higher by the match's duplicate-tiebreak
    /// rule.
    #[must_use]
    pub fn beats(&self, previous: &Combo, duplicate_rule: DuplicateRule) -> bool {
        self.size() == previous.size()
            && self
                .top_card(duplicate_rule)
                .compare(&previous.top_card(duplicate_rule), duplicate_rule)
                == Ordering::Greater
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{Rank, Suit};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit, 0)
    }

    #[test]
    fn empty_combo_is_rejected() {
        assert!(Combo::new(vec![]).is_none());
    }

    #[test]
    fn mixed_rank_combo_is_rejected() {
        let cards = vec![
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Eight, Suit::Clubs),
        ];
        assert!(Combo::new(cards).is_none());
    }

    #[test]
    fn same_rank_combo_is_accepted() {
        let cards = vec![
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Seven, Suit::Diamonds),
        ];
        assert!(Combo::new(cards).is_some());
    }

    #[test]
    fn higher_rank_combo_beats_lower_rank_combo_of_same_size() {
        let low = Combo::new(vec![card(Rank::Seven, Suit::Clubs)]).unwrap();
        let high = Combo::new(vec![card(Rank::Eight, Suit::Diamonds)]).unwrap();
        assert!(high.beats(&low, DuplicateRule::FirstDealtWins));
        assert!(!low.beats(&high, DuplicateRule::FirstDealtWins));
    }

    #[test]
    fn different_size_combo_never_beats() {
        let single = Combo::new(vec![card(Rank::Nine, Suit::Clubs)]).unwrap();
        let pair = Combo::new(vec![
            card(Rank::Ten, Suit::Clubs),
            card(Rank::Ten, Suit::Diamonds),
        ])
        .unwrap();
        assert!(!pair.beats(&single, DuplicateRule::FirstDealtWins));
        assert!(!single.beats(&pair, DuplicateRule::FirstDealtWins));
    }

    #[test]
    fn equal_rank_combo_broken_by_suit() {
        let low = Combo::new(vec![card(Rank::Nine, Suit::Diamonds)]).unwrap();
        let high = Combo::new(vec![card(Rank::Nine, Suit::Clubs)]).unwrap();
        assert!(high.beats(&low, DuplicateRule::FirstDealtWins));
    }

    #[test]
    fn equal_rank_and_suit_combo_never_beats_itself() {
        let a = Combo::new(vec![card(Rank::Nine, Suit::Clubs)]).unwrap();
        let b = Combo::new(vec![card(Rank::Nine, Suit::Clubs)]).unwrap();
        assert!(!a.beats(&b, DuplicateRule::FirstDealtWins));
    }

    #[test]
    fn top_card_is_public_and_returns_the_highest_card_in_the_combo() {
        let combo = Combo::new(vec![
            card(Rank::Seven, Suit::Diamonds),
            card(Rank::Seven, Suit::Clubs),
        ])
        .unwrap();
        assert_eq!(
            combo.top_card(DuplicateRule::FirstDealtWins),
            card(Rank::Seven, Suit::Clubs)
        );
    }

    #[test]
    fn cards_returns_the_cards_the_combo_was_built_from() {
        let cards = vec![
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Seven, Suit::Diamonds),
        ];
        let combo = Combo::new(cards.clone()).unwrap();
        assert_eq!(combo.cards(), &cards[..]);
    }
}
