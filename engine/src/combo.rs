//! A `Combo` is a set of cards of equal rank played together (a single,
//! pair, triple, ...). Straights and bombs are out of scope for this
//! phase (see docs/RULES.md, "Deferred / Out-of-scope Rules").

use std::cmp::Ordering;

use crate::card::{Card, DuplicateRule};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Combo {
    cards: Vec<Card>,
}

impl Combo {
    /// Builds a combo from cards that must all share the same rank and
    /// must be non-empty. Returns `None` otherwise.
    #[must_use]
    pub fn new(cards: Vec<Card>) -> Option<Self> {
        let first = cards.first()?;
        if cards.iter().all(|c| c.rank == first.rank) {
            Some(Self { cards })
        } else {
            None
        }
    }

    #[must_use]
    pub fn size(&self) -> usize {
        self.cards.len()
    }

    /// The representative card used for comparison: since every card in
    /// a combo shares a rank, the highest card (by suit/duplicate
    /// tiebreak) stands in for the whole combo.
    fn top_card(&self, duplicate_rule: DuplicateRule) -> &Card {
        self.cards
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
                .compare(previous.top_card(duplicate_rule), duplicate_rule)
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
}
