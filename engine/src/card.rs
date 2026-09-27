//! Card, suit, and rank types, including the project's house-rule suit
//! ordering and double-deck duplicate-card tiebreak. See docs/RULES.md,
//! "Card Ranking" and "Duplicate cards".

use std::cmp::Ordering;

/// Suit ranking is a house rule for this project (see docs/RULES.md):
/// Diamonds < Hearts < Spades < Clubs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Suit {
    Diamonds,
    Hearts,
    Spades,
    Clubs,
}

/// Standard poker rank order, low to high.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Rank {
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Jack,
    Queen,
    King,
    Ace,
}

/// Which physical deck(s) a match is played with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeckVariant {
    Single,
    Double,
}

/// Resolves ties between two cards that are identical in rank and suit,
/// which can only happen in the `Double` deck variant. See docs/RULES.md,
/// "Duplicate cards".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuplicateRule {
    FirstDealtWins,
    LastDealtWins,
}

/// A single playing card. `deal_index` disambiguates true duplicate
/// rank+suit pairs in the double-deck variant (the copy's position in
/// deal order) and is otherwise ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Card {
    pub rank: Rank,
    pub suit: Suit,
    pub deal_index: u8,
}

impl Card {
    #[must_use]
    pub fn new(rank: Rank, suit: Suit, deal_index: u8) -> Self {
        Self {
            rank,
            suit,
            deal_index,
        }
    }

    /// Total order over cards for a given match's duplicate-tiebreak
    /// rule. Rank is compared first, then suit, then (only when rank and
    /// suit are both equal) deal order.
    #[must_use]
    pub fn compare(&self, other: &Card, duplicate_rule: DuplicateRule) -> Ordering {
        self.rank
            .cmp(&other.rank)
            .then_with(|| self.suit.cmp(&other.suit))
            .then_with(|| match duplicate_rule {
                DuplicateRule::FirstDealtWins => other.deal_index.cmp(&self.deal_index),
                DuplicateRule::LastDealtWins => self.deal_index.cmp(&other.deal_index),
            })
    }
}

/// Groups `hand` by rank, in ascending rank order. Each inner `Vec` is
/// every card of one rank (a "reserve" a strategy might want to keep
/// together rather than split up — see `docs/ROADMAP.md`, Phase 5).
#[must_use]
pub fn rank_groups(hand: &[Card]) -> Vec<Vec<Card>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_orders_low_to_high() {
        assert!(Rank::Two < Rank::Three);
        assert!(Rank::King < Rank::Ace);
    }

    #[test]
    fn suit_orders_per_house_rule() {
        assert!(Suit::Diamonds < Suit::Hearts);
        assert!(Suit::Hearts < Suit::Spades);
        assert!(Suit::Spades < Suit::Clubs);
    }

    #[test]
    fn higher_rank_beats_lower_rank_regardless_of_suit() {
        let low = Card::new(Rank::Seven, Suit::Clubs, 0);
        let high = Card::new(Rank::Eight, Suit::Diamonds, 0);
        assert_eq!(
            low.compare(&high, DuplicateRule::FirstDealtWins),
            Ordering::Less
        );
    }

    #[test]
    fn equal_rank_breaks_tie_by_suit() {
        let diamonds = Card::new(Rank::Nine, Suit::Diamonds, 0);
        let clubs = Card::new(Rank::Nine, Suit::Clubs, 0);
        assert_eq!(
            diamonds.compare(&clubs, DuplicateRule::FirstDealtWins),
            Ordering::Less
        );
    }

    #[test]
    fn true_duplicate_first_dealt_wins() {
        let first = Card::new(Rank::Ten, Suit::Hearts, 0);
        let second = Card::new(Rank::Ten, Suit::Hearts, 1);
        assert_eq!(
            first.compare(&second, DuplicateRule::FirstDealtWins),
            Ordering::Greater
        );
        assert_eq!(
            second.compare(&first, DuplicateRule::FirstDealtWins),
            Ordering::Less
        );
    }

    #[test]
    fn higher_rank_beats_lower_rank_regardless_of_deal_index() {
        let low = Card::new(Rank::Eight, Suit::Clubs, 9);
        let high = Card::new(Rank::Nine, Suit::Diamonds, 0);
        assert_eq!(
            low.compare(&high, DuplicateRule::FirstDealtWins),
            Ordering::Less
        );
    }

    #[test]
    fn equal_rank_breaks_tie_by_suit_regardless_of_deal_index() {
        let diamonds = Card::new(Rank::Nine, Suit::Diamonds, 9);
        let clubs = Card::new(Rank::Nine, Suit::Clubs, 0);
        assert_eq!(
            diamonds.compare(&clubs, DuplicateRule::FirstDealtWins),
            Ordering::Less
        );
    }

    #[test]
    fn true_duplicate_last_dealt_wins() {
        let first = Card::new(Rank::Ten, Suit::Hearts, 0);
        let second = Card::new(Rank::Ten, Suit::Hearts, 1);
        assert_eq!(
            first.compare(&second, DuplicateRule::LastDealtWins),
            Ordering::Less
        );
        assert_eq!(
            second.compare(&first, DuplicateRule::LastDealtWins),
            Ordering::Greater
        );
    }

    #[test]
    fn rank_groups_empty_hand_has_no_groups() {
        assert_eq!(rank_groups(&[]), Vec::<Vec<Card>>::new());
    }

    #[test]
    fn rank_groups_groups_same_rank_cards_together_in_ascending_rank_order() {
        let low_a = Card::new(Rank::Two, Suit::Clubs, 0);
        let low_b = Card::new(Rank::Two, Suit::Hearts, 1);
        let mid = Card::new(Rank::Five, Suit::Clubs, 2);
        let high = Card::new(Rank::King, Suit::Clubs, 3);
        let hand = vec![high, low_a, mid, low_b];

        let groups = rank_groups(&hand);

        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].len(), 2);
        assert!(groups[0].contains(&low_a));
        assert!(groups[0].contains(&low_b));
        assert_eq!(groups[1], vec![mid]);
        assert_eq!(groups[2], vec![high]);
    }
}
