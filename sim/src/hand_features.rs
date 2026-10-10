//! Summary of a freshly dealt hand, used by the cheap luck estimator
//! (`crate::skill`) to measure how good a seat's deal was.

use engine::{Card, Rank};

/// Features of one seat's round-1 hand, taken after the deal and before
/// any exchange.
///
/// Rank groups are the cards of the same rank in the hand (with a double
/// deck a group can hold more than 4 cards). `pairs` counts groups of
/// exactly 2 cards, `triples` groups of exactly 3, `quads` groups of 4
/// or more. Strength is the position in the 52-card order (rank first,
/// then suit), scaled to `[0, 1]`; the double-deck duplicate tiebreak is
/// ignored.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HandFeatures {
    /// Cards of rank Queen or higher.
    pub high_cards: u32,
    pub pairs: u32,
    pub triples: u32,
    pub quads: u32,
    /// Strength of the weakest card (0 for an empty hand).
    pub lowest_strength: f64,
    /// Sum of card strengths divided by the hand size (0 if empty).
    pub total_strength: f64,
}

fn strength(card: Card) -> f64 {
    f64::from(card.rank as u8 * 4 + card.suit as u8) / 51.0
}

impl HandFeatures {
    /// # Panics
    ///
    /// Never in practice: a hand has at most 26 cards.
    #[must_use]
    pub fn from_hand(hand: &[Card]) -> Self {
        let mut per_rank = [0u32; 13];
        let mut high_cards = 0;
        let mut lowest = f64::INFINITY;
        let mut sum = 0.0;
        for &card in hand {
            per_rank[card.rank as usize] += 1;
            high_cards += u32::from(card.rank >= Rank::Queen);
            let s = strength(card);
            lowest = lowest.min(s);
            sum += s;
        }
        let groups = |pred: fn(u32) -> bool| {
            u32::try_from(per_rank.iter().filter(|&&n| pred(n)).count()).expect("at most 13")
        };
        let (lowest_strength, total_strength) = if hand.is_empty() {
            (0.0, 0.0)
        } else {
            #[allow(clippy::cast_precision_loss)] // hands hold at most 26 cards
            (lowest, sum / hand.len() as f64)
        };
        Self {
            high_cards,
            pairs: groups(|n| n == 2),
            triples: groups(|n| n == 3),
            quads: groups(|n| n >= 4),
            lowest_strength,
            total_strength,
        }
    }

    /// The regression design columns, in a fixed order.
    #[must_use]
    pub fn as_vector(&self) -> [f64; 6] {
        [
            f64::from(self.high_cards),
            f64::from(self.pairs),
            f64::from(self.triples),
            f64::from(self.quads),
            self.lowest_strength,
            self.total_strength,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::Suit;

    fn c(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit, 0)
    }

    #[test]
    fn counts_groups_and_high_cards() {
        let hand = [
            c(Rank::Two, Suit::Diamonds),
            c(Rank::Two, Suit::Clubs),
            c(Rank::King, Suit::Hearts),
            c(Rank::King, Suit::Spades),
            c(Rank::King, Suit::Clubs),
            c(Rank::Ace, Suit::Clubs),
            c(Rank::Ace, Suit::Clubs),
            c(Rank::Ace, Suit::Hearts),
            c(Rank::Ace, Suit::Diamonds),
            c(Rank::Ace, Suit::Spades),
        ];
        let f = HandFeatures::from_hand(&hand);
        assert_eq!((f.high_cards, f.pairs, f.triples, f.quads), (8, 1, 1, 1));
        assert!((f.lowest_strength - 0.0).abs() < 1e-12);
        let expected: f64 = hand.iter().map(|&x| strength(x)).sum::<f64>() / 10.0;
        assert!((f.total_strength - expected).abs() < 1e-12);
    }

    #[test]
    fn strongest_card_has_strength_one() {
        let f = HandFeatures::from_hand(&[c(Rank::Ace, Suit::Clubs)]);
        assert!((f.lowest_strength - 1.0).abs() < 1e-12);
        assert!((f.total_strength - 1.0).abs() < 1e-12);
    }
}
