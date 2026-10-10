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
    pub mean_strength: f64,
    /// Cards in the dealt hand. Uneven deals give the first seats one
    /// extra card (at 3, 5 and 6 players), a real seat effect. Absent in
    /// results saved before the feature existed (reads as 0).
    #[serde(default)]
    pub hand_size: u32,
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
        let (lowest_strength, mean_strength) = if hand.is_empty() {
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
            mean_strength,
            hand_size: u32::try_from(hand.len()).expect("at most 26"),
        }
    }

    /// The six hand-quality columns, in a fixed order (everything except
    /// `hand_size`). The luck-adjusted fitness term of the trainer
    /// (`crate::training`) regresses on exactly these.
    #[must_use]
    pub fn as_vector(&self) -> [f64; 6] {
        [
            f64::from(self.high_cards),
            f64::from(self.pairs),
            f64::from(self.triples),
            f64::from(self.quads),
            self.lowest_strength,
            self.mean_strength,
        ]
    }

    /// The estimator's regression design columns: `as_vector` followed by
    /// `hand_size`.
    #[must_use]
    pub fn design_vector(&self) -> [f64; 7] {
        let q = self.as_vector();
        [
            q[0],
            q[1],
            q[2],
            q[3],
            q[4],
            q[5],
            f64::from(self.hand_size),
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
        assert!((f.mean_strength - expected).abs() < 1e-12);
        assert_eq!(f.hand_size, 10);
    }

    #[test]
    fn strongest_card_has_strength_one() {
        let f = HandFeatures::from_hand(&[c(Rank::Ace, Suit::Clubs)]);
        assert!((f.lowest_strength - 1.0).abs() < 1e-12);
        assert!((f.mean_strength - 1.0).abs() < 1e-12);
    }
}
