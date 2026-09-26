//! Building a fresh, unshuffled standard deck. Shuffling is the caller's
//! job (see docs/RULES.md, "Duplicate cards") — this only knows which
//! cards exist.

use crate::card::{Card, DeckVariant, Rank, Suit};

/// All cards for a `variant` deck, in a fixed canonical order (unshuffled,
/// every `deal_index` set to `0`). In the `Double` variant, each
/// (rank, suit) pair appears twice consecutively. Reassign `deal_index` by
/// shuffle position before dealing, so `DuplicateRule` can disambiguate
/// true duplicates (see docs/RULES.md, "Duplicate cards").
#[must_use]
pub fn standard_deck(variant: DeckVariant) -> Vec<Card> {
    let suits = [Suit::Diamonds, Suit::Hearts, Suit::Spades, Suit::Clubs];
    let ranks = [
        Rank::Two,
        Rank::Three,
        Rank::Four,
        Rank::Five,
        Rank::Six,
        Rank::Seven,
        Rank::Eight,
        Rank::Nine,
        Rank::Ten,
        Rank::Jack,
        Rank::Queen,
        Rank::King,
        Rank::Ace,
    ];
    let capacity = if variant == DeckVariant::Double { 104 } else { 52 };
    let mut cards = Vec::with_capacity(capacity);
    for suit in suits {
        for rank in ranks {
            cards.push(Card::new(rank, suit, 0));
            if variant == DeckVariant::Double {
                cards.push(Card::new(rank, suit, 0));
            }
        }
    }
    cards
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn single_deck_has_52_unique_cards() {
        let cards = standard_deck(DeckVariant::Single);
        assert_eq!(cards.len(), 52);
        let mut counts: HashMap<(Rank, Suit), u32> = HashMap::new();
        for card in &cards {
            *counts.entry((card.rank, card.suit)).or_insert(0) += 1;
        }
        assert_eq!(counts.len(), 52);
        assert!(counts.values().all(|&count| count == 1));
    }

    #[test]
    fn double_deck_has_104_cards_each_pair_appearing_twice() {
        let cards = standard_deck(DeckVariant::Double);
        assert_eq!(cards.len(), 104);
        let mut counts: HashMap<(Rank, Suit), u32> = HashMap::new();
        for card in &cards {
            *counts.entry((card.rank, card.suit)).or_insert(0) += 1;
        }
        assert_eq!(counts.len(), 52);
        assert!(counts.values().all(|&count| count == 2));
    }
}
