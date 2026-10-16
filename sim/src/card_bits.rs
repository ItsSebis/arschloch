//! A card as one bit of a `u128`, by its `deal_index`: the sets of cards of a
//! round (seen, unseen, in a hand, on the table) become masks.

use engine::Card;

/// The bit of `card`. Every card of a round has its own `deal_index` (the
/// match runner numbers the deck; decks hold at most 104 cards).
#[inline]
pub(crate) fn card_bit(card: Card) -> u128 {
    1u128
        .checked_shl(u32::from(card.deal_index))
        .expect("deal indices of a round's deck are below 128")
}

/// The mask of all `cards`.
pub(crate) fn card_bits(cards: &[Card]) -> u128 {
    cards.iter().fold(0, |mask, card| mask | card_bit(*card))
}
