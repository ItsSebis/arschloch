//! Dealing an already-ordered deck into hands, and finding the session's
//! very first leader. See docs/RULES.md, "Players & Deck" and "First
//! lead of a trick / round".

use crate::card::{Card, DuplicateRule};
use crate::SeatId;

/// Deals `cards` round-robin into `player_count` hands, in the order
/// given — card `i` goes to seat `i % player_count`, so the earliest
/// seats get the extra card(s) when `cards.len()` doesn't divide evenly,
/// matching docs/RULES.md's documented remainder rule. The caller is
/// responsible for shuffling `cards` beforehand; this function only
/// distributes them.
///
/// Returns `None` if `player_count` is 0, or if some seat would end up
/// with an empty hand (`cards.len() < player_count`).
#[must_use]
pub fn deal(cards: Vec<Card>, player_count: u8) -> Option<Vec<Vec<Card>>> {
    if player_count == 0 || cards.len() < usize::from(player_count) {
        return None;
    }
    let seat_count = usize::from(player_count);
    let mut hands = vec![Vec::new(); seat_count];
    for (i, card) in cards.into_iter().enumerate() {
        hands[i % seat_count].push(card);
    }
    Some(hands)
}

/// Finds which seat holds the single lowest card across all hands, by
/// `Card::compare` under `duplicate_rule`. Used only to seed the very
/// first leader of a session's first round (docs/RULES.md, "First
/// lead"); every later round's leader is that round's Arschloch instead.
///
/// Returns `None` if `hands` is empty or every hand is empty.
#[must_use]
pub fn lowest_card_holder(hands: &[Vec<Card>], duplicate_rule: DuplicateRule) -> Option<SeatId> {
    hands
        .iter()
        .enumerate()
        .flat_map(|(seat, hand)| hand.iter().map(move |card| (seat, card)))
        .min_by(|(_, a), (_, b)| a.compare(b, duplicate_rule))
        .map(|(seat, _)| SeatId::try_from(seat).expect("table sizes are capped at 6 seats"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{Rank, Suit};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit, 0)
    }

    #[test]
    fn deal_splits_cards_round_robin() {
        let cards = vec![
            card(Rank::Two, Suit::Clubs),
            card(Rank::Three, Suit::Clubs),
            card(Rank::Four, Suit::Clubs),
            card(Rank::Five, Suit::Clubs),
        ];
        let hands = deal(cards, 2).unwrap();
        assert_eq!(hands.len(), 2);
        assert_eq!(
            hands[0],
            vec![card(Rank::Two, Suit::Clubs), card(Rank::Four, Suit::Clubs)]
        );
        assert_eq!(
            hands[1],
            vec![card(Rank::Three, Suit::Clubs), card(Rank::Five, Suit::Clubs)]
        );
    }

    #[test]
    fn deal_gives_earliest_seats_the_remainder() {
        let cards = vec![
            card(Rank::Two, Suit::Clubs),
            card(Rank::Three, Suit::Clubs),
            card(Rank::Four, Suit::Clubs),
        ];
        let hands = deal(cards, 2).unwrap();
        assert_eq!(hands[0].len(), 2);
        assert_eq!(hands[1].len(), 1);
    }

    #[test]
    fn deal_rejects_zero_players() {
        assert!(deal(vec![card(Rank::Two, Suit::Clubs)], 0).is_none());
    }

    #[test]
    fn deal_rejects_too_few_cards_for_every_seat() {
        assert!(deal(vec![card(Rank::Two, Suit::Clubs)], 2).is_none());
    }

    #[test]
    fn lowest_card_holder_finds_the_minimum_across_hands() {
        let hands = vec![
            vec![card(Rank::King, Suit::Clubs)],
            vec![
                card(Rank::Two, Suit::Diamonds),
                card(Rank::Ace, Suit::Spades),
            ],
        ];
        assert_eq!(
            lowest_card_holder(&hands, DuplicateRule::FirstDealtWins),
            Some(1)
        );
    }

    #[test]
    fn lowest_card_holder_returns_none_for_empty_hands() {
        let hands: Vec<Vec<Card>> = vec![vec![], vec![]];
        assert_eq!(lowest_card_holder(&hands, DuplicateRule::FirstDealtWins), None);
    }
}
