//! Short card names for reports and the `solve` command: `9H`, `10S`, `AC`.

use engine::{Card, Rank, Suit};

const RANKS: [(Rank, &str); 13] = [
    (Rank::Two, "2"),
    (Rank::Three, "3"),
    (Rank::Four, "4"),
    (Rank::Five, "5"),
    (Rank::Six, "6"),
    (Rank::Seven, "7"),
    (Rank::Eight, "8"),
    (Rank::Nine, "9"),
    (Rank::Ten, "10"),
    (Rank::Jack, "J"),
    (Rank::Queen, "Q"),
    (Rank::King, "K"),
    (Rank::Ace, "A"),
];

const SUITS: [(Suit, char); 4] = [
    (Suit::Diamonds, 'D'),
    (Suit::Hearts, 'H'),
    (Suit::Spades, 'S'),
    (Suit::Clubs, 'C'),
];

/// `9H`, `10S`, `AC`.
#[must_use]
pub fn card_token(card: &Card) -> String {
    let rank = RANKS
        .iter()
        .find(|(r, _)| *r == card.rank)
        .map_or("?", |(_, token)| *token);
    let suit = SUITS
        .iter()
        .find(|(s, _)| *s == card.suit)
        .map_or('?', |(_, c)| *c);
    format!("{rank}{suit}")
}

/// Inverse of [`card_token`], case-insensitive.
#[must_use]
pub fn parse_card_token(token: &str) -> Option<(Rank, Suit)> {
    let token = token.trim().to_ascii_uppercase();
    let suit_char = token.chars().last()?;
    let rank_text = &token[..token.len() - suit_char.len_utf8()];
    let suit = SUITS
        .iter()
        .find(|(_, c)| *c == suit_char)
        .map(|(s, _)| *s)?;
    let rank = RANKS
        .iter()
        .find(|(_, text)| *text == rank_text)
        .map(|(r, _)| *r)?;
    Some((rank, suit))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_round_trip_for_every_card() {
        for variant in [engine::DeckVariant::Single, engine::DeckVariant::Double] {
            for card in engine::standard_deck(variant) {
                assert_eq!(
                    parse_card_token(&card_token(&card)),
                    Some((card.rank, card.suit))
                );
            }
        }
    }

    #[test]
    fn parsing_is_case_insensitive_and_rejects_garbage() {
        assert_eq!(parse_card_token("10s"), Some((Rank::Ten, Suit::Spades)));
        assert_eq!(parse_card_token("ac"), Some((Rank::Ace, Suit::Clubs)));
        assert_eq!(parse_card_token("1H"), None);
        assert_eq!(parse_card_token("9X"), None);
        assert_eq!(parse_card_token(""), None);
    }
}
