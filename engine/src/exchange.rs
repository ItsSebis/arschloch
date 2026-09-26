//! The mandatory pre-round card exchange ("Drücken"). See docs/RULES.md,
//! "Card Exchange (\"Drücken\")".

use crate::card::{Card, DuplicateRule};
use crate::role::{exchange_counts_for_player_count, roles_for_player_count, Role};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExchangeError {
    /// `hands.len() != role_by_seat.len()`.
    SeatCountMismatch,
    /// `hands.len()` isn't a supported table size (3-6).
    UnsupportedPlayerCount,
    /// `role_by_seat` didn't contain every role for this player count
    /// exactly once.
    InvalidRoleMapping,
    /// A seat's hand had fewer cards than its required exchange count.
    NotEnoughCards,
}

/// Applies the mandatory "Drücken" exchange to freshly dealt hands for
/// the round about to start, using the role each seat held at the end
/// of the *previous* round (`role_by_seat[i]` = seat `i`'s previous
/// role). Naive tie-break: the lower role's highest N cards move to the
/// higher role, whose lowest N cards move back, both ordered by
/// `Card::compare` under `duplicate_rule` — no strategic selection
/// (that's a later phase). On any error, `hands` is left completely
/// unchanged.
///
/// # Errors
///
/// - [`ExchangeError::SeatCountMismatch`] if `hands.len() !=
///   role_by_seat.len()`.
/// - [`ExchangeError::UnsupportedPlayerCount`] if `hands.len()` isn't a
///   supported table size (3-6).
/// - [`ExchangeError::InvalidRoleMapping`] if `role_by_seat` doesn't
///   contain every role for this player count exactly once.
/// - [`ExchangeError::NotEnoughCards`] if a seat's hand has fewer cards
///   than its required exchange count.
///
/// # Panics
///
/// Panics if `role_by_seat` passes validation but a role it confirmed is
/// present cannot then be found in it; this would indicate an internal
/// invariant violation, not a normal input error.
pub fn exchange(
    hands: &mut [Vec<Card>],
    role_by_seat: &[Role],
    duplicate_rule: DuplicateRule,
) -> Result<(), ExchangeError> {
    if hands.len() != role_by_seat.len() {
        return Err(ExchangeError::SeatCountMismatch);
    }
    let player_count =
        u8::try_from(hands.len()).map_err(|_| ExchangeError::UnsupportedPlayerCount)?;
    let roles =
        roles_for_player_count(player_count).ok_or(ExchangeError::UnsupportedPlayerCount)?;
    let counts = exchange_counts_for_player_count(player_count)
        .ok_or(ExchangeError::UnsupportedPlayerCount)?;
    validate_role_mapping(role_by_seat, roles)?;

    let seat_of = |role: Role| -> usize {
        role_by_seat
            .iter()
            .position(|&r| r == role)
            .expect("validate_role_mapping already confirmed every role is present exactly once")
    };

    // Validate every pair has enough cards on both sides before mutating
    // anything, so a failed exchange leaves hands untouched.
    for (i, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let count = usize::from(count);
        let high_seat = seat_of(roles[i]);
        let low_seat = seat_of(roles[roles.len() - 1 - i]);
        if hands[low_seat].len() < count || hands[high_seat].len() < count {
            return Err(ExchangeError::NotEnoughCards);
        }
    }

    let mut incoming: Vec<Vec<Card>> = vec![Vec::new(); hands.len()];
    for (i, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let count = usize::from(count);
        let high_seat = seat_of(roles[i]);
        let low_seat = seat_of(roles[roles.len() - 1 - i]);

        let from_low = take_highest(&mut hands[low_seat], count, duplicate_rule);
        let from_high = take_lowest(&mut hands[high_seat], count, duplicate_rule);

        incoming[high_seat].extend(from_low);
        incoming[low_seat].extend(from_high);
    }

    for (hand, extra) in hands.iter_mut().zip(incoming) {
        hand.extend(extra);
    }
    Ok(())
}

fn validate_role_mapping(role_by_seat: &[Role], roles: &[Role]) -> Result<(), ExchangeError> {
    for &role in roles {
        if role_by_seat.iter().filter(|&&r| r == role).count() != 1 {
            return Err(ExchangeError::InvalidRoleMapping);
        }
    }
    Ok(())
}

fn take_highest(hand: &mut Vec<Card>, count: usize, duplicate_rule: DuplicateRule) -> Vec<Card> {
    hand.sort_by(|a, b| a.compare(b, duplicate_rule));
    hand.split_off(hand.len() - count)
}

fn take_lowest(hand: &mut Vec<Card>, count: usize, duplicate_rule: DuplicateRule) -> Vec<Card> {
    hand.sort_by(|a, b| a.compare(b, duplicate_rule));
    hand.drain(0..count).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{Rank, Suit};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit, 0)
    }

    #[test]
    fn four_player_exchange_swaps_highest_and_lowest() {
        let role_by_seat = vec![
            Role::President,
            Role::Vize,
            Role::ViceArschloch,
            Role::Arschloch,
        ];
        let mut hands = vec![
            vec![card(Rank::Two, Suit::Clubs), card(Rank::Three, Suit::Clubs)],
            vec![card(Rank::Four, Suit::Clubs), card(Rank::Five, Suit::Clubs)],
            vec![card(Rank::Six, Suit::Clubs), card(Rank::Seven, Suit::Clubs)],
            vec![card(Rank::King, Suit::Clubs), card(Rank::Ace, Suit::Clubs)],
        ];
        exchange(&mut hands, &role_by_seat, DuplicateRule::FirstDealtWins).unwrap();

        assert_eq!(
            hands[0],
            vec![card(Rank::King, Suit::Clubs), card(Rank::Ace, Suit::Clubs)]
        );
        assert_eq!(
            hands[1],
            vec![
                card(Rank::Five, Suit::Clubs),
                card(Rank::Seven, Suit::Clubs)
            ]
        );
        assert_eq!(
            hands[2],
            vec![card(Rank::Six, Suit::Clubs), card(Rank::Four, Suit::Clubs)]
        );
        assert_eq!(
            hands[3],
            vec![card(Rank::Two, Suit::Clubs), card(Rank::Three, Suit::Clubs)]
        );
    }

    #[test]
    fn three_player_lone_middle_role_is_untouched() {
        let role_by_seat = vec![Role::President, Role::Dorftrottel, Role::Arschloch];
        let dorftrottel_hand = vec![card(Rank::Nine, Suit::Clubs), card(Rank::Ten, Suit::Clubs)];
        let mut hands = vec![
            vec![card(Rank::Two, Suit::Clubs)],
            dorftrottel_hand.clone(),
            vec![card(Rank::Ace, Suit::Clubs)],
        ];
        exchange(&mut hands, &role_by_seat, DuplicateRule::FirstDealtWins).unwrap();
        assert_eq!(hands[1], dorftrottel_hand);
    }

    #[test]
    fn exchange_rejects_seat_count_mismatch() {
        let mut hands = vec![vec![card(Rank::Two, Suit::Clubs)]; 3];
        let role_by_seat = vec![Role::President, Role::Arschloch];
        assert_eq!(
            exchange(&mut hands, &role_by_seat, DuplicateRule::FirstDealtWins),
            Err(ExchangeError::SeatCountMismatch)
        );
    }

    #[test]
    fn exchange_rejects_unsupported_player_count() {
        let mut hands = vec![vec![card(Rank::Two, Suit::Clubs)]; 2];
        let role_by_seat = vec![Role::President, Role::Arschloch];
        assert_eq!(
            exchange(&mut hands, &role_by_seat, DuplicateRule::FirstDealtWins),
            Err(ExchangeError::UnsupportedPlayerCount)
        );
    }

    #[test]
    fn exchange_rejects_invalid_role_mapping() {
        let mut hands = vec![vec![card(Rank::Two, Suit::Clubs)]; 3];
        // President repeated, Arschloch missing.
        let role_by_seat = vec![Role::President, Role::President, Role::Dorftrottel];
        assert_eq!(
            exchange(&mut hands, &role_by_seat, DuplicateRule::FirstDealtWins),
            Err(ExchangeError::InvalidRoleMapping)
        );
    }

    #[test]
    fn exchange_rejects_not_enough_cards() {
        let mut hands = vec![
            vec![],
            vec![card(Rank::Nine, Suit::Clubs), card(Rank::Ten, Suit::Clubs)],
            vec![card(Rank::Ace, Suit::Clubs), card(Rank::King, Suit::Clubs)],
        ];
        let role_by_seat = vec![Role::President, Role::Dorftrottel, Role::Arschloch];
        assert_eq!(
            exchange(&mut hands, &role_by_seat, DuplicateRule::FirstDealtWins),
            Err(ExchangeError::NotEnoughCards)
        );
    }

    #[test]
    fn exchange_leaves_hands_untouched_when_it_fails() {
        let original = vec![
            vec![],
            vec![card(Rank::Nine, Suit::Clubs), card(Rank::Ten, Suit::Clubs)],
            vec![card(Rank::Ace, Suit::Clubs), card(Rank::King, Suit::Clubs)],
        ];
        let mut hands = original.clone();
        let role_by_seat = vec![Role::President, Role::Dorftrottel, Role::Arschloch];
        assert!(exchange(&mut hands, &role_by_seat, DuplicateRule::FirstDealtWins).is_err());
        assert_eq!(hands, original);
    }
}
