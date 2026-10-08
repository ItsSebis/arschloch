//! The mandatory pre-round card exchange ("Drücken"). See docs/RULES.md,
//! "Card Exchange (\"Drücken\")".

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::card::{Card, DuplicateRule};
use crate::role::{exchange_counts_for_player_count, roles_for_player_count, Role};

/// Whether the lower role of an exchange pair chooses which cards to give.
///
/// Under `Forced` (the rules of the game, and the default) it must hand over
/// its highest cards; under `Free` (how the simulator behaved until Phase 15,
/// kept to reproduce earlier results and as a game modifier) it, or its
/// strategy, may give any cards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExchangeRule {
    Free,
    #[default]
    Forced,
}

impl fmt::Display for ExchangeRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Free => "free",
            Self::Forced => "forced",
        })
    }
}

impl FromStr for ExchangeRule {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        match text.trim().to_ascii_lowercase().as_str() {
            "free" => Ok(Self::Free),
            "forced" => Ok(Self::Forced),
            other => Err(format!(
                "unknown exchange rule `{other}`; expected free or forced"
            )),
        }
    }
}

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
    /// A `choose_cards_to_give` callback (see `exchange_with_selection`)
    /// returned something other than exactly `count` distinct cards
    /// drawn from the hand it was given.
    InvalidSelection,
}

/// Like `exchange`, but the lower role's mandatory best-`count` cards
/// are chosen by `choose_cards_to_give` instead of a naive highest-N
/// sort (`docs/ROADMAP.md`, Phase 5, "Smart exchange"). The higher
/// role's worst-`count` cards it gives back are still chosen naively —
/// Phase 5 only makes the giving-away-your-best-cards side
/// strategy-aware.
///
/// Validated before anything is mutated (same failure atomicity as
/// `exchange`): `choose_cards_to_give(seat, hand, count,
/// duplicate_rule)` is called once per exchanging low seat with that
/// seat's untouched hand, and must return exactly `count` distinct
/// cards each present in `hand` — anything else is
/// `ExchangeError::InvalidSelection` and no hand is changed.
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
/// - [`ExchangeError::InvalidSelection`] if `choose_cards_to_give`
///   returns something other than exactly `count` distinct cards from
///   the given hand.
///
/// # Panics
///
/// Panics if `role_by_seat` passes validation but a role it confirmed is
/// present cannot then be found in it; this would indicate an internal
/// invariant violation, not a normal input error.
pub fn exchange_with_selection(
    hands: &mut [Vec<Card>],
    role_by_seat: &[Role],
    duplicate_rule: DuplicateRule,
    mut choose_cards_to_give: impl FnMut(usize, &[Card], usize, DuplicateRule) -> Vec<Card>,
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

    // Plan pass: ask each low seat what it's giving and validate the
    // answer against its still-untouched hand, before mutating
    // anything.
    let mut planned_gives: Vec<Option<Vec<Card>>> = vec![None; hands.len()];
    for (i, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let count = usize::from(count);
        let low_seat = seat_of(roles[roles.len() - 1 - i]);
        let selected = choose_cards_to_give(low_seat, &hands[low_seat], count, duplicate_rule);
        if selected.len() != count || !is_sub_multiset(&selected, &hands[low_seat]) {
            return Err(ExchangeError::InvalidSelection);
        }
        planned_gives[low_seat] = Some(selected);
    }

    // Apply pass: identical shape to the old `exchange`, except the low
    // seat's outgoing cards come from `planned_gives` instead of
    // `take_highest`.
    let mut incoming: Vec<Vec<Card>> = vec![Vec::new(); hands.len()];
    for (i, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let count = usize::from(count);
        let high_seat = seat_of(roles[i]);
        let low_seat = seat_of(roles[roles.len() - 1 - i]);

        let selected = planned_gives[low_seat]
            .take()
            .expect("every exchanging low seat was planned above");
        let from_low = remove_selected(&mut hands[low_seat], &selected);
        let from_high = take_lowest(&mut hands[high_seat], count, duplicate_rule);

        incoming[high_seat].extend(from_low);
        incoming[low_seat].extend(from_high);
    }

    for (hand, extra) in hands.iter_mut().zip(incoming) {
        hand.extend(extra);
    }
    Ok(())
}

/// The naive top/bottom-N exchange from Phase 1, now implemented as
/// `exchange_with_selection` with a selection callback that reproduces
/// the old `take_highest` behavior exactly. See `exchange_with_selection`
/// for errors and panics.
///
/// # Errors
///
/// See [`exchange_with_selection`] for error cases; `exchange` delegates
/// to it.
///
/// # Panics
///
/// See [`exchange_with_selection`] for panic conditions.
pub fn exchange(
    hands: &mut [Vec<Card>],
    role_by_seat: &[Role],
    duplicate_rule: DuplicateRule,
) -> Result<(), ExchangeError> {
    exchange_with_selection(
        hands,
        role_by_seat,
        duplicate_rule,
        |_seat, hand, count, duplicate_rule| {
            let mut sorted = hand.to_vec();
            sorted.sort_by(|a, b| a.compare(b, duplicate_rule));
            sorted.split_off(sorted.len() - count)
        },
    )
}

/// The exchange under `rule`: `Forced` hands over the highest cards and never
/// calls `choose_cards_to_give`; `Free` asks it (see
/// [`exchange_with_selection`]).
///
/// # Errors
///
/// As [`exchange_with_selection`].
pub fn exchange_with_rule(
    hands: &mut [Vec<Card>],
    role_by_seat: &[Role],
    duplicate_rule: DuplicateRule,
    rule: ExchangeRule,
    choose_cards_to_give: impl FnMut(usize, &[Card], usize, DuplicateRule) -> Vec<Card>,
) -> Result<(), ExchangeError> {
    match rule {
        ExchangeRule::Forced => exchange(hands, role_by_seat, duplicate_rule),
        ExchangeRule::Free => {
            exchange_with_selection(hands, role_by_seat, duplicate_rule, choose_cards_to_give)
        }
    }
}

fn validate_role_mapping(role_by_seat: &[Role], roles: &[Role]) -> Result<(), ExchangeError> {
    for &role in roles {
        if role_by_seat.iter().filter(|&&r| r == role).count() != 1 {
            return Err(ExchangeError::InvalidRoleMapping);
        }
    }
    Ok(())
}

/// Removes and returns the `count` lowest cards from `hand` (by
/// `Card::compare` under `duplicate_rule`).
///
/// Sorts `hand` in place (ascending) as part of finding those cards, so
/// the cards left behind end up in ascending order afterward — this is
/// why some tests' expected remaining-hand contents appear pre-sorted.
fn take_lowest(hand: &mut Vec<Card>, count: usize, duplicate_rule: DuplicateRule) -> Vec<Card> {
    hand.sort_by(|a, b| a.compare(b, duplicate_rule));
    hand.drain(0..count).collect()
}

fn is_sub_multiset(selected: &[Card], hand: &[Card]) -> bool {
    let mut remaining = hand.to_vec();
    for card in selected {
        match remaining.iter().position(|c| c == card) {
            Some(pos) => {
                remaining.remove(pos);
            }
            None => return false,
        }
    }
    true
}

fn remove_selected(hand: &mut Vec<Card>, selected: &[Card]) -> Vec<Card> {
    selected
        .iter()
        .map(|card| {
            let pos = hand
                .iter()
                .position(|c| c == card)
                .expect("is_sub_multiset already validated this selection");
            hand.remove(pos)
        })
        .collect()
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

    #[test]
    fn exchange_with_selection_uses_the_provided_cards() {
        // For 4 players, exchange_counts_for_player_count(4) == [2, 1]: the
        // outer pair (President/Arschloch) exchanges 2 cards, the inner pair
        // (Vize/ViceArschloch) exchanges 1. Seat 3's hand has exactly 2
        // cards and the outer pair's count is 2, so its whole hand moves
        // regardless of selection order — that pair can't distinguish naive
        // from custom. Assert on the inner pair instead, where count (1) is
        // smaller than the low seat's hand size (2), so which card is chosen
        // actually matters.
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

        // Opposite of naive: always give the LOWEST `count` cards instead
        // of the highest.
        exchange_with_selection(
            &mut hands,
            &role_by_seat,
            DuplicateRule::FirstDealtWins,
            |_seat, hand, count, duplicate_rule| {
                let mut sorted = hand.to_vec();
                sorted.sort_by(|a, b| a.compare(b, duplicate_rule));
                sorted.truncate(count);
                sorted
            },
        )
        .unwrap();

        // Seat 2 (ViceArschloch, giving 1 card up to seat 1/Vize) gave its
        // LOWEST card (Six) instead of the naive highest (Seven).
        assert!(hands[1].contains(&card(Rank::Six, Suit::Clubs)));
        assert!(!hands[1].contains(&card(Rank::Seven, Suit::Clubs)));
    }

    fn lowest_giver(_seat: usize, hand: &[Card], count: usize, rule: DuplicateRule) -> Vec<Card> {
        let mut sorted = hand.to_vec();
        sorted.sort_by(|a, b| a.compare(b, rule));
        sorted.truncate(count);
        sorted
    }

    fn inner_pair_scenario() -> (Vec<Role>, Vec<Vec<Card>>) {
        (
            vec![
                Role::President,
                Role::Vize,
                Role::ViceArschloch,
                Role::Arschloch,
            ],
            vec![
                vec![card(Rank::Two, Suit::Clubs), card(Rank::Three, Suit::Clubs)],
                vec![card(Rank::Four, Suit::Clubs), card(Rank::Five, Suit::Clubs)],
                vec![card(Rank::Six, Suit::Clubs), card(Rank::Seven, Suit::Clubs)],
                vec![card(Rank::King, Suit::Clubs), card(Rank::Ace, Suit::Clubs)],
            ],
        )
    }

    #[test]
    fn the_forced_rule_never_asks_the_chooser_and_gives_the_highest_cards() {
        let (roles, mut hands) = inner_pair_scenario();
        let mut asked = false;
        exchange_with_rule(
            &mut hands,
            &roles,
            DuplicateRule::FirstDealtWins,
            ExchangeRule::Forced,
            |seat, hand, count, rule| {
                asked = true;
                lowest_giver(seat, hand, count, rule)
            },
        )
        .unwrap();
        assert!(!asked, "a forced exchange has nothing to choose");
        // Seat 2 gave its HIGHEST card (Seven), not the lowest the chooser would.
        assert!(hands[1].contains(&card(Rank::Seven, Suit::Clubs)));
        assert!(!hands[1].contains(&card(Rank::Six, Suit::Clubs)));
        // And it is exactly the old `exchange`.
        let (roles, mut plain) = inner_pair_scenario();
        exchange(&mut plain, &roles, DuplicateRule::FirstDealtWins).unwrap();
        assert_eq!(hands, plain);
    }

    #[test]
    fn the_free_rule_honours_the_chooser() {
        let (roles, mut hands) = inner_pair_scenario();
        exchange_with_rule(
            &mut hands,
            &roles,
            DuplicateRule::FirstDealtWins,
            ExchangeRule::Free,
            lowest_giver,
        )
        .unwrap();
        assert!(hands[1].contains(&card(Rank::Six, Suit::Clubs)));
    }

    #[test]
    fn the_forced_rule_takes_the_n_highest_at_every_table_size() {
        use crate::role::{exchange_counts_for_player_count, roles_for_player_count};
        for players in 3..=6u8 {
            let roles = roles_for_player_count(players).unwrap().to_vec();
            let counts = exchange_counts_for_player_count(players).unwrap();
            // Seat s holds ranks s*13/n .. so the hands are strictly ordered.
            let mut hands: Vec<Vec<Card>> = (0..usize::from(players))
                .map(|seat| {
                    (0..8)
                        .map(|i| {
                            let rank = [
                                Rank::Two,
                                Rank::Three,
                                Rank::Four,
                                Rank::Five,
                                Rank::Six,
                                Rank::Seven,
                                Rank::Eight,
                                Rank::Nine,
                            ][i];
                            Card::new(
                                rank,
                                [Suit::Diamonds, Suit::Hearts, Suit::Spades, Suit::Clubs][seat % 4],
                                u8::try_from(seat * 8 + i).unwrap(),
                            )
                        })
                        .collect()
                })
                .collect();
            let before = hands.clone();
            let mut seat_roles = roles.clone();
            seat_roles.rotate_left(1); // roles by seat: seat 0 is not the President
            exchange_with_rule(
                &mut hands,
                &seat_roles,
                DuplicateRule::FirstDealtWins,
                ExchangeRule::Forced,
                lowest_giver,
            )
            .unwrap();
            for (i, &count) in counts.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                let high = seat_roles.iter().position(|&r| r == roles[i]).unwrap();
                let low = seat_roles
                    .iter()
                    .position(|&r| r == roles[roles.len() - 1 - i])
                    .unwrap();
                let mut giver = before[low].clone();
                giver.sort_by(|a, b| a.compare(b, DuplicateRule::FirstDealtWins));
                let top: Vec<Card> = giver[giver.len() - usize::from(count)..].to_vec();
                assert!(
                    top.iter().all(|c| hands[high].contains(c)),
                    "{players}p: the {count} highest went to the higher role"
                );
                assert!(top.iter().all(|c| !hands[low].contains(c)));
            }
        }
    }

    #[test]
    fn the_rule_names_round_trip() {
        assert_eq!(ExchangeRule::default(), ExchangeRule::Forced);
        assert_eq!("free".parse::<ExchangeRule>(), Ok(ExchangeRule::Free));
        assert_eq!(" FORCED ".parse::<ExchangeRule>(), Ok(ExchangeRule::Forced));
        assert!("sometimes".parse::<ExchangeRule>().is_err());
        assert_eq!(ExchangeRule::Forced.to_string(), "forced");
        assert_eq!(
            serde_json::to_string(&ExchangeRule::Free).unwrap(),
            "\"free\""
        );
    }

    #[test]
    fn exchange_with_selection_rejects_wrong_count() {
        let role_by_seat = vec![Role::President, Role::Dorftrottel, Role::Arschloch];
        let mut hands = vec![
            vec![card(Rank::Two, Suit::Clubs)],
            vec![card(Rank::Five, Suit::Clubs)],
            vec![card(Rank::King, Suit::Clubs)],
        ];
        let result = exchange_with_selection(
            &mut hands,
            &role_by_seat,
            DuplicateRule::FirstDealtWins,
            |_seat, _hand, _count, _duplicate_rule| Vec::new(),
        );
        assert_eq!(result, Err(ExchangeError::InvalidSelection));
    }

    #[test]
    fn exchange_with_selection_rejects_card_not_in_hand() {
        let role_by_seat = vec![Role::President, Role::Dorftrottel, Role::Arschloch];
        let mut hands = vec![
            vec![card(Rank::Two, Suit::Clubs)],
            vec![card(Rank::Five, Suit::Clubs)],
            vec![card(Rank::King, Suit::Clubs)],
        ];
        let result = exchange_with_selection(
            &mut hands,
            &role_by_seat,
            DuplicateRule::FirstDealtWins,
            |_seat, _hand, _count, _duplicate_rule| vec![card(Rank::Nine, Suit::Diamonds)],
        );
        assert_eq!(result, Err(ExchangeError::InvalidSelection));
    }

    #[test]
    fn exchange_with_selection_rejects_duplicate_selection() {
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
        let result = exchange_with_selection(
            &mut hands,
            &role_by_seat,
            DuplicateRule::FirstDealtWins,
            |_seat, hand, count, _duplicate_rule| {
                // Return the same card `count` times instead of `count`
                // distinct cards.
                vec![hand[0]; count]
            },
        );
        assert_eq!(result, Err(ExchangeError::InvalidSelection));
    }

    #[test]
    fn exchange_with_selection_leaves_hands_untouched_when_selection_invalid() {
        // 6 players -> 3 exchanging pairs. Seat order matches
        // roles_for_player_count(6): President, Vize, Offizier, Dummkopf,
        // ViceArschloch, Arschloch.
        let role_by_seat = vec![
            Role::President,
            Role::Vize,
            Role::Offizier,
            Role::Dummkopf,
            Role::ViceArschloch,
            Role::Arschloch,
        ];
        let original_hands = vec![
            vec![
                card(Rank::Two, Suit::Clubs),
                card(Rank::Three, Suit::Clubs),
                card(Rank::Four, Suit::Clubs),
            ],
            vec![card(Rank::Five, Suit::Clubs), card(Rank::Six, Suit::Clubs)],
            vec![
                card(Rank::Seven, Suit::Clubs),
                card(Rank::Eight, Suit::Clubs),
            ],
            vec![card(Rank::Nine, Suit::Clubs), card(Rank::Ten, Suit::Clubs)],
            vec![
                card(Rank::Jack, Suit::Clubs),
                card(Rank::Queen, Suit::Clubs),
            ],
            vec![
                card(Rank::King, Suit::Clubs),
                card(Rank::Ace, Suit::Clubs),
                card(Rank::Two, Suit::Hearts),
            ],
        ];
        let mut hands = original_hands.clone();

        // Valid for the first (outermost) pair's low seat (seat 5,
        // Arschloch, giving 3 cards to President), invalid for the second
        // pair's low seat (seat 4, ViceArschloch, giving to Vize) — proves
        // an earlier valid selection doesn't get applied before a later
        // one is found invalid.
        let result = exchange_with_selection(
            &mut hands,
            &role_by_seat,
            DuplicateRule::FirstDealtWins,
            |seat, hand, count, duplicate_rule| {
                if seat == 4 {
                    Vec::new() // wrong count -> InvalidSelection
                } else {
                    let mut sorted = hand.to_vec();
                    sorted.sort_by(|a, b| a.compare(b, duplicate_rule));
                    sorted.split_off(sorted.len() - count)
                }
            },
        );

        assert_eq!(result, Err(ExchangeError::InvalidSelection));
        assert_eq!(hands, original_hands);
    }
}
