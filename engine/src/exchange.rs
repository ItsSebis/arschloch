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
/// its highest cards; under `Free` (how the simulator behaved in the early baselines,
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
/// sort ("Smart exchange" in `docs/ROADMAP.md`). The higher
/// role's worst-`count` cards it gives back are still chosen naively —
/// this only makes the giving-away-your-best-cards side
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

/// The naive top/bottom-N exchange: `exchange_with_selection` with
/// [`take_highest`] as the selection. See `exchange_with_selection` for
/// errors and panics.
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
        |_seat, hand, count, duplicate_rule| take_highest(hand, count, duplicate_rule),
    )
}

/// The `count` highest cards of `hand` (by `Card::compare` under
/// `duplicate_rule`), ascending; `hand` is left untouched. The forced
/// exchange's choice, reused by strategies with no stronger opinion on which
/// cards to give up.
///
/// # Panics
///
/// If `count` exceeds the hand size.
#[must_use]
pub fn take_highest(hand: &[Card], count: usize, duplicate_rule: DuplicateRule) -> Vec<Card> {
    let mut sorted = hand.to_vec();
    sorted.sort_by(|a, b| a.compare(b, duplicate_rule));
    sorted.split_off(sorted.len() - count)
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
mod tests;
