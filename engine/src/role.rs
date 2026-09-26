//! Roles and the per-player-count role/exchange-count tables from
//! docs/RULES.md, "Roles" and "Card Exchange (\"Drücken\")".

use crate::SeatId;

/// A player's role at the end of a round. Variants are listed here from
/// highest to lowest across all table sizes; which subset applies to a
/// given table is determined by `roles_for_player_count`. The derived
/// `Ord` follows this declaration order, so a higher role compares as
/// *less* (`President < Arschloch`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum Role {
    President,
    Vize,
    Offizier,
    Dorftrottel,
    Dummkopf,
    ViceArschloch,
    Arschloch,
}

/// The ordered list of roles for a table of `player_count` seats, highest
/// role first, as defined in docs/RULES.md, "Roles". Returns `None` for
/// unsupported table sizes (anything other than 3, 4, 5, or 6).
#[must_use]
pub fn roles_for_player_count(player_count: u8) -> Option<&'static [Role]> {
    match player_count {
        3 => Some(&[Role::President, Role::Dorftrottel, Role::Arschloch]),
        4 => Some(&[
            Role::President,
            Role::Vize,
            Role::ViceArschloch,
            Role::Arschloch,
        ]),
        5 => Some(&[
            Role::President,
            Role::Vize,
            Role::Dorftrottel,
            Role::ViceArschloch,
            Role::Arschloch,
        ]),
        6 => Some(&[
            Role::President,
            Role::Vize,
            Role::Offizier,
            Role::Dummkopf,
            Role::ViceArschloch,
            Role::Arschloch,
        ]),
        _ => None,
    }
}

/// How many cards each role-pair exchanges before a new round, indexed
/// from the outermost pair (President/Arschloch) inward, as defined in
/// docs/RULES.md, "Card Exchange". A lone unpaired middle role
/// (Dorftrottel at 3 or 5 players) exchanges 0 cards and appears as a
/// trailing `0` in the table with no partner.
#[must_use]
pub fn exchange_counts_for_player_count(player_count: u8) -> Option<&'static [u8]> {
    match player_count {
        3 => Some(&[1, 0]),
        4 => Some(&[2, 1]),
        5 => Some(&[2, 1, 0]),
        6 => Some(&[3, 2, 1]),
        _ => None,
    }
}

/// Maps a completed round's finishing order (the seat that emptied its
/// hand first, ..., the seat ranked last) to each seat's new `Role`, via
/// `roles_for_player_count`. Returns `None` if `player_count` is
/// unsupported, if `finishing_order`'s length doesn't match it, or if
/// `finishing_order` isn't a valid permutation of every seat exactly
/// once (e.g. a duplicate or out-of-range seat).
#[must_use]
pub fn assign_roles(finishing_order: &[SeatId], player_count: u8) -> Option<Vec<Role>> {
    let roles = roles_for_player_count(player_count)?;
    if finishing_order.len() != roles.len() {
        return None;
    }
    let seat_count = usize::from(player_count);
    let mut seen = vec![false; seat_count];
    for &seat in finishing_order {
        let seat = usize::from(seat);
        if seat >= seat_count || seen[seat] {
            return None;
        }
        seen[seat] = true;
    }
    let mut role_by_seat = vec![roles[roles.len() - 1]; seat_count];
    for (place, &seat) in finishing_order.iter().enumerate() {
        role_by_seat[usize::from(seat)] = roles[place];
    }
    Some(role_by_seat)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_player_roles_match_rules_doc() {
        assert_eq!(
            roles_for_player_count(3),
            Some(&[Role::President, Role::Dorftrottel, Role::Arschloch][..])
        );
    }

    #[test]
    fn four_player_roles_match_rules_doc() {
        assert_eq!(
            roles_for_player_count(4),
            Some(
                &[
                    Role::President,
                    Role::Vize,
                    Role::ViceArschloch,
                    Role::Arschloch
                ][..]
            )
        );
    }

    #[test]
    fn five_player_roles_match_rules_doc() {
        assert_eq!(
            roles_for_player_count(5),
            Some(
                &[
                    Role::President,
                    Role::Vize,
                    Role::Dorftrottel,
                    Role::ViceArschloch,
                    Role::Arschloch,
                ][..]
            )
        );
    }

    #[test]
    fn six_player_roles_match_rules_doc() {
        assert_eq!(
            roles_for_player_count(6),
            Some(
                &[
                    Role::President,
                    Role::Vize,
                    Role::Offizier,
                    Role::Dummkopf,
                    Role::ViceArschloch,
                    Role::Arschloch,
                ][..]
            )
        );
    }

    #[test]
    fn unsupported_player_count_returns_none() {
        assert_eq!(roles_for_player_count(2), None);
        assert_eq!(roles_for_player_count(7), None);
    }

    #[test]
    fn exchange_counts_match_rules_doc() {
        assert_eq!(exchange_counts_for_player_count(3), Some(&[1, 0][..]));
        assert_eq!(exchange_counts_for_player_count(4), Some(&[2, 1][..]));
        assert_eq!(exchange_counts_for_player_count(5), Some(&[2, 1, 0][..]));
        assert_eq!(exchange_counts_for_player_count(6), Some(&[3, 2, 1][..]));
    }

    #[test]
    fn every_supported_table_size_has_matching_role_and_exchange_lengths() {
        for player_count in [3u8, 4, 5, 6] {
            let roles = roles_for_player_count(player_count).unwrap();
            let exchanges = exchange_counts_for_player_count(player_count).unwrap();
            // Exchange table has one entry per role-pair, plus a
            // trailing 0 for a lone middle role at odd-shaped tables
            // (3 and 5 players); role count is always exchanges.len()
            // entries * 2, +/-1 for a lone middle role.
            let has_lone_middle = roles.len() % 2 == 1;
            let expected_exchange_len = roles.len() / 2 + usize::from(has_lone_middle);
            assert_eq!(exchanges.len(), expected_exchange_len);
        }
    }

    #[test]
    fn assign_roles_maps_finishing_order_to_roles_by_seat() {
        // Seat 2 finished first (President), seat 0 second (Dorftrottel),
        // seat 1 last (Arschloch).
        let role_by_seat = assign_roles(&[2, 0, 1], 3).unwrap();
        assert_eq!(role_by_seat[2], Role::President);
        assert_eq!(role_by_seat[0], Role::Dorftrottel);
        assert_eq!(role_by_seat[1], Role::Arschloch);
    }

    #[test]
    fn assign_roles_rejects_unsupported_player_count() {
        assert_eq!(assign_roles(&[0, 1], 2), None);
    }

    #[test]
    fn assign_roles_rejects_wrong_length_finishing_order() {
        assert_eq!(assign_roles(&[0, 1], 3), None);
    }

    #[test]
    fn assign_roles_rejects_duplicate_or_out_of_range_seats() {
        assert_eq!(assign_roles(&[0, 0, 1], 3), None);
        assert_eq!(assign_roles(&[0, 1, 5], 3), None);
    }

    #[test]
    fn role_ordering_follows_the_highest_to_lowest_declaration_order() {
        assert!(Role::President < Role::Arschloch);
        for player_count in [3u8, 4, 5, 6] {
            let roles = roles_for_player_count(player_count).unwrap();
            assert!(roles.windows(2).all(|pair| pair[0] < pair[1]));
        }
    }

    #[test]
    fn role_round_trips_through_json() {
        let json = serde_json::to_string(&Role::President).unwrap();
        let role: Role = serde_json::from_str(&json).unwrap();
        assert_eq!(role, Role::President);
    }
}
