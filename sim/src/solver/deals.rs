//! Every way the unseen cards can be split among the other seats.

use std::cmp::Ordering;

use engine::Card;

use super::observation::Observation;

/// Calls `visit` for every consistent deal (hands by seat; our own seat
/// holds `observation.hand`, seats that are out hold nothing); `visit`
/// returns `false` to stop. Returns how many deals were visited. The order
/// is deterministic.
pub fn for_each_deal(
    observation: &Observation,
    mut visit: impl FnMut(&[Vec<Card>]) -> bool,
) -> u64 {
    let n = usize::from(observation.player_count);
    let own = usize::from(observation.seat);
    let mut hands: Vec<Vec<Card>> = vec![Vec::new(); n];
    hands[own].clone_from(&observation.hand);
    let capacity: Vec<usize> = (0..n)
        .map(|s| {
            if s == own {
                0
            } else {
                observation.hand_sizes[s]
            }
        })
        .collect();
    let mut walker = Walker {
        observation,
        capacity,
        hands,
        visited: 0,
        stopped: false,
    };
    walker.assign(0, &mut visit);
    walker.visited
}

struct Walker<'a> {
    observation: &'a Observation,
    capacity: Vec<usize>,
    hands: Vec<Vec<Card>>,
    visited: u64,
    stopped: bool,
}

impl Walker<'_> {
    /// Gives `unseen[index]` to each seat with room left, in seat order.
    /// Every unseen card is distinct (identical rank and suit differ in
    /// `deal_index`, which decides ties), so each assignment is its own deal.
    fn assign(&mut self, index: usize, visit: &mut impl FnMut(&[Vec<Card>]) -> bool) {
        if self.stopped {
            return;
        }
        if index == self.observation.unseen.len() {
            if self.consistent() {
                self.visited += 1;
                if !visit(&self.hands) {
                    self.stopped = true;
                }
            }
            return;
        }
        let card = self.observation.unseen[index];
        for seat in 0..self.hands.len() {
            if self.capacity[seat] == 0 || self.beats_single_ceiling(seat, card) {
                continue;
            }
            self.capacity[seat] -= 1;
            self.hands[seat].push(card);
            self.assign(index + 1, visit);
            self.hands[seat].pop();
            self.capacity[seat] += 1;
            if self.stopped {
                return;
            }
        }
    }

    /// A card above the seat's lowest ceiling can never be in its hand: pruning
    /// here keeps contradictory ceilings from costing a walk over every assignment.
    fn beats_single_ceiling(&self, seat: usize, card: Card) -> bool {
        let rule = self.observation.duplicate_rule;
        self.observation.ceilings[seat]
            .ceiling(1, rule)
            .is_some_and(|ceiling| card.compare(&ceiling, rule) == Ordering::Greater)
    }

    /// No opponent holds a combo that beats its pass ceiling. Ceilings are
    /// closed upwards (a larger size has the lower, stricter ceiling), so the
    /// size of the whole rank group is the one to check.
    fn consistent(&self) -> bool {
        let rule = self.observation.duplicate_rule;
        self.hands.iter().enumerate().all(|(seat, hand)| {
            if seat == usize::from(self.observation.seat) {
                return true;
            }
            let mut sorted = hand.clone();
            sorted.sort_by(|a, b| a.compare(b, rule));
            sorted.chunk_by(|a, b| a.rank == b.rank).all(|group| {
                let top = &group[group.len() - 1];
                self.observation.ceilings[seat]
                    .ceiling(group.len(), rule)
                    .is_none_or(|ceiling| top.compare(&ceiling, rule) != Ordering::Greater)
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand_reading::{read_pass_ceilings, PassCeilings};
    use crate::solver::Observation;
    use engine::{Card, Combo, DuplicateRule, Rank, Suit};

    fn card(rank: Rank, suit: Suit, index: u8) -> Card {
        Card::new(rank, suit, index)
    }

    /// 3 seats; seat 0 (us) holds `hand`; seats 1 and 2 hold `sizes`.
    fn observation(hand: Vec<Card>, sizes: [usize; 2], unseen: Vec<Card>) -> Observation {
        Observation {
            duplicate_rule: DuplicateRule::FirstDealtWins,
            seat: 0,
            player_count: 3,
            hand_sizes: vec![hand.len(), sizes[0], sizes[1]],
            hand,
            in_trick: vec![true; 3],
            ceilings: vec![PassCeilings::default(); 3],
            unseen,
            on_table: None,
        }
    }

    fn three_cards() -> Vec<Card> {
        vec![
            card(Rank::Four, Suit::Clubs, 0),
            card(Rank::Eight, Suit::Hearts, 1),
            card(Rank::King, Suit::Spades, 2),
        ]
    }

    #[test]
    fn splitting_three_distinct_cards_one_two_gives_three_deals() {
        let o = observation(vec![card(Rank::Two, Suit::Clubs, 3)], [1, 2], three_cards());
        let mut seen = Vec::new();
        let n = for_each_deal(&o, |hands| {
            assert_eq!(hands[0], o.hand);
            assert_eq!(hands[1].len(), 1);
            assert_eq!(hands[2].len(), 2);
            seen.push(hands.to_vec());
            true
        });
        assert_eq!(n, 3);
        seen.sort_by_key(|h| format!("{h:?}"));
        seen.dedup();
        assert_eq!(seen.len(), 3, "every deal is distinct");
    }

    #[test]
    fn a_seat_that_is_out_holds_nothing_and_counts_match_a_multinomial() {
        let unseen = vec![
            card(Rank::Four, Suit::Clubs, 0),
            card(Rank::Eight, Suit::Hearts, 1),
        ];
        let o = observation(vec![card(Rank::Two, Suit::Clubs, 3)], [2, 0], unseen);
        let n = for_each_deal(&o, |hands| {
            assert!(hands[2].is_empty());
            true
        });
        assert_eq!(n, 1);
        // 4 unseen cards, sizes 2/2 -> C(4,2) = 6
        let four = vec![
            card(Rank::Four, Suit::Clubs, 0),
            card(Rank::Eight, Suit::Hearts, 1),
            card(Rank::Nine, Suit::Hearts, 2),
            card(Rank::King, Suit::Spades, 4),
        ];
        let o = observation(vec![card(Rank::Two, Suit::Clubs, 3)], [2, 2], four);
        assert_eq!(for_each_deal(&o, |_| true), 6);
    }

    #[test]
    fn a_pass_ceiling_removes_deals_that_hold_something_that_beats_it() {
        // Seat 1 passed on a single 9 of hearts: it holds nothing above it, so
        // the King of spades must be seat 2's.
        let nine = Combo::new(vec![card(Rank::Nine, Suit::Hearts, 9)]).unwrap();
        let mut o = observation(vec![card(Rank::Two, Suit::Clubs, 3)], [1, 2], three_cards());
        o.ceilings = read_pass_ceilings(3, &[], &[(1, nine, 0)], DuplicateRule::FirstDealtWins);
        let mut deals = Vec::new();
        let n = for_each_deal(&o, |hands| {
            deals.push(hands.to_vec());
            true
        });
        // Seat 1 gets exactly one of {4C, 8H}: 2 deals, the King goes to seat 2.
        assert_eq!(n, 2);
        assert!(deals
            .iter()
            .all(|h| h[2].contains(&card(Rank::King, Suit::Spades, 2))));
    }

    #[test]
    fn contradictory_ceilings_leave_no_deal() {
        let three = Combo::new(vec![card(Rank::Three, Suit::Hearts, 9)]).unwrap();
        let mut o = observation(vec![card(Rank::Two, Suit::Clubs, 3)], [1, 2], three_cards());
        // Both opponents passed on a 3: neither holds anything higher, but
        // every unseen card is higher.
        o.ceilings = read_pass_ceilings(
            3,
            &[],
            &[(1, three, 0), (2, three, 0)],
            DuplicateRule::FirstDealtWins,
        );
        assert_eq!(for_each_deal(&o, |_| true), 0);
    }

    #[test]
    fn contradictory_ceilings_are_found_without_walking_every_assignment() {
        // 18 unseen cards, 6 for each of three opponents: 17 million assignments.
        // Every opponent passed on a 3 although every unseen card is higher, so
        // none is consistent.
        let ranks = [
            Rank::Four,
            Rank::Five,
            Rank::Six,
            Rank::Seven,
            Rank::Eight,
            Rank::Nine,
            Rank::Ten,
            Rank::Jack,
            Rank::Queen,
        ];
        let unseen: Vec<Card> = (0..18u8)
            .map(|i| {
                let suit = if i % 2 == 0 {
                    Suit::Hearts
                } else {
                    Suit::Spades
                };
                card(ranks[usize::from(i) / 2], suit, i + 1)
            })
            .collect();
        let three = Combo::new(vec![card(Rank::Three, Suit::Hearts, 30)]).unwrap();
        let o = Observation {
            duplicate_rule: DuplicateRule::FirstDealtWins,
            seat: 0,
            player_count: 4,
            hand: vec![card(Rank::Two, Suit::Clubs, 0)],
            hand_sizes: vec![1, 6, 6, 6],
            in_trick: vec![true; 4],
            ceilings: read_pass_ceilings(
                4,
                &[],
                &[(1, three, 0), (2, three, 0), (3, three, 0)],
                DuplicateRule::FirstDealtWins,
            ),
            unseen,
            on_table: None,
        };
        let started = std::time::Instant::now();
        assert_eq!(for_each_deal(&o, |_| true), 0);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "{:?}",
            started.elapsed()
        );
    }

    #[test]
    fn visiting_can_stop_early_and_the_order_is_deterministic() {
        let o = observation(vec![card(Rank::Two, Suit::Clubs, 3)], [1, 2], three_cards());
        let mut first = Vec::new();
        for_each_deal(&o, |h| {
            first.push(h.to_vec());
            true
        });
        let mut second = Vec::new();
        for_each_deal(&o, |h| {
            second.push(h.to_vec());
            true
        });
        assert_eq!(first, second);
        assert_eq!(for_each_deal(&o, |_| false), 1);
    }
}
