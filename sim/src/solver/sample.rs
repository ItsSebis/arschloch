//! One random split of the unseen cards that fits what a seat knows: the other
//! seats' hand sizes and (optionally) their pass ceilings. Phase B's sampler
//! draws these instead of enumerating every split.

use engine::Card;
use rand::seq::SliceRandom;
use rand::RngExt;

use super::deals::{beats_single_ceiling, respects_ceilings};
use super::observation::Observation;

/// Attempts per sample before the ceilings are taken to rule out every split.
pub const MAX_ATTEMPTS: usize = 64;

/// A random consistent deal (hands by seat; our own seat holds
/// `observation.hand`, seats that are out hold nothing), or `None` when
/// `MAX_ATTEMPTS` attempts found none.
///
/// Each unseen card goes to a seat drawn with probability proportional to the
/// room left in its hand. Without ceilings that is exactly a uniform random
/// split. With them, the cards are dealt from the highest down (size-1 ceilings
/// exclude high cards, so those are the constrained ones) to the seats whose
/// size-1 ceiling allows them, and a deal that breaks a larger ceiling is drawn
/// again: close to uniform over the consistent deals, not exactly.
///
/// # Panics
///
/// Never in practice: a seat is drawn below the total room left.
pub fn sample_deal(
    observation: &Observation,
    use_ceilings: bool,
    rng: &mut impl rand::Rng,
) -> Option<Vec<Vec<Card>>> {
    let n = usize::from(observation.player_count);
    let own = usize::from(observation.seat);
    let rule = observation.duplicate_rule;
    let constrained = use_ceilings
        && observation
            .ceilings
            .iter()
            .enumerate()
            .any(|(seat, c)| seat != own && *c != crate::hand_reading::PassCeilings::default());
    let mut order = observation.unseen.clone();
    if constrained {
        order.sort_by(|a, b| b.compare(a, rule));
    }
    for _ in 0..MAX_ATTEMPTS {
        if !constrained {
            order.shuffle(rng);
        }
        let mut room: Vec<usize> = (0..n)
            .map(|s| {
                if s == own {
                    0
                } else {
                    observation.hand_sizes[s]
                }
            })
            .collect();
        let mut hands: Vec<Vec<Card>> = vec![Vec::new(); n];
        hands[own].clone_from(&observation.hand);
        let mut dead_end = false;
        for &card in &order {
            let allowed = |seat: usize| {
                room[seat] > 0 && !(constrained && beats_single_ceiling(observation, seat, card))
            };
            let total: usize = (0..n).filter(|&s| allowed(s)).map(|s| room[s]).sum();
            if total == 0 {
                dead_end = true;
                break;
            }
            let mut pick = rng.random_range(0..total);
            let seat = (0..n)
                .filter(|&s| allowed(s))
                .find(|&s| {
                    if pick < room[s] {
                        true
                    } else {
                        pick -= room[s];
                        false
                    }
                })
                .expect("pick is below the total room");
            room[seat] -= 1;
            hands[seat].push(card);
        }
        if !dead_end && (!constrained || respects_ceilings(observation, &hands)) {
            return Some(hands);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand_reading::{read_pass_ceilings, PassCeilings};
    use engine::{Combo, DuplicateRule, Rank, Suit};
    use rand::SeedableRng;

    fn card(rank: Rank, suit: Suit, index: u8) -> Card {
        Card::new(rank, suit, index)
    }

    /// 4 seats; we (seat 0) hold a 2; seats 1-3 hold 3, 2 and 1 of six unseen cards.
    fn observation() -> Observation {
        let ranks = [
            Rank::Four,
            Rank::Six,
            Rank::Eight,
            Rank::Ten,
            Rank::Queen,
            Rank::Ace,
        ];
        Observation {
            duplicate_rule: DuplicateRule::FirstDealtWins,
            seat: 0,
            player_count: 4,
            hand: vec![card(Rank::Two, Suit::Clubs, 0)],
            hand_sizes: vec![1, 3, 2, 1],
            in_trick: vec![true; 4],
            ceilings: vec![PassCeilings::default(); 4],
            unseen: ranks
                .iter()
                .enumerate()
                .map(|(i, &r)| card(r, Suit::Hearts, u8::try_from(i).unwrap() + 1))
                .collect(),
            on_table: None,
        }
    }

    fn check_shape(o: &Observation, hands: &[Vec<Card>]) {
        assert_eq!(hands[0], o.hand);
        for (hand, &size) in hands.iter().zip(&o.hand_sizes).skip(1) {
            assert_eq!(hand.len(), size);
        }
        let mut dealt: Vec<Card> = hands[1..].iter().flatten().copied().collect();
        let mut unseen = o.unseen.clone();
        dealt.sort_by_key(|c| c.deal_index);
        unseen.sort_by_key(|c| c.deal_index);
        assert_eq!(dealt, unseen, "every unseen card is dealt exactly once");
    }

    #[test]
    fn a_sample_fills_every_hand_with_each_unseen_card_once() {
        let o = observation();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        for _ in 0..200 {
            check_shape(&o, &sample_deal(&o, true, &mut rng).unwrap());
        }
    }

    #[test]
    fn without_ceilings_a_card_lands_in_proportion_to_the_room_left() {
        // The Ace goes to seats 1, 2, 3 with probability 3/6, 2/6, 1/6.
        let o = observation();
        let ace = o.unseen[5];
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let mut counts = [0u32; 4];
        let draws = 6000;
        for _ in 0..draws {
            let hands = sample_deal(&o, false, &mut rng).unwrap();
            let seat = hands.iter().position(|h| h.contains(&ace)).unwrap();
            counts[seat] += 1;
        }
        for (seat, expected) in [(1, 3000.0), (2, 2000.0), (3, 1000.0)] {
            let got = f64::from(counts[seat]);
            assert!(
                (got - expected).abs() < 150.0,
                "seat {seat}: {got} of {draws}"
            );
        }
    }

    #[test]
    fn ceilings_are_respected_or_ignored_on_request() {
        // Seat 1 passed on a single 9: no 10, Queen or Ace for it.
        let mut o = observation();
        let nine = Combo::new(vec![card(Rank::Nine, Suit::Spades, 30)]).unwrap();
        o.ceilings = read_pass_ceilings(4, &[], &[(1, nine, 0)], DuplicateRule::FirstDealtWins);
        let high: Vec<Card> = o.unseen[3..].to_vec();
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let mut ignored_once = false;
        for _ in 0..300 {
            let hands = sample_deal(&o, true, &mut rng).unwrap();
            check_shape(&o, &hands);
            assert!(hands[1].iter().all(|c| !high.contains(c)), "{hands:?}");
            let free = sample_deal(&o, false, &mut rng).unwrap();
            ignored_once |= free[1].iter().any(|c| high.contains(c));
        }
        assert!(
            ignored_once,
            "without ceilings seat 1 sometimes gets a high card"
        );
    }

    #[test]
    fn contradictory_ceilings_give_no_deal_unless_ignored() {
        // Every opponent passed on a 3, every unseen card is higher.
        let mut o = observation();
        let three = Combo::new(vec![card(Rank::Three, Suit::Spades, 30)]).unwrap();
        o.ceilings = read_pass_ceilings(
            4,
            &[],
            &[(1, three, 0), (2, three, 0), (3, three, 0)],
            DuplicateRule::FirstDealtWins,
        );
        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        assert!(sample_deal(&o, true, &mut rng).is_none());
        assert!(sample_deal(&o, false, &mut rng).is_some());
    }

    #[test]
    fn a_pair_ceiling_is_checked_on_the_whole_deal() {
        // Seat 1 passed on a pair of 5s: it holds no pair above. Two 9s unseen.
        let mut o = observation();
        o.unseen[1] = card(Rank::Nine, Suit::Hearts, 2);
        o.unseen[2] = card(Rank::Nine, Suit::Spades, 3);
        let fives = Combo::new(vec![
            card(Rank::Five, Suit::Spades, 30),
            card(Rank::Five, Suit::Diamonds, 31),
        ])
        .unwrap();
        o.ceilings = read_pass_ceilings(4, &[], &[(1, fives, 0)], DuplicateRule::FirstDealtWins);
        let nines = [o.unseen[1], o.unseen[2]];
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        for _ in 0..300 {
            let hands = sample_deal(&o, true, &mut rng).unwrap();
            assert!(!nines.iter().all(|c| hands[1].contains(c)), "{hands:?}");
        }
    }

    #[test]
    fn the_same_seed_draws_the_same_deals() {
        let o = observation();
        let draw = |seed| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            (0..20)
                .map(|_| sample_deal(&o, true, &mut rng).unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(9), draw(9));
        assert_ne!(draw(9), draw(10));
    }
}
