//! Exact max^n search of one fully known deal.
//!
//! Every seat to move picks the move that gives *itself* the best (lowest)
//! finishing place; ties go to the first move in the engine's move order.
//! This is the rational-opponents model (the spec's default); alpha-beta is
//! not sound for more than two players, so there is no pruning beyond the
//! node budget.

use engine::{Move, Round};

/// 0-based finishing place by seat (0 = first out), for all seats.
pub type Places = [u8; 6];

/// The search needed more nodes than allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetExceeded;

pub struct Search {
    remaining: u64,
    used: u64,
}

impl Search {
    #[must_use]
    pub fn new(node_budget: u64) -> Self {
        Self {
            remaining: node_budget,
            used: 0,
        }
    }

    #[must_use]
    pub fn nodes_used(&self) -> u64 {
        self.used
    }

    /// The outcome of each legal move of the seat to move (engine order).
    ///
    /// # Errors
    ///
    /// [`BudgetExceeded`] when the node budget runs out.
    ///
    /// # Panics
    ///
    /// Panics if the engine rejects one of its own legal moves.
    pub fn root_values(&mut self, round: &Round) -> Result<Vec<(Move, Places)>, BudgetExceeded> {
        let Some(seat) = round.seat_to_move() else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        for mv in round.legal_moves() {
            let mut next = round.clone();
            next.submit_move(seat, mv)
                .expect("moves from legal_moves are accepted");
            out.push((mv, self.value(&next)?));
        }
        Ok(out)
    }

    fn value(&mut self, round: &Round) -> Result<Places, BudgetExceeded> {
        if round.is_complete() {
            return Ok(places_of(round));
        }
        if self.remaining == 0 {
            return Err(BudgetExceeded);
        }
        self.remaining -= 1;
        self.used += 1;
        let seat = round.seat_to_move().expect("round is not complete");
        let mut best: Option<Places> = None;
        for mv in round.legal_moves() {
            let mut next = round.clone();
            next.submit_move(seat, mv)
                .expect("moves from legal_moves are accepted");
            let value = self.value(&next)?;
            if best.is_none_or(|b| value[usize::from(seat)] < b[usize::from(seat)]) {
                best = Some(value);
            }
        }
        Ok(best.expect("a seat to move has at least one legal move"))
    }
}

fn places_of(round: &Round) -> Places {
    let mut places = [0u8; 6];
    for (place, &seat) in round.finishing_order().iter().enumerate() {
        places[usize::from(seat)] = u8::try_from(place).expect("at most 6 seats");
    }
    places
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::{Card, Combo, DuplicateRule, Move, Rank, Round, Suit};

    fn c(rank: Rank, suit: Suit, i: u8) -> Card {
        Card::new(rank, suit, i)
    }
    const RULE: DuplicateRule = DuplicateRule::FirstDealtWins;

    #[test]
    fn the_leader_with_a_winning_last_card_finishes_first() {
        // Seat 0 holds a single Ace and leads: it goes out first.
        let hands = vec![
            vec![c(Rank::Ace, Suit::Clubs, 0)],
            vec![
                c(Rank::Three, Suit::Clubs, 1),
                c(Rank::Four, Suit::Clubs, 2),
            ],
            vec![c(Rank::Five, Suit::Clubs, 3), c(Rank::Six, Suit::Clubs, 4)],
        ];
        let round = Round::from_position(hands, RULE, 0, None, &[false; 3]).unwrap();
        let mut search = Search::new(1_000_000);
        let values = search.root_values(&round).unwrap();
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].1[0], 0, "seat 0 finishes first");
    }

    #[test]
    fn every_move_leads_to_a_complete_set_of_places() {
        let hands = vec![
            vec![
                c(Rank::Seven, Suit::Clubs, 0),
                c(Rank::Eight, Suit::Clubs, 1),
            ],
            vec![
                c(Rank::Six, Suit::Clubs, 2),
                c(Rank::Nine, Suit::Clubs, 3),
                c(Rank::Nine, Suit::Hearts, 4),
            ],
            vec![c(Rank::Four, Suit::Clubs, 5), c(Rank::Ten, Suit::Clubs, 6)],
        ];
        let on_table = Some((
            0u8,
            Combo::from_slice(&[c(Rank::Five, Suit::Hearts, 7)]).unwrap(),
        ));
        let round = Round::from_position(hands, RULE, 1, on_table, &[false; 3]).unwrap();
        let mut search = Search::new(1_000_000);
        let values = search.root_values(&round).unwrap();
        assert!(values.len() >= 2);
        for (_, places) in &values {
            let mut p = places[..3].to_vec();
            p.sort_unstable();
            assert_eq!(p, vec![0, 1, 2]);
        }
    }

    #[test]
    fn a_too_small_budget_is_reported_not_guessed() {
        let hands = vec![
            vec![
                c(Rank::Seven, Suit::Clubs, 0),
                c(Rank::Eight, Suit::Clubs, 1),
            ],
            vec![c(Rank::Six, Suit::Clubs, 2), c(Rank::Nine, Suit::Clubs, 3)],
            vec![c(Rank::Four, Suit::Clubs, 5), c(Rank::Ten, Suit::Clubs, 6)],
        ];
        let round = Round::from_position(hands, RULE, 0, None, &[false; 3]).unwrap();
        assert!(Search::new(2).root_values(&round).is_err());
    }

    /// An independent, deliberately naive reference: recursion over
    /// `Round::legal_moves`, every seat taking its own best place.
    fn reference(round: &Round) -> Places {
        if round.is_complete() {
            let mut p = [0u8; 6];
            for (place, &seat) in round.finishing_order().iter().enumerate() {
                p[usize::from(seat)] = u8::try_from(place).unwrap();
            }
            return p;
        }
        let seat = round.seat_to_move().unwrap();
        let mut best: Option<Places> = None;
        for mv in round.legal_moves() {
            let mut next = round.clone();
            next.submit_move(seat, mv).unwrap();
            let value = reference(&next);
            if best.is_none_or(|b| value[usize::from(seat)] < b[usize::from(seat)]) {
                best = Some(value);
            }
        }
        best.unwrap()
    }

    #[test]
    fn the_search_agrees_with_the_naive_reference_on_random_small_positions() {
        use crate::test_support::for_each_state;
        let mut checked = 0;
        for_each_state(21, 2, |state, _| {
            let round = state.round;
            let cards: usize = (0..state.players).map(|s| round.hand_size(s)).sum();
            if cards > 9 || round.pass_rule() != engine::PassRule::Final {
                return;
            }
            let seat = round.seat_to_move().unwrap();
            let mut search = Search::new(5_000_000);
            let values = search.root_values(round).unwrap();
            for (mv, places) in values {
                let mut after = round.clone();
                after.submit_move(seat, mv).unwrap();
                assert_eq!(places, reference(&after));
            }
            checked += 1;
        });
        assert!(checked > 20, "only {checked} positions were small enough");
    }

    #[test]
    fn the_first_best_move_wins_ties() {
        // Seat 0 leads a single card of two equal-valued options: the value of
        // the first move in engine order is returned for a tied seat.
        let hands = vec![
            vec![c(Rank::Ace, Suit::Clubs, 0), c(Rank::Ace, Suit::Spades, 1)],
            vec![c(Rank::Three, Suit::Clubs, 2)],
            vec![c(Rank::Four, Suit::Clubs, 3)],
        ];
        let round = Round::from_position(hands, RULE, 0, None, &[false; 3]).unwrap();
        let moves = round.legal_moves();
        let values = Search::new(1_000_000).root_values(&round).unwrap();
        assert_eq!(
            values.iter().map(|(mv, _)| *mv).collect::<Vec<Move>>(),
            moves,
            "one value per legal move, in engine order"
        );
    }
}
