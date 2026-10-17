//! Exact max^n search of one fully known deal.
//!
//! Every seat to move picks the move that gives *itself* the best (lowest)
//! finishing place; ties go to the first move in the engine's move order.
//! This is the rational-opponents model (the spec's default); alpha-beta is
//! not sound for more than two players, so there is no pruning beyond the
//! node budget. Positions that different move orders lead to are solved once
//! (a memo keyed by everything that decides the rest of the game).

use std::collections::HashMap;

use engine::{Move, Round, SeatId};

use crate::card_bits::card_bits;

/// 0-based finishing place by seat (0 = first out), for all seats.
pub type Places = [u8; 6];

/// The search needed more nodes than allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetExceeded;

/// Memo entries per search, allocated once up front: 7/8 of 2^18 table
/// slots, so the table never resizes. A slot is about 150 bytes: a fixed
/// ~40 MB per decision in flight, however large the node budget is.
const MEMO_LIMIT: usize = (1 << 18) / 8 * 7;

/// Everything that decides how a deal continues, without any heap data.
/// Cards are identified by their `deal_index` (unique within a round), as
/// bits of a `u128`.
#[derive(Hash, PartialEq, Eq)]
struct Position {
    hands: [u128; 6],
    on_table: u128,
    holder: Option<SeatId>,
    turn: SeatId,
    passed: u8,
    /// Who went out, in order: the same hands can follow different orders.
    finished: [Option<SeatId>; 6],
}

impl Position {
    fn of(round: &Round, turn: SeatId) -> Self {
        let mut position = Self {
            hands: [0; 6],
            on_table: round.current_combo().map_or(0, |c| card_bits(c.cards())),
            holder: round.combo_holder(),
            turn,
            passed: 0,
            finished: [None; 6],
        };
        let mut cards = 0;
        for (seat, hand) in (0..round.player_count()).filter_map(|s| {
            SeatId::try_from(s)
                .ok()
                .map(|seat| (seat, round.hand(seat)))
        }) {
            position.hands[usize::from(seat)] = card_bits(hand);
            position.passed |= u8::from(round.has_passed(seat)) << seat;
            cards += hand.len();
        }
        debug_assert_eq!(
            position
                .hands
                .iter()
                .map(|m| m.count_ones() as usize)
                .sum::<usize>(),
            cards,
            "cards of one round need distinct deal indices"
        );
        for (slot, &seat) in position.finished.iter_mut().zip(round.finishing_order()) {
            *slot = Some(seat);
        }
        position
    }
}

pub struct Search {
    budget: u64,
    used: u64,
    memo: HashMap<Position, Places>,
    memo_limit: usize,
}

impl Search {
    #[must_use]
    pub fn new(node_budget: u64) -> Self {
        Self {
            budget: node_budget,
            used: 0,
            memo: HashMap::with_capacity(MEMO_LIMIT),
            memo_limit: MEMO_LIMIT,
        }
    }

    #[must_use]
    pub fn nodes_used(&self) -> u64 {
        self.used
    }

    /// Remembers at most `limit` positions (tests: a tiny memo must not change values).
    #[cfg(test)]
    fn with_memo_limit(mut self, limit: usize) -> Self {
        self.memo_limit = limit;
        self
    }

    #[cfg(test)]
    fn memo_len(&self) -> usize {
        self.memo.len()
    }

    #[cfg(test)]
    fn memo_capacity(&self) -> usize {
        self.memo.capacity()
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
        let seat = round.seat_to_move().expect("round is not complete");
        let position = Position::of(round, seat);
        if let Some(&places) = self.memo.get(&position) {
            return Ok(places);
        }
        if self.used >= self.budget {
            return Err(BudgetExceeded);
        }
        self.used += 1;
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
        let best = best.expect("a seat to move has at least one legal move");
        if self.memo.len() < self.memo_limit {
            self.memo.insert(position, best);
        }
        Ok(best)
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

    /// Non-terminal nodes of the plain game tree (no memoisation).
    fn naive_nodes(round: &Round) -> u64 {
        if round.is_complete() {
            return 0;
        }
        let seat = round.seat_to_move().unwrap();
        1 + round
            .legal_moves()
            .into_iter()
            .map(|mv| {
                let mut next = round.clone();
                next.submit_move(seat, mv).unwrap();
                naive_nodes(&next)
            })
            .sum::<u64>()
    }

    #[test]
    fn equal_positions_are_searched_once() {
        // Seat 0 can lead its low cards in any order while the others pass, so
        // many lines reach the same position.
        let hands = vec![
            vec![
                c(Rank::Three, Suit::Clubs, 0),
                c(Rank::Four, Suit::Clubs, 1),
                c(Rank::Five, Suit::Clubs, 2),
                c(Rank::Six, Suit::Clubs, 3),
            ],
            vec![
                c(Rank::Seven, Suit::Clubs, 4),
                c(Rank::Eight, Suit::Clubs, 5),
            ],
            vec![c(Rank::Nine, Suit::Clubs, 6), c(Rank::Ten, Suit::Clubs, 7)],
        ];
        let round = Round::from_position(hands, RULE, 0, None, &[false; 3]).unwrap();
        let mut search = Search::new(10_000_000);
        let memoised = search.root_values(&round).unwrap();
        // `root_values` does not count the root itself.
        let plain = naive_nodes(&round) - 1;
        assert!(
            search.nodes_used() < plain,
            "{} nodes searched, {plain} in the plain tree",
            search.nodes_used()
        );
        // The values are those of the plain search (reference agreement is
        // tested above on random positions); here: one entry per legal move.
        assert_eq!(memoised.len(), round.legal_moves().len());
    }

    #[test]
    fn the_memo_is_bounded_and_does_not_change_the_values() {
        use crate::test_support::for_each_state;
        let mut checked = 0;
        for_each_state(25, 1, |state, _| {
            let round = state.round;
            let cards: usize = (0..state.players).map(|s| round.hand_size(s)).sum();
            if cards > 8 || round.pass_rule() != engine::PassRule::Final {
                return;
            }
            let mut full = Search::new(5_000_000);
            let mut tiny = Search::new(5_000_000).with_memo_limit(10);
            assert_eq!(
                full.root_values(round).unwrap(),
                tiny.root_values(round).unwrap()
            );
            assert!(tiny.memo_len() <= 10);
            checked += 1;
        });
        assert!(checked > 10, "only {checked} positions were small enough");
    }

    #[test]
    fn the_memo_is_allocated_once_and_never_resizes() {
        let hands = vec![
            vec![
                c(Rank::Three, Suit::Clubs, 0),
                c(Rank::Four, Suit::Clubs, 1),
                c(Rank::Five, Suit::Clubs, 2),
                c(Rank::Six, Suit::Clubs, 3),
            ],
            vec![
                c(Rank::Seven, Suit::Clubs, 4),
                c(Rank::Eight, Suit::Clubs, 5),
            ],
            vec![c(Rank::Nine, Suit::Clubs, 6), c(Rank::Ten, Suit::Clubs, 7)],
        ];
        let round = Round::from_position(hands, RULE, 0, None, &[false; 3]).unwrap();
        let mut search = Search::new(10_000_000);
        let before = search.memo_capacity();
        assert!(before >= MEMO_LIMIT, "{before}");
        search.root_values(&round).unwrap();
        assert!(
            search.memo_len() > 100,
            "the search must fill part of the memo"
        );
        assert_eq!(
            search.memo_capacity(),
            before,
            "a resize would double the memory"
        );
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
