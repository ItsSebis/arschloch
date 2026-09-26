//! The single-round trick-taking state machine: hands, combo legality,
//! and finishing order. See docs/RULES.md, "Playing a Round".

use crate::card::{Card, DuplicateRule};
use crate::combo::Combo;
use crate::trick::Trick;
use crate::SeatId;

/// A move a seat can submit on its turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Move {
    Play(Combo),
    Pass,
}

/// Why a submitted move was rejected. The round's state is unchanged
/// when this is returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoveError {
    /// The round has already finished; no more moves can be submitted.
    RoundAlreadyComplete,
    /// It isn't `seat`'s turn.
    NotYourTurn { expected: SeatId },
    /// No combo is currently on the table, so the acting seat must play,
    /// not pass.
    CannotPassOnLead,
    /// The submitted combo contains a card the seat doesn't hold (or
    /// lists a card it holds only once, twice).
    CardNotInHand(Card),
    /// The submitted combo doesn't beat the current combo on the table
    /// (wrong size, or not strictly higher).
    ComboDoesNotBeat,
}

/// A single round's trick-taking state, from a freshly dealt (and, if
/// applicable, already-exchanged) set of hands through to every seat's
/// finishing order.
#[derive(Debug)]
pub struct Round {
    hands: Vec<Vec<Card>>,
    duplicate_rule: DuplicateRule,
    finishing_order: Vec<SeatId>,
    current_combo: Option<Combo>,
    trick: Trick,
}

impl Round {
    /// Starts a new round from `hands` (one per seat, already dealt and,
    /// for rounds after the first, already exchanged), with
    /// `first_leader` leading the first trick. Returns `None` if
    /// `hands.len()` isn't a supported table size (3-6), `first_leader`
    /// is out of range, or any hand starts empty (a round can't begin
    /// with a seat already out).
    #[must_use]
    pub fn new(hands: Vec<Vec<Card>>, duplicate_rule: DuplicateRule, first_leader: SeatId) -> Option<Self> {
        if !(3..=6).contains(&hands.len()) {
            return None;
        }
        if usize::from(first_leader) >= hands.len() {
            return None;
        }
        if hands.iter().any(Vec::is_empty) {
            return None;
        }
        Some(Self {
            hands,
            duplicate_rule,
            finishing_order: Vec::new(),
            current_combo: None,
            trick: Trick::new(first_leader),
        })
    }

    /// The seat that must act next, or `None` if the round is complete.
    #[must_use]
    pub fn seat_to_move(&self) -> Option<SeatId> {
        if self.is_complete() {
            None
        } else {
            Some(self.trick.turn())
        }
    }

    /// `seat`'s current hand.
    ///
    /// # Panics
    ///
    /// Panics if `seat` is not a valid seat for this round
    /// (`usize::from(seat) >= player_count`).
    #[must_use]
    pub fn hand(&self, seat: SeatId) -> &[Card] {
        &self.hands[usize::from(seat)]
    }

    /// Whether `seat` still holds cards (hasn't finished the round yet).
    ///
    /// # Panics
    ///
    /// Panics if `seat` is not a valid seat for this round.
    #[must_use]
    pub fn is_active(&self, seat: SeatId) -> bool {
        !self.hands[usize::from(seat)].is_empty()
    }

    /// Whether every seat but one has emptied its hand.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.finishing_order.len() == self.hands.len()
    }

    /// Seats in the order they emptied their hand (first out first),
    /// including the auto-appended last seat once the round completes.
    #[must_use]
    pub fn finishing_order(&self) -> &[SeatId] {
        &self.finishing_order
    }

    /// The combo currently on the table for the active trick, or `None`
    /// if the active seat must lead a fresh trick.
    #[must_use]
    pub fn current_combo(&self) -> Option<&Combo> {
        self.current_combo.as_ref()
    }

    /// The seat that led the trick currently in progress.
    #[must_use]
    pub fn current_trick_leader(&self) -> SeatId {
        self.trick.leader()
    }

    /// Submits `seat`'s move. On success, the round's state has already
    /// advanced (hand updated, trick/finishing-order progressed as
    /// needed). On failure, the round's state is unchanged.
    pub fn submit_move(&mut self, seat: SeatId, mv: Move) -> Result<(), MoveError> {
        if self.is_complete() {
            return Err(MoveError::RoundAlreadyComplete);
        }
        let expected = self.trick.turn();
        if seat != expected {
            return Err(MoveError::NotYourTurn { expected });
        }

        match mv {
            Move::Pass => {
                if !self.trick.has_current_play() {
                    return Err(MoveError::CannotPassOnLead);
                }
                self.apply_pass(seat);
                Ok(())
            }
            Move::Play(combo) => {
                self.validate_play(seat, &combo)?;
                self.apply_play(seat, combo);
                Ok(())
            }
        }
    }

    fn validate_play(&self, seat: SeatId, combo: &Combo) -> Result<(), MoveError> {
        let mut remaining_hand = self.hands[usize::from(seat)].clone();
        for card in combo.cards() {
            match remaining_hand.iter().position(|c| c == card) {
                Some(index) => {
                    remaining_hand.remove(index);
                }
                None => return Err(MoveError::CardNotInHand(*card)),
            }
        }
        if let Some(current) = &self.current_combo {
            if !combo.beats(current, self.duplicate_rule) {
                return Err(MoveError::ComboDoesNotBeat);
            }
        }
        Ok(())
    }

    fn apply_play(&mut self, seat: SeatId, combo: Combo) {
        let hand = &mut self.hands[usize::from(seat)];
        for card in combo.cards() {
            let position = hand
                .iter()
                .position(|c| c == card)
                .expect("validate_play already confirmed every card in the combo has a matching card in hand");
            hand.remove(position);
        }
        let just_emptied = hand.is_empty();
        self.current_combo = Some(combo);

        if just_emptied {
            self.finishing_order.push(seat);
            if self.check_for_single_seat_remaining() {
                return;
            }
        }

        let active = self.active_mask();
        self.trick.record_play(seat, &active);
    }

    fn apply_pass(&mut self, seat: SeatId) {
        let active = self.active_mask();
        if let Some(new_leader) = self.trick.record_pass(seat, &active) {
            self.current_combo = None;
            self.trick = Trick::new(new_leader);
        }
    }

    /// If exactly one seat still holds cards, appends it to the
    /// finishing order and completes the round. Returns whether this
    /// happened, so `apply_play` can skip further trick bookkeeping.
    fn check_for_single_seat_remaining(&mut self) -> bool {
        let remaining_active_count = self.hands.iter().filter(|h| !h.is_empty()).count();
        if remaining_active_count == 1 {
            let last_seat = SeatId::try_from(
                self.hands
                    .iter()
                    .position(|h| !h.is_empty())
                    .expect("remaining_active_count == 1"),
            )
            .expect("table sizes are capped at 6 seats");
            self.finishing_order.push(last_seat);
            true
        } else {
            false
        }
    }

    fn active_mask(&self) -> Vec<bool> {
        self.hands.iter().map(|h| !h.is_empty()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{Rank, Suit};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit, 0)
    }

    fn combo(cards: Vec<Card>) -> Combo {
        Combo::new(cards).unwrap()
    }

    #[test]
    fn new_round_rejects_unsupported_player_counts() {
        assert!(Round::new(
            vec![vec![card(Rank::Two, Suit::Clubs)]; 2],
            DuplicateRule::FirstDealtWins,
            0
        )
        .is_none());
        assert!(Round::new(
            vec![vec![card(Rank::Two, Suit::Clubs)]; 7],
            DuplicateRule::FirstDealtWins,
            0
        )
        .is_none());
    }

    #[test]
    fn new_round_rejects_out_of_range_leader() {
        let hands = vec![vec![card(Rank::Two, Suit::Clubs)]; 3];
        assert!(Round::new(hands, DuplicateRule::FirstDealtWins, 3).is_none());
    }

    #[test]
    fn new_round_rejects_an_empty_starting_hand() {
        let hands = vec![
            vec![card(Rank::Two, Suit::Clubs)],
            vec![],
            vec![card(Rank::Three, Suit::Clubs)],
        ];
        assert!(Round::new(hands, DuplicateRule::FirstDealtWins, 0).is_none());
    }

    #[test]
    fn leader_must_play_and_cannot_pass() {
        let hands = vec![
            vec![card(Rank::Two, Suit::Clubs)],
            vec![card(Rank::Three, Suit::Clubs)],
            vec![card(Rank::Four, Suit::Clubs)],
        ];
        let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
        assert_eq!(round.submit_move(0, Move::Pass), Err(MoveError::CannotPassOnLead));
    }

    #[test]
    fn rejects_a_move_from_the_wrong_seat() {
        let hands = vec![
            vec![card(Rank::Two, Suit::Clubs)],
            vec![card(Rank::Three, Suit::Clubs)],
            vec![card(Rank::Four, Suit::Clubs)],
        ];
        let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
        let play = Move::Play(combo(vec![card(Rank::Three, Suit::Clubs)]));
        assert_eq!(
            round.submit_move(1, play),
            Err(MoveError::NotYourTurn { expected: 0 })
        );
    }

    #[test]
    fn rejects_a_combo_with_a_card_not_in_hand() {
        let hands = vec![
            vec![card(Rank::Two, Suit::Clubs)],
            vec![card(Rank::Three, Suit::Clubs)],
            vec![card(Rank::Four, Suit::Clubs)],
        ];
        let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
        let play = Move::Play(combo(vec![card(Rank::Nine, Suit::Clubs)]));
        assert_eq!(
            round.submit_move(0, play),
            Err(MoveError::CardNotInHand(card(Rank::Nine, Suit::Clubs)))
        );
    }

    #[test]
    fn rejects_a_combo_that_does_not_beat_the_current_combo() {
        let hands = vec![
            vec![card(Rank::Eight, Suit::Clubs)],
            vec![card(Rank::Seven, Suit::Clubs)],
            vec![card(Rank::Nine, Suit::Clubs)],
        ];
        let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
        round
            .submit_move(0, Move::Play(combo(vec![card(Rank::Eight, Suit::Clubs)])))
            .unwrap();
        let play = Move::Play(combo(vec![card(Rank::Seven, Suit::Clubs)]));
        assert_eq!(round.submit_move(1, play), Err(MoveError::ComboDoesNotBeat));
    }

    #[test]
    fn everyone_passing_returns_the_lead_to_the_original_leader() {
        let hands = vec![
            vec![card(Rank::Eight, Suit::Clubs), card(Rank::Ten, Suit::Clubs)],
            vec![card(Rank::Seven, Suit::Clubs), card(Rank::Six, Suit::Clubs)],
            vec![card(Rank::Nine, Suit::Clubs), card(Rank::Five, Suit::Clubs)],
        ];
        let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
        round
            .submit_move(0, Move::Play(combo(vec![card(Rank::Eight, Suit::Clubs)])))
            .unwrap();
        round.submit_move(1, Move::Pass).unwrap();
        round.submit_move(2, Move::Pass).unwrap();
        assert_eq!(round.seat_to_move(), Some(0));
        assert_eq!(round.current_combo(), None);
        assert_eq!(round.current_trick_leader(), 0);
    }

    #[test]
    fn a_seat_emptying_its_hand_mid_round_is_recorded_and_skipped() {
        // 3 players, each starts with 1 card so seat 0 empties
        // immediately on its lead.
        let hands = vec![
            vec![card(Rank::Two, Suit::Clubs)],
            vec![card(Rank::Three, Suit::Clubs)],
            vec![card(Rank::Four, Suit::Clubs)],
        ];
        let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
        round
            .submit_move(0, Move::Play(combo(vec![card(Rank::Two, Suit::Clubs)])))
            .unwrap();
        assert_eq!(round.finishing_order(), &[0]);
        assert!(!round.is_active(0));
        assert_eq!(round.seat_to_move(), Some(1));

        round
            .submit_move(1, Move::Play(combo(vec![card(Rank::Three, Suit::Clubs)])))
            .unwrap();
        // Only seat 2 remains active: the round auto-completes.
        assert!(round.is_complete());
        assert_eq!(round.finishing_order(), &[0, 1, 2]);
        assert_eq!(round.seat_to_move(), None);
    }
}
