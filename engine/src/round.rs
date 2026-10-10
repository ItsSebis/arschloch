//! The single-round trick-taking state machine: hands, combo legality,
//! and finishing order. See docs/RULES.md, "Playing a Round".

use crate::card::{Card, DuplicateRule};
use crate::combo::Combo;
use crate::trick::{PassRule, Trick};
use crate::SeatId;

/// A move a seat can submit on its turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone)]
pub struct Round {
    hands: Vec<Vec<Card>>,
    duplicate_rule: DuplicateRule,
    finishing_order: Vec<SeatId>,
    current_combo: Option<Combo>,
    trick: Trick,
    pass_rule: PassRule,
    play_history: Vec<(SeatId, Combo)>,
    pass_history: Vec<(SeatId, Combo, usize)>,
}

impl Round {
    /// Starts a new round from `hands` (one per seat, already dealt and,
    /// for rounds after the first, already exchanged), with
    /// `first_leader` leading the first trick. Returns `None` if
    /// `hands.len()` isn't a supported table size (3-6), `first_leader`
    /// is out of range, or any hand starts empty (a round can't begin
    /// with a seat already out).
    #[must_use]
    pub fn new(
        hands: Vec<Vec<Card>>,
        duplicate_rule: DuplicateRule,
        first_leader: SeatId,
    ) -> Option<Self> {
        Self::with_pass_rule(hands, duplicate_rule, PassRule::default(), first_leader)
    }

    /// Like [`Round::new`] with an explicit [`PassRule`] (`new` plays by the
    /// rules of the game: a pass ends your part in the trick).
    #[must_use]
    pub fn with_pass_rule(
        hands: Vec<Vec<Card>>,
        duplicate_rule: DuplicateRule,
        pass_rule: PassRule,
        first_leader: SeatId,
    ) -> Option<Self> {
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
            trick: Trick::new(first_leader, pass_rule),
            pass_rule,
            play_history: Vec::new(),
            pass_history: Vec::new(),
        })
    }

    /// Whether `seat` has passed in the current trick and is out of it
    /// until the trick ends (`PassRule::Final` only; always `false` under
    /// `Free`).
    #[must_use]
    pub fn has_passed(&self, seat: SeatId) -> bool {
        self.trick.has_passed(seat)
    }

    #[must_use]
    pub fn pass_rule(&self) -> PassRule {
        self.pass_rule
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

    /// `seat`'s current hand size only (not contents) — the one piece of
    /// information about *other* seats this genre treats as public.
    ///
    /// # Panics
    ///
    /// Panics if `seat` is not a valid seat for this round (same
    /// unchecked-indexing convention as `hand`).
    #[must_use]
    pub fn hand_size(&self, seat: SeatId) -> usize {
        self.hands[usize::from(seat)].len()
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

    /// Every combo played so far this round, in play order, tagged with
    /// the seat that played it. Passes are omitted — a pass never removes
    /// a card from any hand, so it carries nothing a card-counting
    /// strategy needs.
    #[must_use]
    pub fn play_history(&self) -> &[(SeatId, Combo)] {
        &self.play_history
    }

    /// Every pass this round, in order, tagged with the combo it declined
    /// to beat and `play_history().len()` at that moment (so a later
    /// reduction can tell whether this seat *subsequently* played
    /// something that would have beaten it — see `sim::hand_reading`).
    #[must_use]
    pub fn pass_history(&self) -> &[(SeatId, Combo, usize)] {
        &self.pass_history
    }

    /// All moves currently legal for `self.seat_to_move()`. Empty if the
    /// round is already complete.
    #[must_use]
    pub fn legal_moves(&self) -> Vec<Move> {
        let Some(seat) = self.seat_to_move() else {
            return Vec::new();
        };
        crate::legal_moves::legal_moves(
            &self.hands[usize::from(seat)],
            self.current_combo.as_ref(),
            self.duplicate_rule,
        )
    }

    /// Like [`Round::legal_moves`], but clears `out` and fills it, so a
    /// caller that enumerates moves every turn can reuse one buffer and
    /// allocate nothing.
    pub fn legal_moves_into(&self, out: &mut Vec<Move>) {
        let Some(seat) = self.seat_to_move() else {
            out.clear();
            return;
        };
        crate::legal_moves::legal_moves_into(
            &self.hands[usize::from(seat)],
            self.current_combo.as_ref(),
            self.duplicate_rule,
            out,
        );
    }

    /// Submits `seat`'s move. On success, the round's state has already
    /// advanced (hand updated, trick/finishing-order progressed as
    /// needed). On failure, the round's state is unchanged.
    ///
    /// # Errors
    ///
    /// - [`MoveError::RoundAlreadyComplete`] if the round has already
    ///   finished.
    /// - [`MoveError::NotYourTurn`] if it isn't `seat`'s turn.
    /// - [`MoveError::CannotPassOnLead`] if `mv` is [`Move::Pass`] but no
    ///   combo is currently on the table, so `seat` must lead instead.
    /// - [`MoveError::CardNotInHand`] if `mv` is [`Move::Play`] with a
    ///   combo containing a card `seat` doesn't hold (or holds fewer
    ///   copies of than the combo lists).
    /// - [`MoveError::ComboDoesNotBeat`] if `mv` is [`Move::Play`] with a
    ///   combo that doesn't beat the current combo on the table (wrong
    ///   size, or not strictly higher).
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
        // Each card of the combo needs its own copy in hand: the i-th
        // occurrence of a card in the combo needs at least i copies held.
        let hand = &self.hands[usize::from(seat)];
        let cards = combo.cards();
        for (i, card) in cards.iter().enumerate() {
            let wanted = cards[..=i].iter().filter(|c| *c == card).count();
            if hand.iter().filter(|c| *c == card).count() < wanted {
                return Err(MoveError::CardNotInHand(*card));
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
        self.play_history.push((seat, combo));
        self.current_combo = Some(combo);

        if just_emptied {
            self.finishing_order.push(seat);
            if self.check_for_single_seat_remaining() {
                return;
            }
        }

        let (active, count) = self.active_mask();
        if let Some(new_leader) = self.trick.record_play(seat, &active[..count]) {
            self.current_combo = None;
            self.trick = Trick::new(new_leader, self.pass_rule);
        }
    }

    fn apply_pass(&mut self, seat: SeatId) {
        if let Some(combo) = &self.current_combo {
            self.pass_history
                .push((seat, *combo, self.play_history.len()));
        }
        let (active, count) = self.active_mask();
        if let Some(new_leader) = self.trick.record_pass(seat, &active[..count]) {
            self.current_combo = None;
            self.trick = Trick::new(new_leader, self.pass_rule);
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

    /// Which seats still hold cards, in a fixed array (a table has at most
    /// 6 seats) with the number of seats in use.
    fn active_mask(&self) -> ([bool; 6], usize) {
        let mut mask = [false; 6];
        for (slot, hand) in mask.iter_mut().zip(&self.hands) {
            *slot = !hand.is_empty();
        }
        (mask, self.hands.len())
    }
}

#[cfg(test)]
mod tests;
