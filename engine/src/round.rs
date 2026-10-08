//! The single-round trick-taking state machine: hands, combo legality,
//! and finishing order. See docs/RULES.md, "Playing a Round".

use crate::card::{Card, DuplicateRule};
use crate::combo::Combo;
use crate::trick::{PassRule, Trick};
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
    /// strategy needs (docs/ROADMAP.md, Phase 6).
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
        self.play_history.push((seat, combo.clone()));
        self.current_combo = Some(combo);

        if just_emptied {
            self.finishing_order.push(seat);
            if self.check_for_single_seat_remaining() {
                return;
            }
        }

        let active = self.active_mask();
        if let Some(new_leader) = self.trick.record_play(seat, &active) {
            self.current_combo = None;
            self.trick = Trick::new(new_leader, self.pass_rule);
        }
    }

    fn apply_pass(&mut self, seat: SeatId) {
        if let Some(combo) = &self.current_combo {
            self.pass_history
                .push((seat, combo.clone(), self.play_history.len()));
        }
        let active = self.active_mask();
        if let Some(new_leader) = self.trick.record_pass(seat, &active) {
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
        assert_eq!(
            round.submit_move(0, Move::Pass),
            Err(MoveError::CannotPassOnLead)
        );
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

    #[test]
    fn rejects_a_combo_that_lists_the_same_physical_card_twice() {
        let original_hand = vec![
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Seven, Suit::Diamonds),
        ];
        let hands = vec![
            original_hand.clone(),
            vec![card(Rank::Eight, Suit::Clubs)],
            vec![card(Rank::Nine, Suit::Clubs)],
        ];
        let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
        let play = Move::Play(combo(vec![
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Seven, Suit::Clubs),
        ]));
        assert_eq!(
            round.submit_move(0, play),
            Err(MoveError::CardNotInHand(card(Rank::Seven, Suit::Clubs)))
        );
        assert_eq!(round.hand(0), original_hand.as_slice());
    }

    #[test]
    fn legal_moves_is_empty_once_the_round_is_complete() {
        let hands = vec![
            vec![card(Rank::Two, Suit::Clubs)],
            vec![card(Rank::Three, Suit::Clubs)],
            vec![card(Rank::Four, Suit::Clubs)],
        ];
        let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
        round
            .submit_move(0, Move::Play(combo(vec![card(Rank::Two, Suit::Clubs)])))
            .unwrap();
        round
            .submit_move(1, Move::Play(combo(vec![card(Rank::Three, Suit::Clubs)])))
            .unwrap();
        assert!(round.is_complete());
        assert_eq!(round.legal_moves(), Vec::new());
    }

    #[test]
    fn legal_moves_matches_the_leaders_hand_when_no_combo_is_on_the_table() {
        let hands = vec![
            vec![
                card(Rank::Two, Suit::Clubs),
                card(Rank::Two, Suit::Diamonds),
            ],
            vec![card(Rank::Three, Suit::Clubs)],
            vec![card(Rank::Four, Suit::Clubs)],
        ];
        let round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
        let moves = round.legal_moves();
        assert_ne!(moves, []);
        assert!(!moves.contains(&Move::Pass));
    }

    #[test]
    fn accepts_a_legal_two_card_combo_of_distinct_cards() {
        let hands = vec![
            vec![
                card(Rank::Seven, Suit::Clubs),
                card(Rank::Seven, Suit::Diamonds),
            ],
            vec![card(Rank::Eight, Suit::Clubs)],
            vec![card(Rank::Nine, Suit::Clubs)],
        ];
        let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
        let play = Move::Play(combo(vec![
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Seven, Suit::Diamonds),
        ]));
        assert_eq!(round.submit_move(0, play), Ok(()));
        assert_eq!(round.hand(0), []);
    }

    #[test]
    fn play_history_records_plays_in_order_and_omits_passes() {
        let hands = vec![
            vec![card(Rank::Eight, Suit::Clubs), card(Rank::Ten, Suit::Clubs)],
            vec![card(Rank::Seven, Suit::Clubs), card(Rank::Six, Suit::Clubs)],
            vec![card(Rank::Nine, Suit::Clubs), card(Rank::Five, Suit::Clubs)],
        ];
        let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
        assert_eq!(round.play_history(), &[]);

        let first_play = combo(vec![card(Rank::Eight, Suit::Clubs)]);
        round
            .submit_move(0, Move::Play(first_play.clone()))
            .unwrap();
        assert_eq!(round.play_history(), &[(0, first_play.clone())]);

        round.submit_move(1, Move::Pass).unwrap();
        assert_eq!(
            round.play_history(),
            &[(0, first_play.clone())],
            "a pass must not appear in play_history"
        );

        let second_play = combo(vec![card(Rank::Nine, Suit::Clubs)]);
        round
            .submit_move(2, Move::Play(second_play.clone()))
            .unwrap();
        assert_eq!(round.play_history(), &[(0, first_play), (2, second_play)]);
    }

    #[test]
    fn hand_size_matches_hand_len_and_decreases_after_a_play() {
        let hands = vec![
            vec![card(Rank::Eight, Suit::Clubs), card(Rank::Ten, Suit::Clubs)],
            vec![card(Rank::Seven, Suit::Clubs)],
            vec![card(Rank::Nine, Suit::Clubs)],
        ];
        let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
        assert_eq!(round.hand_size(0), 2);
        assert_eq!(round.hand_size(0), round.hand(0).len());

        round
            .submit_move(0, Move::Play(combo(vec![card(Rank::Eight, Suit::Clubs)])))
            .unwrap();
        assert_eq!(round.hand_size(0), 1);
    }

    #[test]
    fn pass_history_records_the_combo_and_play_count_at_pass_time() {
        let hands = vec![
            vec![card(Rank::Eight, Suit::Clubs), card(Rank::Ten, Suit::Clubs)],
            vec![card(Rank::Seven, Suit::Clubs), card(Rank::Six, Suit::Clubs)],
            vec![card(Rank::Nine, Suit::Clubs), card(Rank::Five, Suit::Clubs)],
        ];
        let mut round =
            Round::with_pass_rule(hands, DuplicateRule::FirstDealtWins, PassRule::Free, 0).unwrap();
        assert_eq!(round.pass_history(), &[]);

        let lead = combo(vec![card(Rank::Eight, Suit::Clubs)]);
        round.submit_move(0, Move::Play(lead.clone())).unwrap(); // play_history now len 1
        round.submit_move(1, Move::Pass).unwrap();
        assert_eq!(round.pass_history(), &[(1, lead.clone(), 1)]);

        let beat = combo(vec![card(Rank::Nine, Suit::Clubs)]);
        round.submit_move(2, Move::Play(beat)).unwrap(); // play_history now len 2
                                                         // Trick resolves (both non-leaders acted); seat 0 leads again with
                                                         // its remaining card.
        let second_lead = combo(vec![card(Rank::Ten, Suit::Clubs)]);
        round
            .submit_move(0, Move::Play(second_lead.clone()))
            .unwrap(); // len 3
        round.submit_move(1, Move::Pass).unwrap();
        assert_eq!(
            round.pass_history(),
            &[(1, lead, 1), (1, second_lead, 3)],
            "each pass is tagged with play_history().len() at that exact moment"
        );
    }

    // ---- the pass rule -------------------------------------------------

    fn single_hands(ranks: &[&[Rank]]) -> Vec<Vec<Card>> {
        ranks
            .iter()
            .enumerate()
            .map(|(seat, hand)| {
                hand.iter()
                    .enumerate()
                    .map(|(i, &rank)| {
                        Card::new(rank, Suit::Clubs, u8::try_from(seat * 10 + i).unwrap())
                    })
                    .collect()
            })
            .collect()
    }

    fn play(round: &mut Round, seat: SeatId, rank: Rank) {
        let card = round
            .hand(seat)
            .iter()
            .copied()
            .find(|c| c.rank == rank)
            .expect("the seat holds that rank");
        round
            .submit_move(seat, Move::Play(combo(vec![card])))
            .unwrap();
    }

    #[test]
    fn under_the_final_rule_a_seat_that_passed_is_skipped_for_the_rest_of_the_trick() {
        use Rank::*;
        // 4 seats: 0 leads 4, 1 passes (out), 2 plays 6, 3 passes (out),
        // 0 plays 9 -> turn must go to 2 (1 and 3 are out), not to 1.
        let hands = single_hands(&[
            &[Four, Nine, Ace],
            &[Five, Seven],
            &[Six, Eight],
            &[Seven, Ten],
        ]);
        let mut round =
            Round::with_pass_rule(hands, DuplicateRule::FirstDealtWins, PassRule::Final, 0)
                .unwrap();
        play(&mut round, 0, Four);
        round.submit_move(1, Move::Pass).unwrap();
        play(&mut round, 2, Six);
        round.submit_move(3, Move::Pass).unwrap();
        assert_eq!(round.seat_to_move(), Some(0));
        play(&mut round, 0, Nine);
        assert_eq!(
            round.seat_to_move(),
            Some(2),
            "seats 1 and 3 passed and are out"
        );
        assert_eq!(
            round.submit_move(1, Move::Pass),
            Err(MoveError::NotYourTurn { expected: 2 })
        );
        // Seat 2 passes: nobody is left but seat 0, who wins and leads.
        round.submit_move(2, Move::Pass).unwrap();
        assert_eq!(round.current_combo(), None);
        assert_eq!(round.seat_to_move(), Some(0));
        assert!(!round.has_passed(1), "a new trick clears the passes");
    }

    #[test]
    fn the_free_rule_lets_a_passed_seat_play_again_in_the_same_trick() {
        use Rank::*;
        let hands = single_hands(&[
            &[Four, Nine, Ace],
            &[Five, Seven],
            &[Six, Eight],
            &[Seven, Ten],
        ]);
        let mut round =
            Round::with_pass_rule(hands, DuplicateRule::FirstDealtWins, PassRule::Free, 0).unwrap();
        play(&mut round, 0, Four);
        round.submit_move(1, Move::Pass).unwrap();
        play(&mut round, 2, Six);
        round.submit_move(3, Move::Pass).unwrap();
        play(&mut round, 0, Nine);
        assert_eq!(round.seat_to_move(), Some(1), "free: seat 1 is back in");
        assert!(!round.has_passed(1), "free never marks a seat as out");
        // Seat 1 can act again (here it passes once more) and play goes on.
        round.submit_move(1, Move::Pass).unwrap();
        assert_eq!(round.seat_to_move(), Some(2));
    }

    #[test]
    fn a_passed_seat_acts_again_in_the_next_trick() {
        use Rank::*;
        let hands = single_hands(&[&[Four, Nine], &[Five, Seven], &[Six, Eight]]);
        let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
        assert_eq!(round.pass_rule(), PassRule::Final, "the default rule");
        play(&mut round, 0, Four);
        round.submit_move(1, Move::Pass).unwrap();
        round.submit_move(2, Move::Pass).unwrap();
        // Trick over: seat 0 leads again, and seat 1 may play in the next one.
        assert_eq!(round.seat_to_move(), Some(0));
        play(&mut round, 0, Nine);
        assert_eq!(round.seat_to_move(), Some(1));
        round.submit_move(1, Move::Pass).unwrap();
    }

    #[test]
    fn when_the_winner_goes_out_the_lead_passes_to_the_next_active_seat_under_final() {
        use Rank::*;
        // Seat 0 leads its last card; 1 passes; 2 beats it with its last card
        // and goes out; seat 0 is already out; only seat 1 (passed) is left in
        // the round with cards? Use 4 seats so someone can still lead.
        let hands = single_hands(&[&[Four], &[Five, Seven], &[Six], &[Eight, Ten]]);
        let mut round =
            Round::with_pass_rule(hands, DuplicateRule::FirstDealtWins, PassRule::Final, 0)
                .unwrap();
        play(&mut round, 0, Four); // seat 0 is out
        round.submit_move(1, Move::Pass).unwrap();
        play(&mut round, 2, Six); // seat 2 is out, winner of the trick
        assert_eq!(round.seat_to_move(), Some(3));
        round.submit_move(3, Move::Pass).unwrap();
        // 1 and 3 passed, 0 and 2 are out: the trick is over and the next
        // active seat after the winner (2) leads: that is seat 3.
        assert_eq!(round.current_combo(), None);
        assert_eq!(round.seat_to_move(), Some(3));
    }

    /// Plays random legal rounds with a lot of passing, for every table
    /// size, both decks and both rules: every round must end, every seat
    /// must appear once in the finishing order, and (under `Final`) nobody is
    /// ever asked to move after passing in the same trick.
    #[test]
    fn random_rounds_always_finish_under_both_rules() {
        use crate::card::DeckVariant;
        use crate::deal::{deal, lowest_card_holder};
        use crate::deck::standard_deck;
        use rand::seq::SliceRandom;
        use rand::{RngExt, SeedableRng};

        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        for rule in [PassRule::Free, PassRule::Final] {
            for variant in [DeckVariant::Single, DeckVariant::Double] {
                for players in 3..=6u8 {
                    for _ in 0..150 {
                        let mut deck = standard_deck(variant);
                        deck.shuffle(&mut rng);
                        for (i, c) in deck.iter_mut().enumerate() {
                            c.deal_index = u8::try_from(i).unwrap();
                        }
                        let hands = deal(deck, players).unwrap();
                        let leader =
                            lowest_card_holder(&hands, DuplicateRule::FirstDealtWins).unwrap();
                        let mut round = Round::with_pass_rule(
                            hands,
                            DuplicateRule::FirstDealtWins,
                            rule,
                            leader,
                        )
                        .unwrap();
                        let mut passed_in_trick: Vec<SeatId> = Vec::new();
                        let mut steps = 0;
                        while let Some(seat) = round.seat_to_move() {
                            steps += 1;
                            assert!(steps < 10_000, "a round never ended ({rule:?})");
                            if round.current_combo().is_none() {
                                passed_in_trick.clear();
                            }
                            if rule == PassRule::Final {
                                assert!(
                                    !passed_in_trick.contains(&seat),
                                    "seat {seat} acts again after passing in the same trick"
                                );
                            }
                            let moves = round.legal_moves();
                            let pass = moves.contains(&Move::Pass) && rng.random_bool(0.5);
                            let mv = if pass {
                                Move::Pass
                            } else {
                                moves[rng.random_range(0..moves.len())].clone()
                            };
                            if mv == Move::Pass {
                                passed_in_trick.push(seat);
                            }
                            round.submit_move(seat, mv).unwrap();
                        }
                        let mut order = round.finishing_order().to_vec();
                        order.sort_unstable();
                        let expected: Vec<SeatId> = (0..players).collect();
                        assert_eq!(order, expected, "{rule:?} {variant:?} {players}p");
                    }
                }
            }
        }
    }
}
