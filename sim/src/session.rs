//! An interactive game: one human seat among strategy-driven seats.
//!
//! A `Session` plays a whole match (several rounds with role carry-over
//! and the card exchange) exactly the way `match_runner::run_match` does,
//! except that one seat's decisions come from a person. It never blocks:
//! after every action it plays the AI seats until the human must act
//! (give cards in the exchange, make a move, start the next round) or the
//! match is over, and logs everything as `GameEvent`s so a front end can
//! replay what the AI seats did. It is a pure state machine: no I/O.
//!
//! The human only ever sees public information: their own hand, every
//! seat's hand size, the plays and passes made, and, for the exchange,
//! the cards that involve them. `View` and `GameEvent` are what a client
//! receives, so nothing else may leak into them.

use std::sync::Arc;

use engine::{
    assign_roles, deal, exchange_counts_for_player_count, exchange_with_rule, lowest_card_holder,
    roles_for_player_count, Card, Combo, DeckVariant, DuplicateRule, ExchangeRule, Move, PassRule,
    Role, Round, SeatId,
};
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::hand_reading::PassTracker;
use crate::match_runner::{shuffled_numbered_deck, turn_context_for};
use crate::strategies::NeatStrategy;
use crate::strategy::Strategy;
use crate::training::role_score;

mod flow;
mod types;
mod view;

pub use self::types::*;
use self::view::explain;

struct PendingExchange {
    hands: Vec<Vec<Card>>,
    roles: Vec<Role>,
    give_count: usize,
}

pub struct Session {
    config: SessionConfig,
    /// One entry per seat; `None` is the human.
    ai: Vec<Option<AiSeat>>,
    rng: rand::rngs::StdRng,
    round: Option<Round>,
    round_deck: Vec<Card>,
    tracker: PassTracker,
    /// Rounds finished so far.
    finished_rounds: usize,
    previous_roles: Option<Vec<Role>>,
    previous_arschloch: Option<SeatId>,
    roles_history: Vec<Vec<Role>>,
    events: Vec<GameEvent>,
    phase: Phase,
    pending: Option<PendingExchange>,
    ids: IdMap,
}

impl Session {
    /// # Errors
    ///
    /// `SessionError::BadConfig` for an unsupported table size, zero
    /// rounds, a seat out of range, or the wrong number of AI seats.
    pub fn new(config: SessionConfig, ai: Vec<AiSeat>) -> Result<Self, SessionError> {
        if !(3..=6).contains(&config.player_count) {
            return Err(SessionError::BadConfig("a table has 3-6 seats".into()));
        }
        if config.rounds == 0 || config.rounds > 1000 {
            return Err(SessionError::BadConfig("rounds must be 1-1000".into()));
        }
        if config.human_seat >= config.player_count {
            return Err(SessionError::BadConfig(
                "the human seat is not at the table".into(),
            ));
        }
        if ai.len() != usize::from(config.player_count) - 1 {
            return Err(SessionError::BadConfig(format!(
                "{} opponents are needed, {} given",
                config.player_count - 1,
                ai.len()
            )));
        }
        let mut ai_iter = ai.into_iter();
        let seats: Vec<Option<AiSeat>> = (0..config.player_count)
            .map(|seat| {
                if seat == config.human_seat {
                    None
                } else {
                    ai_iter.next()
                }
            })
            .collect();
        let ids = if config.hide_deal_order {
            IdMap::shuffled(config.seed)
        } else {
            IdMap::identity()
        };
        let mut session = Self {
            rng: rand::rngs::StdRng::seed_from_u64(config.seed),
            tracker: PassTracker::new(usize::from(config.player_count), config.duplicate_rule),
            config,
            ai: seats,
            round: None,
            round_deck: Vec::new(),
            finished_rounds: 0,
            previous_roles: None,
            previous_arschloch: None,
            roles_history: Vec::new(),
            events: Vec::new(),
            phase: Phase::Playing,
            pending: None,
            ids,
        };
        session.start_round();
        Ok(session)
    }

    fn human(&self) -> SeatId {
        self.config.human_seat
    }

    #[must_use]
    pub fn phase(&self) -> Phase {
        self.phase
    }

    #[must_use]
    pub fn events_since(&self, since: usize) -> &[GameEvent] {
        &self.events[since.min(self.events.len())..]
    }

    /// The names of every seat, the human's as "You".
    #[must_use]
    pub fn seat_names(&self) -> Vec<String> {
        self.ai
            .iter()
            .map(|seat| {
                seat.as_ref()
                    .map_or_else(|| "You".to_owned(), |s| s.name.clone())
            })
            .collect()
    }

    /// The human's role in each finished round.
    #[must_use]
    pub fn human_roles(&self) -> Vec<Role> {
        self.roles_history
            .iter()
            .map(|roles| roles[usize::from(self.human())])
            .collect()
    }

    fn playing_human(&self) -> Result<(), SessionError> {
        if self.phase != Phase::Playing {
            return Err(SessionError::WrongPhase);
        }
        let round = self.round.as_ref().ok_or(SessionError::WrongPhase)?;
        if round.seat_to_move() != Some(self.human()) {
            return Err(SessionError::NotYourTurn);
        }
        Ok(())
    }

    /// Looks `ids` up in `hand`, refusing unknown and repeated ids.
    fn cards_from(map: &IdMap, hand: &[Card], ids: &[u8]) -> Result<Vec<Card>, SessionError> {
        let mut cards = Vec::with_capacity(ids.len());
        for &id in ids {
            let deal = map.from_client[usize::from(id)];
            if cards.iter().any(|c: &Card| c.deal_index == deal) {
                return Err(SessionError::DuplicateCard(id));
            }
            let card = hand
                .iter()
                .find(|c| c.deal_index == deal)
                .ok_or(SessionError::UnknownCard(id))?;
            cards.push(*card);
        }
        Ok(cards)
    }

    /// The human plays the cards with these ids (all of one rank).
    ///
    /// # Errors
    ///
    /// A `SessionError`; the game is unchanged when this fails.
    ///
    /// # Panics
    ///
    /// Only if an internal invariant is broken (a bug), never for bad input.
    pub fn play(&mut self, card_ids: &[u8]) -> Result<(), SessionError> {
        self.playing_human()?;
        let round = self.round.as_ref().expect("checked by playing_human");
        let cards = Self::cards_from(&self.ids, round.hand(self.human()), card_ids)?;
        let combo = Combo::new(cards).ok_or_else(|| {
            SessionError::IllegalMove("choose at least one card, all of the same rank".into())
        })?;
        self.apply(self.human(), &Move::Play(combo))
            .map_err(|e| SessionError::IllegalMove(explain(&e)))?;
        self.advance();
        Ok(())
    }

    /// The human passes.
    ///
    /// # Errors
    ///
    /// A `SessionError`; the game is unchanged when this fails.
    pub fn pass(&mut self) -> Result<(), SessionError> {
        self.playing_human()?;
        self.apply(self.human(), &Move::Pass)
            .map_err(|e| SessionError::IllegalMove(explain(&e)))?;
        self.advance();
        Ok(())
    }

    /// The human, holding a lower role, gives these cards in the exchange.
    ///
    /// # Errors
    ///
    /// A `SessionError`; the game is unchanged when this fails.
    ///
    /// # Panics
    ///
    /// Only if an internal invariant is broken (a bug), never for bad input.
    pub fn give(&mut self, card_ids: &[u8]) -> Result<(), SessionError> {
        if self.phase != Phase::Exchange {
            return Err(SessionError::WrongPhase);
        }
        let pending = self.pending.as_ref().ok_or(SessionError::WrongPhase)?;
        if card_ids.len() != pending.give_count {
            return Err(SessionError::IllegalMove(format!(
                "give exactly {} cards",
                pending.give_count
            )));
        }
        let selection = Self::cards_from(
            &self.ids,
            &pending.hands[usize::from(self.human())],
            card_ids,
        )?;
        let pending = self.pending.take().expect("checked above");
        let arschloch = self
            .previous_arschloch
            .expect("an exchange only follows a finished round");
        self.finish_exchange(pending.hands, &pending.roles, Some(selection), arschloch);
        Ok(())
    }

    /// Starts the next round after `round_over`.
    ///
    /// # Errors
    ///
    /// `SessionError::WrongPhase` unless the round is over and rounds remain.
    pub fn next_round(&mut self) -> Result<(), SessionError> {
        if self.phase != Phase::RoundOver {
            return Err(SessionError::WrongPhase);
        }
        self.start_round();
        Ok(())
    }
}

#[cfg(test)]
mod tests;
