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

use std::fmt;
use std::sync::Arc;

use engine::{
    assign_roles, deal, exchange_counts_for_player_count, exchange_with_rule, lowest_card_holder,
    roles_for_player_count, standard_deck, Card, Combo, DeckVariant, DuplicateRule, ExchangeRule,
    Move, PassRule, Rank, Role, Round, SeatId, Suit,
};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Serialize;

use crate::hand_reading::PassTracker;
use crate::match_runner::turn_context_for;
use crate::strategies::NeatStrategy;
use crate::strategy::Strategy;
use crate::training::role_score;

#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub player_count: u8,
    pub deck_variant: DeckVariant,
    pub duplicate_rule: DuplicateRule,
    pub pass_rule: PassRule,
    pub exchange_rule: ExchangeRule,
    pub rounds: usize,
    pub seed: u64,
    pub human_seat: u8,
    /// Number the cards shown to the client in a random order instead of
    /// their deal position (a card's position reveals which seat it was
    /// dealt to, and so which cards were exchanged).
    pub hide_deal_order: bool,
}

/// A strategy-driven seat.
#[derive(Clone)]
pub struct AiSeat {
    pub name: String,
    pub strategy: Arc<dyn Strategy>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionError {
    /// The move is for a seat that is not the human, or it is not their turn.
    NotYourTurn,
    /// The action does not fit the game's current phase.
    WrongPhase,
    UnknownCard(u8),
    DuplicateCard(u8),
    IllegalMove(String),
    BadConfig(String),
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotYourTurn => write!(f, "it is not your turn"),
            Self::WrongPhase => write!(f, "that action is not possible right now"),
            Self::UnknownCard(id) => write!(f, "card {id} is not in your hand"),
            Self::DuplicateCard(id) => write!(f, "card {id} was given twice"),
            Self::IllegalMove(reason) => write!(f, "{reason}"),
            Self::BadConfig(reason) => write!(f, "invalid game setup: {reason}"),
        }
    }
}

impl std::error::Error for SessionError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// The human holds a lower role and must choose cards to give.
    Exchange,
    /// The human is to move.
    Playing,
    RoundOver,
    MatchOver,
}

/// A card as a client sees it. `id` is unique within a round and is what
/// moves refer to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CardView {
    pub id: u8,
    /// 0 = Two ... 12 = Ace.
    pub rank: u8,
    /// 0 = Diamonds, 1 = Hearts, 2 = Spades, 3 = Clubs.
    pub suit: u8,
    pub label: String,
}

fn rank_label(rank: Rank) -> &'static str {
    match rank {
        Rank::Two => "2",
        Rank::Three => "3",
        Rank::Four => "4",
        Rank::Five => "5",
        Rank::Six => "6",
        Rank::Seven => "7",
        Rank::Eight => "8",
        Rank::Nine => "9",
        Rank::Ten => "10",
        Rank::Jack => "J",
        Rank::Queen => "Q",
        Rank::King => "K",
        Rank::Ace => "A",
    }
}

fn suit_label(suit: Suit) -> &'static str {
    match suit {
        Suit::Diamonds => "\u{2666}",
        Suit::Hearts => "\u{2665}",
        Suit::Spades => "\u{2660}",
        Suit::Clubs => "\u{2663}",
    }
}

impl From<Card> for CardView {
    fn from(card: Card) -> Self {
        Self {
            id: card.deal_index,
            rank: card.rank as u8,
            suit: card.suit as u8,
            label: format!("{}{}", rank_label(card.rank), suit_label(card.suit)),
        }
    }
}

/// The numbers cards go by on the client side of the session.
struct IdMap {
    to_client: Vec<u8>,
    from_client: Vec<u8>,
}

impl IdMap {
    fn identity() -> Self {
        let ids: Vec<u8> = (0..=255).collect();
        Self {
            to_client: ids.clone(),
            from_client: ids,
        }
    }

    fn shuffled(seed: u64) -> Self {
        let mut to_client: Vec<u8> = (0..=255).collect();
        // Its own generator: the game's random stream must not change.
        to_client.shuffle(&mut rand::rngs::StdRng::seed_from_u64(
            seed ^ 0x9E37_79B9_7F4A_7C15,
        ));
        let mut from_client = vec![0u8; 256];
        for (deal, &client) in to_client.iter().enumerate() {
            from_client[usize::from(client)] = u8::try_from(deal).expect("256 entries");
        }
        Self {
            to_client,
            from_client,
        }
    }
}

fn views(ids: &IdMap, cards: &[Card]) -> Vec<CardView> {
    cards
        .iter()
        .copied()
        .map(|card| {
            let mut view = CardView::from(card);
            view.id = ids.to_client[usize::from(card.deal_index)];
            view
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GameEvent {
    RoundStart {
        round: usize,
        /// Roles entering this round, by seat (none in round 1).
        roles: Option<Vec<Role>>,
        leader: SeatId,
        hand_sizes: Vec<usize>,
    },
    /// Who gave how many cards to whom (public); the cards themselves are
    /// only revealed to the seats involved, see `ExchangeYours`.
    Exchange {
        pairs: Vec<ExchangePair>,
    },
    /// The human's own part of the exchange.
    ExchangeYours {
        gave: Vec<CardView>,
        received: Vec<CardView>,
    },
    Play {
        seat: SeatId,
        cards: Vec<CardView>,
        hand_left: usize,
    },
    Pass {
        seat: SeatId,
    },
    /// Everyone else passed; `leader` leads the next trick.
    TrickEnd {
        leader: SeatId,
    },
    Finished {
        seat: SeatId,
        place: usize,
    },
    RoundEnd {
        finishing_order: Vec<SeatId>,
        roles: Vec<Role>,
    },
    MatchEnd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ExchangePair {
    pub from: SeatId,
    pub to: SeatId,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SeatView {
    pub seat: SeatId,
    pub name: String,
    pub is_human: bool,
    pub hand_size: usize,
    pub active: bool,
    /// The role this seat entered the round with.
    pub role: Option<Role>,
    /// Finishing place this round (1 = first out), once finished.
    pub place: Option<usize>,
    /// Passed in the current trick and so out of it (pass rule `final`).
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct TableView {
    pub seat: SeatId,
    pub cards: Vec<CardView>,
}

/// A rank the human can lead or follow with now, and how many of its cards
/// they may play at once (judged by the strongest subset of each size).
#[derive(Debug, Clone, Serialize)]
pub struct Playable {
    pub rank: u8,
    pub card_ids: Vec<u8>,
    pub sizes: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FinalView {
    /// The human's role in each round.
    pub roles: Vec<Role>,
    /// Mean role score of the human over the rounds: +1 always President
    /// .. -1 always last.
    pub score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct View {
    pub phase: Phase,
    /// 1-based round number (the round just finished while `round_over`).
    pub round: usize,
    pub rounds: usize,
    pub player_count: u8,
    pub pass_rule: PassRule,
    pub exchange_rule: ExchangeRule,
    pub human_seat: SeatId,
    pub hand: Vec<CardView>,
    pub seats: Vec<SeatView>,
    pub table: Option<TableView>,
    pub to_move: Option<SeatId>,
    pub must_lead: bool,
    pub playable: Vec<Playable>,
    /// How many cards the human must give (phase `exchange`).
    pub give_count: usize,
    /// Roles of every seat in each finished round.
    pub roles_history: Vec<Vec<Role>>,
    pub event_count: usize,
    #[serde(rename = "final")]
    pub final_result: Option<FinalView>,
}

/// One candidate move for the human with a model's opinion of it.
#[derive(Debug, Clone, Serialize)]
pub struct Advice {
    /// Empty for a pass.
    pub cards: Vec<CardView>,
    pub is_pass: bool,
    pub raw_score: f64,
}

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

    // ------------------------------------------------------------ rounds

    fn start_round(&mut self) {
        let config = self.config.clone();
        let mut deck = standard_deck(config.deck_variant);
        deck.shuffle(&mut self.rng);
        for (index, card) in deck.iter_mut().enumerate() {
            card.deal_index = u8::try_from(index).expect("deck sizes (52/104) fit in u8");
        }
        let hands = deal(deck, config.player_count)
            .expect("standard_deck always yields enough cards for a supported player count");
        if let (Some(roles), Some(arschloch)) =
            (self.previous_roles.clone(), self.previous_arschloch)
        {
            let give_count = self.human_give_count(&roles);
            if give_count > 0 {
                self.pending = Some(PendingExchange {
                    hands,
                    roles,
                    give_count,
                });
                self.phase = Phase::Exchange;
            } else {
                self.finish_exchange(hands, &roles, None, arschloch);
            }
        } else {
            let leader = lowest_card_holder(&hands, config.duplicate_rule)
                .expect("a freshly dealt hand set is never empty");
            self.begin_play(hands, leader);
        }
    }

    /// How many cards the human must choose to give (0 when they are not
    /// a lower role, or their pair exchanges nothing).
    fn human_give_count(&self, roles: &[Role]) -> usize {
        // Under the forced rule there is nothing to choose.
        if self.config.exchange_rule == ExchangeRule::Forced {
            return 0;
        }
        let player_count = self.config.player_count;
        let (Some(order), Some(counts)) = (
            roles_for_player_count(player_count),
            exchange_counts_for_player_count(player_count),
        ) else {
            return 0;
        };
        let mine = roles[usize::from(self.human())];
        counts
            .iter()
            .enumerate()
            .find(|&(i, &count)| count > 0 && order[order.len() - 1 - i] == mine)
            .map_or(0, |(_, &count)| usize::from(count))
    }

    /// Runs the exchange (the human's selection, when they give cards, is
    /// `human_selection`) and starts play with the Arschloch leading.
    fn finish_exchange(
        &mut self,
        mut hands: Vec<Vec<Card>>,
        roles: &[Role],
        human_selection: Option<Vec<Card>>,
        leader: SeatId,
    ) {
        let rule = self.config.duplicate_rule;
        let human = usize::from(self.human());
        let before = hands[human].clone();
        let ai = &self.ai;
        let rng = &mut self.rng;
        let mut selection = human_selection.clone();
        exchange_with_rule(
            &mut hands,
            roles,
            rule,
            self.config.exchange_rule,
            |seat, hand, count, rule| match &ai[seat] {
                Some(ai) => ai.strategy.choose_exchange_cards(hand, count, rule, rng),
                None => selection
                    .take()
                    .expect("the human's selection was collected before the exchange"),
            },
        )
        .expect(
            "previous_roles always came from assign_roles for this player_count, and every \
             selection is validated before the exchange",
        );
        self.record_exchange(roles, &before, &hands[human], human_selection);
        self.begin_play(hands, leader);
    }

    fn record_exchange(
        &mut self,
        roles: &[Role],
        before: &[Card],
        after: &[Card],
        human_selection: Option<Vec<Card>>,
    ) {
        let player_count = self.config.player_count;
        let order = roles_for_player_count(player_count).expect("supported size");
        let counts = exchange_counts_for_player_count(player_count).expect("supported size");
        let seat_of = |role: Role| -> SeatId {
            SeatId::try_from(
                roles
                    .iter()
                    .position(|&r| r == role)
                    .expect("every role is present"),
            )
            .expect("seat fits")
        };
        let pairs: Vec<ExchangePair> = counts
            .iter()
            .enumerate()
            .filter(|&(_, &count)| count > 0)
            .map(|(i, &count)| ExchangePair {
                from: seat_of(order[order.len() - 1 - i]),
                to: seat_of(order[i]),
                count: usize::from(count),
            })
            .collect();
        let involved = pairs
            .iter()
            .any(|p| p.from == self.human() || p.to == self.human());
        self.events.push(GameEvent::Exchange { pairs });
        if involved {
            let gave = match human_selection {
                Some(selection) => views(&self.ids, &selection),
                // No selection: the exchange was forced (a lower role gives its
                // highest cards) or the human was the higher role (hands back
                // its lowest): in both cases whatever left their hand.
                None => views(
                    &self.ids,
                    &before
                        .iter()
                        .copied()
                        .filter(|c| !after.contains(c))
                        .collect::<Vec<_>>(),
                ),
            };
            let received = views(
                &self.ids,
                &after
                    .iter()
                    .copied()
                    .filter(|c| !before.contains(c))
                    .collect::<Vec<_>>(),
            );
            self.events
                .push(GameEvent::ExchangeYours { gave, received });
        }
    }

    fn begin_play(&mut self, hands: Vec<Vec<Card>>, leader: SeatId) {
        self.round_deck = hands.iter().flatten().copied().collect();
        let hand_sizes = hands.iter().map(Vec::len).collect();
        self.round = Some(
            Round::with_pass_rule(
                hands,
                self.config.duplicate_rule,
                self.config.pass_rule,
                leader,
            )
            .expect("player_count/leader are always valid for a supported table size"),
        );
        self.tracker = PassTracker::new(
            usize::from(self.config.player_count),
            self.config.duplicate_rule,
        );
        self.events.push(GameEvent::RoundStart {
            round: self.finished_rounds + 1,
            roles: self.previous_roles.clone(),
            leader,
            hand_sizes,
        });
        self.phase = Phase::Playing;
        self.advance();
    }

    // ------------------------------------------------------------- moves

    /// Plays the AI seats until the human must act or the round ends.
    fn advance(&mut self) {
        loop {
            let (seat, complete) = {
                let round = self.round.as_ref().expect("a round is in progress");
                (round.seat_to_move(), round.is_complete())
            };
            if complete {
                self.end_round();
                return;
            }
            let seat = seat.expect("an incomplete round has a seat to move");
            if seat == self.human() {
                self.phase = Phase::Playing;
                return;
            }
            let chosen = {
                let round = self.round.as_ref().expect("a round is in progress");
                let legal = round.legal_moves();
                let context = turn_context_for(
                    round,
                    seat,
                    self.config.player_count,
                    &self.round_deck,
                    &mut self.tracker,
                );
                let strategy = &self.ai[usize::from(seat)]
                    .as_ref()
                    .expect("only the human seat has no strategy")
                    .strategy;
                strategy.choose_play(&legal, self.config.duplicate_rule, &context, &mut self.rng)
            };
            self.apply(seat, &chosen)
                .expect("strategies only choose from the moves engine just reported as legal");
        }
    }

    /// Submits `mv` for `seat` and logs it.
    fn apply(&mut self, seat: SeatId, mv: &Move) -> Result<(), engine::MoveError> {
        let round = self.round.as_mut().expect("a round is in progress");
        let had_combo = round.current_combo().is_some();
        let finished_before = round.finishing_order().len();
        round.submit_move(seat, mv.clone())?;
        match mv {
            Move::Pass => self.events.push(GameEvent::Pass { seat }),
            Move::Play(combo) => {
                let hand_left = round.hand_size(seat);
                self.events.push(GameEvent::Play {
                    seat,
                    cards: views(&self.ids, combo.cards()),
                    hand_left,
                });
            }
        }
        if round.finishing_order().len() > finished_before {
            self.events.push(GameEvent::Finished {
                seat,
                place: round.finishing_order().len(),
            });
        }
        if had_combo && round.current_combo().is_none() {
            if let Some(leader) = round.seat_to_move() {
                self.events.push(GameEvent::TrickEnd { leader });
            }
        }
        Ok(())
    }

    fn end_round(&mut self) {
        let round = self.round.as_ref().expect("a round is in progress");
        let finishing_order = round.finishing_order().to_vec();
        let roles = assign_roles(&finishing_order, self.config.player_count)
            .expect("finishing_order is always a valid permutation for a supported player count");
        self.previous_arschloch = finishing_order.last().copied();
        self.roles_history.push(roles.clone());
        self.events.push(GameEvent::RoundEnd {
            finishing_order,
            roles: roles.clone(),
        });
        self.previous_roles = Some(roles);
        self.finished_rounds += 1;
        if self.finished_rounds >= self.config.rounds {
            self.events.push(GameEvent::MatchEnd);
            self.phase = Phase::MatchOver;
        } else {
            self.phase = Phase::RoundOver;
        }
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

    /// The moves `Round::legal_moves` lists for the human (canonical
    /// weakest/strongest subsets); empty unless it is their turn.
    #[must_use]
    pub fn legal_moves(&self) -> Vec<Move> {
        if self.playing_human().is_err() {
            return Vec::new();
        }
        self.round
            .as_ref()
            .map(Round::legal_moves)
            .unwrap_or_default()
    }

    /// Every move `Round::legal_moves` lists for the human, ranked by what
    /// `advisor` (a trained model) thinks of it, best first. Only the
    /// human's own options and the model's scores are returned: the model
    /// sees exactly what the human sees.
    ///
    /// # Errors
    ///
    /// `SessionError::WrongPhase` / `NotYourTurn` unless the human is to move.
    ///
    /// # Panics
    ///
    /// Only if an internal invariant is broken (a bug), never for bad input.
    pub fn advice(&self, advisor: &NeatStrategy) -> Result<Vec<Advice>, SessionError> {
        self.playing_human()?;
        let round = self.round.as_ref().expect("checked by playing_human");
        let legal = round.legal_moves();
        let mut tracker = self.tracker.clone();
        let context = turn_context_for(
            round,
            self.human(),
            self.config.player_count,
            &self.round_deck,
            &mut tracker,
        );
        let mut scored = advisor.score_candidates(&legal, self.config.duplicate_rule, &context);
        // Best first; equal scores keep the engine's listing order.
        scored.sort_by(|a, b| b.raw_score.total_cmp(&a.raw_score));
        Ok(scored
            .into_iter()
            .map(|candidate| match candidate.candidate {
                Move::Pass => Advice {
                    cards: Vec::new(),
                    is_pass: true,
                    raw_score: candidate.raw_score,
                },
                Move::Play(combo) => Advice {
                    cards: views(&self.ids, combo.cards()),
                    is_pass: false,
                    raw_score: candidate.raw_score,
                },
            })
            .collect())
    }

    // -------------------------------------------------------------- view

    fn final_view(&self) -> Option<FinalView> {
        (self.phase == Phase::MatchOver).then(|| {
            let roles = self.human_roles();
            #[allow(clippy::cast_precision_loss)] // a handful of rounds
            let score = roles
                .iter()
                .map(|&role| role_score(role, self.config.player_count))
                .sum::<f64>()
                / roles.len().max(1) as f64;
            FinalView { roles, score }
        })
    }

    /// Everything the human may know right now.
    #[must_use]
    pub fn view(&self) -> View {
        let human = self.human();
        let rule = self.config.duplicate_rule;
        let names = self.seat_names();
        let in_exchange = self.phase == Phase::Exchange;
        // During the exchange the previous round is over and the next has not
        // started: no table, no places, nobody to move.
        let round = if in_exchange {
            None
        } else {
            self.round.as_ref()
        };

        let mut hand: Vec<Card> = if in_exchange {
            self.pending
                .as_ref()
                .map(|p| p.hands[usize::from(human)].clone())
                .unwrap_or_default()
        } else {
            round.map(|r| r.hand(human).to_vec()).unwrap_or_default()
        };
        hand.sort_by(|a, b| a.compare(b, rule));

        let places: Vec<Option<usize>> = (0..self.config.player_count)
            .map(|seat| {
                round.and_then(|r| {
                    r.finishing_order()
                        .iter()
                        .position(|&s| s == seat)
                        .map(|p| p + 1)
                })
            })
            .collect();
        let seats = (0..self.config.player_count)
            .map(|seat| SeatView {
                seat,
                name: names[usize::from(seat)].clone(),
                is_human: seat == human,
                hand_size: if in_exchange {
                    self.pending
                        .as_ref()
                        .map_or(0, |p| p.hands[usize::from(seat)].len())
                } else {
                    round.map_or(0, |r| r.hand_size(seat))
                },
                active: !in_exchange && round.is_some_and(|r| r.is_active(seat)),
                role: self
                    .previous_roles
                    .as_ref()
                    .map(|roles| roles[usize::from(seat)]),
                place: places[usize::from(seat)],
                passed: self.phase == Phase::Playing && round.is_some_and(|r| r.has_passed(seat)),
            })
            .collect();

        let to_move = if self.phase == Phase::Playing {
            round.and_then(Round::seat_to_move)
        } else {
            None
        };
        let table = round.and_then(|r| {
            r.current_combo().and_then(|combo| {
                r.play_history().last().map(|(seat, _)| TableView {
                    seat: *seat,
                    cards: views(&self.ids, combo.cards()),
                })
            })
        });
        let must_lead = self.phase == Phase::Playing
            && to_move == Some(human)
            && round.is_some_and(|r| r.current_combo().is_none());
        let playable = if to_move == Some(human) {
            round.map_or_else(Vec::new, |r| {
                playable_ranks(&self.ids, &hand, r.current_combo(), rule)
            })
        } else {
            Vec::new()
        };
        let final_result = self.final_view();
        View {
            phase: self.phase,
            round: if matches!(self.phase, Phase::RoundOver | Phase::MatchOver) {
                self.finished_rounds
            } else {
                self.finished_rounds + 1
            },
            rounds: self.config.rounds,
            player_count: self.config.player_count,
            pass_rule: self.config.pass_rule,
            exchange_rule: self.config.exchange_rule,
            human_seat: human,
            hand: views(&self.ids, &hand),
            seats,
            table,
            to_move,
            must_lead,
            playable,
            give_count: self.pending.as_ref().map_or(0, |p| p.give_count),
            roles_history: self.roles_history.clone(),
            event_count: self.events.len(),
            final_result,
        }
    }
}

/// The ranks the human can play now. A rank offers a size when its
/// strongest subset of that size is playable (always on a lead; against a
/// table combo only the same size, and it must beat it).
fn playable_ranks(
    ids: &IdMap,
    hand: &[Card],
    current: Option<&Combo>,
    rule: DuplicateRule,
) -> Vec<Playable> {
    engine::rank_groups(hand)
        .into_iter()
        .filter_map(|mut group| {
            group.sort_by(|a, b| a.compare(b, rule));
            let sizes: Vec<usize> = (1..=group.len())
                .filter(|&size| match current {
                    None => true,
                    Some(table) => {
                        size == table.size()
                            && Combo::new(group[group.len() - size..].to_vec())
                                .is_some_and(|strongest| strongest.beats(table, rule))
                    }
                })
                .collect();
            (!sizes.is_empty()).then(|| Playable {
                rank: group[0].rank as u8,
                card_ids: group
                    .iter()
                    .map(|c| ids.to_client[usize::from(c.deal_index)])
                    .collect(),
                sizes,
            })
        })
        .collect()
}

fn explain(error: &engine::MoveError) -> String {
    use engine::MoveError;
    match error {
        MoveError::RoundAlreadyComplete => "the round is over".into(),
        MoveError::NotYourTurn { .. } => "it is not your turn".into(),
        MoveError::CannotPassOnLead => "you lead this trick, so you must play".into(),
        MoveError::CardNotInHand(_) => "you do not hold one of those cards".into(),
        MoveError::ComboDoesNotBeat => {
            "those cards do not beat the cards on the table (same number of cards, higher rank)"
                .into()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand_reading::PassCeilings;
    use crate::strategies::LowestLegal;
    use crate::strategy::TurnContext;
    use crate::{run_match, MatchConfig};

    fn ai(count: usize) -> Vec<AiSeat> {
        (0..count)
            .map(|_| AiSeat {
                name: "LowestLegal".into(),
                strategy: Arc::new(LowestLegal),
            })
            .collect()
    }

    fn config(players: u8, deck: DeckVariant, rounds: usize, human_seat: u8) -> SessionConfig {
        SessionConfig {
            player_count: players,
            deck_variant: deck,
            duplicate_rule: DuplicateRule::FirstDealtWins,
            pass_rule: PassRule::default(),
            exchange_rule: ExchangeRule::default(),
            rounds,
            seed: 7,
            human_seat,
            hide_deal_order: false,
        }
    }

    /// What `LowestLegal` would choose, for a scripted human.
    fn scripted_move(session: &Session) -> Move {
        let moves = session.legal_moves();
        let hand: Vec<Card> = Vec::new();
        let context = TurnContext {
            seat: 0,
            hand: &hand,
            opponents: Vec::new(),
            unseen_cards: Vec::new(),
            own_pass_ceilings: PassCeilings::default(),
            current_combo: None,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        LowestLegal.choose_play(&moves, DuplicateRule::FirstDealtWins, &context, &mut rng)
    }

    /// Plays the human's seat like `LowestLegal` until the match is over.
    fn play_out(session: &mut Session) {
        let mut guard = 0;
        while session.phase() != Phase::MatchOver {
            guard += 1;
            assert!(guard < 10_000, "the game does not end");
            match session.phase() {
                Phase::Exchange => {
                    let view = session.view();
                    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
                    let hand: Vec<Card> = view
                        .hand
                        .iter()
                        .map(|c| Card::new(rank_from(c.rank), suit_from(c.suit), c.id))
                        .collect();
                    let give = LowestLegal.choose_exchange_cards(
                        &hand,
                        view.give_count,
                        DuplicateRule::FirstDealtWins,
                        &mut rng,
                    );
                    let ids: Vec<u8> = give.iter().map(|c| c.deal_index).collect();
                    session.give(&ids).unwrap();
                }
                Phase::Playing => match scripted_move(session) {
                    Move::Pass => session.pass().unwrap(),
                    Move::Play(combo) => {
                        let ids: Vec<u8> = combo.cards().iter().map(|c| c.deal_index).collect();
                        session.play(&ids).unwrap();
                    }
                },
                Phase::RoundOver => session.next_round().unwrap(),
                Phase::MatchOver => unreachable!(),
            }
        }
    }

    fn rank_from(rank: u8) -> Rank {
        use Rank::*;
        [
            Two, Three, Four, Five, Six, Seven, Eight, Nine, Ten, Jack, Queen, King, Ace,
        ][usize::from(rank)]
    }

    fn suit_from(suit: u8) -> Suit {
        [Suit::Diamonds, Suit::Hearts, Suit::Spades, Suit::Clubs][usize::from(suit)]
    }

    #[test]
    fn a_session_plays_a_whole_match_for_every_table_size_and_deck() {
        for players in 3..=6u8 {
            for deck in [DeckVariant::Single, DeckVariant::Double] {
                for human_seat in [0, players - 1] {
                    let mut session = Session::new(
                        config(players, deck, 5, human_seat),
                        ai(usize::from(players) - 1),
                    )
                    .unwrap();
                    play_out(&mut session);
                    let view = session.view();
                    assert_eq!(view.roles_history.len(), 5, "{players} {deck:?}");
                    for roles in &view.roles_history {
                        assert_eq!(roles.len(), usize::from(players));
                    }
                    let result = view.final_result.expect("a finished match has a result");
                    assert_eq!(result.roles.len(), 5);
                    assert!((-1.0..=1.0).contains(&result.score));
                }
            }
        }
    }

    #[test]
    fn the_ai_seats_play_exactly_as_run_match_does() {
        // A scripted human that plays like LowestLegal, among LowestLegal
        // seats, must reproduce run_match with four LowestLegal seats: the
        // same shuffles, exchanges and moves.
        for players in 3..=6u8 {
            let mut session = Session::new(
                config(players, DeckVariant::Single, 6, 0),
                ai(usize::from(players) - 1),
            )
            .unwrap();
            play_out(&mut session);
            let strategies: Vec<Arc<dyn Strategy>> = (0..players)
                .map(|_| Arc::new(LowestLegal) as Arc<dyn Strategy>)
                .collect();
            let expected = run_match(
                &MatchConfig {
                    player_count: players,
                    deck_variant: DeckVariant::Single,
                    duplicate_rule: DuplicateRule::FirstDealtWins,
                    rounds: 6,
                    seed: 7,
                    pass_rule: engine::PassRule::default(),
                    exchange_rule: engine::ExchangeRule::default(),
                },
                &strategies,
            );
            assert_eq!(
                session.view().roles_history,
                expected.role_history,
                "{players} players"
            );
        }
    }

    #[test]
    fn the_view_and_events_never_contain_another_seats_cards() {
        let mut session = Session::new(config(4, DeckVariant::Single, 4, 1), ai(3)).unwrap();
        let mut seen_events = 0;
        // Cards the human gave away are legitimately known to be elsewhere.
        let mut known: Vec<u8> = Vec::new();
        let mut guard = 0;
        while session.phase() != Phase::MatchOver {
            guard += 1;
            assert!(guard < 10_000);
            let view = session.view();
            // The human's real hand, independent of what the view claims.
            let own: Vec<u8> = match &session.pending {
                Some(pending) => pending.hands[1].iter().map(|c| c.deal_index).collect(),
                None => session.round.as_ref().map_or_else(Vec::new, |r| {
                    r.hand(1).iter().map(|c| c.deal_index).collect()
                }),
            };
            let events = session.events_since(seen_events).to_vec();
            seen_events += events.len();
            if events
                .iter()
                .any(|e| matches!(e, GameEvent::RoundStart { .. }))
            {
                known.clear();
            }
            for event in &events {
                if let GameEvent::ExchangeYours { gave, .. } = event {
                    known.extend(gave.iter().map(|c| c.id));
                }
            }
            // Ids restart every round, so events spanning a round boundary
            // are not comparable with the current hands; the view always is.
            let crosses_a_round = events
                .iter()
                .any(|e| matches!(e, GameEvent::RoundStart { .. } | GameEvent::RoundEnd { .. }));
            let text = if crosses_a_round {
                serde_json::to_string(&view).unwrap()
            } else {
                serde_json::to_string(&(&view, &events)).unwrap()
            };
            for seat in (0..4u8).filter(|&s| s != 1) {
                let hidden: Vec<u8> = session.round.as_ref().map_or_else(Vec::new, |r| {
                    r.hand(seat).iter().map(|c| c.deal_index).collect()
                });
                for id in hidden {
                    if own.contains(&id) || known.contains(&id) {
                        continue;
                    }
                    assert!(
                        !text.contains(&format!("\"id\":{id},")),
                        "card {id} of seat {seat}'s hand leaked: {text}"
                    );
                }
            }
            match session.phase() {
                Phase::Exchange => {
                    let ids: Vec<u8> = view
                        .hand
                        .iter()
                        .take(view.give_count)
                        .map(|c| c.id)
                        .collect();
                    session.give(&ids).unwrap();
                }
                Phase::Playing => match scripted_move(&session) {
                    Move::Pass => session.pass().unwrap(),
                    Move::Play(combo) => {
                        let ids: Vec<u8> = combo.cards().iter().map(|c| c.deal_index).collect();
                        session.play(&ids).unwrap();
                    }
                },
                Phase::RoundOver => session.next_round().unwrap(),
                Phase::MatchOver => unreachable!(),
            }
        }
    }

    fn json(session: &Session) -> String {
        serde_json::to_string(&(session.view(), session.events_since(0))).unwrap()
    }

    /// A session at the human's first turn (the human leads or follows).
    fn at_first_turn(players: u8, human_seat: u8, seed: u64) -> Session {
        let mut cfg = config(players, DeckVariant::Single, 3, human_seat);
        cfg.seed = seed;
        let mut session = Session::new(cfg, ai(usize::from(players) - 1)).unwrap();
        assert_eq!(session.phase(), Phase::Playing);
        // Let the round proceed until it is the human's turn (it already is,
        // `new` advances the AI seats); keep the type used.
        let _ = &mut session;
        session
    }

    #[test]
    fn rejected_actions_leave_the_game_unchanged() {
        let mut session = at_first_turn(4, 0, 3);
        let before = json(&session);
        let own: Vec<u8> = session.view().hand.iter().map(|c| c.id).collect();
        let absent = (0..52u8).find(|id| !own.contains(id)).unwrap();
        let attempts: Vec<(&str, Result<(), SessionError>)> = vec![
            ("unknown card", session.play(&[absent])),
            ("duplicate card", session.play(&[own[0], own[0]])),
            ("empty", session.play(&[])),
            ("give in the wrong phase", session.give(&[own[0]])),
            ("next round too early", session.next_round()),
        ];
        for (what, result) in attempts {
            assert!(result.is_err(), "{what} was accepted");
            assert_eq!(json(&session), before, "{what} changed the game");
        }
        // Two cards of different ranks.
        let view = session.view();
        let first = &view.hand[0];
        let other = view.hand.iter().find(|c| c.rank != first.rank).unwrap();
        assert!(session.play(&[first.id, other.id]).is_err());
        assert_eq!(json(&session), before);
        // A pass on the lead (the human leads when nothing is on the table).
        if session.view().must_lead {
            assert!(session.pass().is_err());
            assert_eq!(json(&session), before);
        }
    }

    #[test]
    fn nothing_works_after_the_match_ends() {
        let mut session = Session::new(config(3, DeckVariant::Single, 1, 0), ai(2)).unwrap();
        play_out(&mut session);
        let before = json(&session);
        assert_eq!(session.play(&[0]), Err(SessionError::WrongPhase));
        assert_eq!(session.pass(), Err(SessionError::WrongPhase));
        assert_eq!(session.give(&[0]), Err(SessionError::WrongPhase));
        assert_eq!(session.next_round(), Err(SessionError::WrongPhase));
        assert_eq!(json(&session), before);
    }

    #[test]
    fn any_subset_of_a_rank_that_beats_the_table_is_accepted() {
        // Search seeds for a first turn where the human may lead a rank held
        // three times; leading a pair out of the triple (a non-canonical
        // choice: the engine lists only weakest and strongest) must work.
        for seed in 0..200 {
            let mut session = at_first_turn(4, 0, seed);
            let view = session.view();
            if !view.must_lead {
                continue;
            }
            let Some(triple) = view.playable.iter().find(|p| p.card_ids.len() >= 3) else {
                continue;
            };
            // The middle two cards of the group, not the weakest or strongest pair.
            let ids = [triple.card_ids[1], triple.card_ids[2]];
            if triple.card_ids.len() == 3 {
                // pair (1,2) is the strongest pair; use (0,2) instead: neither
                // weakest nor strongest.
                let ids = [triple.card_ids[0], triple.card_ids[2]];
                session.play(&ids).unwrap();
            } else {
                session.play(&ids).unwrap();
            }
            return;
        }
        panic!("no seed produced a leading triple");
    }

    #[test]
    fn the_human_as_a_lower_role_chooses_the_cards_to_give() {
        // Play round 1 out; find a seed where the human ends lower and the
        // second round opens in the exchange phase (free exchange rule).
        for seed in 0..300 {
            let mut cfg = config(4, DeckVariant::Single, 3, 0);
            cfg.exchange_rule = ExchangeRule::Free;
            cfg.seed = seed;
            let mut session = Session::new(cfg, ai(3)).unwrap();
            // Finish round 1 with the scripted human.
            while session.phase() == Phase::Playing {
                match scripted_move(&session) {
                    Move::Pass => session.pass().unwrap(),
                    Move::Play(c) => {
                        let ids: Vec<u8> = c.cards().iter().map(|c| c.deal_index).collect();
                        session.play(&ids).unwrap();
                    }
                }
            }
            session.next_round().unwrap();
            if session.phase() != Phase::Exchange {
                continue;
            }
            let view = session.view();
            assert!(view.give_count >= 1);
            let hand_before = view.hand.clone();
            // Wrong count and a foreign card are refused.
            assert!(session.give(&[]).is_err());
            let ids: Vec<u8> = hand_before
                .iter()
                .rev()
                .take(view.give_count)
                .map(|c| c.id)
                .collect();
            session.give(&ids).unwrap();
            assert_eq!(session.phase(), Phase::Playing);
            let yours = session
                .events_since(0)
                .iter()
                .find_map(|e| match e {
                    GameEvent::ExchangeYours { gave, received } => {
                        Some((gave.clone(), received.clone()))
                    }
                    _ => None,
                })
                .expect("the human sees their own exchange");
            assert_eq!(yours.0.iter().map(|c| c.id).collect::<Vec<_>>(), ids);
            assert_eq!(yours.1.len(), view.give_count);
            return;
        }
        panic!("the human never ended lower in 300 seeds");
    }

    #[test]
    fn the_exchange_view_shows_a_fresh_table_not_the_last_round() {
        for seed in 0..300 {
            let mut cfg = config(4, DeckVariant::Single, 3, 0);
            cfg.exchange_rule = ExchangeRule::Free;
            cfg.seed = seed;
            let mut session = Session::new(cfg, ai(3)).unwrap();
            while session.phase() == Phase::Playing {
                match scripted_move(&session) {
                    Move::Pass => session.pass().unwrap(),
                    Move::Play(c) => {
                        let ids: Vec<u8> = c.cards().iter().map(|c| c.deal_index).collect();
                        session.play(&ids).unwrap();
                    }
                }
            }
            session.next_round().unwrap();
            if session.phase() != Phase::Exchange {
                continue;
            }
            let view = session.view();
            assert!(
                view.table.is_none(),
                "last round's final combo is still shown"
            );
            assert!(
                view.seats.iter().all(|s| s.place.is_none()),
                "last round's places are still shown"
            );
            assert!(view.to_move.is_none() && !view.must_lead && view.playable.is_empty());
            return;
        }
        panic!("the human never ended lower in 300 seeds");
    }

    #[test]
    fn under_the_forced_rule_the_human_as_a_lower_role_gives_the_highest_cards_without_choosing() {
        for seed in 0..300 {
            let mut cfg = config(4, DeckVariant::Single, 3, 0);
            cfg.exchange_rule = ExchangeRule::Forced;
            cfg.seed = seed;
            let mut session = Session::new(cfg, ai(3)).unwrap();
            while session.phase() == Phase::Playing {
                match scripted_move(&session) {
                    Move::Pass => session.pass().unwrap(),
                    Move::Play(c) => {
                        let ids: Vec<u8> = c.cards().iter().map(|c| c.deal_index).collect();
                        session.play(&ids).unwrap();
                    }
                }
            }
            let role = session.view().roles_history[0][0];
            session.next_round().unwrap();
            assert_ne!(
                session.phase(),
                Phase::Exchange,
                "nothing to choose when forced"
            );
            if !matches!(role, Role::ViceArschloch | Role::Arschloch) {
                continue;
            }
            let (gave, received) = session
                .events_since(0)
                .iter()
                .find_map(|e| match e {
                    GameEvent::ExchangeYours { gave, received } => {
                        Some((gave.clone(), received.clone()))
                    }
                    _ => None,
                })
                .expect("a lower role still sees what was taken and received");
            let count = if role == Role::Arschloch { 2 } else { 1 };
            assert_eq!(gave.len(), count);
            assert_eq!(received.len(), count);
            // What was given is at least as high as anything kept (rank order).
            let kept_max = session
                .view()
                .hand
                .iter()
                .filter(|c| {
                    !gave.iter().any(|g| g.id == c.id) && !received.iter().any(|r| r.id == c.id)
                })
                .map(|c| (c.rank, c.suit))
                .max();
            let given_min = gave.iter().map(|c| (c.rank, c.suit)).min().unwrap();
            if let Some(kept) = kept_max {
                assert!(
                    given_min >= kept,
                    "seed {seed}: gave {gave:?} but kept a higher card"
                );
            }
            // And what came back is from the president's low end.
            assert!(
                received.iter().all(|c| c.rank <= 6),
                "received {received:?}"
            );
            return;
        }
        panic!("the human never ended in a lower role in 300 seeds");
    }

    #[test]
    fn the_human_as_a_higher_role_gets_cards_without_choosing() {
        for seed in 0..300 {
            let mut cfg = config(4, DeckVariant::Single, 3, 0);
            cfg.seed = seed;
            let mut session = Session::new(cfg, ai(3)).unwrap();
            while session.phase() == Phase::Playing {
                match scripted_move(&session) {
                    Move::Pass => session.pass().unwrap(),
                    Move::Play(c) => {
                        let ids: Vec<u8> = c.cards().iter().map(|c| c.deal_index).collect();
                        session.play(&ids).unwrap();
                    }
                }
            }
            let role = session.view().roles_history[0][0];
            session.next_round().unwrap();
            if role != Role::President {
                continue;
            }
            assert_ne!(session.phase(), Phase::Exchange);
            let yours = session.events_since(0).iter().any(|e| {
                matches!(e, GameEvent::ExchangeYours { gave, received } if gave.len() == 2 && received.len() == 2)
            });
            assert!(yours, "the president hands back 2 cards and receives 2");
            return;
        }
        panic!("the human was never President in 300 seeds");
    }

    #[test]
    fn bad_setups_are_refused() {
        let bad = |cfg: SessionConfig, n: usize| Session::new(cfg, ai(n)).err().unwrap();
        assert!(matches!(
            bad(config(2, DeckVariant::Single, 1, 0), 1),
            SessionError::BadConfig(_)
        ));
        assert!(matches!(
            bad(config(7, DeckVariant::Single, 1, 0), 6),
            SessionError::BadConfig(_)
        ));
        assert!(matches!(
            bad(config(4, DeckVariant::Single, 0, 0), 3),
            SessionError::BadConfig(_)
        ));
        assert!(matches!(
            bad(config(4, DeckVariant::Single, 1, 4), 3),
            SessionError::BadConfig(_)
        ));
        assert!(matches!(
            bad(config(4, DeckVariant::Single, 1, 0), 2),
            SessionError::BadConfig(_)
        ));
    }

    /// Plays a whole match from what the client sees only.
    fn play_by_view(session: &mut Session) {
        let mut guard = 0;
        while session.phase() != Phase::MatchOver {
            guard += 1;
            assert!(guard < 10_000);
            let view = session.view();
            match view.phase {
                Phase::Exchange => {
                    let ids: Vec<u8> = view
                        .hand
                        .iter()
                        .take(view.give_count)
                        .map(|c| c.id)
                        .collect();
                    session.give(&ids).unwrap();
                }
                Phase::Playing => match view.playable.first() {
                    Some(rank) => {
                        let size = rank.sizes[0];
                        let ids = &rank.card_ids[rank.card_ids.len() - size..];
                        session.play(ids).unwrap();
                    }
                    None => session.pass().unwrap(),
                },
                Phase::RoundOver => session.next_round().unwrap(),
                Phase::MatchOver => unreachable!(),
            }
        }
    }

    #[test]
    fn hidden_deal_order_keeps_the_game_playable_and_unrevealing() {
        // With the deal position as the id, `id % players` is the seat a card
        // was dealt to. Hidden ids must break that for every seat and still
        // play identically.
        let mut revealing = 0;
        let mut total = 0;
        for seed in 0..20 {
            let mut cfg = config(4, DeckVariant::Single, 2, 1);
            cfg.seed = seed;
            cfg.hide_deal_order = true;
            let mut hidden = Session::new(cfg.clone(), ai(3)).unwrap();
            for card in &hidden.view().hand {
                total += 1;
                revealing += usize::from(card.id % 4 == 1);
            }
            play_by_view(&mut hidden);
            cfg.hide_deal_order = false;
            let mut plain = Session::new(cfg, ai(3)).unwrap();
            play_by_view(&mut plain);
            // Same game, only the numbers differ: same roles every round.
            assert_eq!(
                hidden.view().roles_history,
                plain.view().roles_history,
                "seed {seed}"
            );
        }
        // Dealt round-robin, all 13 of seat 1's cards would be 1 mod 4.
        assert!(
            revealing < total / 2,
            "{revealing} of {total} ids still match the seat"
        );
    }

    #[test]
    fn under_the_final_rule_passes_show_in_the_view_and_a_pass_ends_the_humans_trick() {
        let mut saw_an_ai_pass = false;
        let mut humans_passes = 0;
        for seed in 0..60 {
            let mut cfg = config(4, DeckVariant::Single, 2, 0);
            cfg.seed = seed;
            cfg.pass_rule = PassRule::Final;
            let mut session = Session::new(cfg, ai(3)).unwrap();
            let mut guard = 0;
            while session.phase() != Phase::MatchOver && guard < 2000 {
                guard += 1;
                let view = session.view();
                if view.seats.iter().any(|s| s.passed && !s.is_human) {
                    saw_an_ai_pass = true;
                }
                match view.phase {
                    Phase::Exchange => {
                        let ids: Vec<u8> = view
                            .hand
                            .iter()
                            .take(view.give_count)
                            .map(|c| c.id)
                            .collect();
                        session.give(&ids).unwrap();
                    }
                    Phase::Playing if !view.must_lead => {
                        // Always pass when following: the human is out of the
                        // trick, so the next human decision must come after
                        // a TrickEnd.
                        let before = view.event_count;
                        session.pass().unwrap();
                        humans_passes += 1;
                        let ended = session.events_since(before).iter().any(|e| {
                            matches!(e, GameEvent::TrickEnd { .. } | GameEvent::RoundEnd { .. })
                        });
                        assert!(
                            ended,
                            "seed {seed}: the human acted again inside the trick they passed in"
                        );
                    }
                    Phase::Playing => {
                        let rank = view.playable.first().unwrap();
                        let size = rank.sizes[0];
                        session
                            .play(&rank.card_ids[rank.card_ids.len() - size..])
                            .unwrap();
                    }
                    Phase::RoundOver => session.next_round().unwrap(),
                    Phase::MatchOver => unreachable!(),
                }
            }
            assert_eq!(session.phase(), Phase::MatchOver, "seed {seed}");
        }
        assert!(saw_an_ai_pass, "the view never showed a seat as passed");
        assert!(humans_passes > 0);
    }

    #[test]
    fn nobody_is_shown_as_passed_outside_a_trick_in_progress() {
        // The round-over and match-over screens must not show last trick's passes.
        let mut stale = 0;
        for seed in 0..80 {
            let mut cfg = config(4, DeckVariant::Single, 2, 0);
            cfg.seed = seed;
            cfg.pass_rule = PassRule::Final;
            let mut session = Session::new(cfg, ai(3)).unwrap();
            let mut guard = 0;
            while session.phase() != Phase::MatchOver && guard < 3000 {
                guard += 1;
                let view = session.view();
                if view.phase != Phase::Playing && view.seats.iter().any(|s| s.passed) {
                    stale += 1;
                }
                match view.phase {
                    Phase::Exchange => {
                        let ids: Vec<u8> = view
                            .hand
                            .iter()
                            .take(view.give_count)
                            .map(|c| c.id)
                            .collect();
                        session.give(&ids).unwrap();
                    }
                    Phase::Playing if !view.must_lead => session.pass().unwrap(),
                    Phase::Playing => {
                        let rank = view.playable.first().unwrap();
                        let size = rank.sizes[0];
                        session
                            .play(&rank.card_ids[rank.card_ids.len() - size..])
                            .unwrap();
                    }
                    Phase::RoundOver => session.next_round().unwrap(),
                    Phase::MatchOver => unreachable!(),
                }
            }
            let end = session.view();
            if end.seats.iter().any(|s| s.passed) {
                stale += 1;
            }
        }
        assert_eq!(stale, 0, "views outside play showed passed seats");
    }

    #[test]
    fn under_the_free_rule_no_seat_is_ever_shown_as_passed() {
        let mut cfg = config(4, DeckVariant::Single, 2, 0);
        cfg.pass_rule = PassRule::Free;
        let mut session = Session::new(cfg, ai(3)).unwrap();
        play_by_view(&mut session);
        assert_eq!(session.view().pass_rule, PassRule::Free);
        assert!(session.view().seats.iter().all(|s| !s.passed));
    }

    fn champion() -> NeatStrategy {
        NeatStrategy::from_file(std::path::Path::new(
            "../docs/baselines/neat-v1/champion.json",
        ))
        .unwrap()
    }

    #[test]
    fn advice_ranks_the_humans_moves_and_agrees_with_the_models_own_choice() {
        let model = champion();
        let session = at_first_turn(4, 0, 5);
        let advice = session.advice(&model).unwrap();
        let legal = session.legal_moves();
        assert_eq!(advice.len(), legal.len());
        assert!(advice
            .windows(2)
            .all(|pair| pair[0].raw_score >= pair[1].raw_score));
        // What the model itself would play in this exact situation.
        let round = session.round.as_ref().unwrap();
        let mut tracker = session.tracker.clone();
        let context = turn_context_for(round, 0, 4, &session.round_deck, &mut tracker);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let chosen = model.choose_play(&legal, DuplicateRule::FirstDealtWins, &context, &mut rng);
        let top = &advice[0];
        match chosen {
            Move::Pass => assert!(top.is_pass),
            Move::Play(combo) => {
                let ids: Vec<u8> = combo.cards().iter().map(|c| c.deal_index).collect();
                assert_eq!(top.cards.iter().map(|c| c.id).collect::<Vec<_>>(), ids);
            }
        }
    }

    #[test]
    fn advice_is_only_available_on_the_humans_turn() {
        let model = champion();
        let mut session = Session::new(config(3, DeckVariant::Single, 1, 0), ai(2)).unwrap();
        play_out(&mut session);
        assert_eq!(session.advice(&model).err(), Some(SessionError::WrongPhase));
    }
}
