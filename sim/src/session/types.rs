//! The data a session exchanges with a front end: configuration, errors, the
//! card ids, the events it logs and the views it renders.

use std::fmt;

use engine::{Card, Rank, Role, SeatId, Suit};
use serde::Serialize;

use super::{
    Arc, DeckVariant, DuplicateRule, ExchangeRule, PassRule, SeedableRng, SliceRandom, Strategy,
};

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

pub(super) fn rank_label(rank: Rank) -> &'static str {
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

pub(super) fn suit_label(suit: Suit) -> &'static str {
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
pub(super) struct IdMap {
    pub(super) to_client: Vec<u8>,
    pub(super) from_client: Vec<u8>,
}

impl IdMap {
    pub(super) fn identity() -> Self {
        let ids: Vec<u8> = (0..=255).collect();
        Self {
            to_client: ids.clone(),
            from_client: ids,
        }
    }

    pub(super) fn shuffled(seed: u64) -> Self {
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

pub(super) fn views(ids: &IdMap, cards: &[Card]) -> Vec<CardView> {
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
