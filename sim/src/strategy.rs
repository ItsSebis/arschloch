//! The `Strategy` trait: how a simulated seat picks among legal moves.

use engine::{Card, DuplicateRule, Move, SeatId};

/// A seat other than the one currently acting, and what's publicly
/// known about it: its current hand size and whether it's still in
/// the round (a finished seat's hand size is always `0`, but `active`
/// is spelled out so strategies never have to re-derive it).
#[derive(Debug, Clone, Copy)]
pub struct OpponentHand {
    pub seat: SeatId,
    pub hand_size: usize,
    pub active: bool,
}

/// Everything beyond `legal_moves` a `Strategy` needs for card
/// counting and endgame denial (docs/ROADMAP.md, Phase 6). Built fresh
/// on the stack every turn by `match_runner::run_match` — cheap at
/// this project's match sizes, so there's no caching/incremental-
/// update machinery here.
pub struct TurnContext<'a> {
    /// The acting seat, for context that needs to know who's asking.
    pub seat: SeatId,
    /// This seat's full remaining hand.
    pub hand: &'a [Card],
    /// Every *other* seat, in seat order.
    pub opponents: Vec<OpponentHand>,
    /// The exact multiset of cards neither in `hand` nor played by
    /// anyone yet this round — i.e. every card some other still-active
    /// seat currently holds. Deterministic: this is a closed-deck game
    /// with no draw pile.
    pub unseen_cards: Vec<Card>,
}

/// Chooses a move from the moves `engine` reports as legal. Implementors
/// must be `Send + Sync` so a single instance (behind `Arc`) can be
/// shared read-only across many parallel matches; per-match randomness is
/// threaded through via `rng` rather than owned by the strategy, so every
/// match's outcome depends only on its own seed, not on thread
/// scheduling.
pub trait Strategy: Send + Sync {
    /// A short, stable name used to group results by strategy (see
    /// `crate::statistics::aggregate`).
    fn name(&self) -> &'static str;

    /// Picks one entry from `legal_moves` (never empty when a seat is
    /// actually to move — see `engine::Round::legal_moves`).
    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        context: &TurnContext<'_>,
        rng: &mut dyn rand::Rng,
    ) -> Move;

    /// Chooses which `count` cards to give up when this seat holds a
    /// role required to hand over its best cards during the exchange
    /// (`docs/ROADMAP.md`, Phase 5, "Smart exchange"). Must return
    /// exactly `count` distinct cards, each present in `hand`;
    /// `engine::exchange_with_selection` treats anything else as a bug
    /// (`ExchangeError::InvalidSelection`).
    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        duplicate_rule: DuplicateRule,
        rng: &mut dyn rand::Rng,
    ) -> Vec<Card>;
}
