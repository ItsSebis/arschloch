//! The `Strategy` trait: how a simulated seat picks among legal moves.

use engine::{Card, Combo, DuplicateRule, Move, SeatId};

use crate::hand_reading::PassCeilings;

/// A seat other than the one currently acting, and what's publicly
/// known about it: its current hand size and whether it's still in
/// the round (a finished seat's hand size is always `0`, but `active`
/// is spelled out so strategies never have to re-derive it).
#[derive(Debug, Clone, Copy)]
pub struct OpponentHand {
    pub seat: SeatId,
    pub hand_size: usize,
    pub active: bool,
    pub pass_ceilings: PassCeilings,
}

/// Which parts of a `TurnContext` a `Strategy` reads (`Strategy::needs`).
/// The match loops build only what the acting seat's strategy asks for;
/// every other field holds a cheap empty/default value (no opponents, no
/// unseen cards, default pass ceilings) and reading it is a bug. `hand`,
/// `seat` and `current_combo` are always filled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ContextNeeds(u8);

impl ContextNeeds {
    /// Nothing beyond `seat`, `hand` and `current_combo`.
    pub const NONE: Self = Self(0);
    /// `opponents`: seat, `hand_size` and `active` of every other seat
    /// (their `pass_ceilings` stay default unless also requested).
    pub const OPPONENTS: Self = Self(1);
    /// `OpponentHand::pass_ceilings` (implies `OPPONENTS`).
    pub const OPPONENT_PASS_CEILINGS: Self = Self(2 | 1);
    /// `unseen_cards`.
    pub const UNSEEN: Self = Self(4);
    /// `own_pass_ceilings`.
    pub const OWN_PASS_CEILINGS: Self = Self(8);
    /// Everything (the default of `Strategy::needs`).
    pub const ALL: Self = Self(1 | 2 | 4 | 8);

    /// Both sets of needs.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether every need of `other` is also in `self`.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether any pass ceiling (own or opponents') is read.
    #[must_use]
    pub const fn pass_ceilings(self) -> bool {
        self.0 & (2 | 8) != 0
    }
}

/// Everything beyond `legal_moves` a `Strategy` needs for card
/// counting and endgame denial (docs/ROADMAP.md, Phase 6). Built fresh
/// every turn by `match_runner::run_match`, and only the parts the acting
/// seat's strategy declares in `Strategy::needs`: `opponents`,
/// `unseen_cards` and the pass ceilings hold empty/default values when
/// not requested, and reading them then is a bug in the strategy (its
/// `needs` must list everything it reads). `seat`, `hand` and
/// `current_combo` are always filled.
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
    /// This seat's own pass ceilings, as read by anyone else — needed
    /// by `Adaptive`'s deception modifier (a later task) to avoid a
    /// redundant bluff.
    pub own_pass_ceilings: PassCeilings,
    /// The combo currently on the table, or `None` if this seat must
    /// lead (mirrors `engine::Round::current_combo`).
    pub current_combo: Option<&'a Combo>,
}

/// Chooses a move from the moves `engine` reports as legal. Implementors
/// must be `Send + Sync` so a single instance (behind `Arc`) can be
/// shared read-only across many parallel matches; per-match randomness is
/// threaded through via `rng` rather than owned by the strategy, so every
/// match's outcome depends only on its own seed, not on thread
/// scheduling.
pub trait Strategy: Send + Sync {
    /// A short, stable name used to group results by strategy (see
    /// `crate::statistics::aggregate`). Owned content, not necessarily
    /// `'static` — a configurable strategy's name reflects its actual
    /// configuration.
    fn name(&self) -> &str;

    /// Picks one entry from `legal_moves` (never empty when a seat is
    /// actually to move — see `engine::Round::legal_moves`).
    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        context: &TurnContext<'_>,
        rng: &mut dyn rand::Rng,
    ) -> Move;

    /// The `TurnContext` fields `choose_play` reads. The default is
    /// everything, so strategies that do not override it keep working; an
    /// override must list every field the strategy reads, since the
    /// others hold empty/default values.
    fn needs(&self) -> ContextNeeds {
        ContextNeeds::ALL
    }

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

#[cfg(test)]
mod tests {
    use engine::{Card, Combo, DuplicateRule, Move, Rank, Suit};
    use neat::{NeatConfig, Population};
    use rand::{Rng, RngExt, SeedableRng};

    use super::*;
    use crate::hand_reading::{PassCeilings, PassTracker};
    use crate::match_runner::turn_context_for;
    use crate::strategies::{
        Adaptive, AdaptiveConfig, CardCounter, DenialMode, EndgameDenial, GreedyHighest,
        HoldBackPairs, LowestLegal, NeatStrategy, RandomLegal,
    };
    use crate::test_support::for_each_state;

    #[test]
    fn needs_combine_and_nest() {
        let all = ContextNeeds::ALL;
        assert!(all.contains(ContextNeeds::UNSEEN));
        assert!(all.contains(ContextNeeds::OPPONENT_PASS_CEILINGS));
        assert!(ContextNeeds::OPPONENT_PASS_CEILINGS.contains(ContextNeeds::OPPONENTS));
        assert!(!ContextNeeds::OPPONENTS.contains(ContextNeeds::OPPONENT_PASS_CEILINGS));
        assert!(!ContextNeeds::NONE.pass_ceilings());
        assert!(ContextNeeds::OWN_PASS_CEILINGS.pass_ceilings());
        assert!(ContextNeeds::OPPONENT_PASS_CEILINGS.pass_ceilings());
        assert!(!ContextNeeds::OPPONENTS
            .union(ContextNeeds::UNSEEN)
            .pass_ceilings());
    }

    fn evolved_genomes(count: usize) -> Vec<neat::Genome> {
        let config = NeatConfig {
            population_size: 40,
            add_node_rate: 0.3,
            add_connection_rate: 0.3,
            toggle_enable_rate: 0.1,
            ..NeatConfig::default()
        };
        let mut population = Population::new(crate::FEATURE_COUNT, config, 5).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        for _ in 0..15 {
            let fitness = (0..40).map(|_| rng.random_range(0.0..1.0)).collect();
            population.set_fitness(fitness);
            population.advance();
        }
        population.genomes().iter().take(count).cloned().collect()
    }

    fn strategies_under_test() -> Vec<Box<dyn Strategy>> {
        let mut all: Vec<Box<dyn Strategy>> = vec![
            Box::new(LowestLegal),
            Box::new(GreedyHighest),
            Box::new(RandomLegal),
            Box::new(HoldBackPairs),
            Box::new(CardCounter),
            Box::new(EndgameDenial),
        ];
        for (i, genome) in evolved_genomes(4).iter().enumerate() {
            all.push(Box::new(
                NeatStrategy::new(format!("n{i}"), genome).unwrap(),
            ));
        }
        for counting in [false, true] {
            for denial in [
                DenialMode::Off,
                DenialMode::HandSize { close: 2 },
                DenialMode::HandReading { close: 3 },
            ] {
                for deception_rate in [0.0, 0.5, 1.0] {
                    for tempo in [false, true] {
                        for bully in [false, true] {
                            all.push(Box::new(Adaptive::new(AdaptiveConfig {
                                counting,
                                denial,
                                deception_rate,
                                tempo,
                                bully,
                            })));
                        }
                    }
                }
            }
        }
        all.push(Box::new(Adaptive::new(AdaptiveConfig::NONE)));
        all
    }

    /// Pass ceilings that claim "cannot beat anything" at every size.
    fn poison_ceilings() -> PassCeilings {
        let lowest = Combo::new(vec![Card::new(Rank::Two, Suit::Diamonds, 0)]).unwrap();
        crate::hand_reading::read_pass_ceilings(
            1,
            &[],
            &[(0, lowest, 0)],
            DuplicateRule::FirstDealtWins,
        )[0]
    }

    /// The full context with every field `needs` does not declare replaced
    /// by garbage.
    fn poisoned_context<'a>(
        full: &TurnContext<'a>,
        needs: ContextNeeds,
        rng: &mut rand::rngs::StdRng,
    ) -> TurnContext<'a> {
        let mut poisoned = TurnContext {
            seat: full.seat,
            hand: full.hand,
            opponents: full.opponents.clone(),
            unseen_cards: full.unseen_cards.clone(),
            own_pass_ceilings: full.own_pass_ceilings,
            current_combo: full.current_combo,
        };
        if !needs.contains(ContextNeeds::OPPONENTS) {
            poisoned.opponents = (0..rng.random_range(0..6u8))
                .map(|s| OpponentHand {
                    seat: s,
                    hand_size: rng.random_range(0..4),
                    active: rng.random_bool(0.5),
                    pass_ceilings: poison_ceilings(),
                })
                .collect();
        } else if !needs.contains(ContextNeeds::OPPONENT_PASS_CEILINGS) {
            for o in &mut poisoned.opponents {
                o.pass_ceilings = poison_ceilings();
            }
        }
        if !needs.contains(ContextNeeds::UNSEEN) {
            poisoned.unseen_cards = vec![Card::new(
                if rng.random_bool(0.5) {
                    Rank::Two
                } else {
                    Rank::Ace
                },
                Suit::Diamonds,
                200,
            )];
        }
        if !needs.contains(ContextNeeds::OWN_PASS_CEILINGS) {
            poisoned.own_pass_ceilings = poison_ceilings();
        }
        poisoned
    }

    /// Every strategy decides identically (and consumes the same
    /// randomness) from the full context, from the context built for its
    /// declared `needs` only (its own pass tracker skipping turns it was
    /// not asked and catching up later), and from the full context with
    /// every undeclared field replaced by garbage.
    #[test]
    fn strategies_read_only_what_they_declare() {
        let strategies = strategies_under_test();
        let mut trackers: Vec<PassTracker> = Vec::new();
        let mut states = 0usize;
        let mut decisions = 0usize;
        for_each_state(31, 3, |state, rng| {
            let round = state.round;
            // A new round starts with empty histories: restart the trackers.
            if round.play_history().is_empty() && round.pass_history().is_empty() {
                trackers = strategies
                    .iter()
                    .map(|_| PassTracker::new(usize::from(state.players), state.rule))
                    .collect();
            }
            let seat = round.seat_to_move().unwrap();
            let legal = round.legal_moves();
            let mut full_tracker = PassTracker::new(usize::from(state.players), state.rule);
            let full = turn_context_for(
                round,
                seat,
                state.players,
                state.round_deck,
                &mut full_tracker,
                ContextNeeds::ALL,
            );
            states += 1;
            for (strategy, tracker) in strategies.iter().zip(trackers.iter_mut()) {
                // Each strategy is asked on about half of the turns, so its
                // tracker skips and catches up.
                if rng.random_bool(0.5) {
                    continue;
                }
                decisions += 1;
                let needs = strategy.needs();
                let reduced =
                    turn_context_for(round, seat, state.players, state.round_deck, tracker, needs);
                // What is declared must equal the full context.
                if needs.contains(ContextNeeds::OPPONENT_PASS_CEILINGS) {
                    assert_eq!(
                        reduced
                            .opponents
                            .iter()
                            .map(|o| o.pass_ceilings)
                            .collect::<Vec<_>>(),
                        full.opponents
                            .iter()
                            .map(|o| o.pass_ceilings)
                            .collect::<Vec<_>>()
                    );
                }
                if needs.contains(ContextNeeds::OPPONENTS) {
                    assert_eq!(reduced.opponents.len(), full.opponents.len());
                    for (a, b) in reduced.opponents.iter().zip(&full.opponents) {
                        assert_eq!(
                            (a.seat, a.hand_size, a.active),
                            (b.seat, b.hand_size, b.active)
                        );
                    }
                } else {
                    assert_eq!(reduced.opponents.len(), 0);
                }
                if needs.contains(ContextNeeds::UNSEEN) {
                    assert_eq!(reduced.unseen_cards, full.unseen_cards);
                } else {
                    assert_eq!(reduced.unseen_cards.len(), 0);
                }
                if needs.contains(ContextNeeds::OWN_PASS_CEILINGS) {
                    assert_eq!(reduced.own_pass_ceilings, full.own_pass_ceilings);
                }

                let poisoned = poisoned_context(&full, needs, rng);
                let seed = rng.next_u64();
                let mut rngs: Vec<rand::rngs::StdRng> = (0..3)
                    .map(|_| rand::rngs::StdRng::seed_from_u64(seed))
                    .collect();
                let a = strategy.choose_play(&legal, state.rule, &full, &mut rngs[0]);
                let b = strategy.choose_play(&legal, state.rule, &reduced, &mut rngs[1]);
                let c = strategy.choose_play(&legal, state.rule, &poisoned, &mut rngs[2]);
                assert_eq!(a, b, "{} (needs {needs:?})", strategy.name());
                assert_eq!(a, c, "{} (needs {needs:?}), poisoned", strategy.name());
                let next: Vec<u64> = rngs.iter_mut().map(Rng::next_u64).collect();
                assert!(
                    next.iter().all(|&n| n == next[0]),
                    "{} consumed different randomness",
                    strategy.name()
                );
                assert!(matches!(a, Move::Pass | Move::Play(_)));
            }
        });
        assert!(states >= 3000, "only {states} states");
        assert!(decisions >= 50_000, "only {decisions} decisions");
    }
}
