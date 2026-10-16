//! `EndgameSolver`: exact play once few cards are unseen, a fallback
//! strategy before that. See docs/superpowers/specs/2026-10-16-endgame-solver-design.md.

use std::sync::{Arc, Mutex};

use engine::{Card, DuplicateRule, Move};

use crate::solver::config::SolverSpec;
use crate::solver::{analyze, Observation, SolveError, SolverConfig};
use crate::strategy::{ContextNeeds, Strategy, TurnContext};

/// What the solver did so far (all decisions of this instance).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SolverStats {
    /// Decisions with more than one legal move.
    pub decisions: u64,
    /// Decisions the solver tried to solve (supported position, few enough unseen cards).
    pub eligible: u64,
    pub solved: u64,
    pub fell_back_budget: u64,
    pub fell_back_other: u64,
    /// Search nodes spent in total.
    pub nodes: u64,
}

pub struct EndgameSolver {
    config: SolverConfig,
    fallback: Arc<dyn Strategy>,
    name: String,
    stats: Mutex<SolverStats>,
}

impl EndgameSolver {
    #[must_use]
    pub fn new(spec: SolverSpec) -> Self {
        Self {
            config: spec.solver,
            fallback: spec.fallback.build(),
            name: format!(
                "EndgameSolver({},T={})",
                spec.fallback, spec.solver.max_unseen
            ),
            stats: Mutex::new(SolverStats::default()),
        }
    }

    /// Counters are plain sums, so the order in which threads add to them
    /// does not matter.
    #[must_use]
    pub fn stats(&self) -> SolverStats {
        *self.lock()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, SolverStats> {
        self.stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn count(&self, update: impl FnOnce(&mut SolverStats)) {
        update(&mut self.lock());
    }

    /// The solver's move, or `None` to defer to the fallback.
    fn solve(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        context: &TurnContext<'_>,
    ) -> Option<Move> {
        let observation = Observation::from_context(context, duplicate_rule)?;
        if observation.unseen.len() > self.config.max_unseen {
            return None;
        }
        self.count(|s| s.eligible += 1);
        match analyze(&observation, &self.config) {
            Ok(report) => {
                // The solver searches the engine's own moves, so this only
                // guards against a bug; a mismatch must not reach the engine.
                let best = Some(report.best().mv).filter(|mv| legal_moves.contains(mv));
                self.count(|s| {
                    s.nodes += report.nodes;
                    if best.is_some() {
                        s.solved += 1;
                    } else {
                        s.fell_back_other += 1;
                    }
                });
                best
            }
            Err(SolveError::BudgetExceeded) => {
                self.count(|s| s.fell_back_budget += 1);
                None
            }
            Err(_) => {
                self.count(|s| s.fell_back_other += 1);
                None
            }
        }
    }
}

impl Strategy for EndgameSolver {
    fn name(&self) -> &str {
        &self.name
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        context: &TurnContext<'_>,
        rng: &mut dyn rand::Rng,
    ) -> Move {
        if legal_moves.len() == 1 {
            return legal_moves[0];
        }
        self.count(|s| s.decisions += 1);
        self.solve(legal_moves, duplicate_rule, context)
            .unwrap_or_else(|| {
                self.fallback
                    .choose_play(legal_moves, duplicate_rule, context, rng)
            })
    }

    fn needs(&self) -> ContextNeeds {
        ContextNeeds::ALL
    }

    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        duplicate_rule: DuplicateRule,
        rng: &mut dyn rand::Rng,
    ) -> Vec<Card> {
        self.fallback
            .choose_exchange_cards(hand, count, duplicate_rule, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand_reading::PassTracker;
    use crate::match_runner::turn_context_for;
    use crate::solver::config::SolverSpec;
    use crate::strategy::ContextNeeds;
    use crate::test_support::for_each_state;
    use rand::SeedableRng;

    fn spec(max_unseen: usize) -> SolverSpec {
        SolverSpec {
            solver: crate::solver::SolverConfig {
                max_unseen,
                node_budget: 300_000,
            },
            ..SolverSpec::default()
        }
    }

    #[test]
    fn it_always_returns_a_legal_move_and_counts_what_it_did() {
        let solver = EndgameSolver::new(spec(5));
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        for_each_state(41, 3, |state, _| {
            let round = state.round;
            let seat = round.seat_to_move().unwrap();
            let mut tracker = PassTracker::new(usize::from(state.players), state.rule);
            let ctx = turn_context_for(
                round,
                seat,
                state.players,
                state.round_deck,
                &mut tracker,
                ContextNeeds::ALL,
            );
            let legal = round.legal_moves();
            let mv = solver.choose_play(&legal, state.rule, &ctx, &mut rng);
            assert!(legal.contains(&mv));
        });
        let s = solver.stats();
        assert!(s.solved > 0, "the endgames of the random rounds are solved");
        assert_eq!(
            s.solved + s.fell_back_budget + s.fell_back_other,
            s.eligible
        );
        assert!(s.eligible <= s.decisions);
    }

    #[test]
    fn under_the_free_pass_rule_it_defers_to_the_fallback() {
        let solver = EndgameSolver::new(spec(40));
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        for_each_state(43, 1, |state, _| {
            let round = state.round;
            let seat = round.seat_to_move().unwrap();
            let mut tracker = PassTracker::new(usize::from(state.players), state.rule);
            let mut ctx = turn_context_for(
                round,
                seat,
                state.players,
                state.round_deck,
                &mut tracker,
                ContextNeeds::ALL,
            );
            ctx.table.pass_rule = engine::PassRule::Free;
            let legal = round.legal_moves();
            let mv = solver.choose_play(&legal, state.rule, &ctx, &mut rng);
            assert!(legal.contains(&mv));
        });
        assert_eq!(solver.stats().eligible, 0);
        assert!(solver.stats().decisions > 0);
    }

    #[test]
    fn a_single_legal_move_is_returned_without_searching() {
        let solver = EndgameSolver::new(spec(40));
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let mut seen_single = false;
        for_each_state(47, 2, |state, _| {
            let round = state.round;
            let legal = round.legal_moves();
            if legal.len() != 1 {
                return;
            }
            seen_single = true;
            let seat = round.seat_to_move().unwrap();
            let mut tracker = PassTracker::new(usize::from(state.players), state.rule);
            let ctx = turn_context_for(
                round,
                seat,
                state.players,
                state.round_deck,
                &mut tracker,
                ContextNeeds::ALL,
            );
            assert_eq!(
                solver.choose_play(&legal, state.rule, &ctx, &mut rng),
                legal[0]
            );
        });
        assert!(seen_single);
        assert_eq!(solver.stats().eligible, 0);
    }

    #[test]
    fn the_name_shows_the_configuration() {
        assert_eq!(
            EndgameSolver::new(spec(7)).name(),
            "EndgameSolver(adaptive,T=7)"
        );
    }
}
