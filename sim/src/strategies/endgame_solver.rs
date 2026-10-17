//! `EndgameSolver`: exact play once few cards are unseen, a fallback
//! strategy before that (or, with `samples=N`, the phase B sampler). See
//! docs/superpowers/specs/2026-10-16-endgame-solver-design.md and
//! docs/superpowers/specs/2026-10-17-solver-phase-b-design.md.

use std::sync::{Arc, Mutex};

use engine::{Card, DuplicateRule, Move};

use crate::solver::config::SolverSpec;
use crate::solver::leaf::LeafEvaluator;
use crate::solver::{
    analyze, sample_analyze, CeilingMode, Observation, SamplerConfig, SolveError, SolverConfig,
};
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
    /// Decisions the early-game sampler answered.
    pub sampled: u64,
    /// Decisions the sampler could not answer (no consistent deal, `free` rule).
    pub sample_fell_back: u64,
    /// Leaf evaluations (rollouts) the sampler spent in total.
    pub rollouts: u64,
}

pub struct EndgameSolver {
    config: SolverConfig,
    fallback: Arc<dyn Strategy>,
    name: String,
    /// Whether a forced move (one legal move) still goes to the fallback. The
    /// `endgame-solver` strategy returns it directly; a wrapper hands it on, so the
    /// wrapped strategy sees (and draws randomness for) every decision it would
    /// have made alone.
    delegate_forced: bool,
    /// The early-game sampler and its leaf evaluator; `None` = phase A only.
    sampler: Option<(SamplerConfig, Arc<dyn LeafEvaluator>)>,
    stats: Mutex<SolverStats>,
}

/// `T=6`, then `budget=..` and `ceilings=..` when not the defaults and the sampler's
/// settings when on: two configurations never share a name (the default ones keep the
/// names they always had).
fn settings_label(config: &SolverConfig, sampler: Option<&SamplerConfig>) -> String {
    let mut text = format!("T={}", config.max_unseen);
    if config.node_budget != SolverConfig::default().node_budget {
        text.push_str(",budget=");
        text.push_str(&config.node_budget.to_string());
    }
    if config.ceilings != CeilingMode::Hard {
        text.push_str(",ceilings=");
        text.push_str(config.ceilings.as_str());
    }
    if let Some(sampler) = sampler {
        text.push(',');
        text.push_str(&sampler.label());
    }
    text
}

impl EndgameSolver {
    #[must_use]
    pub fn new(spec: SolverSpec) -> Self {
        Self {
            config: spec.solver,
            fallback: spec.fallback.build(),
            name: format!(
                "EndgameSolver({},{})",
                spec.fallback,
                settings_label(&spec.solver, spec.sampler.as_ref())
            ),
            delegate_forced: false,
            sampler: spec.sampler.map(|s| {
                let leaf = s.leaf.evaluator();
                (s, leaf)
            }),
            stats: Mutex::new(SolverStats::default()),
        }
    }

    /// Any strategy with the solver taking over once at most `config.max_unseen`
    /// cards are unseen (`SPEC+solver[:N]`). Until then, and whenever the solver cannot
    /// answer (the `free` pass rule, over budget), `inner` plays, so with
    /// `max_unseen = 0` this plays exactly like `inner`. The exchange is `inner`'s.
    /// The match runner builds the context for the strongest needs, which feeds
    /// `inner`'s pass tracking and hand reading on every turn, whoever then plays.
    #[must_use]
    pub fn wrapping(inner: Arc<dyn Strategy>, config: SolverConfig) -> Self {
        Self::wrapping_with_sampler(inner, config, None)
    }

    /// [`Self::wrapping`] with the early-game sampler deciding, before the exact
    /// endgame, every decision the wrapped strategy would have made (forced moves
    /// still go to the wrapped strategy). `None` is exactly `wrapping`.
    #[must_use]
    pub fn wrapping_with_sampler(
        inner: Arc<dyn Strategy>,
        config: SolverConfig,
        sampler: Option<SamplerConfig>,
    ) -> Self {
        Self {
            config,
            name: format!(
                "{}+Solver({})",
                inner.name(),
                settings_label(&config, sampler.as_ref())
            ),
            fallback: inner,
            delegate_forced: true,
            sampler: sampler.map(|s| {
                let leaf = s.leaf.evaluator();
                (s, leaf)
            }),
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

    /// The sampler's move, or `None` (sampler off, unsupported position, no
    /// consistent deal). Draws one number from `rng` only when it samples.
    fn sample(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        context: &TurnContext<'_>,
        rng: &mut dyn rand::Rng,
    ) -> Option<Move> {
        let (config, leaf) = self.sampler.as_ref()?;
        let Some(observation) = Observation::from_context(context, duplicate_rule) else {
            self.count(|s| s.sample_fell_back += 1);
            return None;
        };
        let seed = rng.next_u64();
        let Ok(report) = sample_analyze(
            &observation,
            config,
            self.config.ceilings,
            leaf.as_ref(),
            seed,
        ) else {
            self.count(|s| s.sample_fell_back += 1);
            return None;
        };
        let best = Some(report.best()).filter(|mv| legal_moves.contains(mv));
        self.count(|s| {
            s.rollouts += report.leaves;
            if best.is_some() {
                s.sampled += 1;
            } else {
                s.sample_fell_back += 1;
            }
        });
        best
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
        if legal_moves.len() == 1 && !self.delegate_forced {
            return legal_moves[0];
        }
        if legal_moves.len() == 1 {
            return self
                .fallback
                .choose_play(legal_moves, duplicate_rule, context, rng);
        }
        self.count(|s| s.decisions += 1);
        self.solve(legal_moves, duplicate_rule, context)
            .or_else(|| self.sample(legal_moves, duplicate_rule, context, rng))
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

    #[test]
    fn configurations_that_differ_have_different_names() {
        let wrap = |config, sampler| {
            EndgameSolver::wrapping_with_sampler(Arc::new(crate::LowestLegal), config, sampler)
                .name()
                .to_owned()
        };
        let base = SolverConfig::default();
        assert_eq!(wrap(base, None), "LowestLegal+Solver(T=6)", "default name");
        let budget = SolverConfig {
            node_budget: 1000,
            ..base
        };
        assert_eq!(wrap(budget, None), "LowestLegal+Solver(T=6,budget=1000)");
        let sampler = SamplerConfig::default();
        let other = SamplerConfig {
            depth: 2,
            ..sampler.clone()
        };
        let names = [
            wrap(base, Some(sampler.clone())),
            wrap(base, Some(other)),
            wrap(budget, Some(sampler)),
        ];
        assert_ne!(names[0], names[1]);
        assert_ne!(names[0], names[2]);
    }
    use crate::solver::config::SolverSpec;
    use crate::strategy::ContextNeeds;
    use crate::test_support::for_each_state;
    use rand::SeedableRng;

    fn spec(max_unseen: usize) -> SolverSpec {
        SolverSpec {
            solver: crate::solver::SolverConfig {
                max_unseen,
                node_budget: 300_000,
                ..crate::solver::SolverConfig::default()
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
            "EndgameSolver(adaptive,T=7,budget=300000)"
        );
        let with_sampler: SolverSpec = "samples=8,depth=2,ceilings=off".parse().unwrap();
        assert_eq!(
            EndgameSolver::new(with_sampler).name(),
            "EndgameSolver(adaptive,T=6,ceilings=off,S=8,D=2,adaptive)"
        );
        let toggle =
            crate::solver::config::parse_solver_toggle(":max-unseen=0,samples=4,leaf=cards")
                .unwrap();
        let wrapped = EndgameSolver::wrapping_with_sampler(
            Arc::new(crate::strategies::LowestLegal),
            toggle.solver,
            toggle.sampler,
        );
        assert_eq!(wrapped.name(), "LowestLegal+Solver(T=0,S=4,D=1,cards)");
    }

    #[test]
    fn the_sampler_answers_the_early_game_and_the_exact_solver_the_end() {
        let mut spec: SolverSpec = "samples=2,leaf=cards".parse().unwrap();
        spec.solver.max_unseen = 4;
        spec.solver.node_budget = 300_000;
        let solver = EndgameSolver::new(spec);
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        for_each_state(53, 2, |state, _| {
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
            assert!(legal.contains(&solver.choose_play(&legal, state.rule, &ctx, &mut rng)));
        });
        let s = solver.stats();
        assert!(s.sampled > 0 && s.solved > 0, "{s:?}");
        // Every decision with a choice under the `final` rule is answered by one of them.
        assert!(
            s.sampled + s.solved + s.sample_fell_back <= s.decisions,
            "{s:?}"
        );
    }
}
