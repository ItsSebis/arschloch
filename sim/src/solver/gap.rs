//! Where do other strategies disagree with the exact solver, and what does
//! it cost? One round per game, every seat played by the strategies under
//! test; at every decision the solver can handle, the chosen move is compared
//! with the solver's best.

use std::sync::Arc;

use engine::{
    deal, lowest_card_holder, standard_deck, DeckVariant, DuplicateRule, Move, Round, SeatId,
};
use rand::seq::SliceRandom;
use rand::SeedableRng;

use super::notation::{cards_text, move_text};
use super::{analyze, MoveReport, Observation, SolveError, SolverConfig};
use crate::hand_reading::PassTracker;
use crate::match_runner::turn_context_for;
use crate::strategy::{ContextNeeds, Strategy};

#[derive(Debug, Clone, Copy)]
pub struct GapConfig {
    pub players: u8,
    pub deck: DeckVariant,
    pub rule: DuplicateRule,
    pub games: usize,
    pub seed: u64,
    pub solver: SolverConfig,
}

/// One costly decision, for reading.
#[derive(Debug, Clone, PartialEq)]
pub struct Disagreement {
    /// Expected places lost against the solver's move (> 0).
    pub cost: f64,
    pub situation: String,
}

/// How one strategy fared against the solver on the decisions it could solve.
#[derive(Debug, Clone, PartialEq)]
pub struct GapRow {
    pub strategy: String,
    /// Decisions the solver solved (more than one legal move).
    pub decisions: u64,
    /// Decisions where the strategy's move is worse in expectation than the solver's.
    pub differing: u64,
    /// Sum of the expected places lost over all decisions.
    pub cost_sum: f64,
    /// The five costliest decisions, costliest first.
    pub worst: Vec<Disagreement>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GapStats {
    /// Decisions with more than one legal move.
    pub decisions: u64,
    pub solvable: u64,
    pub budget_exceeded: u64,
    pub nodes_total: u64,
    pub nodes_max: u64,
}

const KEEP_WORST: usize = 5;
const COST_EPSILON: f64 = 1e-9;

fn situation(
    observation: &Observation,
    strategy: &str,
    chosen: Move,
    best: Move,
    cost: f64,
) -> String {
    let table = match observation.on_table {
        None => "leading".to_owned(),
        Some((holder, combo)) => format!("on {} by seat {holder}", cards_text(combo.cards())),
    };
    format!(
        "seat {}, hand [{}], {table}, {} unseen: {strategy} plays {}, solver prefers {} (costs {cost:.2} places)",
        observation.seat,
        cards_text(&observation.hand),
        observation.unseen.len(),
        move_text(chosen),
        move_text(best),
    )
}

/// Plays `config.games` rounds with `strategies` at the seats (rotated by
/// one seat per game) and compares every solvable decision with the solver.
///
/// # Panics
///
/// Panics if `strategies` does not hold one strategy per seat, or if a
/// strategy chooses an illegal move.
#[must_use]
pub fn run_gap(strategies: &[Arc<dyn Strategy>], config: &GapConfig) -> (Vec<GapRow>, GapStats) {
    let n = usize::from(config.players);
    assert_eq!(strategies.len(), n, "one strategy per seat");
    let mut rows: Vec<GapRow> = strategies
        .iter()
        .map(|s| GapRow {
            strategy: s.name().to_owned(),
            decisions: 0,
            differing: 0,
            cost_sum: 0.0,
            worst: Vec::new(),
        })
        .collect();
    let mut stats = GapStats::default();

    for game in 0..config.games {
        let game_seed = config.seed.wrapping_add(game as u64);
        let mut deal_rng = rand::rngs::StdRng::seed_from_u64(game_seed);
        let mut play_rng = rand::rngs::StdRng::seed_from_u64(game_seed ^ 0xA5A5_5A5A_A5A5_5A5A);
        let mut deck = standard_deck(config.deck);
        deck.shuffle(&mut deal_rng);
        for (index, card) in deck.iter_mut().enumerate() {
            card.deal_index = u8::try_from(index).expect("decks have at most 104 cards");
        }
        let hands = deal(deck, config.players).expect("3-6 players");
        let round_deck: Vec<_> = hands.iter().flatten().copied().collect();
        let leader =
            lowest_card_holder(&hands, config.rule).expect("a dealt hand has a lowest card");
        let mut round = Round::new(hands, config.rule, leader).expect("a dealt table is valid");
        let mut tracker = PassTracker::new(n, config.rule);

        while !round.is_complete() {
            let seat: SeatId = round.seat_to_move().expect("round is not complete");
            let index = (usize::from(seat) + game) % n;
            let strategy = &strategies[index];
            let legal = round.legal_moves();
            let context = turn_context_for(
                &round,
                seat,
                config.players,
                &round_deck,
                &mut tracker,
                ContextNeeds::ALL,
            );
            let chosen = strategy.choose_play(&legal, config.rule, &context, &mut play_rng);
            if legal.len() > 1 {
                stats.decisions += 1;
                if let Some(observation) = Observation::from_context(&context, config.rule) {
                    match analyze(&observation, &config.solver) {
                        Ok(report) => {
                            stats.solvable += 1;
                            stats.nodes_total += report.nodes;
                            stats.nodes_max = stats.nodes_max.max(report.nodes);
                            record(&mut rows[index], &report, &observation, chosen);
                        }
                        Err(SolveError::BudgetExceeded) => stats.budget_exceeded += 1,
                        Err(_) => {}
                    }
                }
            }
            round
                .submit_move(seat, chosen)
                .expect("strategies choose from the legal moves");
        }
    }
    (rows, stats)
}

fn record(row: &mut GapRow, report: &MoveReport, observation: &Observation, chosen: Move) {
    let Some(picked) = report.moves.iter().find(|m| m.mv == chosen) else {
        return;
    };
    let best = report.best();
    let cost = (picked.mean_place - best.mean_place).max(0.0);
    row.decisions += 1;
    row.cost_sum += cost;
    if cost > COST_EPSILON {
        row.differing += 1;
        row.worst.push(Disagreement {
            cost,
            situation: situation(observation, &row.strategy, chosen, best.mv, cost),
        });
        // Stable sort: on equal costs the earlier decision stays first.
        row.worst.sort_by(|a, b| b.cost.total_cmp(&a.cost));
        row.worst.truncate(KEEP_WORST);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::config::SolverSpec;
    use crate::solver::SolverConfig;
    use crate::strategies::{EndgameSolver, GreedyHighest, LowestLegal};
    use engine::{DeckVariant, DuplicateRule};
    use std::sync::Arc;

    fn config(games: usize) -> GapConfig {
        GapConfig {
            players: 4,
            deck: DeckVariant::Single,
            rule: DuplicateRule::FirstDealtWins,
            games,
            seed: 5,
            solver: SolverConfig {
                max_unseen: 4,
                node_budget: 100_000,
            },
        }
    }

    #[test]
    fn the_benchmark_is_deterministic_and_costs_are_never_negative() {
        let table: Vec<Arc<dyn Strategy>> = vec![
            Arc::new(LowestLegal),
            Arc::new(GreedyHighest),
            Arc::new(LowestLegal),
            Arc::new(GreedyHighest),
        ];
        let (rows, stats) = run_gap(&table, &config(20));
        let (rows_again, stats_again) = run_gap(&table, &config(20));
        assert_eq!(format!("{rows:?}"), format!("{rows_again:?}"));
        assert_eq!(stats, stats_again);
        assert!(stats.solvable > 0, "{stats:?}");
        assert_eq!(rows.len(), 4);
        for row in &rows {
            assert!(row.cost_sum >= -1e-9);
            assert!(row.differing <= row.decisions);
            assert!(row.worst.len() <= 5);
            assert!(row.worst.windows(2).all(|w| w[0].cost >= w[1].cost));
        }
    }

    #[test]
    fn some_heuristic_decisions_differ_from_the_solver() {
        let table: Vec<Arc<dyn Strategy>> = vec![
            Arc::new(GreedyHighest),
            Arc::new(GreedyHighest),
            Arc::new(GreedyHighest),
            Arc::new(GreedyHighest),
        ];
        let (rows, _) = run_gap(&table, &config(60));
        let differing: u64 = rows.iter().map(|r| r.differing).sum();
        assert!(
            differing > 0,
            "greedy-highest never plays the exact best move?"
        );
        assert!(rows.iter().any(|r| !r.worst.is_empty()));
        assert!(rows[0].worst.iter().all(|d| d.cost > 0.0));
        assert!(rows
            .iter()
            .flat_map(|r| &r.worst)
            .all(|d| d.situation.contains("solver prefers")));
    }

    #[test]
    fn a_strategy_that_is_the_solver_itself_never_disagrees() {
        let spec = SolverSpec {
            solver: config(1).solver,
            ..SolverSpec::default()
        };
        let table: Vec<Arc<dyn Strategy>> = (0..4)
            .map(|_| Arc::new(EndgameSolver::new(spec)) as Arc<dyn Strategy>)
            .collect();
        let (rows, stats) = run_gap(&table, &config(20));
        assert!(stats.solvable > 0);
        assert!(rows.iter().all(|r| r.differing == 0), "{rows:?}");
    }
}
