//! The endgame solver inside a batch: it plays, it is deterministic and the
//! thread count does not change a single result.

use std::sync::Arc;

use engine::{DeckVariant, DuplicateRule, ExchangeRule, PassRule};
use sim::{run_batch, EndgameSolver, LowestLegal, MatchConfig, SolverConfig, SolverSpec, Strategy};

/// Role history and trick count of every match.
type Summary = Vec<(Vec<Vec<engine::Role>>, u32)>;

/// A small solver: the debug build of the test suite must stay fast.
fn small_solver() -> SolverSpec {
    SolverSpec {
        solver: SolverConfig {
            max_unseen: 4,
            node_budget: 100_000,
        },
        ..SolverSpec::default()
    }
}

fn configs(player_count: u8, deck_variant: DeckVariant) -> Vec<MatchConfig> {
    (0..8)
        .map(|seed| MatchConfig {
            player_count,
            deck_variant,
            duplicate_rule: DuplicateRule::FirstDealtWins,
            rounds: 2,
            seed,
            pass_rule: PassRule::Final,
            exchange_rule: ExchangeRule::default(),
        })
        .collect()
}

fn run_on(
    threads: usize,
    player_count: u8,
    deck_variant: DeckVariant,
) -> (Summary, sim::SolverStats) {
    let solver = Arc::new(EndgameSolver::new(small_solver()));
    let mut strategies: Vec<Arc<dyn Strategy>> = vec![solver.clone()];
    strategies.extend((1..player_count).map(|_| Arc::new(LowestLegal) as Arc<dyn Strategy>));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let results = pool.install(|| run_batch(&configs(player_count, deck_variant), &strategies));
    let summary = results
        .iter()
        .map(|r| (r.role_history.clone(), r.trick_count))
        .collect();
    (summary, solver.stats())
}

#[test]
fn a_batch_with_the_solver_is_identical_on_one_and_many_threads() {
    for (players, deck) in [(4, DeckVariant::Single), (5, DeckVariant::Double)] {
        let (one, stats_one) = run_on(1, players, deck);
        let (many, stats_many) = run_on(4, players, deck);
        assert_eq!(one, many, "{players} players, {deck:?}");
        assert_eq!(stats_one, stats_many, "the counters are deterministic too");
        assert!(
            stats_one.solved > 0,
            "the solver engaged in {players}-player matches: {stats_one:?}"
        );
    }
}
