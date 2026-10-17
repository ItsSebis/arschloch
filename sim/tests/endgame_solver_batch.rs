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
            ..SolverConfig::default()
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

fn play(seats: &[Arc<dyn Strategy>], pass_rule: PassRule) -> Summary {
    let configs: Vec<MatchConfig> = configs(4, DeckVariant::Single)
        .into_iter()
        .map(|c| MatchConfig { pass_rule, ..c })
        .collect();
    run_batch(&configs, seats)
        .iter()
        .map(|r| (r.role_history.clone(), r.trick_count))
        .collect()
}

fn wrapped(inner: Arc<dyn Strategy>, max_unseen: usize) -> Arc<dyn Strategy> {
    Arc::new(EndgameSolver::wrapping(
        inner,
        SolverConfig {
            max_unseen,
            node_budget: 100_000,
            ..SolverConfig::default()
        },
    ))
}

#[test]
fn a_wrapped_strategy_plays_exactly_like_itself_while_the_solver_does_not_engage() {
    // `RandomLegal` draws from the rng on every decision, forced ones included, and
    // `Adaptive` reads the passes: any difference in what the wrapper hands on shows.
    let bare = || -> Vec<Arc<dyn Strategy>> {
        vec![
            Arc::new(sim::RandomLegal),
            Arc::new(sim::Adaptive::new(
                "reading,counting".parse::<sim::AdaptiveConfig>().unwrap(),
            )),
            Arc::new(LowestLegal),
            Arc::new(sim::RandomLegal),
        ]
    };
    let alone = play(&bare(), PassRule::Final);
    let wrap_all = |max_unseen| -> Vec<Arc<dyn Strategy>> {
        bare().into_iter().map(|s| wrapped(s, max_unseen)).collect()
    };
    assert_eq!(play(&wrap_all(0), PassRule::Final), alone, "threshold 0");
    // The solver supports the pass rule `final` only: under `free` it never engages.
    let free_alone = play(&bare(), PassRule::Free);
    assert_eq!(play(&wrap_all(40), PassRule::Free), free_alone, "free");
}

#[test]
fn a_wrapped_strategy_hands_the_endgame_to_the_solver() {
    let inner: Arc<dyn Strategy> = Arc::new(LowestLegal);
    let solver = Arc::new(EndgameSolver::wrapping(
        inner.clone(),
        SolverConfig {
            max_unseen: 4,
            node_budget: 100_000,
            ..SolverConfig::default()
        },
    ));
    let seats: Vec<Arc<dyn Strategy>> =
        vec![solver.clone(), inner.clone(), inner.clone(), inner.clone()];
    let with = play(&seats, PassRule::Final);
    assert!(solver.stats().solved > 0, "{:?}", solver.stats());
    assert_ne!(
        with,
        play(
            &[inner.clone(), inner.clone(), inner.clone(), inner],
            PassRule::Final
        )
    );
    assert_eq!(solver.name(), "LowestLegal+Solver(T=4,budget=100000)");
}

fn sampler(samples: usize, leaf: sim::LeafSpec) -> sim::SamplerConfig {
    sim::SamplerConfig {
        samples,
        leaf,
        ..sim::SamplerConfig::default()
    }
}

#[test]
fn the_sampler_plays_the_early_game_identically_on_one_and_many_threads() {
    let run = |threads: usize| {
        let solver = Arc::new(EndgameSolver::wrapping_with_sampler(
            Arc::new(LowestLegal),
            SolverConfig {
                max_unseen: 3,
                node_budget: 100_000,
                ..SolverConfig::default()
            },
            Some(sampler(2, sim::LeafSpec::Adaptive)),
        ));
        let mut seats: Vec<Arc<dyn Strategy>> = vec![solver.clone()];
        seats.extend((1..4).map(|_| Arc::new(LowestLegal) as Arc<dyn Strategy>));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let summary = pool.install(|| play(&seats, PassRule::Final));
        (summary, solver.stats())
    };
    let (one, stats_one) = run(1);
    let (many, stats_many) = run(4);
    assert_eq!(one, many);
    assert_eq!(stats_one, stats_many);
    assert!(
        stats_one.sampled > 0 && stats_one.solved > 0,
        "{stats_one:?}"
    );
    assert!(stats_one.rollouts >= 2 * stats_one.sampled, "{stats_one:?}");
}

#[test]
fn the_sampler_never_engages_under_the_free_pass_rule() {
    // It needs the same `final`-rule rebuild as the exact solver: under `free` the
    // wrapped strategy plays every move, from the same random stream as alone.
    let bare: Vec<Arc<dyn Strategy>> = vec![
        Arc::new(sim::RandomLegal),
        Arc::new(LowestLegal),
        Arc::new(sim::RandomLegal),
        Arc::new(LowestLegal),
    ];
    let wrapped: Vec<Arc<dyn Strategy>> = bare
        .iter()
        .map(|s| {
            Arc::new(EndgameSolver::wrapping_with_sampler(
                Arc::clone(s),
                SolverConfig::default(),
                Some(sampler(2, sim::LeafSpec::Cards)),
            )) as Arc<dyn Strategy>
        })
        .collect();
    assert_eq!(play(&wrapped, PassRule::Free), play(&bare, PassRule::Free));
}
