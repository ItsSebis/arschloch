//! Robustness: genomes with arbitrary evolved topology must always play
//! legal moves, at every table size, deck and duplicate rule, alongside
//! other strategies, in parallel, and deterministically. `run_match`
//! panics if a strategy ever picks a move `engine` did not offer, so
//! simply completing the matches is the assertion.

use std::sync::Arc;

use engine::{DeckVariant, DuplicateRule};
use neat::{NeatConfig, Network, Population};
use rand::{RngExt, SeedableRng};
use sim::{
    run_batch, CardCounter, LowestLegal, MatchConfig, NeatStrategy, RandomLegal, Strategy,
    FEATURE_COUNT,
};

/// Genomes with real hidden structure: evolve a population for a while
/// against random fitness, so topologies diverge, then take a spread.
fn diverse_genomes(count: usize) -> Vec<neat::Genome> {
    let config = NeatConfig {
        population_size: 60,
        add_node_rate: 0.3,
        add_connection_rate: 0.3,
        toggle_enable_rate: 0.1,
        ..NeatConfig::default()
    };
    let mut population = Population::new(FEATURE_COUNT, config, 5).unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(9);
    for _ in 0..25 {
        let fitness = (0..60).map(|_| rng.random_range(0.0..1.0)).collect();
        population.set_fitness(fitness);
        population.advance();
    }
    population.genomes().iter().take(count).cloned().collect()
}

fn configs(player_count: u8, deck_variant: DeckVariant, rule: DuplicateRule) -> Vec<MatchConfig> {
    (0..12)
        .map(|seed| MatchConfig {
            player_count,
            deck_variant,
            duplicate_rule: rule,
            rounds: 4,
            seed,
            pass_rule: engine::PassRule::default(),
            exchange_rule: engine::ExchangeRule::default(),
        })
        .collect()
}

fn seats(player_count: u8, genomes: &[neat::Genome]) -> Vec<Arc<dyn Strategy>> {
    (0..usize::from(player_count))
        .map(|seat| -> Arc<dyn Strategy> {
            match seat % 4 {
                0 | 1 => Arc::new(
                    NeatStrategy::new(format!("Neat({seat})"), &genomes[seat % genomes.len()])
                        .unwrap(),
                ),
                2 => Arc::new(LowestLegal),
                _ => Arc::new(CardCounter),
            }
        })
        .collect()
}

#[test]
fn evolved_networks_always_play_legal_moves_everywhere() {
    let genomes = diverse_genomes(8);
    assert!(
        genomes.iter().any(|g| g.hidden_count() > 0),
        "the scenario needs real hidden structure"
    );
    for player_count in 3..=6 {
        for deck in [DeckVariant::Single, DeckVariant::Double] {
            for rule in [DuplicateRule::FirstDealtWins, DuplicateRule::LastDealtWins] {
                let results = run_batch(
                    &configs(player_count, deck, rule),
                    &seats(player_count, &genomes),
                );
                assert_eq!(results.len(), 12);
                for result in &results {
                    assert_eq!(result.role_history.len(), 4);
                }
            }
        }
    }
}

#[test]
fn batches_with_networks_are_deterministic_and_thread_safe() {
    let genomes = diverse_genomes(8);
    let run = || {
        let results = run_batch(
            &configs(5, DeckVariant::Single, DuplicateRule::FirstDealtWins),
            &seats(5, &genomes),
        );
        results
            .iter()
            .map(|r| (r.role_history.clone(), r.trick_count, r.pass_counts.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(run(), run());
}

#[test]
fn random_networks_do_not_just_play_like_random_legal() {
    // A sanity check that the scenario exercises the network: the
    // evolved genomes' choices must differ from a uniformly random
    // player's. (Same table, same seeds: only the strategies differ.)
    let genomes = diverse_genomes(4);
    let neat_table: Vec<Arc<dyn Strategy>> = (0..4)
        .map(|i| -> Arc<dyn Strategy> { Arc::new(NeatStrategy::new("n", &genomes[i]).unwrap()) })
        .collect();
    let random_table: Vec<Arc<dyn Strategy>> = (0..4)
        .map(|_| -> Arc<dyn Strategy> { Arc::new(RandomLegal) })
        .collect();
    let config = MatchConfig {
        player_count: 4,
        deck_variant: DeckVariant::Single,
        duplicate_rule: DuplicateRule::FirstDealtWins,
        rounds: 6,
        seed: 3,
        pass_rule: engine::PassRule::default(),
        exchange_rule: engine::ExchangeRule::default(),
    };
    let a = sim::run_match(&config, &neat_table);
    let b = sim::run_match(&config, &random_table);
    assert!(a.role_history != b.role_history || a.pass_counts != b.pass_counts);
    // And the network itself is real: compiled output is bounded.
    let network = Network::compile(&genomes[0]);
    let out = network.activate(&[0.5; FEATURE_COUNT], &mut Vec::new());
    assert!(out.abs() <= 1.0);
}
