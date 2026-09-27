//! A small batch of matches across all three baseline strategies,
//! asserting the statistical invariants docs/ARCHITECTURE.md calls for:
//! well-shaped role history, sane aggregate counts, and a JSON round-trip.

use std::sync::Arc;

use engine::{DeckVariant, DuplicateRule};
use sim::{
    aggregate, run_batch, run_match, GreedyHighest, LowestLegal, MatchConfig, RandomLegal, Strategy,
};

fn baseline_strategies() -> Vec<Arc<dyn Strategy>> {
    vec![
        Arc::new(LowestLegal),
        Arc::new(RandomLegal),
        Arc::new(GreedyHighest),
        Arc::new(RandomLegal),
    ]
}

#[test]
fn a_small_batch_produces_well_shaped_results() {
    let strategies = baseline_strategies();
    let configs: Vec<MatchConfig> = (0..50)
        .map(|seed| MatchConfig {
            player_count: 4,
            deck_variant: DeckVariant::Single,
            duplicate_rule: DuplicateRule::FirstDealtWins,
            rounds: 5,
            seed,
        })
        .collect();

    let results = run_batch(&configs, &strategies);
    assert_eq!(results.len(), 50);

    for result in &results {
        assert_eq!(result.role_history.len(), 5);
        for round_roles in &result.role_history {
            assert_eq!(round_roles.len(), 4);
        }
        assert_eq!(result.strategy_names.len(), 4);
    }

    let stats = aggregate(&results);
    assert_eq!(stats.matches_played, 50);
    let total_role_assignments: u32 = stats
        .role_counts_by_strategy
        .values()
        .flat_map(|counts| counts.values())
        .sum();
    assert_eq!(total_role_assignments, 50 * 5 * 4);
    assert!((0.0..=1.0).contains(&stats.voluntary_pass_rate));

    let json = serde_json::to_string(&stats).expect("Statistics serializes to JSON");
    assert!(json.contains("matches_played"));
}

#[test]
fn identical_seeds_produce_identical_results() {
    let strategies = baseline_strategies();
    let config = MatchConfig {
        player_count: 4,
        deck_variant: DeckVariant::Single,
        duplicate_rule: DuplicateRule::FirstDealtWins,
        rounds: 3,
        seed: 42,
    };
    let first = run_match(&config, &strategies);
    let second = run_match(&config, &strategies);
    assert_eq!(first.role_history, second.role_history);
    assert_eq!(first.trick_count, second.trick_count);
    assert_eq!(first.pass_counts, second.pass_counts);
}
