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
            pass_rule: engine::PassRule::default(),
            exchange_rule: engine::ExchangeRule::default(),
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
    assert!(json.contains("voluntary_pass_rate_by_strategy"));
    assert!(json.contains("role_retention_by_strategy"));
    assert!(json.contains("first_round_placement_variance_by_strategy"));
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
        pass_rule: engine::PassRule::default(),
        exchange_rule: engine::ExchangeRule::default(),
    };
    let first = run_match(&config, &strategies);
    let second = run_match(&config, &strategies);
    assert_eq!(first.role_history, second.role_history);
    assert_eq!(first.trick_count, second.trick_count);
    assert_eq!(first.pass_counts, second.pass_counts);
}

/// The two pass rules play different games (a passed seat is out of the
/// trick under `final`), each deterministically.
#[test]
fn the_pass_rules_differ_and_each_is_deterministic() {
    use engine::PassRule;
    let strategies: Vec<std::sync::Arc<dyn sim::Strategy>> = (0..4)
        .map(|_| std::sync::Arc::new(sim::HoldBackPairs) as std::sync::Arc<dyn sim::Strategy>)
        .collect();
    let configs = |rule| -> Vec<sim::MatchConfig> {
        (0..40)
            .map(|seed| sim::MatchConfig {
                player_count: 4,
                deck_variant: engine::DeckVariant::Single,
                duplicate_rule: engine::DuplicateRule::FirstDealtWins,
                rounds: 6,
                seed,
                pass_rule: rule,
                exchange_rule: engine::ExchangeRule::default(),
            })
            .collect()
    };
    let run = |rule| {
        sim::run_batch(&configs(rule), &strategies)
            .iter()
            .map(|r| (r.role_history.clone(), r.trick_count))
            .collect::<Vec<_>>()
    };
    assert_eq!(run(PassRule::Free), run(PassRule::Free));
    assert_eq!(run(PassRule::Final), run(PassRule::Final));
    assert_ne!(run(PassRule::Free), run(PassRule::Final));
}

/// The forced exchange only changes what strategies that give something other
/// than their highest cards would have done: the others play identical matches
/// under both rules.
#[test]
fn the_forced_exchange_differs_only_for_strategies_that_choose_other_cards() {
    use engine::ExchangeRule;
    let run = |strategy: std::sync::Arc<dyn sim::Strategy>, rule: ExchangeRule| {
        let strategies: Vec<std::sync::Arc<dyn sim::Strategy>> =
            (0..4).map(|_| strategy.clone()).collect();
        let configs: Vec<sim::MatchConfig> = (0..60)
            .map(|seed| sim::MatchConfig {
                player_count: 4,
                deck_variant: engine::DeckVariant::Single,
                duplicate_rule: engine::DuplicateRule::FirstDealtWins,
                rounds: 6,
                seed,
                pass_rule: engine::PassRule::Final,
                exchange_rule: rule,
            })
            .collect();
        sim::run_batch(&configs, &strategies)
            .iter()
            .map(|r| r.role_history.clone())
            .collect::<Vec<_>>()
    };
    for same in [
        std::sync::Arc::new(sim::LowestLegal) as std::sync::Arc<dyn sim::Strategy>,
        std::sync::Arc::new(sim::GreedyHighest),
        std::sync::Arc::new(sim::EndgameDenial),
        std::sync::Arc::new(sim::CardCounter),
    ] {
        assert_eq!(
            run(same.clone(), ExchangeRule::Free),
            run(same.clone(), ExchangeRule::Forced),
            "{}",
            same.name()
        );
    }
    let hold = std::sync::Arc::new(sim::HoldBackPairs) as std::sync::Arc<dyn sim::Strategy>;
    assert_ne!(
        run(hold.clone(), ExchangeRule::Free),
        run(hold, ExchangeRule::Forced)
    );
    let random = std::sync::Arc::new(sim::RandomLegal) as std::sync::Arc<dyn sim::Strategy>;
    assert_ne!(
        run(random.clone(), ExchangeRule::Free),
        run(random, ExchangeRule::Forced)
    );
}
