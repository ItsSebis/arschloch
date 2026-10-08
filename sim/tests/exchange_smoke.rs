//! Exercises `exchange_with_selection` through real strategies across
//! every supported table size, guarding against
//! `ExchangeError::InvalidSelection` ever firing in practice (Phase 5,
//! docs/ROADMAP.md).

use std::sync::Arc;

use engine::{DeckVariant, DuplicateRule};
use sim::{
    run_batch, GreedyHighest, HoldBackPairs, LowestLegal, MatchConfig, RandomLegal, Strategy,
};

fn baseline_strategies(player_count: u8) -> Vec<Arc<dyn Strategy>> {
    let pool: [Arc<dyn Strategy>; 4] = [
        Arc::new(LowestLegal),
        Arc::new(RandomLegal),
        Arc::new(GreedyHighest),
        Arc::new(HoldBackPairs),
    ];
    // Start each table size at a different offset into the (cyclic) pool
    // so every strategy — including HoldBackPairs at index 3 — gets
    // exercised somewhere in the sweep, not just at tables large enough
    // to wrap around from index 0.
    pool.iter()
        .cycle()
        .skip(usize::from(player_count) % pool.len())
        .take(usize::from(player_count))
        .cloned()
        .collect()
}

#[test]
fn every_table_size_runs_many_rounds_without_panicking() {
    for player_count in [3u8, 4, 5, 6] {
        let configs: Vec<MatchConfig> = (0..20u64)
            .map(|seed| MatchConfig {
                player_count,
                deck_variant: DeckVariant::Single,
                duplicate_rule: DuplicateRule::FirstDealtWins,
                rounds: 6,
                seed,
                pass_rule: engine::PassRule::default(),
                exchange_rule: engine::ExchangeRule::default(),
            })
            .collect();
        let results = run_batch(&configs, &baseline_strategies(player_count));
        assert_eq!(results.len(), 20, "player_count {player_count}");
    }
}
