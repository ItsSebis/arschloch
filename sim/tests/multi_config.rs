//! Sweeps every table size (3-6) and both deck variants, alternating the
//! duplicate-tiebreak rule, through full multi-round matches with the
//! four baseline strategies. Every round of every match must end in a
//! role assignment that is exactly the table's role set, each role once.

use std::sync::Arc;

use engine::{roles_for_player_count, DeckVariant, DuplicateRule};
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
fn every_table_size_and_deck_variant_plays_to_completion_with_valid_roles() {
    let rounds = 3;
    let mut config_index = 0usize;
    for player_count in [3u8, 4, 5, 6] {
        for deck_variant in [DeckVariant::Single, DeckVariant::Double] {
            let duplicate_rule = if config_index.is_multiple_of(2) {
                DuplicateRule::FirstDealtWins
            } else {
                DuplicateRule::LastDealtWins
            };
            config_index += 1;

            let configs: Vec<MatchConfig> = (0..4)
                .map(|seed| MatchConfig {
                    player_count,
                    deck_variant,
                    duplicate_rule,
                    rounds,
                    seed,
                })
                .collect();
            let results = run_batch(&configs, &baseline_strategies(player_count));

            let mut expected_roles = roles_for_player_count(player_count).unwrap().to_vec();
            expected_roles.sort();
            let label = format!("{player_count} players, {deck_variant:?}, {duplicate_rule:?}");
            for result in &results {
                assert_eq!(result.player_count, player_count, "{label}");
                assert_eq!(
                    result.strategy_names.len(),
                    usize::from(player_count),
                    "{label}"
                );
                assert_eq!(result.role_history.len(), rounds, "{label}");
                for round_roles in &result.role_history {
                    let mut roles = round_roles.clone();
                    roles.sort();
                    assert_eq!(roles, expected_roles, "{label}");
                }
                // Every round leads at least one trick, and a voluntary
                // pass is a subset of all passes.
                assert!(result.trick_count >= 3, "{label}");
                for seat in 0..result.pass_counts.len() {
                    assert!(
                        result.voluntary_pass_counts[seat] <= result.pass_counts[seat],
                        "{label}"
                    );
                }
            }
        }
    }
}
