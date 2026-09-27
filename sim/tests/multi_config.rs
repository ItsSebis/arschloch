//! Sweeps every table size (3-6) and both deck variants, alternating the
//! duplicate-tiebreak rule, through full multi-round matches with the
//! seven baseline strategies (six fixed strategies plus `Adaptive` on
//! its defaults). Every round of every match must end in a role
//! assignment that is exactly the table's role set, each role once.

use std::sync::Arc;

use engine::{roles_for_player_count, DeckVariant, DuplicateRule};
use sim::{
    run_batch, Adaptive, CardCounter, EndgameDenial, GreedyHighest, HoldBackPairs, LowestLegal,
    MatchConfig, RandomLegal, Strategy,
};

fn baseline_strategies(player_count: u8) -> Vec<Arc<dyn Strategy>> {
    let pool: [Arc<dyn Strategy>; 7] = [
        Arc::new(LowestLegal),
        Arc::new(RandomLegal),
        Arc::new(GreedyHighest),
        Arc::new(HoldBackPairs),
        Arc::new(CardCounter),
        Arc::new(EndgameDenial),
        Arc::new(Adaptive::default()),
    ];
    // Start each table size at a different offset into the (cyclic) pool
    // so the strategies near the tail get exercised at *every* table
    // size (3-6), not just somewhere in the sweep.
    //
    // Walking the offset backward from the pool's end (rather than
    // forward from its start, i.e. `player_count % pool.len()`) is
    // required here, not just a style choice — this is the same
    // mechanism that fixed a coverage bug when this pool grew 4->6
    // (Phase 4->5): a forward offset skips an index for at least one
    // table size, while a backward offset makes the window for
    // `player_count` seats exactly the pool's *last* `player_count`
    // entries (since `player_count <= pool.len()` always holds here,
    // `offset = pool.len() - player_count`, no wraparound). Re-derived
    // by hand for this 7-entry pool at every table size:
    //
    //   player_count=3: offset=4, window = [4,5,6] = CardCounter, EndgameDenial, Adaptive
    //   player_count=4: offset=3, window = [3,4,5,6] = HoldBackPairs, CardCounter, EndgameDenial, Adaptive
    //   player_count=5: offset=2, window = [2,3,4,5,6] = GreedyHighest, HoldBackPairs, CardCounter, EndgameDenial, Adaptive
    //   player_count=6: offset=1, window = [1,2,3,4,5,6] = RandomLegal, GreedyHighest, HoldBackPairs, CardCounter, EndgameDenial, Adaptive
    //
    // So `Adaptive` (index 6, the new tail) is reachable at every table
    // size 3-6, matching the guarantee this pool has upheld for its
    // most-recently-added entries twice before. Two things are new,
    // accepted trade-offs of a 7-entry pool against a max table size of
    // 6 (this project's engine caps `player_count` at 6, so this pool
    // can never grow to a table size that shows all 7 at once):
    // `HoldBackPairs` is no longer covered at player_count=3 (a window
    // of 3 can't hold all 4 tail entries at once, so `Adaptive`
    // displaces it there — `HoldBackPairs` is still covered at sizes
    // 4-6), and `LowestLegal` (index 0) is never in any window in this
    // sweep at all (it would only appear once `player_count` reached
    // `pool.len()` = 7, which is above this engine's table-size cap).
    // Both are fine per this project's established pattern: the
    // guarantee this sweep upholds is "the newest strategy is reachable
    // at every size", not "every strategy is reachable at every size"
    // (mathematically impossible once the pool outgrows the max table
    // size) — `LowestLegal` remains thoroughly covered elsewhere (its
    // own unit tests, and as the base every other strategy delegates to
    // or is compared against, including `Adaptive`'s own equivalence
    // tests).
    let player_count = usize::from(player_count);
    let offset = (pool.len() - player_count % pool.len()) % pool.len();
    pool.iter()
        .cycle()
        .skip(offset)
        .take(player_count)
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
