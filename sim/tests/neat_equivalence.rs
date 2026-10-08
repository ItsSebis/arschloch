//! End-to-end proof that features, scoring and move choice are wired
//! correctly: a network hand-wired to mean "lowest legal" must play
//! *exactly* like `LowestLegal` in complete matches. One wrong sign, a
//! swapped feature or a broken tie-break changes some move in some match
//! and fails this test.

mod common;

use std::sync::Arc;

use engine::{DeckVariant, DuplicateRule};
use sim::{run_match, LowestLegal, MatchConfig, NeatStrategy, Strategy};

fn table(player_count: u8, make: &dyn Fn() -> Arc<dyn Strategy>) -> Vec<Arc<dyn Strategy>> {
    (0..player_count).map(|_| make()).collect()
}

#[test]
fn a_lowest_legal_network_plays_identically_to_lowest_legal() {
    let neat: Arc<dyn Strategy> =
        Arc::new(NeatStrategy::new("Neat(test)", &common::lowest_legal_genome()).unwrap());
    let reference: Arc<dyn Strategy> = Arc::new(LowestLegal);
    for player_count in 3..=6 {
        for duplicate_rule in [DuplicateRule::FirstDealtWins, DuplicateRule::LastDealtWins] {
            for seed in 0..40 {
                let config = MatchConfig {
                    player_count,
                    deck_variant: DeckVariant::Single,
                    duplicate_rule,
                    rounds: 6,
                    seed,
                    pass_rule: engine::PassRule::default(),
                };
                let a = run_match(&config, &table(player_count, &|| neat.clone()));
                let b = run_match(&config, &table(player_count, &|| reference.clone()));
                let context = format!("{player_count} players, {duplicate_rule:?}, seed {seed}");
                assert_eq!(a.role_history, b.role_history, "{context}");
                assert_eq!(a.trick_count, b.trick_count, "{context}");
                assert_eq!(a.pass_counts, b.pass_counts, "{context}");
                assert_eq!(
                    a.voluntary_pass_counts, b.voluntary_pass_counts,
                    "{context}"
                );
            }
        }
    }
}

#[test]
fn flipping_the_strength_weight_changes_play() {
    // Guards the equivalence test above against passing vacuously: a
    // network that prefers the *strongest* card must differ from
    // LowestLegal somewhere.
    let greedy: Arc<dyn Strategy> = Arc::new(
        NeatStrategy::new(
            "Neat(greedy)",
            &common::linear_genome(&[(0, -3.0), (1, -2.0), (2, 0.2)]),
        )
        .unwrap(),
    );
    let reference: Arc<dyn Strategy> = Arc::new(LowestLegal);
    let differs = (0..40).any(|seed| {
        let config = MatchConfig {
            player_count: 4,
            deck_variant: DeckVariant::Single,
            duplicate_rule: DuplicateRule::FirstDealtWins,
            rounds: 6,
            seed,
            pass_rule: engine::PassRule::default(),
        };
        let a = run_match(&config, &table(4, &|| greedy.clone()));
        let b = run_match(&config, &table(4, &|| reference.clone()));
        a.role_history != b.role_history || a.trick_count != b.trick_count
    });
    assert!(differs);
}
