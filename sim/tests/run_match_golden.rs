//! Golden fingerprint of `run_match` output, captured from the
//! implementation before the `play_out` / `RunOptions` refactor: ordinary
//! runs must stay byte-for-byte identical.

use std::sync::Arc;

use engine::{DeckVariant, DuplicateRule};
use sim::{run_match, GreedyHighest, LowestLegal, MatchConfig, RandomLegal, Strategy};

fn fnv(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn fingerprint(deck_variant: DeckVariant, seed: u64) -> u64 {
    let strategies: Vec<Arc<dyn Strategy>> = vec![
        Arc::new(RandomLegal),
        Arc::new(LowestLegal),
        Arc::new(GreedyHighest),
        Arc::new(RandomLegal),
    ];
    let config = MatchConfig {
        player_count: 4,
        deck_variant,
        duplicate_rule: DuplicateRule::FirstDealtWins,
        rounds: 12,
        seed,
        pass_rule: engine::PassRule::default(),
        exchange_rule: engine::ExchangeRule::default(),
    };
    let result = run_match(&config, &strategies);
    fnv(&format!(
        "{:?}|{}|{:?}|{:?}",
        result.role_history, result.trick_count, result.pass_counts, result.voluntary_pass_counts
    ))
}

#[test]
fn run_match_output_matches_the_pre_refactor_fingerprint() {
    let got = [
        fingerprint(DeckVariant::Single, 7),
        fingerprint(DeckVariant::Single, 12345),
        fingerprint(DeckVariant::Double, 99),
    ];
    assert_eq!(
        got,
        [
            1_188_595_008_322_094_258,
            6_800_289_207_550_504_460,
            8_652_417_788_000_432_118
        ],
        "fingerprints"
    );
}
