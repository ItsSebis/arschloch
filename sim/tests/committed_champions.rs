//! The committed champions keep loading and playing the same way when the
//! default feature set of new training changes: each records its own
//! feature set in its file, and a fixed set of matches against classic
//! players gives the same fingerprint it gave when the file was committed.

use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

use engine::{DeckVariant, DuplicateRule};
use sim::{run_match, FeatureSet, LowestLegal, MatchConfig, NeatStrategy, Strategy};

fn fnv(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Fingerprint of 30 matches of `champion` against three lowest-legal players.
fn fingerprint(champion: &str) -> (FeatureSet, u64) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../docs/baselines")
        .join(champion)
        .join("champion.json");
    let neat = NeatStrategy::from_file(&path).unwrap();
    let set = neat.feature_set();
    let neat: Arc<dyn Strategy> = Arc::new(neat);
    let strategies: Vec<Arc<dyn Strategy>> = vec![
        neat,
        Arc::new(LowestLegal),
        Arc::new(LowestLegal),
        Arc::new(LowestLegal),
    ];
    let mut history = String::new();
    for seed in 0..30 {
        let config = MatchConfig {
            player_count: 4,
            deck_variant: DeckVariant::Single,
            duplicate_rule: DuplicateRule::FirstDealtWins,
            rounds: 8,
            seed,
            pass_rule: engine::PassRule::Final,
            exchange_rule: engine::ExchangeRule::default(),
        };
        let result = run_match(&config, &strategies);
        write!(
            history,
            "{:?}|{}|{:?};",
            result.role_history, result.trick_count, result.pass_counts
        )
        .unwrap();
    }
    (set, fnv(&history))
}

#[test]
fn the_base_champions_still_load_as_base_and_play_as_they_did() {
    for (name, expected) in [
        ("neat-v1", 16_830_792_035_496_989_792_u64),
        ("neat-v2", 5_143_831_512_394_719_082),
    ] {
        let (set, got) = fingerprint(name);
        assert_eq!(set, FeatureSet::Base, "{name}");
        assert_eq!(got, expected, "{name}");
    }
}
