//! Runs the dashboard's JavaScript tests (`node --test`) as part of
//! `cargo test`, when Node is installed (they are skipped, loudly, when it
//! is not). The replay test runs against a real training run, so it proves
//! the browser-side network evaluation agrees with the Rust one.

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use neat::NeatConfig;
use sim::training::{DeckChoice, DuplicateChoice, Opponent, TrainConfig, TrainObserver, Trainer};
use sim::{LowestLegal, RandomLegal};

struct Silent;
impl TrainObserver for Silent {}

fn node_available() -> bool {
    Command::new("node")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn real_run() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("arschloch-web-js-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let config = TrainConfig {
        seed: 21,
        player_count: 4,
        deck: DeckChoice::Single,
        duplicate_rule: DuplicateChoice::FirstDealtWins,
        pass_rule: sim::PassRule::default(),
        exchange_rule: sim::ExchangeRule::default(),
        rounds_per_match: 3,
        matches_per_genome: 6,
        reeval_matches: 8,
        generations: 4,
        neat: NeatConfig {
            population_size: 16,
            add_node_rate: 0.4,
            add_connection_rate: 0.4,
            ..NeatConfig::default()
        },
        opponent_specs: vec!["lowest-legal".into(), "random-legal".into()],
        champion_candidates: 1,
        hall_of_fame_size: 0,
        hall_of_fame_interval: 5,
        skill_weight: 0.0,
    };
    let opponents = vec![
        Opponent {
            name: "LowestLegal".into(),
            strategy: Arc::new(LowestLegal),
        },
        Opponent {
            name: "RandomLegal".into(),
            strategy: Arc::new(RandomLegal),
        },
    ];
    Trainer::new(config, opponents, &dir)
        .unwrap()
        .run(&mut Silent)
        .unwrap();
    dir
}

#[test]
fn the_javascript_tests_pass() {
    if !node_available() {
        eprintln!(
            "SKIPPED: node is not installed, so the dashboard's JavaScript tests did not run"
        );
        return;
    }
    let run = real_run();
    let tests = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let output = Command::new("node")
        .arg("--test")
        .arg(tests.join("lib.test.mjs"))
        .arg(tests.join("replay.test.mjs"))
        .arg(tests.join("play-lib.test.mjs"))
        .env("RUN_DIR", &run)
        .output()
        .expect("node runs");
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "node --test failed:\n{text}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(text.contains("fail 0"), "{text}");
    assert!(
        !text.contains("skipped 1"),
        "the replay test must run against the real run:\n{text}"
    );
    std::fs::remove_dir_all(run).ok();
}
