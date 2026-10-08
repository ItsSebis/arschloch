//! Test support: a real (tiny) training run to serve and read.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use neat::NeatConfig;
use sim::training::{DeckChoice, DuplicateChoice, Opponent, TrainConfig, TrainObserver, Trainer};
use sim::{LowestLegal, RandomLegal};

pub fn temp_path(name: &str) -> PathBuf {
    let path =
        std::env::temp_dir().join(format!("arschloch-web-{name}-{}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&path);
    path
}

struct Silent;
impl TrainObserver for Silent {}

/// A finished 3-generation run, created once per test process.
pub fn fixture_run_dir() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir =
            std::env::temp_dir().join(format!("arschloch-web-fixture-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let config = TrainConfig {
            seed: 9,
            player_count: 4,
            deck: DeckChoice::Single,
            duplicate_rule: DuplicateChoice::FirstDealtWins,
            rounds_per_match: 3,
            matches_per_genome: 4,
            reeval_matches: 6,
            generations: 3,
            neat: NeatConfig {
                population_size: 12,
                ..NeatConfig::default()
            },
            opponent_specs: vec!["lowest-legal".into(), "random-legal".into()],
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
    })
    .clone()
}

/// The fixture run's `events.jsonl` lines: start, three generations, end.
pub fn sample_events() -> Vec<String> {
    std::fs::read_to_string(fixture_run_dir().join("events.jsonl"))
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect()
}
