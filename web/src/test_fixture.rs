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
            pass_rule: sim::PassRule::default(),
            exchange_rule: sim::ExchangeRule::default(),
            rounds_per_match: 3,
            matches_per_genome: 4,
            reeval_matches: 6,
            generations: 3,
            neat: NeatConfig {
                population_size: 12,
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

/// A set directory: `run-01` is the finished fixture run, `run-02` has only
/// its start and first generation; `set.json` says run 2 of 3 is current.
pub fn fixture_set_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("arschloch-web-set-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let source = fixture_run_dir();
    for run in ["run-01", "run-02"] {
        std::fs::create_dir_all(dir.join(run).join("decisions")).unwrap();
        std::fs::copy(source.join("best.json"), dir.join(run).join("best.json")).unwrap();
    }
    let lines = sample_events();
    std::fs::write(dir.join("run-01/events.jsonl"), lines.join("\n") + "\n").unwrap();
    std::fs::write(
        dir.join("run-02/events.jsonl"),
        lines[..2].join("\n") + "\n",
    )
    .unwrap();
    write_set(&dir, 2, &[10.0]);
    dir
}

pub fn write_set(dir: &std::path::Path, current_run: u32, finished_secs: &[f64]) {
    let set = sim::training::SetFile {
        total_runs: 3,
        current_run,
        finished_secs: finished_secs.to_vec(),
        from: None,
    };
    set.write(dir).unwrap();
}
