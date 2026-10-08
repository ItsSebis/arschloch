//! End-to-end tests of `cli train`: the real binary, real files.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn run_dir(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("arschloch-cli-train-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_cli"))
        .args(args)
        .output()
        .expect("failed to run cli binary")
}

fn train(out: &Path, extra: &[&str]) -> Output {
    let mut args = vec![
        "train",
        "--out",
        out.to_str().unwrap(),
        "--population",
        "12",
        "--matches-per-genome",
        "4",
        "--reeval-matches",
        "6",
        "--rounds",
        "3",
        "--threads",
        "2",
    ];
    args.extend_from_slice(extra);
    cli(&args)
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// The generation numbers of the per-generation rows in `stdout`.
fn row_generations(stdout: &str) -> Vec<u32> {
    stdout
        .lines()
        .filter(|line| line.contains('±'))
        .filter_map(|line| line.split_whitespace().next()?.parse().ok())
        .collect()
}

#[test]
fn a_run_prints_a_row_per_generation_and_leaves_playable_champions() {
    let out = run_dir("run");
    let result = train(
        &out,
        &[
            "--generations",
            "3",
            "--opponent",
            "lowest-legal",
            "--opponent",
            "random-legal",
        ],
    );
    assert!(result.status.success(), "stderr: {}", text(&result.stderr));
    let stdout = text(&result.stdout);
    assert!(
        stdout.contains("o1=LowestLegal  o2=RandomLegal"),
        "{stdout}"
    );
    assert_eq!(row_generations(&stdout), vec![0, 1, 2], "{stdout}");
    assert!(stdout.contains("done: 3 generations"), "{stdout}");
    assert!(stdout.contains("best champion: generation"), "{stdout}");

    for file in [
        "config.json",
        "events.jsonl",
        "checkpoint.json",
        "best.json",
        "gen-0000.json",
        "gen-0002.json",
    ] {
        assert!(out.join(file).exists(), "{file} missing");
    }
    let events = std::fs::read_to_string(out.join("events.jsonl")).unwrap();
    assert_eq!(events.lines().count(), 5, "start, 3 generations, end");
    for line in events.lines() {
        serde_json::from_str::<serde_json::Value>(line).expect("every event line is JSON");
    }

    // The champion plays in an ordinary simulation run.
    let best = out.join("best.json");
    let played = cli(&[
        "--player-count",
        "4",
        "--matches",
        "10",
        "--rounds",
        "2",
        "--output",
        "/dev/null",
        "--strategy",
        &format!("neat:{}", best.display()),
        "--strategy",
        "lowest-legal",
        "--strategy",
        "lowest-legal",
        "--strategy",
        "lowest-legal",
    ]);
    assert!(played.status.success(), "stderr: {}", text(&played.stderr));
    assert!(text(&played.stdout).contains("Neat(best)"));
    std::fs::remove_dir_all(&out).unwrap();
}

#[test]
fn resume_continues_from_the_checkpoint_and_prints_only_new_generations() {
    let out = run_dir("resume");
    assert!(train(&out, &["--generations", "2"]).status.success());
    let result = cli(&[
        "train",
        "--out",
        out.to_str().unwrap(),
        "--resume",
        "--generations",
        "4",
        "--threads",
        "2",
    ]);
    assert!(result.status.success(), "stderr: {}", text(&result.stderr));
    let stdout = text(&result.stdout);
    assert!(stdout.contains("resumed from generation 2"), "{stdout}");
    assert_eq!(row_generations(&stdout), vec![2, 3], "{stdout}");
    assert!(stdout.contains("done: 4 generations"), "{stdout}");
    let events = std::fs::read_to_string(out.join("events.jsonl")).unwrap();
    let generations = events
        .lines()
        .filter(|l| l.contains(r#""type":"generation""#))
        .count();
    assert_eq!(generations, 4, "each generation is logged exactly once");
    std::fs::remove_dir_all(&out).unwrap();
}

#[test]
fn a_new_run_never_overwrites_an_existing_one() {
    let out = run_dir("overwrite");
    assert!(train(&out, &["--generations", "1"]).status.success());
    let again = train(&out, &["--generations", "1"]);
    assert!(!again.status.success());
    assert!(
        text(&again.stderr).contains("already holds a run"),
        "{}",
        text(&again.stderr)
    );
    std::fs::remove_dir_all(&out).unwrap();
}

#[test]
fn bad_input_fails_with_a_clear_message_and_no_panic() {
    let out = run_dir("bad");
    let bad_opponent = train(&out, &["--opponent", "nonsense"]);
    assert!(!bad_opponent.status.success());
    assert!(
        text(&bad_opponent.stderr).contains("--opponent `nonsense`"),
        "{}",
        text(&bad_opponent.stderr)
    );

    let duplicate = train(
        &out,
        &["--opponent", "lowest-legal", "--opponent", "lowest-legal"],
    );
    assert!(
        text(&duplicate.stderr).contains("duplicates `LowestLegal`"),
        "{}",
        text(&duplicate.stderr)
    );

    let conflict = cli(&[
        "train",
        "--out",
        out.to_str().unwrap(),
        "--resume",
        "--population",
        "9",
    ]);
    assert!(!conflict.status.success());
    assert!(
        text(&conflict.stderr).contains("cannot be used with"),
        "{}",
        text(&conflict.stderr)
    );

    let nothing = cli(&["train", "--out", out.to_str().unwrap(), "--resume"]);
    assert!(!nothing.status.success());
    assert!(
        text(&nothing.stderr).contains("no checkpoint.json"),
        "{}",
        text(&nothing.stderr)
    );
    for output in [&bad_opponent, &duplicate, &conflict, &nothing] {
        assert!(!text(&output.stderr).contains("panicked"));
    }
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn quiet_prints_only_the_summary() {
    let out = run_dir("quiet");
    let result = train(&out, &["--generations", "2", "--quiet"]);
    assert!(result.status.success());
    let stdout = text(&result.stdout);
    assert!(row_generations(&stdout).is_empty(), "{stdout}");
    assert!(!stdout.contains("opponents:"), "{stdout}");
    assert!(stdout.contains("done: 2 generations"), "{stdout}");
    assert_eq!(
        std::fs::read_to_string(out.join("events.jsonl"))
            .unwrap()
            .lines()
            .count(),
        4
    );
    std::fs::remove_dir_all(&out).unwrap();
}

fn genome_file(path: &Path, seed: u64) {
    let population = neat::Population::new(
        sim::FEATURE_COUNT,
        neat::NeatConfig {
            population_size: 4,
            ..neat::NeatConfig::default()
        },
        seed,
    )
    .unwrap();
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    sim::GenomeFile::new(population.genomes()[0].clone())
        .unwrap()
        .save(path)
        .unwrap();
}

#[test]
fn a_neat_opponent_is_frozen_into_the_run_so_resume_cannot_silently_change_the_pool() {
    let out = run_dir("frozen");
    let source = run_dir("frozen-src").join("champ.json");
    genome_file(&source, 1);
    let original = std::fs::read_to_string(&source).unwrap();
    let opponent = format!("neat:{}", source.display());
    let first = train(
        &out,
        &[
            "--generations",
            "1",
            "--opponent",
            &opponent,
            "--opponent",
            "lowest-legal",
        ],
    );
    assert!(first.status.success(), "stderr: {}", text(&first.stderr));

    // The run keeps its own copy of the opponent.
    assert_eq!(
        std::fs::read_to_string(out.join("opponents/0-champ.json")).unwrap(),
        original
    );
    // Someone overwrites (or deletes) the original; the run is unaffected.
    genome_file(&source, 2);
    let resumed = cli(&[
        "train",
        "--out",
        out.to_str().unwrap(),
        "--resume",
        "--generations",
        "2",
        "--threads",
        "2",
    ]);
    assert!(
        resumed.status.success(),
        "stderr: {}",
        text(&resumed.stderr)
    );
    assert_eq!(
        std::fs::read_to_string(out.join("opponents/0-champ.json")).unwrap(),
        original
    );
    assert!(
        text(&resumed.stdout).contains("Neat(0-champ)"),
        "{}",
        text(&resumed.stdout)
    );
    std::fs::remove_dir_all(&out).unwrap();
    std::fs::remove_dir_all(source.parent().unwrap()).unwrap();
}

#[test]
fn two_neat_opponents_with_the_same_file_name_stay_distinct() {
    let out = run_dir("samestem");
    let base = run_dir("samestem-src");
    let (a, b) = (base.join("a/champ.json"), base.join("b/champ.json"));
    genome_file(&a, 1);
    genome_file(&b, 2);
    let result = train(
        &out,
        &[
            "--generations",
            "1",
            "--opponent",
            &format!("neat:{}", a.display()),
            "--opponent",
            &format!("neat:{}", b.display()),
        ],
    );
    assert!(result.status.success(), "stderr: {}", text(&result.stderr));
    let stdout = text(&result.stdout);
    assert!(
        stdout.contains("Neat(0-champ)") && stdout.contains("Neat(1-champ)"),
        "{stdout}"
    );
    std::fs::remove_dir_all(&out).unwrap();
    std::fs::remove_dir_all(&base).unwrap();
}
