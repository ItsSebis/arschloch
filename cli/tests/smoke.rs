//! End-to-end smoke tests: invoke the real built `cli` binary and check
//! its exit status, stdout, and JSON output file.

use std::process::Command;

#[test]
fn happy_path_run_produces_valid_json_and_summary() {
    let output_path = std::env::temp_dir().join(format!(
        "arschloch-cli-smoke-happy-{}.json",
        std::process::id()
    ));

    let output = Command::new(env!("CARGO_BIN_EXE_cli"))
        .args([
            "--player-count",
            "4",
            "--matches",
            "5",
            "--rounds",
            "2",
            "--strategy",
            "lowest-legal",
            "--strategy",
            "greedy-highest",
            "--strategy",
            "random-legal",
            "--strategy",
            "hold-back-pairs",
            "--seed",
            "1",
            "--output",
        ])
        .arg(&output_path)
        .output()
        .expect("failed to run cli binary");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("LowestLegal"));
    assert!(stdout.contains("GreedyHighest"));
    assert!(stdout.contains("RandomLegal"));
    assert!(stdout.contains("HoldBackPairs"));
    assert!(stdout.contains("Voluntary pass rate"));
    assert!(stdout.contains("President retention"));
    assert!(stdout.contains("First-round placement variance"));

    let contents = std::fs::read_to_string(&output_path).expect("output file should exist");
    let value: serde_json::Value =
        serde_json::from_str(&contents).expect("output should be valid JSON");
    assert!(value.get("matches").is_some());
    let statistics = value.get("statistics").expect("statistics key present");
    assert!(statistics.get("voluntary_pass_rate_by_strategy").is_some());
    assert!(statistics.get("role_retention_by_strategy").is_some());
    assert!(statistics
        .get("first_round_placement_variance_by_strategy")
        .is_some());

    std::fs::remove_file(&output_path).ok();
}

#[test]
fn mismatched_strategy_count_fails() {
    let output = Command::new(env!("CARGO_BIN_EXE_cli"))
        .args([
            "--player-count",
            "4",
            "--matches",
            "1",
            "--rounds",
            "1",
            "--strategy",
            "lowest-legal",
        ])
        .output()
        .expect("failed to run cli binary");

    assert!(!output.status.success());
}

fn write_genome_file(name: &str) -> std::path::PathBuf {
    let population = neat::Population::new(
        sim::FEATURE_COUNT,
        neat::NeatConfig {
            population_size: 4,
            ..neat::NeatConfig::default()
        },
        1,
    )
    .unwrap();
    let path = std::env::temp_dir().join(format!("{name}-{}.json", std::process::id()));
    sim::GenomeFile::new(population.genomes()[0].clone())
        .unwrap()
        .save(&path)
        .unwrap();
    path
}

#[test]
fn a_trained_genome_plays_in_a_normal_run() {
    let genome = write_genome_file("smoke-champion");
    let output_path = std::env::temp_dir().join(format!("smoke-neat-{}.json", std::process::id()));
    let output = Command::new(env!("CARGO_BIN_EXE_cli"))
        .args([
            "--player-count",
            "4",
            "--matches",
            "20",
            "--rounds",
            "3",
            "--seed",
            "5",
        ])
        .args(["--strategy", &format!("neat:{}", genome.display())])
        .args(["--strategy", "lowest-legal", "--strategy", "card-counter"])
        .args(["--strategy", "adaptive:reading,tempo,bully", "--output"])
        .arg(&output_path)
        .output()
        .expect("failed to run cli binary");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stem = genome.file_stem().unwrap().to_string_lossy().into_owned();
    assert!(stdout.contains(&format!("Neat({stem})")), "{stdout}");
    assert!(stdout.contains("LowestLegal") && stdout.contains("CardCounter"));
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&output_path).unwrap()).unwrap();
    assert_eq!(json["matches"].as_array().unwrap().len(), 20);

    std::fs::remove_file(&genome).ok();
    std::fs::remove_file(&output_path).ok();
}

#[test]
fn a_missing_genome_file_fails_with_a_clear_message() {
    let output = Command::new(env!("CARGO_BIN_EXE_cli"))
        .args(["--player-count", "3", "--matches", "1"])
        .args(["--strategy", "neat:/nonexistent/champion.json"])
        .args(["--strategy", "lowest-legal", "--strategy", "lowest-legal"])
        .output()
        .expect("failed to run cli binary");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("/nonexistent/champion.json"), "{stderr}");
}

#[test]
fn a_garbage_genome_file_fails_without_a_panic() {
    let path = std::env::temp_dir().join(format!("smoke-garbage-{}.json", std::process::id()));
    std::fs::write(&path, "{ definitely not a genome").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_cli"))
        .args(["--player-count", "3", "--matches", "1"])
        .args(["--strategy", &format!("neat:{}", path.display())])
        .args(["--strategy", "lowest-legal", "--strategy", "lowest-legal"])
        .output()
        .expect("failed to run cli binary");
    std::fs::remove_file(&path).ok();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("malformed genome file"), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
}
