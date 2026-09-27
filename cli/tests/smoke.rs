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
