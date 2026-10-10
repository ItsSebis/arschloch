//! End-to-end tests of `cli evaluate`: the real binary scoring real genome files.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("arschloch-cli-eval-{name}-{}", std::process::id()))
}

fn genome(path: &Path, seed: u64) {
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

fn evaluate(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_cli"))
        .arg("evaluate")
        .args(args)
        .output()
        .expect("run cli")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn it_scores_genomes_against_opponents_and_writes_json() {
    let base = temp("table");
    let (a, b) = (base.join("alpha.json"), base.join("beta.json"));
    genome(&a, 1);
    genome(&b, 2);
    let json = base.join("out.json");
    let output = evaluate(&[
        "--genome",
        a.to_str().unwrap(),
        "--genome",
        b.to_str().unwrap(),
        "--opponent",
        "lowest-legal",
        "--opponent",
        "random-legal",
        "--matches",
        "20",
        "--rounds",
        "3",
        "--seed",
        "5",
        "--json",
        json.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "stderr: {}", text(&output.stderr));
    let stdout = text(&output.stdout);
    for expected in [
        "alpha",
        "beta",
        "LowestLegal",
        "RandomLegal",
        "mixed (all opponents)",
        "±",
    ] {
        assert!(stdout.contains(expected), "{expected} missing:\n{stdout}");
    }
    let value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
    assert_eq!(
        value["deck_variant"], "Single",
        "the file must say which table it measured"
    );
    assert_eq!(value["duplicate_rule"], "FirstDealtWins");
    let results = value["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    let cells = results[0]["cells"].as_array().unwrap();
    assert_eq!(cells.len(), 3, "two opponents plus the mixed cell");
    assert_eq!(cells[0]["opponent"], "LowestLegal");
    assert_eq!(cells[0]["matches"], 20);
    assert_eq!(
        cells[0]["placements"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap())
            .sum::<u64>(),
        20 * 3
    );
    std::fs::remove_dir_all(&base).unwrap();
}

#[test]
fn the_same_arguments_always_give_the_same_numbers() {
    let base = temp("determinism");
    let a = base.join("g.json");
    genome(&a, 3);
    let run = |threads: &str| {
        text(
            &evaluate(&[
                "--genome",
                a.to_str().unwrap(),
                "--opponent",
                "lowest-legal",
                "--matches",
                "30",
                "--rounds",
                "3",
                "--threads",
                threads,
            ])
            .stdout,
        )
    };
    assert_eq!(
        run("1"),
        run("3"),
        "results do not depend on the thread count"
    );
    std::fs::remove_dir_all(&base).unwrap();
}

#[test]
fn a_genome_can_face_another_genome() {
    let base = temp("versus");
    let (a, b) = (base.join("a.json"), base.join("b.json"));
    genome(&a, 1);
    genome(&b, 2);
    let output = evaluate(&[
        "--genome",
        a.to_str().unwrap(),
        "--opponent",
        &format!("neat:{}", b.display()),
        "--matches",
        "10",
        "--rounds",
        "2",
    ]);
    assert!(output.status.success(), "stderr: {}", text(&output.stderr));
    assert!(text(&output.stdout).contains("Neat(b)"));
    std::fs::remove_dir_all(&base).unwrap();
}

#[test]
fn operator_mistakes_fail_clearly_without_a_panic() {
    let missing = evaluate(&["--genome", "/nonexistent/g.json"]);
    assert!(!missing.status.success());
    assert!(
        text(&missing.stderr).contains("/nonexistent/g.json"),
        "{}",
        text(&missing.stderr)
    );

    let base = temp("mistakes");
    let g = base.join("g.json");
    genome(&g, 1);
    let bad_opponent = evaluate(&["--genome", g.to_str().unwrap(), "--opponent", "nonsense"]);
    assert!(
        text(&bad_opponent.stderr).contains("--opponent `nonsense`"),
        "{}",
        text(&bad_opponent.stderr)
    );
    let duplicate = evaluate(&[
        "--genome",
        g.to_str().unwrap(),
        "--opponent",
        "lowest-legal",
        "--opponent",
        "lowest-legal",
    ]);
    assert!(
        text(&duplicate.stderr).contains("duplicates"),
        "{}",
        text(&duplicate.stderr)
    );
    let none = Command::new(env!("CARGO_BIN_EXE_cli"))
        .arg("evaluate")
        .output()
        .unwrap();
    assert!(!none.status.success());
    for output in [&missing, &bad_opponent, &duplicate, &none] {
        assert!(!text(&output.stderr).contains("panicked"));
    }
    std::fs::remove_dir_all(&base).unwrap();
}

#[test]
fn several_table_sizes_and_decks_get_a_block_each_and_the_single_setting_output_is_unchanged() {
    let base = temp("settings");
    let g = base.join("g.json");
    genome(&g, 3);
    let common = |extra: &[&str], json: &Path| {
        let mut args = vec![
            "--genome",
            g.to_str().unwrap(),
            "--opponent",
            "lowest-legal",
            "--matches",
            "12",
            "--rounds",
            "2",
            "--seed",
            "4",
            "--json",
            json.to_str().unwrap(),
        ];
        args.extend_from_slice(extra);
        evaluate(&args)
    };
    let multi_json = base.join("multi.json");
    let output = common(
        &["--player-counts", "3,5", "--decks", "single,double"],
        &multi_json,
    );
    assert!(output.status.success(), "stderr: {}", text(&output.stderr));
    let stdout = text(&output.stdout);
    for label in ["3p-single", "3p-double", "5p-single", "5p-double"] {
        assert!(
            stdout.contains(&format!("=== setting {label} ===")),
            "{stdout}"
        );
    }
    let value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&multi_json).unwrap()).unwrap();
    let settings = value["settings"].as_array().unwrap();
    assert_eq!(settings.len(), 4);
    assert_eq!(settings[2]["player_count"], 5);
    assert_eq!(settings[1]["deck_variant"], "Double");
    // Placements are as long as the table, and every cell reports the spread.
    let cell = &settings[2]["results"][0]["cells"][0];
    assert_eq!(cell["placements"].as_array().unwrap().len(), 5);
    assert!(cell["spread"].as_f64().unwrap() > 0.0);
    // The top-level `results` stay the first setting's.
    assert_eq!(value["results"], settings[0]["results"]);
    assert_eq!(value["player_count"], 3);

    // One setting: no blocks, no `settings` key, the same table as the plain flags.
    let (one_json, plain_json) = (base.join("one.json"), base.join("plain.json"));
    let one = common(&["--player-counts", "5"], &one_json);
    let plain = common(&["--player-count", "5"], &plain_json);
    assert!(one.status.success() && plain.status.success());
    assert_eq!(one.stdout, plain.stdout);
    assert!(!text(&one.stdout).contains("=== setting"));
    let one_value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&one_json).unwrap()).unwrap();
    assert!(one_value.get("settings").is_none());
    assert_eq!(
        std::fs::read_to_string(&one_json).unwrap(),
        std::fs::read_to_string(&plain_json).unwrap()
    );
    // The lists and the single flags exclude each other; bad lists are refused.
    assert!(!common(
        &["--player-counts", "3,4", "--player-count", "5"],
        &one_json
    )
    .status
    .success());
    assert!(!common(
        &["--decks", "single", "--deck-variant", "double"],
        &one_json
    )
    .status
    .success());
    assert!(!common(&["--player-counts", "3,3"], &one_json)
        .status
        .success());
    assert!(!common(&["--decks", "double,double"], &one_json)
        .status
        .success());
    assert!(!common(&["--player-counts", "2,4"], &one_json)
        .status
        .success());
    std::fs::remove_dir_all(&base).unwrap();
}
