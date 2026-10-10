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
    let results = out.join("played.json"); // not /dev/null: this must also run on Windows
    let played = cli(&[
        "--player-count",
        "4",
        "--matches",
        "10",
        "--rounds",
        "2",
        "--output",
        results.to_str().unwrap(),
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

#[test]
fn the_learning_options_show_up_in_the_output_and_the_events() {
    let out = run_dir("learning");
    let result = train(
        &out,
        &[
            "--generations",
            "3",
            "--champion-candidates",
            "3",
            "--hall-of-fame",
            "2",
            "--hall-interval",
            "1",
            "--weight-power",
            "0.2",
        ],
    );
    assert!(result.status.success(), "stderr: {}", text(&result.stderr));
    let stdout = text(&result.stdout);
    assert!(
        stdout.contains("hof"),
        "the hall column appears when the hall is on:\n{stdout}"
    );
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["champion_candidates"], 3);
    assert_eq!(config["hall_of_fame_size"], 2);
    assert!((config["neat"]["weight_perturb_power"].as_f64().unwrap() - 0.2).abs() < 1e-12);
    let events = std::fs::read_to_string(out.join("events.jsonl")).unwrap();
    assert!(
        events.contains(r#""hall_of_fame":[1]"#) || events.contains(r#""hall_of_fame":[1,2]"#),
        "{events}"
    );
    std::fs::remove_dir_all(&out).unwrap();
}

#[test]
fn an_absurd_weight_power_is_refused_before_anything_is_written() {
    let out = run_dir("badpower");
    let result = train(&out, &["--generations", "1", "--weight-power", "-1"]);
    assert!(!result.status.success());
    assert!(!out.join("checkpoint.json").exists());
    assert!(!text(&result.stderr).contains("panicked"));
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn a_bad_option_leaves_nothing_behind_even_with_a_neat_opponent() {
    // The opponent file is frozen into the run directory; that must not
    // happen for a run whose settings turn out to be invalid.
    let base = run_dir("nothing-behind");
    let genome = base.join("g.json");
    genome_file(&genome, 1);
    let out = base.join("run");
    for bad in [
        ["--weight-power", "NaN"],
        ["--weight-power", "inf"],
        ["--weight-power", "0"],
    ] {
        let spec = format!("neat:{}", genome.display());
        let mut args = vec![
            "--generations",
            "1",
            "--opponent",
            spec.as_str(),
            "--opponent",
            "lowest-legal",
        ];
        args.extend(bad);
        let result = train(&out, &args);
        assert!(!result.status.success(), "{bad:?}");
        assert!(!text(&result.stderr).contains("panicked"), "{bad:?}");
        assert!(
            !out.join("opponents").exists(),
            "{bad:?}: a frozen opponent was left behind"
        );
        assert!(!out.join("checkpoint.json").exists(), "{bad:?}");
    }
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn a_tiny_population_works_with_the_default_candidate_count() {
    // The default re-scores the top 5 genomes, which cannot exceed the
    // population: a small population simply uses all of its genomes.
    let out = run_dir("tinypop");
    let result = cli(&[
        "train",
        "--out",
        out.to_str().unwrap(),
        "--population",
        "3",
        "--generations",
        "2",
        "--matches-per-genome",
        "2",
        "--reeval-matches",
        "4",
        "--rounds",
        "2",
        "--threads",
        "1",
        "--quiet",
    ]);
    assert!(result.status.success(), "stderr: {}", text(&result.stderr));
    std::fs::remove_dir_all(&out).unwrap();
}

fn train_from(out: &Path, source: &Path, extra: &[&str]) -> Output {
    let mut args = vec![
        "train",
        "--out",
        out.to_str().unwrap(),
        "--from",
        source.to_str().unwrap(),
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

#[test]
fn a_run_can_build_on_an_earlier_run() {
    let base = run_dir("warm");
    let source = base.join("a");
    assert!(train(&source, &["--generations", "3"]).status.success());
    let out = base.join("b");
    let result = train_from(&out, &source, &["--generations", "2", "--seed", "9"]);
    assert!(result.status.success(), "{}", text(&result.stderr));
    let stdout = text(&result.stdout);
    assert!(stdout.contains("warm start"), "{stdout}");
    assert_eq!(row_generations(&stdout), vec![0, 1], "{stdout}");
    let log = std::fs::read_to_string(out.join("events.jsonl")).unwrap();
    assert!(
        log.lines().next().unwrap().contains("warm_started_from"),
        "{log}"
    );
    // The new run is an ordinary run: it can be resumed.
    assert!(cli(&[
        "train",
        "--out",
        out.to_str().unwrap(),
        "--resume",
        "--generations",
        "3"
    ])
    .status
    .success());
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn a_bad_from_leaves_nothing_behind_and_points_at_resume_or_the_size() {
    let base = run_dir("warm-bad");
    let out = base.join("b");
    let missing = train_from(&out, &base.join("nowhere"), &["--generations", "1"]);
    assert!(!missing.status.success());
    assert!(!text(&missing.stderr).contains("panicked"));
    assert!(!out.exists(), "a refused start creates nothing");
    let source = base.join("a");
    assert!(train(&source, &["--generations", "1"]).status.success());
    let sized = train_from(&out, &source, &["--population", "20"]);
    assert!(!sized.status.success());
    assert!(!out.exists());
    let resumed = train_from(&out, &source, &["--resume"]);
    assert!(!resumed.status.success());
    let _ = std::fs::remove_dir_all(&base);
}

fn generation_means(events_path: &Path) -> Vec<f64> {
    std::fs::read_to_string(events_path)
        .unwrap()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|event| event["type"] == "generation")
        .map(|event| event["fitness"]["mean"].as_f64().unwrap())
        .collect()
}

fn set_json(out: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(out.join("set.json")).unwrap()).unwrap()
}

#[test]
fn a_set_runs_every_run_in_its_own_directory_with_a_set_eta() {
    let out = run_dir("set");
    let result = train(&out, &["--runs", "2", "--generations", "3"]);
    assert!(result.status.success(), "{}", text(&result.stderr));
    for name in ["run-01", "run-02"] {
        assert!(out.join(name).join("best.json").exists(), "{name}");
        assert_eq!(
            generation_means(&out.join(name).join("events.jsonl")).len(),
            3
        );
    }
    assert_ne!(
        generation_means(&out.join("run-01/events.jsonl")),
        generation_means(&out.join("run-02/events.jsonl")),
        "each run has its own seed"
    );
    let set = set_json(&out);
    assert_eq!(set["total_runs"], 2);
    assert_eq!(set["finished_secs"].as_array().unwrap().len(), 2);
    let stdout = text(&result.stdout);
    assert!(
        stdout.contains("run 1/2") && stdout.contains("run 2/2"),
        "{stdout}"
    );
    assert!(stdout.contains("set ETA"), "{stdout}");
    assert!(stdout.contains("set finished"), "{stdout}");
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn runs_1_is_exactly_a_normal_run() {
    let out = run_dir("set-one");
    let result = train(&out, &["--runs", "1", "--generations", "2"]);
    assert!(result.status.success());
    assert!(out.join("checkpoint.json").exists());
    assert!(!out.join("run-01").exists() && !out.join("set.json").exists());
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn a_killed_set_resumes_where_it_stopped() {
    let out = run_dir("set-kill");
    let mut child = Command::new(env!("CARGO_BIN_EXE_cli"))
        .args([
            "train",
            "--out",
            out.to_str().unwrap(),
            "--runs",
            "2",
            "--generations",
            "8",
            "--population",
            "12",
            "--matches-per-genome",
            "6",
            "--reeval-matches",
            "6",
            "--rounds",
            "3",
            "--threads",
            "2",
            "--quiet",
        ])
        .stdout(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let second = out.join("run-02/checkpoint.json");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    while !second.exists() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    child.kill().unwrap();
    child.wait().unwrap();
    let first_events = std::fs::read(out.join("run-01/events.jsonl")).unwrap();
    let result = cli(&[
        "train",
        "--out",
        out.to_str().unwrap(),
        "--resume",
        "--quiet",
    ]);
    assert!(result.status.success(), "{}", text(&result.stderr));
    assert_eq!(
        std::fs::read(out.join("run-01/events.jsonl")).unwrap(),
        first_events,
        "a finished run is not touched"
    );
    assert_eq!(generation_means(&out.join("run-02/events.jsonl")).len(), 8);
    assert_eq!(set_json(&out)["finished_secs"].as_array().unwrap().len(), 2);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn a_set_killed_between_runs_starts_the_next_run_on_resume() {
    let out = run_dir("set-between");
    assert!(train(&out, &["--runs", "2", "--generations", "2"])
        .status
        .success());
    // Pretend the process died right after run 1 finished.
    std::fs::remove_dir_all(out.join("run-02")).unwrap();
    let mut set = set_json(&out);
    set["finished_secs"] = serde_json::json!([set["finished_secs"][0].clone()]);
    set["current_run"] = serde_json::json!(2);
    std::fs::write(out.join("set.json"), set.to_string()).unwrap();
    let result = cli(&[
        "train",
        "--out",
        out.to_str().unwrap(),
        "--resume",
        "--quiet",
    ]);
    assert!(result.status.success(), "{}", text(&result.stderr));
    assert_eq!(generation_means(&out.join("run-02/events.jsonl")).len(), 2);
    assert_ne!(
        generation_means(&out.join("run-01/events.jsonl")),
        generation_means(&out.join("run-02/events.jsonl")),
        "the restarted run still gets its own seed"
    );
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn a_bad_option_leaves_nothing_behind_in_a_set() {
    let out = run_dir("set-bad");
    for bad in [["--weight-power", "NaN"], ["--runs", "0"]] {
        let mut args = vec!["--generations", "1"];
        if bad[0] == "--weight-power" {
            args.extend(["--runs", "2"]);
        }
        args.extend(bad);
        let result = train(&out, &args);
        assert!(!result.status.success(), "{bad:?}");
        assert!(!text(&result.stderr).contains("panicked"), "{bad:?}");
        assert!(
            !out.join("set.json").exists() && !out.join("run-01").exists(),
            "{bad:?}"
        );
    }
    let _ = std::fs::remove_dir_all(&out);
}

/// A finished run whose checkpoint claims another feature set.
fn run_with_foreign_features(base: &Path) -> PathBuf {
    let source = base.join("foreign");
    assert!(train(&source, &["--generations", "1"]).status.success());
    let path = source.join("checkpoint.json");
    let mut checkpoint: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    checkpoint["feature_count"] = serde_json::json!(3);
    std::fs::write(&path, checkpoint.to_string()).unwrap();
    source
}

#[test]
fn a_refused_warm_start_leaves_nothing_behind_in_a_set_or_with_a_neat_opponent() {
    let base = run_dir("warm-foreign");
    let source = run_with_foreign_features(&base);
    let genome = base.join("g.json");
    genome_file(&genome, 1);
    let spec = format!("neat:{}", genome.display());
    for (name, extra) in [
        ("set", vec!["--runs", "2"]),
        (
            "neat",
            vec!["--opponent", spec.as_str(), "--opponent", "lowest-legal"],
        ),
    ] {
        let out = base.join(name);
        let mut args = vec!["--generations", "1"];
        args.extend(extra);
        let result = train_from(&out, &source, &args);
        assert!(!result.status.success(), "{name}");
        assert!(!text(&result.stderr).contains("panicked"), "{name}");
        assert!(!out.exists(), "{name}: a refused start left {out:?} behind");
    }
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn generations_on_a_set_resume_apply_to_every_run_still_to_go() {
    let out = run_dir("set-generations");
    assert!(train(&out, &["--runs", "3", "--generations", "2"])
        .status
        .success());
    std::fs::remove_dir_all(out.join("run-02")).unwrap();
    std::fs::remove_dir_all(out.join("run-03")).unwrap();
    let mut set = set_json(&out);
    set["finished_secs"] = serde_json::json!([set["finished_secs"][0].clone()]);
    set["current_run"] = serde_json::json!(2);
    std::fs::write(out.join("set.json"), set.to_string()).unwrap();
    let result = cli(&[
        "train",
        "--out",
        out.to_str().unwrap(),
        "--resume",
        "--generations",
        "4",
        "--quiet",
    ]);
    assert!(result.status.success(), "{}", text(&result.stderr));
    assert_eq!(generation_means(&out.join("run-01/events.jsonl")).len(), 2);
    for run in ["run-02", "run-03"] {
        assert_eq!(
            generation_means(&out.join(run).join("events.jsonl")).len(),
            4,
            "{run}"
        );
    }
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn a_skill_weighted_run_writes_the_weight_and_the_additive_fields_and_resumes_with_it() {
    let out = run_dir("skill-weight");
    let result = train(
        &out,
        &[
            "--generations",
            "2",
            "--fitness-skill-weight",
            "0.5",
            "--quiet",
        ],
    );
    assert!(result.status.success(), "stderr: {}", text(&result.stderr));
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("config.json")).unwrap()).unwrap();
    assert!((config["skill_weight"].as_f64().unwrap() - 0.5).abs() < 1e-12);
    let events = std::fs::read_to_string(out.join("events.jsonl")).unwrap();
    assert!(events.contains(r#""skill_term""#), "{events}");
    let resumed = cli(&[
        "train",
        "--out",
        out.to_str().unwrap(),
        "--resume",
        "--generations",
        "3",
        "--quiet",
    ]);
    assert!(
        resumed.status.success(),
        "stderr: {}",
        text(&resumed.stderr)
    );
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("config.json")).unwrap()).unwrap();
    assert!((config["skill_weight"].as_f64().unwrap() - 0.5).abs() < 1e-12);
    std::fs::remove_dir_all(&out).unwrap();

    // The default writes neither the weight nor any skill field.
    let plain = run_dir("skill-weight-off");
    let result = train(&plain, &["--generations", "2", "--quiet"]);
    assert!(result.status.success(), "stderr: {}", text(&result.stderr));
    for file in ["config.json", "events.jsonl", "checkpoint.json"] {
        let content = std::fs::read_to_string(plain.join(file)).unwrap();
        assert!(!content.contains("skill"), "{file}");
    }
    std::fs::remove_dir_all(&plain).unwrap();

    let bad = run_dir("skill-weight-bad");
    let result = train(&bad, &["--fitness-skill-weight", "2"]);
    assert!(!result.status.success());
    assert!(!bad.exists());
}
