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

/// FNV-1a, to pin the bytes of files without a hashing dependency.
fn fnv(path: &Path) -> u64 {
    std::fs::read(path)
        .unwrap()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
}

fn small_run(out: &Path, threads: &str, extra: &[&str]) -> Output {
    let mut args = vec![
        "train",
        "--out",
        out.to_str().unwrap(),
        "--population",
        "20",
        "--matches-per-genome",
        "6",
        "--reeval-matches",
        "10",
        "--rounds",
        "3",
        "--threads",
        threads,
        "--quiet",
    ];
    args.extend_from_slice(extra);
    cli(&args)
}

/// FNV-1a of `bytes`.
fn fnv_bytes(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// A hash of every event with its wall-clock fields removed (they differ
/// from run to run), so the golden pins everything else the events say.
fn timeless_events_hash(dir: &Path) -> u64 {
    let lines: Vec<String> = std::fs::read_to_string(dir.join("events.jsonl"))
        .unwrap()
        .lines()
        .map(|line| {
            let mut value: serde_json::Value = serde_json::from_str(line).unwrap();
            let object = value.as_object_mut().unwrap();
            for key in [
                "elapsed_secs",
                "generation_secs",
                "rounds_per_sec",
                "timings",
            ] {
                object.remove(key);
            }
            value.to_string()
        })
        .collect();
    fnv_bytes(lines.join("\n").as_bytes())
}

/// The hashes of the decision files, folded in file-name order.
fn decisions_hash(dir: &Path) -> u64 {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir.join("decisions"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    files.sort();
    files.iter().fold(0xcbf2_9ce4_8422_2325, |h, path| {
        (h ^ fnv(path)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Runs trained before `--features` (and the Phase 13 training options)
/// existed, captured with the build that preceded them: `--features base`
/// (the default until the `hidden` set became it) must reproduce their files
/// byte for byte, and their events (apart from the wall-clock fields) and
/// decision samples too.
#[test]
#[allow(clippy::too_many_lines)] // the golden table is the bulk of it
fn the_base_feature_set_trains_exactly_as_before_the_option_existed() {
    for (name, extra, hashes, decisions, events) in [
        (
            "features-golden-4p",
            vec!["--generations", "3", "--seed", "7"],
            [
                ("config.json", 0x31a9_dd66_51cc_4b2a_u64),
                ("best.json", 0xddb5_ea26_7675_ff14),
                ("gen-0000.json", 0x68a4_edb8_1f21_e2b1),
                ("gen-0002.json", 0x24da_74c1_3169_29f4),
            ],
            0x2673_670c_3a47_bb12_u64,
            0xe0ad_cf2b_4e0d_a49e_u64,
        ),
        (
            "features-golden-5p",
            vec![
                "--generations",
                "4",
                "--seed",
                "3",
                "--player-count",
                "5",
                "--deck-variant",
                "double",
                "--pass-rule",
                "free",
            ],
            [
                ("config.json", 0x4586_20e5_9a23_5273),
                ("best.json", 0x5206_24db_58d1_a72c),
                ("gen-0000.json", 0x7a21_c9cb_6b2e_5f8f),
                ("gen-0002.json", 0x78f3_660f_7e21_7bfb),
            ],
            0xe504_b521_8170_29cd,
            0x2de6_36c6_4235_77e1,
        ),
        (
            "features-golden-hall",
            vec![
                "--generations",
                "4",
                "--seed",
                "11",
                "--hall-of-fame",
                "2",
                "--hall-interval",
                "1",
                "--champion-candidates",
                "3",
            ],
            [
                ("config.json", 0x4faa_348e_e06b_d4da),
                ("best.json", 0xbf7f_3f1a_802e_59fc),
                ("gen-0000.json", 0x5401_a5fd_fd2a_885c),
                ("gen-0002.json", 0xbf7f_3f1a_802e_59fc),
            ],
            0x0838_4bae_06c3_7ae0,
            0x1bea_9f9f_b24a_1bf7,
        ),
    ] {
        let out = run_dir(name);
        let mut base = extra.clone();
        base.extend(["--features", "base"]);
        let result = small_run(&out, "2", &base);
        assert!(result.status.success(), "stderr: {}", text(&result.stderr));
        for (file, expected) in hashes {
            assert_eq!(fnv(&out.join(file)), expected, "{name}/{file}");
        }
        eprintln!(
            "CAPTURE {name} {:#x} {:#x}",
            decisions_hash(&out),
            timeless_events_hash(&out)
        );
        // The sampled decision scores and the event floats are printed from
        // values the platform's libm computes (exp/tanh), which differ in the
        // last digit between Linux, macOS and Windows; the genome files above
        // are what must match everywhere.
        if cfg!(target_os = "linux") {
            assert_eq!(decisions_hash(&out), decisions, "{name} decisions");
            assert_eq!(timeless_events_hash(&out), events, "{name} events");
        }
        let config = std::fs::read_to_string(out.join("config.json")).unwrap();
        for option in ["features", "fitness", "racing", "player_counts", "decks"] {
            assert!(!config.contains(option), "{option}: {config}");
        }
        // Naming the default fitness explicitly changes nothing either.
        let explicit = run_dir(&format!("{name}-explicit"));
        let mut again = base.clone();
        again.extend(["--fitness", "mean"]);
        assert!(small_run(&explicit, "1", &again).status.success());
        for (file, expected) in hashes {
            assert_eq!(
                fnv(&explicit.join(file)),
                expected,
                "{name}/{file} explicit"
            );
        }
        // Same platform, same run: the explicit spelling must equal the default.
        assert_eq!(
            decisions_hash(&explicit),
            decisions_hash(&out),
            "{name} explicit"
        );
        assert_eq!(
            timeless_events_hash(&explicit),
            timeless_events_hash(&out),
            "{name} explicit"
        );
        std::fs::remove_dir_all(&out).unwrap();
        std::fs::remove_dir_all(&explicit).unwrap();
    }
}

fn names_of(path: &Path) -> Vec<String> {
    let file: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    file["feature_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_str().unwrap().to_owned())
        .collect()
}

#[test]
fn every_feature_set_trains_records_its_names_and_plays() {
    for (set, count) in [("trick", 25), ("hidden", 28), ("all", 31)] {
        let out = run_dir(&format!("features-{set}"));
        let result = small_run(
            &out,
            "2",
            &["--generations", "3", "--seed", "5", "--features", set],
        );
        assert!(result.status.success(), "{set}: {}", text(&result.stderr));
        let names = names_of(&out.join("best.json"));
        assert_eq!(names.len(), count, "{set}");
        assert_eq!(names[..20], sim::FEATURE_NAMES, "{set}: v1 prefix");
        assert_eq!(names, sim::FEATURE_NAMES_V2[..count], "{set}");
        let config = std::fs::read_to_string(out.join("config.json")).unwrap();
        assert!(
            config.contains(&format!("\"features\": \"{set}\"")),
            "{config}"
        );
        let checkpoint: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(out.join("checkpoint.json")).unwrap())
                .unwrap();
        assert_eq!(checkpoint["feature_count"], count);
        let start = std::fs::read_to_string(out.join("events.jsonl")).unwrap();
        let start: serde_json::Value = serde_json::from_str(start.lines().next().unwrap()).unwrap();
        assert_eq!(start["feature_names"].as_array().unwrap().len(), count);
        // The champion plays in an ordinary run next to the other sets' players.
        let played = cli(&[
            "--player-count",
            "4",
            "--matches",
            "10",
            "--rounds",
            "2",
            "--output",
            out.join("played.json").to_str().unwrap(),
            "--strategy",
            &format!("neat:{}", out.join("best.json").display()),
            "--strategy",
            "lowest-legal",
            "--strategy",
            "lowest-legal",
            "--strategy",
            "adaptive:reading,tempo,bully",
        ]);
        assert!(played.status.success(), "{set}: {}", text(&played.stderr));
        std::fs::remove_dir_all(&out).unwrap();
    }
}

#[test]
fn an_all_features_run_resumes_identically_and_ignores_the_thread_count() {
    let options = ["--seed", "11", "--features", "all", "--player-count", "4"];
    let straight = run_dir("features-all-straight");
    let mut args = vec!["--generations", "3"];
    args.extend(options);
    assert!(small_run(&straight, "1", &args).status.success());

    let threaded = run_dir("features-all-threads");
    assert!(small_run(&threaded, "4", &args).status.success());

    let resumed = run_dir("features-all-resumed");
    let mut first = vec!["--generations", "2"];
    first.extend(options);
    assert!(small_run(&resumed, "2", &first).status.success());
    let more = cli(&[
        "train",
        "--out",
        resumed.to_str().unwrap(),
        "--resume",
        "--generations",
        "3",
        "--threads",
        "3",
        "--quiet",
    ]);
    assert!(more.status.success(), "{}", text(&more.stderr));

    for file in [
        "config.json",
        "best.json",
        "gen-0000.json",
        "gen-0001.json",
        "gen-0002.json",
    ] {
        assert_eq!(
            fnv(&straight.join(file)),
            fnv(&threaded.join(file)),
            "{file}: threads"
        );
        // (config.json keeps the generation total the run started with.)
        if file != "config.json" {
            assert_eq!(
                fnv(&straight.join(file)),
                fnv(&resumed.join(file)),
                "{file}: resume"
            );
        }
    }
    for dir in [&straight, &threaded, &resumed] {
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn a_warm_start_needs_the_same_feature_set_as_its_source() {
    let base = run_dir("features-warm");
    let source = base.join("trick-source");
    assert!(
        small_run(&source, "2", &["--generations", "1", "--features", "trick"])
            .status
            .success()
    );
    // Same set: builds on the source.
    let same = base.join("same");
    let result = train_from(
        &same,
        &source,
        &["--generations", "1", "--features", "trick", "--quiet"],
    );
    assert!(result.status.success(), "{}", text(&result.stderr));
    assert_eq!(names_of(&same.join("best.json")).len(), 25);
    // Another set (including the default) is refused, with the reason, and
    // nothing is written.
    for other in [None, Some("all")] {
        let out = base.join(format!("other-{}", other.unwrap_or("base")));
        let mut args = vec!["--generations", "1"];
        if let Some(set) = other {
            args.extend(["--features", set]);
        }
        let result = train_from(&out, &source, &args);
        assert!(!result.status.success());
        let stderr = text(&result.stderr);
        assert!(stderr.contains("--features"), "{stderr}");
        assert!(!stderr.contains("panicked"), "{stderr}");
        assert!(!out.exists());
    }
    let _ = std::fs::remove_dir_all(&base);
}

fn config_json(out: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(out.join("config.json")).unwrap()).unwrap()
}

#[test]
fn each_fitness_variant_trains_records_itself_refuses_a_resume_change_and_bad_values() {
    for (spec, expected) in [
        ("risk:0.5", serde_json::json!({"risk": {"lambda": 0.5}})),
        ("tanh:4", serde_json::json!({"tanh": {"kappa": 4.0}})),
    ] {
        let out = run_dir(&format!("fitness-{}", spec.replace(':', "-")));
        let result = small_run(&out, "2", &["--generations", "2", "--fitness", spec]);
        assert!(result.status.success(), "{spec}: {}", text(&result.stderr));
        assert_eq!(config_json(&out)["fitness"], expected);
        let events = std::fs::read_to_string(out.join("events.jsonl")).unwrap();
        assert!(events.contains("\"spread\""), "{spec}: spread is reported");
        // The fitness is fixed for the run: a resume may not change it, but
        // continues with it.
        let conflict = cli(&[
            "train",
            "--out",
            out.to_str().unwrap(),
            "--resume",
            "--fitness",
            "mean",
        ]);
        assert!(!conflict.status.success());
        let resumed = cli(&[
            "train",
            "--out",
            out.to_str().unwrap(),
            "--resume",
            "--generations",
            "3",
            "--quiet",
        ]);
        assert!(resumed.status.success(), "{}", text(&resumed.stderr));
        std::fs::remove_dir_all(&out).unwrap();
    }
    for bad in ["median", "risk:-1", "tanh:0", "risk"] {
        let out = run_dir("fitness-bad");
        let result = small_run(&out, "1", &["--fitness", bad]);
        assert!(!result.status.success(), "{bad}");
        assert!(!out.exists(), "{bad}: nothing is written");
    }
    // Not with the luck-adjusted term.
    let out = run_dir("fitness-skill");
    let result = small_run(
        &out,
        "1",
        &["--fitness", "risk:1", "--fitness-skill-weight", "0.5"],
    );
    assert!(!result.status.success());
    assert!(
        text(&result.stderr).contains("skill_weight"),
        "{}",
        text(&result.stderr)
    );
    assert!(!out.exists());
}

#[test]
fn racing_needs_both_flags_trains_reports_the_shift_and_cannot_change_on_resume() {
    let out = run_dir("racing");
    let result = cli(&[
        "train",
        "--out",
        out.to_str().unwrap(),
        "--population",
        "20",
        "--matches-per-genome",
        "6",
        "--reeval-matches",
        "10",
        "--rounds",
        "3",
        "--generations",
        "2",
        "--threads",
        "2",
        "--racing-top",
        "0.25",
        "--racing-matches",
        "8",
    ]);
    assert!(result.status.success(), "{}", text(&result.stderr));
    let stdout = text(&result.stdout);
    assert!(
        stdout.contains("racing: top 5 genomes x 8 extra matches"),
        "{stdout}"
    );
    assert!(stdout.contains("shift"), "{stdout}");
    let config = config_json(&out);
    assert_eq!(
        (
            config["racing_top"].as_f64(),
            config["racing_matches"].as_u64()
        ),
        (Some(0.25), Some(8))
    );
    let events = std::fs::read_to_string(out.join("events.jsonl")).unwrap();
    assert!(
        events.contains("\"racing\":{\"raced\":5,\"extra_matches\":8"),
        "{events}"
    );
    assert!(!cli(&[
        "train",
        "--out",
        out.to_str().unwrap(),
        "--resume",
        "--racing-top",
        "0.5"
    ])
    .status
    .success());
    std::fs::remove_dir_all(&out).unwrap();
    for bad in [
        vec!["--racing-top", "0.25"],
        vec!["--racing-matches", "8"],
        vec!["--racing-top", "0", "--racing-matches", "8"],
        vec!["--racing-top", "1.5", "--racing-matches", "8"],
    ] {
        let out = run_dir("racing-bad");
        let result = small_run(&out, "1", &bad);
        assert!(!result.status.success(), "{bad:?}");
        assert!(!out.exists());
    }
}

#[test]
#[allow(clippy::too_many_lines)] // one scenario per option of the group
fn several_settings_train_warn_without_the_table_features_and_record_the_lists() {
    let out = run_dir("multi");
    let result = small_run(
        &out,
        "2",
        &[
            "--generations",
            "2",
            "--player-counts",
            "3,5",
            "--decks",
            "single,double",
            "--features",
            "all",
        ],
    );
    assert!(result.status.success(), "{}", text(&result.stderr));
    assert!(
        !text(&result.stderr).contains("warning"),
        "{}",
        text(&result.stderr)
    );
    let config = config_json(&out);
    assert_eq!(config["player_counts"], serde_json::json!([3, 5]));
    assert_eq!(config["decks"], serde_json::json!(["Single", "Double"]));
    assert_eq!(
        config["player_count"], 3,
        "the first entry stays the single value"
    );
    let events = std::fs::read_to_string(out.join("events.jsonl")).unwrap();
    assert!(events.contains("\"by_setting\""), "{events}");
    std::fs::remove_dir_all(&out).unwrap();

    // Without the table inputs it still runs, with a warning (not an error).
    let out = run_dir("multi-base");
    let result = small_run(&out, "2", &["--generations", "1", "--player-counts", "3,4"]);
    assert!(result.status.success(), "{}", text(&result.stderr));
    assert!(
        text(&result.stderr).contains("warning") && text(&result.stderr).contains("--features all"),
        "{}",
        text(&result.stderr)
    );
    std::fs::remove_dir_all(&out).unwrap();

    // A single-entry list is the single flag: the config file does not change.
    let (listed, single) = (run_dir("one-listed"), run_dir("one-single"));
    assert!(small_run(
        &listed,
        "1",
        &[
            "--generations",
            "1",
            "--player-counts",
            "5",
            "--decks",
            "double"
        ]
    )
    .status
    .success());
    assert!(small_run(
        &single,
        "1",
        &[
            "--generations",
            "1",
            "--player-count",
            "5",
            "--deck-variant",
            "double"
        ]
    )
    .status
    .success());
    assert_eq!(
        std::fs::read(listed.join("config.json")).unwrap(),
        std::fs::read(single.join("config.json")).unwrap()
    );
    std::fs::remove_dir_all(&listed).unwrap();
    std::fs::remove_dir_all(&single).unwrap();

    // Too few matches for the settings, a duplicate, and the single flag together with a list.
    for bad in [
        vec!["--player-counts", "3,4,5,6", "--matches-per-genome", "3"],
        vec!["--player-counts", "3,3"],
        vec!["--player-counts", "3,4", "--player-count", "5"],
        vec!["--player-counts", "3,4", "--fitness-skill-weight", "0.5"],
    ] {
        let out = run_dir("multi-bad");
        let result = cli(&[
            "train",
            "--out",
            out.to_str().unwrap(),
            "--population",
            "12",
            "--reeval-matches",
            "10",
            "--rounds",
            "2",
            "--quiet",
        ]
        .iter()
        .chain(bad.iter())
        .copied()
        .collect::<Vec<_>>());
        assert!(!result.status.success(), "{bad:?}");
        assert!(!out.exists(), "{bad:?}");
    }
}

#[test]
fn auto_checks_the_cheap_things_before_the_probe() {
    // A bad candidate spec: reported at once, nothing probed, nothing created.
    let out = run_dir("auto-spec");
    let result = cli(&[
        "train",
        "--auto",
        "--out",
        out.to_str().unwrap(),
        "--opponent",
        "nonsense",
    ]);
    assert!(!result.status.success());
    assert!(
        text(&result.stderr).contains("invalid --opponent `nonsense`"),
        "{}",
        text(&result.stderr)
    );
    assert!(!text(&result.stdout).contains("Probe:"));
    assert!(!out.exists());
    // A busy dashboard port: also before the probe.
    let busy = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = busy.local_addr().unwrap().port().to_string();
    let out = run_dir("auto-port");
    let result = cli(&[
        "train",
        "--auto",
        "--out",
        out.to_str().unwrap(),
        "--serve",
        &port,
    ]);
    assert!(!result.status.success());
    assert!(
        text(&result.stderr).contains("cannot start the dashboard"),
        "{}",
        text(&result.stderr)
    );
    assert!(!text(&result.stdout).contains("Probe:"));
}

#[test]
fn a_run_without_features_trains_hidden_and_an_older_base_run_resumes_as_base() {
    // The default is the `hidden` set, and the run says so in its files.
    let fresh = run_dir("default-hidden");
    assert!(
        small_run(&fresh, "2", &["--generations", "2", "--seed", "5"])
            .status
            .success()
    );
    assert_eq!(names_of(&fresh.join("best.json")).len(), 28);
    let config = std::fs::read_to_string(fresh.join("config.json")).unwrap();
    assert!(config.contains("\"features\": \"hidden\""), "{config}");
    std::fs::remove_dir_all(&fresh).unwrap();

    // A base run (as every run was before the default changed) carries no
    // `features` in its config; resuming it without naming a set stays base.
    let old = run_dir("default-hidden-old-base");
    assert!(small_run(
        &old,
        "2",
        &["--generations", "2", "--seed", "5", "--features", "base"]
    )
    .status
    .success());
    let config = std::fs::read_to_string(old.join("config.json")).unwrap();
    assert!(!config.contains("features"), "{config}");
    let resumed = cli(&[
        "train",
        "--out",
        old.to_str().unwrap(),
        "--resume",
        "--generations",
        "3",
        "--threads",
        "2",
        "--quiet",
    ]);
    assert!(resumed.status.success(), "{}", text(&resumed.stderr));
    assert_eq!(names_of(&old.join("best.json")), sim::FEATURE_NAMES);
    std::fs::remove_dir_all(&old).unwrap();
}
