//! End-to-end tests of `cli analyse-run`: the real binary on runs it
//! trained itself. Structure and tolerances only, no float goldens.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "arschloch-cli-analyse-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_cli"))
        .args(args)
        .output()
        .expect("failed to run cli binary")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn small_run(out: &Path, extra: &[&str]) {
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
        "--generations",
        "6",
        "--quiet",
    ];
    args.extend_from_slice(extra);
    let result = cli(&args);
    assert!(result.status.success(), "{}", text(&result.stderr));
}

fn analyse(paths: &[&Path], extra: &[&str]) -> Output {
    let mut args = vec!["analyse-run"];
    args.extend(paths.iter().map(|p| p.to_str().unwrap()));
    args.extend_from_slice(extra);
    cli(&args)
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn a_run_dir_and_its_events_file_give_the_report_and_the_json() {
    let base = temp("run");
    let run = base.join("myrun");
    small_run(&run, &[]);
    let out_json = base.join("out.json");
    let result = analyse(&[&run], &["--json", out_json.to_str().unwrap()]);
    assert!(result.status.success(), "{}", text(&result.stderr));
    let stdout = text(&result.stdout);
    for expected in [
        "== myrun ==",
        "generations",
        "6",
        "final re-evaluation",
        "stall generation",
        "trend, last third",
        "Per opponent",
        "Verdict:",
    ] {
        assert!(stdout.contains(expected), "{expected} missing in\n{stdout}");
    }
    assert!(
        !stdout.contains("== comparison =="),
        "one run has no comparison"
    );
    let value = json(&out_json);
    let items = value.as_array().unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["label"], "myrun");
    let analysis = &items[0]["analysis"];
    assert_eq!(analysis["generations"], 6);
    for key in [
        "stall_generation",
        "noise",
        "trend",
        "curse",
        "opponents",
        "growth",
        "population",
        "verdict",
    ] {
        assert!(analysis.get(key).is_some(), "{key}");
    }
    assert_ne!(analysis["verdict"].as_array().unwrap().len(), 0);

    let events = run.join("events.jsonl");
    let from_file = analyse(&[&events], &[]);
    assert!(from_file.status.success(), "{}", text(&from_file.stderr));
    assert!(text(&from_file.stdout).contains("== myrun =="));
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn a_set_dir_gives_one_report_per_run_and_a_comparison() {
    let base = temp("set");
    let set = base.join("set");
    small_run(&set, &["--runs", "2"]);
    let single = base.join("single");
    small_run(&single, &[]);
    let result = analyse(&[&set, &single], &[]);
    assert!(result.status.success(), "{}", text(&result.stderr));
    let stdout = text(&result.stdout);
    for expected in [
        "== run-01 ==",
        "== run-02 ==",
        "== single ==",
        "== comparison ==",
        "curse gap",
    ] {
        assert!(stdout.contains(expected), "{expected} missing in\n{stdout}");
    }
    let comparison = stdout.split("== comparison ==").nth(1).unwrap();
    for label in ["run-01", "run-02", "single"] {
        assert!(comparison.contains(label), "{label}");
    }
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn bad_paths_fail_with_a_clear_message_and_no_panic() {
    let base = temp("bad");
    std::fs::create_dir_all(&base).unwrap();
    let missing = base.join("nope");
    let empty = base.join("empty.jsonl");
    std::fs::write(&empty, "").unwrap();
    let corrupt = base.join("corrupt.jsonl");
    std::fs::write(&corrupt, "this is not json\n{\"half\": \n").unwrap();
    let empty_dir = base.join("dir");
    std::fs::create_dir_all(&empty_dir).unwrap();
    for path in [&missing, &empty, &corrupt, &empty_dir] {
        let result = analyse(&[path], &[]);
        let stderr = text(&result.stderr);
        assert!(!result.status.success(), "{}", path.display());
        assert!(stderr.contains(path.to_str().unwrap()), "{stderr}");
        assert!(!stderr.contains("panicked"), "{stderr}");
        assert_ne!(result.status.code(), Some(101), "{stderr}");
    }
    let no_args = cli(&["analyse-run"]);
    assert!(!no_args.status.success());
    let _ = std::fs::remove_dir_all(&base);
}

/// Rewrites a real events file as an old run would have been: no timings,
/// no racing, plus a line that is not an event.
fn strip_to_old_format(source: &Path, target: &Path) {
    let mut lines = Vec::new();
    for line in std::fs::read_to_string(source).unwrap().lines() {
        let mut value: serde_json::Value = serde_json::from_str(line).unwrap();
        if let Some(object) = value.as_object_mut() {
            object.remove("timings");
            object.remove("racing");
        }
        lines.push(value.to_string());
    }
    lines.insert(1, "{\"event\": \"from_the_future\"}".to_owned());
    std::fs::write(target, lines.join("\n") + "\n").unwrap();
}

#[test]
fn an_old_format_file_without_optional_fields_is_analysed_and_reports_skipped_lines() {
    let base = temp("old");
    let run = base.join("run");
    small_run(&run, &[]);
    let old = base.join("old.jsonl");
    strip_to_old_format(&run.join("events.jsonl"), &old);
    let out_json = base.join("old.json");
    let result = analyse(&[&old], &["--json", out_json.to_str().unwrap()]);
    assert!(result.status.success(), "{}", text(&result.stderr));
    let stdout = text(&result.stdout);
    assert!(stdout.contains("== old =="), "{stdout}");
    assert!(stdout.contains("skipped"), "{stdout}");
    assert!(!stdout.contains("seconds per stage"), "{stdout}");
    let analysis = &json(&out_json)[0]["analysis"];
    assert_eq!(analysis["skipped_lines"], 1);
    assert_eq!(analysis["generations"], 6);
    assert!(analysis["timings"].is_null() && analysis["racing"].is_null());
    let _ = std::fs::remove_dir_all(&base);
}
