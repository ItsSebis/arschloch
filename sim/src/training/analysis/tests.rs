//! Unit tests of the run analysis with small hand-computed fixtures.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::json;

use super::super::config::test_support::sample;
use super::super::events::{Event, GenerationEvent, RunEnd, RunStart, SCHEMA_VERSION};
use super::*;

/// One generation event with the fields the analysis reads.
struct Gen {
    generation: u32,
    mean: f64,
    se: f64,
    train: f64,
    opponents: Vec<(&'static str, f64)>,
    nodes: usize,
    histogram: Vec<u32>,
}

fn gen(generation: u32, mean: f64) -> Gen {
    Gen {
        generation,
        mean,
        se: 0.05,
        train: mean,
        opponents: Vec::new(),
        nodes: 1,
        histogram: vec![1; 10],
    }
}

fn stat(mean: f64, se: f64) -> serde_json::Value {
    json!({"mean": mean, "std_error": se, "matches": 10, "placements": [1, 1]})
}

impl Gen {
    fn event(&self) -> GenerationEvent {
        serde_json::from_value(json!({
            "generation": self.generation,
            "elapsed_secs": 1.0,
            "generation_secs": 1.0,
            "rounds_evaluated": 10,
            "total_rounds": 10,
            "rounds_per_sec": 10.0,
            "fitness": {"best": 1.0, "mean": 0.0, "median": 0.0, "min": -1.0,
                        "std_dev": 0.1, "histogram": self.histogram},
            "champion": {
                "train_fitness": self.train,
                "reeval": stat(self.mean, self.se),
                "hidden_nodes": self.nodes,
                "enabled_connections": 5 + self.generation,
                "genome_file": "g.json",
                "is_new_best": false,
            },
            "opponents": self.opponents.iter()
                .map(|(n, m)| json!({"name": n, "score": stat(*m, 0.05)}))
                .collect::<Vec<_>>(),
            "species": [],
            "compatibility_threshold": 3.0 + f64::from(self.generation),
            "complexity": {"mean_hidden_nodes": 1.0, "mean_enabled_connections": 5.0,
                           "innovation_count": 10},
        }))
        .unwrap()
    }
}

fn run_of(gens: &[Gen]) -> RunData {
    RunData {
        label: "fixture".into(),
        start: None,
        generations: gens.iter().map(Gen::event).collect(),
        end: None,
        skipped_lines: 0,
    }
}

fn line(event: &Event) -> String {
    serde_json::to_string(event).unwrap()
}

fn start_event() -> Event {
    Event::RunStart(Box::new(RunStart {
        schema_version: SCHEMA_VERSION,
        config: sample(),
        opponents: vec![],
        feature_names: vec![],
        resumed_from_generation: None,
        warm_started_from: None,
    }))
}

fn temp_dir(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("arschloch-analysis-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_events(path: &Path, gens: &[Gen]) {
    let mut text = line(&start_event());
    for g in gens {
        text.push('\n');
        text.push_str(&line(&Event::Generation(Box::new(g.event()))));
    }
    fs::write(path, text).unwrap();
}

// ---- noise, stall, curse -------------------------------------------------

#[test]
fn noise_and_curse_numbers_match_hand_computation() {
    let means = [0.0, 0.1, 0.3, 0.2];
    let ses = [0.04, 0.06, 0.05, 0.05];
    let trains = [0.1, 0.2, 0.5, 0.3];
    let gens: Vec<Gen> = (0..4)
        .map(|i| Gen {
            se: ses[i],
            train: trains[i],
            ..gen(u32::try_from(i).unwrap(), means[i])
        })
        .collect();
    let mut run = run_of(&gens);
    run.end = Some(RunEnd {
        generations_completed: 4,
        best_generation: Some(2),
        best_reeval: Some(serde_json::from_value(stat(0.3, 0.05)).unwrap()),
        best_heldout: Some(serde_json::from_value(stat(0.2, 0.05)).unwrap()),
        elapsed_secs: 1.0,
    });
    let a = analyse(&run);
    assert!((a.noise.single_se - 0.05).abs() < 1e-12);
    // changes 0.1, 0.2, -0.1: mean 1/15, sample variance 0.07 / 3 = 0.0233...
    let sd = a.noise.change_sd.unwrap();
    assert!((sd - (0.07_f64 / 3.0).sqrt()).abs() < 1e-12, "{sd}");
    // train - reeval: 0.1, 0.1, 0.2, 0.1.
    assert!((a.curse.train_minus_reeval - 0.125).abs() < 1e-12);
    let gap = a.curse.reeval_minus_heldout.unwrap();
    assert!((gap.value - 0.1).abs() < 1e-12);
    assert!((gap.in_heldout_se.unwrap() - 2.0).abs() < 1e-9);
    assert_eq!(a.best_generation, Some(2));
    assert!((a.final_reeval.unwrap().mean - 0.3).abs() < 1e-12);
    assert!((a.final_heldout.unwrap().mean - 0.2).abs() < 1e-12);
    // Four generations: no last-third slope (one point), none overall? n = 4 gives one.
    assert!(a.trend.last_third.is_none());
    assert!(a.trend.overall.is_some());
}

#[test]
fn without_run_end_there_is_no_heldout_gap() {
    let a = analyse(&run_of(&[gen(0, 0.0), gen(1, 0.1)]));
    assert_eq!(a.curse.reeval_minus_heldout, None);
    assert_eq!(a.final_reeval, None);
    assert_eq!(a.noise.change_sd, None, "one change has no spread");
}

#[test]
fn stall_generation_follows_the_smoothed_curve() {
    // Window 10: the first ten means are 0,0,...,0 then jump to 1 and
    // stay: smoothed climbs 0.1 per generation until 1.0 at index 19.
    // With se 0.05 the band starts at 0.95, reached at index 19 (0.95 at
    // index 18 is 0.9; index 19 is 1.0).
    let gens: Vec<Gen> = (0..30)
        .map(|g| gen(g, if g < 10 { 0.0 } else { 1.0 }))
        .collect();
    let a = analyse(&run_of(&gens));
    // smoothed[i] = (i - 9) / 10 for 10 <= i <= 19; >= 0.95 first at 19.
    assert_eq!(a.stall_generation, 19);
    assert!((a.peak_smoothed - 1.0).abs() < 1e-12);
    let s = series(&run_of(&gens));
    assert!((s.smoothed[14] - 0.5).abs() < 1e-12);
}

// ---- trend, opponents, growth, population --------------------------------

fn nine_with_opponents() -> RunData {
    let gens: Vec<Gen> = (0..9)
        .map(|g: u32| {
            let alt = if g.is_multiple_of(2) { 0.01 } else { -0.01 };
            Gen {
                opponents: vec![
                    ("Rising", 0.1 * f64::from(g)),
                    ("Flat", 0.5 + alt),
                    ("Falling", -0.05 * f64::from(g)),
                ],
                ..gen(g, 0.0)
            }
        })
        .collect();
    run_of(&gens)
}

#[test]
fn opponents_are_classified_by_their_slope_interval() {
    let a = analyse(&nine_with_opponents());
    let by = |n: &str| a.opponents.iter().find(|o| o.name == n).unwrap();
    assert_eq!(by("Rising").class, Some(TrendClass::Rising));
    assert_eq!(by("Flat").class, Some(TrendClass::Flat));
    assert_eq!(by("Falling").class, Some(TrendClass::Falling));
    // Slopes are per 100 generations.
    assert!((by("Rising").slope.unwrap().value - 10.0).abs() < 1e-9);
    // Thirds are generations 0..3 and 6..9.
    assert!((by("Rising").first_third_mean - 0.1).abs() < 1e-12);
    assert!((by("Rising").last_third_mean - 0.7).abs() < 1e-12);
    assert!((by("Falling").last_third_mean + 0.35).abs() < 1e-12);
    // A flat champion curve contains zero: a perfect constant has an
    // interval of width zero at zero, which is neither above nor below.
    assert!(a.trend.last_third.unwrap().contains_zero());
}

#[test]
fn trend_is_per_100_generations_on_the_generation_number() {
    let gens: Vec<Gen> = (0..12).map(|g| gen(g, 0.01 * f64::from(g))).collect();
    let a = analyse(&run_of(&gens));
    let overall = a.trend.overall.unwrap();
    assert!((overall.value - 1.0).abs() < 1e-9, "{overall:?}");
    assert!(a.trend.last_third.unwrap().is_positive());
}

#[test]
fn growth_without_gain_needs_node_growth_in_the_last_third_and_a_flat_trend() {
    let make = |nodes: fn(u32) -> usize, mean: fn(u32) -> f64| {
        let gens: Vec<Gen> = (0..9)
            .map(|g| Gen {
                nodes: nodes(g),
                ..gen(g, mean(g))
            })
            .collect();
        analyse(&run_of(&gens)).growth
    };
    let flat = |g: u32| if g.is_multiple_of(2) { 0.1 } else { 0.0 };
    // nodes grow 2 -> 5 in the last third (generation 6 has 3 nodes).
    let growth = make(|g| 1 + g as usize / 2, flat);
    assert_eq!(
        (growth.first_hidden_nodes, growth.last_hidden_nodes),
        (1, 5)
    );
    assert_eq!((growth.first_connections, growth.last_connections), (5, 13));
    assert!(growth.without_gain);
    // No growth in the last third.
    assert!(!make(|g| 1 + g.min(6) as usize, flat).without_gain);
    // Growth with a rising curve is not "without gain".
    assert!(!make(|g| 1 + g as usize, |g| 0.1 * f64::from(g)).without_gain);
}

#[test]
fn population_shape_uses_the_last_third_and_first_last_values() {
    // Top bin holds 6 of 10 genomes in the last third, 1 of 10 before.
    let gens: Vec<Gen> = (0..9)
        .map(|g| Gen {
            histogram: if g >= 6 {
                vec![0, 0, 0, 0, 0, 0, 0, 0, 4, 6]
            } else {
                vec![1; 10]
            },
            ..gen(g, 0.0)
        })
        .collect();
    let p = analyse(&run_of(&gens)).population;
    assert!((p.top_bin_share.unwrap() - 0.6).abs() < 1e-12);
    assert_eq!((p.species_first, p.species_last), (0, 0));
    assert!((p.threshold_first - 3.0).abs() < 1e-12);
    assert!((p.threshold_last - 11.0).abs() < 1e-12);
}

#[test]
fn racing_and_timings_are_totals_and_optional() {
    let a = analyse(&run_of(&[gen(0, 0.0), gen(1, 0.0)]));
    assert!(a.racing.is_none() && a.timings.is_none());

    let mut run = run_of(&[gen(0, 0.0), gen(1, 0.0)]);
    for g in &mut run.generations {
        g.racing = Some(
            serde_json::from_value(json!({
                "raced": 3, "extra_matches": 4, "mean_before": 0.5, "mean_after": 0.3,
                "mean_shift": -0.2, "raced_final": 5,
            }))
            .unwrap(),
        );
        g.timings = Some(super::super::events::StageTimings {
            training_evaluation: 1.5,
            ..Default::default()
        });
    }
    let a = analyse(&run);
    let racing = a.racing.unwrap();
    assert_eq!((racing.generations, racing.extra_matches), (2, 40));
    assert!((racing.mean_shift + 0.2).abs() < 1e-12);
    assert_eq!(racing.extra_share, None, "no run_start, no share");
    assert!((a.timings.unwrap().training_evaluation - 3.0).abs() < 1e-12);
}

#[test]
fn series_carry_one_entry_per_generation() {
    let run = nine_with_opponents();
    let s = series(&run);
    assert_eq!(s.generation, (0..9).collect::<Vec<u32>>());
    assert_eq!(s.reeval_mean.len(), 9);
    assert_eq!(s.opponents.len(), 3);
    assert_eq!(s.opponents[0].name, "Rising");
    assert_eq!(s.opponents[0].means[3], Some(0.1 * 3.0));
    assert!(serde_json::to_string(&s).unwrap().contains("smoothed"));
}

// ---- verdict rules --------------------------------------------------------

fn flat() -> Slope {
    Slope {
        value: 0.0,
        low: -1.0,
        high: 1.0,
    }
}

fn up() -> Slope {
    Slope {
        value: 2.0,
        low: 1.0,
        high: 3.0,
    }
}

fn opponent(name: &str, class: TrendClass) -> OpponentAnalysis {
    OpponentAnalysis {
        name: name.into(),
        first_third_mean: 0.0,
        last_third_mean: 0.0,
        slope: None,
        class: Some(class),
    }
}

/// An analysis that fires no rule: 100 generations, stall at 80 (between
/// the 60% and 95% limits), no last-third trend, quiet everything.
fn quiet() -> RunAnalysis {
    let mut a = analyse(&run_of(&[gen(0, 0.0), gen(1, 0.0)]));
    a.generations = 100;
    a.stall_generation = 80;
    a.trend.last_third = None;
    a.noise = Noise {
        single_se: 0.05,
        change_sd: Some(0.05),
    };
    a.curse.reeval_minus_heldout = None;
    a.opponents.clear();
    a.growth.without_gain = false;
    a.population.top_bin_share = Some(0.1);
    a
}

fn codes(a: &RunAnalysis) -> Vec<VerdictCode> {
    verdict_for(a).iter().map(|v| v.code).collect()
}

#[test]
fn a_quiet_run_has_no_clear_limit() {
    let v = verdict_for(&quiet());
    assert_eq!(v.len(), 1);
    assert_eq!(v[0].code, VerdictCode::NoClearLimit);
    assert_eq!(v[0].message, "no clear limit found");
}

#[test]
fn still_improving_by_late_stall_or_rising_trend() {
    let mut a = quiet();
    a.stall_generation = 94; // 94 < 0.95 * 99 = 94.05
    assert!(!codes(&a).contains(&VerdictCode::StillImproving));
    a.stall_generation = 95;
    let v = verdict_for(&a);
    assert_eq!(v[0].code, VerdictCode::StillImproving);
    assert!(v[0].message.contains("95 of 99"), "{}", v[0].message);
    assert!(v[0].message.contains("--resume"));
    let mut b = quiet();
    b.trend.last_third = Some(up());
    assert_eq!(codes(&b), vec![VerdictCode::StillImproving]);
}

#[test]
fn noise_limited_needs_a_flat_trend_and_a_noise_signal() {
    let mut a = quiet();
    a.trend.last_third = Some(flat());
    a.noise.change_sd = Some(0.09); // < 2 * 0.05
    assert!(!codes(&a).contains(&VerdictCode::NoiseLimited));
    a.noise.change_sd = Some(0.10);
    let v = verdict_for(&a);
    assert!(v.iter().any(|v| v.code == VerdictCode::NoiseLimited
        && v.message.contains("0.1000")
        && v.message.contains("--matches-per-genome")));
    // The curse gap alone also counts, from one held-out standard error.
    let mut b = quiet();
    b.trend.last_third = Some(flat());
    b.curse.reeval_minus_heldout = Some(HeldoutGap {
        value: 0.05,
        in_heldout_se: Some(0.99),
    });
    assert!(!codes(&b).contains(&VerdictCode::NoiseLimited));
    b.curse.reeval_minus_heldout = Some(HeldoutGap {
        value: 0.05,
        in_heldout_se: Some(1.0),
    });
    assert!(codes(&b).contains(&VerdictCode::NoiseLimited));
    // A rising trend is progress above the noise.
    b.trend.last_third = Some(up());
    assert!(!codes(&b).contains(&VerdictCode::NoiseLimited));
}

#[test]
fn pool_limited_needs_a_rising_and_a_flat_opponent() {
    let mut a = quiet();
    a.trend.last_third = Some(flat());
    a.opponents = vec![
        opponent("Easy", TrendClass::Rising),
        opponent("Hard", TrendClass::Flat),
    ];
    let v = verdict_for(&a);
    let pool = v
        .iter()
        .find(|v| v.code == VerdictCode::PoolLimited)
        .unwrap();
    assert!(pool.message.contains("rising: Easy") && pool.message.contains("flat: Hard"));
    assert!(
        !codes(&a).contains(&VerdictCode::ModelLimited),
        "an opponent rises"
    );
    a.opponents = vec![
        opponent("Easy", TrendClass::Rising),
        opponent("Worse", TrendClass::Falling),
    ];
    assert!(!codes(&a).contains(&VerdictCode::PoolLimited));
    a.opponents = vec![opponent("Hard", TrendClass::Flat)];
    assert!(!codes(&a).contains(&VerdictCode::PoolLimited));
}

#[test]
fn model_limited_is_flat_with_nothing_rising_and_not_noise() {
    let mut a = quiet();
    a.trend.last_third = Some(flat());
    a.opponents = vec![opponent("Hard", TrendClass::Flat)];
    assert_eq!(codes(&a), vec![VerdictCode::ModelLimited]);
    assert!(verdict_for(&a)[0].message.contains("--features hidden"));
    a.noise.change_sd = Some(0.2);
    assert!(!codes(&a).contains(&VerdictCode::ModelLimited));
    a.noise.change_sd = Some(0.05);
    a.opponents.push(opponent("Easy", TrendClass::Rising));
    assert!(!codes(&a).contains(&VerdictCode::ModelLimited));
}

#[test]
fn growth_without_gain_fires_from_the_growth_flag() {
    let mut a = quiet();
    a.growth.without_gain = true;
    a.growth.first_hidden_nodes = 1;
    a.growth.last_hidden_nodes = 7;
    let v = verdict_for(&a);
    assert_eq!(v[0].code, VerdictCode::GrowthWithoutGain);
    assert!(v[0].message.contains("1 -> 7"), "{}", v[0].message);
    a.growth.without_gain = false;
    assert!(!codes(&a).contains(&VerdictCode::GrowthWithoutGain));
}

#[test]
fn converged_from_half_the_genomes_in_the_top_bin() {
    let mut a = quiet();
    a.population.top_bin_share = Some(0.49);
    assert!(!codes(&a).contains(&VerdictCode::Converged));
    a.population.top_bin_share = Some(0.5);
    let v = verdict_for(&a);
    assert_eq!(v[0].code, VerdictCode::Converged);
    assert!(v[0].message.contains("50%"));
    a.population.top_bin_share = None;
    assert!(!codes(&a).contains(&VerdictCode::Converged));
}

#[test]
fn stop_early_when_the_stall_is_early_and_still_improving_does_not_apply() {
    let mut a = quiet();
    a.stall_generation = 60; // 60 > 0.6 * 99 = 59.4
    assert!(!codes(&a).contains(&VerdictCode::StopEarly));
    a.stall_generation = 59;
    let v = verdict_for(&a);
    assert_eq!(v[0].code, VerdictCode::StopEarly);
    assert!(
        v[0].message.contains("last 40 generations"),
        "{}",
        v[0].message
    );
    // An early stall with a rising end is still improving, not early.
    a.trend.last_third = Some(up());
    assert_eq!(codes(&a), vec![VerdictCode::StillImproving]);
}

#[test]
fn several_rules_fire_together_in_order_of_importance() {
    let mut a = quiet();
    a.stall_generation = 40;
    a.trend.last_third = Some(flat());
    a.noise.change_sd = Some(0.2);
    a.population.top_bin_share = Some(0.8);
    assert_eq!(
        codes(&a),
        vec![
            VerdictCode::NoiseLimited,
            VerdictCode::Converged,
            VerdictCode::StopEarly
        ]
    );
}

#[test]
fn analyse_attaches_the_verdict() {
    let gens: Vec<Gen> = (0..12).map(|g| gen(g, 0.02 * f64::from(g))).collect();
    let a = analyse(&run_of(&gens));
    assert_eq!(a.verdict[0].code, VerdictCode::StillImproving);
    let json = serde_json::to_value(&a).unwrap();
    assert_eq!(json["verdict"][0]["code"], "still_improving");
}

// ---- loading --------------------------------------------------------------

#[test]
fn a_resumed_run_keeps_the_latest_event_per_generation() {
    let mut text = String::new();
    let mut push = |s: String| {
        text.push_str(&s);
        text.push('\n');
    };
    push(line(&start_event()));
    for g in 0..3 {
        push(line(&Event::Generation(Box::new(gen(g, 0.1).event()))));
    }
    push("not json".into());
    push(String::new());
    push(line(&start_event()));
    push(line(&Event::Generation(Box::new(gen(2, 0.9).event()))));
    push(line(&Event::Generation(Box::new(gen(3, 0.4).event()))));
    push(r#"{"type":"generation","genera"#.into()); // truncated tail
    let run = parse_run("resumed".into(), &text);
    assert_eq!(run.generations.len(), 4);
    assert!((run.generations[2].champion.reeval.mean - 0.9).abs() < 1e-12);
    assert_eq!(run.generations[3].generation, 3);
    assert_eq!(run.skipped_lines, 2);
    assert!(run.start.is_some() && run.end.is_none());
    assert_eq!(analyse(&run).skipped_lines, 2);
}

#[test]
fn a_file_a_run_dir_and_a_set_dir_load_with_labels() {
    let root = temp_dir("load");
    let gens = [gen(0, 0.0), gen(1, 0.1)];
    for name in ["run-b", "run-a"] {
        fs::create_dir_all(root.join(name)).unwrap();
        write_events(&root.join(name).join("events.jsonl"), &gens);
    }
    fs::create_dir_all(root.join("not-a-run")).unwrap();
    fs::create_dir_all(root.join("run-empty")).unwrap();
    fs::write(root.join("run-empty").join("events.jsonl"), "").unwrap();
    write_events(&root.join("loose.jsonl"), &gens);

    let set = load_runs(&root).unwrap();
    let labels: Vec<&str> = set.iter().map(|r| r.label.as_str()).collect();
    assert_eq!(
        labels,
        vec!["loose", "run-a", "run-b"],
        "sorted, loose .jsonl files included, empty runs left out"
    );

    let one = load_runs(&root.join("run-a")).unwrap();
    assert_eq!((one.len(), one[0].label.as_str()), (1, "run-a"));
    let file = load_runs(&root.join("run-b").join("events.jsonl")).unwrap();
    assert_eq!(
        file[0].label, "run-b",
        "events.jsonl is labelled by its directory"
    );
    let loose = load_runs(&root.join("loose.jsonl")).unwrap();
    assert_eq!(loose[0].label, "loose");
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn empty_corrupt_or_missing_input_is_an_error_naming_the_path() {
    let root = temp_dir("errors");
    let empty = root.join("empty.jsonl");
    fs::write(&empty, "").unwrap();
    let corrupt = root.join("corrupt.jsonl");
    fs::write(&corrupt, "garbage\n{\"type\":\"nope\"}\n\u{0}\u{1}").unwrap();
    let only_start = root.join("start.jsonl");
    fs::write(&only_start, line(&start_event())).unwrap();
    for path in [&empty, &corrupt, &only_start, &root.join("sub")] {
        let error = load_runs(path).unwrap_err();
        assert!(
            error.to_string().contains(&path.display().to_string()),
            "{error}"
        );
    }
    assert!(matches!(load_runs(&empty), Err(AnalysisError::NoEvents(_))));
    assert!(matches!(
        load_runs(&root.join("missing")),
        Err(AnalysisError::Io(_))
    ));
    // A directory without runs.
    let dir = root.join("nothing");
    fs::create_dir_all(&dir).unwrap();
    assert!(matches!(load_runs(&dir), Err(AnalysisError::NoEvents(_))));
    fs::remove_dir_all(&root).unwrap();
}

// ---- real data ------------------------------------------------------------

#[test]
fn stall_generation_reproduces_the_phase_13_script_on_a0_seed1() {
    // `analyse.py` prints 109 for A0 seed 1 (120 generations). Skipped
    // silently where the run is not on disk.
    let dir = Path::new("/home/sebi/.cache/phase13/A0/seed1");
    if !dir.join("events.jsonl").is_file() {
        return;
    }
    let runs = load_runs(dir).unwrap();
    let a = analyse(&runs[0]);
    assert_eq!(a.generations, 120);
    assert_eq!(a.stall_generation, 109);
    assert_eq!(runs[0].skipped_lines, 0);
}
