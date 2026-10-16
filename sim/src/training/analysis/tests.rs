//! Unit tests of the run analysis with small hand-computed fixtures.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::json;

use super::super::config::test_support::sample;
use super::super::events::{Event, RunEnd, RunStart, SCHEMA_VERSION};
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
    /// The event as the trainer writes it (every field).
    fn json(&self) -> serde_json::Value {
        json!({
            "type": "generation",
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
        })
    }

    fn event(&self) -> LeanGeneration {
        serde_json::from_value(self.json()).unwrap()
    }

    fn line(&self) -> String {
        self.json().to_string()
    }
}

fn run_of(gens: &[Gen]) -> RunData {
    RunData {
        label: "fixture".into(),
        parent: None,
        start: None,
        starts: 0,
        generations: gens.iter().map(Gen::event).collect(),
        end: None,
        skipped_lines: 0,
    }
}

fn line(event: &Event) -> String {
    serde_json::to_string(event).unwrap()
}

fn start_event(resumed_from: Option<u32>) -> Event {
    Event::RunStart(Box::new(RunStart {
        schema_version: SCHEMA_VERSION,
        config: sample(),
        opponents: vec![],
        feature_names: vec![],
        resumed_from_generation: resumed_from,
        warm_started_from: None,
    }))
}

fn end_event(best_generation: u32) -> Event {
    Event::RunEnd(RunEnd {
        generations_completed: 4,
        best_generation: Some(best_generation),
        best_reeval: Some(serde_json::from_value(stat(0.5, 0.02)).unwrap()),
        best_heldout: Some(serde_json::from_value(stat(0.4, 0.02)).unwrap()),
        elapsed_secs: 1.0,
    })
}

fn lean_end(reeval: (f64, f64), heldout: (f64, f64), best: u32) -> LeanEnd {
    LeanEnd {
        best_generation: Some(best),
        best_reeval: Some(LeanScore {
            mean: reeval.0,
            std_error: reeval.1,
        }),
        best_heldout: Some(LeanScore {
            mean: heldout.0,
            std_error: heldout.1,
        }),
    }
}

fn temp_dir(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("arschloch-analysis-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_events(path: &Path, gens: &[Gen]) {
    let mut text = line(&start_event(None));
    for g in gens {
        text.push('\n');
        text.push_str(&g.line());
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
    run.end = Some(lean_end((0.3, 0.05), (0.2, 0.05), 2));
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
    // 0.1 / sqrt(0.05^2 + 0.05^2) = sqrt(2).
    assert!((gap.in_combined_se.unwrap() - 2.0_f64.sqrt()).abs() < 1e-9);
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
    // Three generations are enough for the spread of two changes.
    let three = analyse(&run_of(&[gen(0, 0.0), gen(1, 0.1), gen(2, 0.5)]));
    assert!(three.noise.change_sd.is_some());
}

#[test]
fn a_gap_without_any_standard_error_has_no_se_units() {
    let mut run = run_of(&[gen(0, 0.0), gen(1, 0.1)]);
    run.end = Some(lean_end((0.3, 0.0), (0.2, 0.0), 1));
    let gap = analyse(&run).curse.reeval_minus_heldout.unwrap();
    assert_eq!((gap.in_heldout_se, gap.in_combined_se), (None, None));
}

#[test]
fn stall_generation_follows_the_smoothed_curve_and_is_a_generation_number() {
    // Window 10: the first ten means are 0,0,...,0 then jump to 1 and
    // stay: smoothed climbs 0.1 per generation until 1.0 at index 19.
    // With se 0.05 the band starts at 0.95, reached at index 19 (0.95 at
    // index 18 is 0.9; index 19 is 1.0).
    let gens: Vec<Gen> = (0..30)
        .map(|g| gen(g, if g < 10 { 0.0 } else { 1.0 }))
        .collect();
    let a = analyse(&run_of(&gens));
    // smoothed[i] = (i - 9) / 10 for 10 <= i <= 19; >= 0.95 first at 19.
    assert_eq!((a.stall_index, a.stall_generation), (19, 19));
    assert!(!a.stall_fallback);
    assert!((a.peak_smoothed - 1.0).abs() < 1e-12);
    let s = series(&run_of(&gens));
    assert!((s.smoothed[14] - 0.5).abs() < 1e-12);
}

#[test]
fn generation_numbers_that_do_not_start_at_zero_or_have_gaps_are_reported_as_numbers() {
    // The same curve with the generations 100, 102, ..., 158 (a resumed run
    // that logged every second generation): index 19 is generation 138.
    let gens: Vec<Gen> = (0..30)
        .map(|i| gen(100 + 2 * i, if i < 10 { 0.0 } else { 1.0 }))
        .collect();
    let a = analyse(&run_of(&gens));
    assert_eq!(a.generations, 30);
    assert_eq!((a.stall_index, a.stall_generation), (19, 138));
    assert_eq!((a.peak_index, a.peak_generation), (19, 138));
    assert_eq!(a.last_generation, 158);
    assert!((a.final_smoothed - 1.0).abs() < 1e-12);
    let marker = series(&run_of(&gens)).generation;
    assert_eq!(marker[a.stall_index], a.stall_generation);
}

#[test]
fn a_run_that_ends_below_its_peak_reports_the_fallback_stall() {
    // Ones from generation 5 to 14, zeros before and after: the smoothed
    // peak 1.0 is at index 14 and the final smoothed value is 0.
    let gens: Vec<Gen> = (0..30)
        .map(|g| gen(g, if (5..15).contains(&g) { 1.0 } else { 0.0 }))
        .collect();
    let a = analyse(&run_of(&gens));
    assert!(a.stall_fallback);
    assert_eq!((a.stall_index, a.stall_generation), (29, 29));
    assert_eq!((a.peak_index, a.peak_generation), (14, 14));
    assert!(a.final_smoothed.abs() < 1e-12);
    let codes: Vec<_> = a.verdict.iter().map(|v| v.code).collect();
    assert!(codes.contains(&VerdictCode::Declining), "{codes:?}");
    assert!(!codes.contains(&VerdictCode::StillImproving));
    let declining = a
        .verdict
        .iter()
        .find(|v| v.code == VerdictCode::Declining)
        .unwrap();
    assert!(
        declining
            .message
            .contains("peaked at generation 14 (1.000), now 0.000"),
        "{}",
        declining.message
    );
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
fn opponents_are_classified_by_their_last_third_slope_interval() {
    let a = analyse(&nine_with_opponents());
    let by = |n: &str| a.opponents.iter().find(|o| o.name == n).unwrap();
    assert_eq!(by("Rising").class, Some(TrendClass::Rising));
    assert_eq!(by("Flat").class, Some(TrendClass::Flat));
    assert_eq!(by("Falling").class, Some(TrendClass::Falling));
    // Slopes are per 100 generations.
    assert!((by("Rising").overall_slope.unwrap().value - 10.0).abs() < 1e-9);
    assert!((by("Rising").last_third_slope.unwrap().value - 10.0).abs() < 1e-9);
    // Thirds are generations 0..3 and 6..9.
    assert!((by("Rising").first_third_mean.unwrap() - 0.1).abs() < 1e-12);
    assert!((by("Rising").last_third_mean.unwrap() - 0.7).abs() < 1e-12);
    assert!((by("Falling").last_third_mean.unwrap() + 0.35).abs() < 1e-12);
    // A flat champion curve contains zero: a perfect constant has an
    // interval of width zero at zero, which is neither above nor below.
    assert!(a.trend.last_third.unwrap().contains_zero());
}

#[test]
fn an_opponent_that_rose_early_but_is_flat_at_the_end_is_flat() {
    // Rises 0.1 per generation until generation 7, constant afterwards: the
    // whole-run slope is positive, the last third (generations 8..12) flat.
    let gens: Vec<Gen> = (0..12)
        .map(|g: u32| Gen {
            opponents: vec![("Beaten", 0.1 * f64::from(g.min(7)))],
            ..gen(g, 0.0)
        })
        .collect();
    let a = analyse(&run_of(&gens));
    let o = &a.opponents[0];
    assert!(o.overall_slope.unwrap().is_positive());
    assert!(o.last_third_slope.unwrap().contains_zero());
    assert_eq!(o.class, Some(TrendClass::Flat));
    assert!((o.last_third_mean.unwrap() - 0.7).abs() < 1e-12);
}

#[test]
fn an_opponent_absent_from_the_last_third_has_no_last_mean_or_class() {
    let gens: Vec<Gen> = (0..9)
        .map(|g: u32| Gen {
            opponents: if g < 6 {
                vec![("Early", 0.2)]
            } else {
                Vec::new()
            },
            ..gen(g, 0.0)
        })
        .collect();
    let o = &analyse(&run_of(&gens)).opponents[0];
    assert!((o.first_third_mean.unwrap() - 0.2).abs() < 1e-12);
    assert_eq!((o.last_third_mean, o.class), (None, None));
}

#[test]
fn trend_is_per_100_generations_on_the_generation_number() {
    let gens: Vec<Gen> = (0..12).map(|g| gen(g, 0.01 * f64::from(g))).collect();
    let a = analyse(&run_of(&gens));
    let overall = a.trend.overall.unwrap();
    assert!((overall.value - 1.0).abs() < 1e-9, "{overall:?}");
    assert!(a.trend.last_third.unwrap().is_positive());
    // Generations 100, 105, ...: 0.01 per index is 0.002 per generation.
    let gapped: Vec<Gen> = (0..12)
        .map(|i| gen(100 + 5 * i, 0.01 * f64::from(i)))
        .collect();
    let overall = analyse(&run_of(&gapped)).trend.overall.unwrap();
    assert!((overall.value - 0.2).abs() < 1e-9, "{overall:?}");
}

/// 18 generations (last third 12..18, three first and three last of it) with
/// a flat, alternating curve; `nodes` gives the hidden nodes per generation.
fn growth_of(nodes: fn(u32) -> usize, mean: fn(u32) -> f64, n: u32) -> Growth {
    let gens: Vec<Gen> = (0..n)
        .map(|g| Gen {
            nodes: nodes(g),
            ..gen(g, mean(g))
        })
        .collect();
    analyse(&run_of(&gens)).growth
}

#[test]
fn growth_without_gain_compares_the_node_means_at_both_ends_of_the_last_third() {
    let flat = |g: u32| if g.is_multiple_of(2) { 0.1 } else { 0.0 };
    // Generations 12..18: nodes 5 5 5 | 6 6 6 (g / 3 + 1).
    let growth = growth_of(|g| 1 + g as usize / 3, flat, 18);
    assert_eq!(
        (growth.first_hidden_nodes, growth.last_hidden_nodes),
        (1, 6)
    );
    assert_eq!((growth.first_connections, growth.last_connections), (5, 22));
    assert_eq!(growth.third_window, 3);
    assert!((growth.third_start_nodes - 5.0).abs() < 1e-12);
    assert!((growth.third_end_nodes - 6.0).abs() < 1e-12);
    assert!(growth.without_gain);
    // A single node gained in the very last generation is not growth: the
    // means over three generations are 4 and 4.333 > 4, so it still counts,
    // but one lost node and then regained does not.
    let wobble = growth_of(|g| if g == 16 { 3 } else { 4 }, flat, 18);
    assert!((wobble.third_start_nodes - 4.0).abs() < 1e-12);
    assert!((wobble.third_end_nodes - (11.0 / 3.0)).abs() < 1e-12);
    assert!(!wobble.without_gain);
    // No growth in the last third.
    assert!(!growth_of(|g| 1 + g.min(11) as usize, flat, 18).without_gain);
    // Growth with a rising curve is not "without gain".
    assert!(!growth_of(|g| 1 + g as usize / 3, |g| 0.1 * f64::from(g), 18).without_gain);
}

#[test]
fn growth_windows_shrink_with_a_short_last_third() {
    let flat = |g: u32| if g.is_multiple_of(2) { 0.1 } else { 0.0 };
    // 9 generations: a third of 3 compares generation 6 with generation 8.
    let growth = growth_of(|g| 1 + g as usize / 2, flat, 9);
    assert_eq!(growth.third_window, 1);
    assert!((growth.third_start_nodes - 4.0).abs() < 1e-12);
    assert!((growth.third_end_nodes - 5.0).abs() < 1e-12);
    assert!(growth.without_gain);
    // 12 generations: a third of 4 uses two each.
    assert_eq!(growth_of(|_| 1, flat, 12).third_window, 2);
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

fn racing_run(population: Option<u64>) -> RunData {
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
    run.start = population.map(|p| {
        serde_json::from_value(json!({
            "config": {"neat": {"population_size": p}, "matches_per_genome": 2},
            "resumed_from_generation": null,
        }))
        .unwrap()
    });
    run
}

#[test]
fn racing_and_timings_are_totals_and_optional() {
    let a = analyse(&run_of(&[gen(0, 0.0), gen(1, 0.0)]));
    assert!(a.racing.is_none() && a.timings.is_none());

    let a = analyse(&racing_run(None));
    let racing = a.racing.unwrap();
    assert_eq!((racing.generations, racing.extra_matches), (2, 40));
    assert!((racing.mean_shift + 0.2).abs() < 1e-12);
    assert_eq!(racing.extra_share, None, "no run_start, no share");
    assert!((a.timings.unwrap().training_evaluation - 3.0).abs() < 1e-12);

    // 2 generations x 10 genomes x 2 matches = 40 training matches.
    let share = analyse(&racing_run(Some(10))).racing.unwrap().extra_share;
    assert!((share.unwrap() - 0.5).abs() < 1e-12);
    // A start event without a population size gives no share.
    assert_eq!(
        analyse(&racing_run(Some(0))).racing.unwrap().extra_share,
        None
    );
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

fn down() -> Slope {
    Slope {
        value: -2.0,
        low: -3.0,
        high: -1.0,
    }
}

fn opponent(name: &str, class: TrendClass) -> OpponentAnalysis {
    OpponentAnalysis {
        name: name.into(),
        first_third_mean: Some(0.0),
        last_third_mean: Some(0.0),
        overall_slope: None,
        last_third_slope: None,
        class: Some(class),
    }
}

/// An analysis that fires no rule: 100 generations numbered 1000..=1099, a
/// stall and a peak at index 80 (between the 60% and 95% limits), no
/// last-third trend, quiet everything.
fn quiet() -> RunAnalysis {
    let mut a = analyse(&run_of(&[gen(0, 0.0), gen(1, 0.0)]));
    a.generations = 100;
    a.last_generation = 1099;
    a.peak_index = 80;
    a.peak_generation = 1080;
    a.peak_smoothed = 0.3;
    a.final_smoothed = 0.3;
    a.stall_index = 80;
    a.stall_generation = 1080;
    a.stall_fallback = false;
    a.best_generation = None;
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

#[allow(clippy::unnecessary_wraps)] // the field is an `Option`, tests assign it directly
fn combined_gap(in_combined_se: f64) -> Option<HeldoutGap> {
    Some(HeldoutGap {
        value: 0.05,
        in_heldout_se: Some(5.0),
        in_combined_se: Some(in_combined_se),
    })
}

#[test]
fn a_quiet_run_has_no_clear_limit() {
    let v = verdict_for(&quiet());
    assert_eq!(v.len(), 1);
    assert_eq!(v[0].code, VerdictCode::NoClearLimit);
    assert_eq!(v[0].message, "no clear limit found");
}

#[test]
fn still_improving_when_the_smoothed_peak_is_at_the_end_or_the_last_third_rises() {
    let mut a = quiet();
    a.peak_index = 94; // 94 < 0.95 * 99 = 94.05
    assert!(!codes(&a).contains(&VerdictCode::StillImproving));
    a.peak_index = 95;
    a.peak_generation = 1095;
    let v = verdict_for(&a);
    assert_eq!(v[0].code, VerdictCode::StillImproving);
    assert!(
        v[0].message.contains("generation 1095 of 1099"),
        "{}",
        v[0].message
    );
    assert!(v[0].message.contains("--resume"));
    let mut b = quiet();
    b.trend.last_third = Some(up());
    assert_eq!(codes(&b), vec![VerdictCode::StillImproving]);
}

#[test]
fn the_stall_being_late_alone_is_not_still_improving() {
    // The old rule: a stall index at the end. A falling curve stalls "at the
    // end" by python's fallback; that is not improvement.
    let mut a = quiet();
    a.stall_index = 99;
    a.stall_generation = 1099;
    assert!(!codes(&a).contains(&VerdictCode::StillImproving));
}

#[test]
fn still_improving_never_fires_on_a_falling_curve() {
    // The peak is at the end of the smoothed curve but the last third falls.
    let mut a = quiet();
    a.peak_index = 99;
    a.trend.last_third = Some(down());
    let c = codes(&a);
    assert!(!c.contains(&VerdictCode::StillImproving), "{c:?}");
    assert!(c.contains(&VerdictCode::Declining));
    // A rising interval cannot be combined with a final value below the peak.
    let mut b = quiet();
    b.trend.last_third = Some(up());
    b.stall_fallback = true;
    let c = codes(&b);
    assert!(!c.contains(&VerdictCode::StillImproving), "{c:?}");
    assert!(c.contains(&VerdictCode::Declining));
}

#[test]
fn declining_when_the_last_third_falls_or_the_run_ended_below_its_peak() {
    let mut a = quiet();
    a.trend.last_third = Some(down());
    a.peak_generation = 1040;
    a.peak_smoothed = 0.3;
    a.final_smoothed = 0.2;
    let v = verdict_for(&a);
    assert_eq!(v[0].code, VerdictCode::Declining);
    assert!(
        v[0].message
            .contains("peaked at generation 1040 (0.300), now 0.200"),
        "{}",
        v[0].message
    );
    assert!(v[0].message.contains("not the best one"));
    assert!(v[0].message.contains("-2.0000 [-3.0000, -1.0000]"));
    // The fallback flag alone is enough.
    let mut b = quiet();
    b.stall_fallback = true;
    assert_eq!(codes(&b), vec![VerdictCode::Declining]);
    // A flat end at the peak is not declining.
    assert!(!codes(&quiet()).contains(&VerdictCode::Declining));
}

#[test]
fn noise_limited_needs_a_flat_trend_and_a_noise_signal() {
    let mut a = quiet();
    a.trend.last_third = Some(flat());
    a.noise.change_sd = Some(0.10); // not > 2 * 0.05
    assert!(!codes(&a).contains(&VerdictCode::NoiseLimited));
    a.noise.change_sd = Some(0.1001);
    let v = verdict_for(&a);
    assert!(v.iter().any(|v| v.code == VerdictCode::NoiseLimited
        && v.message.contains("0.1001")
        && v.message.contains("not distinguishable from zero")
        && v.message.contains("--matches-per-genome")));
    // The curse gap alone also counts, from two combined standard errors.
    let mut b = quiet();
    b.trend.last_third = Some(flat());
    b.curse.reeval_minus_heldout = combined_gap(1.99);
    assert!(!codes(&b).contains(&VerdictCode::NoiseLimited));
    b.curse.reeval_minus_heldout = combined_gap(2.0);
    assert!(codes(&b).contains(&VerdictCode::NoiseLimited));
    // A rising trend is progress above the noise.
    b.trend.last_third = Some(up());
    assert!(!codes(&b).contains(&VerdictCode::NoiseLimited));
}

#[test]
fn noise_limited_needs_a_positive_single_se() {
    let mut a = quiet();
    a.trend.last_third = Some(flat());
    a.noise = Noise {
        single_se: 0.0,
        change_sd: Some(0.0),
    };
    assert!(!codes(&a).contains(&VerdictCode::NoiseLimited));
    a.noise.change_sd = Some(0.2);
    assert!(!codes(&a).contains(&VerdictCode::NoiseLimited));
}

#[test]
fn rules_two_to_five_and_stop_early_wait_for_a_run_that_is_still_improving() {
    // The curve peaks at the very end but the last-third interval is flat:
    // still improving, so no limit is named.
    let mut a = quiet();
    a.peak_index = 99;
    a.trend.last_third = Some(flat());
    a.noise.change_sd = Some(0.5);
    a.curse.reeval_minus_heldout = combined_gap(3.0);
    a.opponents = vec![
        opponent("Easy", TrendClass::Rising),
        opponent("Hard", TrendClass::Flat),
    ];
    a.growth.without_gain = true;
    a.stall_index = 10;
    a.stall_generation = 1010;
    assert_eq!(codes(&a), vec![VerdictCode::StillImproving]);
    // Without the peak at the end the same numbers name the limits.
    a.peak_index = 80;
    let c = codes(&a);
    for code in [
        VerdictCode::NoiseLimited,
        VerdictCode::PoolLimited,
        VerdictCode::GrowthWithoutGain,
        VerdictCode::StopEarly,
    ] {
        assert!(c.contains(&code), "{code:?} in {c:?}");
    }
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
    assert!(pool.message.contains("last third"));
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
fn growth_without_gain_fires_from_the_growth_flag_and_quotes_the_node_means() {
    let mut a = quiet();
    a.growth.without_gain = true;
    a.growth.third_start_nodes = 5.0;
    a.growth.third_end_nodes = 6.5;
    a.growth.third_window = 3;
    let v = verdict_for(&a);
    assert_eq!(v[0].code, VerdictCode::GrowthWithoutGain);
    assert!(
        v[0].message
            .contains("5.0 over the first 3 generations of the last third -> 6.5 over its last 3"),
        "{}",
        v[0].message
    );
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
fn stop_early_names_generation_numbers_and_the_noise_unit() {
    let mut a = quiet();
    a.stall_index = 60; // 60 > 0.6 * 99 = 59.4
    a.stall_generation = 1060;
    assert!(!codes(&a).contains(&VerdictCode::StopEarly));
    a.stall_index = 59;
    a.stall_generation = 1059;
    a.peak_smoothed = 0.3;
    let v = verdict_for(&a);
    assert_eq!(v[0].code, VerdictCode::StopEarly);
    assert!(
        v[0].message.contains(
            "after generation 1059 the smoothed score stayed within one noise unit (0.050) of its peak 0.300"
        ),
        "{}",
        v[0].message
    );
    assert!(
        v[0].message
            .contains("last 40 generations (up to generation 1099)"),
        "{}",
        v[0].message
    );
    // An early stall with a rising end is still improving, not early.
    a.trend.last_third = Some(up());
    assert_eq!(codes(&a), vec![VerdictCode::StillImproving]);
}

#[test]
fn stop_early_is_suppressed_by_a_later_best_generation_or_a_positive_estimate() {
    let mut a = quiet();
    a.stall_index = 40;
    a.stall_generation = 1040;
    assert!(codes(&a).contains(&VerdictCode::StopEarly));
    a.best_generation = Some(1040);
    assert!(
        codes(&a).contains(&VerdictCode::StopEarly),
        "a best generation at the stall is not later"
    );
    a.best_generation = Some(1041);
    assert!(!codes(&a).contains(&VerdictCode::StopEarly));
    a.best_generation = None;
    // Flat interval, but the point estimate still points up.
    a.trend.last_third = Some(Slope {
        value: 0.3,
        low: -1.0,
        high: 1.6,
    });
    assert!(!codes(&a).contains(&VerdictCode::StopEarly));
    a.trend.last_third = Some(Slope {
        value: -0.3,
        low: -1.0,
        high: 0.4,
    });
    assert!(codes(&a).contains(&VerdictCode::StopEarly));
}

#[test]
fn several_rules_fire_together_in_order_of_importance() {
    let mut a = quiet();
    a.stall_index = 40;
    a.stall_generation = 1040;
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
    assert!(json["stall_generation"].is_u64() && json["stall_index"].is_u64());
}

#[test]
fn a_falling_curve_is_declining_not_still_improving() {
    // Rises for 10 generations, then falls steadily for 30.
    let gens: Vec<Gen> = (0..40)
        .map(|g| {
            gen(
                g,
                0.1 * f64::from(g.min(10)) - 0.02 * f64::from(g.saturating_sub(10)),
            )
        })
        .collect();
    let a = analyse(&run_of(&gens));
    let c: Vec<_> = a.verdict.iter().map(|v| v.code).collect();
    assert!(a.trend.last_third.unwrap().is_negative());
    assert!(c.contains(&VerdictCode::Declining), "{c:?}");
    assert!(!c.contains(&VerdictCode::StillImproving), "{c:?}");
}

// ---- loading --------------------------------------------------------------

#[test]
fn a_resumed_run_keeps_the_latest_event_per_generation() {
    let mut text = String::new();
    let mut push = |s: String| {
        text.push_str(&s);
        text.push('\n');
    };
    push(line(&start_event(None)));
    for g in 0..3 {
        push(gen(g, 0.1).line());
    }
    push(line(&end_event(1)));
    push("not json".into());
    push(String::new());
    push(line(&start_event(Some(2))));
    push(gen(2, 0.9).line());
    push(gen(3, 0.4).line());
    push(r#"{"type":"generation","genera"#.into()); // truncated tail
    let run = parse_run("resumed".into(), &text);
    assert_eq!(run.generations.len(), 4);
    assert!((run.generations[2].champion.reeval.mean - 0.9).abs() < 1e-12);
    assert_eq!(run.generations[3].generation, 3);
    assert_eq!(run.skipped_lines, 2);
    assert_eq!(run.starts, 2);
    assert_eq!(
        run.start.as_ref().unwrap().resumed_from_generation,
        Some(2),
        "the last run_start is kept"
    );
    assert!(
        run.end.is_none(),
        "the run_end before the resume no longer describes the end"
    );
    assert_eq!(analyse(&run).skipped_lines, 2);
}

#[test]
fn a_run_end_after_the_last_start_is_kept() {
    let mut text = String::new();
    for l in [
        line(&start_event(None)),
        gen(0, 0.1).line(),
        line(&end_event(0)),
        line(&start_event(Some(1))),
        gen(1, 0.2).line(),
        line(&end_event(1)),
    ] {
        text.push_str(&l);
        text.push('\n');
    }
    let run = parse_run("r".into(), &text);
    assert_eq!(run.end.unwrap().best_generation, Some(1));
}

#[test]
fn events_from_older_versions_with_missing_fields_still_load() {
    // Only what the analysis needs: no fitness, species, timings,
    // complexity, matches, placements, hall of fame, run_start config.
    let text = [
        r#"{"type":"run_start","schema_version":1}"#,
        r#"{"type":"generation","generation":4,"champion":{"reeval":{"mean":0.1,"std_error":0.02}}}"#,
        r#"{"type":"generation","generation":5,"champion":{"reeval":{"mean":0.3},"hidden_nodes":2},"extra":"ignored"}"#,
        r#"{"type":"run_end","best_generation":5}"#,
    ]
    .join("\n");
    let run = parse_run("old".into(), &text);
    assert_eq!((run.generations.len(), run.skipped_lines), (2, 0));
    let a = analyse(&run);
    assert_eq!((a.last_generation, a.generations), (5, 2));
    assert_eq!(a.best_generation, Some(5));
    assert_eq!(a.population.top_bin_share, None);
    assert_eq!(a.growth.last_hidden_nodes, 2);
    assert!(a.curse.train_minus_reeval.abs() < 1e-12, "no train fitness");
    assert!(run.start.unwrap().population_size().is_none());
}

#[test]
fn a_file_a_run_dir_and_a_set_dir_load_with_labels_and_parents() {
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
    assert_eq!(set.skipped_runs.len(), 0);
    let labels: Vec<&str> = set.runs.iter().map(|r| r.label.as_str()).collect();
    assert_eq!(
        labels,
        vec!["loose", "run-a", "run-b"],
        "sorted, loose .jsonl files included, empty runs left out"
    );
    let root_name = root.file_name().unwrap().to_string_lossy().into_owned();
    assert!(set
        .runs
        .iter()
        .all(|r| r.parent.as_deref() == Some(root_name.as_str())));

    let one = load_runs(&root.join("run-a")).unwrap().runs;
    assert_eq!((one.len(), one[0].label.as_str()), (1, "run-a"));
    assert_eq!(one[0].parent.as_deref(), Some(root_name.as_str()));
    let file = load_runs(&root.join("run-b").join("events.jsonl"))
        .unwrap()
        .runs;
    assert_eq!(
        file[0].label, "run-b",
        "events.jsonl is labelled by its directory"
    );
    assert_eq!(file[0].parent.as_deref(), Some(root_name.as_str()));
    let loose = load_runs(&root.join("loose.jsonl")).unwrap().runs;
    assert_eq!(loose[0].label, "loose");
    assert_eq!(loose[0].parent.as_deref(), Some(root_name.as_str()));
    fs::remove_dir_all(&root).unwrap();
}

fn labelled(label: &str, parent: &str) -> RunData {
    RunData {
        label: label.into(),
        parent: Some(parent.into()),
        ..run_of(&[gen(0, 0.0)])
    }
}

fn labels(runs: &[RunData]) -> Vec<&str> {
    runs.iter().map(|r| r.label.as_str()).collect()
}

#[test]
fn equal_labels_or_several_arguments_get_the_parent_directory_in_front() {
    // Distinct labels stay as they are...
    let mut runs = vec![labelled("seed1", "A0"), labelled("seed2", "A0")];
    qualify_labels(&mut runs, false);
    assert_eq!(labels(&runs), ["seed1", "seed2"]);
    // ... unless several path arguments were given.
    qualify_labels(&mut runs, true);
    assert_eq!(labels(&runs), ["A0/seed1", "A0/seed2"]);
    // Equal labels collide: everyone is prefixed.
    let mut runs = vec![
        labelled("seed1", "A0"),
        labelled("seed1", "A2"),
        labelled("seed2", "A0"),
    ];
    qualify_labels(&mut runs, false);
    assert_eq!(labels(&runs), ["A0/seed1", "A2/seed1", "A0/seed2"]);
    // Still equal after that (same parent): numbered.
    let mut runs = vec![labelled("x", "P"), labelled("x", "P"), labelled("x", "P")];
    qualify_labels(&mut runs, false);
    assert_eq!(labels(&runs), ["P/x", "P/x (2)", "P/x (3)"]);
    // No parent known: nothing to prefix.
    let mut runs = vec![
        RunData {
            parent: None,
            ..labelled("x", "")
        },
        labelled("x", "P"),
    ];
    qualify_labels(&mut runs, false);
    assert_eq!(labels(&runs), ["x", "P/x"]);
}

#[cfg(unix)]
#[test]
fn an_unreadable_run_of_a_set_is_skipped_and_reported() {
    use std::os::unix::fs::PermissionsExt;
    let root = temp_dir("unreadable");
    let gens = [gen(0, 0.0), gen(1, 0.1)];
    for name in ["run-a", "run-b"] {
        fs::create_dir_all(root.join(name)).unwrap();
        write_events(&root.join(name).join("events.jsonl"), &gens);
    }
    let broken = root.join("run-b").join("events.jsonl");
    fs::set_permissions(&broken, fs::Permissions::from_mode(0o000)).unwrap();
    if fs::read(&broken).is_ok() {
        // Running as a user that ignores permissions: nothing to test.
        fs::remove_dir_all(&root).unwrap();
        return;
    }
    let loaded = load_runs(&root).unwrap();
    assert_eq!(labels(&loaded.runs), ["run-a"]);
    assert_eq!(loaded.skipped_runs.len(), 1, "{:?}", loaded.skipped_runs);
    assert!(loaded.skipped_runs[0].contains("run-b"));
    // Reading only the broken run is an error, as is a set of nothing else.
    assert!(matches!(
        load_runs(&root.join("run-b")),
        Err(AnalysisError::Io(_))
    ));
    fs::remove_dir_all(root.join("run-a")).unwrap();
    let error = load_runs(&root).unwrap_err();
    assert!(matches!(error, AnalysisError::Io(_)), "{error}");
    fs::set_permissions(&broken, fs::Permissions::from_mode(0o644)).unwrap();
    fs::remove_dir_all(&root).unwrap();
}

#[cfg(unix)]
#[test]
fn within_a_root_runs_that_resolve_outside_are_dropped_and_paths_are_relative() {
    let root = temp_dir("within");
    let outside = temp_dir("within-outside");
    let gens = [gen(0, 0.0), gen(1, 0.1)];
    fs::create_dir_all(root.join("inside")).unwrap();
    write_events(&root.join("inside").join("events.jsonl"), &gens);
    write_events(&outside.join("events.jsonl"), &gens);
    std::os::unix::fs::symlink(&outside, root.join("escape-dir")).unwrap();
    std::os::unix::fs::symlink(outside.join("events.jsonl"), root.join("escape-file.jsonl"))
        .unwrap();
    fs::create_dir_all(root.join("sneaky")).unwrap();
    std::os::unix::fs::symlink(
        outside.join("events.jsonl"),
        root.join("sneaky/events.jsonl"),
    )
    .unwrap();

    // Without a root everything is read.
    assert_eq!(load_runs(&root).unwrap().runs.len(), 4);
    // Within the root only the genuine run is.
    let loaded = load_runs_within(&root, Some(&root)).unwrap();
    assert_eq!(labels(&loaded.runs), ["inside"]);
    // A target that is itself outside is an error without absolute paths.
    let error = load_runs_within(&outside, Some(&root)).unwrap_err();
    let text = error.to_string();
    assert!(!text.contains(&outside.display().to_string()), "{text}");
    assert!(!text.contains(&root.display().to_string()), "{text}");

    // An empty directory inside the root is named relative to it.
    fs::create_dir_all(root.join("empty")).unwrap();
    let error = load_runs_within(&root.join("empty"), Some(&root)).unwrap_err();
    assert_eq!(
        error,
        AnalysisError::NoEvents("empty".to_owned()),
        "relative to the root"
    );
    let missing = load_runs_within(&root.join("missing"), Some(&root)).unwrap_err();
    assert!(!missing.to_string().contains(&root.display().to_string()));
    fs::remove_dir_all(&root).unwrap();
    fs::remove_dir_all(&outside).unwrap();
}

#[test]
fn empty_corrupt_or_missing_input_is_an_error_naming_the_path() {
    let root = temp_dir("errors");
    let empty = root.join("empty.jsonl");
    fs::write(&empty, "").unwrap();
    let corrupt = root.join("corrupt.jsonl");
    fs::write(&corrupt, "garbage\n{\"type\":\"nope\"}\n\u{0}\u{1}").unwrap();
    let only_start = root.join("start.jsonl");
    fs::write(&only_start, line(&start_event(None))).unwrap();
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

/// Twelve generations (61 to 72) of the Phase 13 run A0 seed 5, reduced to
/// the fields the analysis reads, with the run's real `run_end` (best
/// generation 83, outside the excerpt). The numbers below were computed with
/// `docs/baselines/phase-13/analyse.py` (`load_events`, `stall_generation`,
/// `smooth`) and plain Python OLS on this very file:
/// stall index 4, peak of the smoothed curve at index 11, gap
/// `0.648750 - 0.641875 = 0.0068750000000000755`, combined error
/// `hypot(0.016241192906090426, 0.011872654724511426)`, OLS slopes
/// 0.2555361305361269 (whole run) and -0.1291666666666691 (last four
/// generations) per 100 generations.
const A0_SEED5_WINDOW: &str = include_str!("fixtures/a0-seed5-window.jsonl");

#[test]
fn the_committed_phase_13_excerpt_reproduces_the_python_numbers() {
    let run = parse_run("a0-seed5".into(), A0_SEED5_WINDOW);
    assert_eq!((run.generations.len(), run.skipped_lines), (12, 0));
    assert_eq!(run.generations[0].generation, 61, "numbers are not indices");
    let a = analyse(&run);
    // analyse.py: stall index 4 -> generation 61 + 4.
    assert_eq!((a.stall_index, a.stall_generation), (4, 65));
    assert!(!a.stall_fallback);
    assert_eq!((a.peak_index, a.peak_generation), (11, 72));
    assert_eq!(a.last_generation, 72);
    assert!((a.noise.single_se - 0.017_928_100_874_336_418).abs() < 1e-12);
    assert!((a.peak_smoothed - 0.630_333_333_333_333_3).abs() < 1e-12);
    let gap = a.curse.reeval_minus_heldout.unwrap();
    assert!((gap.value - 0.006_875_000_000_000_075_5).abs() < 1e-12);
    assert!((gap.in_combined_se.unwrap() - 0.341_732_777_295_759_07).abs() < 1e-9);
    assert!((a.trend.overall.unwrap().value - 0.255_536_130_536_126_9).abs() < 1e-9);
    assert!((a.trend.last_third.unwrap().value + 0.129_166_666_666_669_1).abs() < 1e-9);
    assert_eq!(a.best_generation, Some(83));
}

/// `analyse.py` stall generation and winner's-curse gap of the Phase 13
/// A0 and A2 seeds 1..=10 (120 generations each), printed by
/// `stall_generation` and `end[best_reeval] - end[best_heldout]`.
const A0_PYTHON: [(usize, f64); 10] = [
    (109, 0.030_833_333_333_332_935),
    (119, 0.039_166_666_666_666_46),
    (119, 0.024_166_666_666_666_448),
    (119, -0.003_333_333_333_332_855),
    (119, 0.006_875_000_000_000_075_5),
    (92, 0.025_625_000_000_000_12),
    (90, 0.036_041_666_666_666_75),
    (97, 0.065_833_333_333_333_08),
    (114, 0.013_333_333_333_333_086),
    (84, 0.051_666_666_666_666_416),
];

/// Skipped where the local runs are not on disk.
#[test]
fn stall_generation_and_gap_reproduce_the_phase_13_script_on_the_a0_seeds() {
    let base = Path::new("/home/sebi/.cache/phase13/A0");
    if !base.join("seed1/events.jsonl").is_file() {
        return;
    }
    for (i, (stall, gap)) in A0_PYTHON.iter().enumerate() {
        let dir = base.join(format!("seed{}", i + 1));
        let run = load_runs(&dir).unwrap().runs.remove(0);
        let a = analyse(&run);
        assert_eq!(a.generations, 120, "seed {}", i + 1);
        assert_eq!(a.stall_index, *stall, "seed {}", i + 1);
        assert_eq!(a.stall_generation, u32::try_from(*stall).unwrap());
        assert!(
            (a.curse.reeval_minus_heldout.unwrap().value - gap).abs() < 1e-12,
            "seed {}",
            i + 1
        );
        assert_eq!(run.skipped_lines, 0);
    }
}

/// The real runs the first verdict called "still improving" although their
/// curves had stopped or fallen.
#[test]
fn the_real_runs_that_stalled_at_the_end_are_not_still_improving() {
    let cases: [(&str, &[usize]); 2] = [
        ("/home/sebi/.cache/phase13/A0", &[2, 3, 4, 5]),
        ("/home/sebi/.cache/phase13/A2", &[3, 4, 5, 8]),
    ];
    for (base, seeds) in cases {
        let base = Path::new(base);
        if !base.join("seed1/events.jsonl").is_file() {
            return;
        }
        for seed in seeds {
            let run = load_runs(&base.join(format!("seed{seed}")))
                .unwrap()
                .runs
                .remove(0);
            let a = analyse(&run);
            assert!(
                !a.verdict
                    .iter()
                    .any(|v| v.code == VerdictCode::StillImproving),
                "{base:?} seed {seed}: {:?}",
                a.verdict
            );
        }
    }
    let pc = Path::new("/home/sebi/Documents/Code/arschloch/runs/PC/4th-ontopOf3rd.jsonl");
    if pc.is_file() {
        let a = analyse(&load_runs(pc).unwrap().runs[0]);
        assert!(!a
            .verdict
            .iter()
            .any(|v| v.code == VerdictCode::StillImproving));
    }
}
