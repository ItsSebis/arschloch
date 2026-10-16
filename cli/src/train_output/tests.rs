use sim::training::events::{ChampionStats, Complexity, FitnessStats, OpponentStat};
use sim::training::{GenerationEvent, RunEnd, RunStart, ScoreStat, SCHEMA_VERSION};

use super::observer::*;
use super::render::*;

fn stat(mean: f64) -> ScoreStat {
    ScoreStat {
        mean,
        std_error: 0.021,
        matches: 100,
        placements: vec![1, 2, 3, 4],
        spread: None,
        fitness: None,
        by_setting: None,
    }
}

fn event(is_new_best: bool) -> GenerationEvent {
    GenerationEvent {
        generation: 42,
        elapsed_secs: 100.0,
        generation_secs: 2.0,
        rounds_evaluated: 168_000,
        total_rounds: 1_000_000,
        rounds_per_sec: 84_000.0,
        fitness: FitnessStats {
            best: 0.412,
            mean: 0.188,
            median: 0.2,
            min: -0.5,
            std_dev: 0.1,
            histogram: vec![0; 10],
            skill_term: None,
        },
        champion: ChampionStats {
            train_fitness: 0.412,
            skill_term: None,
            reeval: stat(0.397),
            heldout: None,
            training_rank: 0,
            hidden_nodes: 7,
            enabled_connections: 23,
            genome_file: "gen-0042.json".into(),
            is_new_best,
        },
        opponents: vec![
            OpponentStat {
                name: "LowestLegal".into(),
                score: stat(0.61),
            },
            OpponentStat {
                name: "Adaptive".into(),
                score: stat(-0.05),
            },
        ],
        hall_of_fame: vec![],
        hall_score: None,
        species: vec![],
        compatibility_threshold: 0.5,
        complexity: Complexity {
            mean_hidden_nodes: 1.0,
            mean_enabled_connections: 20.0,
            innovation_count: 30,
        },
        racing: None,
        timings: None,
    }
}

#[test]
fn a_row_shows_every_headline_number() {
    let row = render_row(&event(false), Some(2470.0), false);
    for expected in [
        "42", "+0.412", "+0.188", "+0.397", "±0.021", "7/23", "+0.61", "-0.05", "84k", "0:41:10",
    ] {
        assert!(row.contains(expected), "{expected} missing from: {row}");
    }
    assert!(!row.ends_with('*'));
}

#[test]
fn a_new_best_is_marked_and_a_missing_eta_is_dashes() {
    let row = render_row(&event(true), None, false);
    assert!(row.ends_with(" *"), "{row}");
    assert!(row.contains("--:--:--"));
}

#[test]
fn the_header_has_one_column_per_opponent() {
    let header = render_header(3, false);
    assert!(header.contains("o1") && header.contains("o2") && header.contains("o3"));
    assert!(!header.contains("o4"));
    assert!(header.contains("ETA") && header.contains("champion"));
}

#[test]
fn the_hall_of_fame_column_appears_only_when_the_hall_is_enabled() {
    assert!(!render_header(2, false).contains("hof"));
    assert!(render_header(2, true).contains("hof"));
    let mut with = event(false);
    with.hall_score = Some(stat(0.37));
    assert!(render_row(&with, None, true).contains("+0.37"));
    let without = event(false);
    assert!(
        render_row(&without, None, true).contains('–'),
        "an empty hall shows a dash"
    );
    assert!(!render_row(&with, None, false).contains("+0.37"));
    assert_eq!(
        render_header(2, true).split_whitespace().count(),
        render_header(2, false).split_whitespace().count() + 1
    );
}

#[test]
fn durations_and_rates_format_compactly() {
    assert_eq!(format_duration(0.0), "0:00:00");
    assert_eq!(format_duration(61.4), "0:01:01");
    assert_eq!(format_duration(3_725.0), "1:02:05");
    assert_eq!(format_duration(-5.0), "0:00:00");
    assert_eq!(format_rate(842.0), "842");
    assert_eq!(format_rate(1_234.0), "1.2k");
    assert_eq!(format_rate(84_000.0), "84k");
    assert_eq!(format_rate(2_500_000.0), "2.5M");
}

#[test]
fn the_summary_names_the_best_generation_and_how_to_play_it() {
    let end = RunEnd {
        generations_completed: 100,
        best_generation: Some(87),
        best_reeval: Some(stat(0.652)),
        best_heldout: Some(stat(0.640)),
        elapsed_secs: 5025.0,
    };
    let text = render_summary(&end, std::path::Path::new("runs/a"));
    assert!(text.contains("100 generations in 1:23:45"), "{text}");
    assert!(
        text.contains("generation 87") && text.contains("+0.640"),
        "{text}"
    );
    assert!(text.contains("held-out"), "{text}");
    assert!(
        !text.contains("+0.652"),
        "the selection score is not the headline: {text}"
    );
    let played = std::path::Path::new("runs/a").join("best.json");
    assert!(
        text.contains(&format!("neat:{}", played.display())),
        "{text}"
    );
    let none = RunEnd {
        best_generation: None,
        best_reeval: None,
        ..end
    };
    assert!(!render_summary(&none, std::path::Path::new("x")).contains("best champion"));
}

#[test]
fn a_set_stopped_early_says_so_and_marks_the_unequal_run() {
    let end = |done| RunEnd {
        generations_completed: done,
        best_generation: Some(3),
        best_reeval: Some(stat(0.5)),
        best_heldout: Some(stat(0.4)),
        elapsed_secs: 10.0,
    };
    let ends = vec![
        ("run-1".to_owned(), end(100)),
        ("run-2".to_owned(), end(45)),
    ];
    let done = render_set_summary(&ends[..1], 10.0, None);
    assert!(done.starts_with("set finished: 1 runs"), "{done}");
    let text = render_set_summary(&ends, 20.0, Some(100));
    assert!(text.starts_with("set stopped early: 2 runs"), "{text}");
    assert!(
        text.contains("stopped after 45 of 100 generations"),
        "{text}"
    );
    assert!(
        text.contains("not comparable") && text.contains("--resume"),
        "{text}"
    );
    assert_eq!(
        text.matches("stopped after").count(),
        1,
        "only the last run is marked: {text}"
    );
}

#[test]
fn the_banner_names_the_pass_rule() {
    let mut start = RunStart {
        schema_version: SCHEMA_VERSION,
        config: sim_config(),
        opponents: vec![],
        feature_names: vec![],
        resumed_from_generation: None,
        warm_started_from: None,
    };
    let text = render_banner(&start, std::path::Path::new("runs/a"));
    assert!(text.contains("pass rule final"), "{text}");
    start.config.pass_rule = sim::PassRule::Free;
    let text = render_banner(&start, std::path::Path::new("runs/a"));
    assert!(text.contains("pass rule free"), "{text}");
}

#[test]
fn the_banner_lists_opponents_and_flags_a_resume() {
    let start = RunStart {
        schema_version: SCHEMA_VERSION,
        config: sim_config(),
        opponents: vec!["LowestLegal".into(), "Adaptive(x)".into()],
        feature_names: vec![],
        resumed_from_generation: Some(12),
        warm_started_from: None,
    };
    let text = render_banner(&start, std::path::Path::new("runs/a"));
    assert!(text.contains("o1=LowestLegal  o2=Adaptive(x)"), "{text}");
    assert!(text.contains("resumed from generation 12"), "{text}");
    assert!(text.contains("4 players"), "{text}");
}

#[test]
fn progress_counts_finished_generations_and_the_share_of_this_one() {
    assert!((progress_fraction(0, 0, 50, 10) - 0.0).abs() < 1e-12);
    assert!((progress_fraction(4, 25, 50, 10) - 0.45).abs() < 1e-12);
    assert!((progress_fraction(9, 50, 50, 10) - 1.0).abs() < 1e-12);
    assert!(
        progress_fraction(0, 5, 0, 10).abs() < 1e-12,
        "no genomes: no progress, no NaN"
    );
    assert!(
        progress_fraction(0, 5, 10, 0).abs() < 1e-12,
        "unknown total: no NaN"
    );
    assert!(
        (progress_fraction(30, 1, 1, 10) - 1.0).abs() < 1e-12,
        "never above 1"
    );
}

#[test]
fn a_progress_line_is_the_documented_format() {
    let line = progress_line(0.4512, 4, 25, 50, 10, None);
    assert_eq!(line, "progress: 0.4512 generation 5/10, genomes 25/50");
}

#[test]
fn in_a_set_the_fraction_and_text_include_the_run_position() {
    // Run 2 of 4, halfway through it: (1 + 0.5) / 4.
    assert!((set_fraction(1, 4, 0.5) - 0.375).abs() < 1e-12);
    assert!(set_fraction(0, 4, 0.0).abs() < 1e-12);
    assert!(
        (set_fraction(3, 4, 1.0) - 1.0).abs() < 1e-12,
        "the last run ends at 1"
    );
    assert!(
        (set_fraction(0, 0, 0.3) - 0.3).abs() < 1e-12,
        "no set: unchanged"
    );
    let line = progress_line(0.375, 4, 25, 50, 10, Some((2, 4)));
    assert_eq!(
        line,
        "progress: 0.3750 run 2/4 generation 5/10, genomes 25/50"
    );
    // The window reads the number right after the prefix, with or without the run.
    let number = line
        .strip_prefix("progress: ")
        .unwrap()
        .split(' ')
        .next()
        .unwrap();
    assert!(number.parse::<f64>().is_ok());
}

/// The line the window reads must be what the trainer prints: this is the
/// contract `gui/src/progress.rs` parses (`progress: FRACTION TEXT`).
#[test]
fn the_line_starts_the_way_the_window_expects() {
    let line = progress_line(1.0, 9, 50, 50, 10, None);
    let rest = line.strip_prefix("progress: ").unwrap();
    let number = rest.split(' ').next().unwrap();
    assert!(number.parse::<f64>().is_ok(), "{line}");
}

fn sim_config() -> sim::training::TrainConfig {
    sim::training::TrainConfig {
        seed: 0,
        player_count: 4,
        deck: sim::training::DeckChoice::Single,
        duplicate_rule: sim::training::DuplicateChoice::FirstDealtWins,
        rounds_per_match: 8,
        matches_per_genome: 100,
        reeval_matches: 200,
        generations: 100,
        neat: neat::NeatConfig::default(),
        opponent_specs: vec![],
        champion_candidates: 1,
        hall_of_fame_size: 0,
        hall_of_fame_interval: 5,
        skill_weight: 0.0,
        features: sim::FeatureSet::Base,
        fitness: sim::training::FitnessKind::Mean,
        racing_top: 0.0,
        racing_matches: 0,
        player_counts: Vec::new(),
        decks: Vec::new(),
        pass_rule: sim::PassRule::default(),
        exchange_rule: sim::ExchangeRule::default(),
    }
}
