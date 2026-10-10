//! End-to-end tests of the trainer: the files a run leaves behind,
//! exact resume, independence from thread count, crash repair and, most
//! importantly, that training actually improves play.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use neat::NeatConfig;
use sim::training::{
    DeckChoice, DuplicateChoice, Event, Opponent, TrainConfig, TrainError, TrainObserver, Trainer,
};
use sim::{LowestLegal, NeatStrategy, RandomLegal};

fn dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("arschloch-train-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    path
}

fn opponents() -> Vec<Opponent> {
    vec![
        Opponent {
            name: "LowestLegal".into(),
            strategy: Arc::new(LowestLegal),
        },
        Opponent {
            name: "RandomLegal".into(),
            strategy: Arc::new(RandomLegal),
        },
    ]
}

fn config(generations: u32) -> TrainConfig {
    TrainConfig {
        seed: 3,
        player_count: 4,
        deck: DeckChoice::Single,
        duplicate_rule: DuplicateChoice::FirstDealtWins,
        pass_rule: engine::PassRule::default(),
        exchange_rule: engine::ExchangeRule::default(),
        rounds_per_match: 4,
        matches_per_genome: 8,
        reeval_matches: 12,
        generations,
        neat: NeatConfig {
            population_size: 16,
            ..NeatConfig::default()
        },
        opponent_specs: vec!["lowest-legal".into(), "random-legal".into()],
        champion_candidates: 1,
        hall_of_fame_size: 0,
        hall_of_fame_interval: 5,
        skill_weight: 0.0,
    }
}

fn events(dir: &Path) -> Vec<Event> {
    fs::read_to_string(dir.join("events.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).expect("every line is an event"))
        .collect()
}

fn checkpoint_population(dir: &Path) -> serde_json::Value {
    let text = fs::read_to_string(dir.join("checkpoint.json")).unwrap();
    serde_json::from_str::<serde_json::Value>(&text).unwrap()["population"].clone()
}

#[derive(Default)]
struct Recorder {
    starts: usize,
    progress: usize,
    last_progress: (usize, usize),
    generations: Vec<u32>,
    finished: bool,
}

impl TrainObserver for Recorder {
    fn on_start(&mut self, _: &sim::training::RunStart) {
        self.starts += 1;
    }
    fn on_eval_progress(&mut self, _: u32, done: usize, total: usize) {
        self.progress += 1;
        self.last_progress = (done, total);
    }
    fn on_generation(&mut self, event: &sim::training::GenerationEvent) {
        self.generations.push(event.generation);
    }
    fn on_finish(&mut self, _: &sim::training::RunEnd) {
        self.finished = true;
    }
}

#[test]
fn a_short_run_leaves_every_artifact_and_notifies_the_observer() {
    let run = dir("artifacts");
    let mut observer = Recorder::default();
    let end = Trainer::new(config(3), opponents(), &run)
        .unwrap()
        .run(&mut observer)
        .unwrap();

    assert_eq!(end.generations_completed, 3);
    assert_eq!(
        (
            observer.starts,
            observer.generations.clone(),
            observer.finished
        ),
        (1, vec![0, 1, 2], true)
    );
    assert_eq!(
        observer.last_progress,
        (16, 16),
        "progress reaches the whole population"
    );
    assert!(
        observer.progress >= 3 * 2,
        "several progress steps per generation"
    );

    for file in [
        "config.json",
        "events.jsonl",
        "checkpoint.json",
        "best.json",
        "gen-0000.json",
        "gen-0001.json",
        "gen-0002.json",
    ] {
        assert!(run.join(file).exists(), "{file} missing");
    }
    assert!(!run.join("checkpoint.json.tmp").exists());

    let log = events(&run);
    assert_eq!(log.len(), 5, "start, 3 generations, end");
    let Event::Generation(first) = &log[1] else {
        panic!("expected a generation event")
    };
    assert_eq!(first.generation, 0);
    assert_eq!(first.opponents.len(), 2);
    assert_eq!(first.opponents[0].name, "LowestLegal");
    assert_eq!(first.fitness.histogram.iter().sum::<u32>(), 16);
    assert_eq!(first.champion.reeval.matches, 12);
    assert!(
        first.champion.is_new_best,
        "the first champion is the first best"
    );
    assert_eq!(first.champion.genome_file, "gen-0000.json");
    // 16 genomes x 8 matches, the champion's fixed-match scores (12 matches
    // each: the mixed pool and 2 opponents), and, because the first
    // champion is a new best, 24 held-out matches; 4 rounds each.
    // ... plus the one match played to record the champion's decisions.
    assert_eq!(first.rounds_evaluated, 4 * (16 * 8 + 12 * 3 + 24 + 1));
    assert!(matches!(log[4], Event::RunEnd(_)));

    assert!(NeatStrategy::from_file(&run.join("best.json")).is_ok());
    assert!(NeatStrategy::from_file(&run.join("gen-0002.json")).is_ok());
    // A new best also records some of the champion's real decisions.
    let decisions: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(run.join("decisions/gen-0000.json")).unwrap())
            .unwrap();
    assert_eq!(decisions["generation"], 0);
    assert_ne!(
        *decisions["decisions"].as_array().unwrap(),
        [] as [serde_json::Value; 0]
    );
    fs::remove_dir_all(&run).unwrap();
}

#[test]
fn resuming_is_identical_to_never_stopping() {
    let straight = dir("straight");
    Trainer::new(config(4), opponents(), &straight)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();

    let split = dir("split");
    Trainer::new(config(2), opponents(), &split)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let mut resumed = Trainer::resume(&split, opponents(), Some(4)).unwrap();
    assert_eq!(resumed.completed_generations(), 2);
    let mut observer = Recorder::default();
    let end = resumed.run(&mut observer).unwrap();
    assert_eq!(end.generations_completed, 4);
    assert_eq!(observer.generations, vec![2, 3]);

    assert_eq!(
        checkpoint_population(&split),
        checkpoint_population(&straight)
    );
    for file in ["best.json", "gen-0002.json", "gen-0003.json"] {
        assert_eq!(
            fs::read_to_string(split.join(file)).unwrap(),
            fs::read_to_string(straight.join(file)).unwrap(),
            "{file}"
        );
    }
    // Each generation appears once in the resumed log, in order.
    let generations: Vec<u32> = events(&split)
        .iter()
        .filter_map(|e| {
            if let Event::Generation(g) = e {
                Some(g.generation)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(generations, vec![0, 1, 2, 3]);
    fs::remove_dir_all(&straight).unwrap();
    fs::remove_dir_all(&split).unwrap();
}

#[test]
fn results_do_not_depend_on_the_thread_count() {
    let run_with = |threads: usize, name: &str| {
        let run = dir(name);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            Trainer::new(config(3), opponents(), &run)
                .unwrap()
                .run(&mut Recorder::default())
                .unwrap();
        });
        let population = checkpoint_population(&run);
        let best = fs::read_to_string(run.join("best.json")).unwrap();
        fs::remove_dir_all(&run).unwrap();
        (population, best)
    };
    assert_eq!(run_with(1, "threads1"), run_with(4, "threads4"));
}

#[test]
fn a_crash_between_the_event_and_the_checkpoint_is_repaired_on_resume() {
    let run = dir("crash");
    Trainer::new(config(2), opponents(), &run)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    // Simulate dying after generation 2's event was logged but before its
    // checkpoint: duplicate the last generation event with generation 2.
    let log = events(&run);
    let Event::Generation(last) = log
        .iter()
        .rev()
        .find(|e| matches!(e, Event::Generation(_)))
        .unwrap()
        .clone()
    else {
        unreachable!()
    };
    let mut phantom = *last;
    phantom.generation = 2;
    let line = serde_json::to_string(&Event::Generation(Box::new(phantom))).unwrap();
    let mut text = fs::read_to_string(run.join("events.jsonl")).unwrap();
    text.push_str(&line);
    text.push('\n');
    fs::write(run.join("events.jsonl"), text).unwrap();

    Trainer::resume(&run, opponents(), Some(3))
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let generations: Vec<u32> = events(&run)
        .iter()
        .filter_map(|e| {
            if let Event::Generation(g) = e {
                Some(g.generation)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(
        generations,
        vec![0, 1, 2],
        "the phantom was replaced by the real generation 2"
    );
    fs::remove_dir_all(&run).unwrap();
}

#[test]
fn resuming_with_a_different_pool_or_no_run_is_refused() {
    let run = dir("mismatch");
    Trainer::new(config(1), opponents(), &run)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let wrong = vec![Opponent {
        name: "GreedyHighest".into(),
        strategy: Arc::new(RandomLegal),
    }];
    let error = Trainer::resume(&run, wrong, None).err().unwrap();
    assert!(matches!(error, TrainError::Mismatch(_)), "{error}");
    let error = Trainer::new(config(1), opponents(), &run).err().unwrap();
    assert!(
        matches!(error, TrainError::Config(_)),
        "a run is never overwritten: {error}"
    );
    let error = Trainer::resume(&dir("nothing-here"), opponents(), None)
        .err()
        .unwrap();
    assert!(matches!(error, TrainError::Checkpoint(_)), "{error}");
    fs::remove_dir_all(&run).unwrap();
}

#[test]
fn invalid_setups_are_refused_before_anything_is_written() {
    let run = dir("invalid");
    let bad = TrainConfig {
        matches_per_genome: 0,
        pass_rule: engine::PassRule::default(),
        exchange_rule: engine::ExchangeRule::default(),
        ..config(1)
    };
    assert!(matches!(
        Trainer::new(bad, opponents(), &run).err().unwrap(),
        TrainError::Config(_)
    ));
    let few = vec![opponents().remove(0)];
    assert!(matches!(
        Trainer::new(config(1), few, &run).err().unwrap(),
        TrainError::Config(_)
    ));
    assert!(!run.join("checkpoint.json").exists());
    let _ = fs::remove_dir_all(&run);
}

#[test]
fn training_improves_play() {
    let run = dir("improves");
    let config = TrainConfig {
        matches_per_genome: 14,
        reeval_matches: 40,
        neat: NeatConfig {
            population_size: 30,
            ..NeatConfig::default()
        },
        ..config(10)
    };
    Trainer::new(config, opponents(), &run)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let generations: Vec<_> = events(&run)
        .into_iter()
        .filter_map(|e| {
            if let Event::Generation(g) = e {
                Some(g)
            } else {
                None
            }
        })
        .collect();
    let first = &generations[0];
    let last = &generations[generations.len() - 1];
    // Random networks start badly (most play poorly); evolution must lift
    // the whole population, not just find one lucky genome.
    assert!(
        last.fitness.mean > first.fitness.mean + 0.25,
        "mean fitness {} -> {}",
        first.fitness.mean,
        last.fitness.mean
    );
    // And the best champion beats random play and mirrors-or-beats the
    // lowest-legal baseline on fresh matches.
    let best = generations
        .iter()
        .map(|g| &g.champion.reeval)
        .max_by(|a, b| a.mean.total_cmp(&b.mean))
        .unwrap();
    assert!(best.mean > 0.3, "best fresh-match score {}", best.mean);
    fs::remove_dir_all(&run).unwrap();
}

#[test]
fn a_run_killed_before_its_first_generation_can_be_resumed() {
    // Trainer::new returns, then the process dies during generation 0:
    // the directory must stay usable (not "no checkpoint" and not
    // "already holds a run").
    let straight = dir("early-straight");
    Trainer::new(config(3), opponents(), &straight)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();

    let killed = dir("early-killed");
    drop(Trainer::new(config(3), opponents(), &killed).unwrap());
    let mut resumed =
        Trainer::resume(&killed, opponents(), None).expect("resumable from the start");
    assert_eq!(resumed.completed_generations(), 0);
    resumed.run(&mut Recorder::default()).unwrap();
    assert_eq!(
        checkpoint_population(&killed),
        checkpoint_population(&straight)
    );
    fs::remove_dir_all(&straight).unwrap();
    fs::remove_dir_all(&killed).unwrap();
}

#[test]
fn a_checkpoint_from_another_feature_set_is_refused_without_touching_the_log() {
    let run = dir("featureset");
    Trainer::new(config(2), opponents(), &run)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let path = run.join("checkpoint.json");
    let mut value: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    value["feature_count"] = serde_json::json!(5);
    fs::write(&path, value.to_string()).unwrap();
    let log_before = fs::read_to_string(run.join("events.jsonl")).unwrap();

    let error = Trainer::resume(&run, opponents(), Some(3))
        .err()
        .expect("must be refused");
    assert!(matches!(error, TrainError::Mismatch(_)), "{error}");
    assert!(error.to_string().contains("feature"), "{error}");
    assert_eq!(
        fs::read_to_string(run.join("events.jsonl")).unwrap(),
        log_before
    );

    value["feature_count"] = serde_json::json!(sim::FEATURE_COUNT);
    value["feature_set_version"] = serde_json::json!(sim::FEATURE_SET_VERSION + 1);
    fs::write(&path, value.to_string()).unwrap();
    let error = Trainer::resume(&run, opponents(), Some(3))
        .err()
        .expect("must be refused");
    assert!(matches!(error, TrainError::Mismatch(_)), "{error}");
    fs::remove_dir_all(&run).unwrap();
}

#[test]
fn the_best_champion_is_confirmed_on_held_out_matches() {
    let run = dir("heldout");
    let end = Trainer::new(config(4), opponents(), &run)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let generations: Vec<_> = events(&run)
        .into_iter()
        .filter_map(|e| {
            if let Event::Generation(g) = e {
                Some(g)
            } else {
                None
            }
        })
        .collect();
    for generation in &generations {
        assert_eq!(
            generation.champion.heldout.is_some(),
            generation.champion.is_new_best,
            "generation {}: only a new best is confirmed",
            generation.generation
        );
    }
    let last_best = generations
        .iter()
        .rev()
        .find(|g| g.champion.is_new_best)
        .unwrap();
    assert_eq!(end.best_generation, Some(last_best.generation));
    assert_eq!(end.best_heldout, last_best.champion.heldout);
    assert!(end.best_heldout.is_some());
    fs::remove_dir_all(&run).unwrap();
}

fn generation_events(dir: &Path) -> Vec<sim::training::GenerationEvent> {
    events(dir)
        .into_iter()
        .filter_map(|e| match e {
            Event::Generation(g) => Some(*g),
            _ => None,
        })
        .collect()
}

#[test]
fn a_hall_of_fame_fills_every_interval_and_keeps_only_the_newest() {
    let run = dir("hall");
    let config = TrainConfig {
        hall_of_fame_size: 2,
        hall_of_fame_interval: 2,
        pass_rule: engine::PassRule::default(),
        exchange_rule: engine::ExchangeRule::default(),
        ..config(8)
    };
    Trainer::new(config, opponents(), &run)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let events = generation_events(&run);
    // A champion is admitted at the end of generations 2, 4, 6; the hall
    // holds the newest two and a generation sees the members admitted
    // before it.
    let seen: Vec<Vec<u32>> = events.iter().map(|g| g.hall_of_fame.clone()).collect();
    assert_eq!(
        seen,
        vec![
            vec![],
            vec![],
            vec![],
            vec![2],
            vec![2],
            vec![2, 4],
            vec![2, 4],
            vec![4, 6]
        ]
    );
    for generation in &events {
        assert_eq!(
            generation.hall_score.is_some(),
            !generation.hall_of_fame.is_empty(),
            "the champion is scored against the hall exactly when it has members"
        );
    }
    // Hall members are extra opponents but the per-opponent report stays
    // on the fixed pool, so the series stay comparable across generations.
    assert!(events.iter().all(|g| g.opponents.len() == 2));
    fs::remove_dir_all(&run).unwrap();
}

#[test]
fn resuming_with_a_hall_of_fame_still_equals_never_stopping() {
    let with_hall = |generations| TrainConfig {
        hall_of_fame_size: 2,
        hall_of_fame_interval: 2,
        pass_rule: engine::PassRule::default(),
        exchange_rule: engine::ExchangeRule::default(),
        ..config(generations)
    };
    let straight = dir("hall-straight");
    Trainer::new(with_hall(6), opponents(), &straight)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let split = dir("hall-split");
    Trainer::new(with_hall(3), opponents(), &split)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    Trainer::resume(&split, opponents(), Some(6))
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();

    let checkpoint = |dir: &Path| -> serde_json::Value {
        serde_json::from_str(&fs::read_to_string(dir.join("checkpoint.json")).unwrap()).unwrap()
    };
    let (a, b) = (checkpoint(&straight), checkpoint(&split));
    assert_eq!(a["population"], b["population"]);
    assert_eq!(a["hall_of_fame"], b["hall_of_fame"]);
    assert_eq!(a["hall_of_fame"].as_array().unwrap().len(), 2);
    fs::remove_dir_all(&straight).unwrap();
    fs::remove_dir_all(&split).unwrap();
}

#[test]
fn a_checkpoint_from_before_the_hall_of_fame_still_resumes() {
    let run = dir("oldcheckpoint");
    Trainer::new(config(2), opponents(), &run)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let path = run.join("checkpoint.json");
    let mut value: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    value.as_object_mut().unwrap().remove("hall_of_fame");
    for key in [
        "champion_candidates",
        "hall_of_fame_size",
        "hall_of_fame_interval",
    ] {
        value["config"].as_object_mut().unwrap().remove(key);
    }
    fs::write(&path, value.to_string()).unwrap();
    let end = Trainer::resume(&run, opponents(), Some(3))
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    assert_eq!(end.generations_completed, 3);
    fs::remove_dir_all(&run).unwrap();
}

#[test]
fn the_champion_is_chosen_among_the_top_candidates_by_the_fixed_matches() {
    // With one candidate the training best is always the champion; with
    // several, the fixed-match score decides, so a champion can come from
    // below the training best (which is just a lucky sample).
    let single = dir("single");
    Trainer::new(config(6), opponents(), &single)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    assert!(generation_events(&single)
        .iter()
        .all(|g| g.champion.training_rank == 0));

    let several = dir("several");
    let config = TrainConfig {
        champion_candidates: 6,
        pass_rule: engine::PassRule::default(),
        exchange_rule: engine::ExchangeRule::default(),
        ..config(6)
    };
    Trainer::new(config, opponents(), &several)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let ranks: Vec<usize> = generation_events(&several)
        .iter()
        .map(|g| g.champion.training_rank)
        .collect();
    assert!(ranks.iter().all(|&r| r < 6), "{ranks:?}");
    assert!(ranks.iter().any(|&r| r > 0), "over six generations the fixed matches overrule the training best at least once: {ranks:?}");
    fs::remove_dir_all(&single).unwrap();
    fs::remove_dir_all(&several).unwrap();
}

#[test]
fn candidate_selection_never_picks_a_worse_champion_than_the_training_best() {
    // The training best is always among the candidates, so the chosen
    // champion's fixed-match score is at least that of the training best:
    // compare two runs that differ only in the candidate count (same seed,
    // so generation 0 evaluates identical genomes).
    let one = dir("cmp-one");
    let many = dir("cmp-many");
    Trainer::new(config(1), opponents(), &one)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let wide = TrainConfig {
        champion_candidates: 8,
        pass_rule: engine::PassRule::default(),
        exchange_rule: engine::ExchangeRule::default(),
        ..config(1)
    };
    Trainer::new(wide, opponents(), &many)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let a = generation_events(&one)[0].champion.reeval.mean;
    let b = generation_events(&many)[0].champion.reeval.mean;
    assert!(b >= a - 1e-12, "{b} < {a}");
    fs::remove_dir_all(&one).unwrap();
    fs::remove_dir_all(&many).unwrap();
}

#[test]
fn candidate_selection_costs_its_own_matches_and_is_accounted_for() {
    // 16 genomes x 8 matches; the 4 candidates ranked on 12 selection
    // matches each; then the champion measured on 12 fresh matches against
    // the mixed pool and 12 against each of the 2 opponents; plus, for a
    // new best, 24 held-out matches; 4 rounds each.
    let run = dir("accounting");
    let config = TrainConfig {
        champion_candidates: 4,
        pass_rule: engine::PassRule::default(),
        exchange_rule: engine::ExchangeRule::default(),
        ..config(1)
    };
    Trainer::new(config, opponents(), &run)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let first = &generation_events(&run)[0];
    // ... plus the one match played to record the champion's decisions.
    assert_eq!(
        first.rounds_evaluated,
        4 * (16 * 8 + 12 * (4 + 1 + 2) + 24 + 1)
    );
    fs::remove_dir_all(&run).unwrap();
}

fn run_start(dir: &Path) -> sim::training::RunStart {
    match events(dir).into_iter().next() {
        Some(Event::RunStart(start)) => *start,
        other => panic!("first event is not a run start: {other:?}"),
    }
}

#[test]
fn a_warm_started_run_begins_from_the_sources_final_population() {
    let source = dir("warm-source");
    Trainer::new(config(6), opponents(), &source)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let source_bytes = fs::read(source.join("checkpoint.json")).unwrap();
    let source_events = generation_events(&source);

    let warm_dir = dir("warm-child");
    let mut warm_config = config(2);
    warm_config.seed = 11;
    let mut warm = Trainer::new_from(warm_config.clone(), opponents(), &warm_dir, &source).unwrap();
    warm.run(&mut Recorder::default()).unwrap();
    let cold_dir = dir("warm-cold");
    Trainer::new(warm_config, opponents(), &cold_dir)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();

    let start = run_start(&warm_dir);
    assert_eq!(
        start.warm_started_from.as_deref(),
        Some(source.display().to_string().as_str())
    );
    let warm_events = generation_events(&warm_dir);
    assert_eq!(warm_events[0].generation, 0, "the new run counts from 0");
    assert!(
        warm_events[0].fitness.mean > generation_events(&cold_dir)[0].fitness.mean,
        "a warm start begins ahead of a cold one: {} vs {}",
        warm_events[0].fitness.mean,
        generation_events(&cold_dir)[0].fitness.mean
    );
    assert!(warm_events[0].fitness.mean > source_events[0].fitness.mean);
    assert_eq!(
        fs::read(source.join("checkpoint.json")).unwrap(),
        source_bytes,
        "the source run is only read"
    );
}

#[test]
fn a_warm_start_from_a_different_population_size_is_refused_and_leaves_nothing() {
    let source = dir("warm-size-source");
    Trainer::new(config(1), opponents(), &source)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let target = dir("warm-size-target");
    let mut bigger = config(1);
    bigger.neat.population_size = 20;
    let error = Trainer::new_from(bigger, opponents(), &target, &source)
        .err()
        .expect("size mismatch");
    assert!(matches!(error, TrainError::Mismatch(_)), "{error}");
    assert!(!target.exists(), "nothing is created for a refused start");
}

#[test]
fn a_warm_start_from_a_missing_run_is_a_clear_error() {
    let target = dir("warm-missing-target");
    let error = Trainer::new_from(config(1), opponents(), &target, &dir("warm-missing-source"))
        .err()
        .expect("no source");
    assert!(matches!(error, TrainError::Checkpoint(_)), "{error}");
    assert!(!target.exists());
}

/// Bit patterns of `(best, mean, median, min, std_dev, champion train
/// fitness, champion re-evaluation)` of `config(2)`'s two generations,
/// captured from the code before the skill weight existed.
const GOLDEN_W0: [[u64; 7]; 2] = [
    [
        13_820_796_656_462_157_140,
        13_827_681_065_267_538_602,
        13_828_677_955_810_055_508,
        13_830_366_805_670_319_445,
        4_597_716_784_971_618_455,
        13_820_796_656_462_157_140,
        13_823_048_456_275_842_388,
    ],
    [
        4_605_305_918_955_279_701,
        4_579_410_221_097_899_364,
        13_811_038_857_269_521_061,
        13_828_302_655_841_107_968,
        4_602_104_100_315_806_600,
        4_605_305_918_955_279_701,
        4_602_428_619_193_348_549,
    ],
];

#[test]
fn the_default_skill_weight_reproduces_the_fitness_from_before_it_existed() {
    let out = dir("golden-w0");
    Trainer::new(config(2), opponents(), &out)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let got: Vec<[u64; 7]> = generation_events(&out)
        .iter()
        .map(|e| {
            [
                e.fitness.best.to_bits(),
                e.fitness.mean.to_bits(),
                e.fitness.median.to_bits(),
                e.fitness.min.to_bits(),
                e.fitness.std_dev.to_bits(),
                e.champion.train_fitness.to_bits(),
                e.champion.reeval.mean.to_bits(),
            ]
        })
        .collect();
    assert_eq!(got, GOLDEN_W0);
    for text in ["events.jsonl", "config.json", "checkpoint.json"] {
        let content = fs::read_to_string(out.join(text)).unwrap();
        assert!(!content.contains("skill"), "{text} mentions the skill term");
    }
    fs::remove_dir_all(&out).unwrap();
}

fn weighted(generations: u32, weight: f64) -> TrainConfig {
    TrainConfig {
        skill_weight: weight,
        ..config(generations)
    }
}

#[test]
fn a_skill_weighted_run_completes_resumes_exactly_and_emits_the_additive_fields() {
    let straight = dir("skill-straight");
    Trainer::new(weighted(3, 0.5), opponents(), &straight)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let split = dir("skill-split");
    Trainer::new(weighted(2, 0.5), opponents(), &split)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let mut resumed = Trainer::resume(&split, opponents(), Some(3)).unwrap();
    assert!((resumed.config().skill_weight - 0.5).abs() < f64::EPSILON);
    resumed.run(&mut Recorder::default()).unwrap();
    assert_eq!(
        checkpoint_population(&split),
        checkpoint_population(&straight)
    );
    for e in generation_events(&split) {
        let stats = e.fitness.skill_term.expect("fitness skill term");
        assert!(stats.mean.is_finite() && stats.std_dev >= 0.0);
        assert!(e.champion.skill_term.is_some());
    }
    // The champion records are still plain mean role scores on fixed matches.
    let first = &generation_events(&straight)[0];
    assert!(first.champion.reeval.mean.abs() <= 1.0);
    fs::remove_dir_all(&straight).unwrap();
    fs::remove_dir_all(&split).unwrap();
}

#[test]
fn a_skill_weighted_run_does_not_depend_on_the_thread_count() {
    let run_with = |threads: usize, name: &str| {
        let run = dir(name);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            Trainer::new(weighted(2, 0.7), opponents(), &run)
                .unwrap()
                .run(&mut Recorder::default())
                .unwrap();
        });
        let population = checkpoint_population(&run);
        let events = fs::read_to_string(run.join("events.jsonl")).unwrap();
        // Wall-clock fields differ between runs; compare the fitness part.
        let fitness: Vec<String> = generation_events(&run)
            .iter()
            .map(|e| format!("{:?} {:?}", e.fitness, e.champion.skill_term))
            .collect();
        assert!(events.contains("skill_term"));
        fs::remove_dir_all(&run).unwrap();
        (population, fitness)
    };
    assert_eq!(run_with(1, "skill-threads1"), run_with(4, "skill-threads4"));
}

#[test]
fn a_nonzero_weight_changes_the_fitness_and_the_old_weight_zero_path_is_untouched() {
    let off = dir("skill-off");
    let on = dir("skill-on");
    Trainer::new(weighted(1, 0.0), opponents(), &off)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    Trainer::new(weighted(1, 1.0), opponents(), &on)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    let (a, b) = (&generation_events(&off)[0], &generation_events(&on)[0]);
    assert!(a.fitness.skill_term.is_none() && a.champion.skill_term.is_none());
    // With weight 1 the fitness is the (round-1) skill term itself.
    let stats = b.fitness.skill_term.as_ref().unwrap();
    assert!((stats.mean - b.fitness.mean).abs() < 1e-12);
    assert!((a.fitness.mean - b.fitness.mean).abs() > 1e-12);
    fs::remove_dir_all(&off).unwrap();
    fs::remove_dir_all(&on).unwrap();
}

#[test]
fn generation_events_carry_stage_timings_and_old_events_still_load() {
    let run = dir("timings");
    Trainer::new(config(2), opponents(), &run)
        .unwrap()
        .run(&mut Recorder::default())
        .unwrap();
    for event in generation_events(&run) {
        let t = event.timings.expect("a new event has timings");
        let stages = [
            t.training_evaluation,
            t.champion_selection,
            t.speciation_and_reproduction,
            t.reevaluation_mixed,
            t.reevaluation_per_opponent,
            t.hall_of_fame,
            t.confirmation,
            t.decision_sample,
            t.checkpoint_and_files,
        ];
        assert!(stages.iter().all(|s| s.is_finite() && *s >= 0.0), "{t:?}");
        assert!(t.training_evaluation > 0.0);
        // The mixed, per-opponent and hall-of-fame re-evaluations run concurrently, so only the
        // longest of them adds to the generation's wall time.
        let sequential = t.training_evaluation
            + t.champion_selection
            + t.speciation_and_reproduction
            + t.confirmation
            + t.decision_sample
            + t.checkpoint_and_files
            + t.reevaluation_mixed
                .max(t.reevaluation_per_opponent)
                .max(t.hall_of_fame);
        assert!(
            sequential <= event.generation_secs + 1e-3,
            "sequential stages plus the longest concurrent one fit in the generation: {t:?} vs {}",
            event.generation_secs
        );
    }
    // An event line from before the field existed has no `timings` key.
    let text = fs::read_to_string(run.join("events.jsonl")).unwrap();
    let line = text.lines().nth(1).unwrap();
    let mut value: serde_json::Value = serde_json::from_str(line).unwrap();
    assert!(value.as_object_mut().unwrap().remove("timings").is_some());
    match serde_json::from_value::<Event>(value).unwrap() {
        Event::Generation(g) => assert!(g.timings.is_none()),
        other => panic!("{other:?}"),
    }
    fs::remove_dir_all(&run).unwrap();
}

/// Everything a generation event says except the wall-clock fields.
fn timeless_events(dir: &Path) -> Vec<serde_json::Value> {
    generation_events(dir)
        .iter()
        .map(|e| {
            let mut value = serde_json::to_value(e).unwrap();
            let object = value.as_object_mut().unwrap();
            for key in [
                "elapsed_secs",
                "generation_secs",
                "rounds_per_sec",
                "timings",
            ] {
                object.remove(key);
            }
            value
        })
        .collect()
}

#[test]
fn the_whole_training_loop_is_identical_on_1_4_and_8_threads() {
    // Champion candidates, hall of fame (interval 1) and the skill term
    // exercise every parallel stage of a generation.
    let full = || TrainConfig {
        champion_candidates: 3,
        hall_of_fame_size: 2,
        hall_of_fame_interval: 1,
        skill_weight: 0.5,
        ..config(4)
    };
    let run_with = |threads: usize| {
        let run = dir(&format!("full-threads{threads}"));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            Trainer::new(full(), opponents(), &run)
                .unwrap()
                .run(&mut Recorder::default())
                .unwrap();
        });
        let result = (
            checkpoint_population(&run),
            fs::read_to_string(run.join("best.json")).unwrap(),
            timeless_events(&run),
        );
        fs::remove_dir_all(&run).unwrap();
        result
    };
    let one = run_with(1);
    assert!(one.2.iter().any(|e| !e["hall_score"].is_null()));
    assert_eq!(one, run_with(4));
    assert_eq!(one, run_with(8));
}

#[derive(Default)]
struct ProgressLog(Vec<(u32, usize, usize)>);

impl TrainObserver for ProgressLog {
    fn on_eval_progress(&mut self, generation: u32, done: usize, total: usize) {
        self.0.push((generation, done, total));
    }
}

#[test]
fn progress_is_monotone_and_ends_at_the_whole_population_on_any_pool() {
    let check = |threads: Option<usize>| {
        let run = dir(&format!("progress-{threads:?}"));
        let mut log = ProgressLog::default();
        let mut trainer = Trainer::new(config(3), opponents(), &run).unwrap();
        match threads {
            // The global pool: the calling thread is not a pool worker.
            None => {
                trainer.run(&mut log).unwrap();
            }
            // Inside a pool: the caller is a worker (a one-thread pool must
            // not deadlock while the caller waits for its own tasks).
            Some(n) => rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap()
                .install(|| {
                    trainer.run(&mut log).unwrap();
                }),
        }
        for generation in 0..3 {
            let steps: Vec<usize> = log
                .0
                .iter()
                .filter(|p| p.0 == generation)
                .map(|p| {
                    assert_eq!(p.2, 16);
                    p.1
                })
                .collect();
            assert!(steps.windows(2).all(|w| w[0] < w[1]), "{steps:?}");
            assert_eq!(steps.last(), Some(&16), "{threads:?}: {steps:?}");
        }
        fs::remove_dir_all(&run).unwrap();
    };
    check(None);
    check(Some(1));
    check(Some(4));
}
