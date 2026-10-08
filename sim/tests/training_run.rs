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
