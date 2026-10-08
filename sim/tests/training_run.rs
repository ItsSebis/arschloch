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
        rounds_per_match: 4,
        matches_per_genome: 8,
        reeval_matches: 12,
        generations,
        neat: NeatConfig {
            population_size: 16,
            ..NeatConfig::default()
        },
        opponent_specs: vec!["lowest-legal".into(), "random-legal".into()],
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
    // 16 genomes x 8 matches + 12 x (1 + 2 opponents), 4 rounds each.
    assert_eq!(first.rounds_evaluated, 4 * (16 * 8 + 12 * 3));
    assert!(matches!(log[4], Event::RunEnd(_)));

    assert!(NeatStrategy::from_file(&run.join("best.json")).is_ok());
    assert!(NeatStrategy::from_file(&run.join("gen-0002.json")).is_ok());
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
