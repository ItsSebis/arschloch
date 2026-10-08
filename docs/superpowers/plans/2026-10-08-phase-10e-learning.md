# Phase 10e — Learning Quality and the v1 Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make trained champions reliably good and prove it: re-select
each generation's champion on fixed matches instead of trusting a noisy
training score, offer a hall of fame and a tunable mutation size, add a
command that scores genomes against any opponents (including ones never
trained against), choose the defaults from real experiments, and commit a
first champion with its recorded comparison against the pre-NEAT
strategies.

**Architecture:** `sim::training` gains `champion_candidates`,
`hall_of_fame_size/interval` in `TrainConfig` (serde defaults keep old
configs and checkpoints loading), a `HallMember` list in the checkpoint,
`select_champion` (top-k by training fitness, re-scored on the fixed
matches) and a hall that joins the training pool while reporting stays on
the fixed pool. `cli train` exposes the options; `cli evaluate` runs the
spec's baseline-comparison protocol; the dashboard shows the hall line.
`docs/baselines/neat-v1/` holds the experiments, the committed champion and
the recorded comparison.

**Tech Stack:** Rust workspace (no new crates), the existing `rayon`
evaluation, shell and Python for the experiment scripts.

**Spec:** `docs/superpowers/specs/2026-10-08-neat-engine-design.md`,
sections 6 (opponent pool, hall of fame, tie handling), 9 (baseline
comparison protocol, success criteria) and 12 (risks: noisy fitness,
overfitting). Task 6 records, as section 7d, what was built and what was
left out.

## How this plan was prepared

The code was written into a scratch copy first and passed `cargo fmt
--check`, `cargo clippy --workspace --all-targets -- -D warnings` and
`cargo test --workspace` (439 tests). Then it was used for real:

- **Experiments** (Task 4): 22 training runs (population 100, 60
  generations) over six configurations, every champion scored on a
  battery that includes opponents it never trained against. Result:
  generalization is not a problem (+0.90 to +0.92 on unseen opponents),
  and top-5 selection plus weight power 0.2 beat the baseline by +0.016
  (paired standard error 0.012) over five seeds, so they became the
  defaults; the hall of fame showed no benefit.
- **The v1 champion** (Task 5): 120 generations at population 150 took
  11.5 minutes; its fixed-match re-selection overruled the training-best
  genome in 75 of 120 generations. Held-out score +0.624 +- 0.013.
  Trained at 4 players only, it beats every opponent at every table size
  from 3 to 6 (+0.42 to +0.96) and is President in 57% of rounds against
  three copies of the strongest hand-written strategy.

The steps are applied by one executor from the same data that renders
this document; the file changes of Tasks 1-3 are generated as diffs
against the repository, so the plan cannot drift from what runs.

## Global Constraints

- No new crates. `Strategy`, `TurnContext`, `run_match`, `run_batch`, the hand-written strategies, `docs/baselines/pre-neat/` and the `neat` crate are **not modified**.
- Old `config.json`/`checkpoint.json` files (without the new fields) must keep loading with the old behaviour: `champion_candidates` 1, `hall_of_fame_size` 0, `hall_of_fame_interval` 5 as *serde defaults*. The *CLI* defaults (5, 0, 5, weight power 0.2) apply to new runs only.
- A run stays a pure function of its `TrainConfig`: the hall is deterministic and persisted in the checkpoint, so resuming equals never stopping, and results do not depend on thread count.
- Champions are compared on the fixed seed set (same deals every generation); the hall never changes what the per-opponent report measures (fixed pool only), so series stay comparable across generations.
- `cli evaluate` uses a seed stream that training never uses (`match_seed(seed, u64::MAX - 3, i)`), the same deals/seats/opponents for every genome, and rejects duplicate opponent names (results would merge).
- Reproducibility: everything in `docs/baselines/neat-v1/summaries/` must be reproducible by `run.sh` (Task 5 checks it with a different thread count).
- Between tasks the non-test build can report `dead_code`/unused warnings for items later tasks start using; any other warning is a defect.
- Per-task verification is `cargo test -p <crate>`; the full gate (`cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`; judge by the exit status, not by a summary pipe) runs in Task 6.

## Review Focus

1. **Champion selection must never pick a worse champion than trusting the training best, and must be deterministic:** `training_run::candidate_selection_never_picks_a_worse_champion_than_the_training_best`, `the_champion_is_chosen_among_the_top_candidates_by_the_fixed_matches`, `trainer::tests::top_indices_orders_best_first_with_ties_going_to_the_lower_index`.
2. **The hall of fame must be exact across resume and old checkpoints:** `training_run::resuming_with_a_hall_of_fame_still_equals_never_stopping`, `a_checkpoint_from_before_the_hall_of_fame_still_resumes`, `config::tests::a_config_saved_before_these_options_existed_still_loads_with_their_defaults`, `a_hall_of_fame_fills_every_interval_and_keeps_only_the_newest`.
3. **`cli evaluate` must be comparable and honest:** same seeds for every genome, a seed stream training never uses, no dependence on thread count, loud failure on bad input (`evaluate_smoke::the_same_arguments_always_give_the_same_numbers`, `operator_mistakes_fail_clearly_without_a_panic`).
4. **The committed baseline must be reproducible:** Task 5's `check_reproducible.sh` re-runs the protocol with 1 thread and compares all 20 recorded files.
5. **Bad option values fail before anything is written:** `train_smoke::an_absurd_weight_power_is_refused_before_anything_is_written`, `config` validation of `champion_candidates` (0 or above the population) and `hall_of_fame_interval` 0.

---

## File Structure

```
sim/src/training/{config,events,run_dir,trainer,mod}.rs   # Task 1
sim/tests/training_run.rs                                 # Task 1
web/src/test_fixture.rs, web/tests/js.rs                  # Task 1 (fixture configs)
cli/src/{train_args,train,train_output}.rs, cli/tests/train_smoke.rs   # Task 1
cli/src/evaluate.rs, cli/src/main.rs, cli/tests/evaluate_smoke.rs      # Task 2
web/assets/app.js                                         # Task 3
docs/baselines/neat-v1/{experiments.md,experiments/*}     # Task 4
docs/baselines/neat-v1/{champion.json,run.sh,README.md,summaries/*}    # Task 5
docs/ROADMAP.md, docs/ARCHITECTURE.md, docs/BUILDING.md, the spec      # Task 6
```

Not in this phase: a weak-to-strong curriculum, matches growing over generations, complexity pressure in ties, the role feature, evolving the exchange step, and the how-to-train guide (the next deliverable).

---

### Task 1: Learning options: top-k champion selection, a hall of fame, weight power

**Files:**
- Modify: `sim/src/training/{config,events,run_dir,trainer,mod}.rs`, `sim/tests/training_run.rs`, `web/src/test_fixture.rs`, `web/tests/js.rs`, `cli/src/{train_args,train,train_output}.rs`, `cli/tests/train_smoke.rs`

**Why:** A 10c review showed the training fitness of a generation's best genome is dominated by noise (its standard error is as large as the gaps between the top genomes), so the recorded champion was often not the strongest. Re-scoring the top few on the fixed matches picks the real best. The hall of fame guards against tuning only to three deterministic opponents.

**Interfaces:**
- Consumes: The 10c/10d trainer (`select`/`reevaluate`/`step`, `Checkpoint`, events), `TrainArgs`, the terminal renderers.
- Produces: `TrainConfig { champion_candidates (serde default 1), hall_of_fame_size (default 0), hall_of_fame_interval (default 5) }` (old configs and checkpoints still load); the generation's champion is the best of the top `champion_candidates` genomes by training fitness *re-scored on the fixed matches*, with `ChampionStats::training_rank` recording where it ranked; a hall of fame of frozen past champions (a fresh champion every `interval` generations, newest `size` kept, persisted in the checkpoint) joins the training opponents while per-opponent reporting stays on the fixed pool, and `GenerationEvent::{hall_of_fame, hall_score}` report it; `cli train --champion-candidates N --hall-of-fame N --hall-interval N --weight-power P` (defaults 5, 0, 5, 0.2: see Task 4) and a `hof` column in the terminal row; `Trainer`'s round accounting now includes the held-out confirmation matches.

- [ ] **Step 1: Edit**

In `sim/tests/training_run.rs` (Tests first: hall of fame fills, caps and resumes exactly; old checkpoints still load; candidate selection uses the fixed matches and never does worse than the training best; round accounting), replace:

```rust
            ..NeatConfig::default()
        },
        opponent_specs: vec!["lowest-legal".into(), "random-legal".into()],
    }
}
```

with:

```rust
            ..NeatConfig::default()
        },
        opponent_specs: vec!["lowest-legal".into(), "random-legal".into()],
        champion_candidates: 1,
        hall_of_fame_size: 0,
        hall_of_fame_interval: 5,
    }
}
```

- [ ] **Step 2: Edit**

In `sim/tests/training_run.rs` (Tests first: hall of fame fills, caps and resumes exactly; old checkpoints still load; candidate selection uses the fixed matches and never does worse than the training best; round accounting), replace:

```rust
        "the first champion is the first best"
    );
    assert_eq!(first.champion.genome_file, "gen-0000.json");
    // 16 genomes x 8 matches + 12 x (1 + 2 opponents), 4 rounds each.
    assert_eq!(first.rounds_evaluated, 4 * (16 * 8 + 12 * 3));
    assert!(matches!(log[4], Event::RunEnd(_)));

    assert!(NeatStrategy::from_file(&run.join("best.json")).is_ok());
```

with:

```rust
        "the first champion is the first best"
    );
    assert_eq!(first.champion.genome_file, "gen-0000.json");
    // 16 genomes x 8 matches, the champion's fixed-match scores (12 matches
    // each: the mixed pool and 2 opponents), and, because the first
    // champion is a new best, 24 held-out matches; 4 rounds each.
    assert_eq!(first.rounds_evaluated, 4 * (16 * 8 + 12 * 3 + 24));
    assert!(matches!(log[4], Event::RunEnd(_)));

    assert!(NeatStrategy::from_file(&run.join("best.json")).is_ok());
```

- [ ] **Step 3: Edit**

In `sim/tests/training_run.rs` (Tests first: hall of fame fills, caps and resumes exactly; old checkpoints still load; candidate selection uses the fixed matches and never does worse than the training best; round accounting), replace:

```rust
    assert!(end.best_heldout.is_some());
    fs::remove_dir_all(&run).unwrap();
}
```

with:

```rust
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
```

- [ ] **Step 4: Edit**

In `cli/tests/train_smoke.rs` (Tests first: the learning options reach the config, events and terminal output; an absurd weight power is refused before anything is written), replace:

```rust
    std::fs::remove_dir_all(&out).unwrap();
    std::fs::remove_dir_all(&base).unwrap();
}
```

with:

```rust
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
```

- [ ] **Step 5: Run (expect failure)**

Run: `cargo test -p sim --test training_run`

Expected: FAIL (compile errors: `TrainConfig` has no `champion_candidates`, `hall_of_fame_size` ...; events have no `hall_of_fame`).

- [ ] **Step 6: Edit**

In `sim/src/training/config.rs`, replace:

```rust
    /// The opponent pool as the `--strategy`-style specs that built it,
    /// so a resume can rebuild exactly the same pool.
    pub opponent_specs: Vec<String>,
}

impl TrainConfig {
```

with:

```rust
    /// The opponent pool as the `--strategy`-style specs that built it,
    /// so a resume can rebuild exactly the same pool.
    pub opponent_specs: Vec<String>,
    /// How many of the genomes with the best *training* fitness are
    /// re-evaluated on the fixed matches to pick the generation's
    /// champion (1 = trust the training fitness). Training fitness is
    /// noisy enough that its best genome is often not the strongest one.
    #[serde(default = "one")]
    pub champion_candidates: usize,
    /// Frozen past champions kept as extra opponents in the training pool
    /// (0 = none), so the population is not tuned only against the fixed
    /// opponents.
    #[serde(default)]
    pub hall_of_fame_size: usize,
    /// A generation's champion joins the hall of fame every this many
    /// generations (the oldest member leaves when it is full).
    #[serde(default = "five")]
    pub hall_of_fame_interval: u32,
}

fn one() -> usize {
    1
}

fn five() -> u32 {
    5
}

impl TrainConfig {
```

- [ ] **Step 7: Edit**

In `sim/src/training/config.rs`, replace:

```rust
        if self.opponent_specs.is_empty() {
            return Err("the opponent pool is empty".into());
        }
        self.neat.validate().map_err(|e| e.to_string())
    }
}
```

with:

```rust
        if self.opponent_specs.is_empty() {
            return Err("the opponent pool is empty".into());
        }
        if self.champion_candidates == 0 || self.champion_candidates > self.neat.population_size {
            return Err(format!(
                "champion_candidates must be between 1 and the population size ({})",
                self.neat.population_size
            ));
        }
        if self.hall_of_fame_interval == 0 {
            return Err("hall_of_fame_interval must be at least 1".into());
        }
        self.neat.validate().map_err(|e| e.to_string())
    }
}
```

- [ ] **Step 8: Edit**

In `sim/src/training/config.rs`, replace:

```rust
                ..NeatConfig::default()
            },
            opponent_specs: vec!["lowest-legal".into()],
        }
    }
}
```

with:

```rust
                ..NeatConfig::default()
            },
            opponent_specs: vec!["lowest-legal".into()],
            champion_candidates: 1,
            hall_of_fame_size: 0,
            hall_of_fame_interval: 5,
        }
    }
}
```

- [ ] **Step 9: Edit**

In `sim/src/training/config.rs`, replace:

```rust
                },
                "pool is empty",
            ),
        ];
        for (config, expected) in cases {
            let error = config.validate().unwrap_err();
            assert!(error.contains(expected), "{expected}: {error}");
        }
    }

    #[test]
```

with:

```rust
                },
                "pool is empty",
            ),
            (
                TrainConfig {
                    champion_candidates: 0,
                    ..sample()
                },
                "champion_candidates",
            ),
            (
                TrainConfig {
                    champion_candidates: 99,
                    ..sample()
                },
                "champion_candidates",
            ),
            (
                TrainConfig {
                    hall_of_fame_interval: 0,
                    ..sample()
                },
                "hall_of_fame_interval",
            ),
        ];
        for (config, expected) in cases {
            let error = config.validate().unwrap_err();
            assert!(error.contains(expected), "{expected}: {error}");
        }
    }

    #[test]
    fn a_config_saved_before_these_options_existed_still_loads_with_their_defaults() {
        let mut value = serde_json::to_value(sample()).unwrap();
        let object = value.as_object_mut().unwrap();
        for key in [
            "champion_candidates",
            "hall_of_fame_size",
            "hall_of_fame_interval",
        ] {
            object.remove(key);
        }
        let old: TrainConfig = serde_json::from_value(value).unwrap();
        assert_eq!(
            (
                old.champion_candidates,
                old.hall_of_fame_size,
                old.hall_of_fame_interval
            ),
            (1, 0, 5)
        );
    }

    #[test]
```

- [ ] **Step 10: Edit**

In `sim/src/training/events.rs`, replace:

```rust
    /// generations inflates `reeval`.
    #[serde(default)]
    pub heldout: Option<ScoreStat>,
    pub hidden_nodes: usize,
    pub enabled_connections: usize,
    /// File name (inside the run directory) of this champion's genome.
```

with:

```rust
    /// generations inflates `reeval`.
    #[serde(default)]
    pub heldout: Option<ScoreStat>,
    /// Where the champion ranked by *training* fitness among this
    /// generation's genomes (0 = it was the training best). Non-zero means
    /// the fixed-match re-evaluation overruled a lucky training score.
    #[serde(default)]
    pub training_rank: usize,
    pub hidden_nodes: usize,
    pub enabled_connections: usize,
    /// File name (inside the run directory) of this champion's genome.
```

- [ ] **Step 11: Edit**

In `sim/src/training/events.rs`, replace:

```rust
    pub fitness: FitnessStats,
    pub champion: ChampionStats,
    pub opponents: Vec<OpponentStat>,
    pub species: Vec<SpeciesStats>,
    pub compatibility_threshold: f64,
    pub complexity: Complexity,
```

with:

```rust
    pub fitness: FitnessStats,
    pub champion: ChampionStats,
    pub opponents: Vec<OpponentStat>,
    /// Generations of the champions that were in the hall of fame (extra
    /// training opponents) while this generation was evaluated.
    #[serde(default)]
    pub hall_of_fame: Vec<u32>,
    /// The champion against tables of hall-of-fame members only (`None`
    /// while the hall is empty).
    #[serde(default)]
    pub hall_score: Option<ScoreStat>,
    pub species: Vec<SpeciesStats>,
    pub compatibility_threshold: f64,
    pub complexity: Complexity,
```

- [ ] **Step 12: Edit**

In `sim/src/training/run_dir.rs`, replace:

```rust
    pub heldout: ScoreStat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub schema_version: u32,
```

with:

```rust
    pub heldout: ScoreStat,
}

/// A frozen past champion serving as an extra training opponent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallMember {
    pub generation: u32,
    pub genome: Genome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub schema_version: u32,
```

- [ ] **Step 13: Edit**

In `sim/src/training/run_dir.rs`, replace:

```rust
    pub opponent_names: Vec<String>,
    pub population: PopulationState,
    pub best: Option<BestRecord>,
    pub total_rounds: u64,
    pub elapsed_secs: f64,
}
```

with:

```rust
    pub opponent_names: Vec<String>,
    pub population: PopulationState,
    pub best: Option<BestRecord>,
    #[serde(default)]
    pub hall_of_fame: Vec<HallMember>,
    pub total_rounds: u64,
    pub elapsed_secs: f64,
}
```

- [ ] **Step 14: Edit**

In `sim/src/training/run_dir.rs`, replace:

```rust
            opponent_names: vec!["LowestLegal".into()],
            population: population.snapshot(),
            best: None,
            total_rounds: 10,
            elapsed_secs: 1.0,
        }
```

with:

```rust
            opponent_names: vec!["LowestLegal".into()],
            population: population.snapshot(),
            best: None,
            hall_of_fame: Vec::new(),
            total_rounds: 10,
            elapsed_secs: 1.0,
        }
```

- [ ] **Step 15: Edit**

In `sim/src/training/run_dir.rs`, replace:

```rust
                train_fitness: 0.0,
                reeval: stat,
                heldout: None,
                hidden_nodes: 0,
                enabled_connections: 0,
                genome_file: String::new(),
                is_new_best: false,
            },
            opponents: vec![],
            species: vec![],
            compatibility_threshold: 0.5,
            complexity: super::super::events::Complexity {
```

with:

```rust
                train_fitness: 0.0,
                reeval: stat,
                heldout: None,
                training_rank: 0,
                hidden_nodes: 0,
                enabled_connections: 0,
                genome_file: String::new(),
                is_new_best: false,
            },
            opponents: vec![],
            hall_of_fame: vec![],
            hall_score: None,
            species: vec![],
            compatibility_threshold: 0.5,
            complexity: super::super::events::Complexity {
```

- [ ] **Step 16: Edit**

In `sim/src/training/trainer.rs`, replace:

```rust
    ChampionStats, Complexity, Event, FitnessStats, GenerationEvent, OpponentStat, RunEnd,
    RunStart, SCHEMA_VERSION,
};
use super::run_dir::{BestRecord, Checkpoint, RunDir, TrainError};
use crate::{NeatStrategy, Strategy, FEATURE_COUNT, FEATURE_NAMES, FEATURE_SET_VERSION};

/// One member of the opponent pool.
```

with:

```rust
    ChampionStats, Complexity, Event, FitnessStats, GenerationEvent, OpponentStat, RunEnd,
    RunStart, SCHEMA_VERSION,
};
use super::run_dir::{BestRecord, Checkpoint, HallMember, RunDir, TrainError};
use crate::{NeatStrategy, Strategy, FEATURE_COUNT, FEATURE_NAMES, FEATURE_SET_VERSION};

/// One member of the opponent pool.
```

- [ ] **Step 17: Edit**

In `sim/src/training/trainer.rs`, replace:

```rust
    dir: RunDir,
    population: Population,
    best: Option<BestRecord>,
    total_rounds: u64,
    elapsed_before: f64,
    resumed_from: Option<u32>,
```

with:

```rust
    dir: RunDir,
    population: Population,
    best: Option<BestRecord>,
    hall: Vec<HallMember>,
    total_rounds: u64,
    elapsed_before: f64,
    resumed_from: Option<u32>,
```

- [ ] **Step 18: Edit**

In `sim/src/training/trainer.rs`, replace:

```rust
    )
}

fn position_of_best(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 {
                (i, v)
            } else {
                best
            }
        })
        .0
}

#[allow(clippy::cast_precision_loss)] // counts are far below 2^52
```

with:

```rust
    )
}

/// The indices of the `k` highest values, best first (ties: lower index).
fn top_indices(values: &[f64], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    order.truncate(k);
    order
}

#[allow(clippy::cast_precision_loss)] // counts are far below 2^52
```

- [ ] **Step 19: Edit**

In `sim/src/training/trainer.rs`, replace:

```rust
            dir,
            population,
            best: None,
            total_rounds: 0,
            elapsed_before: 0.0,
            resumed_from: None,
```

with:

```rust
            dir,
            population,
            best: None,
            hall: Vec::new(),
            total_rounds: 0,
            elapsed_before: 0.0,
            resumed_from: None,
```

- [ ] **Step 20: Edit**

In `sim/src/training/trainer.rs`, replace:

```rust
            opponent_names: self.opponents.iter().map(|o| o.name.clone()).collect(),
            population: self.population.snapshot(),
            best: self.best.clone(),
            total_rounds: self.total_rounds,
            elapsed_secs,
        }
```

with:

```rust
            opponent_names: self.opponents.iter().map(|o| o.name.clone()).collect(),
            population: self.population.snapshot(),
            best: self.best.clone(),
            hall_of_fame: self.hall.clone(),
            total_rounds: self.total_rounds,
            elapsed_secs,
        }
```

- [ ] **Step 21: Edit**

In `sim/src/training/trainer.rs`, replace:

```rust
            dir,
            population,
            best: checkpoint.best,
            total_rounds: checkpoint.total_rounds,
            elapsed_before: checkpoint.elapsed_secs,
            resumed_from: Some(completed),
```

with:

```rust
            dir,
            population,
            best: checkpoint.best,
            hall: checkpoint.hall_of_fame,
            total_rounds: checkpoint.total_rounds,
            elapsed_before: checkpoint.elapsed_secs,
            resumed_from: Some(completed),
```

- [ ] **Step 22: Edit**

In `sim/src/training/trainer.rs`, replace:

```rust
        Ok(end)
    }

    fn evaluate_population(&self, generation: u32, observer: &mut dyn TrainObserver) -> Vec<f64> {
        let table = self.config.table();
        let pool: Vec<Arc<dyn Strategy>> =
            self.opponents.iter().map(|o| o.strategy.clone()).collect();
        let seeds = training_seeds(&self.config, generation);
        let genomes = self.population.genomes();
        let batch = genomes.len().div_ceil(PROGRESS_STEPS);
```

with:

```rust
        Ok(end)
    }

    /// The fixed opponents only: what champions are compared against, so
    /// scores stay comparable across generations as the hall changes.
    fn fixed_pool(&self) -> Vec<Arc<dyn Strategy>> {
        self.opponents.iter().map(|o| o.strategy.clone()).collect()
    }

    fn hall_pool(&self) -> Vec<Arc<dyn Strategy>> {
        self.hall
            .iter()
            .map(|member| -> Arc<dyn Strategy> {
                Arc::new(
                    NeatStrategy::new(format!("HoF({})", member.generation), &member.genome)
                        .expect("hall members use this build's features"),
                )
            })
            .collect()
    }

    /// What genomes train against: the fixed opponents plus the hall.
    fn training_pool(&self) -> Vec<Arc<dyn Strategy>> {
        let mut pool = self.fixed_pool();
        pool.extend(self.hall_pool());
        pool
    }

    fn evaluate_population(&self, generation: u32, observer: &mut dyn TrainObserver) -> Vec<f64> {
        let table = self.config.table();
        let pool = self.training_pool();
        let seeds = training_seeds(&self.config, generation);
        let genomes = self.population.genomes();
        let batch = genomes.len().div_ceil(PROGRESS_STEPS);
```

- [ ] **Step 23: Edit**

In `sim/src/training/trainer.rs`, replace:

```rust
        fitness
    }

    fn reevaluate(&self, champion: &Genome) -> (Score, Vec<OpponentStat>) {
        let table = self.config.table();
        let pool: Vec<Arc<dyn Strategy>> =
            self.opponents.iter().map(|o| o.strategy.clone()).collect();
        let seeds = reeval_seeds(&self.config);
        let candidate = strategy_for(champion);
        let mixed = evaluate(&candidate, &table, Opponents::Mixed(&pool), &seeds);
        let per_opponent = self
            .opponents
            .par_iter()
```

with:

```rust
        fitness
    }

    /// Picks the generation's champion: the best of the `k` genomes with
    /// the highest training fitness *on the fixed matches*. Returns its
    /// index, its rank by training fitness and its fixed-match score.
    fn select_champion(&self, fitness: &[f64]) -> (usize, usize, Score) {
        let table = self.config.table();
        let pool = self.fixed_pool();
        let seeds = reeval_seeds(&self.config);
        let ranked = top_indices(fitness, self.config.champion_candidates);
        let genomes = self.population.genomes();
        let scores: Vec<Score> = ranked
            .par_iter()
            .map(|&index| {
                evaluate(
                    &strategy_for(&genomes[index]),
                    &table,
                    Opponents::Mixed(&pool),
                    &seeds,
                )
            })
            .collect();
        let winner = scores.iter().enumerate().fold(0, |best, (rank, score)| {
            if score.mean > scores[best].mean {
                rank
            } else {
                best
            }
        });
        (ranked[winner], winner, scores[winner].clone())
    }

    /// The champion against each fixed opponent alone and, once the hall
    /// has members, against the hall alone. `mixed` is its score against
    /// the mixed fixed pool (already computed by `select_champion`).
    fn reevaluate(&self, champion: &Genome) -> (Vec<OpponentStat>, Option<Score>) {
        let table = self.config.table();
        let seeds = reeval_seeds(&self.config);
        let candidate = strategy_for(champion);
        let per_opponent = self
            .opponents
            .par_iter()
```

- [ ] **Step 24: Edit**

In `sim/src/training/trainer.rs`, replace:

```rust
                .into(),
            })
            .collect();
        (mixed, per_opponent)
    }

    /// A few real decisions of the champion (for the dashboard's decision
```

with:

```rust
                .into(),
            })
            .collect();
        let hall = self.hall_pool();
        let hall_score = (!hall.is_empty())
            .then(|| evaluate(&candidate, &table, Opponents::Mixed(&hall), &seeds));
        (per_opponent, hall_score)
    }

    /// A few real decisions of the champion (for the dashboard's decision
```

- [ ] **Step 25: Edit**

In `sim/src/training/trainer.rs`, replace:

```rust
        let generation = self.population.generation();

        let fitness = self.evaluate_population(generation, observer);
        let champion_index = position_of_best(&fitness);
        // `advance` replaces the genomes, so take the champion first.
        let champion = self.population.genomes()[champion_index].clone();
        let stats = fitness_stats(&fitness);
        self.population.set_fitness(fitness.clone());
        let report = self.population.advance();

        let (reeval, opponents) = self.reevaluate(&champion);
        let is_new_best = self
            .best
            .as_ref()
```

with:

```rust
        let generation = self.population.generation();

        let fitness = self.evaluate_population(generation, observer);
        let (champion_index, training_rank, reeval) = self.select_champion(&fitness);
        // `advance` replaces the genomes, so take the champion first.
        let champion = self.population.genomes()[champion_index].clone();
        let hall_generations: Vec<u32> = self.hall.iter().map(|m| m.generation).collect();
        let stats = fitness_stats(&fitness);
        self.population.set_fitness(fitness.clone());
        let report = self.population.advance();

        let (opponents, hall_score) = self.reevaluate(&champion);
        let is_new_best = self
            .best
            .as_ref()
```

- [ ] **Step 26: Edit**

In `sim/src/training/trainer.rs`, replace:

```rust
            None
        };

        let table = self.config.table();
        let rounds_per_match = table.rounds as u64;
        let rounds_evaluated = rounds_per_match
            * (self.config.matches_per_genome as u64 * self.config.neat.population_size as u64
                + self.config.reeval_matches as u64 * (1 + self.opponents.len() as u64));
        self.total_rounds += rounds_evaluated;
        let generation_secs = step_started.elapsed().as_secs_f64();
        let elapsed_secs = self.elapsed_before + started.elapsed().as_secs_f64();
```

with:

```rust
            None
        };

        // The hall takes a fresh champion every `interval` generations.
        if self.config.hall_of_fame_size > 0
            && generation > 0
            && generation.is_multiple_of(self.config.hall_of_fame_interval)
        {
            self.hall.push(HallMember {
                generation,
                genome: champion.clone(),
            });
            while self.hall.len() > self.config.hall_of_fame_size {
                self.hall.remove(0);
            }
        }

        let table = self.config.table();
        let rounds_per_match = table.rounds as u64;
        let rounds_evaluated = rounds_per_match
            * (self.config.matches_per_genome as u64 * self.config.neat.population_size as u64
                + self.config.reeval_matches as u64
                    * (self.config.champion_candidates as u64
                        + self.opponents.len() as u64
                        + u64::from(hall_score.is_some()))
                + heldout.as_ref().map_or(0, |h| h.matches as u64));
        self.total_rounds += rounds_evaluated;
        let generation_secs = step_started.elapsed().as_secs_f64();
        let elapsed_secs = self.elapsed_before + started.elapsed().as_secs_f64();
```

- [ ] **Step 27: Edit**

In `sim/src/training/trainer.rs`, replace:

```rust
                train_fitness: fitness[champion_index],
                reeval: reeval.into(),
                heldout: heldout.map(Into::into),
                hidden_nodes: champion.hidden_count(),
                enabled_connections: champion.enabled_connection_count(),
                genome_file,
                is_new_best,
            },
            opponents,
            species: report.species,
            compatibility_threshold: report.compatibility_threshold,
            complexity: Complexity {
```

with:

```rust
                train_fitness: fitness[champion_index],
                reeval: reeval.into(),
                heldout: heldout.map(Into::into),
                training_rank,
                hidden_nodes: champion.hidden_count(),
                enabled_connections: champion.enabled_connection_count(),
                genome_file,
                is_new_best,
            },
            opponents,
            hall_of_fame: hall_generations,
            hall_score: hall_score.map(Into::into),
            species: report.species,
            compatibility_threshold: report.compatibility_threshold,
            complexity: Complexity {
```

- [ ] **Step 28: Edit**

In `sim/src/training/trainer.rs`, replace:

```rust
    use super::*;

    #[test]
    fn champions_are_compared_on_one_fixed_seed_set_that_training_never_uses() {
        let config = sample();
        let fixed = reeval_seeds(&config);
```

with:

```rust
    use super::*;

    #[test]
    fn top_indices_orders_best_first_with_ties_going_to_the_lower_index() {
        let values = [0.5, 0.9, 0.5, -1.0, 0.9];
        assert_eq!(top_indices(&values, 3), vec![1, 4, 0]);
        assert_eq!(top_indices(&values, 1), vec![1]);
        assert_eq!(top_indices(&values, 99), vec![1, 4, 0, 2, 3]);
        assert!(top_indices(&[], 3).is_empty());
    }

    #[test]
    fn champions_are_compared_on_one_fixed_seed_set_that_training_never_uses() {
        let config = sample();
        let fixed = reeval_seeds(&config);
```

- [ ] **Step 29: Edit**

In `sim/src/training/mod.rs`, replace:

```rust
pub use decisions::{record_decisions, DecisionFile, DecisionRecord};
pub use evaluate::{evaluate, match_seed, role_score, Opponents, Score, TableSpec};
pub use events::{Event, GenerationEvent, RunEnd, RunStart, ScoreStat, SCHEMA_VERSION};
pub use run_dir::{load_config, RunDir, TrainError};
pub use trainer::{Opponent, TrainObserver, Trainer};
```

with:

```rust
pub use decisions::{record_decisions, DecisionFile, DecisionRecord};
pub use evaluate::{evaluate, match_seed, role_score, Opponents, Score, TableSpec};
pub use events::{Event, GenerationEvent, RunEnd, RunStart, ScoreStat, SCHEMA_VERSION};
pub use run_dir::{load_config, HallMember, RunDir, TrainError};
pub use trainer::{Opponent, TrainObserver, Trainer};
```

- [ ] **Step 30: Edit**

In `web/src/test_fixture.rs` (The shared fixture config gets the new fields), replace:

```rust
                ..NeatConfig::default()
            },
            opponent_specs: vec!["lowest-legal".into(), "random-legal".into()],
        };
        let opponents = vec![
            Opponent {
```

with:

```rust
                ..NeatConfig::default()
            },
            opponent_specs: vec!["lowest-legal".into(), "random-legal".into()],
            champion_candidates: 1,
            hall_of_fame_size: 0,
            hall_of_fame_interval: 5,
        };
        let opponents = vec![
            Opponent {
```

- [ ] **Step 31: Edit**

In `web/tests/js.rs`, replace:

```rust
            ..NeatConfig::default()
        },
        opponent_specs: vec!["lowest-legal".into(), "random-legal".into()],
    };
    let opponents = vec![
        Opponent {
```

with:

```rust
            ..NeatConfig::default()
        },
        opponent_specs: vec!["lowest-legal".into(), "random-legal".into()],
        champion_candidates: 1,
        hall_of_fame_size: 0,
        hall_of_fame_interval: 5,
    };
    let opponents = vec![
        Opponent {
```

- [ ] **Step 32: Edit**

In `cli/src/train_args.rs`, replace:

```rust
        long,
        conflicts_with_all = [
            "player_count", "deck_variant", "duplicate_rule", "rounds", "population",
            "matches_per_genome", "reeval_matches", "seed", "opponent", "target_species"
        ]
    )]
    pub resume: bool,
```

with:

```rust
        long,
        conflicts_with_all = [
            "player_count", "deck_variant", "duplicate_rule", "rounds", "population",
            "matches_per_genome", "reeval_matches", "seed", "opponent", "target_species",
            "champion_candidates", "hall_of_fame", "hall_interval", "weight_power"
        ]
    )]
    pub resume: bool,
```

- [ ] **Step 33: Edit**

In `cli/src/train_args.rs`, replace:

```rust
    /// How many species the speciation threshold steers toward.
    #[arg(long, default_value_t = 8, value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(1..))]
    pub target_species: usize,

    /// Rayon thread-pool size. 0 lets rayon pick its own default.
    #[arg(long, default_value_t = 0)]
```

with:

```rust
    /// How many species the speciation threshold steers toward.
    #[arg(long, default_value_t = 8, value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(1..))]
    pub target_species: usize,

    /// Re-evaluate this many of each generation's best genomes (by training
    /// fitness) on the fixed matches and take the best of them as the
    /// champion. Training fitness is noisy, so its best genome is often not
    /// the strongest; 1 trusts it. Default 5 (see docs/baselines/neat-v1).
    #[arg(long, default_value_t = 5, value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(1..))]
    pub champion_candidates: usize,

    /// Keep this many frozen past champions as extra training opponents
    /// (0 = none), so the population is not tuned only to the fixed pool.
    #[arg(long, default_value_t = 0)]
    pub hall_of_fame: usize,

    /// A champion joins the hall of fame every this many generations.
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u32).range(1..))]
    pub hall_interval: u32,

    /// Size of a weight perturbation (uniform in plus/minus this). Default
    /// 0.2 (see docs/baselines/neat-v1).
    #[arg(long, default_value_t = 0.2)]
    pub weight_power: f64,

    /// Rayon thread-pool size. 0 lets rayon pick its own default.
    #[arg(long, default_value_t = 0)]
```

- [ ] **Step 34: Edit**

In `cli/src/train_args.rs`, replace:

```rust
    }

    #[test]
    fn out_is_required() {
        assert!(parse(&[]).is_err());
    }
```

with:

```rust
    }

    #[test]
    fn learning_options_default_to_the_measured_settings_and_conflict_with_resume() {
        let args = parse(&["--out", "d"]).unwrap();
        assert_eq!(
            (
                args.champion_candidates,
                args.hall_of_fame,
                args.hall_interval
            ),
            (5, 0, 5)
        );
        assert!((args.weight_power - 0.2).abs() < f64::EPSILON);
        let old_style = parse(&[
            "--out",
            "d",
            "--champion-candidates",
            "1",
            "--hall-of-fame",
            "3",
            "--weight-power",
            "0.5",
        ])
        .unwrap();
        assert_eq!(
            (old_style.champion_candidates, old_style.hall_of_fame),
            (1, 3)
        );
        assert!((old_style.weight_power - 0.5).abs() < f64::EPSILON);
        assert!(parse(&["--out", "d", "--champion-candidates", "0"]).is_err());
        assert!(parse(&["--out", "d", "--hall-interval", "0"]).is_err());
        for conflicting in [
            ["--hall-of-fame", "2"],
            ["--champion-candidates", "3"],
            ["--weight-power", "0.1"],
            ["--hall-interval", "7"],
        ] {
            let mut args = vec!["--out", "d", "--resume"];
            args.extend(conflicting);
            assert!(parse(&args).is_err(), "{conflicting:?}");
        }
    }

    #[test]
    fn out_is_required() {
        assert!(parse(&[]).is_err());
    }
```

- [ ] **Step 35: Edit**

In `cli/src/train.rs`, replace:

```rust
        neat: neat::NeatConfig {
            population_size: args.population,
            target_species: args.target_species,
            ..neat::NeatConfig::default()
        },
        opponent_specs: specs,
    }
}
```

with:

```rust
        neat: neat::NeatConfig {
            population_size: args.population,
            target_species: args.target_species,
            weight_perturb_power: args.weight_power,
            ..neat::NeatConfig::default()
        },
        opponent_specs: specs,
        champion_candidates: args.champion_candidates,
        hall_of_fame_size: args.hall_of_fame,
        hall_of_fame_interval: args.hall_interval,
    }
}
```

- [ ] **Step 36: Edit**

In `cli/src/train_output.rs`, replace:

```rust
}

#[must_use]
pub fn render_header(opponent_count: usize) -> String {
    let mut text = String::from("  gen    best    mean  champion (fresh)  spc  nodes/conn");
    for i in 1..=opponent_count {
        let _ = write!(text, "   o{i:<3}");
    }
    text.push_str("  rounds/s      ETA");
    text
```

with:

```rust
}

#[must_use]
pub fn render_header(opponent_count: usize, show_hall: bool) -> String {
    let mut text = String::from("  gen    best    mean  champion (fresh)  spc  nodes/conn");
    for i in 1..=opponent_count {
        let _ = write!(text, "   o{i:<3}");
    }
    if show_hall {
        text.push_str("   hof ");
    }
    text.push_str("  rounds/s      ETA");
    text
```

- [ ] **Step 37: Edit**

In `cli/src/train_output.rs`, replace:

```rust
/// champion's score against each opponent alone, throughput and ETA. A
/// trailing `*` marks a new best champion.
#[must_use]
pub fn render_row(event: &GenerationEvent, eta_secs: Option<f64>) -> String {
    let champion = &event.champion;
    let mut text = format!(
        "{:>5} {:>+7.3} {:>+7.3}  {:>+7.3} ±{:<6.3}  {:>3}  {:>4}/{:<5}",
```

with:

```rust
/// champion's score against each opponent alone, throughput and ETA. A
/// trailing `*` marks a new best champion.
#[must_use]
pub fn render_row(event: &GenerationEvent, eta_secs: Option<f64>, show_hall: bool) -> String {
    let champion = &event.champion;
    let mut text = format!(
        "{:>5} {:>+7.3} {:>+7.3}  {:>+7.3} ±{:<6.3}  {:>3}  {:>4}/{:<5}",
```

- [ ] **Step 38: Edit**

In `cli/src/train_output.rs`, replace:

```rust
    );
    for opponent in &event.opponents {
        let _ = write!(text, " {:>+6.2}", opponent.score.mean);
    }
    let _ = write!(
        text,
```

with:

```rust
    );
    for opponent in &event.opponents {
        let _ = write!(text, " {:>+6.2}", opponent.score.mean);
    }
    if show_hall {
        match &event.hall_score {
            Some(score) => {
                let _ = write!(text, " {:>+6.2}", score.mean);
            }
            None => text.push_str("      –"),
        }
    }
    let _ = write!(
        text,
```

- [ ] **Step 39: Edit**

In `cli/src/train_output.rs`, replace:

```rust
    text
}

pub struct TerminalObserver {
    quiet: bool,
    out: PathBuf,
    live_progress: bool,
    progress_visible: bool,
    opponent_count: usize,
    total_generations: u32,
    first_generation: Option<u32>,
    generation_secs: Vec<f64>,
```

with:

```rust
    text
}

#[allow(clippy::struct_excessive_bools)] // independent display switches
pub struct TerminalObserver {
    quiet: bool,
    out: PathBuf,
    live_progress: bool,
    progress_visible: bool,
    opponent_count: usize,
    show_hall: bool,
    total_generations: u32,
    first_generation: Option<u32>,
    generation_secs: Vec<f64>,
```

- [ ] **Step 40: Edit**

In `cli/src/train_output.rs`, replace:

```rust
            live_progress: std::io::stderr().is_terminal(),
            progress_visible: false,
            opponent_count: 0,
            total_generations: 0,
            first_generation: None,
            generation_secs: Vec::new(),
```

with:

```rust
            live_progress: std::io::stderr().is_terminal(),
            progress_visible: false,
            opponent_count: 0,
            show_hall: false,
            total_generations: 0,
            first_generation: None,
            generation_secs: Vec::new(),
```

- [ ] **Step 41: Edit**

In `cli/src/train_output.rs`, replace:

```rust
impl TrainObserver for TerminalObserver {
    fn on_start(&mut self, start: &RunStart) {
        self.opponent_count = start.opponents.len();
        self.total_generations = start.config.generations;
        if !self.quiet {
            println!(
                "{}\n\n{}",
                render_banner(start, &self.out),
                render_header(self.opponent_count)
            );
        }
    }
```

with:

```rust
impl TrainObserver for TerminalObserver {
    fn on_start(&mut self, start: &RunStart) {
        self.opponent_count = start.opponents.len();
        self.show_hall = start.config.hall_of_fame_size > 0;
        self.total_generations = start.config.generations;
        if !self.quiet {
            println!(
                "{}\n\n{}",
                render_banner(start, &self.out),
                render_header(self.opponent_count, self.show_hall)
            );
        }
    }
```

- [ ] **Step 42: Edit**

In `cli/src/train_output.rs`, replace:

```rust
            return;
        }
        if (event.generation - first).is_multiple_of(HEADER_EVERY) && event.generation != first {
            println!("{}", render_header(self.opponent_count));
        }
        println!("{}", render_row(event, self.eta(event)));
    }

    fn on_finish(&mut self, end: &RunEnd) {
```

with:

```rust
            return;
        }
        if (event.generation - first).is_multiple_of(HEADER_EVERY) && event.generation != first {
            println!("{}", render_header(self.opponent_count, self.show_hall));
        }
        println!("{}", render_row(event, self.eta(event), self.show_hall));
    }

    fn on_finish(&mut self, end: &RunEnd) {
```

- [ ] **Step 43: Edit**

In `cli/src/train_output.rs`, replace:

```rust
                train_fitness: 0.412,
                reeval: stat(0.397),
                heldout: None,
                hidden_nodes: 7,
                enabled_connections: 23,
                genome_file: "gen-0042.json".into(),
```

with:

```rust
                train_fitness: 0.412,
                reeval: stat(0.397),
                heldout: None,
                training_rank: 0,
                hidden_nodes: 7,
                enabled_connections: 23,
                genome_file: "gen-0042.json".into(),
```

- [ ] **Step 44: Edit**

In `cli/src/train_output.rs`, replace:

```rust
                    score: stat(-0.05),
                },
            ],
            species: vec![],
            compatibility_threshold: 0.5,
            complexity: Complexity {
```

with:

```rust
                    score: stat(-0.05),
                },
            ],
            hall_of_fame: vec![],
            hall_score: None,
            species: vec![],
            compatibility_threshold: 0.5,
            complexity: Complexity {
```

- [ ] **Step 45: Edit**

In `cli/src/train_output.rs`, replace:

```rust

    #[test]
    fn a_row_shows_every_headline_number() {
        let row = render_row(&event(false), Some(2470.0));
        for expected in [
            "42", "+0.412", "+0.188", "+0.397", "±0.021", "7/23", "+0.61", "-0.05", "84k",
            "0:41:10",
```

with:

```rust

    #[test]
    fn a_row_shows_every_headline_number() {
        let row = render_row(&event(false), Some(2470.0), false);
        for expected in [
            "42", "+0.412", "+0.188", "+0.397", "±0.021", "7/23", "+0.61", "-0.05", "84k",
            "0:41:10",
```

- [ ] **Step 46: Edit**

In `cli/src/train_output.rs`, replace:

```rust

    #[test]
    fn a_new_best_is_marked_and_a_missing_eta_is_dashes() {
        let row = render_row(&event(true), None);
        assert!(row.ends_with(" *"), "{row}");
        assert!(row.contains("--:--:--"));
    }

    #[test]
    fn the_header_has_one_column_per_opponent() {
        let header = render_header(3);
        assert!(header.contains("o1") && header.contains("o2") && header.contains("o3"));
        assert!(!header.contains("o4"));
        assert!(header.contains("ETA") && header.contains("champion"));
    }

    #[test]
```

with:

```rust

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
```

- [ ] **Step 47: Edit**

In `cli/src/train_output.rs`, replace:

```rust
            generations: 100,
            neat: neat::NeatConfig::default(),
            opponent_specs: vec![],
        }
    }
}
```

with:

```rust
            generations: 100,
            neat: neat::NeatConfig::default(),
            opponent_specs: vec![],
            champion_candidates: 1,
            hall_of_fame_size: 0,
            hall_of_fame_interval: 5,
        }
    }
}
```

- [ ] **Step 48: Run (expect success)**

Run: `cargo fmt --all && cargo test -p sim -p web`

Expected: PASS (the new `sim` tests: `top_indices`; the hall fills at generations 2, 4, 6 and keeps the newest two; `hall_score` is present exactly when the hall has members; resuming with a hall equals never stopping; a checkpoint written before these options still resumes; the champion's `training_rank` is always below the candidate count and above 0 at least once; and more candidates never give a worse fixed-match score).

- [ ] **Step 49: Run (expect success)**

Run: `cargo test -p cli`

Expected: PASS (including `learning_options_default_to_the_measured_settings_and_conflict_with_resume`, the `hof` column test, and the train smoke tests).

- [ ] **Step 50: Commit**

```bash
git add sim web cli
git commit -F - <<'EOF'
train: top-k champion selection, a hall of fame and weight power (Phase 10e)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F3sC3JdcgTFAox87bBuikA
EOF
```


### Task 2: `cli evaluate`: score genomes against any opponents

**Files:**
- Modify: `cli/src/main.rs`
- Create: `cli/src/evaluate.rs`, `cli/tests/evaluate_smoke.rs`

**Why:** This is the spec's baseline-comparison protocol as a command, and the way to measure generalization to opponents a run never trained against.

**Interfaces:**
- Consumes: `sim::training::{evaluate, match_seed, Opponents, Score, TableSpec}`, `StrategyArg`, `NeatStrategy::from_file`.
- Produces: `cli evaluate --genome PATH... [--opponent SPEC...] [--player-count N] [--rounds R] [--matches M] [--seed S] [--json PATH]`: every genome plays exactly the same deals, seats and opponents on a seed stream training never uses, at tables of one opponent each and at tables mixing them all; a table with one row per opponent plus a mixed row and one column per genome (`+0.574 ±0.019`), optional JSON. The default battery is lowest-legal, endgame-denial, adaptive:reading,tempo,bully, hold-back-pairs, greedy-highest, random-legal; `neat:PATH` opponents pit genomes against each other.

- [ ] **Step 1: Create file**

Create `cli/tests/evaluate_smoke.rs` (Tests first: the real binary on real genome files):

```rust
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
```

- [ ] **Step 2: Create file**

Create `cli/src/evaluate.rs` (Tests first):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn cell(opponent: &str, mean: f64) -> Cell {
        Cell {
            opponent: opponent.into(),
            mean,
            std_error: 0.021,
            matches: 100,
            placements: vec![1, 2, 3, 4],
        }
    }

    fn result(genome: &str, means: [f64; 2]) -> GenomeResult {
        GenomeResult {
            genome: genome.into(),
            cells: vec![cell("LowestLegal", means[0]), cell(MIXED, means[1])],
        }
    }

    #[test]
    fn the_table_has_a_row_per_opponent_plus_mixed_and_a_column_per_genome() {
        let labels = vec!["a".to_owned(), "b".to_owned()];
        let text = render_table(
            &labels,
            &[result("a.json", [0.5, 0.4]), result("b.json", [-0.25, 0.1])],
        );
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3, "{text}");
        assert!(
            lines[0].starts_with("opponent") && lines[0].contains('a') && lines[0].contains('b')
        );
        assert!(
            lines[1].contains("LowestLegal")
                && lines[1].contains("+0.500 ±0.021")
                && lines[1].contains("-0.250 ±0.021")
        );
        assert!(lines[2].starts_with("mixed (all opponents)") && lines[2].contains("+0.400"));
    }

    #[test]
    fn same_named_genome_files_get_distinct_columns() {
        let labels = column_labels(&[
            PathBuf::from("x/best.json"),
            PathBuf::from("y/best.json"),
            PathBuf::from("z/other.json"),
        ]);
        assert_eq!(labels, vec!["0:best", "1:best", "other"]);
    }

    #[test]
    fn the_default_battery_builds_and_bad_or_duplicate_opponents_are_rejected() {
        let specs: Vec<String> = DEFAULT_BATTERY.iter().map(|&s| s.to_owned()).collect();
        assert_eq!(build_opponents(&specs).unwrap().len(), 6);
        let error = build_opponents(&["nonsense".to_owned()]).err().unwrap();
        assert!(format!("{error:#}").contains("--opponent `nonsense`"));
        let twice = build_opponents(&["lowest-legal".to_owned(), "lowest-legal".to_owned()])
            .err()
            .unwrap();
        assert!(twice.to_string().contains("duplicates"), "{twice}");
    }

    #[test]
    fn a_genome_is_required_and_defaults_are_sensible() {
        assert!(EvaluateArgs::try_parse_from(["cli evaluate"]).is_err());
        let args = EvaluateArgs::try_parse_from(["cli evaluate", "--genome", "g.json"]).unwrap();
        assert_eq!((args.player_count, args.matches, args.rounds), (4, 400, 8));
        assert!(args.opponent.is_empty() && args.json.is_none());
    }
}
```

- [ ] **Step 3: Edit**

In `cli/src/main.rs` (Wire the subcommand in), replace:

```rust
//! docs/ARCHITECTURE.md, "cli".

mod args;
mod output;
mod summary;
mod train;
```

with:

```rust
//! docs/ARCHITECTURE.md, "cli".

mod args;
mod evaluate;
mod output;
mod summary;
mod train;
```

- [ ] **Step 4: Edit**

In `cli/src/main.rs` (Wire the subcommand in), replace:

```rust
    match std::env::args().nth(1).as_deref() {
        Some("train") => return train::run(std::env::args().skip(2)),
        Some("watch") => return watch::run(std::env::args().skip(2)),
        _ => {}
    }
    let args = args::Args::parse();
```

with:

```rust
    match std::env::args().nth(1).as_deref() {
        Some("train") => return train::run(std::env::args().skip(2)),
        Some("watch") => return watch::run(std::env::args().skip(2)),
        Some("evaluate") => return evaluate::run(std::env::args().skip(2)),
        _ => {}
    }
    let args = args::Args::parse();
```

- [ ] **Step 5: Run (expect failure)**

Run: `cargo test -p cli`

Expected: FAIL (compile errors: `render_table`, `EvaluateArgs`, `build_opponents` ... are not defined).

- [ ] **Step 6: Implement**

Insert at the very top of `cli/src/evaluate.rs`, above the `#[cfg(test)]` line:

```rust
//! `cli evaluate`: scores evolved genomes against opponents, including
//! opponents they never trained against. This is the baseline-comparison
//! protocol of the design spec as a command: the same seeds and tables
//! for every genome, so results are directly comparable.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use rayon::prelude::*;
use serde::Serialize;
use sim::training::{evaluate, match_seed, Opponents, Score, TableSpec};

use crate::args::{DeckVariantArg, DuplicateRuleArg, StrategyArg};

/// Opponents when none are given: the strongest hand-written strategies
/// (the training pool's defaults) plus ones a default training run never
/// sees.
pub const DEFAULT_BATTERY: [&str; 6] = [
    "lowest-legal",
    "endgame-denial",
    "adaptive:reading,tempo,bully",
    "hold-back-pairs",
    "greedy-highest",
    "random-legal",
];

/// Score evolved genomes against opponents.
#[derive(Parser, Debug)]
#[command(
    name = "cli evaluate",
    about = "Score evolved genomes against opponents, including ones they never trained against",
    long_about = "Plays each genome at tables made only of one opponent (and at tables mixing all \
the opponents) and reports its mean finishing-role score, +1 (always President) to -1 (always \
last), with its standard error. Every genome plays exactly the same deals, seats and opponents, \
so rows are directly comparable. The matches use a seed stream training never uses."
)]
pub struct EvaluateArgs {
    /// A genome file written by `cli train` (repeat to compare several).
    #[arg(long, required = true, value_name = "PATH")]
    pub genome: Vec<PathBuf>,

    /// An opponent, in `--strategy` syntax (repeat for several; `neat:PATH`
    /// pits a genome against another genome). Default: lowest-legal,
    /// endgame-denial, adaptive:reading,tempo,bully, hold-back-pairs,
    /// greedy-highest, random-legal.
    #[arg(long, value_name = "SPEC")]
    pub opponent: Vec<String>,

    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u8).range(3..=6))]
    pub player_count: u8,

    #[arg(long, value_enum, default_value_t = DeckVariantArg::Single)]
    pub deck_variant: DeckVariantArg,

    #[arg(long, value_enum, default_value_t = DuplicateRuleArg::FirstDealtWins)]
    pub duplicate_rule: DuplicateRuleArg,

    /// Rounds per match.
    #[arg(long, default_value_t = 8, value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(1..))]
    pub rounds: usize,

    /// Matches per opponent.
    #[arg(long, default_value_t = 400, value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(1..))]
    pub matches: usize,

    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Rayon thread-pool size. 0 lets rayon pick its own default.
    #[arg(long, default_value_t = 0)]
    pub threads: usize,

    /// Also write the results as JSON.
    #[arg(long, value_name = "PATH")]
    pub json: Option<PathBuf>,
}

/// One cell of the result table.
#[derive(Debug, Clone, Serialize)]
pub struct Cell {
    pub opponent: String,
    pub mean: f64,
    pub std_error: f64,
    pub matches: usize,
    pub placements: Vec<u64>,
}

impl Cell {
    fn new(opponent: &str, score: Score) -> Self {
        Self {
            opponent: opponent.to_owned(),
            mean: score.mean,
            std_error: score.std_error,
            matches: score.matches,
            placements: score.placements,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GenomeResult {
    pub genome: String,
    /// One cell per opponent, then a last `mixed` cell.
    pub cells: Vec<Cell>,
}

/// The label of the all-opponents-mixed row.
pub const MIXED: &str = "mixed (all opponents)";

/// A column heading for a genome: its file stem, made unique if needed.
fn column_labels(paths: &[PathBuf]) -> Vec<String> {
    let stems: Vec<String> = paths
        .iter()
        .map(|p| {
            p.file_stem()
                .map_or_else(|| "genome".into(), |s| s.to_string_lossy().into_owned())
        })
        .collect();
    stems
        .iter()
        .enumerate()
        .map(|(i, stem)| {
            if stems.iter().filter(|s| *s == stem).count() > 1 {
                format!("{i}:{stem}")
            } else {
                stem.clone()
            }
        })
        .collect()
}

/// Renders results as a table: one row per opponent plus the mixed row,
/// one column per genome. Pure, so it is directly testable.
#[must_use]
pub fn render_table(labels: &[String], results: &[GenomeResult]) -> String {
    let rows: Vec<&str> = results[0]
        .cells
        .iter()
        .map(|c| c.opponent.as_str())
        .collect();
    let row_width = rows
        .iter()
        .map(|r| r.len())
        .max()
        .unwrap_or(8)
        .max("opponent".len());
    let column_width = labels.iter().map(String::len).max().unwrap_or(8).max(14);
    let mut text = format!("{:<row_width$}", "opponent");
    for label in labels {
        let _ = write!(text, "  {label:>column_width$}");
    }
    text.push('\n');
    for (row, name) in rows.iter().enumerate() {
        let _ = write!(text, "{name:<row_width$}");
        for result in results {
            let cell = &result.cells[row];
            let _ = write!(
                text,
                "  {:>column_width$}",
                format!("{:+.3} ±{:.3}", cell.mean, cell.std_error)
            );
        }
        text.push('\n');
    }
    text
}

fn build_opponents(specs: &[String]) -> anyhow::Result<Vec<(String, Arc<dyn sim::Strategy>)>> {
    let mut built: Vec<(String, Arc<dyn sim::Strategy>)> = Vec::new();
    for spec in specs {
        let strategy: Arc<dyn sim::Strategy> = spec
            .parse::<StrategyArg>()
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("invalid --opponent `{spec}`"))?
            .build();
        let name = sim::Strategy::name(&*strategy).to_owned();
        anyhow::ensure!(
            built.iter().all(|(n, _)| *n != name),
            "--opponent `{spec}` duplicates `{name}`, which is already listed"
        );
        built.push((name, strategy));
    }
    Ok(built)
}

pub fn run(raw_args: impl Iterator<Item = String>) -> anyhow::Result<()> {
    let args = EvaluateArgs::parse_from(std::iter::once("cli evaluate".to_owned()).chain(raw_args));
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .context("failed to configure thread pool")?;
    }
    let specs: Vec<String> = if args.opponent.is_empty() {
        DEFAULT_BATTERY.iter().map(|&s| s.to_owned()).collect()
    } else {
        args.opponent.clone()
    };
    let opponents = build_opponents(&specs)?;
    let candidates: Vec<(String, Arc<dyn sim::Strategy>)> = args
        .genome
        .iter()
        .map(|path| {
            let strategy = sim::NeatStrategy::from_file(path)
                .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
            Ok((
                path.display().to_string(),
                Arc::new(strategy) as Arc<dyn sim::Strategy>,
            ))
        })
        .collect::<anyhow::Result<_>>()?;

    let table = TableSpec {
        player_count: args.player_count,
        deck_variant: args.deck_variant.into(),
        duplicate_rule: args.duplicate_rule.into(),
        rounds: args.rounds,
    };
    // A seed stream that training (generation streams, fixed and held-out
    // sets) never uses.
    let seeds: Vec<u64> = (0..args.matches as u64)
        .map(|i| match_seed(args.seed, u64::MAX - 3, i))
        .collect();
    let pool: Vec<Arc<dyn sim::Strategy>> = opponents.iter().map(|(_, s)| s.clone()).collect();

    let results: Vec<GenomeResult> = candidates
        .iter()
        .map(|(path, candidate)| {
            let mut cells: Vec<Cell> = opponents
                .par_iter()
                .map(|(name, strategy)| {
                    Cell::new(
                        name,
                        evaluate(candidate, &table, Opponents::Only(strategy), &seeds),
                    )
                })
                .collect();
            cells.push(Cell::new(
                MIXED,
                evaluate(candidate, &table, Opponents::Mixed(&pool), &seeds),
            ));
            GenomeResult {
                genome: path.clone(),
                cells,
            }
        })
        .collect();

    println!(
        "{} players, {:?} deck, {:?}, {} matches x {} rounds per cell, seed {}",
        args.player_count,
        args.deck_variant,
        args.duplicate_rule,
        args.matches,
        args.rounds,
        args.seed
    );
    println!("score: +1 always President .. -1 always last; 0 is even\n");
    print!("{}", render_table(&column_labels(&args.genome), &results));
    if let Some(path) = &args.json {
        let text = serde_json::to_string_pretty(&serde_json::json!({
            "player_count": args.player_count,
            "rounds": args.rounds,
            "matches": args.matches,
            "seed": args.seed,
            "results": results,
        }))?;
        std::fs::write(path, text).with_context(|| format!("cannot write {}", path.display()))?;
    }
    Ok(())
}
```

- [ ] **Step 7: Run (expect success)**

Run: `cargo fmt -p cli && cargo test -p cli`

Expected: PASS (4 unit tests: the table has a row per opponent plus mixed and a column per genome, same-named files get distinct columns, the default battery builds and bad or duplicate opponents are rejected, argument defaults; 4 smoke tests: two genomes against two opponents with JSON output, identical output for 1 and 3 threads, a genome against another genome, operator mistakes fail clearly with no panic).

- [ ] **Step 8: Commit**

```bash
git add cli
git commit -F - <<'EOF'
cli: add `cli evaluate`, scoring genomes against any opponents (Phase 10e)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F3sC3JdcgTFAox87bBuikA
EOF
```


### Task 3: Dashboard: hall-of-fame line and champion-selection note

**Files:**
- Modify: `web/assets/app.js`

**Interfaces:**
- Consumes: `GenerationEvent::hall_score` and `ChampionStats::training_rank` (Task 1).
- Produces: The opponents chart takes opponent names from *every* generation (an opponent that leaves the pool keeps its line and colour, which also fixes a 10d review note) and adds a dashed `hall of fame (past champions)` line when the run has a hall; the network note says when the fixed matches preferred the champion to genomes that looked better in training.

- [ ] **Step 1: Edit**

In `web/assets/app.js`, replace:

```

function renderOpponents() {
  const events = state.events;
  const names = events.length ? events[events.length - 1].opponents.map((o) => o.name) : [];
  lineChart("opponents-chart", "opponents-readout", {
    xs: events.map((e) => e.generation), ref: 0, signed: true,
    series: names.map((name, k) => ({
      name, color: `var(--s${(k % 6) + 1})`,
      values: events.map((e) => e.opponents.find((o) => o.name === name)?.score.mean ?? null),
    })),
  });
}

function renderSpecies() {
```

with:

```

function renderOpponents() {
  const events = state.events;
  // Names from every generation (not just the last), so an opponent that
  // left the pool keeps its line and colour.
  const names = [];
  for (const e of events) for (const o of e.opponents) if (!names.includes(o.name)) names.push(o.name);
  const series = names.map((name, k) => ({
    name, color: `var(--s${(k % 6) + 1})`,
    values: events.map((e) => e.opponents.find((o) => o.name === name)?.score.mean ?? null),
  }));
  if (events.some((e) => e.hall_score)) {
    series.push({ name: "hall of fame (past champions)", color: "var(--s6)", dashed: true, values: events.map((e) => (e.hall_score ? e.hall_score.mean : null)) });
  }
  lineChart("opponents-chart", "opponents-readout", { xs: events.map((e) => e.generation), ref: 0, signed: true, series });
}

function renderSpecies() {
```

- [ ] **Step 2: Edit**

In `web/assets/app.js`, replace:

```
    if (generation !== state.networkGeneration) return; // the user moved on
    renderNetwork($("network"), file, { showDisabled: state.showDisabled });
    $("network-note").textContent = event
      ? `Champion of generation ${generation}: ${event.champion.hidden_nodes} hidden nodes, ${event.champion.enabled_connections} enabled connections, fixed-match score ${fmt(event.champion.reeval.mean, 3, true)}${event.champion.is_new_best ? " (new best)" : ""}.`
      : "";
  } catch (error) {
    $("network").innerHTML = `<p class="muted">no genome for generation ${generation} yet</p>`;
```

with:

```
    if (generation !== state.networkGeneration) return; // the user moved on
    renderNetwork($("network"), file, { showDisabled: state.showDisabled });
    $("network-note").textContent = event
      ? `Champion of generation ${generation}: ${event.champion.hidden_nodes} hidden nodes, ${event.champion.enabled_connections} enabled connections, fixed-match score ${fmt(event.champion.reeval.mean, 3, true)}${event.champion.is_new_best ? " (new best)" : ""}${event.champion.training_rank > 0 ? `; the fixed matches preferred it to ${event.champion.training_rank} genome(s) that looked better in training` : ""}.`
      : "";
  } catch (error) {
    $("network").innerHTML = `<p class="muted">no genome for generation ${generation} yet</p>`;
```

- [ ] **Step 3: Run (expect success)**

Run: `node --check web/assets/app.js`

Expected: No output.

- [ ] **Step 4: Run (expect success)**

Run: `cargo fmt --all && cargo test -p web`

Expected: PASS.

- [ ] **Step 5: Look at it (real browser)**

**Look at it (real browser).** Run a short training with a hall (`cli train --out /tmp/dash-hall --serve 8790 --population 60 --generations 24 --matches-per-genome 30 --reeval-matches 60 --rounds 6 --hall-of-fame 3 --hall-interval 3 --quiet`) and open the page: the opponents chart shows the three opponent lines plus the dashed hall line starting once the hall has members; the network note for some generation says the fixed matches preferred the champion to genomes that looked better in training; no console errors.

- [ ] **Step 6: Commit**

```bash
git add web
git commit -F - <<'EOF'
web: show the hall of fame and champion re-selection in the dashboard (Phase 10e)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F3sC3JdcgTFAox87bBuikA
EOF
```


### Task 4: Record the learning-option experiments and set the defaults

**Files:**
- Create: `docs/baselines/neat-v1/experiments.md`, `docs/baselines/neat-v1/experiments/run_experiments.sh`, `docs/baselines/neat-v1/experiments/analyze.py`

**Why:** The defaults were chosen from 15 + 7 real training runs (40 minutes of compute on 8 cores), each champion scored on opponents it never trained against. The honest summary: generalization is not a problem (every configuration scores +0.90 to +0.92 on unseen opponents), and the combined options are better by about 0.016 (1.3 standard errors over 5 seeds), so they were adopted on principle plus a never-worse measurement, not on a proof. The hall of fame showed no benefit and stays off.

**Interfaces:**
- Consumes: `cli train` with the Task 1 options and `cli evaluate` (Task 2).
- Produces: A reproducible record of which learning options were measured, with what result, and why the CLI defaults are `--champion-candidates 5 --weight-power 0.2` and `--hall-of-fame 0`.

- [ ] **Step 1: Create file**

Create `docs/baselines/neat-v1/experiments.md`:

```markdown
# Phase 10e learning-option experiments

Question: do the new learning options (top-k champion selection, a hall of
fame, a smaller weight-perturbation power) produce better champions, and
do champions trained against three fixed opponents generalize to opponents
they never saw?

**Setup.** Population 100, 60 generations, 40 matches x 6 rounds per
genome, 100 re-evaluation matches, 4 players, training pool
`lowest-legal`, `endgame-denial`, `adaptive:reading,tempo,bully`. Each
run's `best.json` is then scored by `cli evaluate` (600 matches x 8 rounds
per cell, seed 99, a stream training never uses) against the three
training opponents ("trained-3") and against `hold-back-pairs`,
`greedy-highest` and `random-legal`, which training never saw
("unseen-3"). Scores are mean finishing-role scores, +1 always President
to -1 always last. Reproduce with `experiments/run_experiments.sh`.

## Five configurations, three seeds each

| configuration | trained-3 | unseen-3 | mixed |
|---|---|---|---|
| baseline (1 candidate, no hall, power 0.5) | +0.597 | +0.907 | +0.672 |
| top-5 champion selection | +0.619 | +0.904 | +0.684 |
| hall of fame (3 members, every 5 generations) | +0.586 | +0.915 | +0.671 |
| weight power 0.2 | +0.614 | +0.921 | +0.703 |
| weight power 0.1 | +0.603 | +0.918 | +0.699 |

Seed-to-seed spread (one standard deviation) of a single configuration on
trained-3 is about 0.012 to 0.03, so differences of 0.01 to 0.02 between
rows are within noise.

## Baseline against the two best options combined, five seeds

| configuration | trained-3 | unseen-3 | mixed |
|---|---|---|---|
| baseline | +0.603 (sd 0.012) | +0.907 (sd 0.017) | +0.679 (sd 0.017) |
| top-5 selection + weight power 0.2 | +0.619 (sd 0.022) | +0.912 (sd 0.023) | +0.697 (sd 0.022) |

Paired difference on trained-3 (same seeds): **+0.016, standard error
0.012** (per seed +0.048, -0.007, +0.032, +0.025, -0.016).

## What this does and does not show

- **Generalization is not a problem here.** Every configuration scores
  +0.90 to +0.92 against opponents it never trained against, far above
  its score against the trained opponents; there is no sign of overfitting
  to the three-opponent pool.
- **The learning options help a little, not conclusively.** The combined
  configuration is better by about 0.016 (1.3 standard errors) on the
  trained opponents, 0.018 in mixed tables and 0.005 on unseen ones. It was
  adopted as the default because top-k selection is principled (the
  training fitness of a generation's best genome is dominated by noise and
  re-scoring a few candidates on fixed matches removes that), costs about
  12% more time per generation, and was never worse; but five seeds do not
  prove it.
- **The hall of fame showed no benefit** against this pool and stays off by
  default (`--hall-of-fame N` enables it). It may matter against pools that
  include evolved opponents.
- Every champion here already beats the strongest hand-written strategy
  (`adaptive:reading,tempo,bully`) by about +0.55 to +0.6 in a table of
  one champion and three copies of it.
```

- [ ] **Step 2: Create file**

Create `docs/baselines/neat-v1/experiments/run_experiments.sh`:

```
#!/usr/bin/env bash
# Reproduces the Phase 10e learning-option experiments (see ../experiments.md).
# Usage (from the repository root, after `cargo build --release -p cli`):
#   docs/baselines/neat-v1/experiments/run_experiments.sh [outdir]
# Needs roughly 40 minutes on 8 cores: 5 configurations x 3 seeds, then 2
# more baseline seeds and 5 seeds of the combined configuration, each a
# 60-generation training run followed by a held-out evaluation.
set -euo pipefail
CLI=${CLI:-target/release/cli}
OUT=${1:-/tmp/neat-experiments}
mkdir -p "$OUT"
COMMON=(--population 100 --generations 60 --matches-per-genome 40 --reeval-matches 100 --rounds 6 --quiet)

run() { # name seed extra-args...
  local name=$1 seed=$2; shift 2
  local dir=$OUT/${name}_$seed
  [ -f "$dir/best.json" ] && return
  rm -rf "$dir"
  # Explicit flags: the experiment must not depend on today's defaults.
  "$CLI" train --out "$dir" --seed "$seed" "${COMMON[@]}" "$@" > "$dir.log" 2>&1
  # A battery that includes opponents training never saw, on a seed stream
  # training never uses.
  "$CLI" evaluate --genome "$dir/best.json" --matches 600 --seed 99 --json "$dir.eval.json" > "$dir.eval.txt"
  echo "done $name seed $seed"
}

for seed in 1 2 3 4 5; do
  run base  "$seed" --champion-candidates 1 --hall-of-fame 0 --weight-power 0.5
  run combo "$seed" --champion-candidates 5 --hall-of-fame 0 --weight-power 0.2
done
for seed in 1 2 3; do
  run topk "$seed" --champion-candidates 5 --hall-of-fame 0 --weight-power 0.5
  run hof  "$seed" --champion-candidates 1 --hall-of-fame 3 --hall-interval 5 --weight-power 0.5
  run wp02 "$seed" --champion-candidates 1 --hall-of-fame 0 --weight-power 0.2
  run wp01 "$seed" --champion-candidates 1 --hall-of-fame 0 --weight-power 0.1
done
python3 "$(dirname "$0")/analyze.py" "$OUT"
```

- [ ] **Step 3: Create file**

Create `docs/baselines/neat-v1/experiments/analyze.py`:

```
"""Summarises the experiment runs: mean score of each configuration's best
champion against the opponents it trained on and against opponents it never
saw, from the `*.eval.json` files written by run_experiments.sh."""
import json
import os
import statistics as st
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/neat-experiments"
TRAINED = ["LowestLegal", "EndgameDenial", "Adaptive(reading,tempo,bully)"]
LABELS = {
    "base": "baseline (1 candidate, no hall, power 0.5)",
    "topk": "top-5 champion selection",
    "hof": "hall of fame (3, every 5)",
    "wp02": "weight power 0.2",
    "wp01": "weight power 0.1",
    "combo": "top-5 selection + power 0.2",
}


def cells(name, seed):
    path = f"{OUT}/{name}_{seed}.eval.json"
    if not os.path.exists(path):
        return None
    return {c["opponent"]: c["mean"] for c in json.load(open(path))["results"][0]["cells"]}


def summary(name, seed):
    c = cells(name, seed)
    trained = st.mean(c[n] for n in TRAINED)
    unseen = st.mean(v for n, v in c.items() if n not in TRAINED and not n.startswith("mixed"))
    return trained, unseen, c["mixed (all opponents)"]


for name, label in LABELS.items():
    rows = [summary(name, s) for s in range(1, 6) if cells(name, s)]
    if not rows:
        continue
    parts = []
    for i, what in enumerate(("trained-3", "unseen-3", "mixed")):
        values = [r[i] for r in rows]
        spread = st.stdev(values) if len(values) > 1 else 0.0
        parts.append(f"{what} {st.mean(values):+.3f} (sd {spread:.3f})")
    print(f"{label:44s} {len(rows)} seeds   " + "   ".join(parts))

paired = [
    summary("combo", s)[0] - summary("base", s)[0]
    for s in range(1, 6)
    if cells("combo", s) and cells("base", s)
]
if len(paired) > 1:
    se = st.stdev(paired) / len(paired) ** 0.5
    print(f"\npaired difference combo - base on the trained opponents: {st.mean(paired):+.3f} (standard error {se:.3f}, {len(paired)} seeds)")
```

- [ ] **Step 4: Run (expect success)**

Run: `bash -n docs/baselines/neat-v1/experiments/run_experiments.sh && python3 -m py_compile docs/baselines/neat-v1/experiments/analyze.py`

Expected: No output (valid syntax). The experiments themselves were run during preparation (about 40 minutes); re-running them is optional.

- [ ] **Step 5: Run (expect success)**

Run: `chmod +x docs/baselines/neat-v1/experiments/run_experiments.sh`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add docs/baselines/neat-v1
git commit -F - <<'EOF'
docs: record the Phase 10e learning-option experiments

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F3sC3JdcgTFAox87bBuikA
EOF
```


### Task 5: The NEAT v1 baseline: a committed champion and its comparison with the pre-NEAT strategies

**Files:**
- Create: `docs/baselines/neat-v1/champion.json`, `docs/baselines/neat-v1/run.sh`, `docs/baselines/neat-v1/README.md`, `docs/baselines/neat-v1/summaries/*` (generated)

**Interfaces:**
- Consumes: The built release `cli` (Tasks 1-2), the champion trained during preparation.
- Produces: `docs/baselines/neat-v1/champion.json` (plays with `--strategy neat:PATH`), `run.sh` (the pre-NEAT protocol: seed 1000, 2000 matches x 10 rounds at 3-6 players, one champion against copies of each strong opponent, plus `cli evaluate` on a fresh seed stream against the whole battery), the recorded `summaries/` (20 files), and a README with the measured results and caveats. Result: trained at 4 players only, the champion beats every opponent at every table size from 3 to 6 players (+0.42 to +0.96 on a -1..+1 scale) and is President in 57% of rounds against three copies of the best hand-written strategy (33% for that strategy against three `LowestLegal`s in the pre-NEAT baseline) while finishing last in 2%.

- [ ] **Step 1: Create file**

Create `docs/baselines/neat-v1/champion.json` (The champion: the `best.json` of the v1 run (seed 1, population 150, 120 generations, held-out score +0.624 +- 0.013)):

```
{
  "format_version": 1,
  "feature_set_version": 1,
  "feature_names": [
    "is_pass",
    "combo_size",
    "top_strength",
    "hand_below_top",
    "uses_top_group",
    "splits_group",
    "empties_hand",
    "hand_left_after",
    "is_leading",
    "table_combo_size",
    "hand_size",
    "grouped_fraction",
    "singleton_fraction",
    "active_opponents",
    "min_opponent_hand",
    "mean_opponent_hand",
    "opponent_close",
    "unseen_outranking",
    "unseen_beaters",
    "opponents_locked_out"
  ],
  "genome": {
    "num_inputs": 20,
    "nodes": [
      {
        "id": 0,
        "kind": "Input"
      },
      {
        "id": 1,
        "kind": "Input"
      },
      {
        "id": 2,
        "kind": "Input"
      },
      {
        "id": 3,
        "kind": "Input"
      },
      {
        "id": 4,
        "kind": "Input"
      },
      {
        "id": 5,
        "kind": "Input"
      },
      {
        "id": 6,
        "kind": "Input"
      },
      {
        "id": 7,
        "kind": "Input"
      },
      {
        "id": 8,
        "kind": "Input"
      },
      {
        "id": 9,
        "kind": "Input"
      },
      {
        "id": 10,
        "kind": "Input"
      },
      {
        "id": 11,
        "kind": "Input"
      },
      {
        "id": 12,
        "kind": "Input"
      },
      {
        "id": 13,
        "kind": "Input"
      },
      {
        "id": 14,
        "kind": "Input"
      },
      {
        "id": 15,
        "kind": "Input"
      },
      {
        "id": 16,
        "kind": "Input"
      },
      {
        "id": 17,
        "kind": "Input"
      },
      {
        "id": 18,
        "kind": "Input"
      },
      {
        "id": 19,
        "kind": "Input"
      },
      {
        "id": 20,
        "kind": "Bias"
      },
      {
        "id": 21,
        "kind": "Output"
      },
      {
        "id": 28,
        "kind": "Hidden"
      },
      {
        "id": 31,
        "kind": "Hidden"
      },
      {
        "id": 45,
        "kind": "Hidden"
      },
      {
        "id": 81,
        "kind": "Hidden"
      },
      {
        "id": 103,
        "kind": "Hidden"
      }
    ],
    "connections": [
      {
        "innovation": 0,
        "from": 0,
        "to": 21,
        "weight": -1.081300149962809,
        "enabled": true
      },
      {
        "innovation": 1,
        "from": 1,
        "to": 21,
        "weight": -0.9770625522339687,
        "enabled": true
      },
      {
        "innovation": 2,
        "from": 2,
        "to": 21,
        "weight": -0.3112834277840678,
        "enabled": true
      },
      {
        "innovation": 3,
        "from": 3,
        "to": 21,
        "weight": -1.1351122228323578,
        "enabled": true
      },
      {
        "innovation": 4,
        "from": 4,
        "to": 21,
        "weight": 1.0521463781835747,
        "enabled": false
      },
      {
        "innovation": 5,
        "from": 5,
        "to": 21,
        "weight": -0.391292314038183,
        "enabled": true
      },
      {
        "innovation": 6,
        "from": 6,
        "to": 21,
        "weight": 1.6003870727343934,
        "enabled": false
      },
      {
        "innovation": 7,
        "from": 7,
        "to": 21,
        "weight": -0.9181187610938526,
        "enabled": true
      },
      {
        "innovation": 8,
        "from": 8,
        "to": 21,
        "weight": -0.7701550735115897,
        "enabled": true
      },
      {
        "innovation": 9,
        "from": 9,
        "to": 21,
        "weight": 0.644779017963562,
        "enabled": true
      },
      {
        "innovation": 10,
        "from": 10,
        "to": 21,
        "weight": -0.9841063226439261,
        "enabled": true
      },
      {
        "innovation": 11,
        "from": 11,
        "to": 21,
        "weight": 0.9001405382884349,
        "enabled": true
      },
      {
        "innovation": 12,
        "from": 12,
        "to": 21,
        "weight": -1.625175889002219,
        "enabled": true
      },
      {
        "innovation": 13,
        "from": 13,
        "to": 21,
        "weight": 0.7600099334862,
        "enabled": true
      },
      {
        "innovation": 14,
        "from": 14,
        "to": 21,
        "weight": -0.4536825825626004,
        "enabled": true
      },
      {
        "innovation": 15,
        "from": 15,
        "to": 21,
        "weight": 0.04893828229993902,
        "enabled": true
      },
      {
        "innovation": 16,
        "from": 16,
        "to": 21,
        "weight": 0.25346077076888507,
        "enabled": true
      },
      {
        "innovation": 17,
        "from": 17,
        "to": 21,
        "weight": 0.05692822365185135,
        "enabled": true
      },
      {
        "innovation": 18,
        "from": 18,
        "to": 21,
        "weight": -0.24951340628020297,
        "enabled": true
      },
      {
        "innovation": 19,
        "from": 19,
        "to": 21,
        "weight": 0.7336894628176598,
        "enabled": true
      },
      {
        "innovation": 20,
        "from": 20,
        "to": 21,
        "weight": -1.1494961655484162,
        "enabled": true
      },
      {
        "innovation": 35,
        "from": 9,
        "to": 28,
        "weight": -0.0015769218969280269,
        "enabled": true
      },
      {
        "innovation": 36,
        "from": 28,
        "to": 21,
        "weight": 0.27460571874762035,
        "enabled": true
      },
      {
        "innovation": 41,
        "from": 15,
        "to": 31,
        "weight": -0.3403164791956872,
        "enabled": true
      },
      {
        "innovation": 42,
        "from": 31,
        "to": 21,
        "weight": 0.197587623640608,
        "enabled": true
      },
      {
        "innovation": 49,
        "from": 11,
        "to": 31,
        "weight": 0.3157637308513288,
        "enabled": true
      },
      {
        "innovation": 113,
        "from": 6,
        "to": 28,
        "weight": 0.40879557785146714,
        "enabled": true
      },
      {
        "innovation": 123,
        "from": 14,
        "to": 45,
        "weight": 0.3547314545170369,
        "enabled": true
      },
      {
        "innovation": 124,
        "from": 45,
        "to": 21,
        "weight": 0.07343490788484626,
        "enabled": true
      },
      {
        "innovation": 177,
        "from": 0,
        "to": 28,
        "weight": 0.09347846740263105,
        "enabled": true
      },
      {
        "innovation": 346,
        "from": 12,
        "to": 31,
        "weight": 0.7906655338585346,
        "enabled": true
      },
      {
        "innovation": 403,
        "from": 14,
        "to": 81,
        "weight": 0.8117640301711837,
        "enabled": false
      },
      {
        "innovation": 404,
        "from": 81,
        "to": 45,
        "weight": 0.6388020924987192,
        "enabled": true
      },
      {
        "innovation": 415,
        "from": 28,
        "to": 45,
        "weight": 0.22310910300885156,
        "enabled": true
      },
      {
        "innovation": 556,
        "from": 14,
        "to": 103,
        "weight": 1.0319900477780746,
        "enabled": true
      },
      {
        "innovation": 557,
        "from": 103,
        "to": 81,
        "weight": 0.25734180552224695,
        "enabled": true
      }
    ]
  }
}
```

- [ ] **Step 2: Create file**

Create `docs/baselines/neat-v1/run.sh`:

```
#!/usr/bin/env bash
# Compares the committed champion with the hand-written strategies at tables
# of 3-6 players, with the same seeds, match counts and round counts as
# docs/baselines/pre-neat/run.sh, so the results are directly comparable.
# Usage (from the repository root, after `cargo build --release -p cli`):
#   docs/baselines/neat-v1/run.sh [outdir]
# Output is seeded and independent of the thread count.
set -euo pipefail
CLI=${CLI:-target/release/cli}
GENOME=${GENOME:-docs/baselines/neat-v1/champion.json}
OUT=${1:-docs/baselines/neat-v1/summaries}
SEED=1000
MATCHES=2000
ROUNDS=10
THREADS=${THREADS:-4}
mkdir -p "$OUT"
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

for n in 3 4 5 6; do
  # One champion against n-1 copies of each opponent. (The champion sits in
  # every seat in turn: sim rotates the seating across the batch.)
  for opponent in lowest-legal endgame-denial "adaptive:reading,tempo,bully"; do
    specs=("neat:$GENOME")
    for ((i = 1; i < n; i++)); do specs+=("$opponent"); done
    args=()
    for spec in "${specs[@]}"; do args+=(--strategy "$spec"); done
    "$CLI" --player-count "$n" --matches "$MATCHES" --rounds "$ROUNDS" \
      --seed "$SEED" --threads "$THREADS" --output "$TMP" "${args[@]}" \
      > "$OUT/champion-vs-${opponent//[:,=]/_}_${n}p.txt"
  done
  # The same champion on a seed stream training never used, against the
  # whole battery including opponents it never trained against.
  "$CLI" evaluate --genome "$GENOME" --player-count "$n" --matches 600 --seed 99 \
    --threads "$THREADS" --json "$OUT/evaluate_${n}p.json" > "$OUT/evaluate_${n}p.txt"
done
```

- [ ] **Step 3: Run (expect success)**

Run: `chmod +x docs/baselines/neat-v1/run.sh && cargo build --release -p cli`

Expected: PASS.

- [ ] **Step 4: Run (expect success)**

Run: `docs/baselines/neat-v1/run.sh`

Expected: About a minute. Writes `summaries/` (12 `champion-vs-*` files, 4 `evaluate_*p.txt`, 4 `evaluate_*p.json`).

- [ ] **Step 5: Create file**

Create `docs/baselines/neat-v1/README.md`:

```markdown
# NEAT v1 baseline

The first evolved champion, committed so later work can be compared
against it the way `docs/baselines/pre-neat` records the hand-written
strategies. Nothing here changes the pre-NEAT numbers.

- `champion.json`: the genome (a `GenomeFile`; plays with
  `--strategy neat:docs/baselines/neat-v1/champion.json`).
- `run.sh`: re-measures it with the pre-NEAT protocol (seed 1000, 2000
  matches x 10 rounds) at 3-6 players, plus `cli evaluate` on a seed
  stream training never used. Output is seeded and independent of
  `--threads`; `summaries/` holds the recorded output.
- `experiments.md` and `experiments/`: how the learning defaults were
  chosen.

## How it was trained

```bash
target/release/cli train --out runs/v1 --seed 1 --population 150 \
  --generations 120 --matches-per-genome 80 --reeval-matches 200 \
  --rounds 8 --champion-candidates 5 --weight-power 0.2 --quiet
```

4 players, single deck, opponent pool `lowest-legal`, `endgame-denial`,
`adaptive:reading,tempo,bully`. 120 generations took 11.5 minutes on 8
cores. `champion.json` is the run's `best.json` (generation 69; 5 hidden nodes and
33 enabled connections). Held-out score
against the mixed training pool: **+0.624 +- 0.013**. Re-scoring the top 5
genomes on the fixed matches picked a genome other than the training-best
one in 75 of 120 generations, which is how noisy a generation's training
fitness is.

## What it does (`summaries/evaluate_*p.txt`, 600 matches x 8 rounds per cell)

Mean finishing-role score, +1 always President to -1 always last, one
champion against copies of one opponent. Trained on 4 players, evaluated
at every table size without retraining:

| opponent | 3 players | 4 players | 5 players | 6 players |
|---|---|---|---|---|
| LowestLegal | +0.468 | +0.630 | +0.545 | +0.421 |
| EndgameDenial | +0.534 | +0.692 | +0.643 | +0.586 |
| Adaptive(reading,tempo,bully) | +0.468 | +0.609 | +0.592 | +0.555 |
| HoldBackPairs (never trained against) | +0.911 | +0.917 | +0.881 | +0.838 |
| GreedyHighest (never trained against) | +0.960 | +0.939 | +0.927 | +0.895 |
| RandomLegal (never trained against) | +0.960 | +0.914 | +0.826 | +0.744 |
| all of the above mixed | +0.639 | +0.708 | +0.674 | +0.596 |

Standard errors are 0.003-0.016; see the `evaluate_*p.txt` files.

## Against the pre-NEAT baseline (`summaries/champion-vs-*`)

4 players, 2000 matches x 10 rounds (20000 rounds), one champion against
three copies of one opponent. A player at random would be President in
5000 rounds and last in 5000.

| opponent (x3) | champion President | champion last | opponent President | opponent last |
|---|---|---|---|---|
| LowestLegal | 11477 (57%) | 414 (2%) | 8523 (total, 3 seats) | 19586 (total, 3 seats) |
| EndgameDenial | 12512 (63%) | 255 (1%) | 7488 (total) | 19745 (total) |
| Adaptive(reading,tempo,bully) | 11326 (57%) | 452 (2%) | 8674 (total) | 19548 (total) |

For scale, in `docs/baselines/pre-neat` the best hand-written strategy,
`adaptive:reading,tempo,bully`, against three `LowestLegal` players was
President in 6641 rounds (33%) and last in 3507 (18%).

The champion passes voluntarily (a legal play existed) in about 8-10% of
its turns; every hand-written strategy except `HoldBackPairs` and
`RandomLegal` never does.

## Caveats

- One champion from one seed, trained against one pool. It says what
  this approach reaches, not an upper bound; `experiments.md` shows
  seed-to-seed spread of about 0.01-0.03 on these scores.
- It was trained at 4 players. The 3-, 5- and 6-player rows are
  generalization, and its edge shrinks at 6 players against
  `LowestLegal` (+0.421).
- Opponent tables here are identical copies, so no table-position bias
  enters (see the caveats in `docs/baselines/pre-neat/README.md`).
- Scores of the pre-NEAT strategies and of this champion come from
  different table compositions; the rows above only compare each against
  the same opponents where stated.
```

- [ ] **Step 6: Real scenario**

Save as `check_reproducible.sh` (anywhere outside the repository, for example the system temp directory) and run it from the repository root with `bash check_reproducible.sh`:

```bash
#!/usr/bin/env bash
# The recorded summaries must be reproducible: re-run the protocol with a
# different thread count into a temporary directory and compare. The only
# allowed difference is the thread count printed in the first line of the
# flat simulator's output.
set -euo pipefail
export LC_ALL=C
TMP=$(mktemp -d)
THREADS=1 docs/baselines/neat-v1/run.sh "$TMP" > /dev/null
normalize() { sed -E 's/, [0-9]+ threads//' "$1"; }
status=0
for file in docs/baselines/neat-v1/summaries/*; do
  name=$(basename "$file")
  if ! diff <(normalize "$file") <(normalize "$TMP/$name") > /dev/null; then
    echo "DIFFERS: $name"; status=1
  fi
done
[ "$(ls docs/baselines/neat-v1/summaries | wc -l)" = "$(ls "$TMP" | wc -l)" ] || { echo "file counts differ"; status=1; }
echo "compared $(ls "$TMP" | wc -l) files; status $status"
exit $status
```

Expected: `compared 20 files; status 0`: re-running with 1 thread reproduces every recorded summary (the genome file, the match seeds and the engine are deterministic).

- [ ] **Step 7: Commit**

```bash
git add docs/baselines/neat-v1
git commit -F - <<'EOF'
docs: commit the NEAT v1 champion and its comparison with the hand-written strategies (Phase 10e)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F3sC3JdcgTFAox87bBuikA
EOF
```


### Task 6: Docs and the full gate

**Files:**
- Modify: `docs/ROADMAP.md`, `docs/BUILDING.md`, `docs/ARCHITECTURE.md`, `docs/superpowers/specs/2026-10-08-neat-engine-design.md`

**Interfaces:**
- Consumes: Everything above.
- Produces: Docs that match the code; a clean phase-done gate. (The how-to-train guide is the next deliverable.)

- [ ] **Step 1: Edit**

In `docs/ROADMAP.md` (Roadmap), replace:

```markdown
10e: opponent pool
```

with:

```markdown
10e (done): opponent pool
```

- [ ] **Step 2: Edit**

In `docs/superpowers/specs/2026-10-08-neat-engine-design.md`, replace:

```markdown
## 8. Determinism and performance
```

with:

```markdown
## 7d. As built in Phase 10e

- **Learning options** (`cli train`): `--champion-candidates N` re-scores
  the N genomes with the best training fitness on the fixed matches and
  takes the best as the generation's champion (default 5; the event's
  `training_rank` says where it ranked by training fitness: in the v1 run
  it was not the training best in 75 of 120 generations);
  `--hall-of-fame N --hall-interval K` keeps frozen past champions as extra
  training opponents (default off; persisted in the checkpoint, resumable;
  per-opponent reporting stays on the fixed pool and a `hall_score`
  reports the champion against the hall); `--weight-power P` is the size of
  a weight perturbation (default 0.2).
- **`cli evaluate`**: the baseline-comparison protocol (section 9) as a
  command: genomes against any opponents on a seed stream training never
  uses, one row per opponent plus a mixed row, optional JSON. Genomes can
  face each other with `neat:PATH` opponents.
- **Evidence:** `docs/baselines/neat-v1/experiments.md`: five
  configurations over three seeds and a 5-seed comparison of the combined
  options. All configurations generalize to opponents never trained
  against (+0.90 to +0.92); the combined options are better by about 0.016
  (1.3 standard errors) and became the defaults; the hall of fame showed no
  benefit against this pool.
- **Baseline comparison** (`docs/baselines/neat-v1/`): the first
  committed champion and the recorded comparison with the pre-NEAT
  strategies at 3-6 players (success criteria of section 9 met: it beats
  `lowest-legal`, `endgame-denial` and `adaptive:reading,tempo,bully` with
  differences far outside the seed-to-seed spread).
- **Not built:** a weak-to-strong opponent curriculum and matches growing
  over generations (the champion already beats every opponent, so neither
  has a measured need), complexity pressure in ties, the role feature and
  evolving the exchange step.

## 8. Determinism and performance
```

- [ ] **Step 3: Edit**

In `docs/BUILDING.md`, replace:

```markdown
`cli train --help` lists every option.
```

with:

```markdown
Compare genomes (or a genome against hand-written strategies) with
`cli evaluate --genome runs/first/best.json` (a table of mean finishing-role
scores against a default battery of opponents; add `--opponent SPEC` to
choose them, repeat `--genome` to compare several). Learning options:
`--champion-candidates` (default 5), `--hall-of-fame` (default off) and
`--weight-power` (default 0.2); the evidence for the defaults is in
`docs/baselines/neat-v1/experiments.md`, and `docs/baselines/neat-v1`
holds a committed champion with its comparison against the pre-NEAT
strategies.

`cli train --help` lists every option.
```

- [ ] **Step 4: Edit**

In `docs/ARCHITECTURE.md`, replace:

```markdown
`sim::training` (Phase 10c) holds
```

with:

```markdown
`cli evaluate` (Phase 10e) scores genomes against any opponents with
`sim::training::evaluate`. `sim::training` (Phase 10c) holds
```

- [ ] **Step 5: Run (expect success)**

Run: `cargo fmt --check`

Expected: PASS.

- [ ] **Step 6: Run (expect success)**

Run: `cargo clippy --workspace --all-targets -- -D warnings`

Expected: Clean.

- [ ] **Step 7: Run (expect success)**

Run: `cargo test --workspace`

Expected: PASS: every suite green (see the ledger for the count).

- [ ] **Step 8: Commit**

```bash
git add docs Cargo.lock
git commit -F - <<'EOF'
docs: Phase 10e done; learning options, cli evaluate and the v1 baseline as built

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F3sC3JdcgTFAox87bBuikA
EOF
```


---

## Self-Review (done while writing)

- **Spec coverage:** hall of fame and champion handling (section 6), baseline comparison protocol and success criteria (section 9: beats `lowest-legal`, `endgame-denial` and, as the stretch, `adaptive:reading,tempo,bully` far outside the seed spread), noisy-fitness and overfitting risks measured (section 12). Left out and written into the spec by Task 6: curriculum, growing matches, complexity pressure in ties.
- **Placeholder scan:** none; Tasks 1-3 are mechanical diffs, Tasks 4-5 carry their full files and scripts.
- **Type consistency:** `champion_candidates`, `hall_of_fame_size`, `hall_of_fame_interval`, `HallMember`, `training_rank`, `hall_of_fame`, `hall_score`, `EvaluateArgs`, `render_table` are used with the same names throughout.
- **Review Focus:** all five lines map to named tests or Task 5's check.
