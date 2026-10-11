//! Unit tests of the trainer's seed sets and ranking helpers.

use std::path::PathBuf;
use std::sync::Arc;

use super::super::config::test_support::sample;
use super::super::config::{DeckChoice, TrainConfig};
use super::super::evaluate::{evaluate_with, Opponents};
use super::super::events::RacingStats;
use super::super::fitness::{FitnessKind, Moments};
use super::seeds::*;
use super::stages::{expanding_race, PopulationEval};
use super::stats::top_indices;
use super::{strategy_for, Opponent, TrainObserver, Trainer};
use crate::{LowestLegal, RandomLegal};

struct Quiet;
impl TrainObserver for Quiet {}

fn dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "arschloch-trainer-unit-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn trainer(config: TrainConfig, name: &str) -> Trainer {
    let mut config = config;
    config.opponent_specs = vec!["lowest-legal".into(), "random-legal".into()];
    let opponents = vec![
        Opponent {
            name: "LowestLegal".into(),
            strategy: Arc::new(LowestLegal),
        },
        Opponent {
            name: "RandomLegal".into(),
            strategy: Arc::new(RandomLegal),
        },
    ];
    let path = dir(name);
    let trainer = Trainer::new(config, opponents, &path).unwrap();
    let _ = std::fs::remove_dir_all(&path);
    trainer
}

fn racing_config(kind: FitnessKind) -> TrainConfig {
    TrainConfig {
        fitness: kind,
        racing_top: 0.25,
        racing_matches: 4,
        ..sample()
    }
}

/// The extra evaluation of genome `index` that racing must have played.
fn extra_of(trainer: &Trainer, generation: u32, index: usize) -> crate::training::evaluate::Score {
    let config = trainer.config();
    evaluate_with(
        &strategy_for(&trainer.population.genomes()[index]),
        &config.tables(),
        Opponents::Mixed(&trainer.training_pool()),
        &racing_seeds(config, generation),
        config.fitness,
        config.matches_per_genome,
    )
}

/// Whether every un-raced genome's fitness is at most every raced genome's.
fn ordered(eval: &PopulationEval, raced: &[usize]) -> bool {
    let lowest = raced
        .iter()
        .map(|&i| eval.fitness[i])
        .fold(f64::INFINITY, f64::min);
    (0..eval.fitness.len())
        .filter(|i| !raced.contains(i))
        .all(|i| eval.fitness[i] <= lowest)
}

/// Rounds per injected evaluation.
const ROUNDS: u64 = 10;

/// An evaluation whose genome `i` has the mean `means[i]` over `ROUNDS`.
#[allow(clippy::cast_precision_loss)]
fn injected(means: &[f64]) -> PopulationEval {
    PopulationEval {
        fitness: means.to_vec(),
        moments: means
            .iter()
            .map(|&mean| Moments {
                rounds: ROUNDS,
                sum: mean * ROUNDS as f64,
                sum_sq: 0.0,
                sum_g: 0.0,
            })
            .collect(),
        skill: None,
    }
}

/// Runs the expanding race with the extra matches having the given means;
/// returns the stats, the batches asked for and the raced genomes.
#[allow(clippy::cast_precision_loss)]
fn run_injected(
    eval: &mut PopulationEval,
    initial: usize,
    cap: usize,
    extra_means: &[f64],
) -> (RacingStats, Vec<Vec<usize>>, Vec<usize>) {
    let before = eval.moments.clone();
    let mut batches = Vec::new();
    let stats = expanding_race(eval, initial, cap, 7, &FitnessKind::Mean, |indices| {
        batches.push(indices.to_vec());
        indices
            .iter()
            .map(|&i| Moments {
                rounds: ROUNDS,
                sum: extra_means[i] * ROUNDS as f64,
                sum_sq: 0.0,
                sum_g: 0.0,
            })
            .collect()
    });
    let raced = (0..eval.fitness.len())
        .filter(|&i| eval.moments[i] != before[i])
        .collect();
    (stats, batches, raced)
}

const MEANS: [f64; 8] = [0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.1, 0.0];

#[test]
fn the_race_expands_until_no_unraced_genome_outranks_a_raced_one() {
    let mut eval = injected(&MEANS);
    // Genomes 0 and 1 were lucky (pooled 0.5 each); 3 turns out poor.
    let extra = [0.1, 0.2, 0.7, 0.0, 0.55, 0.5, 0.0, 0.0];
    let (stats, batches, raced) = run_injected(&mut eval, 2, 8, &extra);
    // Round 1: lowest raced is 0.5, so 2, 3, 4 (above it) join; then 3
    // pools to 0.3 and genome 5 (0.5) outranks it; then nothing does.
    assert_eq!(batches, [vec![0, 1], vec![2, 3, 4], vec![5]]);
    assert_eq!(raced, [0, 1, 2, 3, 4, 5]);
    assert_eq!(
        (stats.raced, stats.expansion_rounds, stats.raced_final),
        (2, 2, 6)
    );
    assert!(!stats.cap_hit);
    assert!(ordered(&eval, &raced));
    // The first round's statistics are those of the top two alone.
    assert!((stats.mean_before - 0.85).abs() < 1e-12);
    assert!((stats.mean_after - 0.5).abs() < 1e-12);
    assert!((stats.mean_shift + 0.35).abs() < 1e-12);
    // Pooled values are the plain two-batch means; nobody is raced twice.
    assert!((eval.fitness[3] - 0.3).abs() < 1e-12);
    assert!(eval.moments.iter().all(|m| m.rounds <= 2 * ROUNDS));
    assert_eq!((eval.fitness[6], eval.fitness[7]), (0.1, 0.0));
}

#[test]
fn a_race_that_confirms_the_leaders_does_not_expand() {
    let mut eval = injected(&MEANS);
    let extra = [0.9, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let (stats, batches, raced) = run_injected(&mut eval, 2, 8, &extra);
    assert_eq!(batches, [vec![0, 1]]);
    assert_eq!(raced, [0, 1]);
    assert_eq!(
        (stats.expansion_rounds, stats.raced_final, stats.cap_hit),
        (0, 2, false)
    );
    assert!(ordered(&eval, &raced));
}

#[test]
fn the_cap_bounds_the_expansion_and_is_reported() {
    let mut eval = injected(&MEANS);
    let extra = [0.1, 0.2, 0.7, 0.0, 0.55, 0.5, 0.0, 0.0];
    let (stats, batches, raced) = run_injected(&mut eval, 2, 4, &extra);
    // Three genomes qualify but only two fit under the cap: the two with
    // the highest fitness (2 and 3) are raced, then 4 still outranks 3's
    // pooled 0.3 and the cap stops the race.
    assert_eq!(batches, [vec![0, 1], vec![2, 3]]);
    assert_eq!(raced, [0, 1, 2, 3]);
    assert_eq!(
        (stats.expansion_rounds, stats.raced_final, stats.cap_hit),
        (1, 4, true)
    );
    assert!(!ordered(&eval, &raced));
    // A cap below the initial set never shrinks it.
    let mut eval = injected(&MEANS);
    let (stats, _, _) = run_injected(&mut eval, 2, 1, &extra);
    assert_eq!(stats.raced_final, 2);
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn the_expanding_race_keeps_its_invariant_on_arbitrary_populations() {
    // Deterministic pseudo-random fitness and extras (an LCG).
    let mut lcg = 12345u64;
    let mut next = move || {
        lcg = lcg
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((lcg >> 33) as f64 / f64::from(1u32 << 31)) * 2.0 - 1.0
    };
    for case in 0..200 {
        let n = 5 + case % 20;
        let means: Vec<f64> = (0..n).map(|_| next()).collect();
        let extra: Vec<f64> = (0..n).map(|_| next()).collect();
        let initial = 1 + case % 4;
        let cap = (initial * 2).min(n);
        let mut eval = injected(&means);
        let (stats, batches, raced) = run_injected(&mut eval, initial, cap, &extra);
        assert!(raced.len() <= cap.max(initial), "case {case}");
        assert_eq!(raced.len(), stats.raced_final);
        assert_eq!(batches.len(), stats.expansion_rounds as usize + 1);
        assert!(batches.iter().all(|b| b.windows(2).all(|w| w[0] < w[1])));
        assert!(stats.cap_hit || ordered(&eval, &raced), "case {case}");
        assert!(eval.moments.iter().all(|m| m.rounds <= 2 * ROUNDS));
    }
}

#[test]
fn top_indices_orders_best_first_with_ties_going_to_the_lower_index() {
    let values = [0.5, 0.9, 0.5, -1.0, 0.9];
    assert_eq!(top_indices(&values, 3), vec![1, 4, 0]);
    assert_eq!(top_indices(&values, 1), vec![1]);
    assert_eq!(top_indices(&values, 99), vec![1, 4, 0, 2, 3]);
    assert_eq!(top_indices(&[], 3), [] as [usize; 0]);
}

#[test]
fn champion_selection_has_its_own_seed_stream() {
    // The candidates are ranked on `selection_seeds`; the champion's
    // reported score is then measured on `reeval_seeds`. If they shared
    // matches, the reported score would be inflated by the very
    // selection that chose the champion.
    let config = sample();
    let selection = selection_seeds(&config);
    assert_eq!(selection.len(), config.reeval_matches);
    assert_eq!(
        selection,
        selection_seeds(&config),
        "fixed, not per generation"
    );
    for seed in &selection {
        assert!(!reeval_seeds(&config).contains(seed));
        assert!(!heldout_seeds(&config).contains(seed));
        for generation in 0..200 {
            assert!(!training_seeds(&config, generation).contains(seed));
        }
    }
}

#[test]
fn champions_are_compared_on_one_fixed_seed_set_that_training_never_uses() {
    let config = sample();
    let fixed = reeval_seeds(&config);
    let held_out = heldout_seeds(&config);
    assert_eq!(fixed.len(), config.reeval_matches);
    assert_eq!(held_out.len(), config.reeval_matches * 2);
    // The same every generation (paired comparison between champions).
    assert_eq!(fixed, reeval_seeds(&config));
    // Disjoint from each other and from every generation's training matches.
    for seed in fixed.iter().chain(&held_out) {
        assert_eq!(
            fixed.iter().chain(&held_out).filter(|s| *s == seed).count(),
            1,
            "{seed}"
        );
        for generation in 0..200 {
            assert!(!training_seeds(&config, generation).contains(seed));
        }
    }
}

#[test]
fn racing_pools_the_mean_as_the_match_weighted_mean_and_leaves_the_rest_alone() {
    let trainer = trainer(racing_config(FitnessKind::Mean), "race-mean");
    let before = trainer.evaluate_population(0, &mut Quiet);
    let mut after = before.clone();
    let stats = trainer.race(0, &mut after).unwrap();
    let first = top_indices(&before.fitness, 3);
    assert_eq!((stats.raced, stats.extra_matches), (3, 4));
    // The race may have expanded past the top three (see the expansion
    // tests); every genome whose moments changed was raced.
    let raced: Vec<usize> = (0..before.fitness.len())
        .filter(|&i| after.moments[i] != before.moments[i])
        .collect();
    assert!(first.iter().all(|i| raced.contains(i)));
    assert_eq!(raced.len(), stats.raced_final);
    assert!(stats.raced_final <= trainer.config().racing_cap());
    let (m, k) = (6.0, 4.0);
    let mut shift = 0.0;
    for index in 0..before.fitness.len() {
        if raced.contains(&index) {
            let extra = extra_of(&trainer, 0, index);
            let expected = (m * before.fitness[index] + k * extra.mean) / (m + k);
            assert!(
                (after.fitness[index] - expected).abs() < 1e-12,
                "genome {index}: {} vs {expected}",
                after.fitness[index]
            );
            assert_eq!(after.moments[index].rounds, (6 + 4) * 4);
            if first.contains(&index) {
                shift += expected - before.fitness[index];
            }
        } else {
            assert_eq!(
                after.fitness[index].to_bits(),
                before.fitness[index].to_bits()
            );
            assert_eq!(after.moments[index], before.moments[index]);
        }
    }
    assert!((stats.mean_shift - shift / 3.0).abs() < 1e-12);
    assert!((stats.mean_after - stats.mean_before - stats.mean_shift).abs() < 1e-12);
    assert!(stats.cap_hit || ordered(&after, &raced));
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn racing_pools_the_moments_for_risk_not_the_scalars() {
    let lambda = 1.0;
    let trainer = trainer(racing_config(FitnessKind::Risk { lambda }), "race-risk");
    let before = trainer.evaluate_population(0, &mut Quiet);
    let mut after = before.clone();
    trainer.race(0, &mut after).unwrap();
    let mut differs = false;
    for index in top_indices(&before.fitness, 3) {
        let extra = extra_of(&trainer, 0, index);
        // By hand from the sums of both evaluations.
        let (old, new) = (&before.moments[index], &extra.moments);
        let n = (old.rounds + new.rounds) as f64;
        let (sum, sum_sq) = (old.sum + new.sum, old.sum_sq + new.sum_sq);
        let mean = sum / n;
        let expected = mean - lambda * (sum_sq / n - mean * mean).sqrt();
        assert!((after.fitness[index] - expected).abs() < 1e-12);
        let scalars = (6.0 * before.fitness[index] + 4.0 * extra.fitness) / 10.0;
        differs |= (after.fitness[index] - scalars).abs() > 1e-6;
    }
    assert!(differs, "pooling the scalars would give other numbers");
}

#[test]
fn racing_is_deterministic_independent_of_threads_and_off_by_default() {
    let trainer_on = trainer(racing_config(FitnessKind::Tanh { kappa: 3.0 }), "race-det");
    let before = trainer_on.evaluate_population(0, &mut Quiet);
    let run = || {
        let mut eval: PopulationEval = before.clone();
        let stats = trainer_on.race(0, &mut eval).unwrap();
        (eval.fitness, eval.moments, stats)
    };
    let expected = run();
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        assert_eq!(pool.install(run), expected, "{threads} threads");
    }
    let off = trainer(sample(), "race-off");
    let mut eval = off.evaluate_population(0, &mut Quiet);
    let copy = eval.clone();
    assert!(off.race(0, &mut eval).is_none());
    assert_eq!(eval.fitness, copy.fitness);
}

#[test]
fn racing_matches_are_fresh_every_generation_and_never_a_match_already_played() {
    let config = racing_config(FitnessKind::Mean);
    let first = racing_seeds(&config, 0);
    assert_eq!(first.len(), 4);
    assert_ne!(first, racing_seeds(&config, 1));
    for generation in 0..60 {
        for seed in &racing_seeds(&config, generation) {
            assert!(!training_seeds(&config, generation).contains(seed));
            assert!(!reeval_seeds(&config).contains(seed));
            assert!(!selection_seeds(&config).contains(seed));
            assert!(!heldout_seeds(&config).contains(seed));
            if generation > 0 {
                assert!(!first.contains(seed), "generation {generation}");
            }
        }
        // Training uses the even streams, racing the odd ones: no overlap
        // with any generation's training matches either.
        for other in 0..60 {
            for seed in &training_seeds(&config, other) {
                assert!(!racing_seeds(&config, generation).contains(seed));
            }
        }
    }
}

#[test]
fn every_genome_plays_the_same_settings_in_training() {
    let config = TrainConfig {
        player_counts: vec![3, 5],
        decks: vec![DeckChoice::Single, DeckChoice::Double],
        matches_per_genome: 9,
        ..sample()
    };
    let trainer = trainer(config, "multi-train");
    let eval = trainer.evaluate_population(0, &mut Quiet);
    assert_eq!(eval.fitness.len(), 12);
    // Every genome played the same number of rounds, with the same cycle:
    // the moments (counts, and the sums of squares of a fixed table mix) agree
    // in the number of rounds, and the settings alternate over four tables.
    assert!(eval.moments.iter().all(|m| m.rounds == 9 * 4));
    let labels: Vec<String> = trainer
        .config()
        .tables()
        .iter()
        .map(crate::training::TableSpec::label)
        .collect();
    assert_eq!(labels, ["3p-single", "3p-double", "5p-single", "5p-double"]);
    // And the evaluation is the same function of the seeds on any thread count.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let again = pool.install(|| trainer.evaluate_population(0, &mut Quiet));
    assert_eq!(again.fitness, eval.fitness);
}
