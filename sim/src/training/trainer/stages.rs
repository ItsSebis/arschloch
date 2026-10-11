//! The stages of one generation: evaluation, champion selection,
//! re-evaluation against the opponents, confirmation, decision sample and
//! the hall of fame. `Trainer::step` runs them in order.

use std::sync::Arc;
use std::time::Instant;

use neat::Genome;
use rayon::prelude::*;

use super::super::decisions::{record_decisions, DecisionFile};
use super::super::evaluate::{evaluate_with, evaluate_with_round1, Opponents, Round1Obs, Score};
use super::super::events::{OpponentStat, RacingStats, StageTimings};
use super::super::fitness::{FitnessKind, Moments};
use super::super::run_dir::{BestRecord, HallMember, TrainError};
use super::super::skill_term::combine;
use super::progress::{run_with_progress, timed};
use super::seeds::{heldout_seeds, racing_seeds, reeval_seeds, selection_seeds, training_seeds};
use super::stats::top_indices;
use super::{strategy_for, TrainObserver, Trainer, DECISIONS_PER_BEST};
use crate::{NeatStrategy, Strategy};

/// The fitness of every genome of a generation with what it was computed
/// from.
#[derive(Clone)]
pub(super) struct PopulationEval {
    pub fitness: Vec<f64>,
    /// Per-genome sums of the per-round role scores (for racing's pooling).
    pub moments: Vec<Moments>,
    pub skill: Option<Vec<f64>>,
}

impl Trainer {
    /// The fixed opponents only: what champions are compared against, so
    /// scores stay comparable across generations as the hall changes.
    pub(super) fn fixed_pool(&self) -> Vec<Arc<dyn Strategy>> {
        self.opponents.iter().map(|o| o.strategy.clone()).collect()
    }

    pub(super) fn hall_pool(&self) -> Vec<Arc<dyn Strategy>> {
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
    pub(super) fn training_pool(&self) -> Vec<Arc<dyn Strategy>> {
        let mut pool = self.fixed_pool();
        pool.extend(self.hall_pool());
        pool
    }

    /// Every genome's training fitness (the configured one) with the sums
    /// it came from and, with a non-zero `skill_weight`, its luck-adjusted
    /// term (the fitness is then the blend, see `skill_term`). With weight
    /// zero and the mean fitness this is the plain mean role score and
    /// nothing extra is recorded or computed.
    pub(super) fn evaluate_population(
        &self,
        generation: u32,
        observer: &mut dyn TrainObserver,
    ) -> PopulationEval {
        let tables = self.config.tables();
        let kind = self.config.fitness;
        let pool = self.training_pool();
        let seeds = training_seeds(&self.config, generation);
        let genomes = self.population.genomes();
        let with_skill = self.config.skill_weight > 0.0;
        // One parallel pass over all genomes (a rayon task per genome,
        // collected in genome order); the calling thread meanwhile reports
        // progress.
        let per_genome: Vec<(f64, Moments, Vec<Round1Obs>)> = run_with_progress(
            genomes,
            |genome| {
                let candidate = strategy_for(genome);
                if with_skill {
                    // Several settings and skill weights are exclusive
                    // (`TrainConfig::validate`), so there is one table.
                    let (score, obs) = evaluate_with_round1(
                        &candidate,
                        &tables[0],
                        Opponents::Mixed(&pool),
                        &seeds,
                    );
                    (score.mean, score.moments, obs)
                } else {
                    let score = evaluate_with(
                        &candidate,
                        &tables,
                        Opponents::Mixed(&pool),
                        &seeds,
                        kind,
                        0,
                    );
                    (score.fitness, score.moments, Vec::new())
                }
            },
            |done| observer.on_eval_progress(generation, done, genomes.len()),
        );
        let mut fitness = Vec::with_capacity(genomes.len());
        let mut moments = Vec::with_capacity(genomes.len());
        let mut round1: Vec<Vec<Round1Obs>> = Vec::new();
        for (value, sums, obs) in per_genome {
            fitness.push(value);
            moments.push(sums);
            if with_skill {
                round1.push(obs);
            }
        }
        if with_skill {
            let (total, skill) = combine(self.config.skill_weight, &fitness, &round1);
            PopulationEval {
                fitness: total,
                moments,
                skill: Some(skill),
            }
        } else {
            PopulationEval {
                fitness,
                moments,
                skill: None,
            }
        }
    }

    /// Racing: the top `ceil(racing_top * population)` genomes by fitness
    /// play `racing_matches` extra matches (the same fresh seeds, and so the
    /// same tables, seats and opponents, for every one of them) and their
    /// fitness becomes that of all their rounds pooled: the sums of the two
    /// evaluations are added (`Moments::pooled`) and the configured fitness
    /// is taken of the total. For the mean this is the match-weighted mean
    /// `(m * old + k * extra) / (m + k)`; for `risk` and `tanh` the moments
    /// are pooled, not the two scalars, because a standard deviation of the
    /// pooled rounds is not the average of two standard deviations.
    ///
    /// The race expands (see `expanding_race`) so that no un-raced genome
    /// outranks a raced one on its single noisy estimate. Returns `None`
    /// with racing off.
    pub(super) fn race(&self, generation: u32, eval: &mut PopulationEval) -> Option<RacingStats> {
        let initial = self.config.raced_count();
        if initial == 0 {
            return None;
        }
        let tables = self.config.tables();
        let kind = self.config.fitness;
        let pool = self.training_pool();
        let seeds = racing_seeds(&self.config, generation);
        let genomes = self.population.genomes();
        // The extra matches continue the setting and seat cycle after the
        // training matches, so a few extra matches still visit every setting.
        let first_index = self.config.matches_per_genome;
        Some(expanding_race(
            eval,
            initial,
            self.config.racing_cap(),
            self.config.racing_matches,
            &kind,
            |indices| {
                indices
                    .par_iter()
                    .map(|&index| {
                        evaluate_with(
                            &strategy_for(&genomes[index]),
                            &tables,
                            Opponents::Mixed(&pool),
                            &seeds,
                            kind,
                            first_index,
                        )
                        .moments
                    })
                    .collect()
            },
        ))
    }

    /// Picks the generation's champion: the best of the `k` genomes with
    /// the highest training fitness, ranked on the selection matches.
    /// Returns its index and its rank by training fitness (0 = the training
    /// best). With one candidate no matches are played.
    pub(super) fn select_champion(&self, fitness: &[f64]) -> (usize, usize) {
        let ranked = top_indices(fitness, self.config.champion_candidates);
        if ranked.len() == 1 {
            return (ranked[0], 0);
        }
        let tables = self.config.tables();
        let pool = self.fixed_pool();
        let seeds = selection_seeds(&self.config);
        let genomes = self.population.genomes();
        let scores: Vec<Score> = ranked
            .par_iter()
            .map(|&index| {
                evaluate_with(
                    &strategy_for(&genomes[index]),
                    &tables,
                    Opponents::Mixed(&pool),
                    &seeds,
                    self.config.fitness,
                    0,
                )
            })
            .collect();
        let winner = scores.iter().enumerate().fold(0, |best, (rank, score)| {
            if score.fitness > scores[best].fitness {
                rank
            } else {
                best
            }
        });
        (ranked[winner], winner)
    }

    /// The champion on the fixed matches: against the mixed fixed pool,
    /// against each fixed opponent alone and, once the hall has members,
    /// against the hall alone. These matches played no part in choosing it.
    pub(super) fn reevaluate(
        &self,
        champion: &Genome,
        timings: &mut StageTimings,
    ) -> (Score, Vec<OpponentStat>, Option<Score>) {
        let tables = self.config.tables();
        let kind = self.config.fitness;
        let seeds = reeval_seeds(&self.config);
        let candidate = strategy_for(champion);
        // The three evaluations only share their inputs, so they run
        // concurrently (each is itself parallel over matches); the timings
        // are each stage's own wall time and therefore overlap.
        let mixed_stage = || {
            evaluate_with(
                &candidate,
                &tables,
                Opponents::Mixed(&self.fixed_pool()),
                &seeds,
                kind,
                0,
            )
        };
        let per_opponent_stage = || -> Vec<OpponentStat> {
            self.opponents
                .par_iter()
                .map(|opponent| OpponentStat {
                    name: opponent.name.clone(),
                    score: evaluate_with(
                        &candidate,
                        &tables,
                        Opponents::Only(&opponent.strategy),
                        &seeds,
                        kind,
                        0,
                    )
                    .into(),
                })
                .collect()
        };
        let hall_stage = || {
            let hall = self.hall_pool();
            (!hall.is_empty()).then(|| {
                evaluate_with(
                    &candidate,
                    &tables,
                    Opponents::Mixed(&hall),
                    &seeds,
                    kind,
                    0,
                )
            })
        };
        let (mixed, (per_opponent, hall_score)) = rayon::join(
            || timed(mixed_stage),
            || rayon::join(|| timed(per_opponent_stage), || timed(hall_stage)),
        );
        timings.reevaluation_mixed = mixed.1;
        timings.reevaluation_per_opponent = per_opponent.1;
        timings.hall_of_fame = hall_score.1;
        (mixed.0, per_opponent.0, hall_score.0)
    }

    /// A few real decisions of the champion (for the dashboard's decision
    /// inspector), from one held-out match.
    pub(super) fn sample_decisions(&self, generation: u32, champion: &Genome) -> DecisionFile {
        let pool: Vec<Arc<dyn Strategy>> =
            self.opponents.iter().map(|o| o.strategy.clone()).collect();
        record_decisions(
            champion,
            &self.config.table(),
            &pool,
            heldout_seeds(&self.config)[0],
            DECISIONS_PER_BEST,
            generation,
        )
    }

    /// The champion's score on the held-out matches (see `heldout_seeds`).
    pub(super) fn confirm(&self, champion: &Genome) -> Score {
        let pool: Vec<Arc<dyn Strategy>> =
            self.opponents.iter().map(|o| o.strategy.clone()).collect();
        evaluate_with(
            &strategy_for(champion),
            &self.config.tables(),
            Opponents::Mixed(&pool),
            &heldout_seeds(&self.config),
            self.config.fitness,
            0,
        )
    }

    /// Writes a new best champion, confirms it on the held-out matches and
    /// records its decisions; adds the file-writing seconds to `files_secs`.
    pub(super) fn record_new_best(
        &mut self,
        generation: u32,
        champion: &Genome,
        reeval: &Score,
        timings: &mut StageTimings,
        files_secs: &mut f64,
    ) -> Result<Score, TrainError> {
        let stage = Instant::now();
        self.dir.write_best(champion)?;
        *files_secs += stage.elapsed().as_secs_f64();
        let stage = Instant::now();
        let heldout = self.confirm(champion);
        timings.confirmation = stage.elapsed().as_secs_f64();
        let stage = Instant::now();
        let decisions = self.sample_decisions(generation, champion);
        timings.decision_sample = stage.elapsed().as_secs_f64();
        let stage = Instant::now();
        self.dir.write_decisions(&decisions)?;
        *files_secs += stage.elapsed().as_secs_f64();
        self.best = Some(BestRecord {
            generation,
            reeval: reeval.clone().into(),
            heldout: heldout.clone().into(),
        });
        Ok(heldout)
    }

    /// The hall takes a fresh champion every `interval` generations.
    pub(super) fn update_hall(&mut self, generation: u32, champion: &Genome) {
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
    }
}

/// The expanding race on a population's fitness values. `extra(indices)`
/// returns the moments of the extra matches of those genomes (in the order
/// given; it is called with ascending genome indices, a genome is raced at
/// most once, and what it returns must not depend on which other genomes
/// are raced with it).
///
/// Round 0 races the top `initial` genomes. A genome that was not raced
/// keeps one noisy estimate, and the lucky ones are exactly those that sit
/// just below the cutoff: left alone, they could outrank raced genomes whose
/// fitness has been pulled back. So while some un-raced genome has a fitness
/// above the lowest pooled fitness of the raced set, the qualifying genomes
/// are raced too (one batch per round), until the cap on the raced set
/// (`cap`) is reached. Afterwards either every un-raced fitness is at most
/// every raced pooled fitness, or `cap_hit` says the cap stopped the
/// expansion. When more genomes qualify than the cap has room for, the
/// highest-fitness qualifiers are raced first.
pub(super) fn expanding_race(
    eval: &mut PopulationEval,
    initial: usize,
    cap: usize,
    extra_matches: usize,
    kind: &FitnessKind,
    mut extra: impl FnMut(&[usize]) -> Vec<Moments>,
) -> RacingStats {
    let n = eval.fitness.len();
    let cap = cap.clamp(initial.min(n), n);
    let mut raced_flag = vec![false; n];
    let mut raced = 0usize;
    let mut first_set = top_indices(&eval.fitness, initial);
    first_set.sort_unstable();
    let before: f64 = first_set.iter().map(|&i| eval.fitness[i]).sum();
    let mut pending = first_set.clone();
    let mut rounds = 0u32;
    let mut cap_hit = false;
    loop {
        let moments = extra(&pending);
        assert_eq!(moments.len(), pending.len());
        for (&index, extra) in pending.iter().zip(&moments) {
            let pooled = eval.moments[index].pooled(extra);
            eval.moments[index] = pooled;
            eval.fitness[index] = pooled.fitness(kind);
            raced_flag[index] = true;
        }
        raced += pending.len();
        let lowest = (0..n)
            .filter(|&i| raced_flag[i])
            .map(|i| eval.fitness[i])
            .fold(f64::INFINITY, f64::min);
        let qualifying: Vec<usize> = (0..n)
            .filter(|&i| !raced_flag[i] && eval.fitness[i] > lowest)
            .collect();
        if qualifying.is_empty() {
            break;
        }
        let room = cap - raced;
        if room == 0 {
            cap_hit = true;
            break;
        }
        let values: Vec<f64> = qualifying.iter().map(|&i| eval.fitness[i]).collect();
        pending = top_indices(&values, room)
            .into_iter()
            .map(|k| qualifying[k])
            .collect();
        pending.sort_unstable();
        rounds += 1;
    }
    let after: f64 = first_set.iter().map(|&i| eval.fitness[i]).sum();
    #[allow(clippy::cast_precision_loss)]
    let count = first_set.len() as f64;
    RacingStats {
        raced: first_set.len(),
        extra_matches,
        mean_before: before / count,
        mean_after: after / count,
        mean_shift: (after - before) / count,
        expansion_rounds: rounds,
        raced_final: raced,
        cap_hit,
    }
}
