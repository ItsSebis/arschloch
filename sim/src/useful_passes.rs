//! Counterfactual measurement of whether voluntary passes are *useful*
//! (Phase 12, docs/STATISTICS.md ids `useful_pass_share` and
//! `useful_pass_gain`).
//!
//! # Definition
//!
//! A *voluntary* pass is a pass chosen when at least one `Move::Play` was
//! legal. For a sampled voluntary pass of seat `s` the round is cloned at
//! that exact state and finished `K` times for each of three candidate
//! moves: the pass itself, the *weakest* legal play (lowest top card) and
//! the *strongest* legal play (highest top card); these are the two
//! extremes of the engine's canonical candidate set. Every rollout applies
//! the move and finishes the round with [`play_out`] using the same
//! strategies the match uses. Rollout `k` of one decision uses the same
//! deterministic rng seed for all three candidates (derived from the
//! analysis seed, match, round, decision and `k`), so the candidates are
//! compared on common random numbers.
//!
//! `mean_place(m)` is the mean of `s`'s finishing place (1 = best .. n =
//! worst, its index in `Round::finishing_order` plus one) over the `K`
//! rollouts of candidate `m`. The best alternative is the better (lower)
//! of the weakest and strongest play. Then:
//!
//! * `gain = mean_place(best alternative) - mean_place(pass)` (positive:
//!   passing was better); the mean over all sampled voluntary passes,
//!   useful or not, is `useful_pass_gain`, so it can be negative;
//! * the pass is *useful* iff `mean_place(pass) < mean_place(best
//!   alternative) - margin` (`margin` defaults to 0: strictly better);
//!   the share of useful passes among the sampled ones is
//!   `useful_pass_share`.
//!
//! # Replay
//!
//! The analysis never alters the original matches. [`replay_match`]
//! re-simulates a match with a driver that mirrors `run_match_with` move
//! for move (same seeds, same rng stream: rollouts use their own
//! generators and never touch the match stream), so its [`MatchResult`]
//! equals the original and the voluntary passes occur at the same states.
//! A voluntary pass is sampled when a hash of (analysis seed, match seed,
//! match index, round, decision index) falls below `sample_fraction`; the
//! outcome therefore depends only on the inputs, not on thread count.

use std::collections::BTreeMap;
use std::sync::Arc;

use engine::{
    assign_roles, deal, exchange_with_rule, lowest_card_holder, Card, Combo, DuplicateRule, Move,
    Round, SeatId,
};
use rand::SeedableRng;
use rayon::prelude::*;

use crate::extended_stats::{standard_error, wilson_interval, Estimate, Interval};
use crate::hand_features::HandFeatures;
use crate::hand_reading::PassTracker;
use crate::match_config::MatchConfig;
use crate::match_result::MatchResult;
use crate::match_runner::{play_out, shuffled_deck, turn_context_for, PlayCounters, RunOptions};
use crate::strategy::Strategy;
use crate::training::evaluate::mix;

/// Settings of the useful-pass analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UsefulPassOptions {
    /// Rollouts `K` per candidate move (default 16).
    pub rollouts: usize,
    /// Fraction of voluntary passes analysed, `0..=1` (default 0.05).
    pub sample_fraction: f64,
    /// A pass is useful only if it beats the best alternative by more than
    /// this many places (default 0: strictly better).
    pub margin: f64,
    /// Seed of the sampling hash and the rollout generators.
    pub seed: u64,
}

impl Default for UsefulPassOptions {
    fn default() -> Self {
        Self {
            rollouts: 16,
            sample_fraction: 0.05,
            margin: 0.0,
            seed: 0,
        }
    }
}

/// The verdict on one voluntary pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PassVerdict {
    pub mean_place_pass: f64,
    pub mean_place_weakest: f64,
    pub mean_place_strongest: f64,
    /// `min(weakest, strongest)` mean place.
    pub mean_place_best_alternative: f64,
    /// `mean_place_best_alternative - mean_place_pass`.
    pub gain: f64,
    pub useful: bool,
}

/// A share with its binomial standard error and Wilson 95% interval.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct ShareEstimate {
    pub value: f64,
    pub std_error: f64,
    pub n: usize,
    pub interval: Interval,
}

/// The useful-pass numbers of one strategy name.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct StrategyUsefulPasses {
    /// All voluntary passes of the strategy in the analysed matches.
    pub voluntary_passes_total: u64,
    /// How many of them the sampling selected and rolled out.
    pub sampled_voluntary_passes: u64,
    /// How many sampled passes were useful.
    pub useful: u64,
    /// Useful share of the sampled passes; `None` without samples.
    pub useful_pass_share: Option<ShareEstimate>,
    /// Mean gain in places over the sampled passes (standard error over
    /// the sampled passes, treated as independent); `None` without samples.
    pub useful_pass_gain: Option<Estimate>,
}

/// Useful-pass statistics per strategy name.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct UsefulPassStats {
    pub by_strategy: BTreeMap<String, StrategyUsefulPasses>,
}

/// Raw per-strategy samples; merge in a fixed order for determinism.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UsefulPassTally {
    voluntary_total: u64,
    useful: u64,
    gains: Vec<f64>,
}

/// Per-match analysis output, keyed by strategy name.
pub type MatchUsefulPasses = BTreeMap<String, UsefulPassTally>;

fn place_of(round: &Round, seat: SeatId) -> f64 {
    let index = round
        .finishing_order()
        .iter()
        .position(|&s| s == seat)
        .expect("a completed round lists every seat");
    f64::from(u32::try_from(index).expect("tiny table") + 1)
}

fn top_order(a: &Combo, b: &Combo, rule: DuplicateRule) -> std::cmp::Ordering {
    a.top_card(rule)
        .compare(&b.top_card(rule), rule)
        .then_with(|| a.size().cmp(&b.size()))
}

/// The weakest and strongest legal plays among `legal_moves`, `None`
/// without any play.
fn extreme_plays(legal_moves: &[Move], rule: DuplicateRule) -> Option<(Move, Move)> {
    let plays: Vec<&Combo> = legal_moves
        .iter()
        .filter_map(|mv| match mv {
            Move::Play(combo) => Some(combo),
            Move::Pass => None,
        })
        .collect();
    let weakest = plays.iter().min_by(|a, b| top_order(a, b, rule))?;
    let strongest = plays.iter().max_by(|a, b| top_order(a, b, rule))?;
    Some((Move::Play(*(*weakest)), Move::Play(*(*strongest))))
}

/// What a rollout needs besides the round: strategies, config, deck.
struct Env<'a> {
    strategies: &'a [Arc<dyn Strategy>],
    config: &'a MatchConfig,
    round_deck: &'a [Card],
}

/// Mean place of `seat` over `rollouts` finishes of `round` after `mv`;
/// rollout `k` draws from the generator seeded `mix(mix(base_seed) ^ k)`.
#[allow(clippy::cast_precision_loss)]
fn mean_place_after(
    round: &Round,
    seat: SeatId,
    mv: &Move,
    env: &Env<'_>,
    base_seed: u64,
    rollouts: usize,
) -> f64 {
    let mut total = 0.0;
    for k in 0..rollouts {
        let mut rollout = round.clone();
        rollout
            .submit_move(seat, *mv)
            .expect("the move comes from the engine's legal moves");
        let mut rng = rand::rngs::StdRng::seed_from_u64(mix(mix(base_seed) ^ k as u64));
        let mut scratch = PlayCounters::new(env.config.player_count);
        play_out(
            &mut rollout,
            env.strategies,
            env.config,
            env.round_deck,
            &mut rng,
            &mut scratch,
        );
        total += place_of(&rollout, seat);
    }
    total / rollouts as f64
}

/// Judges a pass by `seat`, which must be the seat to move in `round`
/// (the state *before* the pass): runs the rollouts for the pass and for
/// the weakest and strongest legal play. `decision_seed` makes the
/// rollouts' generators (see the module docs). `None` if no play is legal
/// (a forced pass, not a voluntary one).
///
/// # Panics
///
/// Panics if `seat` is not to move, or `options.rollouts == 0`.
#[must_use]
pub fn evaluate_pass(
    round: &Round,
    seat: SeatId,
    strategies: &[Arc<dyn Strategy>],
    config: &MatchConfig,
    round_deck: &[Card],
    decision_seed: u64,
    options: &UsefulPassOptions,
) -> Option<PassVerdict> {
    assert_eq!(round.seat_to_move(), Some(seat), "seat must be to move");
    assert!(options.rollouts > 0, "at least one rollout is needed");
    let (weakest, strongest) = extreme_plays(&round.legal_moves(), config.duplicate_rule)?;
    let env = Env {
        strategies,
        config,
        round_deck,
    };
    let mean = |mv: &Move| mean_place_after(round, seat, mv, &env, decision_seed, options.rollouts);
    let pass = mean(&Move::Pass);
    let weak = mean(&weakest);
    let strong = if strongest == weakest {
        weak
    } else {
        mean(&strongest)
    };
    let best = weak.min(strong);
    Some(PassVerdict {
        mean_place_pass: pass,
        mean_place_weakest: weak,
        mean_place_strongest: strong,
        mean_place_best_alternative: best,
        gain: best - pass,
        useful: pass < best - options.margin,
    })
}

/// Uniform `[0, 1)` value from a hash of the arguments.
#[allow(clippy::cast_precision_loss)]
fn unit_hash(parts: [u64; 5]) -> f64 {
    let mut h = 0x5EED_u64;
    for part in parts {
        h = mix(h ^ part);
    }
    (h >> 11) as f64 / (1u64 << 53) as f64
}

fn decision_seed(
    options: &UsefulPassOptions,
    config: &MatchConfig,
    index: usize,
    r: usize,
    d: usize,
) -> u64 {
    mix(
        mix(mix(mix(options.seed ^ 0xA11C_E5ED) ^ config.seed) ^ index as u64)
            ^ ((r as u64) << 32 | d as u64),
    )
}

/// Re-simulates one match exactly as `run_match_with` would (same result,
/// same move sequence) and analyses the sampled voluntary passes.
/// `match_index` identifies the match in the sampling hash (its position
/// in the batch). `strategies` are the seat strategies actually used.
///
/// # Panics
///
/// As `run_match_with`, and if `options.rollouts == 0` while
/// `sample_fraction > 0`.
#[must_use]
#[allow(clippy::too_many_lines)] // mirrors `run_match_with` line for line
pub fn replay_match(
    config: &MatchConfig,
    strategies: &[Arc<dyn Strategy>],
    run_options: &RunOptions,
    match_index: usize,
    options: &UsefulPassOptions,
) -> (MatchResult, MatchUsefulPasses) {
    assert_eq!(
        strategies.len(),
        usize::from(config.player_count),
        "one strategy is required per seat"
    );
    assert!(config.rounds > 0, "a match needs at least one round");

    let mut analysis = MatchUsefulPasses::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    let mut previous_roles: Option<Vec<engine::Role>> = None;
    let mut previous_arschloch: Option<SeatId> = None;
    let mut role_history = Vec::with_capacity(config.rounds);
    let mut counters = PlayCounters::new(config.player_count);
    let mut first_hand_features = None;

    for round_index in 0..config.rounds {
        let deck = shuffled_deck(config, run_options, round_index, &mut rng);
        let mut hands = deal(deck, config.player_count)
            .expect("standard_deck always yields enough cards for a supported player count");
        if round_index == 0 && run_options.record_deal_features {
            first_hand_features = Some(hands.iter().map(|h| HandFeatures::from_hand(h)).collect());
        }
        let leader = match (&previous_roles, previous_arschloch) {
            (Some(roles), Some(arschloch)) => {
                exchange_with_rule(
                    &mut hands,
                    roles,
                    config.duplicate_rule,
                    config.exchange_rule,
                    |seat, hand, count, duplicate_rule| {
                        strategies[seat].choose_exchange_cards(
                            hand,
                            count,
                            duplicate_rule,
                            &mut rng,
                        )
                    },
                )
                .expect("exchange inputs are valid, as in run_match_with");
                arschloch
            }
            _ => lowest_card_holder(&hands, config.duplicate_rule)
                .expect("a freshly dealt hand set is never empty"),
        };
        let round_deck: Vec<Card> = hands.iter().flatten().copied().collect();
        let mut round =
            Round::with_pass_rule(hands, config.duplicate_rule, config.pass_rule, leader)
                .expect("player_count/leader are always valid for a supported table size");

        // `play_out`'s loop, plus the analysis hook on voluntary passes.
        let mut tracker = PassTracker::new(usize::from(config.player_count), config.duplicate_rule);
        let mut decision = 0usize;
        let mut legal_moves = Vec::new();
        while !round.is_complete() {
            let seat = round.seat_to_move().expect("round is not complete");
            if round.current_combo().is_none() {
                counters.trick_count += 1;
            }
            round.legal_moves_into(&mut legal_moves);
            let context =
                turn_context_for(&round, seat, config.player_count, &round_deck, &mut tracker);
            let chosen = strategies[usize::from(seat)].choose_play(
                &legal_moves,
                config.duplicate_rule,
                &context,
                &mut rng,
            );
            drop(context);

            if chosen == Move::Pass {
                counters.pass_counts[usize::from(seat)] += 1;
                if legal_moves.iter().any(|mv| matches!(mv, Move::Play(_))) {
                    counters.voluntary_pass_counts[usize::from(seat)] += 1;
                    let tally = analysis
                        .entry(strategies[usize::from(seat)].name().to_string())
                        .or_default();
                    tally.voluntary_total += 1;
                    let draw = unit_hash([
                        options.seed,
                        config.seed,
                        match_index as u64,
                        round_index as u64,
                        decision as u64,
                    ]);
                    if draw < options.sample_fraction {
                        let seed =
                            decision_seed(options, config, match_index, round_index, decision);
                        let verdict = evaluate_pass(
                            &round,
                            seat,
                            strategies,
                            config,
                            &round_deck,
                            seed,
                            options,
                        )
                        .expect("a voluntary pass has a legal play");
                        tally.useful += u64::from(verdict.useful);
                        tally.gains.push(verdict.gain);
                    }
                }
            }
            round
                .submit_move(seat, chosen)
                .expect("strategies only choose from the moves engine just reported as legal");
            decision += 1;
        }

        let finishing_order = round.finishing_order().to_vec();
        let roles = assign_roles(&finishing_order, config.player_count)
            .expect("finishing_order is always a valid permutation for a supported player count");
        previous_arschloch = finishing_order.last().copied();
        role_history.push(roles.clone());
        previous_roles = Some(roles);
    }

    let result = MatchResult {
        player_count: config.player_count,
        strategy_names: strategies.iter().map(|s| s.name().to_string()).collect(),
        role_history,
        trick_count: counters.trick_count,
        pass_counts: counters.pass_counts,
        voluntary_pass_counts: counters.voluntary_pass_counts,
        first_hand_features,
    };
    (result, analysis)
}

/// Folds per-match analyses (in the given order) into statistics.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn summarise(matches: &[MatchUsefulPasses]) -> UsefulPassStats {
    let mut merged: BTreeMap<String, UsefulPassTally> = BTreeMap::new();
    for one in matches {
        for (name, tally) in one {
            let into = merged.entry(name.clone()).or_default();
            into.voluntary_total += tally.voluntary_total;
            into.useful += tally.useful;
            into.gains.extend_from_slice(&tally.gains);
        }
    }
    let by_strategy = merged
        .into_iter()
        .map(|(name, tally)| {
            let sampled = tally.gains.len() as u64;
            let share = wilson_interval(tally.useful, sampled).map(|(low, high)| {
                let n = sampled as f64;
                let p = tally.useful as f64 / n;
                ShareEstimate {
                    value: p,
                    std_error: (p * (1.0 - p) / n).sqrt(),
                    n: tally.gains.len(),
                    interval: Interval { low, high },
                }
            });
            let gain = (!tally.gains.is_empty()).then(|| Estimate {
                value: tally.gains.iter().sum::<f64>() / tally.gains.len() as f64,
                std_error: standard_error(&tally.gains),
                n: tally.gains.len(),
            });
            (
                name,
                StrategyUsefulPasses {
                    voluntary_passes_total: tally.voluntary_total,
                    sampled_voluntary_passes: sampled,
                    useful: tally.useful,
                    useful_pass_share: share,
                    useful_pass_gain: gain,
                },
            )
        })
        .collect();
    UsefulPassStats { by_strategy }
}

/// Replays every config (seats rotated by the config's position, exactly
/// as `run_batch` does, with `strategies` the unrotated list) and analyses
/// the sampled voluntary passes. Parallel over matches; the result does
/// not depend on the thread count. The original matches are not altered.
///
/// # Panics
///
/// As `replay_match`.
#[must_use]
pub fn analyse_useful_passes(
    configs: &[MatchConfig],
    strategies: &[Arc<dyn Strategy>],
    options: &UsefulPassOptions,
) -> UsefulPassStats {
    let per_match: Vec<MatchUsefulPasses> = configs
        .par_iter()
        .enumerate()
        .map(|(index, config)| {
            let rotation = index % strategies.len();
            let rotated: Vec<Arc<dyn Strategy>> = strategies
                .iter()
                .cycle()
                .skip(rotation)
                .take(strategies.len())
                .cloned()
                .collect();
            replay_match(config, &rotated, &RunOptions::default(), index, options).1
        })
        .collect();
    summarise(&per_match)
}
