//! The measurement itself: rollouts of one voluntary pass, and the replay of a match
//! that samples its voluntary passes.

use std::sync::Arc;

use engine::{Card, Combo, DuplicateRule, Move, Round, SeatId};
use rand::SeedableRng;

use super::{MatchUsefulPasses, PassVerdict, UsefulPassOptions};
use crate::match_config::MatchConfig;
use crate::match_result::MatchResult;
use crate::match_runner::{play_out, run_match_observed, PlayCounters, RunOptions};
use crate::strategy::Strategy;
use crate::training::evaluate::mix;

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
pub fn replay_match(
    config: &MatchConfig,
    strategies: &[Arc<dyn Strategy>],
    run_options: &RunOptions,
    match_index: usize,
    options: &UsefulPassOptions,
) -> (MatchResult, MatchUsefulPasses) {
    let mut analysis = MatchUsefulPasses::new();
    let result = run_match_observed(config, strategies, run_options, |pass| {
        let tally = analysis
            .entry(strategies[usize::from(pass.seat)].name().to_string())
            .or_default();
        tally.voluntary_total += 1;
        let draw = unit_hash([
            options.seed,
            config.seed,
            match_index as u64,
            pass.round_index as u64,
            pass.decision as u64,
        ]);
        if draw < options.sample_fraction {
            let seed = decision_seed(
                options,
                config,
                match_index,
                pass.round_index,
                pass.decision,
            );
            let verdict = evaluate_pass(
                pass.round,
                pass.seat,
                strategies,
                config,
                pass.round_deck,
                seed,
                options,
            )
            .expect("a voluntary pass has a legal play");
            tally.useful += u64::from(verdict.useful);
            tally.gains.push(verdict.gain);
        }
    });
    (result, analysis)
}
