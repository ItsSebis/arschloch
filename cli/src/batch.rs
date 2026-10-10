//! Running a batch of matches for the main command, including the
//! optional luck/skill measurements (`--skill-score`). With `off` this is
//! exactly `sim::run_batch`: no extra simulation, no extra output.

use std::sync::Arc;

use anyhow::Context;

use crate::args::SkillScoreArg;

/// The luck/skill reports of a run; the `extended.skill` JSON object.
#[derive(Debug, serde::Serialize)]
pub struct SkillOutput {
    pub mode: SkillScoreArg,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duplicate: Option<sim::skill::SkillReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimate: Option<sim::skill::EstimatorReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comparison: Option<sim::skill::ComparisonReport>,
}

/// The matches of a run plus the skill reports, when asked for.
pub struct BatchRun {
    pub results: Vec<sim::MatchResult>,
    pub skill: Option<SkillOutput>,
}

/// Added to every match seed of the estimator's batch in `both` mode, so
/// its deals are different from (and independent of) the duplicate groups'.
pub const ESTIMATOR_SEED_OFFSET: u64 = 0x5EED_E571_4A70_0001;

fn estimator_batch(
    configs: &[sim::MatchConfig],
    strategies: &[Arc<dyn sim::Strategy>],
) -> Vec<sim::MatchResult> {
    sim::run_batch_with(
        configs,
        strategies,
        &sim::RunOptions {
            deal_seed: None,
            record_deal_features: true,
        },
    )
}

/// The useful-pass analysis of `--useful-passes`, or `None` when it is off.
/// It replays the matches of `configs` as ordinary matches (also in the
/// duplicate modes, whose deals it does not reuse), seeded with `--seed`.
#[must_use]
pub fn useful_passes(
    args: &crate::args::Args,
    configs: &[sim::MatchConfig],
    strategies: &[Arc<dyn sim::Strategy>],
) -> Option<sim::useful_passes::UsefulPassStats> {
    let rollouts = args.useful_passes?;
    Some(sim::useful_passes::analyse_useful_passes(
        configs,
        strategies,
        &sim::useful_passes::UsefulPassOptions {
            rollouts,
            sample_fraction: args.useful_pass_sample,
            margin: args.useful_pass_margin,
            seed: args.seed,
        },
    ))
}

/// Plays `configs` with `strategies` as `mode` asks.
///
/// - `Off`: ordinary matches.
/// - `Estimate`: ordinary matches that also record the round-1 hand
///   features; the estimator adjusts for them.
/// - `Duplicate`: groups of matches with identical deals and rotated
///   seats. The groups are the matches of the run.
/// - `Both`: the duplicate groups (the matches of the run, the ground
///   truth) plus a second, independent ordinary batch of the same size
///   with hand features, whose seeds are `ESTIMATOR_SEED_OFFSET` further
///   on. The estimator runs on that batch and is compared with the
///   duplicate report. This costs about twice the matches.
///
/// # Errors
///
/// A duplicate mode fails with the reason (see `sim::duplicate::DuplicateError`)
/// when the batch does not form whole groups.
pub fn run(
    mode: SkillScoreArg,
    configs: &[sim::MatchConfig],
    strategies: &[Arc<dyn sim::Strategy>],
) -> anyhow::Result<BatchRun> {
    match mode {
        SkillScoreArg::Off => Ok(BatchRun {
            results: sim::run_batch(configs, strategies),
            skill: None,
        }),
        SkillScoreArg::Estimate => {
            let results = estimator_batch(configs, strategies);
            let estimate = sim::skill::estimator_report(&results);
            Ok(BatchRun {
                results,
                skill: Some(SkillOutput {
                    mode,
                    duplicate: None,
                    estimate: Some(estimate),
                    comparison: None,
                }),
            })
        }
        SkillScoreArg::Duplicate | SkillScoreArg::Both => {
            let groups = sim::duplicate::try_run_duplicate_batch(configs, strategies, false)
                .context("cannot run the duplicate deals")?;
            let duplicate = sim::skill::skill_report(&groups);
            let results: Vec<sim::MatchResult> =
                groups.into_iter().flat_map(|g| g.matches).collect();
            let (estimate, comparison) = if mode == SkillScoreArg::Both {
                let shifted: Vec<sim::MatchConfig> = configs
                    .iter()
                    .map(|c| sim::MatchConfig {
                        seed: c.seed.wrapping_add(ESTIMATOR_SEED_OFFSET),
                        ..*c
                    })
                    .collect();
                let estimate = sim::skill::estimator_report(&estimator_batch(&shifted, strategies));
                let comparison = sim::skill::compare(&duplicate, &estimate);
                (Some(estimate), Some(comparison))
            } else {
                (None, None)
            };
            Ok(BatchRun {
                results,
                skill: Some(SkillOutput {
                    mode,
                    duplicate: Some(duplicate),
                    estimate,
                    comparison,
                }),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup(matches: usize) -> (Vec<sim::MatchConfig>, Vec<Arc<dyn sim::Strategy>>) {
        let strategies: Vec<Arc<dyn sim::Strategy>> = vec![
            Arc::new(sim::LowestLegal),
            Arc::new(sim::GreedyHighest),
            Arc::new(sim::CardCounter),
        ];
        let configs = (0..matches as u64)
            .map(|seed| sim::MatchConfig {
                player_count: 3,
                deck_variant: sim::DeckVariant::Single,
                duplicate_rule: sim::DuplicateRule::FirstDealtWins,
                rounds: 2,
                seed,
                pass_rule: sim::PassRule::default(),
                exchange_rule: sim::ExchangeRule::default(),
            })
            .collect();
        (configs, strategies)
    }

    #[test]
    fn off_is_the_plain_batch_with_no_skill_output() {
        let (configs, strategies) = setup(6);
        let run = run(SkillScoreArg::Off, &configs, &strategies).unwrap();
        let plain = sim::run_batch(&configs, &strategies);
        assert_eq!(format!("{:?}", run.results), format!("{plain:?}"));
        assert!(run.skill.is_none());
    }

    #[test]
    fn each_mode_fills_exactly_its_reports() {
        let (configs, strategies) = setup(6);
        let cases = [
            (SkillScoreArg::Estimate, false, true, false),
            (SkillScoreArg::Duplicate, true, false, false),
            (SkillScoreArg::Both, true, true, true),
        ];
        for (mode, dup, est, cmp) in cases {
            let run = run(mode, &configs, &strategies).unwrap();
            let skill = run.skill.unwrap();
            assert_eq!(run.results.len(), 6);
            assert_eq!(
                (
                    skill.duplicate.is_some(),
                    skill.estimate.is_some(),
                    skill.comparison.is_some()
                ),
                (dup, est, cmp),
                "{mode:?}"
            );
        }
    }

    #[test]
    fn both_compares_with_an_independent_batch_so_agreement_is_not_trivial() {
        let strategies: Vec<Arc<dyn sim::Strategy>> = vec![
            Arc::new(sim::LowestLegal),
            Arc::new(sim::GreedyHighest),
            Arc::new(sim::RandomLegal),
            Arc::new(sim::HoldBackPairs),
        ];
        let configs: Vec<sim::MatchConfig> = (0..600u64)
            .map(|seed| sim::MatchConfig {
                player_count: 4,
                deck_variant: sim::DeckVariant::Single,
                duplicate_rule: sim::DuplicateRule::FirstDealtWins,
                rounds: 1,
                seed,
                pass_rule: sim::PassRule::default(),
                exchange_rule: sim::ExchangeRule::default(),
            })
            .collect();
        let run = run(SkillScoreArg::Both, &configs, &strategies).unwrap();
        let skill = run.skill.unwrap();
        let comparison = skill.comparison.unwrap();
        assert_eq!(skill.estimate.unwrap().match_count, 600);
        assert_eq!(comparison.strategies.len(), 4);
        // Different deals: the two estimates differ, but only by noise.
        assert!(
            comparison
                .strategies
                .iter()
                .any(|s| s.difference_in_se.abs() > 1e-9),
            "an independent batch cannot reproduce the duplicate numbers exactly"
        );
        for s in &comparison.strategies {
            assert!(s.difference_in_se.abs() < 4.5, "{}: {:?}", s.name, s);
        }
        assert!((comparison.mean_variance_removed_ratio - 1.0).abs() > 1e-9);
    }

    #[test]
    fn a_ragged_duplicate_batch_is_a_clear_error() {
        let (configs, strategies) = setup(7);
        let error = run(SkillScoreArg::Duplicate, &configs, &strategies)
            .err()
            .expect("7 matches do not form groups of 3");
        let text = format!("{error:#}");
        assert!(text.contains("multiple of 3"), "{text}");
    }
}
