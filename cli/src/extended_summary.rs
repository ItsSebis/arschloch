//! The extended summary sections, printed after the original
//! summary (`summary::render_summary`, whose text is unchanged): average
//! rank, mean score and strength rating per strategy, the position bias per
//! seat and, when a skill score was asked for, the luck-adjusted scores.
//! With `--explain` each section is followed by the catalogue's one-line
//! meaning of its numbers.

use std::fmt::Write as _;

use sim::extended_stats::{wilson_interval, Estimate, ExtendedStatistics};
use sim::stats_catalog::render_explain;
use sim::useful_passes::UsefulPassStats;

use crate::args::{Args, SkillScoreArg};
use crate::batch::SkillOutput;

/// A left-aligned table with a two-space gap; widths come from the data.
fn table(header: &[&str], rows: &[Vec<String>]) -> String {
    let widths: Vec<usize> = (0..header.len())
        .map(|c| {
            rows.iter()
                .map(|r| r[c].chars().count())
                .chain([header[c].chars().count()])
                .max()
                .unwrap_or(0)
        })
        .collect();
    let mut out = String::new();
    for row in std::iter::once(header.iter().map(|h| (*h).to_owned()).collect())
        .chain(rows.iter().cloned())
    {
        let cells: Vec<String> = row
            .iter()
            .enumerate()
            .map(|(c, cell)| format!("{cell:<w$}", w = widths[c]))
            .collect();
        let _ = writeln!(out, "{}", cells.join("  ").trim_end());
    }
    out
}

/// An estimate as `value ±error`.
fn with_error(e: &Estimate, decimals: usize, signed: bool, show_error: bool) -> String {
    let (value, std_error) = (e.value, e.std_error);
    let value = if signed {
        format!("{value:+.decimals$}")
    } else {
        format!("{value:.decimals$}")
    };
    if show_error {
        format!("{value} ±{std_error:.decimals$}")
    } else {
        value
    }
}

/// Appends a blank line and, with `--explain`, the explanation block.
fn section(out: &mut String, args: &Args, body: &str, ids: &[&str]) {
    out.push('\n');
    out.push_str(body);
    if args.explain {
        out.push('\n');
        for line in render_explain(ids).lines() {
            let _ = writeln!(out, "  {line}");
        }
    }
}

fn strategy_table(statistics: &sim::Statistics, extended: &ExtendedStatistics) -> String {
    let rows: Vec<Vec<String>> = extended
        .by_strategy
        .iter()
        .map(|(name, s)| {
            let president_count = statistics
                .role_counts_by_strategy
                .get(name)
                .and_then(|by_role| by_role.get(&engine::Role::President))
                .copied()
                .unwrap_or(0);
            let trials: u32 = statistics
                .role_counts_by_strategy
                .get(name)
                .map_or(0, |by_role| by_role.values().sum());
            let president = match wilson_interval(u64::from(president_count), u64::from(trials)) {
                Some((low, high)) => format!(
                    "{:.1}% [{:.1}, {:.1}]",
                    100.0 * f64::from(president_count) / f64::from(trials),
                    100.0 * low,
                    100.0 * high
                ),
                None => "(not enough data)".to_owned(),
            };
            vec![
                name.clone(),
                with_error(&s.avg_rank, 2, false, true),
                with_error(&s.mean_role_score, 3, true, true),
                s.strength_rating.as_ref().map_or_else(
                    || "n/a".to_owned(),
                    |r| match r.std_error {
                        Some(e) => format!("{:+.0} ±{e:.0}", r.value),
                        None => format!("{:+.0}", r.value),
                    },
                ),
                president,
            ]
        })
        .collect();
    table(
        &[
            "Strategy",
            "avg rank ±SE",
            "mean score ±SE",
            "strength rating ±SE",
            "President % [95% Wilson]",
        ],
        &rows,
    )
}

fn seat_lines(extended: &ExtendedStatistics) -> String {
    let rows: Vec<Vec<String>> = extended
        .by_seat
        .iter()
        .enumerate()
        .map(|(seat, s)| {
            vec![
                format!("seat {seat}"),
                with_error(&s.avg_rank, 2, false, true),
                with_error(&s.mean_role_score, 3, true, true),
            ]
        })
        .collect();
    table(&["Seat", "avg rank ±SE", "mean score ±SE"], &rows)
}

#[allow(clippy::too_many_lines)] // one linear render of the skill table and its notes
fn skill_section(skill: &SkillOutput) -> String {
    let mut out = String::new();
    let mode = skill.mode;
    let rows: Vec<Vec<String>> = if let Some(dup) = &skill.duplicate {
        dup.strategies
            .iter()
            .map(|d| {
                let est = skill
                    .estimate
                    .as_ref()
                    .and_then(|e| e.strategies.iter().find(|e| e.name == d.name));
                vec![
                    d.name.clone(),
                    with_error(&d.plain_mean_role_score, 3, true, true),
                    with_error(&d.skill_score_duplicate, 3, true, true),
                    est.map_or_else(
                        || "-".to_owned(),
                        |e| with_error(&e.skill_score_estimate, 3, true, true),
                    ),
                    format!("{:.2}", d.variance_reduction),
                    format!("{:.0}%", 100.0 * d.luck_share),
                ]
            })
            .collect()
    } else if let Some(est) = &skill.estimate {
        est.strategies
            .iter()
            .map(|e| {
                vec![
                    e.name.clone(),
                    with_error(&e.plain_mean_role_score_round1, 3, true, true),
                    "-".to_owned(),
                    with_error(&e.skill_score_estimate, 3, true, true),
                    format!("{:.2}", e.variance_reduction),
                    format!("{:.0}%", 100.0 * e.luck_share),
                ]
            })
            .collect()
    } else {
        Vec::new()
    };
    let _ = writeln!(
        out,
        "Luck-adjusted score (--skill-score {}; score +1 best .. -1 worst, ± = standard error):",
        match mode {
            SkillScoreArg::Off => "off",
            SkillScoreArg::Estimate => "estimate",
            SkillScoreArg::Duplicate => "duplicate",
            SkillScoreArg::Both => "both",
        }
    );
    out.push_str(&table(
        &[
            "Strategy",
            "plain score ±SE",
            "skill (duplicate) ±SE",
            "skill (estimate) ±SE",
            "variance reduction M",
            "luck share",
        ],
        &rows,
    ));
    if let Some(dup) = &skill.duplicate {
        let _ = writeln!(
            out,
            "Duplicate: {} groups of {} matches, {} rounds per match; plain score, duplicate skill, M and luck share cover all rounds.",
            dup.group_count, dup.k, dup.rounds_per_match
        );
        out.push_str(
            "Duplicate deals cancel the luck of the deal in round 1; later rounds are luck-reduced, not luck-free.\n",
        );
        out.push_str(
            "In duplicate mode the plain score equals the duplicate skill value by construction (the mean of the group means is the mean of the matches); its ±SE is what an ordinary run of the same size would show, not an independent estimate.\n",
        );
    }
    if skill
        .estimate
        .as_ref()
        .is_some_and(|e| e.match_count > 0 && e.strategies.is_empty())
    {
        out.push_str("Estimate: n/a: a single strategy name.\n");
    }
    if skill.estimate.is_some() {
        let note = if skill.duplicate.is_some() {
            "The estimate covers round 1 only and comes from a second, independent batch of the same number of matches (--skill-score both plays about twice the matches); the comparison judges it against the duplicate round-1 skill."
        } else {
            "The estimate covers round 1 only (plain score and M too) and removes just the luck its hand features explain."
        };
        let _ = writeln!(out, "{note}");
    }
    if let Some(cmp) = skill
        .comparison
        .as_ref()
        .filter(|c| !c.strategies.is_empty())
    {
        let _ = writeln!(out, "Comparison (round 1): {}", cmp.verdict);
        for s in &cmp.strategies {
            let _ = writeln!(
                out,
                "  {}: duplicate {} vs estimate {} ({:+.1} SE apart)",
                s.name,
                with_error(&s.skill_score_duplicate_round1, 3, true, true),
                with_error(&s.skill_score_estimate, 3, true, true),
                s.difference_in_se
            );
        }
    }
    out
}

fn useful_pass_section(
    args: &Args,
    extended: &ExtendedStatistics,
    useful: &UsefulPassStats,
) -> String {
    let rows: Vec<Vec<String>> = extended
        .by_strategy
        .keys()
        .map(|name| {
            let mut row = vec![name.clone()];
            match useful.by_strategy.get(name) {
                Some(s) if s.voluntary_passes_total > 0 => {
                    row.push(s.voluntary_passes_total.to_string());
                    row.push(s.sampled_voluntary_passes.to_string());
                    match (&s.useful_pass_share, &s.useful_pass_gain) {
                        (Some(share), Some(gain)) => {
                            row.push(format!(
                                "{:.1}% [{:.1}, {:.1}]",
                                100.0 * share.value,
                                100.0 * share.interval.low,
                                100.0 * share.interval.high
                            ));
                            row.push(with_error(gain, 3, true, true));
                        }
                        _ => row.extend(["no sampled passes".to_owned(), "-".to_owned()]),
                    }
                }
                _ => row.extend([
                    "no voluntary passes".to_owned(),
                    "-".to_owned(),
                    "-".to_owned(),
                    "-".to_owned(),
                ]),
            }
            row
        })
        .collect();
    format!(
        "Useful voluntary passes ({} rollouts per candidate move, {:.0}% of the passes sampled, margin {}; replays the batch's matches as ordinary matches):\n{}A pass is useful if the rest of the round goes better after it than after the best of the weakest and strongest legal play; gain is in places.\n",
        args.useful_passes.unwrap_or(0),
        100.0 * args.useful_pass_sample,
        args.useful_pass_margin,
        table(
            &["Strategy", "voluntary passes", "sampled", "useful share [95% Wilson]", "mean gain ±SE"],
            &rows,
        )
    )
}

/// The extended sections, to be printed straight after the original
/// summary. Pure, so it can be unit-tested without capturing stdout.
#[must_use]
pub fn render_extended_summary(
    args: &Args,
    statistics: &sim::Statistics,
    extended: &ExtendedStatistics,
    skill: Option<&SkillOutput>,
    useful: Option<&UsefulPassStats>,
) -> String {
    let mut out = String::new();
    section(
        &mut out,
        args,
        &format!(
            "Average place (1 = best), mean score (+1 best .. -1 worst), strength rating (Elo-like points, field mean 0), share of rounds as President; ± = standard error over matches:\n{}",
            strategy_table(statistics, extended)
        ),
        &["avg_rank", "mean_role_score", "strength_rating", "retention_interval"],
    );
    section(
        &mut out,
        args,
        &format!(
            "By seat (position bias; a fair table gives every seat the same numbers):\n{}",
            seat_lines(extended)
        ),
        &["avg_rank", "mean_role_score"],
    );
    if let Some(skill) = skill {
        let mut ids = Vec::new();
        if skill.duplicate.is_some() {
            ids.push("skill_score_duplicate");
        }
        if skill.estimate.is_some() {
            ids.push("skill_score_estimate");
        }
        ids.extend(["variance_reduction", "luck_share"]);
        if skill.comparison.is_some() {
            ids.push("estimator_agreement");
        }
        section(&mut out, args, &skill_section(skill), &ids);
    }
    if let Some(useful) = useful {
        section(
            &mut out,
            args,
            &useful_pass_section(args, extended, useful),
            &["useful_pass_share", "useful_pass_gain"],
        );
    }
    if args.explain {
        out.push_str("\nAbout the first sections of this summary:\n");
        for line in render_explain(&[
            "role_counts_by_strategy",
            "voluntary_pass_rate",
            "voluntary_pass_rate_by_strategy",
            "role_retention_by_strategy",
            "first_round_placement_variance_by_strategy",
        ])
        .lines()
        {
            let _ = writeln!(out, "  {line}");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{DeckVariantArg, DuplicateRuleArg, FixedStrategy, StrategyArg};
    use std::path::PathBuf;
    use std::sync::Arc;

    fn args(skill_score: SkillScoreArg, explain: bool) -> Args {
        Args {
            player_count: 3,
            deck_variant: DeckVariantArg::Single,
            duplicate_rule: DuplicateRuleArg::FirstDealtWins,
            pass_rule: engine::PassRule::default(),
            exchange_rule: engine::ExchangeRule::default(),
            matches: 30,
            rounds: 3,
            strategies: vec![StrategyArg::Fixed(FixedStrategy::LowestLegal); 3],
            threads: 0,
            seed: 1,
            output: PathBuf::from("results.json"),
            skill_score,
            explain,
            bootstrap_resamples: 20,
            useful_passes: None,
            useful_pass_sample: 0.05,
            useful_pass_margin: 0.0,
        }
    }

    /// A small real run, rendered.
    fn render(mode: SkillScoreArg, explain: bool) -> String {
        render_with(mode, explain, false)
    }

    fn render_with(mode: SkillScoreArg, explain: bool, with_useful: bool) -> String {
        let mut args = args(mode, explain);
        if with_useful {
            args.useful_passes = Some(2);
            args.useful_pass_sample = 1.0;
        }
        let strategies: Vec<Arc<dyn sim::Strategy>> = vec![
            Arc::new(sim::LowestLegal),
            Arc::new(sim::GreedyHighest),
            Arc::new(sim::CardCounter),
        ];
        let configs: Vec<sim::MatchConfig> = (0..30)
            .map(|seed| sim::MatchConfig {
                player_count: 3,
                deck_variant: sim::DeckVariant::Single,
                duplicate_rule: sim::DuplicateRule::FirstDealtWins,
                rounds: 3,
                seed,
                pass_rule: sim::PassRule::default(),
                exchange_rule: sim::ExchangeRule::default(),
            })
            .collect();
        let run = crate::batch::run(mode, &configs, &strategies).unwrap();
        let useful = crate::batch::useful_passes(&args, &configs, &strategies);
        let statistics = sim::aggregate(&run.results);
        let options = sim::extended_stats::ExtendedOptions {
            bootstrap_resamples: args.bootstrap_resamples,
            ..Default::default()
        };
        let extended = sim::extended_stats::aggregate_extended_with(&run.results, &options);
        render_extended_summary(
            &args,
            &statistics,
            &extended,
            run.skill.as_ref(),
            useful.as_ref(),
        )
    }

    #[test]
    fn the_strategy_table_and_seat_lines_are_always_shown() {
        let text = render(SkillScoreArg::Off, false);
        assert!(text.starts_with('\n'));
        assert!(text.contains("avg rank ±SE") && text.contains("strength rating ±SE"));
        assert!(text.contains("President % [95% Wilson]"));
        for name in ["LowestLegal", "GreedyHighest", "CardCounter"] {
            assert!(text.contains(name), "{name}");
        }
        assert!(text.contains("seat 0") && text.contains("seat 2"));
        assert!(!text.contains("Luck-adjusted"));
        assert!(
            !text.contains("Average finishing place:"),
            "no --explain text"
        );
    }

    #[test]
    fn explain_adds_the_catalogue_lines_after_each_section() {
        let text = render(SkillScoreArg::Both, true);
        for line in [
            "Average finishing place: ",
            "Skill score (duplicate deals): ",
            "Skill score (cheap estimate): ",
            "Luck share: ",
            "Voluntary pass rate: ",
        ] {
            assert!(text.contains(line), "{line}");
        }
        let plain = render(SkillScoreArg::Both, false);
        assert!(!plain.contains("Luck share: "));
    }

    #[test]
    fn skill_table_and_honest_note_depend_on_the_mode() {
        let dup = render(SkillScoreArg::Duplicate, false);
        assert!(dup.contains("skill (duplicate) ±SE") && dup.contains("variance reduction M"));
        assert!(dup.contains("luck-reduced, not luck-free"));
        assert!(!dup.contains("Comparison"));
        let est = render(SkillScoreArg::Estimate, false);
        assert!(est.contains("round 1 only") && !est.contains("luck-reduced"));
        let both = render(SkillScoreArg::Both, false);
        assert!(both.contains("Comparison (round 1):") && both.contains("independent batch"));
    }

    #[test]
    fn the_useful_pass_section_appears_only_with_its_flag() {
        let off = render_with(SkillScoreArg::Off, false, false);
        assert!(!off.contains("Useful voluntary passes"));
        let on = render_with(SkillScoreArg::Off, true, true);
        assert!(on.contains("Useful voluntary passes (2 rollouts per candidate move, 100%"));
        assert!(on.contains("useful share [95% Wilson]") && on.contains("mean gain ±SE"));
        assert!(
            on.contains("no voluntary passes"),
            "LowestLegal never passes on purpose"
        );
        assert!(on.contains("Useful pass share: ") && on.contains("Useful pass gain: "));
    }

    #[test]
    fn table_pads_columns_to_the_widest_cell() {
        let text = table(
            &["A", "B"],
            &[
                vec!["long name".into(), "1".into()],
                vec!["x".into(), "22".into()],
            ],
        );
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "A          B");
        assert_eq!(lines[2], "x          22");
    }
}
