//! A human-readable summary table for stdout: run configuration, one row
//! per strategy of role counts, and diversification/role-sustainment/
//! luck-vs-skill signals broken out per strategy.

use std::fmt::Write as _;

use crate::args::Args;

/// Renders `statistics` (and the run configuration in `args`) as a
/// human-readable report. Pure and allocation-only, so it's directly
/// unit-testable without capturing stdout.
#[must_use]
pub fn render_summary(args: &Args, statistics: &sim::Statistics) -> String {
    let mut out = String::new();

    let deck_variant = format!("{:?}", args.deck_variant);
    let duplicate_rule = format!("{:?}", args.duplicate_rule);
    let _ = writeln!(
        out,
        "{} players, {deck_variant} deck, {duplicate_rule}, {} matches x {} rounds, seed {}, {} threads",
        args.player_count, args.matches, args.rounds, args.seed, args.threads
    );
    let _ = writeln!(out);

    let roles = engine::roles_for_player_count(args.player_count)
        .expect("clap already validated player_count is 3-6");

    // The strategy-name column must fit the longest actual name (e.g. an
    // `Adaptive(...)` spec can far exceed a fixed strategy's name), so
    // its width is computed from the data rather than hardcoded.
    let name_width = statistics
        .role_counts_by_strategy
        .keys()
        .map(String::len)
        .max()
        .unwrap_or(0)
        .max("Strategy".len())
        + 2;

    let _ = write!(out, "{:<name_width$}", "Strategy");
    for role in roles {
        let role_label = format!("{role:?}");
        let _ = write!(out, "{role_label:<16}");
    }
    let _ = writeln!(out);

    for (strategy_name, counts_by_role) in &statistics.role_counts_by_strategy {
        let _ = write!(out, "{strategy_name:<name_width$}");
        for role in roles {
            let count = counts_by_role.get(role).copied().unwrap_or(0);
            let _ = write!(out, "{count:<16}");
        }
        let _ = writeln!(out);
    }
    let _ = writeln!(out);

    let pass_rate_percent = statistics.voluntary_pass_rate * 100.0;
    let _ = writeln!(out, "Voluntary pass rate: {pass_rate_percent:.2}%");

    let _ = writeln!(out);
    let _ = writeln!(out, "Voluntary pass rate by strategy:");
    for strategy_name in statistics.role_counts_by_strategy.keys() {
        match statistics
            .voluntary_pass_rate_by_strategy
            .get(strategy_name)
        {
            Some(rate) => {
                let _ = writeln!(out, "  {strategy_name:<name_width$}{:.2}%", rate * 100.0);
            }
            None => {
                let _ = writeln!(out, "  {strategy_name:<name_width$}(not enough data)");
            }
        }
    }

    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "President retention (stayed President the very next round):"
    );
    for strategy_name in statistics.role_counts_by_strategy.keys() {
        let retention = statistics
            .role_retention_by_strategy
            .get(strategy_name)
            .and_then(|by_role| by_role.get(&engine::Role::President));
        match retention {
            Some(r) if r.held > 0 => {
                let rate = f64::from(r.retained_next_round) / f64::from(r.held) * 100.0;
                let _ = writeln!(
                    out,
                    "  {strategy_name:<name_width$}{rate:.2}% ({}/{})",
                    r.retained_next_round, r.held
                );
            }
            _ => {
                let _ = writeln!(out, "  {strategy_name:<name_width$}(not enough data)");
            }
        }
    }

    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "First-round placement variance (lower = more skill-driven, higher = more luck-driven):"
    );
    for strategy_name in statistics.role_counts_by_strategy.keys() {
        let variance = statistics
            .first_round_placement_variance_by_strategy
            .get(strategy_name)
            .copied()
            .flatten();
        match variance {
            Some(v) => {
                let _ = writeln!(out, "  {strategy_name:<name_width$}{v:.2}");
            }
            None => {
                let _ = writeln!(out, "  {strategy_name:<name_width$}(not enough data)");
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{DeckVariantArg, DuplicateRuleArg, FixedStrategy, StrategyArg};
    use engine::Role;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn sample_args() -> Args {
        Args {
            player_count: 3,
            deck_variant: DeckVariantArg::Single,
            duplicate_rule: DuplicateRuleArg::FirstDealtWins,
            matches: 10,
            rounds: 2,
            strategies: vec![
                StrategyArg::Fixed(FixedStrategy::LowestLegal),
                StrategyArg::Fixed(FixedStrategy::GreedyHighest),
                StrategyArg::Fixed(FixedStrategy::RandomLegal),
            ],
            threads: 0,
            seed: 42,
            output: PathBuf::from("results.json"),
        }
    }

    fn sample_statistics() -> sim::Statistics {
        let mut lowest_legal_counts = BTreeMap::new();
        lowest_legal_counts.insert(Role::President, 4);
        lowest_legal_counts.insert(Role::Arschloch, 6);
        let mut role_counts = BTreeMap::new();
        role_counts.insert("LowestLegal".to_string(), lowest_legal_counts);

        sim::Statistics {
            matches_played: 10,
            role_counts_by_strategy: role_counts,
            voluntary_pass_rate: 0.125,
            voluntary_pass_rate_by_strategy: BTreeMap::from([("LowestLegal".to_string(), 0.1)]),
            role_retention_by_strategy: BTreeMap::from([(
                "LowestLegal".to_string(),
                BTreeMap::from([(
                    Role::President,
                    sim::RoleRetention {
                        held: 4,
                        retained_next_round: 3,
                    },
                )]),
            )]),
            first_round_placement_variance_by_strategy: BTreeMap::from([(
                "LowestLegal".to_string(),
                None,
            )]),
        }
    }

    #[test]
    fn render_summary_includes_strategy_and_role_names() {
        let summary = render_summary(&sample_args(), &sample_statistics());
        assert!(summary.contains("LowestLegal"));
        assert!(summary.contains("President"));
        assert!(summary.contains("Dorftrottel"));
        assert!(summary.contains("Arschloch"));
    }

    #[test]
    fn render_summary_includes_formatted_pass_rate() {
        let summary = render_summary(&sample_args(), &sample_statistics());
        assert!(summary.contains("Voluntary pass rate: 12.50%"));
    }

    #[test]
    fn render_summary_includes_per_strategy_pass_rate() {
        let summary = render_summary(&sample_args(), &sample_statistics());
        assert!(summary.contains("10.00%"));
    }

    #[test]
    fn render_summary_includes_president_retention() {
        let summary = render_summary(&sample_args(), &sample_statistics());
        assert!(summary.contains("75.00% (3/4)"));
    }

    #[test]
    fn render_summary_shows_not_enough_data_for_missing_variance() {
        let summary = render_summary(&sample_args(), &sample_statistics());
        assert!(summary.contains("(not enough data)"));
    }
}
