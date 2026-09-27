//! A human-readable summary table for stdout: run configuration, one row
//! per strategy of role counts, and the pooled voluntary-pass rate.

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

    let _ = write!(out, "{:<16}", "Strategy");
    for role in roles {
        let role_label = format!("{role:?}");
        let _ = write!(out, "{role_label:<16}");
    }
    let _ = writeln!(out);

    for (strategy_name, counts_by_role) in &statistics.role_counts_by_strategy {
        let _ = write!(out, "{strategy_name:<16}");
        for role in roles {
            let count = counts_by_role.get(role).copied().unwrap_or(0);
            let _ = write!(out, "{count:<16}");
        }
        let _ = writeln!(out);
    }
    let _ = writeln!(out);

    let pass_rate_percent = statistics.voluntary_pass_rate * 100.0;
    let _ = writeln!(out, "Voluntary pass rate: {pass_rate_percent:.2}%");

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{DeckVariantArg, DuplicateRuleArg, StrategyArg};
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
                StrategyArg::LowestLegal,
                StrategyArg::GreedyHighest,
                StrategyArg::RandomLegal,
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
}
