//! Command-line argument parsing: the CLI-facing surface (`clap`), plus
//! conversions into `engine`/`sim` types and cross-field validation that
//! `clap` can't express declaratively. See docs/ARCHITECTURE.md, "cli".

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;

/// One batch-run configuration, as parsed from the command line.
#[derive(Parser, Debug)]
#[command(about = "Simulate Arschloch matches and report role/statistics output")]
pub struct Args {
    /// Table size (3-6 seats).
    #[arg(long, value_parser = 3..=6)]
    pub player_count: u8,

    #[arg(long, value_enum, default_value_t = DeckVariantArg::Single)]
    pub deck_variant: DeckVariantArg,

    #[arg(long, value_enum, default_value_t = DuplicateRuleArg::FirstDealtWins)]
    pub duplicate_rule: DuplicateRuleArg,

    /// How many independent matches to simulate.
    #[arg(long, value_parser = 1..)]
    pub matches: usize,

    /// How many rounds each match plays (role carry-over between rounds).
    #[arg(long, default_value_t = 1, value_parser = 1..)]
    pub rounds: usize,

    /// One per seat, in seat order. Repeat the flag once per seat, e.g.
    /// `--strategy lowest-legal --strategy greedy-highest`.
    #[arg(long = "strategy", value_enum, required = true)]
    pub strategies: Vec<StrategyArg>,

    /// Rayon thread-pool size. 0 lets rayon pick its own default.
    #[arg(long, default_value_t = 0)]
    pub threads: usize,

    /// Base seed; match `i`'s seed is `seed.wrapping_add(i)`.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Where to write the JSON results file.
    #[arg(long, default_value = "results.json")]
    pub output: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum DeckVariantArg {
    Single,
    Double,
}

impl From<DeckVariantArg> for engine::DeckVariant {
    fn from(value: DeckVariantArg) -> Self {
        match value {
            DeckVariantArg::Single => engine::DeckVariant::Single,
            DeckVariantArg::Double => engine::DeckVariant::Double,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum DuplicateRuleArg {
    FirstDealtWins,
    LastDealtWins,
}

impl From<DuplicateRuleArg> for engine::DuplicateRule {
    fn from(value: DuplicateRuleArg) -> Self {
        match value {
            DuplicateRuleArg::FirstDealtWins => engine::DuplicateRule::FirstDealtWins,
            DuplicateRuleArg::LastDealtWins => engine::DuplicateRule::LastDealtWins,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum StrategyArg {
    LowestLegal,
    GreedyHighest,
    RandomLegal,
}

impl StrategyArg {
    /// Builds the concrete strategy instance this variant names.
    #[must_use]
    pub fn build(self) -> Arc<dyn sim::Strategy> {
        match self {
            StrategyArg::LowestLegal => Arc::new(sim::LowestLegal),
            StrategyArg::GreedyHighest => Arc::new(sim::GreedyHighest),
            StrategyArg::RandomLegal => Arc::new(sim::RandomLegal),
        }
    }
}

/// Cross-field checks `clap` can't express declaratively.
///
/// # Errors
///
/// Returns an error if `args.strategies.len()` doesn't equal
/// `args.player_count` — exactly one strategy is required per seat.
pub fn validate(args: &Args) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.strategies.len() == usize::from(args.player_count),
        "expected {} strategies (one per seat), got {}",
        args.player_count,
        args.strategies.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args_with_strategies(player_count: u8, strategy_count: usize) -> Args {
        Args {
            player_count,
            deck_variant: DeckVariantArg::Single,
            duplicate_rule: DuplicateRuleArg::FirstDealtWins,
            matches: 1,
            rounds: 1,
            strategies: vec![StrategyArg::LowestLegal; strategy_count],
            threads: 0,
            seed: 0,
            output: PathBuf::from("results.json"),
        }
    }

    #[test]
    fn validate_accepts_matching_strategy_count() {
        assert!(validate(&args_with_strategies(4, 4)).is_ok());
    }

    #[test]
    fn validate_rejects_strategy_count_mismatch() {
        assert!(validate(&args_with_strategies(4, 3)).is_err());
    }

    #[test]
    fn strategy_arg_builds_matching_strategy_names() {
        assert_eq!(StrategyArg::LowestLegal.build().name(), "LowestLegal");
        assert_eq!(StrategyArg::GreedyHighest.build().name(), "GreedyHighest");
        assert_eq!(StrategyArg::RandomLegal.build().name(), "RandomLegal");
    }

    #[test]
    fn deck_variant_arg_converts_to_engine_type() {
        assert_eq!(
            engine::DeckVariant::from(DeckVariantArg::Single),
            engine::DeckVariant::Single
        );
        assert_eq!(
            engine::DeckVariant::from(DeckVariantArg::Double),
            engine::DeckVariant::Double
        );
    }

    #[test]
    fn duplicate_rule_arg_converts_to_engine_type() {
        assert_eq!(
            engine::DuplicateRule::from(DuplicateRuleArg::FirstDealtWins),
            engine::DuplicateRule::FirstDealtWins
        );
        assert_eq!(
            engine::DuplicateRule::from(DuplicateRuleArg::LastDealtWins),
            engine::DuplicateRule::LastDealtWins
        );
    }
}
