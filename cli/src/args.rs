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
    #[arg(long, value_parser = clap::value_parser!(u8).range(3..=6))]
    pub player_count: u8,

    #[arg(long, value_enum, default_value_t = DeckVariantArg::Single)]
    pub deck_variant: DeckVariantArg,

    #[arg(long, value_enum, default_value_t = DuplicateRuleArg::FirstDealtWins)]
    pub duplicate_rule: DuplicateRuleArg,

    /// How many independent matches to simulate.
    #[arg(long, value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(1..))]
    pub matches: usize,

    /// How many rounds each match plays (role carry-over between rounds).
    #[arg(
        long,
        default_value_t = 1,
        value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(1..)
    )]
    pub rounds: usize,

    /// One per seat, setting the seating for the batch's first match,
    /// in seat order (e.g. `--strategy lowest-legal --strategy
    /// greedy-highest`). `sim::run_batch` then rotates this seating by
    /// one position for each subsequent match, so which physical seat
    /// each strategy occupies varies across the batch (this cancels
    /// `deal`'s documented uneven-remainder seat bias — see
    /// docs/RULES.md, "Players & Deck"). Must supply exactly
    /// `player_count`.
    ///
    /// Each value is a spec, `SPEC := FIXED | "adaptive" | "adaptive:"
    /// OPTIONS`: either one of the six fixed strategy names
    /// (`lowest-legal`, `greedy-highest`, `random-legal`,
    /// `hold-back-pairs`, `card-counter`, `endgame-denial`), or
    /// `adaptive` (defaults — see `sim::AdaptiveConfig::default`), or
    /// `adaptive:OPTIONS` where `OPTIONS` is a comma-separated modifier
    /// list parsed by `sim::AdaptiveConfig`'s `FromStr` (e.g.
    /// `counting`, `reading,deception=0.2`, `none`). For example, a
    /// three-seat table: `--strategy lowest-legal --strategy
    /// adaptive:counting --strategy adaptive:reading,deception=0.2`.
    #[arg(long = "strategy", required = true, value_name = "SPEC")]
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

/// The six fixed (non-configurable) strategies, named as `clap`
/// possible values (kebab-case).
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum FixedStrategy {
    LowestLegal,
    GreedyHighest,
    RandomLegal,
    HoldBackPairs,
    CardCounter,
    EndgameDenial,
}

/// One `--strategy` spec: either a fixed strategy name, or a
/// configurable `sim::Adaptive` spec (`adaptive` or
/// `adaptive:OPTIONS`). See `Args::strategies`'s doc comment for the
/// full grammar.
#[derive(Clone, Debug, PartialEq)]
pub enum StrategyArg {
    Fixed(FixedStrategy),
    Adaptive(sim::AdaptiveConfig),
}

impl std::str::FromStr for StrategyArg {
    type Err = String;

    fn from_str(spec: &str) -> Result<Self, String> {
        let spec = spec.trim();
        let (head, options) = match spec.split_once(':') {
            Some((h, o)) => (h.trim(), Some(o)),
            None => (spec, None),
        };
        if head == "adaptive" {
            return match options {
                None => Ok(Self::Adaptive(sim::AdaptiveConfig::default())),
                Some(o) => o.parse().map(Self::Adaptive).map_err(|e| e.to_string()),
            };
        }
        if options.is_some() {
            return Err(format!(
                "strategy `{head}` takes no options (only `adaptive:` does)"
            ));
        }
        <FixedStrategy as clap::ValueEnum>::from_str(head, false)
            .map(Self::Fixed)
            .map_err(|_| {
                format!(
                    "unknown strategy `{head}`; expected one of: {}, adaptive[:OPTIONS]",
                    fixed_strategy_names(),
                )
            })
    }
}

/// The `clap`-possible-value names of every `FixedStrategy` variant,
/// comma-joined, for use in `StrategyArg::from_str`'s error message.
fn fixed_strategy_names() -> String {
    use clap::ValueEnum;
    FixedStrategy::value_variants()
        .iter()
        .filter_map(clap::ValueEnum::to_possible_value)
        .map(|p| p.get_name().to_owned())
        .collect::<Vec<_>>()
        .join(", ")
}

impl StrategyArg {
    /// Builds the concrete strategy instance this spec names.
    #[must_use]
    pub fn build(self) -> Arc<dyn sim::Strategy> {
        match self {
            Self::Fixed(FixedStrategy::LowestLegal) => Arc::new(sim::LowestLegal),
            Self::Fixed(FixedStrategy::GreedyHighest) => Arc::new(sim::GreedyHighest),
            Self::Fixed(FixedStrategy::RandomLegal) => Arc::new(sim::RandomLegal),
            Self::Fixed(FixedStrategy::HoldBackPairs) => Arc::new(sim::HoldBackPairs),
            Self::Fixed(FixedStrategy::CardCounter) => Arc::new(sim::CardCounter),
            Self::Fixed(FixedStrategy::EndgameDenial) => Arc::new(sim::EndgameDenial),
            Self::Adaptive(config) => Arc::new(sim::Adaptive::new(config)),
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
            strategies: vec![StrategyArg::Fixed(FixedStrategy::LowestLegal); strategy_count],
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
        assert_eq!(
            StrategyArg::Fixed(FixedStrategy::LowestLegal)
                .build()
                .name(),
            "LowestLegal"
        );
        assert_eq!(
            StrategyArg::Fixed(FixedStrategy::GreedyHighest)
                .build()
                .name(),
            "GreedyHighest"
        );
        assert_eq!(
            StrategyArg::Fixed(FixedStrategy::RandomLegal)
                .build()
                .name(),
            "RandomLegal"
        );
        assert_eq!(
            StrategyArg::Fixed(FixedStrategy::HoldBackPairs)
                .build()
                .name(),
            "HoldBackPairs"
        );
        assert_eq!(
            StrategyArg::Fixed(FixedStrategy::CardCounter)
                .build()
                .name(),
            "CardCounter"
        );
        assert_eq!(
            StrategyArg::Fixed(FixedStrategy::EndgameDenial)
                .build()
                .name(),
            "EndgameDenial"
        );
    }

    #[test]
    fn parses_fixed_strategy_names_unchanged() {
        assert_eq!(
            "lowest-legal".parse::<StrategyArg>().unwrap(),
            StrategyArg::Fixed(FixedStrategy::LowestLegal)
        );
        assert_eq!(
            "card-counter".parse::<StrategyArg>().unwrap(),
            StrategyArg::Fixed(FixedStrategy::CardCounter)
        );
    }

    #[test]
    fn parses_bare_adaptive_to_defaults() {
        assert_eq!(
            "adaptive".parse::<StrategyArg>().unwrap(),
            StrategyArg::Adaptive(sim::AdaptiveConfig::default())
        );
    }

    #[test]
    fn parses_configured_adaptive() {
        let parsed = "adaptive:counting".parse::<StrategyArg>().unwrap();
        assert_eq!(
            parsed,
            StrategyArg::Adaptive(sim::AdaptiveConfig {
                counting: true,
                denial: sim::DenialMode::Off,
                deception_rate: 0.0,
                tempo: false,
                bully: false,
            })
        );
    }

    #[test]
    fn rejects_options_on_a_fixed_strategy() {
        assert!("lowest-legal:counting".parse::<StrategyArg>().is_err());
    }

    #[test]
    fn rejects_unknown_strategy_name() {
        assert!("nonexistent".parse::<StrategyArg>().is_err());
    }

    #[test]
    fn build_produces_an_adaptive_strategy_instance() {
        let arg: StrategyArg = "adaptive:reading,deception=0.2".parse().unwrap();
        let strategy = arg.build();
        assert!(strategy.name().starts_with("Adaptive("));
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

    // The tests above all construct `Args` via a struct literal, so none of
    // them exercise clap's actual parsing pipeline (attribute macros,
    // `value_parser`s, required-arg checks). The tests below drive
    // `Args::try_parse_from` with real argv to close that gap — in
    // particular, to catch a `value_parser` that is mistyped relative to its
    // field (e.g. an `i64`-typed range parser attached to a `u8`/`usize`
    // field), which previously caused a runtime panic on every parse rather
    // than a compile error.

    #[test]
    fn try_parse_from_accepts_player_count_at_range_boundaries() {
        for player_count in ["3", "6"] {
            let argv = [
                "arschloch",
                "--player-count",
                player_count,
                "--matches",
                "1",
                "--strategy",
                "lowest-legal",
            ];
            let parsed = Args::try_parse_from(argv);
            assert!(
                parsed.is_ok(),
                "player_count={player_count} should be accepted, got {parsed:?}"
            );
        }
    }

    #[test]
    fn try_parse_from_rejects_player_count_outside_range() {
        for player_count in ["2", "7"] {
            let argv = [
                "arschloch",
                "--player-count",
                player_count,
                "--matches",
                "1",
                "--strategy",
                "lowest-legal",
            ];
            assert!(
                Args::try_parse_from(argv).is_err(),
                "player_count={player_count} should be rejected"
            );
        }
    }

    #[test]
    fn try_parse_from_accepts_minimum_matches_and_rounds() {
        let argv = [
            "arschloch",
            "--player-count",
            "4",
            "--matches",
            "1",
            "--rounds",
            "1",
            "--strategy",
            "lowest-legal",
        ];
        let parsed = Args::try_parse_from(argv).expect("minimum matches/rounds should parse");
        assert_eq!(parsed.matches, 1);
        assert_eq!(parsed.rounds, 1);
    }

    #[test]
    fn try_parse_from_rejects_zero_matches_and_zero_rounds() {
        let zero_matches = [
            "arschloch",
            "--player-count",
            "4",
            "--matches",
            "0",
            "--strategy",
            "lowest-legal",
        ];
        assert!(Args::try_parse_from(zero_matches).is_err());

        let zero_rounds = [
            "arschloch",
            "--player-count",
            "4",
            "--matches",
            "1",
            "--rounds",
            "0",
            "--strategy",
            "lowest-legal",
        ];
        assert!(Args::try_parse_from(zero_rounds).is_err());
    }
}
