//! Command-line argument parsing: the CLI-facing surface (`clap`), plus
//! conversions into `engine`/`sim` types and cross-field validation that
//! `clap` can't express declaratively. See docs/ARCHITECTURE.md, "cli".

use std::path::{Path, PathBuf};
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

    /// What a pass means for the rest of the trick: `final` (the rules of the
    /// game) takes a seat out of the trick once it passes; `free` is how the
    /// simulator played before the rule was fixed, kept to reproduce older
    /// results.
    #[arg(long, value_parser = clap::value_parser!(engine::PassRule), default_value_t = engine::PassRule::Final)]
    pub pass_rule: engine::PassRule,

    /// Whether the lower role of an exchange pair must give its highest cards
    /// (`forced`, the rules of the game) or may choose which cards to give
    /// (`free`, how the simulator behaved before Phase 15; also lets strategies
    /// that keep pairs together use their own choice).
    #[arg(long, value_parser = clap::value_parser!(engine::ExchangeRule), default_value_t = engine::ExchangeRule::Forced)]
    pub exchange_rule: engine::ExchangeRule,

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
    /// OPTIONS | "neat:" PATH`: either one of the six fixed strategy names
    /// (`lowest-legal`, `greedy-highest`, `random-legal`,
    /// `hold-back-pairs`, `card-counter`, `endgame-denial`), or
    /// `adaptive` (defaults — see `sim::AdaptiveConfig::default`), or
    /// `adaptive:OPTIONS` where `OPTIONS` is a comma-separated modifier
    /// list parsed by `sim::AdaptiveConfig`'s `FromStr` (e.g.
    /// `counting`, `reading,deception=0.2`, `none`), or
    /// `neat:PATH`, a trained genome file (see `sim::GenomeFile`),
    /// reported in results as `Neat(<file name without extension>)`. For
    /// example, a three-seat table: `--strategy lowest-legal --strategy
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

    /// Separate skill from the luck of the deal. `estimate` adjusts an
    /// ordinary run for the quality of the dealt round-1 hands (nearly
    /// free); `duplicate` replays every deal with the strategies rotated
    /// through all seats (exact, needs `--matches` to be a multiple of
    /// `--player-count`, and those groups are the matches of the run);
    /// `both` does both and compares them. `off` adds no work and no output.
    #[arg(long, value_enum, default_value_t = SkillScoreArg::Off)]
    pub skill_score: SkillScoreArg,

    /// Print a one-line explanation of every statistic shown in the summary
    /// (the full reference is docs/STATISTICS.md, `cli stats-doc`).
    #[arg(long)]
    pub explain: bool,

    /// Bootstrap resamples for the standard error of the strength rating;
    /// 0 skips the standard error.
    #[arg(long, default_value_t = 200)]
    pub bootstrap_resamples: usize,

    /// Measure whether voluntary passes are useful: for a sample of them the
    /// rest of the round is replayed K times after the pass and after the
    /// weakest and strongest legal play. Off unless given; costs K x 3
    /// rollouts per sampled pass. Replays the matches of the batch as
    /// ordinary matches, with `--seed` as the analysis seed.
    #[arg(
        long,
        value_name = "K",
        value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(1..)
    )]
    pub useful_passes: Option<usize>,

    /// Fraction of the voluntary passes analysed by `--useful-passes`, 0..=1.
    #[arg(long, default_value_t = 0.05, value_parser = parse_fraction)]
    pub useful_pass_sample: f64,

    /// A pass counts as useful only if it beats the best alternative by more
    /// than this many places.
    #[arg(long, default_value_t = 0.0, value_parser = parse_margin)]
    pub useful_pass_margin: f64,
}

fn parse_fraction(text: &str) -> Result<f64, String> {
    let value: f64 = text
        .parse()
        .map_err(|_| format!("`{text}` is not a number"))?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err("must be between 0 and 1".to_owned())
    }
}

fn parse_margin(text: &str) -> Result<f64, String> {
    let value: f64 = text
        .parse()
        .map_err(|_| format!("`{text}` is not a number"))?;
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err("must be a finite number, 0 or more".to_owned())
    }
}

/// Which luck/skill measurement a batch runs (`--skill-score`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SkillScoreArg {
    Off,
    Estimate,
    Duplicate,
    Both,
}

impl SkillScoreArg {
    /// Whether the run plays duplicate groups.
    #[must_use]
    pub fn uses_duplicate(self) -> bool {
        matches!(self, Self::Duplicate | Self::Both)
    }
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
    Neat(NeatSpec),
}

/// A `neat:PATH` spec. The genome file is loaded and checked while the
/// argument is parsed, so a missing, malformed or stale file is reported
/// as an argument error before any simulation starts.
#[derive(Clone, Debug)]
pub struct NeatSpec {
    path: PathBuf,
    strategy: Arc<sim::NeatStrategy>,
}

impl PartialEq for NeatSpec {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
    }
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
        if head == "neat" {
            let path = options
                .map(str::trim)
                .filter(|p| !p.is_empty())
                .ok_or("`neat` needs a genome file: neat:PATH")?;
            let path = PathBuf::from(path);
            let strategy = sim::NeatStrategy::from_file(&path).map_err(|e| e.to_string())?;
            return Ok(Self::Neat(NeatSpec {
                path,
                strategy: Arc::new(strategy),
            }));
        }
        if options.is_some() {
            return Err(format!(
                "strategy `{head}` takes no options (only `adaptive:` and `neat:` do)"
            ));
        }
        <FixedStrategy as clap::ValueEnum>::from_str(head, false)
            .map(Self::Fixed)
            .map_err(|_| {
                format!(
                    "unknown strategy `{head}`; expected one of: {}, adaptive[:OPTIONS], neat:PATH",
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
            Self::Neat(spec) => spec.strategy,
        }
    }
}

/// Cross-field checks `clap` can't express declaratively.
///
/// # Errors
///
/// Returns an error if `args.strategies.len()` doesn't equal
/// `args.player_count` — exactly one strategy is required per seat — or if
/// a duplicate skill score is asked for and `args.matches` is not a
/// multiple of `args.player_count`.
pub fn validate(args: &Args) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.strategies.len() == usize::from(args.player_count),
        "expected {} strategies (one per seat), got {}",
        args.player_count,
        args.strategies.len()
    );
    anyhow::ensure!(
        !args.skill_score.uses_duplicate() || args.matches.is_multiple_of(args.strategies.len()),
        "--skill-score {} plays groups of {} matches (one per rotation of the {} strategies), \
         so --matches must be a multiple of --player-count ({}), got {}",
        if args.skill_score == SkillScoreArg::Both {
            "both"
        } else {
            "duplicate"
        },
        args.player_count,
        args.player_count,
        args.player_count,
        args.matches
    );
    // Results are grouped by player name, so two *different* genome
    // files sharing a name (`runs/a/champion.json`, `runs/b/champion.json`)
    // would be silently merged into one row.
    let mut names: Vec<(&str, &Path)> = Vec::new();
    for arg in &args.strategies {
        if let StrategyArg::Neat(spec) = arg {
            let name = sim::Strategy::name(&*spec.strategy);
            match names.iter().find(|(seen, _)| *seen == name) {
                Some((_, path)) => anyhow::ensure!(
                    *path == spec.path.as_path(),
                    "{name} would name two different genome files ({} and {}); rename one so \
                     their results stay separate",
                    path.display(),
                    spec.path.display()
                ),
                None => names.push((name, &spec.path)),
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
