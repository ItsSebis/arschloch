//! `cli train-advisor`: probes this machine and the candidate opponents and
//! recommends settings for `cli train` (docs/TRAINING.md, "Choosing settings").

// Match and round counts and durations here are small and non-negative.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod plan;
mod probe;
mod references;
mod render;

use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, ValueEnum};
use engine::{DuplicateRule, ExchangeRule, PassRule};
use sim::training::TableSpec;

use self::plan::{
    command_line, matches_for_error, parse_duration, plan_run, pooled_sd, RunPlan, Verdict,
    POOL_SIZE, TARGET_ERROR,
};
use self::probe::{run_probe, speed_of, Row, MIN_LUCK_OBSERVATIONS};
use self::references::build_references;
use self::render::render;
use crate::args::DeckVariantArg;
use crate::evaluate::DEFAULT_BATTERY;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Preset {
    /// A few minutes: a sanity run.
    Quick,
    /// Two hours.
    Thorough,
    /// Ten hours.
    Overnight,
}

impl Preset {
    pub fn budget_seconds(self) -> f64 {
        match self {
            Self::Quick => 300.0,
            Self::Thorough => 7200.0,
            Self::Overnight => 36_000.0,
        }
    }
}

/// Recommend settings for `cli train` from a short probe of this machine.
#[derive(Parser, Debug)]
#[command(
    name = "cli train-advisor",
    about = "Recommend settings for `cli train` from a short probe of this machine"
)]
pub struct AdvisorArgs {
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u8).range(3..=6))]
    pub player_count: u8,
    #[arg(long, value_enum, default_value_t = DeckVariantArg::Single)]
    pub deck_variant: DeckVariantArg,
    /// How long the training may take (`30m`, `2h`, `1h30m`). Default: the preset's.
    #[arg(long)]
    pub time_budget: Option<String>,
    #[arg(long, value_enum, default_value_t = Preset::Thorough)]
    pub preset: Preset,
    /// A candidate opponent to judge (`--strategy` syntax); repeat for several.
    /// Default: the evaluation battery of `cli evaluate`.
    #[arg(long, value_name = "SPEC")]
    pub opponent: Vec<String>,
    /// A player the candidates are scored against (stands in for the model); repeat for
    /// several. `neat:PATH` is a starting genome. Without any reference option: `adaptive`.
    #[arg(long, value_name = "SPEC")]
    pub reference: Vec<String>,
    /// Score the candidates against the genomes (`*.json`) of this folder (at most six are used).
    #[arg(long, value_name = "DIR")]
    pub reference_models: Option<PathBuf>,
    /// Score the candidates against the hall of fame of an earlier run in this run folder.
    #[arg(long, value_name = "RUN_DIR")]
    pub reference_hall: Option<PathBuf>,
    /// Probe matches per candidate opponent (at least 2: the spread needs two), split over the
    /// reference players (at least 2 against each).
    #[arg(long, default_value_t = 40, value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(2..))]
    pub probe_matches: usize,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Run directory written into the recommended command.
    #[arg(long, default_value = "runs/auto")]
    pub out: String,
}

/// What the advisor recommends.
pub struct Advice {
    pub text: String,
    pub plan: RunPlan,
    pub pool: Vec<String>,
}

/// The advice for `args`: runs the probe. With `auto` the text ends in "starting the run".
pub fn advise(args: &AdvisorArgs, auto: bool) -> anyhow::Result<Advice> {
    let budget = match &args.time_budget {
        Some(text) => parse_duration(text)
            .with_context(|| format!("--time-budget `{text}`: write e.g. 30m, 2h or 1h30m"))?,
        None => args.preset.budget_seconds(),
    };
    let candidates: Vec<String> = if args.opponent.is_empty() {
        DEFAULT_BATTERY.iter().map(|&s| s.to_owned()).collect()
    } else {
        args.opponent.clone()
    };
    let references = build_references(args)?;
    let table = TableSpec {
        player_count: args.player_count,
        deck_variant: args.deck_variant.into(),
        duplicate_rule: DuplicateRule::FirstDealtWins,
        pass_rule: PassRule::Final,
        exchange_rule: ExchangeRule::Forced,
        rounds: 8,
    };
    let mut probe = run_probe(args, &candidates, &references, &table)?;
    let mut kept: Vec<&Row> = probe
        .rows
        .iter()
        .filter(|r| r.verdict == Verdict::Keep)
        .collect();
    kept.sort_by(|a, b| a.mean.total_cmp(&b.mean)); // hardest first
    let mut pool: Vec<String> = kept
        .iter()
        .take(POOL_SIZE)
        .map(|r| r.spec.clone())
        .collect();
    let mut warning = String::new();
    if pool.is_empty() {
        pool = probe
            .rows
            .iter()
            .take(POOL_SIZE)
            .map(|r| r.spec.clone())
            .collect();
        warning = format!("Warning: no candidate had headroom against the reference player; keeping the first {POOL_SIZE}.\n\n");
    }
    // Spread, speed and luck of the opponents that will actually be played, not of every candidate.
    let chosen: Vec<&Row> = probe
        .rows
        .iter()
        .filter(|r| pool.contains(&r.spec))
        .collect();
    probe.sd_match = pooled_sd(&chosen.iter().map(|r| r.sd).collect::<Vec<_>>());
    probe.rounds_per_sec = speed_of(&chosen);
    let (var_plain, var_adj): (f64, f64) = chosen.iter().fold((0.0, 0.0), |(p, a), r| {
        (p + r.var_round1, a + r.var_adjusted)
    });
    probe.luck_observations = chosen.iter().map(|r| r.observations).sum();
    probe.luck_share = (var_plain > 0.0 && probe.luck_observations >= MIN_LUCK_OBSERVATIONS)
        .then(|| (1.0 - var_adj / var_plain).max(0.0));
    let matches = matches_for_error(probe.sd_match, TARGET_ERROR);
    let plan = plan_run(
        budget,
        probe.rounds_per_sec,
        table.rounds,
        matches,
        150,
        pool.len(),
    );
    if let Some(problem) = plan.problem() {
        let _ = writeln!(warning, "Warning: {problem}.\n");
    }
    let deck = args
        .deck_variant
        .to_possible_value()
        .map_or_else(String::new, |v| v.get_name().to_owned());
    let command = command_line(&plan, args.player_count, &deck, &pool, &args.out);
    let text = warning + &render(args, &references, &probe, &pool, &plan, &command, auto);
    Ok(Advice { text, plan, pool })
}

pub fn run(raw_args: impl Iterator<Item = String>) -> anyhow::Result<()> {
    let args =
        AdvisorArgs::parse_from(std::iter::once("cli train-advisor".to_owned()).chain(raw_args));
    print!("{}", advise(&args, false)?.text);
    Ok(())
}
#[cfg(test)]
mod tests;
