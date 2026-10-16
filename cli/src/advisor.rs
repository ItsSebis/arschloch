//! `cli train-advisor`: probes this machine and the candidate opponents and
//! recommends settings for `cli train` (docs/TRAINING.md, "Choosing settings").

// Match and round counts and durations here are small and non-negative.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::fmt::Write as _;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use clap::{Parser, ValueEnum};
use engine::{DuplicateRule, ExchangeRule, PassRule};
use sim::training::{evaluate, match_seed, Opponents, TableSpec};
use sim::Strategy;

use crate::args::{DeckVariantArg, StrategyArg};
use crate::evaluate::DEFAULT_BATTERY;

pub const MIN_MATCHES: usize = 20;
pub const MAX_MATCHES: usize = 400;
pub const MIN_GENERATIONS: u32 = 20;
const MAX_GENERATIONS: u32 = 1000;
const MIN_POPULATION: usize = 30;
/// The standard error of one fitness measurement the run should reach.
const TARGET_ERROR: f64 = 0.02;
/// Share of the time budget the plan may spend: the rest covers what the
/// estimate leaves out (checkpoints, a candidate slower than the probe).
const BUDGET_SHARE: f64 = 0.7;
/// Candidates and matches of the champion re-evaluations, as `cli train` defaults them.
const CHAMPION_CANDIDATES: usize = 5;
const REEVAL_MATCHES: usize = 200;
/// Opponents in the recommended pool at most.
const POOL_SIZE: usize = 4;

/// `90s`, `30m`, `2h`, `1h30m` in seconds.
pub fn parse_duration(text: &str) -> Option<f64> {
    let (mut total, mut number, mut any) = (0.0, String::new(), false);
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            number.push(ch);
            continue;
        }
        let unit = match ch {
            's' => 1.0,
            'm' => 60.0,
            'h' => 3600.0,
            _ => return None,
        };
        total += number.parse::<f64>().ok()? * unit;
        number.clear();
        any = true;
    }
    (any && number.is_empty() && total > 0.0).then_some(total)
}

/// Matches per genome for a standard error of `target` when one match's score
/// varies with standard deviation `sd`: rounded up to tens, within the limits.
pub fn matches_for_error(sd: f64, target: f64) -> usize {
    let needed = ((sd / target).powi(2).ceil() as usize).div_ceil(10) * 10;
    needed.clamp(MIN_MATCHES, MAX_MATCHES)
}

#[derive(Debug, Clone, Copy)]
pub struct RunPlan {
    pub population: usize,
    pub generations: u32,
    pub matches_per_genome: usize,
    pub rounds: usize,
    pub hall_of_fame: usize,
    pub hall_interval: u32,
    pub estimated_seconds: f64,
    pub budget_seconds: f64,
}

/// Population, generations and matches per genome that fit `budget` seconds at
/// `rounds_per_sec`. The wanted `matches` and `max_population` shrink (population
/// first) until at least `MIN_GENERATIONS` fit.
pub fn plan_run(
    budget: f64,
    rounds_per_sec: f64,
    rounds: usize,
    matches: usize,
    max_population: usize,
) -> RunPlan {
    // One generation: every genome's matches, the champion candidates' matches and one re-evaluation.
    let seconds_per_generation = |population: usize, matches: usize| {
        ((population + CHAMPION_CANDIDATES) * matches + REEVAL_MATCHES) as f64 * rounds as f64
            / rounds_per_sec
    };
    let usable = budget * BUDGET_SHARE;
    let (mut population, mut matches) = (max_population, matches);
    let generations = loop {
        let fit = (usable / seconds_per_generation(population, matches)).floor() as u32;
        if fit >= MIN_GENERATIONS || (population <= MIN_POPULATION && matches <= MIN_MATCHES) {
            break fit.clamp(1, MAX_GENERATIONS);
        }
        if population > MIN_POPULATION {
            population = (population * 2 / 3).max(MIN_POPULATION);
        } else {
            matches = (matches / 2).max(MIN_MATCHES);
        }
    };
    RunPlan {
        population,
        generations,
        matches_per_genome: matches,
        rounds,
        hall_of_fame: if generations >= 60 { 5 } else { 0 },
        hall_interval: (generations / 20).max(5),
        estimated_seconds: f64::from(generations) * seconds_per_generation(population, matches),
        budget_seconds: budget,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Keep,
    TooWeak,
    TooStrong,
}

/// Is an opponent that the reference player scores `reference_mean` against
/// (+1 best .. -1 worst) useful for training?
pub fn judge(reference_mean: f64) -> Verdict {
    if reference_mean > 0.85 {
        Verdict::TooWeak
    } else if reference_mean < -0.6 {
        Verdict::TooStrong
    } else {
        Verdict::Keep
    }
}

impl Verdict {
    pub fn reason(self) -> &'static str {
        match self {
            Self::Keep => "kept: there is room to improve and to lose",
            Self::TooWeak => "dropped, too weak: the reference player wins almost every game, so it teaches little",
            Self::TooStrong => "dropped, too strong: the reference player almost never wins, so there is no signal",
        }
    }
}

/// The `cli train` command line that carries the plan out.
pub fn command_line(
    plan: &RunPlan,
    players: u8,
    deck: &str,
    opponents: &[String],
    out: &str,
) -> String {
    let mut line = format!(
        "cli train --out {out} --player-count {players} --deck-variant {deck} --population {} --generations {} \
         --matches-per-genome {} --rounds {}",
        plan.population, plan.generations, plan.matches_per_genome, plan.rounds
    );
    if plan.hall_of_fame > 0 {
        let _ = write!(
            line,
            " --hall-of-fame {} --hall-interval {}",
            plan.hall_of_fame, plan.hall_interval
        );
    }
    for opponent in opponents {
        let _ = write!(line, " --opponent {opponent}");
    }
    line
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Preset {
    /// A few minutes: a sanity run.
    Quick,
    /// Two hours.
    Thorough,
    /// Ten hours.
    Overnight,
}

impl Preset {
    fn budget_seconds(self) -> f64 {
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
struct AdvisorArgs {
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u8).range(3..=6))]
    player_count: u8,
    #[arg(long, value_enum, default_value_t = DeckVariantArg::Single)]
    deck_variant: DeckVariantArg,
    /// How long the training may take (`30m`, `2h`, `1h30m`). Default: the preset's.
    #[arg(long)]
    time_budget: Option<String>,
    #[arg(long, value_enum, default_value_t = Preset::Thorough)]
    preset: Preset,
    /// A candidate opponent to judge (`--strategy` syntax); repeat for several.
    /// Default: the evaluation battery of `cli evaluate`.
    #[arg(long, value_name = "SPEC")]
    opponent: Vec<String>,
    /// The player the candidates are scored against (stands in for the model).
    #[arg(long, default_value = "adaptive")]
    reference: String,
    /// Probe matches per candidate opponent.
    #[arg(long, default_value_t = 40)]
    probe_matches: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Run directory written into the recommended command.
    #[arg(long, default_value = "runs/auto")]
    out: String,
}

/// What the probe measured.
struct Probe {
    /// Per candidate: spec, the reference player's score against it, its standard error, the verdict.
    rows: Vec<(String, f64, f64, Verdict)>,
    rounds_per_sec: f64,
    /// Standard deviation of one match's score.
    sd_match: f64,
}

fn run_probe(
    args: &AdvisorArgs,
    candidates: &[String],
    reference: &Arc<dyn Strategy>,
    table: &TableSpec,
) -> anyhow::Result<Probe> {
    let (mut rows, mut rounds, mut sd_sum) = (Vec::new(), 0usize, 0.0);
    let started = Instant::now();
    for (index, spec) in candidates.iter().enumerate() {
        let opponent = spec
            .parse::<StrategyArg>()
            .map_err(anyhow::Error::msg)?
            .build();
        let seeds: Vec<u64> = (0..args.probe_matches as u64)
            .map(|i| match_seed(args.seed, 0xAD + index as u64, i))
            .collect();
        let score = evaluate(reference, table, Opponents::Only(&opponent), &seeds);
        rounds += score.matches * table.rounds;
        sd_sum += score.std_error * (score.matches as f64).sqrt();
        rows.push((spec.clone(), score.mean, score.std_error, judge(score.mean)));
    }
    Ok(Probe {
        rows,
        rounds_per_sec: rounds as f64 / started.elapsed().as_secs_f64().max(1e-9),
        sd_match: sd_sum / candidates.len() as f64,
    })
}

/// The text the advisor prints: what was measured, what was chosen and why.
fn render(
    args: &AdvisorArgs,
    probe: &Probe,
    pool: &[String],
    plan: &RunPlan,
    command: &str,
) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "Probe: {} matches per candidate, reference player `{}`.",
        args.probe_matches, args.reference
    );
    let _ = writeln!(
        out,
        "\nCandidate opponents (the reference player's score against each, +1 best .. -1 worst):"
    );
    for (spec, mean, error, verdict) in &probe.rows {
        let _ = writeln!(
            out,
            "  {spec:<44} {mean:+.3} +- {error:.3}  {}",
            verdict.reason()
        );
    }
    let _ = writeln!(
        out,
        "\nRecommended pool: {}.\n  Why: the strongest {} opponents that left headroom.",
        pool.join(", "),
        POOL_SIZE
    );
    let _ = writeln!(
        out,
        "Matches per genome: {}.\n  Why: one match's score varies by {:.3} (measured), so this many matches bring the \
         standard error of a fitness measurement to about {TARGET_ERROR}{}.",
        plan.matches_per_genome,
        probe.sd_match,
        if plan.matches_per_genome < matches_for_error(probe.sd_match, TARGET_ERROR) {
            " only if the time allowed it: it did not, so the error is larger"
        } else {
            ""
        }
    );
    let _ = writeln!(
        out,
        "Population {} and {} generations.\n  Why: this machine plays about {:.0} rounds per second; the plan needs about {:.0} s of the {:.0} s budget \
         (the rest is kept for what the estimate leaves out).",
        plan.population, plan.generations, probe.rounds_per_sec, plan.estimated_seconds, plan.budget_seconds
    );
    if plan.hall_of_fame > 0 {
        let _ = writeln!(out, "Hall of fame {} every {} generations.\n  Why: a long run needs past champions as opponents.", plan.hall_of_fame, plan.hall_interval);
    }
    let _ = writeln!(out, "\nRun it with:\n  {command}");
    out
}

/// The advice for `args`: runs the probe and returns the text to print.
fn advise(args: &AdvisorArgs) -> anyhow::Result<String> {
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
    let reference = args
        .reference
        .parse::<StrategyArg>()
        .map_err(anyhow::Error::msg)?
        .build();
    let table = TableSpec {
        player_count: args.player_count,
        deck_variant: args.deck_variant.into(),
        duplicate_rule: DuplicateRule::FirstDealtWins,
        pass_rule: PassRule::Final,
        exchange_rule: ExchangeRule::Forced,
        rounds: 8,
    };
    let probe = run_probe(args, &candidates, &reference, &table)?;
    let mut kept: Vec<_> = probe.rows.iter().filter(|r| r.3 == Verdict::Keep).collect();
    kept.sort_by(|a, b| a.1.total_cmp(&b.1)); // hardest first
    let mut pool: Vec<String> = kept.iter().take(POOL_SIZE).map(|r| r.0.clone()).collect();
    let mut warning = String::new();
    if pool.is_empty() {
        pool = probe
            .rows
            .iter()
            .take(POOL_SIZE)
            .map(|r| r.0.clone())
            .collect();
        warning = format!("Warning: no candidate had headroom against the reference player; keeping the first {POOL_SIZE}.\n\n");
    }
    let matches = matches_for_error(probe.sd_match, TARGET_ERROR);
    let plan = plan_run(budget, probe.rounds_per_sec, table.rounds, matches, 150);
    let deck = format!("{:?}", args.deck_variant).to_lowercase();
    let command = command_line(&plan, args.player_count, &deck, &pool, &args.out);
    Ok(warning + &render(args, &probe, &pool, &plan, &command))
}

pub fn run(raw_args: impl Iterator<Item = String>) -> anyhow::Result<()> {
    let args =
        AdvisorArgs::parse_from(std::iter::once("cli train-advisor".to_owned()).chain(raw_args));
    print!("{}", advise(&args)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_are_read_in_seconds_minutes_and_hours() {
        assert_eq!(parse_duration("90s"), Some(90.0));
        assert_eq!(parse_duration("30m"), Some(1800.0));
        assert_eq!(parse_duration("2h"), Some(7200.0));
        assert_eq!(parse_duration("1h30m"), Some(5400.0));
        for bad in ["", "30", "m", "1x", "-5m", "0m"] {
            assert_eq!(parse_duration(bad), None, "{bad}");
        }
    }

    #[test]
    fn noisier_matches_need_more_of_them_to_reach_the_target_error() {
        // standard error = sd / sqrt(n): sd 0.4 and a target of 0.02 need 400 matches.
        assert_eq!(matches_for_error(0.4, 0.02), 400);
        assert_eq!(matches_for_error(0.2, 0.02), 100);
        assert_eq!(
            matches_for_error(0.01, 0.02),
            MIN_MATCHES,
            "never fewer than the floor"
        );
        assert_eq!(
            matches_for_error(5.0, 0.02),
            MAX_MATCHES,
            "never more than the cap"
        );
        assert_eq!(matches_for_error(0.213, 0.02) % 10, 0, "rounded up to tens");
    }

    #[test]
    fn the_budget_decides_generations_and_a_tight_one_shrinks_the_population() {
        // 1000 rounds/s, 100 matches x 8 rounds per genome.
        let roomy = plan_run(36_000.0, 1000.0, 8, 100, 150);
        assert_eq!(roomy.population, 150);
        assert!(roomy.generations >= 100, "{roomy:?}");
        let tight = plan_run(300.0, 1000.0, 8, 100, 150);
        assert!(tight.population < 150, "{tight:?}");
        assert!(tight.generations >= MIN_GENERATIONS, "{tight:?}");
        // The plan fits the time it was given.
        for plan in [roomy, tight] {
            assert!(plan.estimated_seconds <= plan.budget_seconds, "{plan:?}");
        }
    }

    #[test]
    fn opponents_with_no_headroom_are_dropped_with_a_reason() {
        assert_eq!(judge(0.95), Verdict::TooWeak);
        assert_eq!(judge(0.30), Verdict::Keep);
        assert_eq!(judge(-0.20), Verdict::Keep);
        assert_eq!(judge(-0.80), Verdict::TooStrong);
        assert!(Verdict::TooWeak.reason().contains("wins almost every"));
        assert!(Verdict::TooStrong.reason().contains("almost never"));
    }

    #[test]
    fn the_recommended_command_carries_every_chosen_setting() {
        let plan = RunPlan {
            population: 80,
            generations: 120,
            matches_per_genome: 150,
            rounds: 8,
            hall_of_fame: 5,
            hall_interval: 6,
            estimated_seconds: 100.0,
            budget_seconds: 200.0,
        };
        let command = command_line(
            &plan,
            4,
            "single",
            &["lowest-legal".into(), "adaptive:reading".into()],
            "runs/auto",
        );
        assert_eq!(
            command,
            "cli train --out runs/auto --player-count 4 --deck-variant single --population 80 --generations 120 \
             --matches-per-genome 150 --rounds 8 --hall-of-fame 5 --hall-interval 6 \
             --opponent lowest-legal --opponent adaptive:reading"
        );
    }

    #[test]
    fn the_advice_runs_a_real_probe_and_ends_in_a_runnable_command() {
        let args = AdvisorArgs::try_parse_from([
            "cli train-advisor",
            "--probe-matches",
            "8",
            "--time-budget",
            "10m",
            "--opponent",
            "lowest-legal",
            "--opponent",
            "random-legal",
            "--out",
            "runs/x",
        ])
        .unwrap();
        let text = advise(&args).unwrap();
        assert!(
            text.contains("lowest-legal") && text.contains("random-legal"),
            "{text}"
        );
        assert!(text.contains("rounds per second"), "{text}");
        let command = text.lines().last().unwrap().trim();
        assert!(
            command.starts_with("cli train --out runs/x --player-count 4 "),
            "{command}"
        );
        assert!(command.contains("--opponent "), "{command}");
        // `cli train` itself accepts the recommended settings.
        let words = command.split_whitespace().skip(2); // `cli train` is dispatched before parsing
        let train =
            crate::train_args::TrainArgs::try_parse_from(std::iter::once("cli train").chain(words));
        assert!(train.is_ok(), "{:?}: {command}", train.err());
    }

    #[test]
    fn a_bad_budget_is_explained() {
        let args =
            AdvisorArgs::try_parse_from(["cli train-advisor", "--time-budget", "soon"]).unwrap();
        assert!(advise(&args)
            .unwrap_err()
            .to_string()
            .contains("--time-budget"));
    }
}
