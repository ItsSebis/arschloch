//! `cli train-advisor`: probes this machine and the candidate opponents and
//! recommends settings for `cli train` (docs/TRAINING.md, "Choosing settings").

// Match and round counts and durations here are small and non-negative.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use clap::{Parser, ValueEnum};
use engine::{DuplicateRule, ExchangeRule, PassRule};
use sim::training::evaluate::{evaluate_with_round1, Round1Obs};
use sim::training::skill_term::{adjusted_score, fit_deal_model};
use sim::training::{match_seed, Opponents, RunDir, TableSpec};
use sim::{NeatStrategy, Strategy};

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
/// Reference players at most: the probe matches of a candidate are split over them.
const MAX_REFERENCES: usize = 6;

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

impl RunPlan {
    /// Why this plan cannot do what the advice promises (at least `MIN_GENERATIONS`
    /// within the budget), in words; `None` when it can.
    pub fn problem(&self) -> Option<String> {
        (self.generations < MIN_GENERATIONS || self.estimated_seconds > self.budget_seconds).then(|| {
            format!(
                "only {} generations fit and the plan needs about {:.0} s of the {:.0} s budget even at the smallest settings: give it more time",
                self.generations, self.estimated_seconds, self.budget_seconds
            )
        })
    }
}

/// The spread of one match's score over several opponents: the root of the mean
/// variance (deviations of different opponents are not averaged directly).
pub fn pooled_sd(sds: &[f64]) -> f64 {
    if sds.is_empty() {
        return 0.0;
    }
    (sds.iter().map(|sd| sd * sd).sum::<f64>() / sds.len() as f64).sqrt()
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
    pool_len: usize,
) -> RunPlan {
    // One generation, as the trainer counts it: every genome's matches, plus one
    // re-evaluation of each champion candidate, the champion and every pool opponent.
    let seconds_per_generation = |population: usize, matches: usize| {
        (population * matches + REEVAL_MATCHES * (CHAMPION_CANDIDATES + 1 + pool_len)) as f64
            * rounds as f64
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
    // A folder with spaces is quoted so the line can be pasted into a shell and
    // split again by the desktop window.
    let out = if out.contains(char::is_whitespace) {
        format!("\"{out}\"")
    } else {
        out.to_owned()
    };
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

/// `count` of `items`, spread evenly over the list (all of them when it is short enough).
fn thin<T: Clone>(items: &[T], count: usize) -> Vec<T> {
    if items.len() <= count {
        return items.to_vec();
    }
    (0..count)
        .map(|i| items[i * items.len() / count].clone())
        .collect()
}

/// A player the candidates are scored against.
#[derive(Clone)]
struct Reference {
    name: String,
    strategy: Arc<dyn Strategy>,
}

/// The reference players the options ask for; `adaptive` when none was asked for.
fn build_references(args: &AdvisorArgs) -> anyhow::Result<Vec<Reference>> {
    let mut found = Vec::new();
    for spec in &args.reference {
        let strategy = spec
            .parse::<StrategyArg>()
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("invalid --reference `{spec}`"))?
            .build();
        found.push(Reference {
            name: spec.clone(),
            strategy,
        });
    }
    if let Some(folder) = &args.reference_models {
        found.extend(models_in(folder)?);
    }
    if let Some(run) = &args.reference_hall {
        found.extend(hall_of(run)?);
    }
    if found.is_empty() {
        found.push(Reference {
            name: "adaptive".to_owned(),
            strategy: "adaptive"
                .parse::<StrategyArg>()
                .map_err(anyhow::Error::msg)?
                .build(),
        });
    }
    Ok(thin(&found, MAX_REFERENCES))
}

/// Every genome file of `folder` that this build can load, by file name.
fn models_in(folder: &Path) -> anyhow::Result<Vec<Reference>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(folder)
        .with_context(|| {
            format!(
                "--reference-models {}: cannot read the folder",
                folder.display()
            )
        })?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    let models: Vec<Reference> = files
        .iter()
        .filter_map(|path| {
            let strategy = NeatStrategy::from_file(path).ok()?;
            let name = path.file_stem()?.to_string_lossy().into_owned();
            Some(Reference {
                name: format!("neat:{name}"),
                strategy: Arc::new(strategy),
            })
        })
        .collect();
    anyhow::ensure!(
        !models.is_empty(),
        "--reference-models {}: no genome file (*.json written by `cli train`) that this build can load",
        folder.display()
    );
    Ok(models)
}

/// The hall-of-fame champions stored in the checkpoint of an earlier run.
fn hall_of(run: &Path) -> anyhow::Result<Vec<Reference>> {
    let checkpoint = RunDir::open_existing(run)
        .and_then(|dir| dir.read_checkpoint())
        .with_context(|| format!("--reference-hall {}: not a run folder", run.display()))?;
    anyhow::ensure!(
        !checkpoint.hall_of_fame.is_empty(),
        "--reference-hall {}: the run has no hall of fame yet (it needs --hall-of-fame and a few generations)",
        run.display()
    );
    checkpoint
        .hall_of_fame
        .iter()
        .map(|member| {
            let name = format!("HoF({})", member.generation);
            let strategy = NeatStrategy::new(name.clone(), &member.genome)
                .with_context(|| format!("--reference-hall {}: {name}", run.display()))?;
            Ok(Reference {
                name,
                strategy: Arc::new(strategy),
            })
        })
        .collect()
}

/// What the probe measured about one candidate opponent.
struct Row {
    spec: String,
    /// The reference players' mean score against it (+1 best .. -1 worst) and its standard error.
    mean: f64,
    error: f64,
    verdict: Verdict,
    /// The spread of one match's score against it.
    sd: f64,
    /// Rounds played and seconds spent probing this candidate.
    rounds: usize,
    seconds: f64,
    /// Variance of one match's round-1 score, and of the same score after the regression on
    /// the six hand features (`skill_term`): what the luck adjustment would leave of it.
    var_round1: f64,
    var_adjusted: f64,
}

/// What the probe measured.
struct Probe {
    rows: Vec<Row>,
    /// Rounds per second over the opponents of the recommended pool only (set once the pool
    /// is chosen): a dropped candidate that plays slower or faster must not set the speed.
    rounds_per_sec: f64,
    /// Standard deviation of one match's score.
    sd_match: f64,
    /// Share of the round-1 variance that the hand explains (pool opponents only).
    luck_share: f64,
}

/// Rounds per second over the given probe rows.
fn speed_of(rows: &[&Row]) -> f64 {
    let seconds: f64 = rows.iter().map(|r| r.seconds).sum();
    let rounds: usize = rows.iter().map(|r| r.rounds).sum();
    rounds as f64 / seconds.max(1e-9)
}

fn variance(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

/// Variances of the round-1 scores and of the luck-adjusted ones, from one fit over all of them.
fn round1_variances(observations: &[Round1Obs]) -> (f64, f64) {
    let Some(model) = fit_deal_model(std::slice::from_ref(&observations.to_vec())) else {
        return (0.0, 0.0);
    };
    let plain: Vec<f64> = observations.iter().map(|o| o.score).collect();
    let adjusted: Vec<f64> = observations
        .iter()
        .map(|o| adjusted_score(&model, std::slice::from_ref(o)))
        .collect();
    (variance(&plain), variance(&adjusted))
}

fn run_probe(
    args: &AdvisorArgs,
    candidates: &[String],
    references: &[Reference],
    table: &TableSpec,
) -> anyhow::Result<Probe> {
    let mut rows = Vec::new();
    // The probe matches of a candidate are split over the references.
    let per_reference = args.probe_matches.div_ceil(references.len()).max(2);
    for (index, spec) in candidates.iter().enumerate() {
        let opponent = spec
            .parse::<StrategyArg>()
            .map_err(anyhow::Error::msg)?
            .build();
        let (mut means, mut variances, mut sds) = (Vec::new(), 0.0, Vec::new());
        let (mut rounds, mut observations) = (0usize, Vec::new());
        let started = Instant::now();
        for (r, reference) in references.iter().enumerate() {
            let seeds: Vec<u64> = (0..per_reference as u64)
                .map(|i| match_seed(args.seed, 0xAD + index as u64 + 0x1000 * r as u64, i))
                .collect();
            let (score, obs) = evaluate_with_round1(
                &reference.strategy,
                table,
                Opponents::Only(&opponent),
                &seeds,
            );
            rounds += score.matches * table.rounds;
            means.push(score.mean);
            variances += score.std_error * score.std_error;
            sds.push(score.std_error * (score.matches as f64).sqrt());
            observations.extend(obs);
        }
        let seconds = started.elapsed().as_secs_f64();
        let k = references.len() as f64;
        let mean = means.iter().sum::<f64>() / k;
        let (var_round1, var_adjusted) = round1_variances(&observations);
        rows.push(Row {
            spec: spec.clone(),
            mean,
            error: variances.sqrt() / k,
            verdict: judge(mean),
            sd: pooled_sd(&sds),
            rounds,
            seconds,
            var_round1,
            var_adjusted,
        });
    }
    Ok(Probe {
        rows,
        rounds_per_sec: 0.0,
        sd_match: 0.0,
        luck_share: 0.0,
    })
}

/// The text the advisor prints: what was measured, what was chosen and why. With `auto` the
/// last line says the run starts now, instead of a command to copy.
fn render(
    args: &AdvisorArgs,
    references: &[Reference],
    probe: &Probe,
    pool: &[String],
    plan: &RunPlan,
    command: &str,
    auto: bool,
) -> String {
    let mut out = String::new();
    let names: Vec<String> = references.iter().map(|r| format!("`{}`", r.name)).collect();
    let _ = writeln!(
        out,
        "Probe: {} matches per candidate, reference player{} {}.",
        args.probe_matches,
        if references.len() > 1 { "s" } else { "" },
        names.join(", ")
    );
    if references.len() > 1 {
        let _ = writeln!(
            out,
            "  The matches are split over the references ({} against each); a candidate's score is their mean.",
            args.probe_matches.div_ceil(references.len()).max(2)
        );
    }
    let _ = writeln!(
        out,
        "\nCandidate opponents (the reference player's score against each, +1 best .. -1 worst):"
    );
    for row in &probe.rows {
        let _ = writeln!(
            out,
            "  {:<44} {:+.3} +- {:.3}  {}",
            row.spec,
            row.mean,
            row.error,
            row.verdict.reason()
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
        "Population {} and {} generations.\n  Why: the pool opponents were played at about {:.0} rounds per second; the plan needs about {:.0} s of the {:.0} s budget \
         (the rest is kept for what the estimate leaves out).",
        plan.population, plan.generations, probe.rounds_per_sec, plan.estimated_seconds, plan.budget_seconds
    );
    if plan.hall_of_fame > 0 {
        let _ = writeln!(out, "Hall of fame {} every {} generations.\n  Why: a long run needs past champions as opponents.", plan.hall_of_fame, plan.hall_interval);
    }
    let _ = writeln!(
        out,
        "Luck adjustment: the six hand features explain {:.0}% of the variance of a round-1 score against the pool.\n  \
         Why this is only reported: every genome of a generation plays the same deals, so `--fitness-skill-weight` cannot cancel \
         deal luck inside a generation (it only weights round 1 more). The duplicate-deal skill score cuts this noise when \
         comparing finished players (`cli --skill-score duplicate`, `cli evaluate`).",
        100.0 * probe.luck_share
    );
    if auto {
        let _ = writeln!(
            out,
            "\nStarting the run with these settings (the same as:\n  {command}\n)"
        );
    } else {
        let _ = writeln!(out, "\nRun it with:\n  {command}");
    }
    out
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
    probe.luck_share = if var_plain > 0.0 {
        (1.0 - var_adj / var_plain).max(0.0)
    } else {
        0.0
    };
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
        let roomy = plan_run(36_000.0, 1000.0, 8, 100, 150, 4);
        assert_eq!(roomy.population, 150);
        assert!(roomy.generations >= 100, "{roomy:?}");
        let tight = plan_run(900.0, 1000.0, 8, 100, 150, 4);
        assert!(tight.population < 150, "{tight:?}");
        assert!(tight.generations >= MIN_GENERATIONS, "{tight:?}");
        // The plan fits the time it was given.
        for plan in [roomy, tight] {
            assert!(plan.estimated_seconds <= plan.budget_seconds, "{plan:?}");
        }
    }

    #[test]
    fn the_cost_model_counts_what_the_trainer_plays() {
        // The trainer plays population x matches plus reeval (200) matches for each of the
        // champion candidates (5), the champion itself and every pool opponent.
        let plan = plan_run(3600.0, 1000.0, 8, 100, 150, 4);
        let per_generation =
            (plan.population * plan.matches_per_genome + 200 * (5 + 1 + 4)) as f64 * 8.0 / 1000.0;
        let expected = f64::from(plan.generations) * per_generation;
        assert!((plan.estimated_seconds - expected).abs() < 1e-6, "{plan:?}");
        // A bigger pool costs more per generation, so fewer generations fit.
        let bigger_pool = plan_run(3600.0, 1000.0, 8, 100, 150, 8);
        assert!(bigger_pool.generations <= plan.generations);
    }

    #[test]
    fn a_plan_that_cannot_fit_says_so() {
        let hopeless = plan_run(10.0, 1000.0, 8, 100, 150, 4);
        let problem = hopeless
            .problem()
            .expect("10 seconds cannot train anything");
        assert!(
            problem.contains("generations") && problem.contains("budget"),
            "{problem}"
        );
        assert!(plan_run(36_000.0, 1000.0, 8, 100, 150, 4)
            .problem()
            .is_none());
    }

    #[test]
    fn the_spread_is_pooled_as_a_variance_not_an_average_of_deviations() {
        // Two opponents with spreads 0.1 and 0.5: the pooled spread is the root of the mean variance.
        let pooled = pooled_sd(&[0.1, 0.5]);
        assert!((pooled - (0.13_f64).sqrt()).abs() < 1e-12, "{pooled}");
        assert!(pooled > 0.3, "larger than the plain average 0.3");
        assert!(pooled_sd(&[]).abs() < 1e-12);
    }

    #[test]
    fn a_probe_needs_at_least_two_matches() {
        for bad in ["0", "1"] {
            assert!(
                AdvisorArgs::try_parse_from(["cli train-advisor", "--probe-matches", bad]).is_err(),
                "{bad}"
            );
        }
        assert!(AdvisorArgs::try_parse_from(["cli train-advisor", "--probe-matches", "2"]).is_ok());
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
    fn a_run_folder_with_spaces_is_quoted() {
        let plan = RunPlan {
            population: 10,
            generations: 2,
            matches_per_genome: 4,
            rounds: 1,
            hall_of_fame: 0,
            hall_interval: 0,
            estimated_seconds: 1.0,
            budget_seconds: 2.0,
        };
        let command = command_line(&plan, 4, "single", &[], "my runs/a b");
        assert!(
            command.starts_with("cli train --out \"my runs/a b\" --player-count 4 "),
            "{command}"
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
        let text = advise(&args, false).unwrap().text;
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
        assert!(advise(&args, false)
            .err()
            .unwrap()
            .to_string()
            .contains("--time-budget"));
    }

    fn row(spec: &str, rounds: usize, seconds: f64) -> Row {
        Row {
            spec: spec.into(),
            mean: 0.0,
            error: 0.0,
            verdict: Verdict::Keep,
            sd: 0.3,
            rounds,
            seconds,
            var_round1: 1.0,
            var_adjusted: 0.8,
        }
    }

    #[test]
    fn the_speed_is_that_of_the_chosen_pool_not_of_every_candidate() {
        let rows = [row("fast", 1000, 1.0), row("slow-and-dropped", 1000, 99.0)];
        let chosen: Vec<&Row> = rows.iter().filter(|r| r.spec == "fast").collect();
        assert!((speed_of(&chosen) - 1000.0).abs() < 1e-9);
        let all: Vec<&Row> = rows.iter().collect();
        assert!(
            (speed_of(&all) - 20.0).abs() < 1e-9,
            "what the old code measured"
        );
    }

    #[test]
    fn references_are_thinned_evenly_and_default_to_adaptive() {
        assert_eq!(thin(&[1, 2, 3], 6), vec![1, 2, 3]);
        assert_eq!(thin(&(0..12).collect::<Vec<_>>(), 4), vec![0, 3, 6, 9]);
        let args = AdvisorArgs::try_parse_from(["cli train-advisor"]).unwrap();
        let refs = build_references(&args).unwrap();
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].name, "adaptive");
    }

    #[test]
    fn bad_references_are_explained() {
        let bad = |extra: &[&str]| {
            let args = AdvisorArgs::try_parse_from(
                std::iter::once("cli train-advisor").chain(extra.iter().copied()),
            )
            .unwrap();
            format!("{:#}", build_references(&args).err().unwrap())
        };
        assert!(bad(&["--reference", "nonsense"]).contains("--reference `nonsense`"));
        assert!(bad(&["--reference-models", "/nonexistent-folder"]).contains("--reference-models"));
        assert!(bad(&["--reference-hall", "/nonexistent-folder"]).contains("--reference-hall"));
        let empty =
            std::env::temp_dir().join(format!("arschloch-advisor-empty-{}", std::process::id()));
        std::fs::create_dir_all(&empty).unwrap();
        assert!(bad(&["--reference-models", empty.to_str().unwrap()]).contains("no genome file"));
        std::fs::remove_dir_all(&empty).unwrap();
    }

    #[test]
    fn the_luck_adjustment_cannot_leave_more_variance_than_the_scores_had() {
        // The score is the hand's strength plus noise: the regression removes the first part.
        let observations: Vec<Round1Obs> = (0..60)
            .map(|i| {
                let x = f64::from(i % 10) / 10.0;
                let noise = f64::from((i * 7) % 5) / 50.0;
                Round1Obs {
                    score: 2.0 * x + noise,
                    features: [x, 0.0, 0.0, 0.0, 0.0, 0.0],
                }
            })
            .collect();
        let (plain, adjusted) = round1_variances(&observations);
        assert!(plain > 0.0 && adjusted < 0.2 * plain, "{plain} {adjusted}");
        assert_eq!(round1_variances(&[]), (0.0, 0.0));
    }

    #[test]
    fn several_references_are_split_over_the_probe_matches_and_the_run_is_auto_ready() {
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
            "--reference",
            "lowest-legal",
            "--reference",
            "random-legal",
            "--out",
            "runs/x",
        ])
        .unwrap();
        let advice = advise(&args, true).unwrap();
        assert!(
            advice
                .text
                .contains("references `lowest-legal`, `random-legal`")
                || advice.text.contains("`lowest-legal`, `random-legal`"),
            "{}",
            advice.text
        );
        assert!(advice.text.contains("4 against each"), "{}", advice.text);
        assert!(advice.text.contains("Luck adjustment"), "{}", advice.text);
        assert!(advice.text.contains("Starting the run"), "{}", advice.text);
        assert!(!advice.pool.is_empty() && advice.plan.population >= MIN_POPULATION);
    }
}
