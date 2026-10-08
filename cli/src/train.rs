//! `cli train`: wires the arguments to `sim::training::Trainer`.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use sim::training::{
    load_config, run_dir_name, DeckChoice, DuplicateChoice, Opponent, RunEnd, SetFile, TrainConfig,
    Trainer,
};

use crate::args::{DeckVariantArg, DuplicateRuleArg, StrategyArg};
use crate::train_args::{TrainArgs, DEFAULT_OPPONENTS};
use crate::train_output::{
    render_set_run_banner, render_set_summary, SetContext, TerminalObserver,
};

/// Resolves a stored `neat:@RELATIVE` spec (a frozen copy inside the run
/// directory) to a loadable `neat:PATH`; every other spec is unchanged.
fn resolve_spec(spec: &str, out: &Path) -> String {
    match spec.trim().strip_prefix("neat:@") {
        Some(relative) => format!("neat:{}", out.join(relative).display()),
        None => spec.to_owned(),
    }
}

/// Builds the opponent pool from `--strategy`-style specs (frozen
/// `neat:@...` specs are resolved against `out`), rejecting two
/// opponents with the same display name (their results would merge).
fn build_opponents(specs: &[String], out: &Path) -> anyhow::Result<Vec<Opponent>> {
    let mut opponents: Vec<Opponent> = Vec::new();
    for spec in specs {
        let strategy: Arc<dyn sim::Strategy> = resolve_spec(spec, out)
            .parse::<StrategyArg>()
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("invalid --opponent `{spec}`"))?
            .build();
        let name = strategy.name().to_owned();
        anyhow::ensure!(
            opponents.iter().all(|o| o.name != name),
            "--opponent `{spec}` duplicates `{name}`, which is already in the pool"
        );
        opponents.push(Opponent { name, strategy });
    }
    Ok(opponents)
}

/// Copies every `neat:PATH` opponent into `<out>/opponents/` and returns
/// the specs to record: `neat:@opponents/N-<stem>.json`. The run then
/// owns its opponents, so a resume cannot silently face a different
/// player if the original file is later changed, moved or deleted (and
/// two files with the same name stay distinct).
fn freeze_neat_specs(specs: &[String], out: &Path) -> anyhow::Result<Vec<String>> {
    let mut frozen = Vec::with_capacity(specs.len());
    for (index, spec) in specs.iter().enumerate() {
        let Some(path) = spec.trim().strip_prefix("neat:") else {
            frozen.push(spec.clone());
            continue;
        };
        let source = Path::new(path.trim());
        let stem = source
            .file_stem()
            .map_or_else(|| "genome".into(), |s| s.to_string_lossy());
        let relative = format!("opponents/{index}-{stem}.json");
        let target = out.join(&relative);
        std::fs::create_dir_all(out.join("opponents"))
            .and_then(|()| std::fs::copy(source, &target).map(|_| ()))
            .with_context(|| {
                format!(
                    "cannot freeze --opponent `{spec}` into {}",
                    target.display()
                )
            })?;
        frozen.push(format!("neat:@{relative}"));
    }
    Ok(frozen)
}

fn new_config(args: &TrainArgs, specs: Vec<String>) -> TrainConfig {
    TrainConfig {
        seed: args.seed,
        player_count: args.player_count,
        deck: match args.deck_variant {
            DeckVariantArg::Single => DeckChoice::Single,
            DeckVariantArg::Double => DeckChoice::Double,
        },
        duplicate_rule: match args.duplicate_rule {
            DuplicateRuleArg::FirstDealtWins => DuplicateChoice::FirstDealtWins,
            DuplicateRuleArg::LastDealtWins => DuplicateChoice::LastDealtWins,
        },
        rounds_per_match: args.rounds,
        matches_per_genome: args.matches_per_genome,
        reeval_matches: args.reeval_matches,
        generations: args.generations.unwrap_or(100),
        neat: neat::NeatConfig {
            population_size: args.population,
            target_species: args.target_species,
            weight_perturb_power: args.weight_power,
            ..neat::NeatConfig::default()
        },
        opponent_specs: specs,
        // Cannot re-score more genomes than there are.
        champion_candidates: args.champion_candidates.min(args.population),
        hall_of_fame_size: args.hall_of_fame,
        hall_of_fame_interval: args.hall_interval,
    }
}

/// Applies `--from` and checks every setting, writing nothing: returns the
/// arguments with the population size settled and the opponent specs.
fn checked_args(args: &TrainArgs) -> anyhow::Result<(TrainArgs, Vec<String>)> {
    let specs: Vec<String> = if args.opponent.is_empty() {
        DEFAULT_OPPONENTS.iter().map(|&s| s.to_owned()).collect()
    } else {
        args.opponent.clone()
    };
    let mut args = args.clone();
    if let Some(from) = &args.from {
        // The genomes are kept as they are, so the size is the source's.
        args.population = load_config(from)
            .with_context(|| {
                format!(
                    "cannot build on {}: it holds no usable run (to continue a run unchanged use --resume)",
                    from.display()
                )
            })?
            .neat
            .population_size;
    }
    if let Some(from) = &args.from {
        Trainer::check_warm_source(from, args.population)?;
    }
    for spec in &specs {
        spec.parse::<StrategyArg>()
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("invalid --opponent `{spec}`"))?;
    }
    new_config(&args, specs.clone())
        .validate()
        .map_err(|reason| anyhow::anyhow!("invalid training setup: {reason}"))?;
    Ok((args, specs))
}

/// Builds the trainer of a fresh run in `out`. Everything is validated
/// before anything is frozen or written, so a bad setup leaves nothing
/// behind, and a directory that already holds a run is never touched.
fn new_trainer(args: &TrainArgs, out: &Path) -> anyhow::Result<Trainer> {
    anyhow::ensure!(
        !out.join("checkpoint.json").exists(),
        "{} already holds a run; resume it or choose another directory",
        out.display()
    );
    let (args, specs) = checked_args(args)?;
    let frozen = freeze_neat_specs(&specs, out)?;
    let opponents = build_opponents(&frozen, out)?;
    let config = new_config(&args, frozen);
    Ok(match &args.from {
        Some(from) => Trainer::new_from(config, opponents, out, from)?,
        None => Trainer::new(config, opponents, out)?,
    })
}

fn resume_trainer(out: &Path, generations: Option<u32>) -> anyhow::Result<Trainer> {
    let config = load_config(out)?;
    let opponents = build_opponents(&config.opponent_specs, out)?;
    Ok(Trainer::resume(out, opponents, generations)?)
}

/// Starts run `index` of a resumed set that had not begun yet: the first
/// run's settings with this run's seed (and the same warm-start source).
fn next_set_trainer(
    set_dir: &Path,
    index: u32,
    set: &SetFile,
    generations: Option<u32>,
) -> anyhow::Result<Trainer> {
    let first = set_dir.join(run_dir_name(1));
    let mut config = load_config(&first)?;
    config.seed += u64::from(index - 1);
    if let Some(total) = generations {
        config.generations = total;
    }
    let out = set_dir.join(run_dir_name(index));
    // The first run froze its neat opponents inside its own directory;
    // every run owns a copy.
    let frozen = first.join("opponents");
    if frozen.is_dir() {
        std::fs::create_dir_all(out.join("opponents"))?;
        for entry in std::fs::read_dir(&frozen)? {
            let entry = entry?;
            std::fs::copy(entry.path(), out.join("opponents").join(entry.file_name()))?;
        }
    }
    let opponents = build_opponents(&config.opponent_specs, &out)?;
    Ok(match &set.from {
        Some(from) => Trainer::new_from(config, opponents, &out, Path::new(from))?,
        None => Trainer::new(config, opponents, &out)?,
    })
}

fn run_set(args: &TrainArgs, mut set: SetFile, resuming: bool) -> anyhow::Result<()> {
    let started = std::time::Instant::now();
    let mut ends: Vec<(String, RunEnd)> = Vec::new();
    let first_unfinished = u32::try_from(set.finished_secs.len()).unwrap_or(u32::MAX) + 1;
    for index in first_unfinished..=set.total_runs {
        let out = args.out.join(run_dir_name(index));
        let mut trainer = if resuming && out.join("checkpoint.json").exists() {
            resume_trainer(&out, args.generations)?
        } else if resuming {
            next_set_trainer(&args.out, index, &set, args.generations)?
        } else {
            let mut run_args = args.clone();
            run_args.seed = args.seed + u64::from(index - 1);
            new_trainer(&run_args, &out)?
        };
        // Recorded only once the run exists, so a run that cannot even be
        // created leaves no set file behind.
        set.current_run = index;
        set.write(&args.out)?;
        let seed = trainer.config().seed;
        if !args.quiet {
            println!("\n{}", render_set_run_banner(index, set.total_runs, seed));
        }
        let observer = TerminalObserver::new(out.clone(), args.quiet).with_set(SetContext {
            total_runs: set.total_runs,
            finished_secs: set.finished_secs.clone(),
        });
        let mut observer = observer;
        let end = trainer.run(&mut observer)?;
        set.finished_secs.push(end.elapsed_secs);
        set.current_run = index.min(set.total_runs);
        set.write(&args.out)?;
        ends.push((run_dir_name(index), end));
    }
    println!(
        "\n{}",
        render_set_summary(&ends, started.elapsed().as_secs_f64())
    );
    Ok(())
}

pub fn run(raw_args: impl Iterator<Item = String>) -> anyhow::Result<()> {
    let args = TrainArgs::parse_from(std::iter::once("cli train".to_owned()).chain(raw_args));
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .context("failed to configure thread pool")?;
    }

    // A set is recognised by its set file when resuming, by --runs when
    // starting; a plain run (`--runs 1`) is unchanged.
    let existing_set = if args.resume {
        SetFile::read(&args.out)?
    } else {
        None
    };
    if args.runs > 1 {
        anyhow::ensure!(
            !args.out.join("set.json").exists() && !args.out.join("checkpoint.json").exists(),
            "{} already holds a run or set; resume it or choose another directory",
            args.out.display()
        );
        // Validate before creating anything: a bad setup leaves no trace.
        checked_args(&args)?;
    }

    // Start the dashboard first: a busy port fails before any run state
    // is created, and the dashboard tolerates a directory that is still
    // empty.
    let dashboard = match args.serve {
        Some(port) => {
            let dashboard = web::Dashboard::start(args.out.clone(), port).with_context(|| {
                format!("cannot start the dashboard on port {port} (is it in use? pick another with --serve PORT)")
            })?;
            println!("dashboard: {}", dashboard.url());
            Some(dashboard)
        }
        None => None,
    };

    if let Some(set) = existing_set {
        run_set(&args, set, true)?;
    } else if args.runs > 1 {
        let set = SetFile {
            total_runs: args.runs,
            current_run: 1,
            finished_secs: Vec::new(),
            from: args.from.as_ref().map(|p| p.display().to_string()),
        };
        run_set(&args, set, false)?;
    } else {
        let mut trainer = if args.resume {
            resume_trainer(&args.out, args.generations)?
        } else {
            new_trainer(&args, &args.out)?
        };
        let mut observer = TerminalObserver::new(args.out.clone(), args.quiet);
        trainer.run(&mut observer)?;
    }
    if let Some(dashboard) = dashboard {
        // The dashboard lives in this process: exiting now would leave the
        // browser without the last generations and the "finished" state.
        println!(
            "run finished; the dashboard stays up at {} (Ctrl-C to exit)",
            dashboard.url()
        );
        dashboard.wait();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_pool_builds_with_distinct_names() {
        let specs: Vec<String> = DEFAULT_OPPONENTS.iter().map(|&s| s.to_owned()).collect();
        let pool = build_opponents(&specs, Path::new(".")).unwrap();
        let names: Vec<&str> = pool.iter().map(|o| o.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "LowestLegal",
                "EndgameDenial",
                "Adaptive(reading,tempo,bully)"
            ]
        );
    }

    #[test]
    fn a_bad_spec_names_the_offending_option() {
        let error = build_opponents(&["nonsense".to_owned()], Path::new("."))
            .err()
            .unwrap();
        assert!(
            format!("{error:#}").contains("--opponent `nonsense`"),
            "{error:#}"
        );
    }

    #[test]
    fn a_duplicate_opponent_is_rejected() {
        let specs = vec!["lowest-legal".to_owned(), "lowest-legal".to_owned()];
        let error = build_opponents(&specs, Path::new("."))
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("duplicates `LowestLegal`"), "{error}");
    }
}
