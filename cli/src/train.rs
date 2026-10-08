//! `cli train`: wires the arguments to `sim::training::Trainer`.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use sim::training::{load_config, DeckChoice, DuplicateChoice, Opponent, TrainConfig, Trainer};

use crate::args::{DeckVariantArg, DuplicateRuleArg, StrategyArg};
use crate::train_args::{TrainArgs, DEFAULT_OPPONENTS};
use crate::train_output::TerminalObserver;

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

pub fn run(raw_args: impl Iterator<Item = String>) -> anyhow::Result<()> {
    let mut args = TrainArgs::parse_from(std::iter::once("cli train".to_owned()).chain(raw_args));
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .context("failed to configure thread pool")?;
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

    let mut trainer = if args.resume {
        let config = load_config(&args.out)?;
        let opponents = build_opponents(&config.opponent_specs, &args.out)?;
        Trainer::resume(&args.out, opponents, args.generations)?
    } else {
        let specs: Vec<String> = if args.opponent.is_empty() {
            DEFAULT_OPPONENTS.iter().map(|&s| s.to_owned()).collect()
        } else {
            args.opponent.clone()
        };
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
        // Fail on a bad spec before anything is written, and never touch a
        // directory that already holds a run.
        anyhow::ensure!(
            !args.out.join("checkpoint.json").exists(),
            "{} already holds a run; resume it or choose another directory",
            args.out.display()
        );
        for spec in &specs {
            spec.parse::<StrategyArg>()
                .map_err(anyhow::Error::msg)
                .with_context(|| format!("invalid --opponent `{spec}`"))?;
        }
        // Validate the whole setup before freezing anything into the run
        // directory: an invalid run must leave nothing behind.
        new_config(&args, specs.clone())
            .validate()
            .map_err(|reason| anyhow::anyhow!("invalid training setup: {reason}"))?;
        let frozen = freeze_neat_specs(&specs, &args.out)?;
        let opponents = build_opponents(&frozen, &args.out)?;
        let config = new_config(&args, frozen);
        match &args.from {
            Some(from) => Trainer::new_from(config, opponents, &args.out, from)?,
            None => Trainer::new(config, opponents, &args.out)?,
        }
    };

    let mut observer = TerminalObserver::new(args.out.clone(), args.quiet);
    trainer.run(&mut observer)?;
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
