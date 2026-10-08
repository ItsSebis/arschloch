//! `cli train`: wires the arguments to `sim::training::Trainer`.

use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use sim::training::{load_config, DeckChoice, DuplicateChoice, Opponent, TrainConfig, Trainer};

use crate::args::{DeckVariantArg, DuplicateRuleArg, StrategyArg};
use crate::train_args::{TrainArgs, DEFAULT_OPPONENTS};
use crate::train_output::TerminalObserver;

/// Builds the opponent pool from `--strategy`-style specs, rejecting two
/// opponents with the same display name (their results would merge).
fn build_opponents(specs: &[String]) -> anyhow::Result<Vec<Opponent>> {
    let mut opponents: Vec<Opponent> = Vec::new();
    for spec in specs {
        let strategy: Arc<dyn sim::Strategy> = spec
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
            ..neat::NeatConfig::default()
        },
        opponent_specs: specs,
    }
}

pub fn run(raw_args: impl Iterator<Item = String>) -> anyhow::Result<()> {
    let args = TrainArgs::parse_from(std::iter::once("cli train".to_owned()).chain(raw_args));
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .context("failed to configure thread pool")?;
    }

    let mut trainer = if args.resume {
        let config = load_config(&args.out)?;
        let opponents = build_opponents(&config.opponent_specs)?;
        Trainer::resume(&args.out, opponents, args.generations)?
    } else {
        let specs: Vec<String> = if args.opponent.is_empty() {
            DEFAULT_OPPONENTS.iter().map(|&s| s.to_owned()).collect()
        } else {
            args.opponent.clone()
        };
        let opponents = build_opponents(&specs)?;
        Trainer::new(new_config(&args, specs), opponents, &args.out)?
    };

    let mut observer = TerminalObserver::new(args.out.clone(), args.quiet);
    trainer.run(&mut observer)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_pool_builds_with_distinct_names() {
        let specs: Vec<String> = DEFAULT_OPPONENTS.iter().map(|&s| s.to_owned()).collect();
        let pool = build_opponents(&specs).unwrap();
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
        let error = build_opponents(&["nonsense".to_owned()]).err().unwrap();
        assert!(
            format!("{error:#}").contains("--opponent `nonsense`"),
            "{error:#}"
        );
    }

    #[test]
    fn a_duplicate_opponent_is_rejected() {
        let specs = vec!["lowest-legal".to_owned(), "lowest-legal".to_owned()];
        let error = build_opponents(&specs).err().unwrap().to_string();
        assert!(error.contains("duplicates `LowestLegal`"), "{error}");
    }
}
