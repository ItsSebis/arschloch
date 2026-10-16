//! `cli solver-gap`: where do other strategies disagree with the exact
//! solver, and what does it cost (docs/SOLVER.md)?

use std::fmt::Write as _;
use std::sync::Arc;

use anyhow::Context;
use engine::{DeckVariant, DuplicateRule};
use sim::solver::gap::{run_gap, GapConfig, GapRow, GapStats};
use sim::{SolverConfig, Strategy};

use crate::args::StrategyArg;

const USAGE: &str =
    "usage: cli solver-gap --strategy SPEC ... [--player-count N] [--deck single|double] \
[--games N] [--seed N] [--max-unseen N] [--budget N]   (see docs/SOLVER.md)";

pub struct Options {
    pub strategies: Vec<Arc<dyn Strategy>>,
    pub config: GapConfig,
}

fn number<T: std::str::FromStr>(flag: &str, value: Option<String>) -> Result<T, String> {
    value
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| format!("{flag} needs a number"))
}

pub fn parse_args(args: Vec<String>) -> Result<Options, String> {
    let mut strategies = Vec::new();
    let mut players: Option<u8> = None;
    let mut config = GapConfig {
        players: 0,
        deck: DeckVariant::Single,
        rule: DuplicateRule::FirstDealtWins,
        games: 1000,
        seed: 0,
        solver: SolverConfig {
            max_unseen: 6,
            node_budget: 2_000_000,
        },
    };
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--strategy" => {
                let spec = args.next().ok_or("--strategy needs a spec")?;
                strategies.push(spec.parse::<StrategyArg>()?.build());
            }
            "--player-count" => players = Some(number(&flag, args.next())?),
            "--games" => config.games = number(&flag, args.next())?,
            "--seed" => config.seed = number(&flag, args.next())?,
            "--max-unseen" => config.solver.max_unseen = number(&flag, args.next())?,
            "--budget" => config.solver.node_budget = number(&flag, args.next())?,
            "--deck" => {
                config.deck = match args.next().as_deref() {
                    Some("single") => DeckVariant::Single,
                    Some("double") => DeckVariant::Double,
                    _ => return Err("--deck must be single or double".to_owned()),
                };
            }
            other => return Err(format!("unexpected argument `{other}`")),
        }
    }
    let seats = u8::try_from(strategies.len()).unwrap_or(0);
    if !(3..=6).contains(&seats) || players.is_some_and(|p| p != seats) {
        return Err(format!(
            "give one --strategy per seat, 3 to 6 of them (and --player-count, if given, must match); got {}",
            strategies.len()
        ));
    }
    config.players = seats;
    Ok(Options { strategies, config })
}

#[allow(clippy::cast_precision_loss)] // counts are far below 2^52
pub fn render(rows: &[GapRow], stats: &GapStats, config: &GapConfig) -> String {
    let ratio = |part: u64, whole: u64| {
        if whole == 0 {
            0.0
        } else {
            part as f64 / whole as f64
        }
    };
    let mut out = format!(
        "{} players, {:?} deck, {} games, seed {}, solver T={} budget {}\n",
        config.players,
        config.deck,
        config.games,
        config.seed,
        config.solver.max_unseen,
        config.solver.node_budget
    );
    let _ = writeln!(
        out,
        "decisions with a choice: {}; solved: {} ({:.1}%); over budget: {}; nodes per solved decision: mean {:.0}, max {}\n",
        stats.decisions,
        stats.solvable,
        100.0 * ratio(stats.solvable, stats.decisions),
        stats.budget_exceeded,
        ratio(stats.nodes_total, stats.solvable),
        stats.nodes_max
    );
    let _ = writeln!(
        out,
        "{:<34} {:>7} {:>8} {:>7} {:>15}",
        "strategy", "solved", "differ", "share", "places lost each"
    );
    for row in rows {
        let cost_each = if row.decisions == 0 {
            0.0
        } else {
            row.cost_sum / row.decisions as f64
        };
        let _ = writeln!(
            out,
            "{:<34} {:>7} {:>8} {:>6.1}% {:>15.4}",
            row.strategy,
            row.decisions,
            row.differing,
            100.0 * ratio(row.differing, row.decisions),
            cost_each
        );
    }
    for row in rows.iter().filter(|r| !r.worst.is_empty()) {
        let _ = writeln!(out, "\ncostliest decisions of {}:", row.strategy);
        for d in &row.worst {
            let _ = writeln!(out, "  {}", d.situation);
        }
    }
    out
}

pub fn run(args: impl Iterator<Item = String>) -> anyhow::Result<()> {
    let args: Vec<String> = args.collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return Ok(());
    }
    let options = parse_args(args)
        .map_err(|e| anyhow::anyhow!("{e}\n{USAGE}"))
        .context("solver-gap")?;
    let (rows, stats) = run_gap(&options.strategies, &options.config);
    print!("{}", render(&rows, &stats, &options.config));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_owned).collect()
    }

    #[test]
    fn options_are_read_and_the_player_count_follows_the_strategies() {
        let options = parse_args(words(
            "--strategy lowest-legal --strategy adaptive --strategy greedy-highest --games 50 --seed 9 --max-unseen 5 --budget 1000 --deck double",
        ))
        .unwrap();
        assert_eq!(options.strategies.len(), 3);
        assert_eq!(options.config.players, 3);
        assert_eq!(options.config.games, 50);
        assert_eq!(options.config.seed, 9);
        assert_eq!(options.config.solver.max_unseen, 5);
        assert_eq!(options.config.solver.node_budget, 1000);
        assert_eq!(options.config.deck, engine::DeckVariant::Double);
    }

    #[test]
    fn mistakes_are_reported() {
        assert!(
            parse_args(words("--strategy lowest-legal")).is_err(),
            "one seat is not a table"
        );
        assert!(parse_args(words(
            "--player-count 4 --strategy lowest-legal --strategy lowest-legal --strategy lowest-legal"
        ))
        .is_err());
        assert!(parse_args(words("--games x --strategy a")).is_err());
        assert!(parse_args(words("--bogus")).is_err());
        assert!(parse_args(words(
            "--strategy lowest-legal --strategy lowest-legal --strategy nope"
        ))
        .is_err());
    }

    #[test]
    fn the_report_names_every_strategy_and_the_costliest_decision() {
        let options = parse_args(words(
            "--strategy greedy-highest --strategy greedy-highest --strategy lowest-legal --games 40 --max-unseen 4 --budget 100000",
        ))
        .unwrap();
        let (rows, stats) = sim::solver::gap::run_gap(&options.strategies, &options.config);
        let text = render(&rows, &stats, &options.config);
        assert!(text.contains("GreedyHighest"), "{text}");
        assert!(text.contains("LowestLegal"), "{text}");
        assert!(text.contains("solved"), "{text}");
    }
}
