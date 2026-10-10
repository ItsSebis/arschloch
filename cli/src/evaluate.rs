//! `cli evaluate`: scores evolved genomes against opponents, including
//! opponents they never trained against. This is the baseline-comparison
//! protocol of the design spec as a command: the same seeds and tables
//! for every genome, so results are directly comparable.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use rayon::prelude::*;
use serde::Serialize;
use sim::training::{evaluate, match_seed, Opponents, Score, TableSpec};

use crate::args::{DeckVariantArg, DuplicateRuleArg, StrategyArg};

/// Opponents when none are given: the strongest hand-written strategies
/// (the training pool's defaults), stronger variants a default training run
/// never sees, and three weak strategies as a sanity floor.
pub const DEFAULT_BATTERY: [&str; 8] = [
    "lowest-legal",
    "endgame-denial",
    "adaptive:reading,tempo,bully",
    // Stronger strategies a default training run never sees: variants of
    // the best hand-written one with modifiers outside the training pool.
    "adaptive:counting,reading",
    "adaptive:reading,deception=0.2,tempo,bully",
    // Weak ones: a sanity floor, not evidence of generalization.
    "hold-back-pairs",
    "greedy-highest",
    "random-legal",
];

/// Score evolved genomes against opponents.
#[derive(Parser, Debug)]
#[command(
    name = "cli evaluate",
    about = "Score evolved genomes against opponents, including ones they never trained against",
    long_about = "Plays each genome at tables made only of one opponent (and at tables mixing all \
the opponents) and reports its mean finishing-role score, +1 (always President) to -1 (always \
last), with its standard error. Every genome plays exactly the same deals, seats and opponents, \
so rows are directly comparable. The matches use a seed stream training never uses."
)]
pub struct EvaluateArgs {
    /// A genome file written by `cli train` (repeat to compare several).
    #[arg(long, required = true, value_name = "PATH")]
    pub genome: Vec<PathBuf>,

    /// An opponent, in `--strategy` syntax (repeat for several; `neat:PATH`
    /// pits a genome against another genome). Default: lowest-legal,
    /// endgame-denial, adaptive:reading,tempo,bully, two stronger adaptive
    /// variants, hold-back-pairs, greedy-highest, random-legal.
    #[arg(long, value_name = "SPEC")]
    pub opponent: Vec<String>,

    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u8).range(3..=6))]
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

    /// Rounds per match.
    #[arg(long, default_value_t = 8, value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(1..))]
    pub rounds: usize,

    /// Matches per opponent.
    #[arg(long, default_value_t = 400, value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(1..))]
    pub matches: usize,

    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Rayon thread-pool size. 0 lets rayon pick its own default.
    #[arg(long, default_value_t = 0)]
    pub threads: usize,

    /// Also write the results as JSON.
    #[arg(long, value_name = "PATH")]
    pub json: Option<PathBuf>,

    /// Print a one-line explanation of each statistic in the tables.
    #[arg(long)]
    pub explain: bool,
}

/// One cell of the result table.
#[derive(Debug, Clone, Serialize)]
pub struct Cell {
    pub opponent: String,
    pub mean: f64,
    pub std_error: f64,
    pub matches: usize,
    pub placements: Vec<u64>,
    /// Average finishing place, 1 = best. The role score is linear in the
    /// place, so this is exactly the mean score restated.
    pub avg_rank: f64,
    /// Standard error of `avg_rank` across matches.
    pub avg_rank_std_error: f64,
}

impl Cell {
    fn new(opponent: &str, score: Score) -> Self {
        #[allow(clippy::cast_precision_loss)] // table sizes are 3-6
        let half_span = (score.placements.len().saturating_sub(1)) as f64 / 2.0;
        Self {
            opponent: opponent.to_owned(),
            mean: score.mean,
            std_error: score.std_error,
            matches: score.matches,
            avg_rank: 1.0 + (1.0 - score.mean) * half_span,
            avg_rank_std_error: score.std_error * half_span,
            placements: score.placements,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GenomeResult {
    pub genome: String,
    /// One cell per opponent, then a last `mixed` cell.
    pub cells: Vec<Cell>,
}

/// The label of the all-opponents-mixed row.
pub const MIXED: &str = "mixed (all opponents)";

/// A column heading for a genome: its file stem, made unique if needed.
fn column_labels(paths: &[PathBuf]) -> Vec<String> {
    let stems: Vec<String> = paths
        .iter()
        .map(|p| {
            p.file_stem()
                .map_or_else(|| "genome".into(), |s| s.to_string_lossy().into_owned())
        })
        .collect();
    stems
        .iter()
        .enumerate()
        .map(|(i, stem)| {
            if stems.iter().filter(|s| *s == stem).count() > 1 {
                format!("{i}:{stem}")
            } else {
                stem.clone()
            }
        })
        .collect()
}

/// Renders results as a table: one row per opponent plus the mixed row,
/// one column per genome. Pure, so it is directly testable.
#[must_use]
pub fn render_table(labels: &[String], results: &[GenomeResult]) -> String {
    render_cells(labels, results, |cell| {
        format!("{:+.3} ±{:.3}", cell.mean, cell.std_error)
    })
}

/// The same table with the average place (1 = best) in each cell.
#[must_use]
pub fn render_rank_table(labels: &[String], results: &[GenomeResult]) -> String {
    render_cells(labels, results, |cell| {
        format!("{:.2} ±{:.2}", cell.avg_rank, cell.avg_rank_std_error)
    })
}

fn render_cells(
    labels: &[String],
    results: &[GenomeResult],
    format_cell: impl Fn(&Cell) -> String,
) -> String {
    let rows: Vec<&str> = results[0]
        .cells
        .iter()
        .map(|c| c.opponent.as_str())
        .collect();
    let row_width = rows
        .iter()
        .map(|r| r.len())
        .max()
        .unwrap_or(8)
        .max("opponent".len());
    let column_width = labels.iter().map(String::len).max().unwrap_or(8).max(14);
    let mut text = format!("{:<row_width$}", "opponent");
    for label in labels {
        let _ = write!(text, "  {label:>column_width$}");
    }
    text.push('\n');
    for (row, name) in rows.iter().enumerate() {
        let _ = write!(text, "{name:<row_width$}");
        for result in results {
            let cell = &result.cells[row];
            let _ = write!(text, "  {:>column_width$}", format_cell(cell));
        }
        text.push('\n');
    }
    text
}

fn build_opponents(specs: &[String]) -> anyhow::Result<Vec<(String, Arc<dyn sim::Strategy>)>> {
    let mut built: Vec<(String, Arc<dyn sim::Strategy>)> = Vec::new();
    for spec in specs {
        let strategy: Arc<dyn sim::Strategy> = spec
            .parse::<StrategyArg>()
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("invalid --opponent `{spec}`"))?
            .build();
        let name = sim::Strategy::name(&*strategy).to_owned();
        anyhow::ensure!(
            built.iter().all(|(n, _)| *n != name),
            "--opponent `{spec}` duplicates `{name}`, which is already listed"
        );
        built.push((name, strategy));
    }
    Ok(built)
}

pub fn run(raw_args: impl Iterator<Item = String>) -> anyhow::Result<()> {
    let args = EvaluateArgs::parse_from(std::iter::once("cli evaluate".to_owned()).chain(raw_args));
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .context("failed to configure thread pool")?;
    }
    let specs: Vec<String> = if args.opponent.is_empty() {
        DEFAULT_BATTERY.iter().map(|&s| s.to_owned()).collect()
    } else {
        args.opponent.clone()
    };
    let opponents = build_opponents(&specs)?;
    let candidates: Vec<(String, Arc<dyn sim::Strategy>)> = args
        .genome
        .iter()
        .map(|path| {
            let strategy = sim::NeatStrategy::from_file(path)
                .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
            Ok((
                path.display().to_string(),
                Arc::new(strategy) as Arc<dyn sim::Strategy>,
            ))
        })
        .collect::<anyhow::Result<_>>()?;

    let table = TableSpec {
        player_count: args.player_count,
        deck_variant: args.deck_variant.into(),
        duplicate_rule: args.duplicate_rule.into(),
        rounds: args.rounds,
        pass_rule: args.pass_rule,
        exchange_rule: args.exchange_rule,
    };
    // A seed stream that training (generation streams, fixed and held-out
    // sets) never uses.
    let seeds: Vec<u64> = (0..args.matches as u64)
        .map(|i| match_seed(args.seed, u64::MAX - 3, i))
        .collect();
    let pool: Vec<Arc<dyn sim::Strategy>> = opponents.iter().map(|(_, s)| s.clone()).collect();

    let results: Vec<GenomeResult> = candidates
        .iter()
        .map(|(path, candidate)| {
            let mut cells: Vec<Cell> = opponents
                .par_iter()
                .map(|(name, strategy)| {
                    Cell::new(
                        name,
                        evaluate(candidate, &table, Opponents::Only(strategy), &seeds),
                    )
                })
                .collect();
            cells.push(Cell::new(
                MIXED,
                evaluate(candidate, &table, Opponents::Mixed(&pool), &seeds),
            ));
            GenomeResult {
                genome: path.clone(),
                cells,
            }
        })
        .collect();

    println!(
        "{} players, {:?} deck, {:?}, pass rule {}, exchange rule {}, {} matches x {} rounds per cell, seed {}",
        args.player_count,
        args.deck_variant,
        args.duplicate_rule,
        args.pass_rule,
        args.exchange_rule,
        args.matches,
        args.rounds,
        args.seed
    );
    println!("score: +1 always President .. -1 always last; 0 is even\n");
    let labels = column_labels(&args.genome);
    print!("{}", render_table(&labels, &results));
    println!(
        "\naverage place: 1 = best .. {} = last, ± standard error\n",
        args.player_count
    );
    print!("{}", render_rank_table(&labels, &results));
    if args.explain {
        println!(
            "\n{}",
            sim::stats_catalog::render_explain(&["mean_role_score", "avg_rank"]).trim_end()
        );
    }
    if let Some(path) = &args.json {
        let text = serde_json::to_string_pretty(&serde_json::json!({
            "player_count": args.player_count,
            "deck_variant": format!("{:?}", args.deck_variant),
            "duplicate_rule": format!("{:?}", args.duplicate_rule),
            "pass_rule": args.pass_rule.to_string(),
            "exchange_rule": args.exchange_rule.to_string(),
            "rounds": args.rounds,
            "matches": args.matches,
            "seed": args.seed,
            "results": results,
            "statistics_catalog": sim::stats_catalog::catalog(),
        }))?;
        std::fs::write(path, text).with_context(|| format!("cannot write {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(opponent: &str, mean: f64) -> Cell {
        Cell {
            opponent: opponent.into(),
            mean,
            std_error: 0.021,
            matches: 100,
            placements: vec![1, 2, 3, 4],
            avg_rank: 2.5,
            avg_rank_std_error: 0.03,
        }
    }

    fn result(genome: &str, means: [f64; 2]) -> GenomeResult {
        GenomeResult {
            genome: genome.into(),
            cells: vec![cell("LowestLegal", means[0]), cell(MIXED, means[1])],
        }
    }

    #[test]
    fn the_table_has_a_row_per_opponent_plus_mixed_and_a_column_per_genome() {
        let labels = vec!["a".to_owned(), "b".to_owned()];
        let text = render_table(
            &labels,
            &[result("a.json", [0.5, 0.4]), result("b.json", [-0.25, 0.1])],
        );
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3, "{text}");
        assert!(
            lines[0].starts_with("opponent") && lines[0].contains('a') && lines[0].contains('b')
        );
        assert!(
            lines[1].contains("LowestLegal")
                && lines[1].contains("+0.500 ±0.021")
                && lines[1].contains("-0.250 ±0.021")
        );
        assert!(lines[2].starts_with("mixed (all opponents)") && lines[2].contains("+0.400"));
    }

    #[test]
    fn the_rank_table_shows_average_places_and_cells_restate_the_score() {
        let labels = vec!["a".to_owned()];
        let text = render_rank_table(&labels, &[result("a.json", [0.5, 0.4])]);
        assert!(text.contains("2.50 ±0.03"), "{text}");
        // four places: score +1 is place 1, -1 is place 4, 0 is 2.5.
        let score = |mean: f64| Score {
            mean,
            std_error: 0.1,
            matches: 10,
            placements: vec![1, 1, 1, 1],
        };
        let best = Cell::new("x", score(1.0));
        let even = Cell::new("x", score(0.0));
        let worst = Cell::new("x", score(-1.0));
        assert!((best.avg_rank - 1.0).abs() < 1e-12);
        assert!((even.avg_rank - 2.5).abs() < 1e-12);
        assert!((worst.avg_rank - 4.0).abs() < 1e-12);
        assert!((even.avg_rank_std_error - 0.15).abs() < 1e-12);
    }

    #[test]
    fn same_named_genome_files_get_distinct_columns() {
        let labels = column_labels(&[
            PathBuf::from("x/best.json"),
            PathBuf::from("y/best.json"),
            PathBuf::from("z/other.json"),
        ]);
        assert_eq!(labels, vec!["0:best", "1:best", "other"]);
    }

    #[test]
    fn the_default_battery_builds_and_bad_or_duplicate_opponents_are_rejected() {
        let specs: Vec<String> = DEFAULT_BATTERY.iter().map(|&s| s.to_owned()).collect();
        assert_eq!(build_opponents(&specs).unwrap().len(), 8);
        let error = build_opponents(&["nonsense".to_owned()]).err().unwrap();
        assert!(format!("{error:#}").contains("--opponent `nonsense`"));
        let twice = build_opponents(&["lowest-legal".to_owned(), "lowest-legal".to_owned()])
            .err()
            .unwrap();
        assert!(twice.to_string().contains("duplicates"), "{twice}");
    }

    #[test]
    fn a_genome_is_required_and_defaults_are_sensible() {
        assert!(EvaluateArgs::try_parse_from(["cli evaluate"]).is_err());
        let args = EvaluateArgs::try_parse_from(["cli evaluate", "--genome", "g.json"]).unwrap();
        assert_eq!((args.player_count, args.matches, args.rounds), (4, 400, 8));
        assert!(args.opponent.is_empty() && args.json.is_none());
    }
}
