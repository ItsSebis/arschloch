//! Arguments of `cli train`.

use std::path::PathBuf;

use clap::Parser;

use crate::args::{DeckVariantArg, DuplicateRuleArg};

/// The opponents a run trains against unless `--opponent` is given: the
/// strongest hand-written strategies as of the pre-NEAT baseline
/// (`CardCounter` is omitted: it plays identically to `LowestLegal`).
pub const DEFAULT_OPPONENTS: [&str; 3] = [
    "lowest-legal",
    "endgame-denial",
    "adaptive:reading,tempo,bully",
];

/// A strictly positive, finite number (clap parses `NaN`/`inf` as floats).
fn positive_finite(text: &str) -> Result<f64, String> {
    let value: f64 = text
        .parse()
        .map_err(|e| format!("`{text}` is not a number: {e}"))?;
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(format!("`{text}` must be a positive, finite number"))
    }
}

/// Evolve a NEAT player, generation by generation.
#[derive(Parser, Debug, Clone)]
#[command(
    name = "cli train",
    about = "Evolve a NEAT player against a pool of opponents, saving every champion",
    long_about = "Evolve a NEAT player against a pool of opponents. Each generation every \
genome plays the same matches against opponents drawn from the pool; the next generation is \
bred from the best, and the generation's champion is re-evaluated on fresh matches. Everything \
is written to --out: events.jsonl (one JSON event per generation), checkpoint.json (resume \
point), best.json and gen-NNNN.json (champions, playable with `cli --strategy neat:PATH`)."
)]
pub struct TrainArgs {
    /// Run directory (created if needed; a new run refuses a directory
    /// that already holds one).
    #[arg(long)]
    pub out: PathBuf,

    /// Continue the run in --out from its last checkpoint. Only
    /// --generations, --threads and --quiet may accompany it: the run's
    /// other settings are read from its checkpoint.
    #[arg(
        long,
        conflicts_with_all = [
            "player_count", "deck_variant", "duplicate_rule", "rounds", "population",
            "matches_per_genome", "reeval_matches", "seed", "opponent", "target_species",
            "champion_candidates", "hall_of_fame", "hall_interval", "weight_power", "runs"
        ]
    )]
    pub resume: bool,

    /// Train this many independent runs one after another as a *set*: with
    /// more than one, --out becomes the set directory holding run-01,
    /// run-02, ... (run k uses seed SEED+k-1) and set.json, and the output
    /// shows the time left for the whole set. A killed set continues with
    /// --resume.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=1000))]
    pub runs: u32,

    /// Start from the final population of the earlier run in this
    /// directory instead of a random one (the earlier run is only read).
    /// The settings of the new run are the ones given here; its
    /// population size is the earlier run's, so --population cannot be
    /// given. To continue the earlier run itself unchanged, use --resume.
    #[arg(long, value_name = "RUN_DIR", conflicts_with_all = ["resume", "population"])]
    pub from: Option<PathBuf>,

    /// Table size (3-6 seats).
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u8).range(3..=6))]
    pub player_count: u8,

    #[arg(long, value_enum, default_value_t = DeckVariantArg::Single)]
    pub deck_variant: DeckVariantArg,

    #[arg(long, value_enum, default_value_t = DuplicateRuleArg::FirstDealtWins)]
    pub duplicate_rule: DuplicateRuleArg,

    /// Rounds per evaluation match (role carry-over between rounds).
    #[arg(long, default_value_t = 8, value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(1..))]
    pub rounds: usize,

    /// Genomes per generation.
    #[arg(long, default_value_t = 150, value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(2..))]
    pub population: usize,

    /// Total generations to run (default 100). With --resume, the new
    /// total to run up to.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub generations: Option<u32>,

    /// Matches each genome plays per generation (all genomes play the
    /// same ones).
    #[arg(long, default_value_t = 100, value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(1..))]
    pub matches_per_genome: usize,

    /// Matches in each fresh re-evaluation of a generation's champion.
    #[arg(long, default_value_t = 200, value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(1..))]
    pub reeval_matches: usize,

    /// Seeds everything; the same seed reproduces the same run.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// An opponent in the training pool, in `--strategy` syntax (for
    /// example `adaptive:reading,tempo,bully` or `neat:PATH`); repeat for
    /// several. Default: lowest-legal, endgame-denial and
    /// adaptive:reading,tempo,bully.
    #[arg(long, value_name = "SPEC")]
    pub opponent: Vec<String>,

    /// How many species the speciation threshold steers toward.
    #[arg(long, default_value_t = 8, value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(1..))]
    pub target_species: usize,

    /// Re-evaluate this many of each generation's best genomes (by training
    /// fitness) on the fixed matches and take the best of them as the
    /// champion. Training fitness is noisy, so its best genome is often not
    /// the strongest; 1 trusts it. Default 5 (see docs/baselines/neat-v1).
    #[arg(long, default_value_t = 5, value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(1..))]
    pub champion_candidates: usize,

    /// Keep this many frozen past champions as extra training opponents
    /// (0 = none), so the population is not tuned only to the fixed pool.
    #[arg(long, default_value_t = 0)]
    pub hall_of_fame: usize,

    /// A champion joins the hall of fame every this many generations.
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u32).range(1..))]
    pub hall_interval: u32,

    /// Size of a weight perturbation (uniform in plus/minus this). Default
    /// 0.2 (see docs/baselines/neat-v1).
    #[arg(long, default_value_t = 0.2, value_parser = positive_finite)]
    pub weight_power: f64,

    /// Rayon thread-pool size. 0 lets rayon pick its own default.
    #[arg(long, default_value_t = 0)]
    pub threads: usize,

    /// Serve the live dashboard at http://127.0.0.1:PORT/ (default port
    /// 8080) while training, and keep serving after the run ends until
    /// Ctrl-C (the dashboard lives in this process). It only reads the run
    /// directory, so a slow or closed browser cannot affect the run.
    #[arg(long, num_args = 0..=1, default_missing_value = "8080", value_name = "PORT")]
    pub serve: Option<u16>,

    /// Print only the final summary (the event log is still written).
    #[arg(long)]
    pub quiet: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<TrainArgs, clap::Error> {
        TrainArgs::try_parse_from(std::iter::once("cli train").chain(args.iter().copied()))
    }

    #[test]
    fn only_out_is_required_and_the_defaults_are_sensible() {
        let args = parse(&["--out", "runs/a"]).unwrap();
        assert_eq!(
            (args.player_count, args.population, args.rounds),
            (4, 150, 8)
        );
        assert_eq!(args.generations, None);
        assert_eq!(args.opponent, [] as [String; 0]);
        assert!(!args.resume && !args.quiet);
    }

    #[test]
    fn from_conflicts_with_resume_and_population() {
        let ok = parse(&["--out", "d", "--from", "old"]).unwrap();
        assert_eq!(ok.from.as_deref(), Some(std::path::Path::new("old")));
        assert!(parse(&["--out", "d", "--from", "old", "--resume"]).is_err());
        assert!(parse(&["--out", "d", "--from", "old", "--population", "20"]).is_err());
        assert_eq!(parse(&["--out", "d"]).unwrap().from, None);
    }

    #[test]
    fn runs_defaults_to_one_and_conflicts_with_resume() {
        assert_eq!(parse(&["--out", "d"]).unwrap().runs, 1);
        assert_eq!(parse(&["--out", "d", "--runs", "5"]).unwrap().runs, 5);
        assert!(parse(&["--out", "d", "--runs", "0"]).is_err());
        assert!(parse(&["--out", "d", "--resume", "--runs", "2"]).is_err());
    }

    #[test]
    fn serve_takes_an_optional_port() {
        assert_eq!(parse(&["--out", "d"]).unwrap().serve, None);
        assert_eq!(parse(&["--out", "d", "--serve"]).unwrap().serve, Some(8080));
        assert_eq!(
            parse(&["--out", "d", "--serve", "9100"]).unwrap().serve,
            Some(9100)
        );
        assert_eq!(
            parse(&["--out", "d", "--resume", "--serve"]).unwrap().serve,
            Some(8080)
        );
        assert!(parse(&["--out", "d", "--serve", "notaport"]).is_err());
    }

    #[test]
    fn learning_options_default_to_the_measured_settings_and_conflict_with_resume() {
        let args = parse(&["--out", "d"]).unwrap();
        assert_eq!(
            (
                args.champion_candidates,
                args.hall_of_fame,
                args.hall_interval
            ),
            (5, 0, 5)
        );
        assert!((args.weight_power - 0.2).abs() < f64::EPSILON);
        let old_style = parse(&[
            "--out",
            "d",
            "--champion-candidates",
            "1",
            "--hall-of-fame",
            "3",
            "--weight-power",
            "0.5",
        ])
        .unwrap();
        assert_eq!(
            (old_style.champion_candidates, old_style.hall_of_fame),
            (1, 3)
        );
        assert!((old_style.weight_power - 0.5).abs() < f64::EPSILON);
        assert!(parse(&["--out", "d", "--champion-candidates", "0"]).is_err());
        assert!(parse(&["--out", "d", "--hall-interval", "0"]).is_err());
        for conflicting in [
            ["--hall-of-fame", "2"],
            ["--champion-candidates", "3"],
            ["--weight-power", "0.1"],
            ["--hall-interval", "7"],
        ] {
            let mut args = vec!["--out", "d", "--resume"];
            args.extend(conflicting);
            assert!(parse(&args).is_err(), "{conflicting:?}");
        }
    }

    #[test]
    fn the_weight_power_must_be_a_positive_finite_number() {
        for bad in ["NaN", "inf", "-inf", "0", "-0.5", "abc"] {
            assert!(
                parse(&["--out", "d", "--weight-power", bad]).is_err(),
                "{bad}"
            );
        }
        assert!(parse(&["--out", "d", "--weight-power", "0.05"]).is_ok());
    }

    #[test]
    fn out_is_required() {
        assert!(parse(&[]).is_err());
    }

    #[test]
    fn resume_accepts_only_generations_threads_and_quiet() {
        assert!(parse(&[
            "--out",
            "d",
            "--resume",
            "--generations",
            "50",
            "--threads",
            "2",
            "--quiet"
        ])
        .is_ok());
        for conflicting in [
            ["--population", "10"],
            ["--seed", "3"],
            ["--opponent", "lowest-legal"],
            ["--player-count", "5"],
        ] {
            let mut args = vec!["--out", "d", "--resume"];
            args.extend(conflicting);
            assert!(parse(&args).is_err(), "{conflicting:?}");
        }
    }

    #[test]
    fn out_of_range_values_are_rejected() {
        assert!(parse(&["--out", "d", "--player-count", "2"]).is_err());
        assert!(parse(&["--out", "d", "--population", "1"]).is_err());
        assert!(parse(&["--out", "d", "--generations", "0"]).is_err());
    }

    #[test]
    fn opponents_can_be_repeated() {
        let args = parse(&[
            "--out",
            "d",
            "--opponent",
            "lowest-legal",
            "--opponent",
            "adaptive:counting",
        ])
        .unwrap();
        assert_eq!(args.opponent, vec!["lowest-legal", "adaptive:counting"]);
    }
}
