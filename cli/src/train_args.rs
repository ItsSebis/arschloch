//! Arguments of `cli train`.

use std::path::PathBuf;

use clap::builder::TypedValueParser as _;
use clap::Parser;
use sim::training::FitnessKind;
use sim::FeatureSet;

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

/// A finite number from 0 to 1.
fn unit_interval(text: &str) -> Result<f64, String> {
    let value: f64 = text
        .parse()
        .map_err(|e| format!("`{text}` is not a number: {e}"))?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(format!("`{text}` must be between 0 and 1"))
    }
}

/// A finite number above 0 and at most 1.
fn positive_unit(text: &str) -> Result<f64, String> {
    let value: f64 = text
        .parse()
        .map_err(|e| format!("`{text}` is not a number: {e}"))?;
    if value > 0.0 && value <= 1.0 {
        Ok(value)
    } else {
        Err(format!("`{text}` must be above 0 and at most 1"))
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
            "player_count", "deck_variant", "duplicate_rule", "pass_rule", "exchange_rule", "rounds", "population",
            "matches_per_genome", "reeval_matches", "seed", "opponent", "target_species",
            "champion_candidates", "hall_of_fame", "hall_interval", "weight_power", "runs",
            "fitness_skill_weight", "features", "fitness", "racing_top", "racing_matches",
            "player_counts", "decks"
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

    /// Train one model on several table sizes, a comma list such as `3,4,5,6`
    /// (each 3-6). Mutually exclusive with --player-count. With --decks the
    /// settings are every combination; each generation's matches cycle through
    /// them (match i is played at setting i mod the number of settings, the
    /// same for every genome). Use `--features all` so the network can see the
    /// table (the `table` feature group); needs --matches-per-genome and
    /// --reeval-matches of at least the number of settings. See docs/TRAINING.md.
    #[arg(
        long,
        value_delimiter = ',',
        value_parser = clap::value_parser!(u8).range(3..=6),
        conflicts_with = "player_count"
    )]
    pub player_counts: Vec<u8>,

    #[arg(long, value_enum, default_value_t = DeckVariantArg::Single)]
    pub deck_variant: DeckVariantArg,

    /// Train on several decks, a comma list of `single`, `double` (see
    /// --player-counts). Mutually exclusive with --deck-variant.
    #[arg(
        long,
        value_delimiter = ',',
        value_enum,
        conflicts_with = "deck_variant"
    )]
    pub decks: Vec<DeckVariantArg>,

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

    /// Weight W (0 to 1) of a luck-adjusted term in the training fitness:
    /// `(1 - W) * mean role score + W * luck-adjusted round-1 score`. Default
    /// 0 trains on the plain mean role score, exactly as before. Champion
    /// choice, re-evaluation and held-out scores always use the plain mean
    /// role score. An open experiment, see docs/TRAINING.md.
    #[arg(long, default_value_t = 0.0, value_parser = unit_interval)]
    pub fitness_skill_weight: f64,

    /// What evolution maximises: `mean` (the mean role score, default),
    /// `risk:L` (mean minus L standard deviations of the per-round role score:
    /// rewards consistency) or `tanh:K` (mean of tanh(K*s)/tanh(K) over the
    /// per-round role scores s: flattens the top, so a safe Vize is worth
    /// nearly a President). Champion choice, re-evaluation and the held-out
    /// confirmation use it too. Cannot be combined with --fitness-skill-weight.
    /// An open experiment, see docs/TRAINING.md.
    #[arg(long, value_parser = clap::value_parser!(FitnessKind), default_value_t = FitnessKind::Mean)]
    pub fitness: FitnessKind,

    /// Racing: the top fraction F (0 < F <= 1) of each generation's genomes by
    /// fitness plays --racing-matches extra matches (same fresh matches for
    /// all of them) and its fitness is pooled with them before selection, which
    /// pulls back genomes that were merely lucky. Needs --racing-matches.
    /// Cannot be combined with --fitness-skill-weight. See docs/TRAINING.md.
    #[arg(
        long,
        value_name = "F",
        requires = "racing_matches",
        value_parser = positive_unit
    )]
    pub racing_top: Option<f64>,

    /// Extra matches K each raced genome plays (needs --racing-top).
    #[arg(
        long,
        value_name = "K",
        requires = "racing_top",
        value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(1..)
    )]
    pub racing_matches: Option<usize>,

    /// Which features the evolved networks see: `base` (the 20 of feature set
    /// v1, loadable by the casino port), `trick` (+ who is still in the
    /// trick), `hidden` (+ a probabilistic reading of the unseen cards) or
    /// `all` (+ player count, deck and pass rule). Each is a prefix of the
    /// next, and the genome files record the names they were trained with.
    /// See docs/TRAINING.md.
    #[arg(
        long,
        default_value = "base",
        value_parser = clap::builder::PossibleValuesParser::new(["base", "trick", "hidden", "all"])
            .map(|name| FeatureSet::ALL_SETS
                .into_iter()
                .find(|set| set.to_string() == name)
                .expect("the possible values are the feature set names"))
    )]
    pub features: FeatureSet,

    /// Choose the settings for you: run the advisor (`cli train-advisor`, a probe of
    /// this machine, one to a few minutes), print what it chose and why, and train with it.
    /// The population, generations, matches per genome, hall of fame and the opponent pool
    /// are the advisor's, so they cannot be given (nor can `--reeval-matches`,
    /// `--champion-candidates`, `--racing-*`, `--fitness`, `--features` and the rule flags:
    /// the advisor's cost estimate assumes their defaults); `--opponent` then lists the *candidates*
    /// the advisor judges (default: the battery of `cli evaluate`). The settings land in the
    /// run's `config.json` as for any run, and the advice is saved as `advisor.txt` there.
    #[arg(
        long,
        conflicts_with_all = [
            "resume", "from", "runs", "population", "generations", "matches_per_genome", "hall_of_fame",
            "hall_interval", "rounds", "player_counts", "decks",
            // The advisor priced a run with the default evaluation, fitness, features and
            // rules; a flag that changes the cost or the meaning of a generation would make
            // its plan wrong, so these are not accepted either.
            "reeval_matches", "champion_candidates", "racing_top", "racing_matches", "fitness",
            "features", "duplicate_rule", "pass_rule", "exchange_rule"
        ]
    )]
    pub auto: bool,

    /// With --auto: how long the training may take (`30m`, `2h`, `1h30m`). Default: the preset's.
    #[arg(long, requires = "auto")]
    pub time_budget: Option<String>,

    /// With --auto: `quick` (5 minutes), `thorough` (2 hours, default) or `overnight` (10 hours).
    #[arg(long, value_enum, requires = "auto")]
    pub preset: Option<crate::advisor::Preset>,

    /// With --auto: a player the candidates are scored against instead of `adaptive` (repeatable;
    /// `neat:PATH` is a starting genome).
    #[arg(long, value_name = "SPEC", requires = "auto")]
    pub reference: Vec<String>,

    /// With --auto: score the candidates against the genomes (`*.json`) of this folder.
    #[arg(long, value_name = "DIR", requires = "auto")]
    pub reference_models: Option<PathBuf>,

    /// With --auto: score the candidates against the hall of fame of the earlier run in this folder.
    #[arg(long, value_name = "RUN_DIR", requires = "auto")]
    pub reference_hall: Option<PathBuf>,

    /// With --auto: probe matches per candidate opponent (default 40).
    #[arg(long, requires = "auto", value_parser = clap::builder::RangedI64ValueParser::<usize>::new().range(2..))]
    pub probe_matches: Option<usize>,

    /// Watch the run's own curve with the run analysis (`cli analyse-run`): every 5 generations
    /// from generation 30, `advise` prints the verdict once it is flat, `stop` also ends the run
    /// when the last 40% of it added nothing visible. May accompany --resume; `--resume
    /// --generations N` continues a stopped run.
    #[arg(long, value_enum)]
    pub when_flat: Option<crate::flat::WhenFlat>,

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
mod tests;
