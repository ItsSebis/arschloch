//! What `cli train` prints while it runs: a banner, one line per
//! generation, a refreshed progress line on a terminal, and a summary.
//! The line renderers are pure functions of an event so they can be
//! tested without capturing stdout.

use std::fmt::Write as _;
use std::io::{IsTerminal, Write as _};
use std::path::PathBuf;

use sim::training::{GenerationEvent, RunEnd, RunStart, TrainObserver};

/// How often (in generations) the column header is repeated.
const HEADER_EVERY: u32 = 20;

#[must_use]
pub fn format_duration(secs: f64) -> String {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let total = secs.max(0.0).round() as u64;
    format!(
        "{}:{:02}:{:02}",
        total / 3600,
        total % 3600 / 60,
        total % 60
    )
}

/// Rounds per second, compactly (`842`, `12.3k`, `1.2M`).
#[must_use]
pub fn format_rate(per_sec: f64) -> String {
    if per_sec >= 1e6 {
        format!("{:.1}M", per_sec / 1e6)
    } else if per_sec >= 1e4 {
        format!("{:.0}k", per_sec / 1e3)
    } else if per_sec >= 1e3 {
        format!("{:.1}k", per_sec / 1e3)
    } else {
        format!("{per_sec:.0}")
    }
}

#[must_use]
pub fn render_banner(start: &RunStart, out: &std::path::Path) -> String {
    let config = &start.config;
    let mut text = String::new();
    let _ = writeln!(
        text,
        "training: {} players, {:?} deck, {:?}, pass rule {}, exchange rule {} | population {}, {} generations | {} matches x {} rounds per genome, seed {}",
        config.player_count,
        config.deck,
        config.duplicate_rule,
        config.pass_rule,
        config.exchange_rule,
        config.neat.population_size,
        config.generations,
        config.matches_per_genome,
        config.rounds_per_match,
        config.seed
    );
    // Only the Phase 13 options that are on are mentioned, so a default
    // run's banner is what it was.
    let tables = config.tables();
    if tables.len() > 1 {
        let labels: Vec<String> = tables.iter().map(sim::training::TableSpec::label).collect();
        let _ = writeln!(
            text,
            "settings: {} (matches cycle through them)",
            labels.join(", ")
        );
    }
    if !config.fitness.is_mean() {
        let _ = writeln!(
            text,
            "fitness: {} (best/mean columns are this fitness; the champion column is its mean role score, sd = spread of its per-round score)",
            config.fitness
        );
    }
    if config.racing_matches > 0 {
        let _ = writeln!(
            text,
            "racing: top {} genomes x {} extra matches (shift = mean fitness change of the raced genomes)",
            config.raced_count(),
            config.racing_matches
        );
    }
    let legend: Vec<String> = start
        .opponents
        .iter()
        .enumerate()
        .map(|(i, name)| format!("o{}={name}", i + 1))
        .collect();
    let _ = writeln!(text, "opponents: {}", legend.join("  "));
    if let Some(source) = &start.warm_started_from {
        let _ = writeln!(
            text,
            "warm start: {source} (its final population of {} genomes)",
            config.neat.population_size
        );
    }
    let _ = write!(text, "output: {}", out.display());
    if let Some(generation) = start.resumed_from_generation {
        let _ = write!(text, "  (resumed from generation {generation})");
    }
    text
}

#[must_use]
pub fn render_header(opponent_count: usize, show_hall: bool) -> String {
    let mut text = String::from("  gen    best    mean  champion (fresh)  spc  nodes/conn");
    for i in 1..=opponent_count {
        let _ = write!(text, "   o{i:<3}");
    }
    if show_hall {
        text.push_str("   hof ");
    }
    text.push_str("  rounds/s      ETA");
    text
}

/// One generation as a line: selection fitness, the champion's fresh
/// score with its standard error, species count, champion size, the
/// champion's score against each opponent alone, throughput and ETA. A
/// trailing `*` marks a new best champion.
#[must_use]
pub fn render_row(event: &GenerationEvent, eta_secs: Option<f64>, show_hall: bool) -> String {
    let champion = &event.champion;
    let mut text = format!(
        "{:>5} {:>+7.3} {:>+7.3}  {:>+7.3} ±{:<6.3}  {:>3}  {:>4}/{:<5}",
        event.generation,
        event.fitness.best,
        event.fitness.mean,
        champion.reeval.mean,
        champion.reeval.std_error,
        event.species.len(),
        champion.hidden_nodes,
        champion.enabled_connections,
    );
    for opponent in &event.opponents {
        let _ = write!(text, " {:>+6.2}", opponent.score.mean);
    }
    if show_hall {
        match &event.hall_score {
            Some(score) => {
                let _ = write!(text, " {:>+6.2}", score.mean);
            }
            None => text.push_str("      –"),
        }
    }
    let _ = write!(
        text,
        "  {:>8}  {:>8}",
        format_rate(event.rounds_per_sec),
        eta_secs.map_or_else(|| "--:--:--".to_owned(), format_duration)
    );
    if let Some(spread) = champion.reeval.spread {
        let _ = write!(text, " sd{spread:.2}");
    }
    if let Some(racing) = &event.racing {
        let _ = write!(text, " shift{:+.3}", racing.mean_shift);
    }
    if champion.is_new_best {
        text.push_str(" *");
    }
    text
}

#[must_use]
pub fn render_summary(end: &RunEnd, out: &std::path::Path) -> String {
    let mut text = format!(
        "done: {} generations in {}",
        end.generations_completed,
        format_duration(end.elapsed_secs)
    );
    // The held-out score was not used to pick the champion, so it is not
    // inflated by selection (the re-evaluation score is, a little).
    let confirmed = end.best_heldout.as_ref().or(end.best_reeval.as_ref());
    if let (Some(generation), Some(score)) = (end.best_generation, confirmed) {
        let _ = write!(
            text,
            "\nbest champion: generation {generation}, score {:+.3} ±{:.3} against the pool (held-out matches)\nplay it: cli --player-count 4 --matches 1000 --strategy neat:{} --strategy lowest-legal --strategy lowest-legal --strategy lowest-legal",
            score.mean,
            score.std_error,
            out.join("best.json").display()
        );
    }
    text
}

/// Where a run sits in a set, so its rows can show the set's ETA.
#[derive(Debug, Clone)]
pub struct SetContext {
    pub total_runs: u32,
    /// Wall-clock seconds of the runs that already finished.
    pub finished_secs: Vec<f64>,
}

/// The banner line that starts run `index` of a set.
#[must_use]
pub fn render_set_run_banner(index: u32, total: u32, seed: u64) -> String {
    format!("=== run {index}/{total} (seed {seed}) ===")
}

/// The closing summary of a set: one line per run, then the total time.
#[must_use]
pub fn render_set_summary(ends: &[(String, RunEnd)], total_secs: f64) -> String {
    let mut text = format!(
        "set finished: {} runs in {}",
        ends.len(),
        format_duration(total_secs)
    );
    for (name, end) in ends {
        let confirmed = end.best_heldout.as_ref().or(end.best_reeval.as_ref());
        match (end.best_generation, confirmed) {
            (Some(generation), Some(score)) => {
                let _ = write!(
                    text,
                    "\n  {name}: best generation {generation}, held-out {:+.3} ±{:.3} ({})",
                    score.mean,
                    score.std_error,
                    format_duration(end.elapsed_secs)
                );
            }
            _ => {
                let _ = write!(text, "\n  {name}: no champion");
            }
        }
    }
    text
}

/// With this environment variable set, `cli train` prints `progress: FRACTION TEXT` lines on
/// stderr (the desktop window's progress bar reads them; the output is unchanged without it).
pub const PROGRESS_ENV: &str = "ARSCHLOCH_PROGRESS";
/// A progress line is printed when the fraction grew by at least this much.
const PROGRESS_STEP: f64 = 0.005;

/// How far the run is: finished generations plus the share of this one's genomes evaluated.
fn progress_fraction(generation: u32, done: usize, total: usize, generations: u32) -> f64 {
    if generations == 0 || total == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let within = done as f64 / total as f64;
    ((f64::from(generation) + within) / f64::from(generations)).clamp(0.0, 1.0)
}

/// Where a run's fraction lands in a set of `total_runs` runs, `finished` of them done.
fn set_fraction(finished: usize, total_runs: u32, run_fraction: f64) -> f64 {
    if total_runs == 0 {
        return run_fraction;
    }
    #[allow(clippy::cast_precision_loss)]
    let before = finished as f64;
    ((before + run_fraction) / f64::from(total_runs)).clamp(0.0, 1.0)
}

/// `position` is `(k, N)` when this is run `k` of a set of `N`: the text then says so
/// (the fraction passed in is already the set's).
fn progress_line(
    fraction: f64,
    generation: u32,
    done: usize,
    total: usize,
    generations: u32,
    position: Option<(usize, u32)>,
) -> String {
    let run = position.map_or_else(String::new, |(k, n)| format!("run {k}/{n} "));
    format!(
        "progress: {fraction:.4} {run}generation {}/{generations}, genomes {done}/{total}",
        generation + 1
    )
}

#[allow(clippy::struct_excessive_bools)] // independent display switches
pub struct TerminalObserver {
    set: Option<SetContext>,
    quiet: bool,
    out: PathBuf,
    live_progress: bool,
    progress_visible: bool,
    /// `ARSCHLOCH_PROGRESS` is set: print machine-readable `progress:` lines (the desktop window's bar).
    progress_lines: bool,
    last_progress: f64,
    opponent_count: usize,
    show_hall: bool,
    total_generations: u32,
    first_generation: Option<u32>,
    generation_secs: Vec<f64>,
    /// `--when-flat`: looks at the run's own events while it trains.
    flat: Option<crate::flat::FlatWatch>,
}

impl TerminalObserver {
    #[must_use]
    pub fn new(out: PathBuf, quiet: bool) -> Self {
        Self {
            set: None,
            quiet,
            out,
            live_progress: std::io::stderr().is_terminal(),
            progress_visible: false,
            progress_lines: std::env::var_os(PROGRESS_ENV).is_some(),
            last_progress: -1.0,
            opponent_count: 0,
            show_hall: false,
            total_generations: 0,
            first_generation: None,
            generation_secs: Vec::new(),
            flat: None,
        }
    }

    /// Advises or stops when the run's analysis finds the curve flat (`--when-flat`).
    #[must_use]
    pub fn with_flat(mut self, mode: crate::flat::WhenFlat) -> Self {
        self.flat = Some(crate::flat::FlatWatch::new(self.out.clone(), mode));
        self
    }

    /// Marks this run as part of a set (its rows then end with the set's ETA).
    #[must_use]
    pub fn with_set(mut self, set: SetContext) -> Self {
        self.set = Some(set);
        self
    }

    fn clear_progress(&mut self) {
        if self.progress_visible {
            eprint!("\r{:60}\r", "");
            self.progress_visible = false;
        }
    }

    fn header(&self) -> String {
        let mut header = render_header(self.opponent_count, self.show_hall);
        if self.set.is_some() {
            header.push_str("  set ETA");
        }
        header
    }

    fn eta(&self, event: &GenerationEvent) -> Option<f64> {
        if self.generation_secs.is_empty() {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let average = self.generation_secs.iter().sum::<f64>() / self.generation_secs.len() as f64;
        let remaining = self.total_generations.saturating_sub(event.generation + 1);
        Some(average * f64::from(remaining))
    }
}

impl TrainObserver for TerminalObserver {
    fn on_start(&mut self, start: &RunStart) {
        self.opponent_count = start.opponents.len();
        self.show_hall = start.config.hall_of_fame_size > 0;
        self.total_generations = start.config.generations;
        if !self.quiet {
            println!("{}\n\n{}", render_banner(start, &self.out), self.header());
        }
    }

    fn on_eval_progress(&mut self, generation: u32, done: usize, total: usize) {
        if self.progress_lines {
            let mut fraction = progress_fraction(generation, done, total, self.total_generations);
            let position = self
                .set
                .as_ref()
                .map(|set| (set.finished_secs.len() + 1, set.total_runs));
            if let Some(set) = &self.set {
                fraction = set_fraction(set.finished_secs.len(), set.total_runs, fraction);
            }
            if fraction - self.last_progress >= PROGRESS_STEP || done == total {
                self.last_progress = fraction;
                eprintln!(
                    "{}",
                    progress_line(
                        fraction,
                        generation,
                        done,
                        total,
                        self.total_generations,
                        position
                    )
                );
            }
        }
        if self.quiet || !self.live_progress {
            return;
        }
        eprint!("\r  generation {generation}: evaluated {done}/{total} genomes ");
        let _ = std::io::stderr().flush();
        self.progress_visible = true;
    }

    fn on_generation(&mut self, event: &GenerationEvent) {
        self.clear_progress();
        self.generation_secs.push(event.generation_secs);
        let first = *self.first_generation.get_or_insert(event.generation);
        let advice = self
            .flat
            .as_mut()
            .and_then(|flat| flat.after_generation(event.generation));
        if self.quiet {
            if let Some(advice) = advice {
                println!("{advice}");
            }
            return;
        }
        if (event.generation - first).is_multiple_of(HEADER_EVERY) && event.generation != first {
            println!("{}", self.header());
        }
        let mut row = render_row(event, self.eta(event), self.show_hall);
        if let Some(set) = &self.set {
            let eta = sim::training::set_eta(
                &set.finished_secs,
                set.total_runs,
                self.eta(event),
                event.elapsed_secs,
            );
            let _ = write!(
                row,
                "  set {}",
                eta.map_or_else(|| "--:--:--".to_owned(), format_duration)
            );
        }
        println!("{row}");
        if let Some(advice) = advice {
            println!("{advice}");
        }
    }

    fn should_stop(&mut self) -> bool {
        self.flat
            .as_ref()
            .is_some_and(crate::flat::FlatWatch::stopping)
    }

    fn on_finish(&mut self, end: &RunEnd) {
        self.clear_progress();
        println!("\n{}", render_summary(end, &self.out));
    }
}

#[cfg(test)]
mod tests {
    use sim::training::events::{ChampionStats, Complexity, FitnessStats, OpponentStat};
    use sim::training::{ScoreStat, SCHEMA_VERSION};

    use super::*;

    fn stat(mean: f64) -> ScoreStat {
        ScoreStat {
            mean,
            std_error: 0.021,
            matches: 100,
            placements: vec![1, 2, 3, 4],
            spread: None,
            fitness: None,
            by_setting: None,
        }
    }

    fn event(is_new_best: bool) -> GenerationEvent {
        GenerationEvent {
            generation: 42,
            elapsed_secs: 100.0,
            generation_secs: 2.0,
            rounds_evaluated: 168_000,
            total_rounds: 1_000_000,
            rounds_per_sec: 84_000.0,
            fitness: FitnessStats {
                best: 0.412,
                mean: 0.188,
                median: 0.2,
                min: -0.5,
                std_dev: 0.1,
                histogram: vec![0; 10],
                skill_term: None,
            },
            champion: ChampionStats {
                train_fitness: 0.412,
                skill_term: None,
                reeval: stat(0.397),
                heldout: None,
                training_rank: 0,
                hidden_nodes: 7,
                enabled_connections: 23,
                genome_file: "gen-0042.json".into(),
                is_new_best,
            },
            opponents: vec![
                OpponentStat {
                    name: "LowestLegal".into(),
                    score: stat(0.61),
                },
                OpponentStat {
                    name: "Adaptive".into(),
                    score: stat(-0.05),
                },
            ],
            hall_of_fame: vec![],
            hall_score: None,
            species: vec![],
            compatibility_threshold: 0.5,
            complexity: Complexity {
                mean_hidden_nodes: 1.0,
                mean_enabled_connections: 20.0,
                innovation_count: 30,
            },
            racing: None,
            timings: None,
        }
    }

    #[test]
    fn a_row_shows_every_headline_number() {
        let row = render_row(&event(false), Some(2470.0), false);
        for expected in [
            "42", "+0.412", "+0.188", "+0.397", "±0.021", "7/23", "+0.61", "-0.05", "84k",
            "0:41:10",
        ] {
            assert!(row.contains(expected), "{expected} missing from: {row}");
        }
        assert!(!row.ends_with('*'));
    }

    #[test]
    fn a_new_best_is_marked_and_a_missing_eta_is_dashes() {
        let row = render_row(&event(true), None, false);
        assert!(row.ends_with(" *"), "{row}");
        assert!(row.contains("--:--:--"));
    }

    #[test]
    fn the_header_has_one_column_per_opponent() {
        let header = render_header(3, false);
        assert!(header.contains("o1") && header.contains("o2") && header.contains("o3"));
        assert!(!header.contains("o4"));
        assert!(header.contains("ETA") && header.contains("champion"));
    }

    #[test]
    fn the_hall_of_fame_column_appears_only_when_the_hall_is_enabled() {
        assert!(!render_header(2, false).contains("hof"));
        assert!(render_header(2, true).contains("hof"));
        let mut with = event(false);
        with.hall_score = Some(stat(0.37));
        assert!(render_row(&with, None, true).contains("+0.37"));
        let without = event(false);
        assert!(
            render_row(&without, None, true).contains('–'),
            "an empty hall shows a dash"
        );
        assert!(!render_row(&with, None, false).contains("+0.37"));
        assert_eq!(
            render_header(2, true).split_whitespace().count(),
            render_header(2, false).split_whitespace().count() + 1
        );
    }

    #[test]
    fn durations_and_rates_format_compactly() {
        assert_eq!(format_duration(0.0), "0:00:00");
        assert_eq!(format_duration(61.4), "0:01:01");
        assert_eq!(format_duration(3_725.0), "1:02:05");
        assert_eq!(format_duration(-5.0), "0:00:00");
        assert_eq!(format_rate(842.0), "842");
        assert_eq!(format_rate(1_234.0), "1.2k");
        assert_eq!(format_rate(84_000.0), "84k");
        assert_eq!(format_rate(2_500_000.0), "2.5M");
    }

    #[test]
    fn the_summary_names_the_best_generation_and_how_to_play_it() {
        let end = RunEnd {
            generations_completed: 100,
            best_generation: Some(87),
            best_reeval: Some(stat(0.652)),
            best_heldout: Some(stat(0.640)),
            elapsed_secs: 5025.0,
        };
        let text = render_summary(&end, std::path::Path::new("runs/a"));
        assert!(text.contains("100 generations in 1:23:45"), "{text}");
        assert!(
            text.contains("generation 87") && text.contains("+0.640"),
            "{text}"
        );
        assert!(text.contains("held-out"), "{text}");
        assert!(
            !text.contains("+0.652"),
            "the selection score is not the headline: {text}"
        );
        let played = std::path::Path::new("runs/a").join("best.json");
        assert!(
            text.contains(&format!("neat:{}", played.display())),
            "{text}"
        );
        let none = RunEnd {
            best_generation: None,
            best_reeval: None,
            ..end
        };
        assert!(!render_summary(&none, std::path::Path::new("x")).contains("best champion"));
    }

    #[test]
    fn the_banner_names_the_pass_rule() {
        let mut start = RunStart {
            schema_version: SCHEMA_VERSION,
            config: sim_config(),
            opponents: vec![],
            feature_names: vec![],
            resumed_from_generation: None,
            warm_started_from: None,
        };
        let text = render_banner(&start, std::path::Path::new("runs/a"));
        assert!(text.contains("pass rule final"), "{text}");
        start.config.pass_rule = sim::PassRule::Free;
        let text = render_banner(&start, std::path::Path::new("runs/a"));
        assert!(text.contains("pass rule free"), "{text}");
    }

    #[test]
    fn the_banner_lists_opponents_and_flags_a_resume() {
        let start = RunStart {
            schema_version: SCHEMA_VERSION,
            config: sim_config(),
            opponents: vec!["LowestLegal".into(), "Adaptive(x)".into()],
            feature_names: vec![],
            resumed_from_generation: Some(12),
            warm_started_from: None,
        };
        let text = render_banner(&start, std::path::Path::new("runs/a"));
        assert!(text.contains("o1=LowestLegal  o2=Adaptive(x)"), "{text}");
        assert!(text.contains("resumed from generation 12"), "{text}");
        assert!(text.contains("4 players"), "{text}");
    }

    #[test]
    fn progress_counts_finished_generations_and_the_share_of_this_one() {
        assert!((progress_fraction(0, 0, 50, 10) - 0.0).abs() < 1e-12);
        assert!((progress_fraction(4, 25, 50, 10) - 0.45).abs() < 1e-12);
        assert!((progress_fraction(9, 50, 50, 10) - 1.0).abs() < 1e-12);
        assert!(
            progress_fraction(0, 5, 0, 10).abs() < 1e-12,
            "no genomes: no progress, no NaN"
        );
        assert!(
            progress_fraction(0, 5, 10, 0).abs() < 1e-12,
            "unknown total: no NaN"
        );
        assert!(
            (progress_fraction(30, 1, 1, 10) - 1.0).abs() < 1e-12,
            "never above 1"
        );
    }

    #[test]
    fn a_progress_line_is_the_documented_format() {
        let line = progress_line(0.4512, 4, 25, 50, 10, None);
        assert_eq!(line, "progress: 0.4512 generation 5/10, genomes 25/50");
    }

    #[test]
    fn in_a_set_the_fraction_and_text_include_the_run_position() {
        // Run 2 of 4, halfway through it: (1 + 0.5) / 4.
        assert!((set_fraction(1, 4, 0.5) - 0.375).abs() < 1e-12);
        assert!(set_fraction(0, 4, 0.0).abs() < 1e-12);
        assert!(
            (set_fraction(3, 4, 1.0) - 1.0).abs() < 1e-12,
            "the last run ends at 1"
        );
        assert!(
            (set_fraction(0, 0, 0.3) - 0.3).abs() < 1e-12,
            "no set: unchanged"
        );
        let line = progress_line(0.375, 4, 25, 50, 10, Some((2, 4)));
        assert_eq!(
            line,
            "progress: 0.3750 run 2/4 generation 5/10, genomes 25/50"
        );
        // The window reads the number right after the prefix, with or without the run.
        let number = line
            .strip_prefix("progress: ")
            .unwrap()
            .split(' ')
            .next()
            .unwrap();
        assert!(number.parse::<f64>().is_ok());
    }

    /// The line the window reads must be what the trainer prints: this is the
    /// contract `gui/src/progress.rs` parses (`progress: FRACTION TEXT`).
    #[test]
    fn the_line_starts_the_way_the_window_expects() {
        let line = progress_line(1.0, 9, 50, 50, 10, None);
        let rest = line.strip_prefix("progress: ").unwrap();
        let number = rest.split(' ').next().unwrap();
        assert!(number.parse::<f64>().is_ok(), "{line}");
    }

    fn sim_config() -> sim::training::TrainConfig {
        sim::training::TrainConfig {
            seed: 0,
            player_count: 4,
            deck: sim::training::DeckChoice::Single,
            duplicate_rule: sim::training::DuplicateChoice::FirstDealtWins,
            rounds_per_match: 8,
            matches_per_genome: 100,
            reeval_matches: 200,
            generations: 100,
            neat: neat::NeatConfig::default(),
            opponent_specs: vec![],
            champion_candidates: 1,
            hall_of_fame_size: 0,
            hall_of_fame_interval: 5,
            skill_weight: 0.0,
            features: sim::FeatureSet::Base,
            fitness: sim::training::FitnessKind::Mean,
            racing_top: 0.0,
            racing_matches: 0,
            player_counts: Vec::new(),
            decks: Vec::new(),
            pass_rule: sim::PassRule::default(),
            exchange_rule: sim::ExchangeRule::default(),
        }
    }
}
