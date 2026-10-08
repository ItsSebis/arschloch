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
        "training: {} players, {:?} deck, {:?} | population {}, {} generations | {} matches x {} rounds per genome, seed {}",
        config.player_count,
        config.deck,
        config.duplicate_rule,
        config.neat.population_size,
        config.generations,
        config.matches_per_genome,
        config.rounds_per_match,
        config.seed
    );
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

#[allow(clippy::struct_excessive_bools)] // independent display switches
pub struct TerminalObserver {
    quiet: bool,
    out: PathBuf,
    live_progress: bool,
    progress_visible: bool,
    opponent_count: usize,
    show_hall: bool,
    total_generations: u32,
    first_generation: Option<u32>,
    generation_secs: Vec<f64>,
}

impl TerminalObserver {
    #[must_use]
    pub fn new(out: PathBuf, quiet: bool) -> Self {
        Self {
            quiet,
            out,
            live_progress: std::io::stderr().is_terminal(),
            progress_visible: false,
            opponent_count: 0,
            show_hall: false,
            total_generations: 0,
            first_generation: None,
            generation_secs: Vec::new(),
        }
    }

    fn clear_progress(&mut self) {
        if self.progress_visible {
            eprint!("\r{:60}\r", "");
            self.progress_visible = false;
        }
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
            println!(
                "{}\n\n{}",
                render_banner(start, &self.out),
                render_header(self.opponent_count, self.show_hall)
            );
        }
    }

    fn on_eval_progress(&mut self, generation: u32, done: usize, total: usize) {
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
        if self.quiet {
            return;
        }
        if (event.generation - first).is_multiple_of(HEADER_EVERY) && event.generation != first {
            println!("{}", render_header(self.opponent_count, self.show_hall));
        }
        println!("{}", render_row(event, self.eta(event), self.show_hall));
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
            },
            champion: ChampionStats {
                train_fitness: 0.412,
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
        }
    }
}
