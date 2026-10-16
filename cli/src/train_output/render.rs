//! The line renderers: banner, header, rows and summaries, as pure functions of an event.

use std::fmt::Write as _;

use sim::training::{GenerationEvent, RunEnd, RunStart};

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
///
/// `stopped_at` is the planned number of generations when the last run in `ends` was stopped
/// early (`--when-flat stop`): the set then did not finish, that run is marked, and the summary
/// says its numbers are not comparable with runs that trained to the end.
#[must_use]
pub fn render_set_summary(
    ends: &[(String, RunEnd)],
    total_secs: f64,
    stopped_at: Option<u32>,
) -> String {
    let mut text = format!(
        "set {}: {} runs in {}",
        if stopped_at.is_some() {
            "stopped early"
        } else {
            "finished"
        },
        ends.len(),
        format_duration(total_secs)
    );
    for (position, (name, end)) in ends.iter().enumerate() {
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
        if let Some(planned) = stopped_at.filter(|_| position + 1 == ends.len()) {
            let _ = write!(
                text,
                "\n    stopped after {} of {planned} generations",
                end.generations_completed
            );
        }
    }
    if stopped_at.is_some() {
        text.push_str(
            "\n  the stopped run and the runs not started are not comparable with runs that trained to the end; \
             `--resume` on the set directory continues it",
        );
    }
    text
}
