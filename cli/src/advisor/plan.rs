//! The plan: how long the run may take, how many matches a genome needs, and the
//! `cli train` command line that carries the plan out.

use std::fmt::Write as _;

pub const MIN_MATCHES: usize = 20;
pub const MAX_MATCHES: usize = 400;
pub const MIN_GENERATIONS: u32 = 20;
pub(super) const MAX_GENERATIONS: u32 = 1000;
pub(super) const MIN_POPULATION: usize = 30;
/// The standard error of one fitness measurement the run should reach.
pub(super) const TARGET_ERROR: f64 = 0.02;
/// Share of the time budget the plan may spend: the rest covers what the
/// estimate leaves out (checkpoints, a candidate slower than the probe).
pub(super) const BUDGET_SHARE: f64 = 0.7;
/// Candidates and matches of the champion re-evaluations, as `cli train` defaults them.
pub(super) const CHAMPION_CANDIDATES: usize = 5;
pub(super) const REEVAL_MATCHES: usize = 200;
/// Opponents in the recommended pool at most.
pub(super) const POOL_SIZE: usize = 4;

/// `90s`, `30m`, `2h`, `1h30m` in seconds.
pub fn parse_duration(text: &str) -> Option<f64> {
    let (mut total, mut number, mut any) = (0.0, String::new(), false);
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            number.push(ch);
            continue;
        }
        let unit = match ch {
            's' => 1.0,
            'm' => 60.0,
            'h' => 3600.0,
            _ => return None,
        };
        total += number.parse::<f64>().ok()? * unit;
        number.clear();
        any = true;
    }
    (any && number.is_empty() && total > 0.0).then_some(total)
}

/// Matches per genome for a standard error of `target` when one match's score
/// varies with standard deviation `sd`: rounded up to tens, within the limits.
pub fn matches_for_error(sd: f64, target: f64) -> usize {
    let needed = ((sd / target).powi(2).ceil() as usize).div_ceil(10) * 10;
    needed.clamp(MIN_MATCHES, MAX_MATCHES)
}

#[derive(Debug, Clone, Copy)]
pub struct RunPlan {
    pub population: usize,
    pub generations: u32,
    pub matches_per_genome: usize,
    pub rounds: usize,
    pub hall_of_fame: usize,
    pub hall_interval: u32,
    pub estimated_seconds: f64,
    pub budget_seconds: f64,
}

impl RunPlan {
    /// Why this plan cannot do what the advice promises (at least `MIN_GENERATIONS`
    /// within the budget), in words; `None` when it can.
    pub fn problem(&self) -> Option<String> {
        (self.generations < MIN_GENERATIONS || self.estimated_seconds > self.budget_seconds).then(|| {
            format!(
                "only {} generations fit and the plan needs about {:.0} s of the {:.0} s budget even at the smallest settings: give it more time",
                self.generations, self.estimated_seconds, self.budget_seconds
            )
        })
    }
}

/// The spread of one match's score over several opponents: the root of the mean
/// variance (deviations of different opponents are not averaged directly).
pub fn pooled_sd(sds: &[f64]) -> f64 {
    if sds.is_empty() {
        return 0.0;
    }
    (sds.iter().map(|sd| sd * sd).sum::<f64>() / sds.len() as f64).sqrt()
}

/// Population, generations and matches per genome that fit `budget` seconds at
/// `rounds_per_sec`. The wanted `matches` and `max_population` shrink (population
/// first) until at least `MIN_GENERATIONS` fit.
pub fn plan_run(
    budget: f64,
    rounds_per_sec: f64,
    rounds: usize,
    matches: usize,
    max_population: usize,
    pool_len: usize,
) -> RunPlan {
    // One generation, as the trainer counts it: every genome's matches, plus one
    // re-evaluation of each champion candidate, the champion and every pool opponent.
    let seconds_per_generation = |population: usize, matches: usize| {
        (population * matches + REEVAL_MATCHES * (CHAMPION_CANDIDATES + 1 + pool_len)) as f64
            * rounds as f64
            / rounds_per_sec
    };
    let usable = budget * BUDGET_SHARE;
    let (mut population, mut matches) = (max_population, matches);
    let generations = loop {
        let fit = (usable / seconds_per_generation(population, matches)).floor() as u32;
        if fit >= MIN_GENERATIONS || (population <= MIN_POPULATION && matches <= MIN_MATCHES) {
            break fit.clamp(1, MAX_GENERATIONS);
        }
        if population > MIN_POPULATION {
            population = (population * 2 / 3).max(MIN_POPULATION);
        } else {
            matches = (matches / 2).max(MIN_MATCHES);
        }
    };
    RunPlan {
        population,
        generations,
        matches_per_genome: matches,
        rounds,
        hall_of_fame: if generations >= 60 { 5 } else { 0 },
        hall_interval: (generations / 20).max(5),
        estimated_seconds: f64::from(generations) * seconds_per_generation(population, matches),
        budget_seconds: budget,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Keep,
    TooWeak,
    TooStrong,
}

/// Is an opponent that the reference player scores `reference_mean` against
/// (+1 best .. -1 worst) useful for training?
pub fn judge(reference_mean: f64) -> Verdict {
    if reference_mean > 0.85 {
        Verdict::TooWeak
    } else if reference_mean < -0.6 {
        Verdict::TooStrong
    } else {
        Verdict::Keep
    }
}

impl Verdict {
    pub fn reason(self) -> &'static str {
        match self {
            Self::Keep => "kept: there is room to improve and to lose",
            Self::TooWeak => "dropped, too weak: the reference player wins almost every game, so it teaches little",
            Self::TooStrong => "dropped, too strong: the reference player almost never wins, so there is no signal",
        }
    }
}

/// The `cli train` command line that carries the plan out.
pub fn command_line(
    plan: &RunPlan,
    players: u8,
    deck: &str,
    opponents: &[String],
    out: &str,
) -> String {
    // A folder with spaces is quoted so the line can be pasted into a shell and
    // split again by the desktop window.
    let out = if out.contains(char::is_whitespace) {
        format!("\"{out}\"")
    } else {
        out.to_owned()
    };
    let mut line = format!(
        "cli train --out {out} --player-count {players} --deck-variant {deck} --population {} --generations {} \
         --matches-per-genome {} --rounds {}",
        plan.population, plan.generations, plan.matches_per_genome, plan.rounds
    );
    if plan.hall_of_fame > 0 {
        let _ = write!(
            line,
            " --hall-of-fame {} --hall-interval {}",
            plan.hall_of_fame, plan.hall_interval
        );
    }
    for opponent in opponents {
        let _ = write!(line, " --opponent {opponent}");
    }
    line
}
