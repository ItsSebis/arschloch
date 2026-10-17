//! The training form's model: the arguments of `cli train-advisor`, and the
//! recommended `cli train` command it prints, split into editable fields.
//! No drawing code, so it is unit-tested; the window runs the real commands.

use crate::form::Deck;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    Quick,
    Thorough,
    Overnight,
}

impl Preset {
    pub const ALL: &'static [Self] = &[Self::Quick, Self::Thorough, Self::Overnight];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Quick => "quick",
            Self::Thorough => "thorough",
            Self::Overnight => "overnight",
        }
    }
}

/// What the user tells the advisor.
#[derive(Clone, Debug, PartialEq)]
pub struct AdvisorForm {
    pub players: u8,
    pub deck: Deck,
    pub preset: Preset,
    /// `30m`, `2h`, `1h30m`; empty takes the preset's budget.
    pub time_budget: String,
    /// Candidate opponents, separated by spaces; empty lets the advisor use its battery.
    pub opponents: String,
    pub probe_matches: usize,
    pub seed: u64,
    /// The run folder written into the recommended command.
    pub out: String,
}

impl Default for AdvisorForm {
    fn default() -> Self {
        Self {
            players: 4,
            deck: Deck::Single,
            preset: Preset::Quick,
            time_budget: String::new(),
            opponents: String::new(),
            probe_matches: 40,
            seed: 0,
            out: "runs/auto".to_owned(),
        }
    }
}

impl AdvisorForm {
    pub fn problem(&self) -> Option<String> {
        if self.probe_matches < 2 {
            return Some("the probe needs at least 2 matches per opponent".to_owned());
        }
        if self.out.trim().is_empty() {
            return Some("give the run folder".to_owned());
        }
        None
    }

    pub fn args(&self) -> Vec<String> {
        let mut args = vec!["train-advisor".to_owned()];
        let mut push = |flag: &str, value: String| {
            args.push(flag.to_owned());
            args.push(value);
        };
        push("--player-count", self.players.to_string());
        push("--deck-variant", self.deck.as_str().to_owned());
        push("--preset", self.preset.as_str().to_owned());
        if !self.time_budget.trim().is_empty() {
            push("--time-budget", self.time_budget.trim().to_owned());
        }
        push("--probe-matches", self.probe_matches.to_string());
        push("--seed", self.seed.to_string());
        push("--out", self.out.trim().to_owned());
        for opponent in self.opponents.split_whitespace() {
            push("--opponent", opponent.to_owned());
        }
        args
    }
}

/// The recommended `cli train` command as editable fields. Flags this form does
/// not know (a later advisor may print more) are kept in `extra` and passed on.
#[derive(Clone, Debug, PartialEq)]
pub struct Recommendation {
    pub out: String,
    pub players: u8,
    pub deck: String,
    pub population: usize,
    pub generations: usize,
    pub matches_per_genome: usize,
    pub rounds: usize,
    /// 0: no hall of fame.
    pub hall_of_fame: usize,
    pub hall_interval: usize,
    pub opponents: Vec<String>,
    pub extra: Vec<(String, Option<String>)>,
}

/// The advisor's recommendation: the last line of its output that is a `cli train` command.
pub fn find_command(output: &str) -> Option<&str> {
    output
        .lines()
        .map(str::trim)
        .rfind(|l| l.starts_with("cli train "))
}

/// Splits a command line at spaces; a `"double quoted"` part (a folder with
/// spaces, as the advisor prints it) stays one word.
fn split_words(line: &str) -> Result<Vec<String>, String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut started = false;
    let mut quoted = false;
    for c in line.chars() {
        match c {
            '"' => {
                quoted = !quoted;
                started = true;
            }
            c if c.is_whitespace() && !quoted => {
                if started {
                    words.push(std::mem::take(&mut word));
                    started = false;
                }
            }
            c => {
                word.push(c);
                started = true;
            }
        }
    }
    if quoted {
        return Err(format!("unbalanced quote in `{line}`"));
    }
    if started {
        words.push(word);
    }
    Ok(words)
}

impl Recommendation {
    /// Splits a `cli train ...` command line.
    ///
    /// # Errors
    ///
    /// The line is not a `cli train` command, or a number in it is not one.
    pub fn parse(command: &str) -> Result<Self, String> {
        let words = split_words(command)?;
        let mut words = words.iter().map(String::as_str);
        if words.next() != Some("cli") || words.next() != Some("train") {
            return Err(format!("`{command}` is not a `cli train` command"));
        }
        let mut rec = Self {
            out: String::new(),
            players: 4,
            deck: "single".to_owned(),
            population: 150,
            generations: 100,
            matches_per_genome: 100,
            rounds: 8,
            hall_of_fame: 0,
            hall_interval: 0,
            opponents: Vec::new(),
            extra: Vec::new(),
        };
        let mut words = words.peekable();
        while let Some(flag) = words.next() {
            let value = words
                .next_if(|next| !next.starts_with("--"))
                .map(str::to_owned);
            let number = |flag: &str, value: &Option<String>| -> Result<usize, String> {
                value
                    .as_deref()
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| format!("{flag} needs a number in `{command}`"))
            };
            match flag {
                "--out" => rec.out = value.unwrap_or_default(),
                "--player-count" => {
                    rec.players = u8::try_from(number(flag, &value)?)
                        .map_err(|_| "--player-count is out of range".to_owned())?;
                }
                "--deck-variant" => rec.deck = value.unwrap_or_default(),
                "--population" => rec.population = number(flag, &value)?,
                "--generations" => rec.generations = number(flag, &value)?,
                "--matches-per-genome" => rec.matches_per_genome = number(flag, &value)?,
                "--rounds" => rec.rounds = number(flag, &value)?,
                "--hall-of-fame" => rec.hall_of_fame = number(flag, &value)?,
                "--hall-interval" => rec.hall_interval = number(flag, &value)?,
                "--opponent" => rec.opponents.extend(value),
                other => rec.extra.push((other.to_owned(), value)),
            }
        }
        Ok(rec)
    }

    pub fn problem(&self) -> Option<String> {
        if self.out.trim().is_empty() {
            return Some("give the run folder".to_owned());
        }
        if self.population == 0
            || self.generations == 0
            || self.matches_per_genome == 0
            || self.rounds == 0
        {
            return Some(
                "population, generations, matches and rounds must be at least 1".to_owned(),
            );
        }
        None
    }

    /// The arguments of `cli train` (without the program name).
    pub fn args(&self) -> Vec<String> {
        let mut args = vec!["train".to_owned()];
        let mut push = |flag: &str, value: String| {
            args.push(flag.to_owned());
            args.push(value);
        };
        push("--out", self.out.trim().to_owned());
        push("--player-count", self.players.to_string());
        push("--deck-variant", self.deck.clone());
        push("--population", self.population.to_string());
        push("--generations", self.generations.to_string());
        push("--matches-per-genome", self.matches_per_genome.to_string());
        push("--rounds", self.rounds.to_string());
        if self.hall_of_fame > 0 {
            push("--hall-of-fame", self.hall_of_fame.to_string());
            // `cli train` rejects `--hall-interval 0`: leave it to its default.
            if self.hall_interval > 0 {
                push("--hall-interval", self.hall_interval.to_string());
            }
        }
        for opponent in &self.opponents {
            push("--opponent", opponent.clone());
        }
        for (flag, value) in &self.extra {
            args.push(flag.clone());
            args.extend(value.clone());
        }
        args
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{real_cli, TempDir};

    const COMMAND: &str = "cli train --out runs/auto --player-count 4 --deck-variant single --population 80 --generations 120 \
         --matches-per-genome 150 --rounds 8 --hall-of-fame 5 --hall-interval 6 \
         --opponent lowest-legal --opponent adaptive:reading";

    #[test]
    fn the_default_form_asks_for_a_quick_plan() {
        let form = AdvisorForm::default();
        assert_eq!(form.problem(), None);
        assert_eq!(
            form.args(),
            [
                "train-advisor",
                "--player-count",
                "4",
                "--deck-variant",
                "single",
                "--preset",
                "quick",
                "--probe-matches",
                "40",
                "--seed",
                "0",
                "--out",
                "runs/auto"
            ]
        );
    }

    #[test]
    fn budget_and_opponents_are_passed_when_given() {
        let form = AdvisorForm {
            time_budget: " 30m ".to_owned(),
            opponents: "lowest-legal  adaptive:reading".to_owned(),
            ..AdvisorForm::default()
        };
        let args = form.args().join(" ");
        assert!(args.contains("--time-budget 30m"), "{args}");
        assert!(
            args.ends_with("--opponent lowest-legal --opponent adaptive:reading"),
            "{args}"
        );
        assert!(AdvisorForm {
            probe_matches: 1,
            ..AdvisorForm::default()
        }
        .problem()
        .is_some());
        assert!(AdvisorForm {
            out: " ".to_owned(),
            ..AdvisorForm::default()
        }
        .problem()
        .is_some());
    }

    #[test]
    fn the_recommendation_is_the_last_cli_train_line() {
        let text = format!("Probe...\n  cli train is mentioned here\n\n{COMMAND}\n");
        assert_eq!(find_command(&text), Some(COMMAND));
        assert_eq!(find_command("nothing here"), None);
    }

    #[test]
    fn a_recommendation_round_trips_into_train_arguments() {
        let rec = Recommendation::parse(COMMAND).unwrap();
        assert_eq!(
            (
                rec.population,
                rec.generations,
                rec.matches_per_genome,
                rec.rounds
            ),
            (80, 120, 150, 8)
        );
        assert_eq!((rec.hall_of_fame, rec.hall_interval), (5, 6));
        assert_eq!(rec.opponents, ["lowest-legal", "adaptive:reading"]);
        assert_eq!(rec.out, "runs/auto");
        assert_eq!(rec.problem(), None);
        let rebuilt = format!("cli {}", rec.args().join(" "));
        assert_eq!(rebuilt, COMMAND, "nothing is lost on the way");
    }

    #[test]
    fn a_quoted_run_folder_with_spaces_stays_one_argument() {
        let command = COMMAND.replace("--out runs/auto", "--out \"my runs/a b\"");
        let rec = Recommendation::parse(&command).unwrap();
        assert_eq!(rec.out, "my runs/a b");
        assert!(rec.extra.is_empty(), "{:?}", rec.extra);
        assert_eq!(rec.population, 80);
        let args = rec.args();
        let at = args.iter().position(|a| a == "--out").unwrap();
        assert_eq!(args[at + 1], "my runs/a b");
        assert!(Recommendation::parse("cli train --out \"oops").is_err());
    }

    #[test]
    fn flags_a_later_advisor_adds_are_passed_through() {
        let rec = Recommendation::parse(&format!(
            "{COMMAND} --fitness consistency --quiet --threads 4"
        ))
        .unwrap();
        assert_eq!(
            rec.extra,
            [
                ("--fitness".to_owned(), Some("consistency".to_owned())),
                ("--quiet".to_owned(), None),
                ("--threads".to_owned(), Some("4".to_owned()))
            ]
        );
        assert!(rec
            .args()
            .join(" ")
            .ends_with("--fitness consistency --quiet --threads 4"));
    }

    #[test]
    fn edits_change_the_arguments_and_no_hall_drops_its_flags() {
        let mut rec = Recommendation::parse(COMMAND).unwrap();
        rec.generations = 30;
        rec.hall_of_fame = 0;
        let args = rec.args().join(" ");
        assert!(
            args.contains("--generations 30") && !args.contains("hall"),
            "{args}"
        );
        rec.population = 0;
        assert!(rec.problem().is_some());
    }

    #[test]
    fn a_hall_without_an_interval_does_not_send_the_rejected_zero() {
        let mut rec = Recommendation::parse(COMMAND).unwrap();
        rec.hall_interval = 0;
        let args = rec.args().join(" ");
        assert!(args.contains("--hall-of-fame 5"), "{args}");
        assert!(!args.contains("--hall-interval"), "{args}");
    }

    #[test]
    fn something_else_is_not_a_recommendation() {
        assert!(Recommendation::parse("cli evaluate x").is_err());
        assert!(Recommendation::parse("cli train --population many").is_err());
    }

    /// The real advisor, with a tiny probe, prints a command this form reads and
    /// the real `cli train` accepts (`--help` is not enough: a one-generation run).
    #[test]
    fn the_real_advisor_feeds_the_form_and_the_real_trainer_accepts_it() {
        let Some(cli) = real_cli() else { return };
        let dir = TempDir::new("advisor");
        let form = AdvisorForm {
            opponents: "lowest-legal random-legal".to_owned(),
            probe_matches: 8,
            time_budget: "10m".to_owned(),
            out: dir.0.join("run").to_string_lossy().into_owned(),
            ..AdvisorForm::default()
        };
        let out = std::process::Command::new(&cli)
            .args(form.args())
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(
            out.status.success(),
            "{text}{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let command = find_command(&text).expect("a recommended command");
        let mut rec = Recommendation::parse(command).unwrap();
        assert_eq!(rec.opponents.len(), 2, "{command}");
        assert_eq!(rec.problem(), None);
        // Shrink it so the test is quick, then run the real trainer.
        rec.population = 10;
        rec.generations = 1;
        rec.matches_per_genome = 2;
        rec.rounds = 1;
        rec.hall_of_fame = 0;
        let run = std::process::Command::new(&cli)
            .args(rec.args())
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
    }
}
