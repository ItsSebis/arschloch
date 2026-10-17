//! The simulation form: every option of a plain `cli` run, turned into the
//! command line the CLI accepts. No drawing code, so it is unit-tested.

use serde::{Deserialize, Serialize};

/// A closed set of command-line values: the enum, its text and the list the
/// window offers.
macro_rules! choices {
    ($name:ident { $($variant:ident = $text:literal),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
        pub enum $name { $($variant),+ }

        impl $name {
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];

            pub fn as_str(self) -> &'static str {
                match self { $(Self::$variant => $text),+ }
            }
        }
    };
}

choices!(Deck { Single = "single", Double = "double" });
choices!(DuplicateRule { FirstDealtWins = "first-dealt-wins", LastDealtWins = "last-dealt-wins" });
choices!(PassRule { Final = "final", Free = "free" });
choices!(ExchangeRule { Forced = "forced", Free = "free" });
choices!(SkillScore { Off = "off", Estimate = "estimate", Duplicate = "duplicate", Both = "both" });

/// Whether a spec ends in `+solver` or `+solver:OPTIONS` (the endgame solver finishes the
/// round for that strategy).
pub fn has_solver(spec: &str) -> bool {
    solver_suffix_at(spec.trim()).is_some()
}

const SOLVER_SUFFIX: &str = "+solver";

/// Where the toggle starts: the last `+solver` that ends the spec or is followed by `:`
/// (the GUI does not link `sim`; the rule is the one of `sim::split_solver_suffix`, and a
/// test here runs the table of cases that sim's test uses).
fn solver_suffix_at(spec: &str) -> Option<usize> {
    let at = spec.rfind(SOLVER_SUFFIX)?;
    let rest = &spec[at + SOLVER_SUFFIX.len()..];
    (rest.is_empty() || rest.starts_with(':')).then_some(at)
}

/// The spec with the `+solver` toggle added or removed; the options of an existing toggle
/// are kept when it stays on.
pub fn with_solver(spec: &str, on: bool) -> String {
    let spec = spec.trim();
    match (solver_suffix_at(spec), on) {
        // `+solver` alone is no spec (the strategy is missing): nothing to toggle.
        (None, true) if spec.is_empty() => String::new(),
        (Some(at), false) => spec[..at].to_owned(),
        (None, true) => format!("{spec}{SOLVER_SUFFIX}"),
        _ => spec.to_owned(),
    }
}

/// A spec split into the strategy and its `+solver[:OPTIONS]` toggle (empty when off).
pub fn split_solver(spec: &str) -> (&str, &str) {
    let spec = spec.trim();
    solver_suffix_at(spec).map_or((spec, ""), |at| spec.split_at(at))
}

/// `neat:PATH` for the genome picked, behind which the seat's `+solver` toggle is kept.
pub fn with_genome(spec: &str, path: &str) -> String {
    format!("neat:{path}{}", split_solver(spec).1)
}

/// The strategy every new seat starts with.
const DEFAULT_STRATEGY: &str = "adaptive";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Form {
    pub player_count: u8,
    pub deck_variant: Deck,
    pub duplicate_rule: DuplicateRule,
    pub pass_rule: PassRule,
    pub exchange_rule: ExchangeRule,
    pub matches: usize,
    pub rounds: usize,
    pub seed: u64,
    /// 0 lets the runtime choose.
    pub threads: usize,
    pub skill_score: SkillScore,
    pub bootstrap_resamples: usize,
    pub output: String,
    /// One strategy spec per seat (`adaptive`, `neat:PATH`, `endgame-solver:...`).
    pub strategies: Vec<String>,
    pub explain: bool,
    /// Rollouts per sampled voluntary pass; `None` skips the analysis.
    pub useful_passes: Option<usize>,
    pub useful_pass_sample: f64,
    pub useful_pass_margin: f64,
}

impl Default for Form {
    fn default() -> Self {
        Self {
            player_count: 4,
            deck_variant: Deck::Single,
            duplicate_rule: DuplicateRule::FirstDealtWins,
            pass_rule: PassRule::Final,
            exchange_rule: ExchangeRule::Forced,
            matches: 1000,
            rounds: 5,
            seed: 0,
            threads: 0,
            skill_score: SkillScore::Off,
            bootstrap_resamples: 200,
            output: "results.json".to_owned(),
            strategies: vec![DEFAULT_STRATEGY.to_owned(); 4],
            explain: false,
            useful_passes: None,
            useful_pass_sample: 0.05,
            useful_pass_margin: 0.0,
        }
    }
}

impl Form {
    /// Changes the table size (clamped to 3-6) and keeps one strategy per seat.
    pub fn set_player_count(&mut self, count: u8) {
        self.player_count = count.clamp(3, 6);
        self.strategies
            .resize(usize::from(self.player_count), DEFAULT_STRATEGY.to_owned());
    }

    /// What the CLI would reject, in words; `None` when the form is runnable.
    pub fn problem(&self) -> Option<String> {
        let players = usize::from(self.player_count);
        if self.matches == 0 || self.rounds == 0 {
            return Some("matches and rounds must be at least 1".to_owned());
        }
        if matches!(self.skill_score, SkillScore::Duplicate | SkillScore::Both)
            && !self.matches.is_multiple_of(players)
        {
            return Some(format!(
                "the duplicate skill score plays groups of {players} matches: matches must be a multiple of {players}"
            ));
        }
        self.strategies
            .iter()
            .position(|s| s.trim().is_empty())
            .map(|seat| format!("seat {} has no strategy", seat + 1))
    }

    /// The arguments of a plain `cli` run.
    pub fn args(&self) -> Vec<String> {
        let mut args: Vec<String> = Vec::new();
        let mut push = |flag: &str, value: String| {
            args.push(flag.to_owned());
            args.push(value);
        };
        push("--player-count", self.player_count.to_string());
        push("--deck-variant", self.deck_variant.as_str().to_owned());
        push("--duplicate-rule", self.duplicate_rule.as_str().to_owned());
        push("--pass-rule", self.pass_rule.as_str().to_owned());
        push("--exchange-rule", self.exchange_rule.as_str().to_owned());
        push("--matches", self.matches.to_string());
        push("--rounds", self.rounds.to_string());
        push("--seed", self.seed.to_string());
        push("--threads", self.threads.to_string());
        push("--skill-score", self.skill_score.as_str().to_owned());
        push(
            "--bootstrap-resamples",
            self.bootstrap_resamples.to_string(),
        );
        push("--output", self.output.clone());
        for strategy in &self.strategies {
            push("--strategy", strategy.trim().to_owned());
        }
        if let Some(rollouts) = self.useful_passes {
            push("--useful-passes", rollouts.to_string());
            push("--useful-pass-sample", self.useful_pass_sample.to_string());
            push("--useful-pass-margin", self.useful_pass_margin.to_string());
        }
        if self.explain {
            args.push("--explain".to_owned());
        }
        args
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_form_is_a_valid_four_player_run() {
        let args = Form::default().args();
        assert_eq!(
            args,
            [
                "--player-count",
                "4",
                "--deck-variant",
                "single",
                "--duplicate-rule",
                "first-dealt-wins",
                "--pass-rule",
                "final",
                "--exchange-rule",
                "forced",
                "--matches",
                "1000",
                "--rounds",
                "5",
                "--seed",
                "0",
                "--threads",
                "0",
                "--skill-score",
                "off",
                "--bootstrap-resamples",
                "200",
                "--output",
                "results.json",
                "--strategy",
                "adaptive",
                "--strategy",
                "adaptive",
                "--strategy",
                "adaptive",
                "--strategy",
                "adaptive",
            ]
        );
    }

    /// The real `cli` (built next to the test binary's directory when the whole
    /// workspace is tested) accepts what the form produces and rejects nonsense.
    #[test]
    fn the_real_cli_accepts_every_option_the_form_builds() {
        let exe = std::env::current_exe().unwrap();
        let name = if cfg!(windows) { "cli.exe" } else { "cli" };
        let Some(cli) = exe
            .ancestors()
            .nth(2)
            .map(|dir| dir.join(name))
            .filter(|p| p.exists())
        else {
            // CI builds the whole workspace, so a missing binary there is a real failure.
            assert!(
                std::env::var_os("CI").is_none(),
                "no `cli` binary next to the test binary"
            );
            eprintln!(
                "skipped: no `cli` binary next to the test binary (run `cargo build -p cli` first)"
            );
            return;
        };
        let output = std::env::temp_dir().join("arschloch-gui-parity.json");
        let form = Form {
            matches: 4,
            rounds: 1,
            threads: 1,
            bootstrap_resamples: 0,
            output: output.display().to_string(),
            explain: true,
            useful_passes: Some(1),
            skill_score: SkillScore::Both,
            strategies: [
                "lowest-legal",
                "adaptive:reading",
                "endgame-solver:max-unseen=2",
                "card-counter",
            ]
            .map(str::to_owned)
            .to_vec(),
            ..Form::default()
        };
        let run = |args: Vec<String>| {
            std::process::Command::new(&cli)
                .args(args)
                .output()
                .unwrap()
        };
        let ok = run(form.args());
        assert!(
            ok.status.success(),
            "{}",
            String::from_utf8_lossy(&ok.stderr)
        );
        let mut broken = form.args();
        broken.push("--no-such-option".to_owned());
        assert!(
            !run(broken).status.success(),
            "the negative control must fail"
        );
        let _ = std::fs::remove_file(output);
    }

    #[test]
    fn the_solver_toggle_adds_and_removes_the_suffix() {
        assert!(!has_solver("adaptive:reading"));
        assert!(has_solver("adaptive:reading+solver:8"));
        assert_eq!(
            with_solver("adaptive:reading", true),
            "adaptive:reading+solver"
        );
        assert_eq!(with_solver("adaptive+solver:8", false), "adaptive");
        assert_eq!(with_solver("adaptive+solver:8", true), "adaptive+solver:8");
        assert_eq!(with_solver("adaptive", false), "adaptive");
        assert_eq!(with_solver("", true), "", "no spec, no `+solver` alone");
        assert_eq!(with_solver("  ", true), "");
    }

    #[test]
    fn picking_a_genome_keeps_the_solver_toggle() {
        assert_eq!(
            split_solver("neat:a.json+solver:8"),
            ("neat:a.json", "+solver:8")
        );
        assert_eq!(split_solver("neat:a.json"), ("neat:a.json", ""));
        assert_eq!(
            with_genome("neat:a.json+solver:8", "/b.json"),
            "neat:/b.json+solver:8"
        );
        assert_eq!(with_genome("adaptive", "/b.json"), "neat:/b.json");
    }

    #[test]
    fn the_suffix_rule_is_the_one_of_sim() {
        // The table is sim's (`sim/src/solver/suffix_cases.txt`): spec|base|rest, `-` = none.
        for line in include_str!("../../sim/src/solver/suffix_cases.txt").lines() {
            let cells: Vec<&str> = line.split('|').collect();
            let (spec, base, rest) = (cells[0], cells[1], cells[2]);
            let got =
                solver_suffix_at(spec).map(|at| (&spec[..at], &spec[at + SOLVER_SUFFIX.len()..]));
            assert_eq!(got, (base != "-").then_some((base, rest)), "{spec}");
        }
    }

    #[test]
    fn one_strategy_per_seat_follows_the_player_count() {
        let mut form = Form::default();
        form.set_player_count(6);
        assert_eq!(form.strategies.len(), 6);
        form.strategies[5] = "lowest-legal".to_owned();
        form.set_player_count(3);
        assert_eq!(form.strategies.len(), 3);
        form.set_player_count(9);
        assert_eq!(form.player_count, 6, "clamped to the supported 3-6");
        assert_eq!(form.strategies.len(), 6);
    }

    #[test]
    fn optional_parts_only_appear_when_switched_on() {
        let form = Form {
            explain: true,
            useful_passes: Some(20),
            skill_score: SkillScore::Duplicate,
            ..Form::default()
        };
        let args = form.args();
        assert!(args.contains(&"--explain".to_owned()));
        let at = args.iter().position(|a| a == "--useful-passes").unwrap();
        assert_eq!(args[at + 1], "20");
        assert!(args.contains(&"--useful-pass-sample".to_owned()));
        let skill = args.iter().position(|a| a == "--skill-score").unwrap();
        assert_eq!(args[skill + 1], "duplicate");
        assert!(!Form::default().args().contains(&"--explain".to_owned()));
        assert!(!Form::default()
            .args()
            .contains(&"--useful-passes".to_owned()));
    }

    #[test]
    fn a_form_round_trips_through_json_for_presets() {
        let mut form = Form::default();
        form.set_player_count(5);
        form.seed = 42;
        form.strategies[0] = "endgame-solver:max-unseen=6".to_owned();
        let json = serde_json::to_string(&form).unwrap();
        assert_eq!(serde_json::from_str::<Form>(&json).unwrap(), form);
    }

    #[test]
    fn a_duplicate_skill_score_needs_matches_in_whole_groups() {
        let mut form = Form {
            skill_score: SkillScore::Duplicate,
            matches: 10, // 4 players: not a multiple of 4
            ..Form::default()
        };
        assert!(form.problem().unwrap().contains("multiple"));
        form.matches = 12;
        assert!(form.problem().is_none());
        form.skill_score = SkillScore::Off;
        form.matches = 10;
        assert!(form.problem().is_none());
        form.strategies[1] = "  ".to_owned();
        assert!(form.problem().unwrap().contains("seat 2"));
    }
}
