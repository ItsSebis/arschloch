//! The progress of the running command, read from the lines it prints.
//!
//! Commands print `progress: FRACTION TEXT` on stderr when the environment
//! variable `ARSCHLOCH_PROGRESS` is set (the window sets it for every job).
//! FRACTION is a number from 0 to 1; TEXT is shown next to the bar.

/// The environment variable that makes a command print progress lines.
pub const ENV: &str = "ARSCHLOCH_PROGRESS";

#[derive(Clone, Debug, PartialEq)]
pub struct Progress {
    /// 0.0 to 1.0.
    pub fraction: f32,
    pub text: String,
}

/// Reads one output line; `None` when it is not a progress line.
pub fn parse(line: &str) -> Option<Progress> {
    let rest = line.trim().strip_prefix("progress:")?.trim_start();
    let (number, text) = rest.split_once(' ').unwrap_or((rest, ""));
    let fraction: f32 = number.parse().ok().filter(|f: &f32| f.is_finite())?;
    Some(Progress {
        fraction: fraction.clamp(0.0, 1.0),
        text: text.trim().to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_progress_line_gives_fraction_and_text() {
        assert_eq!(
            parse("progress: 0.2500 generation 3/12, genomes 20/40"),
            Some(Progress {
                fraction: 0.25,
                text: "generation 3/12, genomes 20/40".to_owned()
            })
        );
        assert_eq!(parse("progress: 1").unwrap().text, "");
    }

    #[test]
    fn the_fraction_is_clamped_and_other_lines_are_not_progress() {
        assert!((parse("progress: 7 over").unwrap().fraction - 1.0).abs() < f32::EPSILON);
        assert!(parse("progress: -3 under").unwrap().fraction.abs() < f32::EPSILON);
        for line in [
            "",
            "generation 3",
            "progress:",
            "progress: soon",
            "progress: NaN x",
            "  gen progress: 0.5",
        ] {
            assert_eq!(parse(line), None, "{line:?}");
        }
    }

    /// The real `cli train` prints lines this module reads, with the variable
    /// the window sets, and prints none without it.
    #[test]
    fn the_real_trainer_prints_progress_lines_only_when_asked() {
        let Some(cli) = crate::testing::real_cli() else {
            return;
        };
        let dir = crate::testing::TempDir::new("progress");
        let run = |ask: bool, name: &str| {
            let mut command = std::process::Command::new(&cli);
            command
                .args(["train", "--out", &dir.0.join(name).to_string_lossy()])
                .args(["--population", "10", "--generations", "3"])
                .args(["--matches-per-genome", "2", "--rounds", "1"])
                .args(["--opponent", "lowest-legal", "--threads", "1"]);
            if ask {
                command.env(ENV, "1");
            } else {
                command.env_remove(ENV);
            }
            let out = command.output().unwrap();
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
            String::from_utf8_lossy(&out.stderr).into_owned()
                + &String::from_utf8_lossy(&out.stdout)
        };
        let asked = run(true, "a");
        let progress: Vec<Progress> = asked.lines().filter_map(parse).collect();
        assert!(progress.len() >= 3, "{asked}");
        assert!(
            progress.windows(2).all(|w| w[0].fraction <= w[1].fraction),
            "{progress:?}"
        );
        let last = progress.last().unwrap();
        assert!((last.fraction - 1.0).abs() < 1e-6, "{last:?}");
        assert!(last.text.contains("generation 3/3"), "{last:?}");
        assert!(run(false, "b").lines().all(|l| parse(l).is_none()));
    }
}
