//! `cli analyse-run PATH...`: where did a training run stop improving, and
//! why? A plain-language report from the `events.jsonl` the trainer wrote.
//! The numbers come from `sim::training::analyse`; this file only reads
//! the paths and lays the result out.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::Parser;
use serde::Serialize;
use sim::training::{analyse, load_runs, qualify_labels, RunAnalysis, RunData, Slope};

/// Analyse training runs.
#[derive(Parser, Debug)]
#[command(
    name = "cli analyse-run",
    about = "Say where training runs stopped improving and why (from events.jsonl)"
)]
pub struct AnalyseRunArgs {
    /// An `events.jsonl` file, a run directory, or a set directory (one run
    /// per subdirectory). Several paths compare several runs.
    /// With several paths (or equal names) runs are shown as `parent/name`.
    #[arg(required = true)]
    pub paths: Vec<PathBuf>,

    /// Also write the analyses as a JSON array of `{label, analysis}`.
    #[arg(long, value_name = "PATH")]
    pub json: Option<PathBuf>,
}

#[derive(Serialize)]
struct Labelled<'a> {
    label: &'a str,
    analysis: &'a RunAnalysis,
}

pub fn run(raw_args: impl Iterator<Item = String>) -> anyhow::Result<()> {
    let args =
        AnalyseRunArgs::parse_from(std::iter::once("cli analyse-run".to_owned()).chain(raw_args));
    let mut runs: Vec<RunData> = Vec::new();
    for path in &args.paths {
        // The error already names the path.
        let loaded = load_runs(path)?;
        for skipped in &loaded.skipped_runs {
            eprintln!("warning: run left out, could not be read: {skipped}");
        }
        runs.extend(loaded.runs);
    }
    qualify_labels(&mut runs, args.paths.len() > 1);
    let analyses = runs
        .iter()
        .map(|r| {
            anyhow::ensure!(
                !r.generations.is_empty(),
                "{}: no generation events to analyse",
                r.label
            );
            Ok(analyse(r))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    for (run, analysis) in runs.iter().zip(&analyses) {
        println!("{}", render_run(&run.label, analysis));
    }
    if runs.len() > 1 {
        let rows: Vec<(&str, &RunAnalysis)> = runs
            .iter()
            .map(|r| r.label.as_str())
            .zip(&analyses)
            .collect();
        println!("{}", render_comparison(&rows));
    }
    if let Some(path) = &args.json {
        write_json(path, &runs, &analyses)?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

fn write_json(path: &Path, runs: &[RunData], analyses: &[RunAnalysis]) -> anyhow::Result<()> {
    let items: Vec<Labelled<'_>> = runs
        .iter()
        .zip(analyses)
        .map(|(r, a)| Labelled {
            label: &r.label,
            analysis: a,
        })
        .collect();
    let text = serde_json::to_string_pretty(&items)?;
    std::fs::write(path, text).with_context(|| {
        format!(
            "the analysis was printed above but the JSON could not be written to {}",
            path.display()
        )
    })
}

fn score(s: Option<sim::training::analysis::ScoreSummary>) -> String {
    s.map_or_else(
        || "-".to_owned(),
        |s| format!("{:.3} +- {:.3}", s.mean, s.std_error),
    )
}

/// `value [low, high]` per 100 generations, or `-`.
fn slope(s: Option<Slope>) -> String {
    s.map_or_else(
        || "-".to_owned(),
        |s| format!("{:+.3} [{:+.3}, {:+.3}]", s.value, s.low, s.high),
    )
}

fn slope_value(s: Option<Slope>) -> String {
    s.map_or_else(|| "-".to_owned(), |s| format!("{:+.3}", s.value))
}

fn mean_or_dash(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_owned(), |v| format!("{v:.3}"))
}

fn code_name(code: sim::training::analysis::VerdictCode) -> String {
    serde_json::to_value(code)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_else(|| format!("{code:?}"))
}

fn gap(a: &RunAnalysis) -> String {
    a.curse.reeval_minus_heldout.map_or_else(
        || "-".to_owned(),
        |g| match g.in_combined_se {
            Some(se) => format!("{:+.3} ({se:+.1} combined SE)", g.value),
            None => format!("{:+.3}", g.value),
        },
    )
}

/// Left-aligns every cell to its column width.
fn table(header: &[&str], rows: &[Vec<String>]) -> String {
    let widths: Vec<usize> = (0..header.len())
        .map(|c| {
            rows.iter()
                .map(|r| r[c].chars().count())
                .chain([header[c].chars().count()])
                .max()
                .unwrap_or(0)
        })
        .collect();
    let line = |cells: Vec<&str>| {
        let padded: Vec<String> = cells
            .iter()
            .zip(&widths)
            .map(|(c, w)| format!("{c:<w$}"))
            .collect();
        format!("{}\n", padded.join("  ").trim_end())
    };
    let mut out = line(header.to_vec());
    for row in rows {
        out.push_str(&line(row.iter().map(String::as_str).collect()));
    }
    out
}

fn facts(a: &RunAnalysis) -> Vec<(&'static str, String)> {
    let best = a
        .best_generation
        .map_or_else(|| "-".to_owned(), |g| g.to_string());
    let change_sd = a
        .noise
        .change_sd
        .map_or_else(|| "-".to_owned(), |v| format!("{v:.3}"));
    let top_bin = a
        .population
        .top_bin_share
        .map_or_else(|| "-".to_owned(), |v| format!("{:.0}%", v * 100.0));
    vec![
        ("generations", a.generations.to_string()),
        ("best generation", best),
        ("final re-evaluation", score(a.final_reeval)),
        ("held-out", score(a.final_heldout)),
        (
            "stall generation",
            format!(
                "{} (last generation {}), smoothed peak {:.3} at generation {}{}",
                a.stall_generation,
                a.last_generation,
                a.peak_smoothed,
                a.peak_generation,
                if a.stall_fallback {
                    ", final smoothed value below the peak band"
                } else {
                    ""
                }
            ),
        ),
        (
            "noise",
            format!(
                "one evaluation +- {:.3}, generation-to-generation sd {change_sd}",
                a.noise.single_se
            ),
        ),
        ("trend, whole run /100 gen", slope(a.trend.overall)),
        ("trend, last third /100 gen", slope(a.trend.last_third)),
        ("winner's curse (re-eval - held-out)", gap(a)),
        (
            "train fitness - re-eval (mean)",
            format!("{:+.3}", a.curse.train_minus_reeval),
        ),
        (
            "hidden nodes / connections",
            format!(
                "{} -> {} / {} -> {}",
                a.growth.first_hidden_nodes,
                a.growth.last_hidden_nodes,
                a.growth.first_connections,
                a.growth.last_connections
            ),
        ),
        (
            "species / threshold",
            format!(
                "{} -> {} / {:.2} -> {:.2}",
                a.population.species_first,
                a.population.species_last,
                a.population.threshold_first,
                a.population.threshold_last
            ),
        ),
        ("top fitness bin (last third)", top_bin),
    ]
}

fn render_run(label: &str, a: &RunAnalysis) -> String {
    let mut out = format!("== {label} ==\n");
    let facts = facts(a);
    let width = facts.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
    for (key, value) in &facts {
        let _ = writeln!(out, "  {key:<width$}  {value}");
    }
    if let Some(r) = &a.racing {
        let share = r.extra_share.map_or_else(String::new, |s| {
            format!(", {:.1}% of all matches", s * 100.0)
        });
        let _ = writeln!(
            out,
            "  racing: {} generations, {} extra matches{share}, mean fitness shift {:+.3}",
            r.generations, r.extra_matches, r.mean_shift
        );
    }
    if let Some(t) = &a.timings {
        let stages = [
            ("training", t.training_evaluation),
            ("selection", t.champion_selection),
            ("reproduction", t.speciation_and_reproduction),
            ("re-eval mixed", t.reevaluation_mixed),
            ("re-eval per opponent", t.reevaluation_per_opponent),
            ("hall of fame", t.hall_of_fame),
            ("confirmation", t.confirmation),
            ("decision sample", t.decision_sample),
            ("files", t.checkpoint_and_files),
        ];
        let parts: Vec<String> = stages
            .iter()
            .map(|(name, secs)| format!("{name} {secs:.1}s"))
            .collect();
        let _ = writeln!(out, "  seconds per stage: {}", parts.join(", "));
    }
    if a.skipped_lines > 0 {
        let _ = writeln!(
            out,
            "  note: {} line(s) were not events and were skipped (old format or truncated)",
            a.skipped_lines
        );
    }
    if !a.opponents.is_empty() {
        let rows: Vec<Vec<String>> = a
            .opponents
            .iter()
            .map(|o| {
                vec![
                    o.name.clone(),
                    o.class
                        .map_or_else(|| "-".to_owned(), |c| format!("{c:?}").to_lowercase()),
                    mean_or_dash(o.first_third_mean),
                    mean_or_dash(o.last_third_mean),
                    slope(o.last_third_slope),
                    slope(o.overall_slope),
                ]
            })
            .collect();
        out.push_str("\nPer opponent (champion score; class from the last third):\n");
        out.push_str(&table(
            &[
                "opponent",
                "class",
                "first third",
                "last third",
                "last third /100 gen",
                "whole run /100 gen",
            ],
            &rows,
        ));
    }
    out.push_str("\nVerdict:\n");
    for v in &a.verdict {
        let _ = writeln!(out, "  [{}] {}", code_name(v.code), v.message);
    }
    out
}

fn render_comparison(rows: &[(&str, &RunAnalysis)]) -> String {
    let cells: Vec<Vec<String>> = rows
        .iter()
        .map(|(label, a)| {
            vec![
                (*label).to_owned(),
                a.generations.to_string(),
                a.best_generation
                    .map_or_else(|| "-".to_owned(), |g| g.to_string()),
                a.stall_generation.to_string(),
                a.final_reeval
                    .map_or_else(|| "-".to_owned(), |s| format!("{:.3}", s.mean)),
                slope_value(a.trend.last_third),
                gap(a),
                a.verdict
                    .iter()
                    .map(|v| code_name(v.code))
                    .collect::<Vec<_>>()
                    .join(","),
            ]
        })
        .collect();
    let mut out = String::from("== comparison ==\n");
    out.push_str(&table(
        &[
            "run",
            "gens",
            "best gen",
            "stall gen",
            "final re-eval",
            "last-third /100",
            "curse gap",
            "verdict",
        ],
        &cells,
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_least_one_path_is_required_and_json_is_optional() {
        assert!(AnalyseRunArgs::try_parse_from(["cli analyse-run"]).is_err());
        let args = AnalyseRunArgs::try_parse_from(["x", "a", "b", "--json", "o.json"]).unwrap();
        assert_eq!(args.paths.len(), 2);
        assert_eq!(args.json, Some(PathBuf::from("o.json")));
    }

    #[test]
    fn slopes_and_missing_values_format_compactly() {
        let s = Slope {
            value: 0.5,
            low: -0.25,
            high: 1.0,
        };
        assert_eq!(slope(Some(s)), "+0.500 [-0.250, +1.000]");
        assert_eq!(slope_value(Some(s)), "+0.500");
        assert_eq!((slope(None), slope_value(None)), ("-".into(), "-".into()));
    }

    #[test]
    fn tables_pad_columns_to_the_widest_cell() {
        let t = table(
            &["a", "bb"],
            &[vec!["xxx".into(), "y".into()], vec!["z".into(), "w".into()]],
        );
        assert_eq!(t, "a    bb\nxxx  y\nz    w\n");
    }

    #[test]
    fn verdict_codes_print_in_snake_case() {
        assert_eq!(
            code_name(sim::training::analysis::VerdictCode::NoiseLimited),
            "noise_limited"
        );
    }
}
