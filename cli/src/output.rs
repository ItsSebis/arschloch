//! Writing a batch run's results (matches + aggregated statistics) as a
//! JSON file. The shape mirrors `sim`'s existing types directly — no new
//! schema — so a later phase reading this file (see docs/ARCHITECTURE.md,
//! "web") sees exactly `sim::MatchResult`/`sim::Statistics` as-is.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::Context;

/// The `extended` key: the extended statistics, the useful-pass numbers
/// merged into each strategy's entry when that analysis ran, and, when a
/// skill score was asked for, the luck/skill reports under `skill`.
#[derive(serde::Serialize)]
struct ExtendedOutput<'a> {
    by_strategy: BTreeMap<&'a String, StrategyOutput<'a>>,
    by_seat: &'a [sim::extended_stats::SeatStats],
    role_retention_intervals:
        &'a BTreeMap<String, BTreeMap<engine::Role, sim::extended_stats::Interval>>,
    voluntary_pass_rate_intervals: &'a BTreeMap<String, sim::extended_stats::Interval>,
    #[serde(skip_serializing_if = "Option::is_none")]
    skill: Option<&'a crate::batch::SkillOutput>,
}

#[derive(serde::Serialize)]
struct StrategyOutput<'a> {
    #[serde(flatten)]
    stats: &'a sim::extended_stats::StrategyStats,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    useful_passes: Option<UsefulOutput<'a>>,
}

/// A strategy without any sampled pass still gets its keys (null values).
#[derive(serde::Serialize)]
struct UsefulOutput<'a> {
    voluntary_passes_total: u64,
    sampled_voluntary_passes: u64,
    useful_passes: u64,
    useful_pass_share: Option<&'a sim::useful_passes::ShareEstimate>,
    useful_pass_gain: Option<&'a sim::extended_stats::Estimate>,
}

fn extended_output<'a>(
    extended: &'a sim::extended_stats::ExtendedStatistics,
    skill: Option<&'a crate::batch::SkillOutput>,
    useful: Option<&'a sim::useful_passes::UsefulPassStats>,
) -> ExtendedOutput<'a> {
    let by_strategy = extended
        .by_strategy
        .iter()
        .map(|(name, stats)| {
            let useful_passes = useful.map(|u| match u.by_strategy.get(name) {
                Some(s) => UsefulOutput {
                    voluntary_passes_total: s.voluntary_passes_total,
                    sampled_voluntary_passes: s.sampled_voluntary_passes,
                    useful_passes: s.useful,
                    useful_pass_share: s.useful_pass_share.as_ref(),
                    useful_pass_gain: s.useful_pass_gain.as_ref(),
                },
                None => UsefulOutput {
                    voluntary_passes_total: 0,
                    sampled_voluntary_passes: 0,
                    useful_passes: 0,
                    useful_pass_share: None,
                    useful_pass_gain: None,
                },
            });
            (
                name,
                StrategyOutput {
                    stats,
                    useful_passes,
                },
            )
        })
        .collect();
    ExtendedOutput {
        by_strategy,
        by_seat: &extended.by_seat,
        role_retention_intervals: &extended.role_retention_intervals,
        voluntary_pass_rate_intervals: &extended.voluntary_pass_rate_intervals,
        skill,
    }
}

/// `matches` and `statistics` are the keys every version of the file has;
/// `extended` and `statistics_catalog` are the additive Phase 12 keys.
#[derive(serde::Serialize)]
struct RunOutput<'a> {
    matches: &'a [sim::MatchResult],
    statistics: &'a sim::Statistics,
    extended: ExtendedOutput<'a>,
    statistics_catalog: &'static [sim::stats_catalog::StatInfo],
}

/// Creates (or truncates) the file at `path`, failing fast before any
/// batch simulation runs rather than after — a bad `--output` path
/// should not cost the caller a completed run's worth of computed
/// results.
///
/// # Errors
///
/// Returns an error if `path` can't be created (e.g. its parent
/// directory doesn't exist, or permissions are denied).
pub fn create_output_file(path: &Path) -> anyhow::Result<File> {
    File::create(path).with_context(|| format!("failed to create {}", path.display()))
}

/// Writes `matches`, `statistics`, the extended statistics (with the skill
/// reports, if any) and the statistics catalogue to `file` as
/// pretty-printed JSON, buffering writes and flushing explicitly so a late I/O error isn't
/// silently lost when the writer drops.
///
/// # Errors
///
/// Returns an error if the results can't be serialized (not expected in
/// practice — every field of `MatchResult`/`Statistics` is a plain
/// serializable type — but `serde_json::to_writer_pretty` returns a
/// `Result`, so this surfaces it rather than unwrapping) or if the final
/// flush fails.
pub fn write_json_output(
    file: File,
    matches: &[sim::MatchResult],
    statistics: &sim::Statistics,
    extended: &sim::extended_stats::ExtendedStatistics,
    skill: Option<&crate::batch::SkillOutput>,
    useful: Option<&sim::useful_passes::UsefulPassStats>,
) -> anyhow::Result<()> {
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(
        &mut writer,
        &RunOutput {
            matches,
            statistics,
            extended: extended_output(extended, skill, useful),
            statistics_catalog: sim::stats_catalog::catalog(),
        },
    )
    .context("failed to serialize results")?;
    writer.flush().context("failed to flush results file")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::Role;
    use std::collections::BTreeMap;

    fn sample_result() -> sim::MatchResult {
        sim::MatchResult {
            player_count: 3,
            strategy_names: vec![
                "LowestLegal".to_string(),
                "GreedyHighest".to_string(),
                "RandomLegal".to_string(),
            ],
            role_history: vec![vec![Role::President, Role::Dorftrottel, Role::Arschloch]],
            trick_count: 3,
            pass_counts: vec![1, 0, 0],
            voluntary_pass_counts: vec![0, 0, 0],
            first_hand_features: None,
        }
    }

    fn sample_statistics() -> sim::Statistics {
        let mut president_count = BTreeMap::new();
        president_count.insert(Role::President, 1);
        let mut role_counts = BTreeMap::new();
        role_counts.insert("LowestLegal".to_string(), president_count);

        sim::Statistics {
            matches_played: 1,
            role_counts_by_strategy: role_counts,
            voluntary_pass_rate: 0.0,
            voluntary_pass_rate_by_strategy: BTreeMap::new(),
            role_retention_by_strategy: BTreeMap::new(),
            first_round_placement_variance_by_strategy: BTreeMap::new(),
        }
    }

    #[test]
    fn write_json_output_writes_parseable_json_with_expected_keys() {
        let path =
            std::env::temp_dir().join(format!("arschloch-output-test-{}.json", std::process::id()));
        let matches = vec![sample_result()];
        let statistics = sample_statistics();

        let file = create_output_file(&path).unwrap();
        let extended = sim::extended_stats::aggregate_extended(&matches);
        write_json_output(file, &matches, &statistics, &extended, None, None).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert!(value.get("matches").is_some());
        assert!(value.get("statistics").is_some());
        assert!(value.get("extended").is_some());
        assert!(value["extended"].get("skill").is_none());
        let by_strategy = value["extended"]["by_strategy"].as_object().unwrap();
        assert!(by_strategy
            .values()
            .all(|s| s.get("useful_pass_share").is_none()));
        assert!(value["statistics_catalog"]
            .as_array()
            .is_some_and(|c| !c.is_empty()));

        std::fs::remove_file(&path).ok();
    }

    /// Whether `path` (the catalogue's `json_path` syntax: `.` separates
    /// segments, `<name>` is every key of an object, `[]` every element of
    /// an array, `{a,b}` several fields) resolves in `value`. Every key or
    /// element must carry the rest of the path; a null leaf counts.
    fn resolves(value: &serde_json::Value, path: &str) -> bool {
        let Some((head, rest)) = path.split_once('.') else {
            return leaf(value, path);
        };
        match head {
            "[]" => value
                .as_array()
                .is_some_and(|a| !a.is_empty() && a.iter().all(|v| resolves(v, rest))),
            h if h.starts_with('<') => value
                .as_object()
                .is_some_and(|o| !o.is_empty() && o.values().all(|v| resolves(v, rest))),
            h => value.get(h).is_some_and(|v| resolves(v, rest)),
        }
    }

    fn leaf(value: &serde_json::Value, last: &str) -> bool {
        if last.starts_with('<') {
            return value.as_object().is_some_and(|o| !o.is_empty());
        }
        match last.strip_prefix('{').and_then(|l| l.strip_suffix('}')) {
            Some(fields) => fields.split(',').all(|f| value.get(f).is_some()),
            None => value.get(last).is_some(),
        }
    }

    #[test]
    fn the_path_syntax_resolves_as_documented() {
        let v = serde_json::json!({"a": {"x": {"k": 1, "m": null}, "y": {"k": 2, "m": 3}},
                                    "l": [{"v": 1}, {"v": 2}], "e": []});
        assert!(resolves(&v, "a.<n>.{k,m}") && resolves(&v, "l.[].v") && resolves(&v, "a.x.m"));
        assert!(!resolves(&v, "a.<n>.z") && !resolves(&v, "l.[].w"));
        assert!(!resolves(&v, "e.[].v") && !resolves(&v, "q"));
    }

    /// Every catalogued statistic must be where the catalogue says, in the
    /// JSON of a real run with every skill mode and the useful-pass
    /// analysis on (their keys only exist with their flags).
    #[test]
    fn every_catalogue_json_path_resolves_in_a_real_run() {
        let strategies: Vec<std::sync::Arc<dyn sim::Strategy>> = vec![
            std::sync::Arc::new(sim::LowestLegal),
            std::sync::Arc::new(sim::GreedyHighest),
            std::sync::Arc::new(sim::CardCounter),
        ];
        let configs: Vec<sim::MatchConfig> = (0..24)
            .map(|seed| sim::MatchConfig {
                player_count: 3,
                deck_variant: sim::DeckVariant::Single,
                duplicate_rule: sim::DuplicateRule::FirstDealtWins,
                rounds: 3,
                seed,
                pass_rule: sim::PassRule::default(),
                exchange_rule: sim::ExchangeRule::default(),
            })
            .collect();
        let run =
            crate::batch::run(crate::args::SkillScoreArg::Both, &configs, &strategies).unwrap();
        let useful = sim::useful_passes::analyse_useful_passes(
            &configs,
            &strategies,
            &sim::useful_passes::UsefulPassOptions {
                rollouts: 2,
                sample_fraction: 1.0,
                margin: 0.0,
                seed: 1,
            },
        );
        let statistics = sim::aggregate(&run.results);
        let extended = sim::extended_stats::aggregate_extended(&run.results);

        let path = std::env::temp_dir().join(format!(
            "arschloch-catalog-paths-{}.json",
            std::process::id()
        ));
        let file = create_output_file(&path).unwrap();
        write_json_output(
            file,
            &run.results,
            &statistics,
            &extended,
            run.skill.as_ref(),
            Some(&useful),
        )
        .unwrap();
        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        std::fs::remove_file(&path).ok();

        let mut checked = 0;
        for stat in sim::stats_catalog::catalog() {
            for path in stat.json_path.split("; ") {
                assert!(
                    resolves(&value, path),
                    "{}: `{path}` not in the JSON",
                    stat.id
                );
                checked += 1;
            }
        }
        assert!(checked >= 15, "only {checked} paths checked");
        let catalog = value["statistics_catalog"].as_array().unwrap();
        assert_eq!(catalog.len(), sim::stats_catalog::catalog().len());
    }
}
