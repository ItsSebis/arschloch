//! Writing a batch run's results (matches + aggregated statistics) as a
//! JSON file. The shape mirrors `sim`'s existing types directly — no new
//! schema — so a later phase reading this file (see docs/ARCHITECTURE.md,
//! "web") sees exactly `sim::MatchResult`/`sim::Statistics` as-is.

use std::fs::File;
use std::path::Path;

use anyhow::Context;

#[derive(serde::Serialize)]
struct RunOutput<'a> {
    matches: &'a [sim::MatchResult],
    statistics: &'a sim::Statistics,
}

/// Writes `matches` and `statistics` to `path` as pretty-printed JSON.
///
/// # Errors
///
/// Returns an error if `path` can't be created or the results can't be
/// serialized (serialization failure is not expected in practice — every
/// field of `MatchResult`/`Statistics` is a plain serializable type — but
/// `serde_json::to_writer_pretty` returns a `Result`, so this surfaces it
/// rather than unwrapping).
pub fn write_json_output(
    path: &Path,
    matches: &[sim::MatchResult],
    statistics: &sim::Statistics,
) -> anyhow::Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    serde_json::to_writer_pretty(
        file,
        &RunOutput {
            matches,
            statistics,
        },
    )
    .context("failed to serialize results")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::Role;
    use std::collections::BTreeMap;

    fn sample_result() -> sim::MatchResult {
        sim::MatchResult {
            player_count: 2,
            strategy_names: vec!["LowestLegal".to_string(), "GreedyHighest".to_string()],
            role_history: vec![vec![Role::President, Role::Arschloch]],
            trick_count: 3,
            pass_count: 1,
            voluntary_pass_count: 0,
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
        }
    }

    #[test]
    fn write_json_output_writes_parseable_json_with_expected_keys() {
        let path =
            std::env::temp_dir().join(format!("arschloch-output-test-{}.json", std::process::id()));
        let matches = vec![sample_result()];
        let statistics = sample_statistics();

        write_json_output(&path, &matches, &statistics).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert!(value.get("matches").is_some());
        assert!(value.get("statistics").is_some());

        std::fs::remove_file(&path).ok();
    }
}
