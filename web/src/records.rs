//! The human's results, kept as one JSON line per finished game so the
//! file survives crashes and can be read (or edited) with ordinary tools.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::json;
use sim::training::role_score;
use sim::{roles_for_player_count, Role};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub finished_unix: u64,
    pub player_count: u8,
    pub deck: String,
    pub duplicate_rule: String,
    /// `final` or `free`. Games recorded before the rule existed were played
    /// under `free`.
    #[serde(default = "legacy_pass_rule")]
    pub pass_rule: String,
    /// `forced` or `free`; records from before the rule existed read as `free`.
    #[serde(default = "legacy_pass_rule")]
    pub exchange_rule: String,
    pub rounds: usize,
    /// The opponents' labels, in seat order (the human's seat skipped).
    pub opponents: Vec<String>,
    /// The human's role in each round.
    pub roles: Vec<Role>,
    pub score: f64,
}

/// Locks `mutex`, carrying on after a panic elsewhere (a poisoned lock must
/// not take every later game down with it).
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

pub struct RecordStore {
    path: Option<PathBuf>,
    /// Set (once per failure) when the file cannot be written; the game
    /// itself is never affected.
    write_error: Mutex<Option<String>>,
    /// Serialises appends so concurrent finishes never interleave a line.
    lock: Mutex<()>,
}

fn legacy_pass_rule() -> String {
    "free".to_owned()
}

impl Record {
    /// Whether the record describes a possible game: a damaged line that
    /// still parses must not be able to crash the summary.
    fn is_valid(&self) -> bool {
        let Some(table) = roles_for_player_count(self.player_count) else {
            return false;
        };
        self.score.is_finite()
            && !self.roles.is_empty()
            && self.roles.iter().all(|r| table.contains(r))
    }
}

impl RecordStore {
    /// `None` keeps nothing on disk (results are only shown for the
    /// games recorded since the server started, in memory).
    #[must_use]
    pub fn new(path: Option<PathBuf>) -> Self {
        Self {
            path,
            write_error: Mutex::new(None),
            lock: Mutex::new(()),
        }
    }

    pub fn append(&self, record: &Record) {
        let _guard = lock(&self.lock);
        let Some(path) = &self.path else {
            return;
        };
        let Ok(line) = serde_json::to_string(record) else {
            return;
        };
        let result = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .and_then(|mut file| writeln!(file, "{line}"));
        if let Err(error) = result {
            *lock(&self.write_error) = Some(format!("cannot write {}: {error}", path.display()));
        }
    }

    /// Every readable record, oldest first; damaged lines are skipped.
    #[must_use]
    pub fn read_all(&self) -> Vec<Record> {
        let _guard = lock(&self.lock);
        let Some(path) = &self.path else {
            return Vec::new();
        };
        std::fs::read_to_string(path)
            .map(|text| {
                text.lines()
                    .filter_map(|line| serde_json::from_str::<Record>(line).ok())
                    .filter(Record::is_valid)
                    .collect()
            })
            .unwrap_or_default()
    }

    #[must_use]
    pub fn write_error(&self) -> Option<String> {
        lock(&self.write_error).clone()
    }

    /// The summary the page shows.
    #[must_use]
    pub fn summary(&self) -> serde_json::Value {
        let records = self.read_all();
        let mut by_opponent: BTreeMap<(String, String, String), (usize, f64)> = BTreeMap::new();
        let mut by_table: BTreeMap<(u8, Vec<String>, String, String), TableStats> = BTreeMap::new();
        for record in &records {
            let mut names = record.opponents.clone();
            names.sort();
            let stats = by_table
                .entry((
                    record.player_count,
                    names.clone(),
                    record.pass_rule.clone(),
                    record.exchange_rule.clone(),
                ))
                .or_default();
            stats.add(record);
            for name in names.iter().collect::<std::collections::BTreeSet<_>>() {
                let entry = by_opponent
                    .entry((
                        name.clone(),
                        record.pass_rule.clone(),
                        record.exchange_rule.clone(),
                    ))
                    .or_default();
                entry.0 += 1;
                entry.1 += record.score;
            }
        }
        #[allow(clippy::cast_precision_loss)] // game counts are small
        let opponents: Vec<_> = by_opponent
            .iter()
            .map(|((name, rule, exchange), (games, total))| {
                json!({"opponent": name, "pass_rule": rule, "exchange_rule": exchange, "games": games, "mean_score": total / *games as f64})
            })
            .collect();
        let tables: Vec<_> = by_table
            .iter()
            .map(|((players, names, rule, exchange), stats)| {
                stats.to_json(*players, names, rule, exchange)
            })
            .collect();
        let recent: Vec<_> = records.iter().rev().take(20).collect();
        json!({
            "total_games": records.len(),
            "write_error": self.write_error(),
            "by_opponent": opponents,
            "by_table": tables,
            "recent": recent,
        })
    }
}

#[derive(Default)]
struct TableStats {
    games: usize,
    rounds: usize,
    score_sum: f64,
    president: usize,
    last: usize,
}

impl TableStats {
    fn add(&mut self, record: &Record) {
        self.games += 1;
        self.score_sum += record.score;
        self.rounds += record.roles.len();
        for &role in &record.roles {
            if role_score(role, record.player_count) >= 1.0 {
                self.president += 1;
            }
            if role_score(role, record.player_count) <= -1.0 {
                self.last += 1;
            }
        }
    }

    #[allow(clippy::cast_precision_loss)] // counts are small
    fn to_json(
        &self,
        players: u8,
        names: &[String],
        rule: &str,
        exchange: &str,
    ) -> serde_json::Value {
        let rounds = self.rounds.max(1) as f64;
        json!({
            "players": players,
            "opponents": names,
            "pass_rule": rule,
            "exchange_rule": exchange,
            "games": self.games,
            "mean_score": self.score_sum / self.games.max(1) as f64,
            "president_rate": self.president as f64 / rounds,
            "last_rate": self.last as f64 / rounds,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "arschloch-records-{name}-{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        path
    }

    fn record(score: f64, roles: Vec<Role>, opponents: &[&str]) -> Record {
        Record {
            finished_unix: 1,
            player_count: 4,
            deck: "single".into(),
            duplicate_rule: "first_dealt_wins".into(),
            pass_rule: "free".into(),
            exchange_rule: "free".into(),
            rounds: roles.len(),
            opponents: opponents.iter().map(|s| (*s).to_owned()).collect(),
            roles,
            score,
        }
    }

    #[test]
    fn records_round_trip_and_damaged_lines_are_skipped() {
        let path = temp("roundtrip");
        let store = RecordStore::new(Some(path.clone()));
        store.append(&record(
            0.5,
            vec![Role::President, Role::Vize],
            &["A", "B", "C"],
        ));
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"{ not json\n\n")
            .unwrap();
        store.append(&record(-1.0, vec![Role::Arschloch], &["A", "B", "C"]));
        let all = store.read_all();
        assert_eq!(all.len(), 2);
        assert!((all[1].score + 1.0).abs() < f64::EPSILON);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn a_parseable_but_impossible_record_is_skipped_like_a_damaged_one() {
        let path = temp("impossible");
        let store = RecordStore::new(Some(path.clone()));
        store.append(&record(0.5, vec![Role::President], &["A", "B", "C"]));
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        // A 3-player table has no Vize; 9 players do not exist; no roles at all.
        let mut bad = record(0.0, vec![Role::Vize], &["A", "B"]);
        bad.player_count = 3;
        let mut nine = record(0.0, vec![Role::Vize], &["A"]);
        nine.player_count = 9;
        let empty = record(0.0, vec![], &["A", "B", "C"]);
        for line in [&bad, &nine, &empty] {
            writeln!(file, "{}", serde_json::to_string(line).unwrap()).unwrap();
        }
        drop(file);
        assert_eq!(store.read_all().len(), 1);
        assert_eq!(store.summary()["total_games"], 1);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn games_under_different_pass_rules_are_never_mixed_and_old_records_read_as_free() {
        let path = temp("rules");
        let store = RecordStore::new(Some(path.clone()));
        let mut final_game = record(1.0, vec![Role::President], &["A", "B", "C"]);
        final_game.pass_rule = "final".into();
        let free_game = record(-1.0, vec![Role::Arschloch], &["A", "B", "C"]);
        store.append(&final_game);
        store.append(&free_game);
        // A line written before the field existed.
        let mut old: serde_json::Value = serde_json::to_value(&free_game).unwrap();
        old.as_object_mut().unwrap().remove("pass_rule");
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        writeln!(file, "{old}").unwrap();
        drop(file);
        let all = store.read_all();
        assert_eq!(all.len(), 3);
        assert_eq!(
            all[2].pass_rule, "free",
            "old records were played under free"
        );
        let summary = store.summary();
        let tables = summary["by_table"].as_array().unwrap();
        assert_eq!(tables.len(), 2, "one table per rule: {tables:?}");
        let by_rule = |rule: &str| tables.iter().find(|t| t["pass_rule"] == rule).unwrap();
        assert_eq!(by_rule("final")["games"], 1);
        assert_eq!(by_rule("free")["games"], 2);
        for o in summary["by_opponent"].as_array().unwrap() {
            assert!(o["pass_rule"] == "final" || o["pass_rule"] == "free");
        }
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn games_under_different_exchange_rules_are_never_mixed() {
        let path = temp("exchange");
        let store = RecordStore::new(Some(path.clone()));
        let mut forced = record(1.0, vec![Role::President], &["A", "B", "C"]);
        forced.pass_rule = "final".into();
        forced.exchange_rule = "forced".into();
        let mut free_exchange = forced.clone();
        free_exchange.exchange_rule = "free".into();
        store.append(&forced);
        store.append(&free_exchange);
        let summary = store.summary();
        assert_eq!(summary["by_table"].as_array().unwrap().len(), 2);
        assert_eq!(summary["by_opponent"].as_array().unwrap().len(), 6);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn the_summary_aggregates_by_opponent_and_by_table() {
        let path = temp("summary");
        let store = RecordStore::new(Some(path.clone()));
        store.append(&record(
            1.0,
            vec![Role::President, Role::President],
            &["Champ", "Low", "Low"],
        ));
        store.append(&record(
            -1.0,
            vec![Role::Arschloch, Role::Arschloch],
            &["Low", "Champ", "Low"],
        ));
        store.append(&record(0.0, vec![Role::Vize], &["Other", "Low", "Low"]));
        let summary = store.summary();
        assert_eq!(summary["total_games"], 3);
        let champ = summary["by_opponent"]
            .as_array()
            .unwrap()
            .iter()
            .find(|o| o["opponent"] == "Champ")
            .unwrap();
        assert_eq!(champ["games"], 2);
        assert!(champ["mean_score"].as_f64().unwrap().abs() < 1e-9);
        // The same opponents in a different seat order are one table.
        let tables = summary["by_table"].as_array().unwrap();
        assert_eq!(tables.len(), 2);
        let table = tables.iter().find(|t| t["games"] == 2).unwrap();
        assert!((table["president_rate"].as_f64().unwrap() - 0.5).abs() < 1e-9);
        assert!((table["last_rate"].as_f64().unwrap() - 0.5).abs() < 1e-9);
        assert_eq!(summary["recent"].as_array().unwrap().len(), 3);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn an_unwritable_path_reports_once_and_never_panics() {
        let store = RecordStore::new(Some(
            std::env::temp_dir().join("no-such-dir-xyz").join("r.jsonl"),
        ));
        store.append(&record(0.0, vec![Role::Vize], &["A"]));
        assert!(store.write_error().unwrap().contains("cannot write"));
        assert_eq!(store.summary()["total_games"], 0);
    }

    #[test]
    fn concurrent_appends_never_interleave_lines() {
        let path = temp("concurrent");
        let store = std::sync::Arc::new(RecordStore::new(Some(path.clone())));
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let store = std::sync::Arc::clone(&store);
                std::thread::spawn(move || {
                    for _ in 0..20 {
                        store.append(&record(f64::from(i), vec![Role::Vize; 5], &["A", "B", "C"]));
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(store.read_all().len(), 160);
        std::fs::remove_file(path).ok();
    }
}
