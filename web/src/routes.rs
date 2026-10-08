//! The dashboard's HTTP routes as a pure function of (method, target):
//! testable without sockets.
//!
//! - `GET /`, `/app.js`, `/lib.js`, `/style.css`: the embedded page;
//! - `GET /api/state`: run settings, progress and whether it finished;
//! - `GET /api/events?since=N`: generation events after generation `N`;
//! - `GET /api/genome/N|best`: a champion genome file;
//! - `GET /api/decisions/N`: the recorded decisions of a new-best champion.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::json;
use sim::training::SCHEMA_VERSION;

use crate::event_index::EventIndex;

pub struct Response {
    pub status: u16,
    pub content_type: &'static str,
    pub body: Vec<u8>,
}

impl Response {
    fn json(status: u16, value: &serde_json::Value) -> Self {
        Self {
            status,
            content_type: "application/json; charset=utf-8",
            body: serde_json::to_vec(value).expect("json serializes"),
        }
    }

    fn error(status: u16, message: &str) -> Self {
        Self::json(status, &json!({ "error": message }))
    }

    fn asset(content_type: &'static str, text: &'static str) -> Self {
        Self {
            status: 200,
            content_type,
            body: text.as_bytes().to_vec(),
        }
    }
}

pub struct App {
    run_dir: PathBuf,
    index: Mutex<EventIndex>,
}

/// A generation number from a URL segment: plain digits only.
fn parse_generation(segment: &str) -> Option<u32> {
    if segment.is_empty() || !segment.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    segment.parse().ok()
}

fn query_value<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .find(|(k, _)| *k == key)
        .map(|(_, v)| v)
}

impl App {
    #[must_use]
    pub fn new(run_dir: PathBuf) -> Self {
        let index = Mutex::new(EventIndex::new(run_dir.join("events.jsonl")));
        Self { run_dir, index }
    }

    #[must_use]
    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }

    /// Handles one request. `target` is the request target as sent
    /// (path plus optional query).
    #[must_use]
    pub fn handle(&self, method: &str, target: &str) -> Response {
        if method != "GET" {
            return Response::error(405, "only GET is supported");
        }
        let (path, query) = target.split_once('?').unwrap_or((target, ""));
        match path {
            "/" => Response::asset(
                "text/html; charset=utf-8",
                include_str!("../assets/index.html"),
            ),
            "/app.js" => Response::asset(
                "text/javascript; charset=utf-8",
                include_str!("../assets/app.js"),
            ),
            "/lib.js" => Response::asset(
                "text/javascript; charset=utf-8",
                include_str!("../assets/lib.js"),
            ),
            "/style.css" => Response::asset(
                "text/css; charset=utf-8",
                include_str!("../assets/style.css"),
            ),
            "/api/state" => self.state(),
            "/api/events" => self.events(query),
            _ => {
                if let Some(rest) = path.strip_prefix("/api/genome/") {
                    return self.genome(rest);
                }
                if let Some(rest) = path.strip_prefix("/api/decisions/") {
                    return self.decisions(rest);
                }
                Response::error(404, "not found")
            }
        }
    }

    fn refreshed<T>(&self, read: impl FnOnce(&EventIndex) -> T) -> T {
        let mut index = self.index.lock().expect("index lock");
        index.refresh();
        read(&index)
    }

    fn state(&self) -> Response {
        let value = self.refreshed(|index| {
            json!({
                "schema_version": SCHEMA_VERSION,
                "run_start": index.run_start(),
                "generations_logged": index.generations().len(),
                "last_generation": index.generations().iter().map(|g| g.generation).max(),
                "finished": index.finished(),
                "epoch": index.epoch(),
            })
        });
        Response::json(200, &value)
    }

    fn events(&self, query: &str) -> Response {
        let since = match query_value(query, "since") {
            None => None,
            Some(text) => match parse_generation(text) {
                Some(generation) => Some(generation),
                None => return Response::error(400, "since must be a generation number"),
            },
        };
        let value = self.refreshed(|index| {
            let events: Vec<&sim::training::GenerationEvent> = index
                .generations()
                .iter()
                .filter(|g| since.is_none_or(|s| g.generation > s))
                .collect();
            json!({ "epoch": index.epoch(), "finished": index.finished(), "events": events })
        });
        Response::json(200, &value)
    }

    fn file(&self, relative: &str) -> Response {
        match std::fs::read(self.run_dir.join(relative)) {
            Ok(body) => Response {
                status: 200,
                content_type: "application/json; charset=utf-8",
                body,
            },
            Err(_) => Response::error(404, "not found"),
        }
    }

    fn genome(&self, which: &str) -> Response {
        if which == "best" {
            return self.file("best.json");
        }
        match parse_generation(which) {
            Some(generation) => self.file(&format!("gen-{generation:04}.json")),
            None => Response::error(404, "not found"),
        }
    }

    fn decisions(&self, which: &str) -> Response {
        match parse_generation(which) {
            Some(generation) => self.file(&format!("decisions/gen-{generation:04}.json")),
            None => Response::error(404, "not found"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixture::fixture_run_dir;

    fn json_of(response: &Response) -> serde_json::Value {
        serde_json::from_slice(&response.body).expect("a JSON body")
    }

    fn app() -> App {
        App::new(fixture_run_dir())
    }

    #[test]
    fn the_page_and_its_assets_are_served_with_the_right_types() {
        let app = app();
        for (path, content_type, marker) in [
            ("/", "text/html", "<title>"),
            ("/app.js", "text/javascript", "fetch"),
            ("/lib.js", "text/javascript", "export"),
            ("/style.css", "text/css", "--"),
        ] {
            let response = app.handle("GET", path);
            assert_eq!(response.status, 200, "{path}");
            assert!(response.content_type.starts_with(content_type), "{path}");
            assert!(
                String::from_utf8_lossy(&response.body).contains(marker),
                "{path}"
            );
        }
    }

    #[test]
    fn state_describes_the_run() {
        let response = app().handle("GET", "/api/state");
        assert_eq!(response.status, 200);
        let state = json_of(&response);
        assert_eq!(state["schema_version"], SCHEMA_VERSION);
        assert_eq!(state["generations_logged"], 3);
        assert_eq!(state["last_generation"], 2);
        assert_eq!(state["finished"], true);
        assert_eq!(state["run_start"]["config"]["player_count"], 4);
        assert_eq!(state["run_start"]["opponents"][0], "LowestLegal");
    }

    #[test]
    fn events_can_be_fetched_whole_or_after_a_generation() {
        let app = app();
        let all = json_of(&app.handle("GET", "/api/events"));
        assert_eq!(all["events"].as_array().unwrap().len(), 3);
        let after = json_of(&app.handle("GET", "/api/events?since=0"));
        let generations: Vec<u64> = after["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["generation"].as_u64().unwrap())
            .collect();
        assert_eq!(generations, vec![1, 2]);
        let none = json_of(&app.handle("GET", "/api/events?since=2"));
        assert!(none["events"].as_array().unwrap().is_empty());
        assert_eq!(
            all["epoch"],
            json_of(&app.handle("GET", "/api/state"))["epoch"]
        );
        assert_eq!(all["finished"], true);
    }

    #[test]
    fn a_malformed_since_is_a_400_not_a_guess() {
        for query in ["since=abc", "since=-1", "since=", "since=1.5"] {
            let response = app().handle("GET", &format!("/api/events?{query}"));
            assert_eq!(response.status, 400, "{query}");
        }
    }

    #[test]
    fn genomes_and_decisions_come_from_the_run_directory() {
        let app = app();
        let genome = json_of(&app.handle("GET", "/api/genome/0"));
        assert!(genome["genome"]["nodes"].is_array());
        assert_eq!(
            genome["feature_names"].as_array().unwrap().len(),
            sim::FEATURE_COUNT
        );
        let best = app.handle("GET", "/api/genome/best");
        assert_eq!(best.status, 200);
        let decisions = json_of(&app.handle("GET", "/api/decisions/0"));
        assert_eq!(decisions["generation"], 0);
        assert!(!decisions["decisions"].as_array().unwrap().is_empty());
    }

    #[test]
    fn unknown_or_hostile_paths_are_404_and_never_leave_the_run_directory() {
        let app = app();
        for path in [
            "/nope",
            "/api/genome/999",
            "/api/genome/..%2f..%2fetc%2fpasswd",
            "/api/genome/../../etc/passwd",
            "/api/genome/-1",
            "/api/genome/0x10",
            "/api/decisions/",
            "/api/decisions/best",
            "/api/events/extra",
        ] {
            assert_eq!(app.handle("GET", path).status, 404, "{path}");
        }
    }

    #[test]
    fn only_get_is_supported() {
        for method in ["POST", "PUT", "DELETE", "HEAD"] {
            assert_eq!(app().handle(method, "/api/state").status, 405, "{method}");
        }
    }

    #[test]
    fn an_empty_directory_is_a_valid_idle_state() {
        let empty =
            std::env::temp_dir().join(format!("arschloch-web-empty-{}", std::process::id()));
        std::fs::create_dir_all(&empty).unwrap();
        let app = App::new(empty.clone());
        let state = json_of(&app.handle("GET", "/api/state"));
        assert_eq!(state["generations_logged"], 0);
        assert!(state["run_start"].is_null() && state["last_generation"].is_null());
        assert_eq!(
            json_of(&app.handle("GET", "/api/events"))["events"]
                .as_array()
                .unwrap()
                .len(),
            0
        );
        assert_eq!(app.handle("GET", "/api/genome/best").status, 404);
        std::fs::remove_dir_all(empty).ok();
    }
}
