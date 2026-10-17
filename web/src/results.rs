//! `cli view FILE`: a page for the `results.json` of a simulation batch.

use std::path::PathBuf;

use crate::routes::Response;
use crate::server::{Handler, HttpRequest};

/// Serves one `results.json`, re-read on every request so a rerun shows up
/// after a reload, and optionally a second one to compare it with.
pub struct ResultsApp {
    file: PathBuf,
    compare: Option<PathBuf>,
}

impl ResultsApp {
    #[must_use]
    pub fn new(file: PathBuf) -> Self {
        Self {
            file,
            compare: None,
        }
    }

    /// Also serve a second results file (`/api/compare`) for the side-by-side view.
    #[must_use]
    pub fn with_compare(mut self, file: PathBuf) -> Self {
        self.compare = Some(file);
        self
    }

    #[must_use]
    pub fn handle(&self, method: &str, target: &str) -> Response {
        if method != "GET" {
            return Response::error(405, "only GET is supported");
        }
        let path = target.split_once('?').map_or(target, |(path, _)| path);
        let js = "text/javascript; charset=utf-8";
        match path {
            "/" => Response::asset(
                "text/html; charset=utf-8",
                include_str!("../assets/results.html"),
            ),
            "/results.js" => Response::asset(js, include_str!("../assets/results.js")),
            "/results_lib.js" => Response::asset(js, include_str!("../assets/results_lib.js")),
            "/lib.js" => Response::asset(js, include_str!("../assets/lib.js")),
            "/style.css" => Response::asset(
                "text/css; charset=utf-8",
                include_str!("../assets/style.css"),
            ),
            "/api/results" => serve_json(&self.file),
            "/api/compare" => match &self.compare {
                Some(file) => serve_json(file),
                None => Response::error(404, "no second results file was given"),
            },
            _ => Response::error(404, "not found"),
        }
    }
}

fn serve_json(file: &std::path::Path) -> Response {
    let shown = file.display();
    match std::fs::read(file) {
        Err(error) => Response::error(404, &format!("cannot read {shown}: {error}")),
        // Validate without building a tree, then serve the bytes as they are.
        Ok(bytes) => match serde_json::from_slice::<serde::de::IgnoredAny>(&bytes) {
            Ok(_) => Response {
                status: 200,
                content_type: "application/json; charset=utf-8",
                body: bytes,
            },
            Err(error) => Response::error(422, &format!("{shown} is not valid JSON: {error}")),
        },
    }
}

impl Handler for ResultsApp {
    fn handle(&self, request: &HttpRequest) -> Response {
        ResultsApp::handle(self, &request.method, &request.target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(name: &str, text: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "arschloch-results-{name}-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, text).unwrap();
        path
    }

    #[test]
    fn the_page_and_its_script_are_served() {
        let app = ResultsApp::new("unused.json".into());
        let page = app.handle("GET", "/");
        assert_eq!(page.status, 200);
        assert!(page.content_type.starts_with("text/html"));
        assert!(String::from_utf8(page.body)
            .unwrap()
            .contains("/results.js"));
        assert_eq!(app.handle("GET", "/results.js").status, 200);
        assert_eq!(app.handle("GET", "/results_lib.js").status, 200);
        assert_eq!(app.handle("GET", "/style.css").status, 200);
    }

    #[test]
    fn the_api_returns_the_file_afresh_on_every_request() {
        let path = temp_file("fresh", r#"{"statistics":{"matches_played":1}}"#);
        let app = ResultsApp::new(path.clone());
        let first = app.handle("GET", "/api/results");
        assert_eq!(first.status, 200);
        assert!(first.content_type.starts_with("application/json"));
        let value: serde_json::Value = serde_json::from_slice(&first.body).unwrap();
        assert_eq!(value["statistics"]["matches_played"], 1);
        std::fs::write(&path, r#"{"statistics":{"matches_played":2}}"#).unwrap();
        let second: serde_json::Value =
            serde_json::from_slice(&app.handle("GET", "/api/results").body).unwrap();
        assert_eq!(
            second["statistics"]["matches_played"], 2,
            "a rerun shows up on reload"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn the_file_is_served_byte_for_byte_without_being_rebuilt_in_memory() {
        let text = r#"{"z": 1, "a": [1, 2,  3]}"#; // key order and spacing survive only if nothing re-serialises it
        let path = temp_file("bytes", text);
        let response = ResultsApp::new(path.clone()).handle("GET", "/api/results");
        assert_eq!(response.status, 200);
        assert_eq!(String::from_utf8(response.body).unwrap(), text);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_missing_or_broken_file_is_said_in_words() {
        let missing =
            ResultsApp::new("definitely/not/here.json".into()).handle("GET", "/api/results");
        assert_eq!(missing.status, 404);
        assert!(String::from_utf8(missing.body)
            .unwrap()
            .contains("here.json"));
        let path = temp_file("broken", "not json");
        let broken = ResultsApp::new(path.clone()).handle("GET", "/api/results");
        assert_eq!(broken.status, 422);
        assert!(String::from_utf8(broken.body)
            .unwrap()
            .contains("not valid JSON"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_second_file_is_served_only_when_given() {
        let none = ResultsApp::new("unused.json".into());
        assert_eq!(none.handle("GET", "/api/compare").status, 404);
        let path = temp_file("compare", r#"{"statistics":{"matches_played":7}}"#);
        let app = ResultsApp::new("unused.json".into()).with_compare(path.clone());
        let response = app.handle("GET", "/api/compare");
        assert_eq!(response.status, 200);
        let value: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(value["statistics"]["matches_played"], 7);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn nothing_else_is_served() {
        let app = ResultsApp::new("unused.json".into());
        assert_eq!(app.handle("GET", "/etc/passwd").status, 404);
        assert_eq!(app.handle("GET", "/api/results/../x").status, 404);
        assert_eq!(app.handle("POST", "/api/results").status, 405);
    }
}
