//! The play-against-the-models application: a catalog of opponents, the
//! games in progress, and the results. Its routes (see `PlayApp::handle`)
//! are JSON over GET/POST; the page itself is static.
//!
//! - `GET  /`, `/play.js`, `/play-lib.js`, `/play.css`: the page
//! - `GET  /api/catalog`: the opponents a game can be set up with
//! - `POST /api/games`: start a game
//! - `GET  /api/games/ID?events_since=N`: the view and the events from N
//! - `POST /api/games/ID/{play,pass,give,next}`: the human acts
//! - `GET  /api/games/ID/advice?model=ID`: a model ranks the legal moves
//! - `GET  /api/records`: the human's results

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use sim::session::{AiSeat, Phase, Session, SessionConfig, SessionError};
use sim::{DeckVariant, DuplicateRule, NeatStrategy, Strategy};

use crate::records::{Record, RecordStore};
use crate::routes::Response;
use crate::server::{Handler, HttpRequest};

/// Games kept at once; the least recently used is dropped for a new one.
const MAX_GAMES: usize = 16;
const MAX_ROUNDS: usize = 50;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OpponentKind {
    Model,
    Strategy,
}

/// One opponent a game can be set up with.
#[derive(Clone)]
pub struct CatalogEntry {
    pub id: String,
    pub label: String,
    pub kind: OpponentKind,
    pub strategy: Arc<dyn Strategy>,
    /// Set for trained models, which can also give advice.
    pub model: Option<Arc<NeatStrategy>>,
}

impl CatalogEntry {
    /// A hand-written strategy.
    #[must_use]
    pub fn strategy(id: &str, label: &str, strategy: Arc<dyn Strategy>) -> Self {
        Self {
            id: id.to_owned(),
            label: label.to_owned(),
            kind: OpponentKind::Strategy,
            strategy,
            model: None,
        }
    }

    /// A trained model.
    #[must_use]
    pub fn model(id: &str, label: &str, model: Arc<NeatStrategy>) -> Self {
        Self {
            id: id.to_owned(),
            label: label.to_owned(),
            kind: OpponentKind::Model,
            strategy: model.clone(),
            model: Some(model),
        }
    }
}

struct Setup {
    players: u8,
    deck: DeckVariant,
    rule: DuplicateRule,
    rounds: usize,
    entries: Vec<CatalogEntry>,
    human_seat: u8,
    seed: u64,
}

struct Game {
    session: Session,
    opponents: Vec<String>,
    deck: String,
    duplicate_rule: String,
    recorded: bool,
    last_used: u64,
}

#[derive(Default)]
struct Games {
    map: HashMap<String, Game>,
    clock: u64,
}

pub struct PlayApp {
    catalog: Vec<CatalogEntry>,
    games: Mutex<Games>,
    records: RecordStore,
}

fn bad(status: u16, message: &str) -> Response {
    Response::json(status, &json!({ "error": message }))
}

fn session_error(error: &SessionError) -> Response {
    bad(400, &error.to_string())
}

/// Decodes `%XX` escapes and `+` (a space) in a query-string value;
/// malformed escapes are kept as they are.
fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' if bytes.len() >= i + 3 => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3])
                    .ok()
                    .and_then(|h| u8::from_str_radix(h, 16).ok());
                if let Some(value) = hex {
                    out.push(value);
                    i += 2;
                } else {
                    out.push(b'%');
                }
            }
            other => out.push(other),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn query_value(query: &str, key: &str) -> Option<String> {
    query
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .find(|(k, _)| *k == key)
        .map(|(_, v)| percent_decode(v))
}

fn new_id() -> String {
    format!("{:032x}", rand::random::<u128>())
}

impl PlayApp {
    #[must_use]
    pub fn new(catalog: Vec<CatalogEntry>, records: RecordStore) -> Self {
        Self {
            catalog,
            games: Mutex::new(Games::default()),
            records,
        }
    }

    fn entry(&self, id: &str) -> Option<&CatalogEntry> {
        self.catalog.iter().find(|entry| entry.id == id)
    }

    fn catalog_response(&self) -> Response {
        let entries: Vec<Value> = self
            .catalog
            .iter()
            .map(|entry| {
                json!({
                    "id": entry.id,
                    "label": entry.label,
                    "kind": if entry.kind == OpponentKind::Model { "model" } else { "strategy" },
                })
            })
            .collect();
        Response::json(200, &json!({ "opponents": entries }))
    }

    /// Checks a game-setup request and resolves its opponents.
    fn parse_setup(&self, body: &[u8]) -> Result<Setup, Response> {
        let Ok(request) = serde_json::from_slice::<Value>(body) else {
            return Err(bad(400, "the request is not valid JSON"));
        };
        let Some(players) = request["players"]
            .as_u64()
            .and_then(|p| u8::try_from(p).ok())
            .filter(|p| (3..=6).contains(p))
        else {
            return Err(bad(400, "players must be 3-6"));
        };
        let deck = match request["deck"].as_str().unwrap_or("single") {
            "single" => DeckVariant::Single,
            "double" => DeckVariant::Double,
            _ => return Err(bad(400, "deck must be single or double")),
        };
        let rule = match request["duplicate_rule"]
            .as_str()
            .unwrap_or("first_dealt_wins")
        {
            "first_dealt_wins" => DuplicateRule::FirstDealtWins,
            "last_dealt_wins" => DuplicateRule::LastDealtWins,
            _ => {
                return Err(bad(
                    400,
                    "duplicate_rule must be first_dealt_wins or last_dealt_wins",
                ))
            }
        };
        let rounds = match request["rounds"].as_u64() {
            None => 8,
            Some(r) => match usize::try_from(r)
                .ok()
                .filter(|r| (1..=MAX_ROUNDS).contains(r))
            {
                Some(r) => r,
                None => return Err(bad(400, &format!("rounds must be 1-{MAX_ROUNDS}"))),
            },
        };
        let Some(ids) = request["opponents"].as_array() else {
            return Err(bad(400, "opponents must be a list of catalog ids"));
        };
        if ids.len() != usize::from(players) - 1 {
            return Err(bad(400, &format!("{} opponents are needed", players - 1)));
        }
        let mut entries = Vec::new();
        for id in ids {
            let Some(entry) = id.as_str().and_then(|id| self.entry(id)) else {
                return Err(bad(400, "an opponent is not in the catalog"));
            };
            entries.push(entry.clone());
        }
        let human_seat = match request["human_seat"].as_u64() {
            None => u8::try_from(rand::random::<u32>() % u32::from(players)).unwrap_or(0),
            Some(seat) => match u8::try_from(seat).ok().filter(|s| *s < players) {
                Some(seat) => seat,
                None => return Err(bad(400, "human_seat is not at the table")),
            },
        };
        Ok(Setup {
            players,
            deck,
            rule,
            rounds,
            entries,
            human_seat,
            seed: request["seed"].as_u64().unwrap_or_else(rand::random),
        })
    }

    fn create_game(&self, body: &[u8]) -> Response {
        let setup = match self.parse_setup(body) {
            Ok(setup) => setup,
            Err(response) => return response,
        };
        let Setup {
            players,
            deck,
            rule,
            rounds,
            entries,
            human_seat,
            seed,
        } = setup;
        let session = match Session::new(
            SessionConfig {
                player_count: players,
                deck_variant: deck,
                duplicate_rule: rule,
                rounds,
                seed,
                human_seat,
            },
            entries
                .iter()
                .map(|entry| AiSeat {
                    name: entry.label.clone(),
                    strategy: Arc::clone(&entry.strategy),
                })
                .collect(),
        ) {
            Ok(session) => session,
            Err(error) => return session_error(&error),
        };
        let id = new_id();
        let response = Response::json(
            200,
            &json!({
                "id": id,
                "view": session.view(),
                "events_from": 0,
                "events": session.events_since(0),
            }),
        );
        let mut games = self
            .games
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if games.map.len() >= MAX_GAMES {
            let oldest = games
                .map
                .iter()
                .min_by_key(|(_, game)| game.last_used)
                .map(|(id, _)| id.clone());
            if let Some(oldest) = oldest {
                games.map.remove(&oldest);
            }
        }
        games.clock += 1;
        let last_used = games.clock;
        games.map.insert(
            id,
            Game {
                session,
                opponents: entries.iter().map(|e| e.label.clone()).collect(),
                deck: if deck == DeckVariant::Single {
                    "single"
                } else {
                    "double"
                }
                .into(),
                duplicate_rule: if rule == DuplicateRule::FirstDealtWins {
                    "first_dealt_wins"
                } else {
                    "last_dealt_wins"
                }
                .into(),
                recorded: false,
                last_used,
            },
        );
        response
    }

    /// Runs `with` on the game `id`, if it exists.
    fn with_game<T>(&self, id: &str, with: impl FnOnce(&mut Game) -> T) -> Option<T> {
        let mut games = self
            .games
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        games.clock += 1;
        let clock = games.clock;
        let game = games.map.get_mut(id)?;
        game.last_used = clock;
        Some(with(game))
    }

    fn game_response(&self, id: &str, since: usize) -> Response {
        self.with_game(id, |game| {
            Response::json(
                200,
                &json!({
                    "view": game.session.view(),
                    "events_from": since.min(game.session.view().event_count),
                    "events": game.session.events_since(since),
                }),
            )
        })
        .unwrap_or_else(|| bad(404, "no such game (it may have been dropped)"))
    }

    fn act(&self, id: &str, action: &str, body: &[u8]) -> Response {
        let parsed = if body.is_empty() {
            Value::Null
        } else {
            match serde_json::from_slice::<Value>(body) {
                Ok(value) => value,
                Err(_) => return bad(400, "the request is not valid JSON"),
            }
        };
        let cards: Result<Vec<u8>, ()> =
            parsed["cards"].as_array().map_or(Ok(Vec::new()), |items| {
                items
                    .iter()
                    .map(|item| item.as_u64().and_then(|n| u8::try_from(n).ok()).ok_or(()))
                    .collect()
            });
        let Ok(cards) = cards else {
            return bad(400, "cards must be a list of card ids");
        };
        let outcome = self.with_game(id, |game| {
            let before = game.session.view().event_count;
            let result = match action {
                "play" => game.session.play(&cards),
                "pass" => game.session.pass(),
                "give" => game.session.give(&cards),
                "next" => game.session.next_round(),
                _ => return bad(404, "no such action"),
            };
            if let Err(error) = result {
                return session_error(&error);
            }
            if game.session.phase() == Phase::MatchOver && !game.recorded {
                game.recorded = true;
                self.record(game);
            }
            Response::json(
                200,
                &json!({
                    "view": game.session.view(),
                    "events_from": before,
                    "events": game.session.events_since(before),
                }),
            )
        });
        outcome.unwrap_or_else(|| bad(404, "no such game (it may have been dropped)"))
    }

    fn record(&self, game: &Game) {
        let Some(result) = game.session.view().final_result else {
            return;
        };
        let view = game.session.view();
        self.records.append(&Record {
            finished_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            player_count: view.player_count,
            deck: game.deck.clone(),
            duplicate_rule: game.duplicate_rule.clone(),
            rounds: view.rounds,
            opponents: game.opponents.clone(),
            roles: result.roles,
            score: result.score,
        });
    }

    fn advice(&self, id: &str, query: &str) -> Response {
        let Some(entry) = query_value(query, "model").and_then(|m| self.entry(&m)) else {
            return bad(400, "model must be a catalog id");
        };
        let Some(model) = &entry.model else {
            return bad(400, "only trained models give advice");
        };
        self.with_game(id, |game| match game.session.advice(model) {
            Ok(advice) => Response::json(200, &json!({ "model": entry.label, "advice": advice })),
            Err(error) => session_error(&error),
        })
        .unwrap_or_else(|| bad(404, "no such game (it may have been dropped)"))
    }
}

impl Handler for PlayApp {
    fn accepts_post(&self) -> bool {
        true
    }

    fn handle(&self, request: &HttpRequest) -> Response {
        let (path, query) = request
            .target
            .split_once('?')
            .unwrap_or((request.target.as_str(), ""));
        let post = request.method == "POST";
        match (post, path) {
            (false, "/") => Response::asset(
                "text/html; charset=utf-8",
                include_str!("../assets/play.html"),
            ),
            (false, "/play.js") => Response::asset(
                "text/javascript; charset=utf-8",
                include_str!("../assets/play.js"),
            ),
            (false, "/play-lib.js") => Response::asset(
                "text/javascript; charset=utf-8",
                include_str!("../assets/play-lib.js"),
            ),
            (false, "/play.css") => Response::asset(
                "text/css; charset=utf-8",
                include_str!("../assets/play.css"),
            ),
            (false, "/api/catalog") => self.catalog_response(),
            (false, "/api/records") => Response::json(200, &self.records.summary()),
            (true, "/api/games") => self.create_game(&request.body),
            _ => {
                let Some(rest) = path.strip_prefix("/api/games/") else {
                    return bad(404, "not found");
                };
                let (id, action) = rest.split_once('/').unwrap_or((rest, ""));
                match (post, action) {
                    (false, "") => {
                        let since = query_value(query, "events_since")
                            .and_then(|v| v.parse::<usize>().ok())
                            .unwrap_or(0);
                        self.game_response(id, since)
                    }
                    (false, "advice") => self.advice(id, query),
                    (true, "play" | "pass" | "give" | "next") => {
                        self.act(id, action, &request.body)
                    }
                    _ => bad(404, "not found"),
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)] // small test numbers
mod tests {
    use std::path::PathBuf;

    use sim::{LowestLegal, RandomLegal};

    use super::*;

    fn champion() -> Arc<NeatStrategy> {
        Arc::new(
            NeatStrategy::from_file(std::path::Path::new(
                "../docs/baselines/neat-v1/champion.json",
            ))
            .unwrap(),
        )
    }

    fn records_path(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "arschloch-play-{name}-{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        path
    }

    fn app(name: &str) -> (PlayApp, PathBuf) {
        let path = records_path(name);
        let catalog = vec![
            CatalogEntry::strategy("lowest-legal", "LowestLegal", Arc::new(LowestLegal)),
            CatalogEntry::strategy("random-legal", "RandomLegal", Arc::new(RandomLegal)),
            CatalogEntry::model("model:champion", "Neat(champion)", champion()),
        ];
        (
            PlayApp::new(catalog, RecordStore::new(Some(path.clone()))),
            path,
        )
    }

    fn call(app: &PlayApp, method: &str, target: &str, body: &str) -> (u16, Value) {
        let response = app.handle(&HttpRequest {
            method: method.to_owned(),
            target: target.to_owned(),
            body: body.as_bytes().to_vec(),
        });
        let value = serde_json::from_slice(&response.body).unwrap_or(Value::Null);
        (response.status, value)
    }

    fn new_game(app: &PlayApp, players: u8, rounds: u32, opponents: &[&str]) -> (String, Value) {
        let body = json!({
            "players": players, "rounds": rounds, "human_seat": 0,
            "opponents": opponents, "seed": 5,
        });
        let (status, reply) = call(app, "POST", "/api/games", &body.to_string());
        assert_eq!(status, 200, "{reply}");
        (
            reply["id"].as_str().unwrap().to_owned(),
            reply["view"].clone(),
        )
    }

    /// One scripted human action; returns the new view.
    fn step(app: &PlayApp, id: &str, view: &Value) -> Value {
        let url = |action: &str| format!("/api/games/{id}/{action}");
        let (status, reply) = match view["phase"].as_str().unwrap() {
            "exchange" => {
                let n = view["give_count"].as_u64().unwrap() as usize;
                let cards: Vec<u64> = view["hand"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .take(n)
                    .map(|c| c["id"].as_u64().unwrap())
                    .collect();
                call(
                    app,
                    "POST",
                    &url("give"),
                    &json!({ "cards": cards }).to_string(),
                )
            }
            "playing" => match view["playable"].as_array().unwrap().first() {
                Some(rank) => {
                    let size = rank["sizes"][0].as_u64().unwrap() as usize;
                    let ids = rank["card_ids"].as_array().unwrap();
                    // Strongest subset: the group is sorted weakest first.
                    let cards: Vec<&Value> = ids[ids.len() - size..].iter().collect();
                    call(
                        app,
                        "POST",
                        &url("play"),
                        &json!({ "cards": cards }).to_string(),
                    )
                }
                None => call(app, "POST", &url("pass"), "{}"),
            },
            "round_over" => call(app, "POST", &url("next"), "{}"),
            other => panic!("unexpected phase {other}"),
        };
        assert_eq!(status, 200, "{reply}");
        reply["view"].clone()
    }

    #[test]
    fn a_whole_match_runs_through_the_api_and_is_recorded_once() {
        let (app, path) = app("whole");
        let (id, mut view) = new_game(
            &app,
            4,
            3,
            &["lowest-legal", "random-legal", "model:champion"],
        );
        let mut guard = 0;
        while view["phase"] != "match_over" {
            guard += 1;
            assert!(guard < 5000, "the game never ends");
            view = step(&app, &id, &view);
        }
        assert_eq!(view["final"]["roles"].as_array().unwrap().len(), 3);
        // Further actions are refused and do not record again.
        for action in ["next", "pass"] {
            let (status, _) = call(&app, "POST", &format!("/api/games/{id}/{action}"), "{}");
            assert_eq!(status, 400, "{action}");
        }
        let (_, records) = call(&app, "GET", "/api/records", "");
        assert_eq!(records["total_games"], 1);
        assert_eq!(records["by_opponent"].as_array().unwrap().len(), 3);
        assert_eq!(std::fs::read_to_string(&path).unwrap().lines().count(), 1);
        // The finished game can still be fetched.
        let (status, fetched) = call(&app, "GET", &format!("/api/games/{id}"), "");
        assert_eq!(status, 200);
        assert_eq!(fetched["view"]["phase"], "match_over");
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn events_can_be_fetched_from_any_point() {
        let (app, path) = app("events");
        let (id, view) = new_game(&app, 3, 1, &["lowest-legal", "lowest-legal"]);
        let total = view["event_count"].as_u64().unwrap();
        let (_, all) = call(&app, "GET", &format!("/api/games/{id}"), "");
        assert_eq!(all["events"].as_array().unwrap().len() as u64, total);
        let (_, tail) = call(
            &app,
            "GET",
            &format!("/api/games/{id}?events_since={}", total - 1),
            "",
        );
        assert_eq!(tail["events"].as_array().unwrap().len(), 1);
        let (_, none) = call(
            &app,
            "GET",
            &format!("/api/games/{id}?events_since=999999"),
            "",
        );
        assert_eq!(none["events"].as_array().unwrap().len(), 0);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn bad_requests_are_400_or_404_and_leave_the_game_unchanged() {
        let (app, path) = app("bad");
        let (id, _) = new_game(
            &app,
            4,
            2,
            &["lowest-legal", "lowest-legal", "lowest-legal"],
        );
        let (_, before) = call(&app, "GET", &format!("/api/games/{id}"), "");
        let act = |action: &str, body: &str| {
            call(&app, "POST", &format!("/api/games/{id}/{action}"), body)
        };
        for (what, (status, _)) in [
            ("unparseable", act("play", "{ nope")),
            ("cards not a list", act("play", r#"{"cards": 5}"#)),
            ("cards not numbers", act("play", r#"{"cards": ["a"]}"#)),
            ("card out of range", act("play", r#"{"cards": [999]}"#)),
            ("empty play", act("play", r#"{"cards": []}"#)),
            (
                "give outside the exchange",
                act("give", r#"{"cards": [1]}"#),
            ),
            ("next too early", act("next", "{}")),
        ] {
            assert_eq!(status, 400, "{what}");
        }
        assert_eq!(act("dance", "{}").0, 404);
        let (_, after) = call(&app, "GET", &format!("/api/games/{id}"), "");
        assert_eq!(before, after, "rejected requests changed the game");
        assert_eq!(call(&app, "GET", "/api/games/nope", "").0, 404);
        assert_eq!(call(&app, "POST", "/api/games/nope/pass", "{}").0, 404);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn a_game_must_be_set_up_properly() {
        let (app, path) = app("setup");
        let make = |body: Value| call(&app, "POST", "/api/games", &body.to_string());
        let ok = ["lowest-legal", "lowest-legal", "lowest-legal"];
        assert_eq!(
            make(json!({"players": 2, "opponents": ["lowest-legal"]})).0,
            400
        );
        assert_eq!(make(json!({"players": 7, "opponents": ok})).0, 400);
        assert_eq!(
            make(json!({"players": 4, "opponents": ["lowest-legal"]})).0,
            400
        );
        assert_eq!(
            make(json!({"players": 4, "opponents": ["lowest-legal", "lowest-legal", "ghost"]})).0,
            400
        );
        assert_eq!(
            make(json!({"players": 4, "opponents": ok, "rounds": 0})).0,
            400
        );
        assert_eq!(
            make(json!({"players": 4, "opponents": ok, "rounds": 100_000})).0,
            400
        );
        assert_eq!(
            make(json!({"players": 4, "opponents": ok, "human_seat": 4})).0,
            400
        );
        assert_eq!(
            make(json!({"players": 4, "opponents": ok, "deck": "triple"})).0,
            400
        );
        assert_eq!(
            make(json!({"players": 4, "opponents": ok, "duplicate_rule": "x"})).0,
            400
        );
        assert_eq!(call(&app, "POST", "/api/games", "not json").0, 400);
        let (status, reply) = make(
            json!({"players": 5, "opponents": vec!["lowest-legal"; 4], "deck": "double", "duplicate_rule": "last_dealt_wins"}),
        );
        assert_eq!(status, 200, "{reply}");
        assert_eq!(reply["id"].as_str().unwrap().len(), 32);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn only_the_newest_games_are_kept() {
        let (app, path) = app("evict");
        let ids: Vec<String> = (0..=MAX_GAMES)
            .map(|_| new_game(&app, 3, 1, &["lowest-legal", "lowest-legal"]).0)
            .collect();
        assert_eq!(
            call(&app, "GET", &format!("/api/games/{}", ids[0]), "").0,
            404
        );
        assert_eq!(
            call(&app, "GET", &format!("/api/games/{}", ids[MAX_GAMES]), "").0,
            200
        );
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len(), "ids are distinct");
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn a_used_game_is_not_the_one_evicted() {
        let (app, path) = app("lru");
        let first = new_game(&app, 3, 1, &["lowest-legal", "lowest-legal"]).0;
        let rest: Vec<String> = (1..MAX_GAMES)
            .map(|_| new_game(&app, 3, 1, &["lowest-legal", "lowest-legal"]).0)
            .collect();
        // Touch the oldest, then add one more: the second-oldest goes.
        assert_eq!(call(&app, "GET", &format!("/api/games/{first}"), "").0, 200);
        let _ = new_game(&app, 3, 1, &["lowest-legal", "lowest-legal"]);
        assert_eq!(call(&app, "GET", &format!("/api/games/{first}"), "").0, 200);
        assert_eq!(
            call(&app, "GET", &format!("/api/games/{}", rest[0]), "").0,
            404
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn advice_comes_from_a_model_on_the_humans_turn_only() {
        let (app, path) = app("advice");
        let (id, view) = new_game(
            &app,
            4,
            1,
            &["lowest-legal", "lowest-legal", "lowest-legal"],
        );
        assert_eq!(view["phase"], "playing");
        let (status, reply) = call(
            &app,
            "GET",
            &format!("/api/games/{id}/advice?model=model:champion"),
            "",
        );
        assert_eq!(status, 200, "{reply}");
        assert_eq!(reply["model"], "Neat(champion)");
        let advice = reply["advice"].as_array().unwrap();
        assert_ne!(advice.len(), 0);
        assert!(advice
            .windows(2)
            .all(|w| w[0]["raw_score"].as_f64() >= w[1]["raw_score"].as_f64()));
        for bad in ["model=lowest-legal", "model=ghost", ""] {
            assert_eq!(
                call(&app, "GET", &format!("/api/games/{id}/advice?{bad}"), "").0,
                400,
                "{bad}"
            );
        }
        assert_eq!(
            call(
                &app,
                "GET",
                "/api/games/nope/advice?model=model:champion",
                ""
            )
            .0,
            404
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn percent_encoded_query_values_are_decoded() {
        let (app, path) = app("encoded");
        let (id, _) = new_game(
            &app,
            4,
            1,
            &["lowest-legal", "lowest-legal", "lowest-legal"],
        );
        // Browsers encode the colon of `model:champion` in a query string.
        let (status, reply) = call(
            &app,
            "GET",
            &format!("/api/games/{id}/advice?model=model%3Achampion"),
            "",
        );
        assert_eq!(status, 200, "{reply}");
        assert_eq!(percent_decode("a%3Ab%2Fc+d%zz%4"), "a:b/c d%zz%4");
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn the_catalog_and_the_page_are_served() {
        let (app, path) = app("catalog");
        let (_, catalog) = call(&app, "GET", "/api/catalog", "");
        let kinds: Vec<(&str, &str)> = catalog["opponents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| (o["id"].as_str().unwrap(), o["kind"].as_str().unwrap()))
            .collect();
        assert_eq!(
            kinds,
            vec![
                ("lowest-legal", "strategy"),
                ("random-legal", "strategy"),
                ("model:champion", "model")
            ]
        );
        let page = app.handle(&HttpRequest {
            method: "GET".into(),
            target: "/".into(),
            body: Vec::new(),
        });
        assert_eq!(page.status, 200);
        assert_eq!(
            app.handle(&HttpRequest {
                method: "GET".into(),
                target: "/etc/passwd".into(),
                body: Vec::new()
            })
            .status,
            404
        );
        std::fs::remove_file(path).ok();
    }
}
