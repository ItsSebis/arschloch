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
        PlayApp::new(catalog, RecordStore::new(Some(path.clone()))).with_fixed_seeds(),
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
    let act =
        |action: &str, body: &str| call(&app, "POST", &format!("/api/games/{id}/{action}"), body);
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
fn the_pass_rule_is_chosen_per_game_and_defaults_to_the_rules_of_the_game() {
    let (app, path) = app("passrule");
    let make = |extra: Value| {
        let mut body =
            json!({"players": 3, "opponents": ["lowest-legal", "lowest-legal"], "human_seat": 0});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        call(&app, "POST", "/api/games", &body.to_string())
    };
    assert_eq!(make(json!({})).1["view"]["pass_rule"], "final");
    let (status, free) = make(json!({"pass_rule": "free"}));
    assert_eq!(status, 200);
    assert_eq!(free["view"]["pass_rule"], "free");
    assert_eq!(make(json!({"pass_rule": "sometimes"})).0, 400);
    assert_eq!(make(json!({"pass_rule": 5})).0, 400);
    std::fs::remove_file(path).ok();
}

#[test]
fn the_exchange_rule_is_chosen_per_game_and_forced_means_the_human_never_chooses() {
    let (app, path) = app("exchangerule");
    let make = |extra: Value| {
        let mut body = json!({"players": 4, "opponents": ["lowest-legal", "lowest-legal", "lowest-legal"], "human_seat": 0, "rounds": 3});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        call(&app, "POST", "/api/games", &body.to_string())
    };
    assert_eq!(make(json!({})).1["view"]["exchange_rule"], "forced");
    assert_eq!(
        make(json!({"exchange_rule": "free"})).1["view"]["exchange_rule"],
        "free"
    );
    assert_eq!(make(json!({"exchange_rule": "sometimes"})).0, 400);
    assert_eq!(make(json!({"exchange_rule": 5})).0, 400);
    // A whole forced match never reaches the exchange phase.
    let (id, mut view) = new_game(
        &app,
        4,
        3,
        &["lowest-legal", "lowest-legal", "lowest-legal"],
    );
    let mut guard = 0;
    while view["phase"] != "match_over" {
        guard += 1;
        assert!(guard < 5000);
        assert_ne!(
            view["phase"], "exchange",
            "a forced exchange has nothing to choose"
        );
        view = step(&app, &id, &view);
    }
    std::fs::remove_file(path).ok();
}

#[test]
fn a_client_cannot_fix_the_deal() {
    // Otherwise: the same seed from another seat shows that seat's hand.
    let catalog = vec![CatalogEntry::strategy(
        "lowest-legal",
        "LowestLegal",
        Arc::new(LowestLegal),
    )];
    let plain = PlayApp::new(catalog, RecordStore::new(None));
    let hand = |seat: u32| {
        let body = json!({"players": 3, "opponents": ["lowest-legal", "lowest-legal"], "seed": 42, "human_seat": seat});
        let (status, reply) = call(&plain, "POST", "/api/games", &body.to_string());
        assert_eq!(status, 200);
        reply["view"]["hand"].to_string()
    };
    // With the seed honoured, seat 0 of one game and seat 0 of another
    // would be identical; ignoring it they differ (13 of 52 cards by chance
    // is astronomically unlikely to repeat).
    assert_ne!(hand(0), hand(0));
    std::fs::remove_file(records_path("noseed")).ok();
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
