//! End-to-end tests of `cli play`: the real binary, a real socket.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use serde_json::{json, Value};

fn temp(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "arschloch-play-smoke-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&path);
    let _ = std::fs::remove_file(&path);
    path
}

struct Play {
    child: Child,
    address: String,
}

impl Drop for Play {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start(extra: &[&str], records: &std::path::Path) -> Play {
    let mut child = Command::new(env!("CARGO_BIN_EXE_cli"))
        .args([
            "play",
            "--port",
            "0",
            "--records",
            records.to_str().unwrap(),
        ])
        .args(extra)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn cli play");
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    let line = lines.next().expect("a first line").unwrap();
    let address = line
        .split("http://")
        .nth(1)
        .unwrap_or_else(|| panic!("no url in {line:?}"))
        .split(['/', ' '])
        .next()
        .unwrap()
        .to_owned();
    Play { child, address }
}

fn request(address: &str, method: &str, target: &str, body: Option<&Value>) -> (u16, Value) {
    let mut stream = TcpStream::connect(address).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(20)))
        .unwrap();
    let payload = body.map(Value::to_string).unwrap_or_default();
    let head = if body.is_some() {
        format!(
            "{method} {target} HTTP/1.1\r\nHost: {address}\r\nOrigin: http://{address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{payload}",
            payload.len()
        )
    } else {
        format!("{method} {target} HTTP/1.1\r\nHost: {address}\r\n\r\n")
    };
    stream.write_all(head.as_bytes()).unwrap();
    let mut reply = String::new();
    stream.read_to_string(&mut reply).unwrap();
    let status = reply.split_whitespace().nth(1).unwrap().parse().unwrap();
    let body = reply.split_once("\r\n\r\n").map_or("", |(_, b)| b);
    (status, serde_json::from_str(body).unwrap_or(Value::Null))
}

#[test]
fn a_match_can_be_played_through_the_real_server_and_is_recorded() {
    let records = temp("records.jsonl");
    let play = start(&[], &records);
    let (status, catalog) = request(&play.address, "GET", "/api/catalog", None);
    assert_eq!(status, 200);
    let ids: Vec<&str> = catalog["opponents"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o["id"].as_str().unwrap())
        .collect();
    assert!(
        ids.contains(&"model:champion-v1") && ids.contains(&"lowest-legal"),
        "{ids:?}"
    );

    let (status, game) = request(
        &play.address,
        "POST",
        "/api/games",
        Some(
            &json!({"players": 3, "rounds": 2, "opponents": ["model:champion-v1", "lowest-legal"], "human_seat": 1, "seed": 3}),
        ),
    );
    assert_eq!(status, 200, "{game}");
    let id = game["id"].as_str().unwrap().to_owned();
    let mut view = game["view"].clone();
    for _ in 0..2000 {
        if view["phase"] == "match_over" {
            break;
        }
        let url = |action: &str| format!("/api/games/{id}/{action}");
        let (status, reply) = match view["phase"].as_str().unwrap() {
            "exchange" => {
                let n = usize::try_from(view["give_count"].as_u64().unwrap()).unwrap();
                let cards: Vec<Value> = view["hand"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .take(n)
                    .map(|c| c["id"].clone())
                    .collect();
                request(
                    &play.address,
                    "POST",
                    &url("give"),
                    Some(&json!({"cards": cards})),
                )
            }
            "playing" => match view["playable"].as_array().unwrap().first() {
                Some(rank) => {
                    let size = usize::try_from(rank["sizes"][0].as_u64().unwrap()).unwrap();
                    let ids = rank["card_ids"].as_array().unwrap();
                    let cards = &ids[ids.len() - size..];
                    request(
                        &play.address,
                        "POST",
                        &url("play"),
                        Some(&json!({"cards": cards})),
                    )
                }
                None => request(&play.address, "POST", &url("pass"), Some(&json!({}))),
            },
            _ => request(&play.address, "POST", &url("next"), Some(&json!({}))),
        };
        assert_eq!(status, 200, "{reply}");
        view = reply["view"].clone();
    }
    assert_eq!(view["phase"], "match_over");
    let (_, summary) = request(&play.address, "GET", "/api/records", None);
    assert_eq!(summary["total_games"], 1);
    assert_eq!(
        std::fs::read_to_string(&records).unwrap().lines().count(),
        1
    );
    // A cross-site page cannot drive the game.
    let mut stream = TcpStream::connect(&play.address).unwrap();
    write!(
        stream,
        "POST /api/games HTTP/1.1\r\nHost: {}\r\nOrigin: http://evil.example\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{{}}",
        play.address
    )
    .unwrap();
    let mut reply = String::new();
    stream.read_to_string(&mut reply).unwrap();
    assert!(reply.starts_with("HTTP/1.1 403"), "{reply}");
    let _ = std::fs::remove_file(records);
}

#[test]
fn a_given_model_shows_up_in_the_catalog() {
    let base = temp("model");
    std::fs::create_dir_all(&base).unwrap();
    let genome = base.join("my-model.json");
    std::fs::copy("../docs/baselines/neat-v1/champion.json", &genome).unwrap();
    let play = start(
        &["--model", genome.to_str().unwrap()],
        &base.join("r.jsonl"),
    );
    let (_, catalog) = request(&play.address, "GET", "/api/catalog", None);
    let ids: Vec<&str> = catalog["opponents"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o["id"].as_str().unwrap())
        .collect();
    assert!(ids.contains(&"model:my-model"), "{ids:?}");
    let _ = std::fs::remove_dir_all(base);
}

#[test]
fn a_bad_model_or_a_busy_port_fails_cleanly() {
    let output = Command::new(env!("CARGO_BIN_EXE_cli"))
        .args(["play", "--port", "0", "--model", "/nonexistent/m.json"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("/nonexistent/m.json") && !stderr.contains("panicked"),
        "{stderr}"
    );

    let records = temp("busy.jsonl");
    let first = start(&[], &records);
    let port = first.address.rsplit(':').next().unwrap().to_owned();
    let second = Command::new(env!("CARGO_BIN_EXE_cli"))
        .args([
            "play",
            "--port",
            &port,
            "--records",
            records.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!second.status.success());
    assert!(String::from_utf8_lossy(&second.stderr).contains("--port"));
}
