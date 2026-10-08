//! End-to-end tests of the dashboard commands: the real binary serving a
//! real run over a real socket.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

fn run_dir(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("arschloch-cli-dash-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

const SMALL: [&str; 12] = [
    "--population",
    "12",
    "--matches-per-genome",
    "4",
    "--reeval-matches",
    "6",
    "--rounds",
    "3",
    "--threads",
    "2",
    "--generations",
    "2",
];

/// A child process whose stdout lines arrive on a channel; killed on drop.
struct Server {
    child: Child,
    lines: mpsc::Receiver<String>,
}

impl Server {
    fn spawn(args: &[&str]) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_cli"))
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn cli");
        let stdout = child.stdout.take().unwrap();
        let (sender, lines) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if sender.send(line).is_err() {
                    break;
                }
            }
        });
        Self { child, lines }
    }

    fn next_line_containing(&self, needle: &str) -> String {
        let deadline = Instant::now() + Duration::from_secs(60);
        while Instant::now() < deadline {
            if let Ok(line) = self.lines.recv_timeout(Duration::from_millis(200)) {
                if line.contains(needle) {
                    return line;
                }
            }
        }
        panic!("no stdout line containing {needle:?} within 60 s");
    }

    fn address(&self) -> String {
        let line = self.next_line_containing("dashboard: http://");
        let url = line.split("http://").nth(1).unwrap();
        url.split(['/', ' ']).next().unwrap().to_owned()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn get(address: &str, target: &str) -> String {
    let mut stream = TcpStream::connect(address).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    write!(stream, "GET {target} HTTP/1.1\r\nHost: {address}\r\n\r\n").unwrap();
    let mut reply = String::new();
    stream.read_to_string(&mut reply).unwrap();
    reply
}

fn json(reply: &str) -> serde_json::Value {
    serde_json::from_str(reply.split_once("\r\n\r\n").unwrap().1).unwrap()
}

fn wait_until_finished(address: &str) -> serde_json::Value {
    let deadline = Instant::now() + Duration::from_secs(90);
    while Instant::now() < deadline {
        let state = json(&get(address, "/api/state"));
        if state["finished"] == true {
            return state;
        }
        thread::sleep(Duration::from_millis(200));
    }
    panic!("the run never reported finished");
}

#[test]
fn train_serve_shows_the_run_and_stays_up_after_it_finishes() {
    let out = run_dir("serve");
    let mut args = vec![
        "train",
        "--out",
        out.to_str().unwrap(),
        "--serve",
        "0",
        "--quiet",
    ];
    args.extend(SMALL);
    let server = Server::spawn(&args);
    let address = server.address();

    let state = wait_until_finished(&address);
    assert_eq!(state["generations_logged"], 2);
    assert_eq!(state["run_start"]["config"]["generations"], 2);
    let events = json(&get(&address, "/api/events"));
    assert_eq!(events["events"].as_array().unwrap().len(), 2);
    assert_eq!(events["finished"], true);

    // The process keeps serving after the run ends, and says so.
    server.next_line_containing("dashboard stays up");
    assert_eq!(json(&get(&address, "/api/state"))["finished"], true);
    let genome = json(&get(&address, "/api/genome/best"));
    assert_eq!(
        genome["feature_names"].as_array().unwrap().len(),
        sim::FEATURE_COUNT
    );
    let page = get(&address, "/");
    assert!(page.contains("<title>NEAT training</title>"));
    drop(server);
    std::fs::remove_dir_all(&out).unwrap();
}

#[test]
fn watch_serves_a_finished_run_directory() {
    let out = run_dir("watch");
    let mut args = vec!["train", "--out", out.to_str().unwrap(), "--quiet"];
    args.extend(SMALL);
    assert!(Command::new(env!("CARGO_BIN_EXE_cli"))
        .args(&args)
        .status()
        .unwrap()
        .success());

    let server = Server::spawn(&["watch", out.to_str().unwrap(), "--port", "0"]);
    let address = server.address();
    let state = json(&get(&address, "/api/state"));
    assert_eq!(
        (
            state["generations_logged"].clone(),
            state["finished"].clone()
        ),
        (2.into(), true.into())
    );
    let decisions = json(&get(&address, "/api/decisions/0"));
    assert_ne!(
        *decisions["decisions"].as_array().unwrap(),
        [] as [serde_json::Value; 0]
    );
    assert!(get(&address, "/api/genome/999").starts_with("HTTP/1.1 404"));
    drop(server);
    std::fs::remove_dir_all(&out).unwrap();
}

#[test]
fn watch_follows_a_run_that_starts_later() {
    // The dashboard tolerates a directory that has no run yet.
    let out = run_dir("later");
    std::fs::create_dir_all(&out).unwrap();
    let server = Server::spawn(&["watch", out.to_str().unwrap(), "--port", "0"]);
    let address = server.address();
    assert_eq!(json(&get(&address, "/api/state"))["generations_logged"], 0);
    let mut args = vec!["train", "--out", out.to_str().unwrap(), "--quiet"];
    args.extend(SMALL);
    assert!(Command::new(env!("CARGO_BIN_EXE_cli"))
        .args(&args)
        .status()
        .unwrap()
        .success());
    assert_eq!(wait_until_finished(&address)["generations_logged"], 2);
    drop(server);
    std::fs::remove_dir_all(&out).unwrap();
}

#[test]
fn a_busy_port_fails_before_any_run_state_is_created() {
    let busy = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = busy.local_addr().unwrap().port().to_string();
    let out = run_dir("busy");
    let output = Command::new(env!("CARGO_BIN_EXE_cli"))
        .args([
            "train",
            "--out",
            out.to_str().unwrap(),
            "--serve",
            &port,
            "--generations",
            "1",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("cannot start the dashboard") && stderr.contains(&port),
        "{stderr}"
    );
    assert!(!stderr.contains("panicked"));
    assert!(!out.join("checkpoint.json").exists(), "nothing was created");

    let watch = Command::new(env!("CARGO_BIN_EXE_cli"))
        .args(["watch", out.to_str().unwrap(), "--port", &port])
        .output()
        .unwrap();
    assert!(!watch.status.success());
    let nothing = Command::new(env!("CARGO_BIN_EXE_cli"))
        .args(["watch", "/nonexistent/run"])
        .output()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&nothing.stderr).contains("is not a directory"),
        "{}",
        String::from_utf8_lossy(&nothing.stderr)
    );
}

#[test]
fn train_serve_follows_a_set_through_its_runs() {
    let out = run_dir("serve-set");
    let mut args = vec![
        "train",
        "--out",
        out.to_str().unwrap(),
        "--runs",
        "2",
        "--serve",
        "0",
        "--quiet",
    ];
    args.extend(SMALL);
    let server = Server::spawn(&args);
    let address = server.address();
    server.next_line_containing("dashboard stays up");
    let state = json(&get(&address, "/api/state"));
    assert_eq!(state["set"]["total_runs"], 2);
    assert_eq!(state["set"]["finished_secs"].as_array().unwrap().len(), 2);
    assert_eq!(state["finished"], true);
    assert_eq!(
        state["run_start"]["config"]["seed"], 1,
        "the page shows the last run (seed 0 + 1)"
    );
    assert_eq!(
        json(&get(&address, "/api/genome/best"))["format_version"],
        1
    );
    drop(server);
    std::fs::remove_dir_all(&out).unwrap();
}
