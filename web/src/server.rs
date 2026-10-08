//! A deliberately small blocking HTTP/1.1 server on `std::net`: one
//! thread accepts, one short-lived thread serves each connection, every
//! response closes the connection. It binds to the loopback interface
//! only (use an SSH tunnel to view a remote run), and a slow or stuck
//! client can only ever hold its own thread (reads and writes time out).

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::routes::{App, Response};

const IO_TIMEOUT: Duration = Duration::from_secs(5);
/// Requests are tiny `GET`s; anything larger is not a browser.
const MAX_REQUEST_BYTES: usize = 16 * 1024;

pub struct Dashboard {
    address: SocketAddr,
    accept_thread: JoinHandle<()>,
}

impl Dashboard {
    /// Starts serving `run_dir` on `127.0.0.1:port` (`0` picks a free
    /// port) in background threads. The threads live until the process
    /// exits.
    ///
    /// # Errors
    ///
    /// Returns the bind error (for example, the port is in use).
    pub fn start(run_dir: PathBuf, port: u16) -> std::io::Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", port))?;
        let address = listener.local_addr()?;
        let app = Arc::new(App::new(run_dir));
        let accept_thread = thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let app = Arc::clone(&app);
                thread::spawn(move || serve_connection(stream, &app));
            }
        });
        Ok(Self {
            address,
            accept_thread,
        })
    }

    #[must_use]
    pub fn address(&self) -> SocketAddr {
        self.address
    }

    #[must_use]
    pub fn url(&self) -> String {
        format!("http://{}/", self.address)
    }

    /// Blocks for as long as the server runs (until the process exits).
    pub fn wait(self) {
        let _ = self.accept_thread.join();
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        _ => "Internal Server Error",
    }
}

struct Request {
    method: String,
    target: String,
    host: Option<String>,
}

/// Only requests addressed to this machine are served. A hostile website
/// can point its own domain name at 127.0.0.1 (DNS rebinding) and make the
/// victim's browser send requests with that name in `Host`; refusing every
/// other name keeps the run's files out of its reach. HTTP/1.0 clients may
/// omit the header.
fn host_allowed(host: Option<&str>) -> bool {
    let Some(host) = host else {
        return true;
    };
    let name = match host.strip_prefix('[') {
        Some(rest) => rest.split(']').next().unwrap_or(""),
        None => host.split(':').next().unwrap_or(""),
    };
    matches!(
        name.to_ascii_lowercase().as_str(),
        "127.0.0.1" | "localhost" | "::1"
    )
}

/// Reads the request head.
fn read_request(stream: &mut TcpStream) -> Option<Request> {
    let mut received = Vec::new();
    let mut chunk = [0u8; 1024];
    while !received.windows(4).any(|w| w == b"\r\n\r\n") {
        if received.len() > MAX_REQUEST_BYTES {
            return None;
        }
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            return None;
        }
        received.extend_from_slice(&chunk[..read]);
    }
    let head = String::from_utf8_lossy(&received);
    let mut parts = head.lines().next()?.split_whitespace();
    let (method, target) = (parts.next()?, parts.next()?);
    let host = head.lines().skip(1).find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.trim()
            .eq_ignore_ascii_case("host")
            .then(|| value.trim().to_owned())
    });
    Some(Request {
        method: method.to_owned(),
        target: target.to_owned(),
        host,
    })
}

fn write_response(stream: &mut TcpStream, response: &Response) {
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n",
        response.status,
        reason(response.status),
        response.content_type,
        response.body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(&response.body);
    let _ = stream.flush();
}

fn serve_connection(mut stream: TcpStream, app: &App) {
    let _ = stream.set_read_timeout(Some(IO_TIMEOUT));
    let _ = stream.set_write_timeout(Some(IO_TIMEOUT));
    let response = match read_request(&mut stream) {
        Some(request) if !host_allowed(request.host.as_deref()) => Response {
            status: 403,
            content_type: "text/plain; charset=utf-8",
            body: b"this server only answers requests addressed to localhost".to_vec(),
        },
        Some(request) => app.handle(&request.method, &request.target),
        None => Response {
            status: 400,
            content_type: "text/plain; charset=utf-8",
            body: b"bad request".to_vec(),
        },
    };
    write_response(&mut stream, &response);
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpStream;

    use super::*;
    use crate::test_fixture::fixture_run_dir;

    fn raw_request(address: SocketAddr, request: &[u8]) -> String {
        let mut stream = TcpStream::connect(address).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        stream.write_all(request).unwrap();
        let mut reply = Vec::new();
        stream.read_to_end(&mut reply).unwrap();
        String::from_utf8_lossy(&reply).into_owned()
    }

    fn get(address: SocketAddr, target: &str) -> String {
        raw_request(
            address,
            format!("GET {target} HTTP/1.1\r\nHost: localhost\r\n\r\n").as_bytes(),
        )
    }

    fn body(reply: &str) -> &str {
        reply.split_once("\r\n\r\n").map_or("", |(_, b)| b)
    }

    #[test]
    fn it_serves_the_api_over_a_real_socket() {
        let dashboard = Dashboard::start(fixture_run_dir(), 0).unwrap();
        assert!(dashboard.address().ip().is_loopback());
        let reply = get(dashboard.address(), "/api/state");
        assert!(reply.starts_with("HTTP/1.1 200 OK\r\n"), "{reply}");
        assert!(reply.contains("Content-Type: application/json"));
        assert!(reply.contains("Cache-Control: no-store"));
        let state: serde_json::Value = serde_json::from_str(body(&reply)).unwrap();
        assert_eq!(state["generations_logged"], 3);
        let declared: usize = reply
            .lines()
            .find_map(|l| l.strip_prefix("Content-Length: "))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert_eq!(declared, body(&reply).len(), "Content-Length is exact");
        let page = get(dashboard.address(), "/");
        assert!(page.contains("text/html") && body(&page).contains("<title>"));
    }

    #[test]
    fn a_foreign_host_header_is_refused_to_defeat_dns_rebinding() {
        // A hostile site can point its own name at 127.0.0.1 and have the
        // victim's browser talk to this server with that name as Host.
        let dashboard = Dashboard::start(fixture_run_dir(), 0).unwrap();
        let address = dashboard.address();
        let port = address.port();
        let with_host = |host: &str| {
            raw_request(
                address,
                format!("GET /api/state HTTP/1.1\r\nHost: {host}\r\n\r\n").as_bytes(),
            )
        };
        for hostile in [
            "evil.example",
            &format!("evil.example:{port}"),
            "127.0.0.1.evil.example",
            "10.0.0.5",
        ] {
            let reply = with_host(hostile);
            assert!(
                reply.starts_with("HTTP/1.1 403 Forbidden"),
                "{hostile}: {reply}"
            );
            assert!(
                !reply.contains("generations_logged"),
                "no data may leak to {hostile}"
            );
        }
        for friendly in [
            format!("127.0.0.1:{port}"),
            format!("localhost:{port}"),
            "localhost".to_owned(),
            format!("[::1]:{port}"),
            format!("LOCALHOST:{port}"),
        ] {
            assert!(
                with_host(&friendly).starts_with("HTTP/1.1 200 OK"),
                "{friendly}"
            );
        }
        // HTTP/1.0 clients may omit Host entirely.
        assert!(
            raw_request(address, b"GET /api/state HTTP/1.0\r\n\r\n").starts_with("HTTP/1.1 200 OK")
        );
    }

    #[test]
    fn errors_come_back_as_proper_statuses() {
        let dashboard = Dashboard::start(fixture_run_dir(), 0).unwrap();
        assert!(get(dashboard.address(), "/nope").starts_with("HTTP/1.1 404 Not Found"));
        assert!(
            get(dashboard.address(), "/api/events?since=x").starts_with("HTTP/1.1 400 Bad Request")
        );
        let post = raw_request(
            dashboard.address(),
            b"POST /api/state HTTP/1.1\r\nContent-Length: 0\r\n\r\n",
        );
        assert!(
            post.starts_with("HTTP/1.1 405 Method Not Allowed"),
            "{post}"
        );
    }

    #[test]
    fn junk_and_oversized_requests_are_rejected_without_hurting_the_server() {
        let dashboard = Dashboard::start(fixture_run_dir(), 0).unwrap();
        let junk = raw_request(dashboard.address(), b"\x00\x01\x02 not http\r\n\r\n");
        assert!(
            junk.starts_with("HTTP/1.1 400")
                || junk.starts_with("HTTP/1.1 405")
                || junk.starts_with("HTTP/1.1 404"),
            "{junk}"
        );
        let huge = vec![b'a'; MAX_REQUEST_BYTES + 4096];
        let mut stream = TcpStream::connect(dashboard.address()).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        let _ = stream.write_all(&huge);
        let mut reply = Vec::new();
        let _ = stream.read_to_end(&mut reply);
        assert!(String::from_utf8_lossy(&reply).starts_with("HTTP/1.1 400") || reply.is_empty());
        assert!(
            get(dashboard.address(), "/api/state").starts_with("HTTP/1.1 200"),
            "still serving"
        );
    }

    #[test]
    fn many_concurrent_clients_all_get_answers() {
        let dashboard = Dashboard::start(fixture_run_dir(), 0).unwrap();
        let address = dashboard.address();
        let handles: Vec<_> = (0..16)
            .map(|_| thread::spawn(move || get(address, "/api/events?since=0")))
            .collect();
        for handle in handles {
            assert!(handle.join().unwrap().starts_with("HTTP/1.1 200"));
        }
    }

    #[test]
    fn binding_a_busy_port_is_an_error_not_a_panic() {
        let first = Dashboard::start(fixture_run_dir(), 0).unwrap();
        let second = Dashboard::start(fixture_run_dir(), first.address().port());
        assert!(second.is_err());
    }
}
