//! A deliberately small blocking HTTP/1.1 server on `std::net`: one
//! thread accepts, one short-lived thread serves each connection, every
//! response closes the connection. It binds to the loopback interface
//! only (use an SSH tunnel to view a remote run), and a slow or stuck
//! client can only ever hold its own thread (reads and writes time out).
//!
//! The server is generic over a `Handler`. `GET` is always available; a
//! handler that returns `true` from `accepts_post` also receives `POST`
//! requests with a JSON body, which must carry `Content-Type:
//! application/json` and a `Content-Length` (at most 16 KB), and whose
//! `Origin`, when present, must be this machine: a page on another site
//! must not be able to drive the application.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::routes::{App, Response};

const IO_TIMEOUT: Duration = Duration::from_secs(5);
/// Requests (head, and separately a body) are small; anything larger is
/// not a browser.
const MAX_REQUEST_BYTES: usize = 16 * 1024;

pub struct HttpRequest {
    pub method: String,
    /// The request target as sent: path plus optional query.
    pub target: String,
    pub body: Vec<u8>,
}

pub trait Handler: Send + Sync + 'static {
    fn handle(&self, request: &HttpRequest) -> Response;

    /// Whether `POST` requests are passed on (otherwise they are 405).
    fn accepts_post(&self) -> bool {
        false
    }
}

impl Handler for App {
    fn handle(&self, request: &HttpRequest) -> Response {
        App::handle(self, &request.method, &request.target)
    }
}

pub struct Server {
    address: SocketAddr,
    accept_thread: JoinHandle<()>,
}

impl Server {
    /// Starts serving `handler` on `127.0.0.1:port` (`0` picks a free
    /// port) in background threads. The threads live until the process
    /// exits.
    ///
    /// # Errors
    ///
    /// Returns the bind error (for example, the port is in use).
    pub fn start<H: Handler>(handler: Arc<H>, port: u16) -> std::io::Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", port))?;
        let address = listener.local_addr()?;
        let accept_thread = thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let handler = Arc::clone(&handler);
                thread::spawn(move || serve_connection(stream, handler.as_ref()));
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

/// The training dashboard for a run directory.
pub struct Dashboard {
    server: Server,
}

impl Dashboard {
    /// Starts serving `run_dir` on `127.0.0.1:port`.
    ///
    /// # Errors
    ///
    /// Returns the bind error (for example, the port is in use).
    pub fn start(run_dir: PathBuf, port: u16) -> std::io::Result<Self> {
        Ok(Self {
            server: Server::start(Arc::new(App::new(run_dir)), port)?,
        })
    }

    #[must_use]
    pub fn address(&self) -> SocketAddr {
        self.server.address()
    }

    #[must_use]
    pub fn url(&self) -> String {
        self.server.url()
    }

    /// Blocks for as long as the server runs (until the process exits).
    pub fn wait(self) {
        self.server.wait();
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        411 => "Length Required",
        413 => "Payload Too Large",
        415 => "Unsupported Media Type",
        _ => "Internal Server Error",
    }
}

fn plain(status: u16, message: &str) -> Response {
    Response {
        status,
        content_type: "text/plain; charset=utf-8",
        body: message.as_bytes().to_vec(),
    }
}

struct Head {
    method: String,
    target: String,
    host: Option<String>,
    origin: Option<String>,
    content_type: Option<String>,
    content_length: Option<String>,
    chunked: bool,
}

/// The host part of `host` or of an `Origin` value (`http://host:port`),
/// lower-cased, without port or brackets.
fn host_name(value: &str) -> String {
    let value = value.split_once("://").map_or(value, |(_, rest)| rest);
    let value = value.split('/').next().unwrap_or("");
    let name = match value.strip_prefix('[') {
        Some(rest) => rest.split(']').next().unwrap_or(""),
        None => value.split(':').next().unwrap_or(""),
    };
    name.to_ascii_lowercase()
}

fn is_local_name(name: &str) -> bool {
    matches!(name, "127.0.0.1" | "localhost" | "::1")
}

/// Only requests addressed to this machine are served. A hostile website
/// can point its own domain name at 127.0.0.1 (DNS rebinding) and make the
/// victim's browser send requests with that name in `Host`; refusing every
/// other name keeps the run's files out of its reach. HTTP/1.0 clients may
/// omit the header.
fn host_allowed(host: Option<&str>) -> bool {
    host.is_none_or(|host| is_local_name(&host_name(host)))
}

/// Reads the request head and returns it with any body bytes that arrived
/// in the same reads.
fn read_head(stream: &mut TcpStream) -> Option<(Head, Vec<u8>)> {
    let mut received = Vec::new();
    let mut chunk = [0u8; 1024];
    let head_end = loop {
        if let Some(position) = received.windows(4).position(|w| w == b"\r\n\r\n") {
            break position;
        }
        if received.len() > MAX_REQUEST_BYTES {
            return None;
        }
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            return None;
        }
        received.extend_from_slice(&chunk[..read]);
    };
    let text = String::from_utf8_lossy(&received[..head_end]).into_owned();
    let mut lines = text.lines();
    let mut parts = lines.next()?.split_whitespace();
    let (method, target) = (parts.next()?, parts.next()?);
    let mut head = Head {
        method: method.to_owned(),
        target: target.to_owned(),
        host: None,
        origin: None,
        content_type: None,
        content_length: None,
        chunked: false,
    };
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim().to_owned();
        match name.trim().to_ascii_lowercase().as_str() {
            "host" => head.host = Some(value),
            "origin" => head.origin = Some(value),
            "content-type" => head.content_type = Some(value),
            "content-length" => head.content_length = Some(value),
            "transfer-encoding" => head.chunked = true,
            _ => {}
        }
    }
    Some((head, received[head_end + 4..].to_vec()))
}

/// Reads exactly `length` body bytes (some may already be in `body`).
fn read_body(stream: &mut TcpStream, mut body: Vec<u8>, length: usize) -> Option<Vec<u8>> {
    let mut chunk = [0u8; 1024];
    while body.len() < length {
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            return None;
        }
        body.extend_from_slice(&chunk[..read]);
    }
    body.truncate(length);
    Some(body)
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

fn respond(stream: &mut TcpStream, handler: &dyn Handler) -> Response {
    let Some((head, early_body)) = read_head(stream) else {
        return plain(400, "bad request");
    };
    if !host_allowed(head.host.as_deref()) {
        return plain(
            403,
            "this server only answers requests addressed to localhost",
        );
    }
    let body = match head.method.as_str() {
        "GET" => Vec::new(),
        "POST" if handler.accepts_post() => {
            if head
                .origin
                .as_deref()
                .is_some_and(|origin| !is_local_name(&host_name(origin)))
            {
                return plain(403, "requests from other sites are refused");
            }
            let json = head.content_type.as_deref().is_some_and(|value| {
                value
                    .split(';')
                    .next()
                    .is_some_and(|kind| kind.trim().eq_ignore_ascii_case("application/json"))
            });
            if !json {
                return plain(415, "send application/json");
            }
            if head.chunked {
                return plain(411, "chunked bodies are not supported; send Content-Length");
            }
            let Some(length) = head
                .content_length
                .as_deref()
                .and_then(|v| v.parse::<usize>().ok())
            else {
                return plain(411, "Content-Length is required");
            };
            if length > MAX_REQUEST_BYTES {
                return plain(413, "the request body is too large");
            }
            match read_body(stream, early_body, length) {
                Some(body) => body,
                None => return plain(400, "the request body did not arrive"),
            }
        }
        _ => return plain(405, "only GET (and POST where offered) is supported"),
    };
    handler.handle(&HttpRequest {
        method: head.method,
        target: head.target,
        body,
    })
}

fn serve_connection(mut stream: TcpStream, handler: &dyn Handler) {
    let _ = stream.set_read_timeout(Some(IO_TIMEOUT));
    let _ = stream.set_write_timeout(Some(IO_TIMEOUT));
    let response = respond(&mut stream, handler);
    write_response(&mut stream, &response);
    // A request refused before its body was read leaves bytes unread;
    // closing on them would reset the connection and could destroy the
    // response. Finish our side, then let the client's bytes drain.
    let _ = stream.shutdown(std::net::Shutdown::Write);
    let _ = stream.set_read_timeout(Some(Duration::from_millis(300)));
    let mut sink = [0u8; 4096];
    let mut drained = 0;
    while drained < 4 * MAX_REQUEST_BYTES {
        match stream.read(&mut sink) {
            Ok(0) | Err(_) => break,
            Ok(read) => drained += read,
        }
    }
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

    /// Echoes the method, target and body, and accepts POST.
    struct Echo;
    impl Handler for Echo {
        fn handle(&self, request: &HttpRequest) -> Response {
            Response {
                status: 200,
                content_type: "text/plain; charset=utf-8",
                body: format!(
                    "{} {} [{}]",
                    request.method,
                    request.target,
                    String::from_utf8_lossy(&request.body)
                )
                .into_bytes(),
            }
        }
        fn accepts_post(&self) -> bool {
            true
        }
    }

    fn echo_server() -> Server {
        Server::start(Arc::new(Echo), 0).unwrap()
    }

    fn post(address: SocketAddr, extra_headers: &str, body: &str) -> String {
        raw_request(
            address,
            format!(
                "POST /api/x HTTP/1.1\r\nHost: localhost\r\n{extra_headers}Content-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .as_bytes(),
        )
    }

    const JSON: &str = "Content-Type: application/json\r\n";

    #[test]
    fn a_json_post_reaches_the_handler_with_its_body() {
        let server = echo_server();
        let reply = post(server.address(), JSON, "{\"a\":1}");
        assert!(reply.starts_with("HTTP/1.1 200 OK"), "{reply}");
        assert_eq!(body(&reply), "POST /api/x [{\"a\":1}]");
        let charset = post(
            server.address(),
            "Content-Type: application/json; charset=utf-8\r\n",
            "{}",
        );
        assert!(charset.starts_with("HTTP/1.1 200"), "{charset}");
    }

    #[test]
    fn a_post_is_refused_unless_it_is_json_of_a_known_length_from_this_machine() {
        let server = echo_server();
        let address = server.address();
        let status = |reply: &str| reply.lines().next().unwrap_or("").to_owned();
        // Wrong or missing content type (the form encodings a hostile page can send).
        for headers in [
            "Content-Type: text/plain\r\n",
            "Content-Type: application/x-www-form-urlencoded\r\n",
            "",
        ] {
            let reply = post(address, headers, "{}");
            assert!(
                reply.starts_with("HTTP/1.1 415"),
                "{headers:?}: {}",
                status(&reply)
            );
        }
        // A page on another site (it cannot send JSON cross-origin without a
        // preflight we never answer, but check the header anyway).
        for origin in [
            "http://evil.example",
            "null",
            "http://127.0.0.1.evil.example:80",
        ] {
            let reply = post(address, &format!("{JSON}Origin: {origin}\r\n"), "{}");
            assert!(
                reply.starts_with("HTTP/1.1 403"),
                "{origin}: {}",
                status(&reply)
            );
        }
        for origin in [
            format!("http://127.0.0.1:{}", address.port()),
            format!("http://localhost:{}", address.port()),
            "http://[::1]:8080".to_owned(),
        ] {
            let reply = post(address, &format!("{JSON}Origin: {origin}\r\n"), "{}");
            assert!(
                reply.starts_with("HTTP/1.1 200"),
                "{origin}: {}",
                status(&reply)
            );
        }
        // A hostile Host (DNS rebinding) is refused for POST as for GET.
        let rebinding = raw_request(
            address,
            format!(
                "POST /api/x HTTP/1.1\r\nHost: evil.example\r\n{JSON}Content-Length: 2\r\n\r\n{{}}"
            )
            .as_bytes(),
        );
        assert!(rebinding.starts_with("HTTP/1.1 403"), "{rebinding}");
        // No length, chunked, or too large.
        let no_length = raw_request(
            address,
            format!("POST /api/x HTTP/1.1\r\nHost: localhost\r\n{JSON}\r\n{{}}").as_bytes(),
        );
        assert!(no_length.starts_with("HTTP/1.1 411"), "{no_length}");
        let chunked = raw_request(
            address,
            format!(
                "POST /api/x HTTP/1.1\r\nHost: localhost\r\n{JSON}Transfer-Encoding: chunked\r\n\r\n2\r\n{{}}\r\n0\r\n\r\n"
            )
            .as_bytes(),
        );
        assert!(
            chunked.starts_with("HTTP/1.1 411") || chunked.starts_with("HTTP/1.1 400"),
            "{chunked}"
        );
        let big = "x".repeat(MAX_REQUEST_BYTES + 1);
        assert!(post(address, JSON, &big).starts_with("HTTP/1.1 413"));
        // Still serving.
        assert!(post(address, JSON, "{}").starts_with("HTTP/1.1 200"));
    }

    #[test]
    fn a_body_shorter_than_declared_times_out_without_hurting_the_server() {
        let server = echo_server();
        let mut stream = TcpStream::connect(server.address()).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(20)))
            .unwrap();
        stream
            .write_all(
                format!(
                    "POST /api/x HTTP/1.1\r\nHost: localhost\r\n{JSON}Content-Length: 50\r\n\r\n{{}}"
                )
                .as_bytes(),
            )
            .unwrap();
        let mut reply = Vec::new();
        let _ = stream.read_to_end(&mut reply);
        assert!(String::from_utf8_lossy(&reply).starts_with("HTTP/1.1 400"));
        assert!(post(server.address(), JSON, "{}").starts_with("HTTP/1.1 200"));
    }

    #[test]
    fn get_still_works_on_a_post_handler_and_other_methods_are_405() {
        let server = echo_server();
        assert_eq!(
            body(&get(server.address(), "/hello?x=1")),
            "GET /hello?x=1 []"
        );
        let put = raw_request(
            server.address(),
            b"PUT /x HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n",
        );
        assert!(put.starts_with("HTTP/1.1 405"), "{put}");
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
