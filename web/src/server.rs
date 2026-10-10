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
mod tests;
