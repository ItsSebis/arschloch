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
    assert!(raw_request(address, b"GET /api/state HTTP/1.0\r\n\r\n").starts_with("HTTP/1.1 200 OK"));
}

#[test]
fn errors_come_back_as_proper_statuses() {
    let dashboard = Dashboard::start(fixture_run_dir(), 0).unwrap();
    assert!(get(dashboard.address(), "/nope").starts_with("HTTP/1.1 404 Not Found"));
    assert!(get(dashboard.address(), "/api/events?since=x").starts_with("HTTP/1.1 400 Bad Request"));
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
