//! The training dashboard: a small HTTP server that shows a run directory
//! (`events.jsonl`, `gen-NNNN.json`, `best.json`, `decisions/`) in a
//! browser, live while training and afterwards. It only ever *reads* the
//! run's files, so a slow or closed browser cannot affect training.
//! See docs/superpowers/specs/2026-10-08-neat-engine-design.md, 7a.

mod event_index;
mod routes;
mod server;
#[cfg(test)]
mod test_fixture;

pub use event_index::EventIndex;
pub use routes::{App, Response};
pub use server::{Dashboard, Handler, HttpRequest, Server};
