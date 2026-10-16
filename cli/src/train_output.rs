//! What `cli train` prints while it runs: a banner, one line per
//! generation, a refreshed progress line on a terminal, and a summary.
//! The line renderers (`render`) are pure functions of an event so they can be
//! tested without capturing stdout; the observer (`observer`) does the printing.

mod observer;
mod render;
#[cfg(test)]
mod tests;

pub use self::observer::TerminalObserver;
pub use self::render::{render_set_run_banner, render_set_summary, SetContext};
