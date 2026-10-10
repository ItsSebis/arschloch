//! `cli stats-doc`: prints the statistics reference generated from the
//! catalogue (`docs/STATISTICS.md`).

/// Prints the Markdown reference to stdout.
#[allow(clippy::unnecessary_wraps)] // same signature as the other subcommands
pub fn run(_args: impl Iterator<Item = String>) -> anyhow::Result<()> {
    print!("{}", sim::stats_catalog::render_markdown());
    Ok(())
}
