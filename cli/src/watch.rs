//! `cli watch RUN_DIR`: the dashboard for a run directory, on its own.
//! It follows a run that is training in another process and shows a
//! finished one.

use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;

/// Show a training run in the browser.
#[derive(Parser, Debug)]
#[command(
    name = "cli watch",
    about = "Serve the training dashboard for a run directory (live while it trains, or afterwards)"
)]
pub struct WatchArgs {
    /// A run directory written by `cli train`.
    pub dir: PathBuf,

    /// Port to serve on, on 127.0.0.1 only (use an SSH tunnel to view a
    /// remote run).
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
}

pub fn run(raw_args: impl Iterator<Item = String>) -> anyhow::Result<()> {
    let args = WatchArgs::parse_from(std::iter::once("cli watch".to_owned()).chain(raw_args));
    anyhow::ensure!(
        args.dir.is_dir(),
        "{} is not a directory (expected a run directory written by `cli train`)",
        args.dir.display()
    );
    let dashboard = web::Dashboard::start(args.dir.clone(), args.port).with_context(|| {
        format!(
            "cannot start the dashboard on port {} (is it in use? pick another with --port)",
            args.port
        )
    })?;
    println!("dashboard: {}  (Ctrl-C to stop)", dashboard.url());
    dashboard.wait();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_directory_is_required_and_the_port_defaults_to_8080() {
        let args = WatchArgs::try_parse_from(["cli watch", "runs/a"]).unwrap();
        assert_eq!((args.dir, args.port), (PathBuf::from("runs/a"), 8080));
        assert!(WatchArgs::try_parse_from(["cli watch"]).is_err());
        assert_eq!(
            WatchArgs::try_parse_from(["cli watch", "d", "--port", "0"])
                .unwrap()
                .port,
            0
        );
    }
}
