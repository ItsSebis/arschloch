//! `cli view FILE`: the `results.json` of a simulation batch as a page in the
//! browser (Phase 11b).

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use clap::Parser;

/// Show the results of a simulation batch in the browser.
#[derive(Parser, Debug)]
#[command(
    name = "cli view",
    about = "Serve a page for the results.json of a simulation batch (reload the page after a new run)"
)]
pub struct ViewArgs {
    /// A results file written by a plain `cli` run (`--output`).
    pub file: PathBuf,

    /// Port to serve on, on 127.0.0.1 only.
    #[arg(long, default_value_t = 8100)]
    pub port: u16,
}

pub fn run(raw_args: impl Iterator<Item = String>) -> anyhow::Result<()> {
    let args = ViewArgs::parse_from(std::iter::once("cli view".to_owned()).chain(raw_args));
    anyhow::ensure!(
        args.file.is_file(),
        "{} is not a file (expected the results.json of a `cli` run)",
        args.file.display()
    );
    let server = web::Server::start(Arc::new(web::ResultsApp::new(args.file)), args.port)
        .with_context(|| {
            format!(
                "cannot start the page on port {} (is it in use? pick another with --port)",
                args.port
            )
        })?;
    println!("results: {}  (Ctrl-C to stop)", server.url());
    server.wait();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_file_is_required_and_the_port_defaults_to_8100() {
        let args = ViewArgs::try_parse_from(["cli view", "results.json"]).unwrap();
        assert_eq!(args.file, PathBuf::from("results.json"));
        assert_eq!(args.port, 8100);
        assert!(ViewArgs::try_parse_from(["cli view"]).is_err());
        assert_eq!(
            ViewArgs::try_parse_from(["cli view", "r.json", "--port", "9000"])
                .unwrap()
                .port,
            9000
        );
    }

    #[test]
    fn a_missing_file_is_refused_before_serving() {
        let error = run(["definitely/not/here.json".to_owned()].into_iter()).unwrap_err();
        assert!(error.to_string().contains("here.json"), "{error}");
    }
}
