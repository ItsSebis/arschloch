//! `cli play`: play against trained models and the hand-written
//! strategies in the browser. The server holds the game; see
//! `web::PlayApp` and docs/PLAYING.md.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use clap::{Parser, ValueEnum};
use sim::{GenomeFile, NeatStrategy, Strategy};
use web::{CatalogEntry, PlayApp, RecordStore, Server};

use crate::args::{FixedStrategy, StrategyArg};

/// The committed baseline champion, built in so `cli play` works from a
/// release archive without any file.
const BUILT_IN_CHAMPION: &str = include_str!("../../docs/baselines/neat-v2/champion.json");

/// Play against the models in the browser.
#[derive(Parser, Debug)]
#[command(
    name = "cli play",
    about = "Play Arschloch in the browser against trained models and hand-written strategies",
    long_about = "Serves a game page on 127.0.0.1. Pick the table size and each opponent in the \
page; the server plays the other seats with the same strategies the simulator uses. Your finished \
games are kept in a records file. A built-in champion is always available; add your own trained \
models with --model."
)]
pub struct PlayArgs {
    /// A trained model to offer as an opponent: a genome file written by
    /// `cli train` (best.json, gen-NNNN.json) or a run directory (its
    /// best.json). Repeat for several.
    #[arg(long, value_name = "PATH")]
    pub model: Vec<PathBuf>,

    /// Port to serve on, on 127.0.0.1 only.
    #[arg(long, default_value_t = 8090)]
    pub port: u16,

    /// Where finished games are recorded (one JSON line each).
    #[arg(long, value_name = "FILE", default_value = "play-records.jsonl")]
    pub records: PathBuf,
}

/// The genome file a `--model` path names.
fn genome_path(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.join("best.json")
    } else {
        path.to_owned()
    }
}

fn model_label(path: &Path) -> String {
    let named = if path.is_dir() {
        path.file_name()
    } else {
        path.file_stem()
    };
    named.map_or_else(|| "model".into(), |s| s.to_string_lossy().into_owned())
}

fn load_model(label: &str, path: &Path) -> anyhow::Result<NeatStrategy> {
    let file = GenomeFile::load(&genome_path(path))
        .with_context(|| format!("cannot use --model {}", path.display()))?;
    NeatStrategy::new(format!("Neat({label})"), &file.genome)
        .with_context(|| format!("cannot use --model {}", path.display()))
}

/// The hand-written strategies (and two adaptive presets), the built-in
/// champion, then the `--model`s.
///
/// # Errors
///
/// A `--model` that cannot be loaded.
pub fn build_catalog(models: &[PathBuf]) -> anyhow::Result<Vec<CatalogEntry>> {
    let mut catalog: Vec<CatalogEntry> = Vec::new();
    for variant in FixedStrategy::value_variants() {
        let id = variant
            .to_possible_value()
            .map(|v| v.get_name().to_owned())
            .context("strategy without a name")?;
        let strategy: Arc<dyn Strategy> = StrategyArg::Fixed(*variant).build();
        let label = strategy.name().to_owned();
        catalog.push(CatalogEntry::strategy(&id, &label, strategy));
    }
    for spec in ["adaptive:reading,tempo,bully", "adaptive:counting,reading"] {
        let strategy = spec
            .parse::<StrategyArg>()
            .map_err(anyhow::Error::msg)?
            .build();
        let label = strategy.name().to_owned();
        catalog.push(CatalogEntry::strategy(spec, &label, strategy));
    }
    let champion = GenomeFile::from_json(BUILT_IN_CHAMPION).context("built-in champion")?;
    catalog.push(CatalogEntry::model(
        "model:champion-v2",
        "Neat(champion-v2)",
        Arc::new(NeatStrategy::new("Neat(champion-v2)", &champion.genome)?),
    ));
    for (index, path) in models.iter().enumerate() {
        let mut label = model_label(path);
        // Two models with the same name must stay distinguishable.
        if catalog.iter().any(|e| e.label == format!("Neat({label})")) {
            label = format!("{label}-{}", index + 1);
        }
        let model = load_model(&label, path)?;
        catalog.push(CatalogEntry::model(
            &format!("model:{label}"),
            &format!("Neat({label})"),
            Arc::new(model),
        ));
    }
    Ok(catalog)
}

pub fn run(raw_args: impl Iterator<Item = String>) -> anyhow::Result<()> {
    let args = PlayArgs::parse_from(std::iter::once("cli play".to_owned()).chain(raw_args));
    // Everything that can fail on input fails before the port is bound.
    let catalog = build_catalog(&args.model)?;
    let app = PlayApp::new(catalog, RecordStore::new(Some(args.records.clone())));
    let server = Server::start(Arc::new(app), args.port).with_context(|| {
        format!(
            "cannot start the game on port {} (is it in use? pick another with --port)",
            args.port
        )
    })?;
    println!("play: {}  (Ctrl-C to stop)", server.url());
    println!("results are kept in {}", args.records.display());
    server.wait();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_catalog_has_the_classics_the_champion_and_the_given_models() {
        let genome = std::env::temp_dir().join(format!("play-cat-{}.json", std::process::id()));
        std::fs::write(&genome, BUILT_IN_CHAMPION).unwrap();
        let catalog = build_catalog(&[genome.clone(), genome.clone()]).unwrap();
        let ids: Vec<&str> = catalog.iter().map(|e| e.id.as_str()).collect();
        for expected in [
            "lowest-legal",
            "card-counter",
            "endgame-denial",
            "adaptive:reading,tempo,bully",
            "model:champion-v2",
        ] {
            assert!(ids.contains(&expected), "{expected} in {ids:?}");
        }
        // Two models with the same file name stay distinct.
        let stem = genome.file_stem().unwrap().to_string_lossy().into_owned();
        assert!(ids.contains(&format!("model:{stem}").as_str()));
        assert!(ids.contains(&format!("model:{stem}-2").as_str()), "{ids:?}");
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len());
        std::fs::remove_file(genome).ok();
    }

    #[test]
    fn a_bad_model_is_an_error_naming_the_path() {
        let error = build_catalog(&[PathBuf::from("/nonexistent/x.json")])
            .err()
            .unwrap();
        assert!(
            format!("{error:#}").contains("/nonexistent/x.json"),
            "{error:#}"
        );
    }

    #[test]
    fn a_run_directory_means_its_best_json() {
        assert_eq!(genome_path(Path::new("runs/a")), Path::new("runs/a")); // not a dir here
        let dir = std::env::temp_dir().join(format!("play-dir-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(genome_path(&dir), dir.join("best.json"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn the_defaults_are_a_local_port_and_a_records_file() {
        let args = PlayArgs::try_parse_from(["cli play"]).unwrap();
        assert_eq!((args.port, args.model.len()), (8090, 0));
        assert_eq!(args.records, PathBuf::from("play-records.jsonl"));
    }
}
