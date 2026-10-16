//! The reference players the candidate opponents are scored against.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use sim::training::RunDir;
use sim::{NeatStrategy, Strategy};

use super::AdvisorArgs;
use crate::args::StrategyArg;

/// Reference players at most: the probe matches of a candidate are split over them.
pub(super) const MAX_REFERENCES: usize = 6;

/// `count` of `items`, spread evenly over the list (all of them when it is short enough).
pub(super) fn thin<T: Clone>(items: &[T], count: usize) -> Vec<T> {
    if items.len() <= count {
        return items.to_vec();
    }
    (0..count)
        .map(|i| items[i * items.len() / count].clone())
        .collect()
}

/// A player the candidates are scored against.
#[derive(Clone)]
pub(super) struct Reference {
    pub(super) name: String,
    pub(super) strategy: Arc<dyn Strategy>,
}

/// The reference players the options ask for; `adaptive` when none was asked for.
pub(super) fn build_references(args: &AdvisorArgs) -> anyhow::Result<Vec<Reference>> {
    let mut found = Vec::new();
    for spec in &args.reference {
        let strategy = spec
            .parse::<StrategyArg>()
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("invalid --reference `{spec}`"))?
            .build();
        found.push(Reference {
            name: spec.clone(),
            strategy,
        });
    }
    if let Some(folder) = &args.reference_models {
        found.extend(models_in(folder)?);
    }
    if let Some(run) = &args.reference_hall {
        found.extend(hall_of(run)?);
    }
    if found.is_empty() {
        found.push(Reference {
            name: "adaptive".to_owned(),
            strategy: "adaptive"
                .parse::<StrategyArg>()
                .map_err(anyhow::Error::msg)?
                .build(),
        });
    }
    Ok(thin(&found, MAX_REFERENCES))
}

/// Every genome file of `folder` that this build can load, by file name.
pub(super) fn models_in(folder: &Path) -> anyhow::Result<Vec<Reference>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(folder)
        .with_context(|| {
            format!(
                "--reference-models {}: cannot read the folder",
                folder.display()
            )
        })?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    let models: Vec<Reference> = files
        .iter()
        .filter_map(|path| {
            let strategy = NeatStrategy::from_file(path).ok()?;
            let name = path.file_stem()?.to_string_lossy().into_owned();
            Some(Reference {
                name: format!("neat:{name}"),
                strategy: Arc::new(strategy),
            })
        })
        .collect();
    anyhow::ensure!(
        !models.is_empty(),
        "--reference-models {}: no genome file (*.json written by `cli train`) that this build can load",
        folder.display()
    );
    Ok(models)
}

/// The hall-of-fame champions stored in the checkpoint of an earlier run.
pub(super) fn hall_of(run: &Path) -> anyhow::Result<Vec<Reference>> {
    let checkpoint = RunDir::open_existing(run)
        .and_then(|dir| dir.read_checkpoint())
        .with_context(|| format!("--reference-hall {}: not a run folder", run.display()))?;
    anyhow::ensure!(
        !checkpoint.hall_of_fame.is_empty(),
        "--reference-hall {}: the run has no hall of fame yet (it needs --hall-of-fame and a few generations)",
        run.display()
    );
    checkpoint
        .hall_of_fame
        .iter()
        .map(|member| {
            let name = format!("HoF({})", member.generation);
            let strategy = NeatStrategy::new(name.clone(), &member.genome)
                .with_context(|| format!("--reference-hall {}: {name}", run.display()))?;
            Ok(Reference {
                name,
                strategy: Arc::new(strategy),
            })
        })
        .collect()
}
