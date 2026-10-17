//! Starting a run: a new one, a warm start from an earlier one, and resuming from a
//! checkpoint.

use std::path::Path;

use neat::Population;

use super::{
    match_seed, Checkpoint, FeatureSet, Opponent, RunDir, TrainConfig, TrainError, Trainer,
    FEATURE_SET_VERSION, SCHEMA_VERSION,
};

impl Trainer {
    /// Starts a new run in `dir`.
    ///
    /// # Errors
    ///
    /// `TrainError::Config` for an invalid setup (or a directory that
    /// already holds a run), `TrainError::Io` if files cannot be written.
    pub fn new(
        config: TrainConfig,
        opponents: Vec<Opponent>,
        dir: &Path,
    ) -> Result<Self, TrainError> {
        Self::start(config, opponents, dir, None)
    }

    /// Starts a new run in `dir` from the final population of the run in
    /// `source`, which is only read. The new run has its own settings,
    /// seed and generation count (from 0); the population size must match
    /// the source's because its genomes are kept as they are.
    ///
    /// # Errors
    ///
    /// `TrainError::Checkpoint` if `source` holds no usable run,
    /// `TrainError::Mismatch` if it was trained on another feature set or
    /// has another population size, otherwise as `new`. Nothing is
    /// created in `dir` when the source is refused.
    pub fn new_from(
        config: TrainConfig,
        opponents: Vec<Opponent>,
        dir: &Path,
        source: &Path,
    ) -> Result<Self, TrainError> {
        Self::start(config, opponents, dir, Some(source))
    }

    /// Reads the run in `source` and checks that a warm start of a run
    /// with `population_size` genomes and the feature set `features` can
    /// be built on it, without writing
    /// anything. Callers use it to refuse a bad source before creating
    /// any file.
    ///
    /// # Errors
    ///
    /// `TrainError::Checkpoint` if `source` holds no usable run,
    /// `TrainError::Mismatch` for another feature set or population size
    /// (a warm start keeps the genomes as they are, so it cannot add or
    /// drop inputs: train with the source's `--features`).
    pub fn check_warm_source(
        source: &Path,
        population_size: usize,
        features: FeatureSet,
    ) -> Result<Checkpoint, TrainError> {
        let checkpoint = RunDir::open_existing(source)?.read_checkpoint()?;
        if checkpoint.feature_count != features.len()
            || checkpoint.feature_set_version != FEATURE_SET_VERSION
        {
            return Err(TrainError::Mismatch(format!(
                "{} was trained with {} features (feature-set version {}) but this run uses `--features {features}` ({} features, version {}); \
                 a warm start keeps its genomes as they are, so give it the same --features as the source (`--resume` can still continue it)",
                source.display(),
                checkpoint.feature_count,
                checkpoint.feature_set_version,
                features.len(),
                FEATURE_SET_VERSION
            )));
        }
        let source_size = checkpoint.config.neat.population_size;
        if source_size != population_size {
            return Err(TrainError::Mismatch(format!(
                "{} has a population of {source_size} but this run asks for {population_size}; \
                 a warm start keeps the genomes it is given",
                source.display()
            )));
        }
        Ok(checkpoint)
    }

    fn start(
        config: TrainConfig,
        opponents: Vec<Opponent>,
        dir: &Path,
        source: Option<&Path>,
    ) -> Result<Self, TrainError> {
        config.validate().map_err(TrainError::Config)?;
        if opponents.len() != config.opponent_specs.len() {
            return Err(TrainError::Config(format!(
                "{} opponents were built for {} specs",
                opponents.len(),
                config.opponent_specs.len()
            )));
        }
        let seed = match_seed(config.seed, u64::MAX, 0);
        // Everything that can refuse the source happens before the new
        // directory is touched.
        let (population, warm_started_from) = if let Some(source) = source {
            let checkpoint =
                Self::check_warm_source(source, config.neat.population_size, config.features)?;
            let population =
                Population::warm_start(checkpoint.population, config.neat.clone(), seed)
                    .map_err(|e| TrainError::Checkpoint(e.to_string()))?;
            (population, Some(source.display().to_string()))
        } else {
            let population = Population::new(config.features.len(), config.neat.clone(), seed)
                .map_err(|e| TrainError::Config(e.to_string()))?;
            (population, None)
        };
        let dir = RunDir::create_new(dir)?;
        dir.write_config(&config)?;
        let trainer = Self {
            config,
            opponents,
            dir,
            population,
            best: None,
            hall: Vec::new(),
            total_rounds: 0,
            elapsed_before: 0.0,
            resumed_from: None,
            warm_started_from,
        };
        // A run killed during generation 0 must still be resumable (and
        // must not look like a run to overwrite), so the directory holds a
        // checkpoint from the very start.
        trainer.dir.write_checkpoint(&trainer.checkpoint(0.0))?;
        Ok(trainer)
    }

    pub(super) fn checkpoint(&self, elapsed_secs: f64) -> Checkpoint {
        Checkpoint {
            schema_version: SCHEMA_VERSION,
            feature_count: self.config.features.len(),
            feature_set_version: FEATURE_SET_VERSION,
            config: self.config.clone(),
            opponent_names: self.opponents.iter().map(|o| o.name.clone()).collect(),
            population: self.population.snapshot(),
            best: self.best.clone(),
            hall_of_fame: self.hall.clone(),
            total_rounds: self.total_rounds,
            elapsed_secs,
        }
    }

    /// Continues the run in `dir` from its last checkpoint. `generations`
    /// optionally raises (or lowers) the total to run up to.
    ///
    /// # Errors
    ///
    /// `TrainError::Checkpoint` for unusable files, `TrainError::Mismatch`
    /// if `opponents` are not the pool the run started with.
    pub fn resume(
        dir: &Path,
        opponents: Vec<Opponent>,
        generations: Option<u32>,
    ) -> Result<Self, TrainError> {
        let dir = RunDir::open_existing(dir)?;
        let checkpoint = dir.read_checkpoint()?;
        if checkpoint.feature_count != checkpoint.config.features.len()
            || checkpoint.feature_set_version != FEATURE_SET_VERSION
        {
            return Err(TrainError::Mismatch(format!(
                "the run was trained with {} features (feature-set version {}) but this build has {} (version {}) for its `{}` feature set; \
                 its genomes cannot be continued",
                checkpoint.feature_count,
                checkpoint.feature_set_version,
                checkpoint.config.features.len(),
                FEATURE_SET_VERSION,
                checkpoint.config.features
            )));
        }
        let names: Vec<&str> = opponents.iter().map(|o| o.name.as_str()).collect();
        if names
            != checkpoint
                .opponent_names
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        {
            return Err(TrainError::Mismatch(format!(
                "the run used opponents {:?} but {:?} were given",
                checkpoint.opponent_names, names
            )));
        }
        let mut config = checkpoint.config;
        if let Some(total) = generations {
            config.generations = total;
        }
        config.validate().map_err(TrainError::Config)?;
        let population = Population::restore(checkpoint.population)
            .map_err(|e| TrainError::Checkpoint(e.to_string()))?;
        let completed = population.generation();
        dir.truncate_events_from(completed)?;
        Ok(Self {
            config,
            opponents,
            dir,
            population,
            best: checkpoint.best,
            hall: checkpoint.hall_of_fame,
            total_rounds: checkpoint.total_rounds,
            elapsed_before: checkpoint.elapsed_secs,
            resumed_from: Some(completed),
            warm_started_from: None,
        })
    }
}
