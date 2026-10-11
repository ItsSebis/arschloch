//! The on-disk form of a trained genome.
//!
//! A genome is only meaningful together with the feature set it was
//! trained against, so the file records the feature names and the load
//! refuses a file whose features differ from this build's: a stale
//! genome fails loudly instead of silently playing garbage.

use std::fmt;
use std::path::Path;

use neat::Genome;
use serde::{Deserialize, Serialize};

use super::feature_set::{FeatureSet, FEATURE_NAMES_V2};
use super::features::FEATURE_SET_VERSION;

/// Bumped when the file layout (not the feature set) changes.
pub const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenomeFile {
    pub format_version: u32,
    /// See `FEATURE_SET_VERSION`: catches changed feature semantics that
    /// the names alone cannot.
    pub feature_set_version: u32,
    pub feature_names: Vec<String>,
    pub genome: Genome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenomeFileError {
    Io(String),
    Parse(String),
    /// The file is well-formed but does not fit this build.
    Mismatch(String),
}

impl fmt::Display for GenomeFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(reason) => write!(f, "cannot access genome file: {reason}"),
            Self::Parse(reason) => write!(f, "malformed genome file: {reason}"),
            Self::Mismatch(reason) => write!(f, "genome file does not fit this build: {reason}"),
        }
    }
}

impl std::error::Error for GenomeFileError {}

impl GenomeFile {
    /// Wraps `genome` for saving, recording the names of the feature set
    /// its input count identifies.
    ///
    /// # Errors
    ///
    /// Returns `GenomeFileError::Mismatch` if the genome's input count is
    /// not that of a `FeatureSet`.
    pub fn new(genome: Genome) -> Result<Self, GenomeFileError> {
        let inputs = genome.num_inputs();
        let features = FeatureSet::from_input_count(inputs).ok_or_else(|| {
            GenomeFileError::Mismatch(format!(
                "genome takes {inputs} inputs, which is no feature set of this build"
            ))
        })?;
        let file = Self {
            format_version: FORMAT_VERSION,
            feature_set_version: FEATURE_SET_VERSION,
            feature_names: features.names().iter().map(|&n| n.to_owned()).collect(),
            genome,
        };
        file.check_fits_this_build()?;
        Ok(file)
    }

    fn check_fits_this_build(&self) -> Result<(), GenomeFileError> {
        if self.format_version != FORMAT_VERSION {
            return Err(GenomeFileError::Mismatch(format!(
                "format version {} (this build reads {FORMAT_VERSION})",
                self.format_version
            )));
        }
        if self.feature_set_version != FEATURE_SET_VERSION {
            return Err(GenomeFileError::Mismatch(format!(
                "feature set version {} (this build uses {FEATURE_SET_VERSION}): the features \
                 keep their names but were rescaled or redefined since this genome was trained",
                self.feature_set_version
            )));
        }
        let Some(features) = FeatureSet::from_names(&self.feature_names) else {
            return Err(GenomeFileError::Mismatch(format!(
                "trained on features {:?}, which is not a prefix of this build's {FEATURE_NAMES_V2:?} \
                 ending at a group boundary ({}, {}, {} or {} features)",
                self.feature_names,
                FeatureSet::Base.len(),
                FeatureSet::Trick.len(),
                FeatureSet::Hidden.len(),
                FeatureSet::All.len(),
            )));
        };
        if self.genome.num_inputs() != features.len() {
            return Err(GenomeFileError::Mismatch(format!(
                "the file lists {} features but its genome takes {} inputs",
                features.len(),
                self.genome.num_inputs()
            )));
        }
        Ok(())
    }

    /// Parses and checks a genome file's JSON text.
    ///
    /// # Errors
    ///
    /// `Parse` for malformed JSON or a structurally invalid genome,
    /// `Mismatch` for a file trained against a different feature set.
    pub fn from_json(text: &str) -> Result<Self, GenomeFileError> {
        let file: Self =
            serde_json::from_str(text).map_err(|e| GenomeFileError::Parse(e.to_string()))?;
        file.check_fits_this_build()?;
        Ok(file)
    }

    /// # Errors
    ///
    /// `Io` if the file cannot be read, otherwise as `from_json`.
    pub fn load(path: &Path) -> Result<Self, GenomeFileError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| GenomeFileError::Io(format!("{}: {e}", path.display())))?;
        Self::from_json(&text)
    }

    /// Writes pretty-printed JSON, replacing any existing file.
    ///
    /// # Errors
    ///
    /// `Io` if the file cannot be written.
    pub fn save(&self, path: &Path) -> Result<(), GenomeFileError> {
        let text = serde_json::to_string_pretty(self)
            .map_err(|e| GenomeFileError::Parse(e.to_string()))?;
        std::fs::write(path, text)
            .map_err(|e| GenomeFileError::Io(format!("{}: {e}", path.display())))
    }
}

#[cfg(test)]
mod tests {
    use neat::{InnovationTracker, NeatConfig};
    use rand::SeedableRng;

    use super::*;
    use crate::strategies::neat_player::features::{FEATURE_COUNT, FEATURE_SET_VERSION};

    fn sample_genome() -> Genome {
        let mut tracker = InnovationTracker::new(u32::try_from(FEATURE_COUNT).unwrap() + 2);
        Genome::minimal(
            FEATURE_COUNT,
            &mut tracker,
            &NeatConfig::default(),
            &mut rand::rngs::StdRng::seed_from_u64(1),
        )
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "arschloch-genome-{name}-{}.json",
            std::process::id()
        ))
    }

    #[test]
    fn save_then_load_round_trips() {
        let file = GenomeFile::new(sample_genome()).unwrap();
        let path = temp_path("roundtrip");
        file.save(&path).unwrap();
        let loaded = GenomeFile::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(loaded, file);
    }

    #[test]
    fn a_genome_with_the_wrong_input_count_is_refused() {
        let mut tracker = InnovationTracker::new(5);
        let genome = Genome::minimal(
            3,
            &mut tracker,
            &NeatConfig::default(),
            &mut rand::rngs::StdRng::seed_from_u64(1),
        );
        let error = GenomeFile::new(genome).unwrap_err();
        assert!(matches!(error, GenomeFileError::Mismatch(_)), "{error}");
    }

    #[test]
    fn a_file_trained_on_other_features_is_refused_with_both_lists_named() {
        let mut file = GenomeFile::new(sample_genome()).unwrap();
        file.feature_names[0] = "something_else".into();
        let error = GenomeFile::from_json(&serde_json::to_string(&file).unwrap()).unwrap_err();
        let GenomeFileError::Mismatch(reason) = &error else {
            panic!("expected Mismatch, got {error}");
        };
        assert!(
            reason.contains("something_else") && reason.contains("is_pass"),
            "{reason}"
        );
    }

    fn genome_with(inputs: usize) -> Genome {
        let mut tracker = InnovationTracker::new(u32::try_from(inputs).unwrap() + 2);
        Genome::minimal(
            inputs,
            &mut tracker,
            &NeatConfig::default(),
            &mut rand::rngs::StdRng::seed_from_u64(1),
        )
    }

    #[test]
    fn every_feature_set_round_trips_and_records_exactly_its_names() {
        for set in FeatureSet::ALL_SETS {
            let file = GenomeFile::new(genome_with(set.len())).unwrap();
            assert_eq!(file.feature_names, set.names());
            assert_eq!(file.feature_set_version, FEATURE_SET_VERSION);
            let loaded = GenomeFile::from_json(&serde_json::to_string(&file).unwrap()).unwrap();
            assert_eq!(loaded, file);
        }
        // A v1 file is exactly what it always was.
        let v1 = GenomeFile::new(genome_with(FEATURE_COUNT)).unwrap();
        assert_eq!(v1.feature_names.len(), FEATURE_COUNT);
        assert_eq!(v1.feature_names[19], "opponents_locked_out");
    }

    #[test]
    fn names_out_of_order_or_beyond_the_known_list_are_refused() {
        let mut file = GenomeFile::new(genome_with(FeatureSet::Trick.len())).unwrap();
        file.feature_names.swap(20, 21);
        assert!(matches!(
            GenomeFile::from_json(&serde_json::to_string(&file).unwrap()),
            Err(GenomeFileError::Mismatch(_))
        ));
        let mut longer = GenomeFile::new(genome_with(FeatureSet::All.len())).unwrap();
        longer.feature_names.push("brand_new".into());
        longer.genome = genome_with(FeatureSet::All.len() + 1);
        let error = GenomeFile::from_json(&serde_json::to_string(&longer).unwrap()).unwrap_err();
        assert!(matches!(error, GenomeFileError::Mismatch(_)), "{error}");
        // A prefix that stops inside a group is not a feature set.
        let mut cut = GenomeFile::new(genome_with(FEATURE_COUNT)).unwrap();
        cut.feature_names = FEATURE_NAMES_V2[..22]
            .iter()
            .map(|n| (*n).to_owned())
            .collect();
        cut.genome = genome_with(22);
        assert!(matches!(
            GenomeFile::from_json(&serde_json::to_string(&cut).unwrap()),
            Err(GenomeFileError::Mismatch(_))
        ));
    }

    #[test]
    fn names_and_input_count_must_agree() {
        // 20 names but a genome of 22 inputs, and the other way round.
        let mut file = GenomeFile::new(genome_with(FEATURE_COUNT)).unwrap();
        file.genome = genome_with(22);
        let error = GenomeFile::from_json(&serde_json::to_string(&file).unwrap()).unwrap_err();
        let GenomeFileError::Mismatch(reason) = &error else {
            panic!("expected Mismatch, got {error}");
        };
        assert!(reason.contains("20") && reason.contains("22"), "{reason}");
        let mut file = GenomeFile::new(genome_with(FeatureSet::Trick.len())).unwrap();
        file.genome = genome_with(FEATURE_COUNT);
        assert!(GenomeFile::from_json(&serde_json::to_string(&file).unwrap()).is_err());
        assert!(GenomeFile::new(genome_with(22)).is_err());
    }

    #[test]
    fn an_unknown_format_version_is_refused() {
        let mut file = GenomeFile::new(sample_genome()).unwrap();
        file.format_version = 99;
        let error = GenomeFile::from_json(&serde_json::to_string(&file).unwrap()).unwrap_err();
        assert!(matches!(error, GenomeFileError::Mismatch(_)), "{error}");
    }

    #[test]
    fn a_file_from_another_feature_set_version_is_refused() {
        // Same names, same order, but a changed scale or formula: only
        // the version can tell, and the genome would silently misplay.
        let mut value = serde_json::to_value(GenomeFile::new(sample_genome()).unwrap()).unwrap();
        value["feature_set_version"] = serde_json::json!(FEATURE_SET_VERSION + 1);
        let error = GenomeFile::from_json(&value.to_string()).unwrap_err();
        let GenomeFileError::Mismatch(reason) = &error else {
            panic!("expected Mismatch, got {error}");
        };
        assert!(reason.contains("feature set version"), "{reason}");
    }

    #[test]
    fn a_file_without_a_feature_set_version_is_refused() {
        let mut value = serde_json::to_value(GenomeFile::new(sample_genome()).unwrap()).unwrap();
        value.as_object_mut().unwrap().remove("feature_set_version");
        assert!(matches!(
            GenomeFile::from_json(&value.to_string()),
            Err(GenomeFileError::Parse(_))
        ));
    }

    #[test]
    fn garbage_and_invalid_genomes_are_parse_errors_not_panics() {
        assert!(matches!(
            GenomeFile::from_json("not json"),
            Err(GenomeFileError::Parse(_))
        ));
        let file = GenomeFile::new(sample_genome()).unwrap();
        let mut value = serde_json::to_value(&file).unwrap();
        value["genome"]["connections"][0]["to"] = serde_json::json!(9999);
        assert!(matches!(
            GenomeFile::from_json(&value.to_string()),
            Err(GenomeFileError::Parse(_))
        ));
    }

    #[test]
    fn a_missing_file_is_an_io_error_naming_the_path() {
        let error = GenomeFile::load(Path::new("/nonexistent/genome.json")).unwrap_err();
        let GenomeFileError::Io(reason) = &error else {
            panic!("expected Io, got {error}");
        };
        assert!(reason.contains("/nonexistent/genome.json"), "{reason}");
    }
}
