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

use super::features::{FEATURE_COUNT, FEATURE_NAMES};

/// Bumped when the file layout (not the feature set) changes.
pub const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenomeFile {
    pub format_version: u32,
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
    /// Wraps `genome` for saving.
    ///
    /// # Errors
    ///
    /// Returns `GenomeFileError::Mismatch` if the genome does not take
    /// exactly `FEATURE_COUNT` inputs.
    pub fn new(genome: Genome) -> Result<Self, GenomeFileError> {
        let file = Self {
            format_version: FORMAT_VERSION,
            feature_names: FEATURE_NAMES.iter().map(|&n| n.to_owned()).collect(),
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
        if self.feature_names != FEATURE_NAMES {
            return Err(GenomeFileError::Mismatch(format!(
                "trained on features {:?}, this build uses {FEATURE_NAMES:?}",
                self.feature_names
            )));
        }
        if self.genome.num_inputs() != FEATURE_COUNT {
            return Err(GenomeFileError::Mismatch(format!(
                "genome takes {} inputs, this build has {FEATURE_COUNT} features",
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

    #[test]
    fn an_unknown_format_version_is_refused() {
        let mut file = GenomeFile::new(sample_genome()).unwrap();
        file.format_version = 99;
        let error = GenomeFile::from_json(&serde_json::to_string(&file).unwrap()).unwrap_err();
        assert!(matches!(error, GenomeFileError::Mismatch(_)), "{error}");
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
