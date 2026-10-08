//! The one error type of the `neat` crate.

use std::fmt;

/// Everything that can go wrong at this crate's boundaries: a bad
/// configuration, or a genome (built by hand or read from JSON) that
/// breaks a structural invariant. Internal code treats invariant
/// violations as bugs and panics instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NeatError {
    InvalidConfig(String),
    InvalidGenome(String),
}

impl fmt::Display for NeatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(reason) => write!(f, "invalid NEAT config: {reason}"),
            Self::InvalidGenome(reason) => write!(f, "invalid genome: {reason}"),
        }
    }
}

impl std::error::Error for NeatError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_names_the_kind_and_reason() {
        let config = NeatError::InvalidConfig("population_size is 0".into());
        let genome = NeatError::InvalidGenome("cycle".into());
        assert_eq!(
            config.to_string(),
            "invalid NEAT config: population_size is 0"
        );
        assert_eq!(genome.to_string(), "invalid genome: cycle");
    }
}
