//! What training maximises: the fitness variants and the sums they are
//! computed from.
//!
//! Every variant is a function of the candidate's *per-round role scores*
//! `s` (+1 for President .. -1 for last, linear in the place, so already
//! comparable across table sizes) over the rounds of an evaluation. An
//! evaluation therefore keeps four running sums (`Moments`); two
//! evaluations of the same genome can be pooled by adding their sums,
//! which is how racing combines its estimates (see `trainer::racing`).

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// The scalar the evolution maximises.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitnessKind {
    /// The mean role score (the only fitness before Phase 13).
    #[default]
    Mean,
    /// `mean - lambda * spread`: rewards consistency (`lambda >= 0`).
    Risk { lambda: f64 },
    /// The mean of `tanh(kappa * s) / tanh(kappa)`: concave at the top, so
    /// the step from Vize to President is worth less than the step from
    /// Arschloch to Vize (`kappa > 0`).
    Tanh { kappa: f64 },
}

impl FitnessKind {
    #[must_use]
    pub fn is_mean(&self) -> bool {
        matches!(self, Self::Mean)
    }

    /// # Errors
    ///
    /// Returns a message naming the invalid parameter.
    pub fn validate(&self) -> Result<(), String> {
        match *self {
            Self::Mean => Ok(()),
            Self::Risk { lambda } if lambda.is_finite() && lambda >= 0.0 => Ok(()),
            Self::Risk { lambda } => Err(format!(
                "fitness risk:{lambda}: lambda must be a finite number >= 0"
            )),
            Self::Tanh { kappa } if kappa.is_finite() && kappa > 0.0 => Ok(()),
            Self::Tanh { kappa } => Err(format!(
                "fitness tanh:{kappa}: kappa must be a finite number > 0"
            )),
        }
    }

    /// The value one round's role score `s` contributes to `Moments::sum_g`:
    /// `tanh(kappa * s) / tanh(kappa)` for `Tanh`, 0 for the others (which
    /// do not use it).
    #[must_use]
    pub fn squash(&self, s: f64) -> f64 {
        match *self {
            Self::Tanh { kappa } => (kappa * s).tanh() / kappa.tanh(),
            _ => 0.0,
        }
    }
}

impl fmt::Display for FitnessKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mean => write!(f, "mean"),
            Self::Risk { lambda } => write!(f, "risk:{lambda}"),
            Self::Tanh { kappa } => write!(f, "tanh:{kappa}"),
        }
    }
}

impl FromStr for FitnessKind {
    type Err = String;

    /// `mean`, `risk:L` or `tanh:K`.
    fn from_str(text: &str) -> Result<Self, String> {
        let number = |what: &str, value: &str| {
            value
                .trim()
                .parse::<f64>()
                .map_err(|e| format!("`{text}`: {what} `{value}` is not a number ({e})"))
        };
        let kind = match text.trim().split_once(':') {
            None if text.trim() == "mean" => Self::Mean,
            Some(("risk", value)) => Self::Risk {
                lambda: number("lambda", value)?,
            },
            Some(("tanh", value)) => Self::Tanh {
                kappa: number("kappa", value)?,
            },
            _ => {
                return Err(format!(
                    "`{text}` is not a fitness: use `mean`, `risk:LAMBDA` (for example risk:0.5) or `tanh:KAPPA` (for example tanh:4)"
                ))
            }
        };
        kind.validate()?;
        Ok(kind)
    }
}

/// The running sums of the per-round role scores of an evaluation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Moments {
    pub rounds: u64,
    /// Sum of `s`.
    pub sum: f64,
    /// Sum of `s * s`.
    pub sum_sq: f64,
    /// Sum of `FitnessKind::squash(s)` (only kept for `Tanh`).
    pub sum_g: f64,
}

impl Moments {
    pub fn add_round(&mut self, s: f64, g: f64) {
        self.rounds += 1;
        self.sum += s;
        self.sum_sq += s * s;
        self.sum_g += g;
    }

    /// The sums of both evaluations together: the moments of one evaluation
    /// that had played all their rounds.
    #[must_use]
    pub fn pooled(&self, other: &Self) -> Self {
        Self {
            rounds: self.rounds + other.rounds,
            sum: self.sum + other.sum,
            sum_sq: self.sum_sq + other.sum_sq,
            sum_g: self.sum_g + other.sum_g,
        }
    }

    #[allow(clippy::cast_precision_loss)] // round counts are far below 2^52
    fn count(&self) -> f64 {
        self.rounds as f64
    }

    #[must_use]
    pub fn mean(&self) -> f64 {
        self.sum / self.count()
    }

    /// The (population) standard deviation of the per-round role score:
    /// `sqrt(sum_sq / n - mean^2)`.
    #[must_use]
    pub fn spread(&self) -> f64 {
        let mean = self.mean();
        (self.sum_sq / self.count() - mean * mean).max(0.0).sqrt()
    }

    /// The fitness these sums give: `mean`, `mean - lambda * spread` or
    /// `sum_g / n`.
    #[must_use]
    pub fn fitness(&self, kind: &FitnessKind) -> f64 {
        match *kind {
            FitnessKind::Mean => self.mean(),
            FitnessKind::Risk { lambda } => self.mean() - lambda * self.spread(),
            FitnessKind::Tanh { .. } => self.sum_g / self.count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moments(kind: &FitnessKind, scores: &[f64]) -> Moments {
        let mut m = Moments::default();
        for &s in scores {
            m.add_round(s, kind.squash(s));
        }
        m
    }

    #[test]
    fn parsing_and_display_round_trip_and_errors_are_clear() {
        for (text, kind) in [
            ("mean", FitnessKind::Mean),
            ("risk:0.5", FitnessKind::Risk { lambda: 0.5 }),
            ("risk:0", FitnessKind::Risk { lambda: 0.0 }),
            ("tanh:4", FitnessKind::Tanh { kappa: 4.0 }),
        ] {
            assert_eq!(text.parse::<FitnessKind>(), Ok(kind));
            assert_eq!(kind.to_string(), text);
        }
        for bad in [
            "", "median", "risk", "risk:", "risk:-1", "risk:x", "tanh:0", "tanh:-2", "tanh:nan",
            "risk:inf", "mean:3",
        ] {
            let error = bad.parse::<FitnessKind>().unwrap_err();
            assert!(!error.is_empty(), "{bad}");
        }
        assert!("tanh:0"
            .parse::<FitnessKind>()
            .unwrap_err()
            .contains("kappa"));
        assert!("risk:-1"
            .parse::<FitnessKind>()
            .unwrap_err()
            .contains("lambda"));
    }

    #[test]
    fn serde_uses_snake_case_tags() {
        assert_eq!(
            serde_json::to_string(&FitnessKind::Mean).unwrap(),
            "\"mean\""
        );
        assert_eq!(
            serde_json::to_string(&FitnessKind::Risk { lambda: 0.5 }).unwrap(),
            r#"{"risk":{"lambda":0.5}}"#
        );
        let back: FitnessKind = serde_json::from_str(r#"{"tanh":{"kappa":4.0}}"#).unwrap();
        assert_eq!(back, FitnessKind::Tanh { kappa: 4.0 });
    }

    #[test]
    fn risk_is_the_mean_minus_lambda_standard_deviations_by_hand() {
        // Scores 1, 0, -1, 0: mean 0, mean of squares 0.5, so spread sqrt(0.5).
        let kind = FitnessKind::Risk { lambda: 0.5 };
        let m = moments(&kind, &[1.0, 0.0, -1.0, 0.0]);
        assert!((m.mean()).abs() < 1e-15);
        assert!((m.spread() - 0.5f64.sqrt()).abs() < 1e-15);
        assert!((m.fitness(&kind) + 0.5 * 0.5f64.sqrt()).abs() < 1e-15);
        // Scores 1, 1, 0, 0: mean 0.5, mean of squares 0.5, variance 0.25.
        let m = moments(&kind, &[1.0, 1.0, 0.0, 0.0]);
        assert!((m.spread() - 0.5).abs() < 1e-15);
        assert!((m.fitness(&kind) - 0.25).abs() < 1e-15);
        // A constant score has no spread, whatever lambda.
        let m = moments(&kind, &[0.5; 7]);
        assert!(m.spread().abs() < 1e-7);
        assert!((m.fitness(&kind) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn tanh_squashes_each_round_before_averaging_by_hand() {
        let kind = FitnessKind::Tanh { kappa: 2.0 };
        let g = |s: f64| (2.0 * s).tanh() / 2.0f64.tanh();
        let m = moments(&kind, &[1.0, 0.5, -1.0]);
        let expected = (g(1.0) + g(0.5) + g(-1.0)) / 3.0;
        assert!((m.fitness(&kind) - expected).abs() < 1e-15);
        // g(1) = 1 and g(-1) = -1, g is odd and concave above zero.
        assert!((g(1.0) - 1.0).abs() < 1e-15 && (g(-1.0) + 1.0).abs() < 1e-15);
        assert!(g(0.5) > 0.5, "the top is flattened: {}", g(0.5));
        // The design example: with kappa 4 a Vize (+1/3 at four seats) is
        // worth about 0.87 of a President.
        let four = FitnessKind::Tanh { kappa: 4.0 };
        assert!((four.squash(1.0 / 3.0) - 0.871).abs() < 1e-3);
    }

    #[test]
    fn mean_is_the_plain_mean_and_pooling_adds_the_sums() {
        let kind = FitnessKind::Risk { lambda: 1.0 };
        let a = moments(&kind, &[1.0, 0.0]);
        let b = moments(&kind, &[-1.0, 1.0, 1.0, 0.0]);
        let all = moments(&kind, &[1.0, 0.0, -1.0, 1.0, 1.0, 0.0]);
        assert_eq!(a.pooled(&b), all);
        assert!((all.fitness(&FitnessKind::Mean) - 2.0 / 6.0).abs() < 1e-15);
        // Pool the moments, not the scalars: the pooled risk fitness is not
        // the weighted mean of the two risk fitnesses.
        let weighted = (2.0 * a.fitness(&kind) + 4.0 * b.fitness(&kind)) / 6.0;
        assert!((all.fitness(&kind) - weighted).abs() > 1e-3);
    }

    #[test]
    fn default_is_mean_and_validation_rejects_bad_parameters() {
        assert_eq!(FitnessKind::default(), FitnessKind::Mean);
        assert!(FitnessKind::Risk { lambda: f64::NAN }.validate().is_err());
        assert!(FitnessKind::Tanh { kappa: 0.0 }.validate().is_err());
        assert!(FitnessKind::Tanh { kappa: 3.0 }.validate().is_ok());
    }
}
