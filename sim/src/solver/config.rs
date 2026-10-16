//! Strategy names and the options of `endgame-solver[:OPTIONS]`.

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use crate::strategies::{
    Adaptive, AdaptiveConfig, CardCounter, EndgameDenial, GreedyHighest, HoldBackPairs, LowestLegal,
};
use crate::strategy::Strategy;

use super::SolverConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategyName {
    LowestLegal,
    GreedyHighest,
    HoldBackPairs,
    CardCounter,
    EndgameDenial,
    Adaptive,
}

impl StrategyName {
    #[must_use]
    pub fn build(self) -> Arc<dyn Strategy> {
        match self {
            Self::LowestLegal => Arc::new(LowestLegal),
            Self::GreedyHighest => Arc::new(GreedyHighest),
            Self::HoldBackPairs => Arc::new(HoldBackPairs),
            Self::CardCounter => Arc::new(CardCounter),
            Self::EndgameDenial => Arc::new(EndgameDenial),
            Self::Adaptive => Arc::new(Adaptive::new(AdaptiveConfig::default())),
        }
    }
}

impl fmt::Display for StrategyName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::LowestLegal => "lowest-legal",
            Self::GreedyHighest => "greedy-highest",
            Self::HoldBackPairs => "hold-back-pairs",
            Self::CardCounter => "card-counter",
            Self::EndgameDenial => "endgame-denial",
            Self::Adaptive => "adaptive",
        })
    }
}

impl FromStr for StrategyName {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        Ok(match text.trim() {
            "lowest-legal" => Self::LowestLegal,
            "greedy-highest" => Self::GreedyHighest,
            "hold-back-pairs" => Self::HoldBackPairs,
            "card-counter" => Self::CardCounter,
            "endgame-denial" => Self::EndgameDenial,
            "adaptive" => Self::Adaptive,
            other => {
                return Err(format!(
                    "unknown fallback `{other}`; expected lowest-legal, greedy-highest, \
                     hold-back-pairs, card-counter, endgame-denial or adaptive"
                ))
            }
        })
    }
}

/// The options of `endgame-solver[:OPTIONS]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolverSpec {
    pub solver: SolverConfig,
    pub fallback: StrategyName,
}

impl Default for SolverSpec {
    fn default() -> Self {
        Self {
            solver: SolverConfig::default(),
            fallback: StrategyName::Adaptive,
        }
    }
}

impl FromStr for SolverSpec {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        let mut spec = Self::default();
        for option in text.split(',').map(str::trim).filter(|o| !o.is_empty()) {
            let (key, value) = option
                .split_once('=')
                .ok_or_else(|| format!("`{option}` is not KEY=VALUE"))?;
            match key {
                "max-unseen" => {
                    spec.solver.max_unseen = value
                        .parse()
                        .map_err(|_| format!("max-unseen: `{value}` is not a number"))?;
                }
                "budget" => {
                    spec.solver.node_budget = value
                        .parse()
                        .map_err(|_| format!("budget: `{value}` is not a number"))?;
                }
                "fallback" => spec.fallback = value.parse()?,
                other => {
                    return Err(format!(
                        "unknown solver option `{other}`; expected max-unseen, budget or fallback"
                    ))
                }
            }
        }
        Ok(spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_parse_and_unknown_ones_are_rejected() {
        let spec: SolverSpec = "max-unseen=8,budget=100,fallback=lowest-legal"
            .parse()
            .unwrap();
        assert_eq!(spec.solver.max_unseen, 8);
        assert_eq!(spec.solver.node_budget, 100);
        assert_eq!(spec.fallback, StrategyName::LowestLegal);
        assert_eq!("".parse::<SolverSpec>().unwrap(), SolverSpec::default());
        assert!("depth=3".parse::<SolverSpec>().is_err());
        assert!("max-unseen=x".parse::<SolverSpec>().is_err());
        assert!("fallback=nope".parse::<SolverSpec>().is_err());
    }

    #[test]
    fn strategy_names_round_trip() {
        for name in [
            StrategyName::LowestLegal,
            StrategyName::GreedyHighest,
            StrategyName::HoldBackPairs,
            StrategyName::CardCounter,
            StrategyName::EndgameDenial,
            StrategyName::Adaptive,
        ] {
            assert_eq!(name.to_string().parse::<StrategyName>().unwrap(), name);
        }
    }
}
