//! Strategy names and the options of `endgame-solver[:OPTIONS]`.

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use crate::strategies::{
    Adaptive, AdaptiveConfig, CardCounter, EndgameDenial, GreedyHighest, HoldBackPairs, LowestLegal,
};
use crate::strategy::Strategy;

use super::sampler::{LeafSpec, SamplerConfig, MAX_DEPTH};
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

/// The suffix that turns any strategy spec into "that strategy, with the solver
/// finishing the round".
pub const SOLVER_SUFFIX: &str = "+solver";

/// Splits `BASE+solver[:OPTIONS]` into `BASE` and what follows `+solver` (the empty
/// string, or `:OPTIONS`); `None` when the spec has no such suffix. A path that merely
/// contains `+solver` (`neat:a+solver.json`) is not a suffix: it must end the spec or
/// be followed by `:`. The suffix is the last `+solver` of the spec, so a `+` in a
/// neat path cannot be escaped: a genome file whose name ends in `+solver` (or
/// `+solver:..`) needs another name.
/// The cases are tabled in `suffix_cases.txt`, which the GUI's own copy of the rule
/// (`gui/src/form.rs`, it does not link `sim`) is tested against too.
#[must_use]
pub fn split_solver_suffix(spec: &str) -> Option<(&str, &str)> {
    let at = spec.rfind(SOLVER_SUFFIX)?;
    let rest = &spec[at + SOLVER_SUFFIX.len()..];
    (rest.is_empty() || rest.starts_with(':')).then(|| (&spec[..at], rest))
}

/// What a `+solver[:OPTIONS]` suffix asks for: the exact endgame's limits and,
/// with `samples=N`, the early-game sampler (phase B).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SolverToggle {
    pub solver: SolverConfig,
    pub sampler: Option<SamplerConfig>,
}

/// The options of a `+solver[:OPTIONS]` suffix: `:N` is `max-unseen=N`, otherwise a comma
/// separated list of `max-unseen=N`, `budget=N`, `ceilings=hard|soft|off` and the sampler's
/// `samples=N`, `depth=D`, `leaf=adaptive|cards|neat:PATH`, `rollouts=N` (no `fallback`: the
/// wrapped strategy is the fallback). `rest` is what `split_solver_suffix` returned.
///
/// # Errors
///
/// An unknown option, a value that does not parse, sampler options without `samples`, or
/// a `:` with nothing after it (`x+solver:`).
pub fn parse_solver_toggle(rest: &str) -> Result<SolverToggle, String> {
    let options = rest.strip_prefix(':').unwrap_or(rest).trim();
    if options.is_empty() && !rest.is_empty() {
        return Err(
            "`+solver:` needs options after the colon (`+solver:N`, or `+solver` alone)".to_owned(),
        );
    }
    let mut toggle = SolverToggle::default();
    if let Ok(limit) = options.parse::<usize>() {
        toggle.solver.max_unseen = limit;
        return Ok(toggle);
    }
    let mut sampler = SamplerOptions::default();
    for option in options.split(',').map(str::trim).filter(|o| !o.is_empty()) {
        let (key, value) = option.split_once('=').ok_or_else(|| {
            format!("`{option}` is not KEY=VALUE (`+solver:N` sets max-unseen to N)")
        })?;
        if !apply_option(key, value, &mut toggle.solver, &mut sampler)? {
            return Err(format!(
                "unknown +solver option `{key}`; expected max-unseen, budget, ceilings, samples, \
                 depth, leaf or rollouts"
            ));
        }
    }
    toggle.sampler = sampler.finish()?;
    Ok(toggle)
}

/// The sampler options seen so far; `samples` switches the sampler on.
#[derive(Default)]
struct SamplerOptions {
    samples: Option<usize>,
    depth: Option<u8>,
    leaf: Option<LeafSpec>,
    rollouts: Option<u64>,
}

impl SamplerOptions {
    fn finish(self) -> Result<Option<SamplerConfig>, String> {
        let others = self.depth.is_some() || self.leaf.is_some() || self.rollouts.is_some();
        match self.samples {
            Some(samples) if samples > 0 => {
                let defaults = SamplerConfig::default();
                Ok(Some(SamplerConfig {
                    samples,
                    depth: self.depth.unwrap_or(defaults.depth),
                    leaf: self.leaf.unwrap_or(defaults.leaf),
                    rollouts: self.rollouts.unwrap_or(defaults.rollouts),
                }))
            }
            _ if others => Err(
                "depth, leaf and rollouts belong to the sampler: give samples=N (N > 0) too"
                    .to_owned(),
            ),
            _ => Ok(None),
        }
    }
}

/// Applies one option shared by both spec forms; `Ok(false)` for a key it does not know.
fn apply_option(
    key: &str,
    value: &str,
    solver: &mut SolverConfig,
    sampler: &mut SamplerOptions,
) -> Result<bool, String> {
    let number = |what: &str| format!("{what}: `{value}` is not a number");
    match key {
        "max-unseen" => solver.max_unseen = value.parse().map_err(|_| number(key))?,
        "budget" => solver.node_budget = value.parse().map_err(|_| number(key))?,
        "ceilings" => solver.ceilings = value.parse()?,
        "samples" => sampler.samples = Some(value.parse().map_err(|_| number(key))?),
        "depth" => {
            let depth: u8 = value.parse().map_err(|_| number(key))?;
            if !(1..=MAX_DEPTH).contains(&depth) {
                return Err(format!("depth: must be 1 to {MAX_DEPTH}, got {depth}"));
            }
            sampler.depth = Some(depth);
        }
        "leaf" => sampler.leaf = Some(LeafSpec::parse(value)?),
        "rollouts" => {
            let rollouts: u64 = value.parse().map_err(|_| number(key))?;
            if rollouts == 0 {
                return Err("rollouts: must be at least 1".to_owned());
            }
            sampler.rollouts = Some(rollouts);
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// The options of `endgame-solver[:OPTIONS]`.
#[derive(Debug, Clone, PartialEq)]
pub struct SolverSpec {
    pub solver: SolverConfig,
    pub fallback: StrategyName,
    /// The early-game sampler (`samples=N` and its options); `None` plays the
    /// fallback until the exact endgame, as in phase A.
    pub sampler: Option<SamplerConfig>,
}

impl Default for SolverSpec {
    fn default() -> Self {
        Self {
            solver: SolverConfig::default(),
            fallback: StrategyName::Adaptive,
            sampler: None,
        }
    }
}

impl FromStr for SolverSpec {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        let mut spec = Self::default();
        let mut sampler = SamplerOptions::default();
        for option in text.split(',').map(str::trim).filter(|o| !o.is_empty()) {
            let (key, value) = option
                .split_once('=')
                .ok_or_else(|| format!("`{option}` is not KEY=VALUE"))?;
            if key == "fallback" {
                spec.fallback = value.parse()?;
            } else if !apply_option(key, value, &mut spec.solver, &mut sampler)? {
                return Err(format!(
                    "unknown solver option `{key}`; expected max-unseen, budget, fallback, \
                     ceilings, samples, depth, leaf or rollouts"
                ));
            }
        }
        spec.sampler = sampler.finish()?;
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
        assert!(
            "depth=3".parse::<SolverSpec>().is_err(),
            "depth needs samples"
        );
        assert!("nodes=3".parse::<SolverSpec>().is_err());
        assert!("max-unseen=x".parse::<SolverSpec>().is_err());
        assert!("fallback=nope".parse::<SolverSpec>().is_err());
    }

    #[test]
    fn the_solver_suffix_splits_off_and_parses() {
        assert_eq!(
            split_solver_suffix("adaptive+solver"),
            Some(("adaptive", ""))
        );
        assert_eq!(
            split_solver_suffix("adaptive:reading,tempo+solver:8"),
            Some(("adaptive:reading,tempo", ":8"))
        );
        assert_eq!(split_solver_suffix("adaptive"), None);
        assert_eq!(split_solver_suffix("neat:a+solver.json"), None);
        assert_eq!(
            split_solver_suffix("neat:a+solver.json+solver"),
            Some(("neat:a+solver.json", ""))
        );
        for line in include_str!("suffix_cases.txt").lines() {
            let mut cells = line.split('|');
            let (spec, base, rest) = (
                cells.next().unwrap(),
                cells.next().unwrap(),
                cells.next().unwrap(),
            );
            let want = (base != "-").then_some((base, rest));
            assert_eq!(split_solver_suffix(spec), want, "{spec}");
        }
        assert!(parse_solver_toggle(":").is_err(), "empty options");
        assert!(parse_solver_toggle(": ").is_err(), "blank options");
        assert_eq!(parse_solver_toggle("").unwrap(), SolverToggle::default());
        assert_eq!(parse_solver_toggle(":8").unwrap().solver.max_unseen, 8);
        let both = parse_solver_toggle(":max-unseen=4,budget=99").unwrap();
        assert_eq!((both.solver.max_unseen, both.solver.node_budget), (4, 99));
        assert_eq!(both.sampler, None, "phase A options leave the sampler off");
        assert!(parse_solver_toggle(":fallback=adaptive").is_err());
        assert!(parse_solver_toggle(":x").is_err());
        assert!(parse_solver_toggle(":max-unseen=x").is_err());
    }

    #[test]
    fn sampler_options_switch_the_sampler_on() {
        let toggle = parse_solver_toggle(":samples=8,depth=2,leaf=cards,rollouts=500").unwrap();
        assert_eq!(
            toggle.sampler,
            Some(SamplerConfig {
                samples: 8,
                depth: 2,
                leaf: LeafSpec::Cards,
                rollouts: 500,
            })
        );
        assert_eq!(toggle.solver, SolverConfig::default());
        let defaults = parse_solver_toggle(":max-unseen=0,samples=4").unwrap();
        assert_eq!(defaults.solver.max_unseen, 0);
        assert_eq!(
            defaults.sampler,
            Some(SamplerConfig {
                samples: 4,
                ..SamplerConfig::default()
            })
        );
        assert_eq!(parse_solver_toggle(":samples=0").unwrap().sampler, None);
        for bad in [
            ":depth=2",
            ":samples=0,leaf=cards",
            ":samples=4,depth=0",
            ":samples=4,depth=9",
            ":samples=4,leaf=nope",
            ":samples=4,rollouts=0",
            ":samples=x",
            ":ceilings=maybe",
        ] {
            assert!(parse_solver_toggle(bad).is_err(), "{bad}");
        }
        let ceilings = parse_solver_toggle(":ceilings=off").unwrap();
        assert_eq!(ceilings.solver.ceilings, crate::solver::CeilingMode::Off);
        let spec: SolverSpec = "samples=8,fallback=lowest-legal,ceilings=soft"
            .parse()
            .unwrap();
        assert_eq!(spec.sampler.unwrap().samples, 8);
        assert_eq!(spec.fallback, StrategyName::LowestLegal);
        assert_eq!(spec.solver.ceilings, crate::solver::CeilingMode::Soft);
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
