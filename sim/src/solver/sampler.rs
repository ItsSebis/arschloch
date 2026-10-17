//! Phase B: the early game. Sample consistent deals of the unseen cards,
//! search each one to a bounded depth with max^n, value the positions at the
//! depth limit with a leaf evaluator and average per move ("perfect-information
//! Monte Carlo"). See docs/superpowers/specs/2026-10-17-solver-phase-b-design.md.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use engine::{Card, DuplicateRule, Move, Round};
use rand::{Rng, SeedableRng};

use super::leaf::{finished_values, CardCount, LeafEvaluator, Rollout, Values};
use super::observation::Observation;
use super::sample::sample_deal;
use super::{CeilingMode, SolveError};
use crate::strategies::{Adaptive, AdaptiveConfig, NeatStrategy};
use crate::training::evaluate::mix;

/// What values a position at the depth limit.
#[derive(Clone)]
pub enum LeafSpec {
    /// Roll the round out with `adaptive` (`Adaptive(counting,reading)`) in every seat.
    Adaptive,
    /// The static `CardCount` guess, no rollout.
    Cards,
    /// Roll the round out with a trained genome in every seat.
    Neat {
        path: PathBuf,
        genome: Arc<NeatStrategy>,
    },
}

impl LeafSpec {
    /// `adaptive`, `cards` or `neat:PATH` (the genome file is loaded here).
    ///
    /// # Errors
    ///
    /// An unknown name or a genome file that does not load.
    pub fn parse(text: &str) -> Result<Self, String> {
        match text.trim() {
            "adaptive" => Ok(Self::Adaptive),
            "cards" => Ok(Self::Cards),
            other => {
                let path = other
                    .strip_prefix("neat:")
                    .map(str::trim)
                    .filter(|p| !p.is_empty())
                    .ok_or_else(|| {
                        format!("leaf: `{other}` is not one of adaptive, cards or neat:PATH")
                    })?;
                let genome = NeatStrategy::from_file(Path::new(path))
                    .map_err(|e| format!("leaf: {path}: {e}"))?;
                Ok(Self::Neat {
                    path: PathBuf::from(path),
                    genome: Arc::new(genome),
                })
            }
        }
    }

    /// How the strategy's name shows it: `adaptive`, `cards`, `neat(<file stem>)`.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::Adaptive => "adaptive".to_owned(),
            Self::Cards => "cards".to_owned(),
            Self::Neat { path, .. } => format!(
                "neat({})",
                path.file_stem()
                    .map_or_else(String::new, |s| s.to_string_lossy().into_owned())
            ),
        }
    }

    #[must_use]
    pub fn evaluator(&self) -> Arc<dyn LeafEvaluator> {
        match self {
            Self::Adaptive => Arc::new(Rollout::new(Arc::new(Adaptive::new(
                AdaptiveConfig::default(),
            )))),
            Self::Cards => Arc::new(CardCount),
            Self::Neat { genome, .. } => Arc::new(Rollout::new(Arc::clone(genome) as _)),
        }
    }
}

impl PartialEq for LeafSpec {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Adaptive, Self::Adaptive) | (Self::Cards, Self::Cards) => true,
            (Self::Neat { path: a, .. }, Self::Neat { path: b, .. }) => a == b,
            _ => false,
        }
    }
}

impl fmt::Debug for LeafSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Neat { path, .. } => write!(f, "Neat({})", path.display()),
            other => f.write_str(&other.label()),
        }
    }
}

/// The sampler's settings (`samples=`, `depth=`, `leaf=`, `rollouts=`).
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerConfig {
    /// Deals sampled per decision.
    pub samples: usize,
    /// Plies searched with max^n in each deal; forced moves do not count.
    pub depth: u8,
    pub leaf: LeafSpec,
    /// Leaf evaluations allowed per decision; sampling stops after the sample
    /// that crosses it (one sample is always completed).
    pub rollouts: u64,
}

/// `samples` when `samples=N` is given without a value elsewhere.
pub const DEFAULT_SAMPLES: usize = 16;
pub const DEFAULT_ROLLOUTS: u64 = 20_000;
pub const MAX_DEPTH: u8 = 4;

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            samples: DEFAULT_SAMPLES,
            depth: 1,
            leaf: LeafSpec::Adaptive,
            rollouts: DEFAULT_ROLLOUTS,
        }
    }
}

impl SamplerConfig {
    /// `S=16,D=1,adaptive` plus the budget when it is not the default.
    #[must_use]
    pub fn label(&self) -> String {
        let mut text = format!("S={},D={},{}", self.samples, self.depth, self.leaf.label());
        if self.rollouts != DEFAULT_ROLLOUTS {
            text.push_str(",R=");
            text.push_str(&self.rollouts.to_string());
        }
        text
    }
}

/// The sampler's verdict on one decision.
#[derive(Debug, Clone, PartialEq)]
pub struct SampleReport {
    /// Every legal move (engine order) with the acting seat's mean 0-based
    /// place over the samples.
    pub moves: Vec<(Move, f64)>,
    /// Deals searched.
    pub samples: u64,
    /// Leaf evaluations spent.
    pub leaves: u64,
}

impl SampleReport {
    /// The move with the lowest mean place, the first one on ties.
    ///
    /// # Panics
    ///
    /// Panics on an empty report (`sample_analyze` never returns one).
    #[must_use]
    pub fn best(&self) -> Move {
        self.moves
            .iter()
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .expect("sample_analyze never returns an empty report")
            .0
    }
}

/// Searches `config.samples` sampled deals of the observed position.
///
/// # Errors
///
/// `NoConsistentDeal` when no sample respects the ceilings (`Hard`), and
/// `InvalidPosition` when the observation cannot be rebuilt.
///
/// # Panics
///
/// Panics if the engine rejects one of its own legal moves.
#[allow(clippy::cast_precision_loss)] // sample counts are tiny
pub fn sample_analyze(
    observation: &Observation,
    config: &SamplerConfig,
    ceilings: CeilingMode,
    leaf: &dyn LeafEvaluator,
    seed: u64,
) -> Result<SampleReport, SolveError> {
    let seat = usize::from(observation.seat);
    let cards = observation.cards_in_play();
    let mut use_ceilings = ceilings != CeilingMode::Off;
    let mut sums: Vec<(Move, f64)> = Vec::new();
    let mut samples = 0u64;
    let mut leaves = 0u64;
    for i in 0..config.samples {
        if samples > 0 && leaves >= config.rollouts {
            break;
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(mix(seed.wrapping_add(i as u64)));
        let mut deal = sample_deal(observation, use_ceilings, &mut rng);
        if deal.is_none() && use_ceilings && ceilings == CeilingMode::Soft {
            use_ceilings = false;
            deal = sample_deal(observation, false, &mut rng);
        }
        // Under `Hard`, a sample that found no deal is skipped: the ceilings
        // may still allow some (the sampling is random), or none at all.
        let Some(deal) = deal else { continue };
        let round = observation
            .round_for(&deal)
            .ok_or(SolveError::InvalidPosition)?;
        let mut search = DepthSearch {
            leaf,
            cards: &cards,
            rule: observation.duplicate_rule,
            seed: rng.next_u64(),
            leaves: 0,
        };
        let seat_to_move = round.seat_to_move().ok_or(SolveError::InvalidPosition)?;
        let legal = round.legal_moves();
        if sums.is_empty() {
            sums = legal.iter().map(|&mv| (mv, 0.0)).collect();
        }
        for (slot, mv) in sums.iter_mut().zip(legal) {
            let mut next = round.clone();
            next.submit_move(seat_to_move, mv)
                .expect("moves from legal_moves are accepted");
            slot.1 += f64::from(search.value(&next, config.depth.saturating_sub(1))[seat]);
        }
        leaves += search.leaves;
        samples += 1;
    }
    if samples == 0 {
        return Err(SolveError::NoConsistentDeal);
    }
    let n = samples as f64;
    Ok(SampleReport {
        moves: sums.into_iter().map(|(mv, sum)| (mv, sum / n)).collect(),
        samples,
        leaves,
    })
}

/// Depth-limited max^n over one sampled deal.
struct DepthSearch<'a> {
    leaf: &'a dyn LeafEvaluator,
    cards: &'a [Card],
    rule: DuplicateRule,
    /// One rollout seed for every leaf of the deal: moves are compared on the
    /// same luck (common random numbers).
    seed: u64,
    leaves: u64,
}

impl DepthSearch<'_> {
    fn value(&mut self, round: &Round, depth: u8) -> Values {
        let Some(seat) = round.seat_to_move() else {
            return finished_values(round);
        };
        let moves = round.legal_moves();
        if moves.len() == 1 {
            // A forced move is not a choice: it does not use up a ply.
            let mut next = round.clone();
            next.submit_move(seat, moves[0])
                .expect("moves from legal_moves are accepted");
            return self.value(&next, depth);
        }
        if depth == 0 {
            self.leaves += 1;
            return self.leaf.values(round, self.cards, self.rule, self.seed);
        }
        let mut best: Option<Values> = None;
        for mv in moves {
            let mut next = round.clone();
            next.submit_move(seat, mv)
                .expect("moves from legal_moves are accepted");
            let value = self.value(&next, depth - 1);
            if best.is_none_or(|b| value[usize::from(seat)] < b[usize::from(seat)]) {
                best = Some(value);
            }
        }
        best.expect("a seat to move has at least one legal move")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand_reading::{read_pass_ceilings, PassTracker};
    use crate::match_runner::turn_context_for;
    use crate::solver::search::Search;
    use crate::strategy::ContextNeeds;
    use crate::test_support::for_each_state;
    use engine::{Combo, Rank, Suit};

    /// Observations of random mid-round positions (`Final` rule only).
    fn observations(seed: u64, mut visit: impl FnMut(&Observation, &Round)) {
        for_each_state(seed, 2, |state, _| {
            let round = state.round;
            if round.pass_rule() != engine::PassRule::Final {
                return;
            }
            let seat = round.seat_to_move().unwrap();
            let mut tracker = PassTracker::new(usize::from(state.players), state.rule);
            let context = turn_context_for(
                round,
                seat,
                state.players,
                state.round_deck,
                &mut tracker,
                ContextNeeds::ALL,
            );
            if let Some(o) = Observation::from_context(&context, state.rule) {
                visit(&o, round);
            }
        });
    }

    fn config(samples: usize, depth: u8, leaf: LeafSpec) -> SamplerConfig {
        SamplerConfig {
            samples,
            depth,
            leaf,
            rollouts: DEFAULT_ROLLOUTS,
        }
    }

    #[test]
    fn deep_enough_it_is_the_exact_search_of_each_sampled_deal() {
        // With every card known (nothing unseen but one opponent's whole hand
        // decided by sizes) and a depth past the end, the leaf is never used and
        // the value per move is the exact max^n value.
        let mut checked = 0;
        observations(61, |o, round| {
            let cards: usize = o.hand_sizes.iter().sum();
            if cards > 7 {
                return;
            }
            let others = o
                .hand_sizes
                .iter()
                .enumerate()
                .filter(|&(s, &n)| s != usize::from(o.seat) && n > 0);
            if others.count() != 1 {
                return; // one opponent left: the deal is fully known
            }
            let config = config(2, 40, LeafSpec::Cards);
            let report = sample_analyze(o, &config, CeilingMode::Off, &CardCount, 3).unwrap();
            let exact = Search::new(5_000_000).root_values(round).unwrap();
            let out_before = round.finishing_order().len();
            let rebuilt_out = o.hand_sizes.iter().filter(|&&n| n == 0).count();
            assert_eq!(out_before, rebuilt_out);
            for ((mv, mean), (exact_mv, places)) in report.moves.iter().zip(&exact) {
                assert_eq!(mv, exact_mv);
                let want = f64::from(places[usize::from(o.seat)]);
                assert!((mean - want).abs() < 1e-9, "{mean} vs {want}");
            }
            checked += 1;
        });
        assert!(checked > 5, "only {checked} positions");
    }

    #[test]
    fn it_answers_every_early_position_with_legal_moves_deterministically() {
        let leaf = LeafSpec::Adaptive.evaluator();
        let mut checked = 0;
        observations(63, |o, round| {
            if o.unseen.len() < 12 || checked >= 25 {
                return;
            }
            let config = config(3, 1, LeafSpec::Adaptive);
            let a = sample_analyze(o, &config, CeilingMode::Soft, leaf.as_ref(), 11).unwrap();
            let b = sample_analyze(o, &config, CeilingMode::Soft, leaf.as_ref(), 11).unwrap();
            assert_eq!(a, b, "same seed, same report");
            assert_eq!(
                a.moves.iter().map(|m| m.0).collect::<Vec<_>>(),
                round.legal_moves()
            );
            assert!(round.legal_moves().contains(&a.best()));
            assert_eq!(a.samples, 3);
            checked += 1;
        });
        assert!(checked >= 10, "only {checked} positions");
    }

    #[test]
    fn depth_two_searches_more_leaves_and_the_budget_stops_sampling() {
        let leaf = CardCount;
        let mut checked = 0;
        observations(65, |o, round| {
            if o.unseen.len() < 12 || round.legal_moves().len() < 3 || checked >= 10 {
                return;
            }
            let one = sample_analyze(
                o,
                &config(4, 1, LeafSpec::Cards),
                CeilingMode::Off,
                &leaf,
                1,
            )
            .unwrap();
            let two = sample_analyze(
                o,
                &config(4, 2, LeafSpec::Cards),
                CeilingMode::Off,
                &leaf,
                1,
            )
            .unwrap();
            assert!(two.leaves > one.leaves, "{} vs {}", two.leaves, one.leaves);
            let tight = SamplerConfig {
                rollouts: 1,
                ..config(4, 1, LeafSpec::Cards)
            };
            let cut = sample_analyze(o, &tight, CeilingMode::Off, &leaf, 1).unwrap();
            assert_eq!(cut.samples, 1, "one sample is always completed");
            checked += 1;
        });
        assert!(checked >= 5, "only {checked} positions");
    }

    #[test]
    fn contradictory_ceilings_fail_hard_and_pass_soft() {
        let c = |rank, suit, i| Card::new(rank, suit, i);
        let three = Combo::new(vec![c(Rank::Three, Suit::Hearts, 30)]).unwrap();
        let o = Observation {
            duplicate_rule: DuplicateRule::FirstDealtWins,
            seat: 0,
            player_count: 3,
            hand: vec![c(Rank::Two, Suit::Clubs, 0), c(Rank::Five, Suit::Clubs, 5)],
            hand_sizes: vec![2, 1, 2],
            in_trick: vec![true; 3],
            ceilings: read_pass_ceilings(
                3,
                &[],
                &[(1, three, 0), (2, three, 0)],
                DuplicateRule::FirstDealtWins,
            ),
            unseen: vec![
                c(Rank::Four, Suit::Clubs, 1),
                c(Rank::Eight, Suit::Hearts, 2),
                c(Rank::King, Suit::Spades, 3),
            ],
            on_table: None,
        };
        let leaf = LeafSpec::Adaptive.evaluator();
        let config = config(4, 1, LeafSpec::Adaptive);
        assert_eq!(
            sample_analyze(&o, &config, CeilingMode::Hard, leaf.as_ref(), 1),
            Err(SolveError::NoConsistentDeal)
        );
        for mode in [CeilingMode::Soft, CeilingMode::Off] {
            assert_eq!(
                sample_analyze(&o, &config, mode, leaf.as_ref(), 1)
                    .unwrap()
                    .samples,
                4
            );
        }
    }

    #[test]
    fn leaf_specs_parse_and_label() {
        assert_eq!(LeafSpec::parse("adaptive").unwrap(), LeafSpec::Adaptive);
        assert_eq!(LeafSpec::parse("cards").unwrap().label(), "cards");
        assert!(LeafSpec::parse("neat:").is_err());
        assert!(LeafSpec::parse("neat:/no/such/file.json").is_err());
        assert!(LeafSpec::parse("random").is_err());
        let champion = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../docs/baselines/neat-v2/champion.json"
        );
        let neat = LeafSpec::parse(&format!("neat:{champion}")).unwrap();
        assert_eq!(neat.label(), "neat(champion)");
        assert_eq!(
            SamplerConfig::default().label(),
            "S=16,D=1,adaptive",
            "the default budget is not shown"
        );
    }
}
