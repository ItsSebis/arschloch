//! `NeatStrategy`: plays by scoring every legal move with an evolved
//! neural network and choosing the highest score. See
//! docs/superpowers/specs/2026-10-08-neat-engine-design.md, section 4.
//!
//! The network never produces a move, only a number per candidate, so
//! the chosen move is always one `engine` reported as legal. The
//! strategy holds an immutable compiled network and no other state, so
//! one instance is shared across parallel matches.

mod features;
mod genome_file;

use std::cmp::Ordering;
use std::path::Path;

use engine::{Card, DuplicateRule, Move};
use neat::{Genome, Network};

pub use features::{TurnSummary, FEATURE_COUNT, FEATURE_NAMES};
pub use genome_file::{GenomeFile, GenomeFileError, FORMAT_VERSION};

use crate::strategy::{Strategy, TurnContext};

#[derive(Debug, Clone)]
pub struct NeatStrategy {
    name: String,
    network: Network,
}

impl NeatStrategy {
    /// Builds a player from `genome`, reported under `name` in results.
    ///
    /// # Errors
    ///
    /// Returns `GenomeFileError::Mismatch` if the genome does not take
    /// exactly `FEATURE_COUNT` inputs.
    pub fn new(name: impl Into<String>, genome: &Genome) -> Result<Self, GenomeFileError> {
        if genome.num_inputs() != FEATURE_COUNT {
            return Err(GenomeFileError::Mismatch(format!(
                "genome takes {} inputs, this build has {FEATURE_COUNT} features",
                genome.num_inputs()
            )));
        }
        Ok(Self {
            name: name.into(),
            network: Network::compile(genome),
        })
    }

    /// Loads a genome file and names the player `Neat(<file stem>)`.
    ///
    /// # Errors
    ///
    /// Any `GenomeFileError` from loading or checking the file.
    pub fn from_file(path: &Path) -> Result<Self, GenomeFileError> {
        let file = GenomeFile::load(path)?;
        let stem = path
            .file_stem()
            .map_or_else(|| "genome".into(), |s| s.to_string_lossy());
        Self::new(format!("Neat({stem})"), &file.genome)
    }
}

impl Strategy for NeatStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        context: &TurnContext<'_>,
        _rng: &mut dyn rand::Rng,
    ) -> Move {
        let summary = TurnSummary::new(context, duplicate_rule);
        let mut scratch = Vec::new();
        let mut best: Option<(&Move, f64)> = None;
        for candidate in legal_moves {
            let score = self
                .network
                .activate(&summary.features(candidate), &mut scratch);
            // Strictly better only: ties keep the earlier-listed move, so
            // the choice is a pure function of the legal-move order.
            if best.is_none_or(|(_, top)| score.total_cmp(&top) == Ordering::Greater) {
                best = Some((candidate, score));
            }
        }
        best.expect("a seat to move always has at least one legal move")
            .0
            .clone()
    }

    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        duplicate_rule: DuplicateRule,
        _rng: &mut dyn rand::Rng,
    ) -> Vec<Card> {
        crate::strategies::take_highest_naive(hand, count, duplicate_rule)
    }
}

#[cfg(test)]
mod tests {
    use engine::{Combo, Rank, Suit};
    use neat::{ConnectionGene, NodeGene, NodeKind};
    use rand::SeedableRng;

    use super::*;
    use crate::hand_reading::PassCeilings;

    /// A genome scoring `sum(weight * feature)` for the given
    /// `(feature index, weight)` pairs, wired directly to the output.
    pub(crate) fn linear_genome(weights: &[(usize, f64)]) -> Genome {
        let inputs = FEATURE_COUNT;
        let mut nodes: Vec<NodeGene> = (0..inputs)
            .map(|id| NodeGene {
                id: u32::try_from(id).unwrap(),
                kind: NodeKind::Input,
            })
            .collect();
        nodes.push(NodeGene {
            id: u32::try_from(inputs).unwrap(),
            kind: NodeKind::Bias,
        });
        nodes.push(NodeGene {
            id: u32::try_from(inputs + 1).unwrap(),
            kind: NodeKind::Output,
        });
        let connections = weights
            .iter()
            .enumerate()
            .map(|(innovation, &(feature, weight))| ConnectionGene {
                innovation: u32::try_from(innovation).unwrap(),
                from: u32::try_from(feature).unwrap(),
                to: u32::try_from(inputs + 1).unwrap(),
                weight,
                enabled: true,
            })
            .collect();
        Genome::from_parts(inputs, nodes, connections).unwrap()
    }

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit, 0)
    }

    fn single(rank: Rank, suit: Suit) -> Move {
        Move::Play(Combo::new(vec![card(rank, suit)]).unwrap())
    }

    fn context(hand: &[Card]) -> TurnContext<'_> {
        TurnContext {
            seat: 0,
            hand,
            opponents: Vec::new(),
            unseen_cards: Vec::new(),
            own_pass_ceilings: PassCeilings::default(),
            current_combo: None,
        }
    }

    fn rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(0)
    }

    const RULE: DuplicateRule = DuplicateRule::FirstDealtWins;

    #[test]
    fn a_negative_strength_weight_picks_the_weakest_card() {
        let strategy = NeatStrategy::new("t", &linear_genome(&[(2, -1.0)])).unwrap();
        let hand = [
            card(Rank::Nine, Suit::Spades),
            card(Rank::Four, Suit::Hearts),
        ];
        let moves = vec![
            single(Rank::Nine, Suit::Spades),
            single(Rank::Four, Suit::Hearts),
        ];
        let chosen = strategy.choose_play(&moves, RULE, &context(&hand), &mut rng());
        assert_eq!(chosen, single(Rank::Four, Suit::Hearts));
    }

    #[test]
    fn a_positive_strength_weight_picks_the_strongest_card() {
        let strategy = NeatStrategy::new("t", &linear_genome(&[(2, 1.0)])).unwrap();
        let hand = [
            card(Rank::Nine, Suit::Spades),
            card(Rank::Four, Suit::Hearts),
        ];
        let moves = vec![
            single(Rank::Nine, Suit::Spades),
            single(Rank::Four, Suit::Hearts),
        ];
        let chosen = strategy.choose_play(&moves, RULE, &context(&hand), &mut rng());
        assert_eq!(chosen, single(Rank::Nine, Suit::Spades));
    }

    #[test]
    fn the_pass_flag_weight_decides_between_passing_and_playing() {
        let hand = [card(Rank::Nine, Suit::Spades)];
        let on_table = Combo::new(vec![card(Rank::Four, Suit::Hearts)]).unwrap();
        let mut ctx = context(&hand);
        ctx.current_combo = Some(&on_table);
        let moves = vec![single(Rank::Nine, Suit::Spades), Move::Pass];
        let keen = NeatStrategy::new("t", &linear_genome(&[(0, -1.0)])).unwrap();
        assert_eq!(
            keen.choose_play(&moves, RULE, &ctx, &mut rng()),
            single(Rank::Nine, Suit::Spades)
        );
        let shy = NeatStrategy::new("t", &linear_genome(&[(0, 1.0)])).unwrap();
        assert_eq!(shy.choose_play(&moves, RULE, &ctx, &mut rng()), Move::Pass);
    }

    #[test]
    fn exact_ties_keep_the_first_listed_move() {
        // No connections: every candidate scores tanh(0) = 0.
        let strategy = NeatStrategy::new("t", &linear_genome(&[])).unwrap();
        let hand = [
            card(Rank::Nine, Suit::Spades),
            card(Rank::Four, Suit::Hearts),
        ];
        let moves = vec![
            single(Rank::Nine, Suit::Spades),
            single(Rank::Four, Suit::Hearts),
        ];
        let chosen = strategy.choose_play(&moves, RULE, &context(&hand), &mut rng());
        assert_eq!(chosen, moves[0]);
    }

    #[test]
    fn a_genome_for_a_different_feature_count_is_refused() {
        let mut tracker = neat::InnovationTracker::new(5);
        let small = Genome::minimal(
            3,
            &mut tracker,
            &neat::NeatConfig::default(),
            &mut rand::rngs::StdRng::seed_from_u64(1),
        );
        assert!(matches!(
            NeatStrategy::new("t", &small),
            Err(GenomeFileError::Mismatch(_))
        ));
    }

    #[test]
    fn from_file_names_the_player_after_the_file() {
        let file = GenomeFile::new(linear_genome(&[(2, -1.0)])).unwrap();
        let path = std::env::temp_dir().join(format!("my-champion-{}.json", std::process::id()));
        file.save(&path).unwrap();
        let strategy = NeatStrategy::from_file(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(
            strategy.name().starts_with("Neat(my-champion-"),
            "{}",
            strategy.name()
        );
    }

    #[test]
    fn the_exchange_gives_up_the_highest_cards() {
        let strategy = NeatStrategy::new("t", &linear_genome(&[])).unwrap();
        let hand = [
            card(Rank::Two, Suit::Clubs),
            card(Rank::Ace, Suit::Clubs),
            card(Rank::King, Suit::Hearts),
        ];
        let given = strategy.choose_exchange_cards(&hand, 2, RULE, &mut rng());
        assert_eq!(given.len(), 2);
        assert!(given.contains(&card(Rank::Ace, Suit::Clubs)));
        assert!(given.contains(&card(Rank::King, Suit::Hearts)));
    }

    #[test]
    fn the_strategy_can_be_shared_across_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NeatStrategy>();
    }
}
