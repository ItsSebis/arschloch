//! A champion's recorded decisions, for the dashboard's decision
//! inspector: real situations from a real match, with every legal move,
//! its feature vector and the network's score, so a viewer (or the
//! browser, which re-runs the network) can see why one move beat another.

use std::sync::{Arc, Mutex};

use engine::{Card, DuplicateRule, Move, Rank, Suit};
use neat::Genome;
use serde::{Deserialize, Serialize};

use super::evaluate::TableSpec;
use crate::{
    run_match, MatchConfig, NeatStrategy, Strategy, TurnContext, FEATURE_NAMES, FEATURE_SET_VERSION,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateRecord {
    /// `pass` or the cards played, such as `9♠` or `3♦ 3♥`.
    pub description: String,
    pub features: Vec<f64>,
    /// The output node's sum before `tanh`: what the choice is made on.
    pub raw_score: f64,
    pub activation: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionRecord {
    /// The deciding seat's hand, weakest card first.
    pub hand: Vec<String>,
    /// The combo to beat, or `None` when leading.
    pub table: Option<String>,
    /// Hand sizes of the opponents still in the round.
    pub opponent_hands: Vec<usize>,
    pub candidates: Vec<CandidateRecord>,
    /// Index into `candidates` of the move the champion played.
    pub chosen: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionFile {
    pub generation: u32,
    pub feature_set_version: u32,
    pub feature_names: Vec<String>,
    pub decisions: Vec<DecisionRecord>,
}

#[must_use]
pub fn card_label(card: &Card) -> String {
    let rank = match card.rank {
        Rank::Two => "2",
        Rank::Three => "3",
        Rank::Four => "4",
        Rank::Five => "5",
        Rank::Six => "6",
        Rank::Seven => "7",
        Rank::Eight => "8",
        Rank::Nine => "9",
        Rank::Ten => "T",
        Rank::Jack => "J",
        Rank::Queen => "Q",
        Rank::King => "K",
        Rank::Ace => "A",
    };
    let suit = match card.suit {
        Suit::Diamonds => "♦",
        Suit::Hearts => "♥",
        Suit::Spades => "♠",
        Suit::Clubs => "♣",
    };
    format!("{rank}{suit}")
}

#[must_use]
pub fn move_label(candidate: &Move) -> String {
    match candidate {
        Move::Pass => "pass".to_owned(),
        Move::Play(combo) => combo
            .cards()
            .iter()
            .map(card_label)
            .collect::<Vec<_>>()
            .join(" "),
    }
}

/// Wraps a `NeatStrategy`, playing exactly as it does while logging each
/// decision with at least two options.
struct Recorder {
    inner: NeatStrategy,
    log: Mutex<Vec<DecisionRecord>>,
}

impl Strategy for Recorder {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        context: &TurnContext<'_>,
        rng: &mut dyn rand::Rng,
    ) -> Move {
        let chosen_move = self
            .inner
            .choose_play(legal_moves, duplicate_rule, context, rng);
        if legal_moves.len() >= 2 {
            let mut hand = context.hand.to_vec();
            hand.sort_by(|a, b| a.compare(b, duplicate_rule));
            let candidates = self
                .inner
                .score_candidates(legal_moves, duplicate_rule, context)
                .into_iter()
                .map(|scored| CandidateRecord {
                    description: move_label(&scored.candidate),
                    features: scored.features.to_vec(),
                    raw_score: scored.raw_score,
                    activation: scored.activation,
                })
                .collect();
            let record = DecisionRecord {
                hand: hand.iter().map(card_label).collect(),
                table: context
                    .current_combo
                    .map(|combo| move_label(&Move::Play(combo.clone()))),
                opponent_hands: context
                    .opponents
                    .iter()
                    .filter(|o| o.active)
                    .map(|o| o.hand_size)
                    .collect(),
                candidates,
                chosen: legal_moves
                    .iter()
                    .position(|m| *m == chosen_move)
                    .expect("the chosen move is one of the legal moves"),
            };
            self.log.lock().expect("recorder lock").push(record);
        }
        chosen_move
    }

    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        duplicate_rule: DuplicateRule,
        rng: &mut dyn rand::Rng,
    ) -> Vec<Card> {
        self.inner
            .choose_exchange_cards(hand, count, duplicate_rule, rng)
    }
}

/// Plays one match with `genome` in seat 0 against pool members in the
/// other seats (in pool order, wrapping) and keeps up to `limit` of its
/// decisions that had a real choice, evenly spread over the match.
/// Deterministic for a given `seed`.
///
/// # Panics
///
/// Panics if `pool` is empty or the genome does not fit this build's
/// features (trained genomes always do).
#[must_use]
pub fn record_decisions(
    genome: &Genome,
    table: &TableSpec,
    pool: &[Arc<dyn Strategy>],
    seed: u64,
    limit: usize,
    generation: u32,
) -> DecisionFile {
    let recorder = Arc::new(Recorder {
        inner: NeatStrategy::new("champion", genome)
            .expect("trained genomes use this build's features"),
        log: Mutex::new(Vec::new()),
    });
    let strategies: Vec<Arc<dyn Strategy>> = (0..usize::from(table.player_count))
        .map(|seat| -> Arc<dyn Strategy> {
            if seat == 0 {
                recorder.clone()
            } else {
                pool[(seat - 1) % pool.len()].clone()
            }
        })
        .collect();
    let _ = run_match(
        &MatchConfig {
            player_count: table.player_count,
            deck_variant: table.deck_variant,
            duplicate_rule: table.duplicate_rule,
            rounds: table.rounds,
            seed,
        },
        &strategies,
    );
    let all = recorder.log.lock().expect("recorder lock").clone();
    let decisions = if all.len() <= limit {
        all
    } else {
        (0..limit)
            .map(|i| all[i * all.len() / limit].clone())
            .collect()
    };
    DecisionFile {
        generation,
        feature_set_version: FEATURE_SET_VERSION,
        feature_names: FEATURE_NAMES.iter().map(|&n| n.to_owned()).collect(),
        decisions,
    }
}

#[cfg(test)]
mod tests {
    use engine::{Combo, DeckVariant};
    use neat::{InnovationTracker, NeatConfig};
    use rand::SeedableRng;

    use super::*;
    use crate::{CardCounter, LowestLegal, FEATURE_COUNT};

    fn genome(seed: u64) -> Genome {
        let mut tracker = InnovationTracker::new(u32::try_from(FEATURE_COUNT).unwrap() + 2);
        Genome::minimal(
            FEATURE_COUNT,
            &mut tracker,
            &NeatConfig::default(),
            &mut rand::rngs::StdRng::seed_from_u64(seed),
        )
    }

    fn table() -> TableSpec {
        TableSpec {
            player_count: 4,
            deck_variant: DeckVariant::Single,
            duplicate_rule: DuplicateRule::FirstDealtWins,
            rounds: 3,
        }
    }

    fn pool() -> Vec<Arc<dyn Strategy>> {
        vec![Arc::new(LowestLegal), Arc::new(CardCounter)]
    }

    #[test]
    fn cards_and_moves_have_readable_labels() {
        let ten = Card::new(Rank::Ten, Suit::Spades, 0);
        assert_eq!(card_label(&ten), "T♠");
        assert_eq!(move_label(&Move::Pass), "pass");
        let pair = Combo::new(vec![
            Card::new(Rank::Three, Suit::Diamonds, 0),
            Card::new(Rank::Three, Suit::Hearts, 0),
        ])
        .unwrap();
        assert_eq!(move_label(&Move::Play(pair)), "3♦ 3♥");
    }

    #[test]
    fn recorded_decisions_are_real_choices_that_match_the_scores() {
        let file = record_decisions(&genome(3), &table(), &pool(), 11, 12, 7);
        assert_eq!(file.generation, 7);
        assert_eq!(file.feature_names.len(), FEATURE_COUNT);
        assert!(!file.decisions.is_empty() && file.decisions.len() <= 12);
        for decision in &file.decisions {
            assert!(decision.candidates.len() >= 2);
            assert!(decision.chosen < decision.candidates.len());
            assert!(decision
                .candidates
                .iter()
                .all(|c| c.features.len() == FEATURE_COUNT));
            let best = decision
                .candidates
                .iter()
                .map(|c| c.raw_score)
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(
                (decision.candidates[decision.chosen].raw_score - best).abs() < 1e-12,
                "the champion plays its highest-scoring candidate"
            );
            for candidate in &decision.candidates {
                assert!((candidate.activation - candidate.raw_score.tanh()).abs() < 1e-12);
            }
            assert!(!decision.hand.is_empty());
        }
    }

    #[test]
    fn leading_decisions_have_no_table_and_following_ones_do() {
        let file = record_decisions(&genome(4), &table(), &pool(), 5, 200, 0);
        assert!(
            file.decisions.iter().any(|d| d.table.is_none()),
            "some leads"
        );
        assert!(
            file.decisions.iter().any(|d| d.table.is_some()),
            "some follows"
        );
        let following = file.decisions.iter().find(|d| d.table.is_some()).unwrap();
        assert!(following.candidates.iter().any(|c| c.description == "pass"));
        let leading = file.decisions.iter().find(|d| d.table.is_none()).unwrap();
        assert!(leading.candidates.iter().all(|c| c.description != "pass"));
    }

    #[test]
    fn recording_is_deterministic_and_respects_the_limit() {
        let a = record_decisions(&genome(3), &table(), &pool(), 11, 5, 1);
        let b = record_decisions(&genome(3), &table(), &pool(), 11, 5, 1);
        assert_eq!(a, b);
        assert!(a.decisions.len() <= 5);
        let all = record_decisions(&genome(3), &table(), &pool(), 11, usize::MAX, 1);
        assert!(all.decisions.len() >= a.decisions.len());
    }
}
