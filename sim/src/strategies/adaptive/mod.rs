//! `Adaptive`: a single configurable strategy that layers card-counting,
//! endgame denial, and deception on top of a `LowestLegal`/`CardCounter`
//! base (docs/ROADMAP.md, Phase 7). See `config` for the toggles,
//! `denial`/`deception` for the modifiers themselves.

mod config;
mod deception;
mod denial;

pub use config::{AdaptiveConfig, DenialMode};

use engine::{Card, DuplicateRule, Move};

use crate::strategies::{CardCounter, LowestLegal};
use crate::strategy::{Strategy, TurnContext};

/// A configurable strategy: `LowestLegal` (or `CardCounter`, if
/// `config.counting`) as its base play selection, with endgame denial
/// and deception layered on top — see `AdaptiveConfig` for what each
/// modifier does and how to enable it.
#[derive(Debug, Clone)]
pub struct Adaptive {
    config: AdaptiveConfig,
    name: String,
}

impl Adaptive {
    /// # Panics
    /// If `config` is invalid (`AdaptiveConfig::is_valid`). The CLI
    /// validates during parsing, so reaching this invalid is a
    /// programming error, not user input.
    #[must_use]
    pub fn new(config: AdaptiveConfig) -> Self {
        assert!(config.is_valid(), "invalid AdaptiveConfig: {config:?}");
        Self {
            name: format!("Adaptive({config})"),
            config,
        }
    }

    #[must_use]
    pub fn config(&self) -> &AdaptiveConfig {
        &self.config
    }
}

impl Default for Adaptive {
    fn default() -> Self {
        Self::new(AdaptiveConfig::default())
    }
}

impl Strategy for Adaptive {
    fn name(&self) -> &str {
        &self.name
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        context: &TurnContext<'_>,
        rng: &mut dyn rand::Rng,
    ) -> Move {
        if !legal_moves.iter().any(|m| matches!(m, Move::Play(_))) {
            return Move::Pass;
        }

        if let Some(mv) = denial::respond(
            self.config.denial,
            legal_moves,
            duplicate_rule,
            context,
            rng,
        ) {
            return mv;
        }

        let base = if self.config.counting {
            CardCounter.choose_play(legal_moves, duplicate_rule, context, rng)
        } else {
            LowestLegal.choose_play(legal_moves, duplicate_rule, context, rng)
        };

        let hand_len = context.hand.len();
        let could_finish_instead = legal_moves
            .iter()
            .any(|m| matches!(m, Move::Play(c) if c.size() == hand_len));
        if legal_moves.len() > 1
            && !could_finish_instead
            && deception::should_bluff_pass(
                self.config.deception_rate,
                legal_moves,
                duplicate_rule,
                context,
                rng,
            )
        {
            return Move::Pass;
        }

        base
    }

    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        duplicate_rule: DuplicateRule,
        rng: &mut dyn rand::Rng,
    ) -> Vec<Card> {
        LowestLegal.choose_exchange_cards(hand, count, duplicate_rule, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::DeckVariant;
    use std::sync::Arc;

    use crate::strategies::EndgameDenial;
    use crate::{run_batch, MatchConfig};

    fn configs(seeds: impl Iterator<Item = u64>, player_count: u8) -> Vec<MatchConfig> {
        seeds
            .map(|seed| MatchConfig {
                player_count,
                deck_variant: DeckVariant::Single,
                duplicate_rule: DuplicateRule::FirstDealtWins,
                rounds: 5,
                seed,
            })
            .collect()
    }

    #[test]
    fn adaptive_none_matches_lowest_legal_exactly() {
        let cfgs = configs(0..30, 4);
        let adaptive: Arc<dyn Strategy> = Arc::new(Adaptive::new(AdaptiveConfig::NONE));
        let lowest: Arc<dyn Strategy> = Arc::new(LowestLegal);
        let a = run_batch(&cfgs, &vec![adaptive; 4]);
        let b = run_batch(&cfgs, &vec![lowest; 4]);
        for (ra, rb) in a.iter().zip(&b) {
            assert_eq!(ra.role_history, rb.role_history);
            assert_eq!(ra.pass_counts, rb.pass_counts);
            assert_eq!(ra.trick_count, rb.trick_count);
        }
    }

    #[test]
    fn adaptive_counting_only_matches_card_counter_exactly() {
        let cfgs = configs(0..30, 4);
        let config = AdaptiveConfig {
            counting: true,
            denial: DenialMode::Off,
            deception_rate: 0.0,
        };
        let adaptive: Arc<dyn Strategy> = Arc::new(Adaptive::new(config));
        let counter: Arc<dyn Strategy> = Arc::new(CardCounter);
        let a = run_batch(&cfgs, &vec![adaptive; 4]);
        let b = run_batch(&cfgs, &vec![counter; 4]);
        for (ra, rb) in a.iter().zip(&b) {
            assert_eq!(ra.role_history, rb.role_history);
        }
    }

    #[test]
    fn adaptive_hand_size_denial_matches_endgame_denial_exactly() {
        let cfgs = configs(0..30, 4);
        let config = AdaptiveConfig {
            counting: false,
            denial: DenialMode::HandSize { close: 2 },
            deception_rate: 0.0,
        };
        let adaptive: Arc<dyn Strategy> = Arc::new(Adaptive::new(config));
        let denier: Arc<dyn Strategy> = Arc::new(EndgameDenial);
        let a = run_batch(&cfgs, &vec![adaptive; 4]);
        let b = run_batch(&cfgs, &vec![denier; 4]);
        for (ra, rb) in a.iter().zip(&b) {
            assert_eq!(ra.role_history, rb.role_history);
        }
    }

    #[test]
    fn adaptive_default_runs_to_completion_at_every_table_size() {
        for player_count in [3u8, 4, 5, 6] {
            let cfgs = configs(0..20, player_count);
            let adaptive: Arc<dyn Strategy> = Arc::new(Adaptive::default());
            let strategies = vec![adaptive; usize::from(player_count)];
            let results = run_batch(&cfgs, &strategies);
            assert_eq!(results.len(), 20, "player_count {player_count}");
        }
    }

    #[test]
    fn name_reflects_configuration() {
        assert_eq!(Adaptive::new(AdaptiveConfig::NONE).name(), "Adaptive(none)");
        assert!(Adaptive::new(AdaptiveConfig {
            counting: true,
            denial: DenialMode::Off,
            deception_rate: 0.0
        })
        .name()
        .contains("counting"));
        assert!(Adaptive::new(AdaptiveConfig {
            counting: false,
            denial: DenialMode::HandReading { close: 2 },
            deception_rate: 0.0
        })
        .name()
        .contains("reading"));
    }
}
