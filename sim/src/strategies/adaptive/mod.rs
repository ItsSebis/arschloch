//! `Adaptive`: a single configurable strategy that layers card-counting,
//! endgame denial, deception, trick-lead tempo, and lead-order bullying
//! on top of a `LowestLegal`/`CardCounter` base (docs/ROADMAP.md, Phase 7,
//! Phase 8, and Phase 9). See `config` for the toggles,
//! `denial`/`deception`/`tempo`/`bully` for the modifiers themselves, and
//! `safety` for the proof `denial` and `tempo` share.

mod bully;
mod config;
mod deception;
mod denial;
mod safety;
mod tempo;

pub use config::{AdaptiveConfig, DenialMode};

use engine::{Card, DuplicateRule, Move};

use crate::strategies::{CardCounter, LowestLegal};
use crate::strategy::{ContextNeeds, Strategy, TurnContext};

/// A configurable strategy: `LowestLegal` (or `CardCounter`, if
/// `config.counting`) as its base play selection, with endgame denial,
/// lead-order bullying, trick-lead tempo, and deception layered on top —
/// see `AdaptiveConfig` for what each modifier does and how to enable it.
#[derive(Debug, Clone)]
pub struct Adaptive {
    config: AdaptiveConfig,
    name: String,
    needs: ContextNeeds,
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
            needs: needs_of(&config),
            config,
        }
    }

    #[must_use]
    pub fn config(&self) -> &AdaptiveConfig {
        &self.config
    }
}

/// The context fields `choose_play` can read under `config`: the union of
/// the base (`CardCounter` reads the unseen cards, `LowestLegal` nothing)
/// and every enabled modifier. `bully` reads only the hand; `denial`
/// (`HandSize`) the opponents' hand sizes; `denial` (`HandReading`) and
/// `tempo` go through `safety`, which reads opponent pass ceilings and
/// the unseen cards; `deception` reads the opponents' sizes and this
/// seat's own ceilings.
fn needs_of(config: &AdaptiveConfig) -> ContextNeeds {
    let mut needs = ContextNeeds::NONE;
    if config.counting {
        needs = needs.union(ContextNeeds::UNSEEN);
    }
    match config.denial {
        DenialMode::Off => {}
        DenialMode::HandSize { .. } => needs = needs.union(ContextNeeds::OPPONENTS),
        DenialMode::HandReading { .. } => {
            needs = needs
                .union(ContextNeeds::OPPONENT_PASS_CEILINGS)
                .union(ContextNeeds::UNSEEN);
        }
    }
    if config.tempo {
        needs = needs
            .union(ContextNeeds::OPPONENT_PASS_CEILINGS)
            .union(ContextNeeds::UNSEEN);
    }
    if config.deception_rate > 0.0 {
        needs = needs
            .union(ContextNeeds::OPPONENTS)
            .union(ContextNeeds::OWN_PASS_CEILINGS);
    }
    needs
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

        if let Some(mv) = bully::respond(self.config.bully, legal_moves, duplicate_rule, context) {
            return mv;
        }

        if let Some(mv) = tempo::respond(self.config.tempo, legal_moves, duplicate_rule, context) {
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

    fn needs(&self) -> ContextNeeds {
        self.needs
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
    use engine::{Combo, DeckVariant, Rank, Suit};
    use rand::SeedableRng;
    use std::sync::Arc;

    use crate::hand_reading::PassCeilings;
    use crate::strategies::EndgameDenial;
    use crate::strategy::OpponentHand;
    use crate::{run_batch, MatchConfig};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit, 0)
    }

    fn play(cards: Vec<Card>) -> Move {
        Move::Play(Combo::new(cards).unwrap())
    }

    fn configs(seeds: impl Iterator<Item = u64>, player_count: u8) -> Vec<MatchConfig> {
        seeds
            .map(|seed| MatchConfig {
                player_count,
                deck_variant: DeckVariant::Single,
                duplicate_rule: DuplicateRule::FirstDealtWins,
                rounds: 5,
                seed,
                pass_rule: engine::PassRule::default(),
                exchange_rule: engine::ExchangeRule::default(),
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
            tempo: false,
            bully: false,
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
            tempo: false,
            bully: false,
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
            deception_rate: 0.0,
            tempo: false,
            bully: false,
        })
        .name()
        .contains("counting"));
        assert!(Adaptive::new(AdaptiveConfig {
            counting: false,
            denial: DenialMode::HandReading { close: 2 },
            deception_rate: 0.0,
            tempo: false,
            bully: false,
        })
        .name()
        .contains("reading"));
        assert!(Adaptive::new(AdaptiveConfig {
            counting: false,
            denial: DenialMode::Off,
            deception_rate: 0.0,
            tempo: true,
            bully: false,
        })
        .name()
        .contains("tempo"));
        assert!(Adaptive::new(AdaptiveConfig {
            counting: false,
            denial: DenialMode::Off,
            deception_rate: 0.0,
            tempo: false,
            bully: true,
        })
        .name()
        .contains("bully"));
    }

    #[test]
    fn tempo_overrides_base_selection_when_the_cheaper_card_would_lose_the_lead() {
        let hand = vec![card(Rank::King, Suit::Hearts), card(Rank::Ace, Suit::Clubs)];
        let context = TurnContext {
            seat: 0,
            hand: &hand,
            opponents: vec![OpponentHand {
                seat: 1,
                hand_size: 2,
                active: true,
                pass_ceilings: PassCeilings::default(),
            }],
            unseen_cards: vec![
                card(Rank::Queen, Suit::Diamonds),
                card(Rank::Ace, Suit::Diamonds),
            ],
            own_pass_ceilings: PassCeilings::default(),
            current_combo: None,
        };
        let legal = vec![
            play(vec![card(Rank::King, Suit::Hearts)]),
            play(vec![card(Rank::Ace, Suit::Clubs)]),
        ];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        // Plain LowestLegal picks the cheaper King — nothing in its
        // logic knows that King is beatable by the unseen Ace.
        assert_eq!(
            LowestLegal.choose_play(&legal, DuplicateRule::FirstDealtWins, &context, &mut rng),
            play(vec![card(Rank::King, Suit::Hearts)])
        );

        // With tempo enabled and this seat down to its last 2 cards,
        // Adaptive overrides that choice with the only provably safe
        // play (Ace of Clubs beats the unseen Ace of Diamonds outright).
        let adaptive = Adaptive::new(AdaptiveConfig {
            counting: false,
            denial: DenialMode::Off,
            deception_rate: 0.0,
            tempo: true,
            bully: false,
        });
        assert_eq!(
            adaptive.choose_play(&legal, DuplicateRule::FirstDealtWins, &context, &mut rng),
            play(vec![card(Rank::Ace, Suit::Clubs)])
        );
    }

    #[test]
    fn tempo_only_runs_to_completion_at_every_table_size() {
        for player_count in [3u8, 4, 5, 6] {
            let cfgs = configs(0..20, player_count);
            let config = AdaptiveConfig {
                counting: false,
                denial: DenialMode::Off,
                deception_rate: 0.0,
                tempo: true,
                bully: false,
            };
            let adaptive: Arc<dyn Strategy> = Arc::new(Adaptive::new(config));
            let strategies = vec![adaptive; usize::from(player_count)];
            let results = run_batch(&cfgs, &strategies);
            assert_eq!(results.len(), 20, "player_count {player_count}");
        }
    }

    #[test]
    fn bully_overrides_base_selection_to_lead_the_cheapest_whole_group() {
        let hand = vec![
            card(Rank::Two, Suit::Diamonds),
            card(Rank::Two, Suit::Hearts),
            card(Rank::Three, Suit::Hearts),
            card(Rank::Four, Suit::Diamonds),
            card(Rank::Four, Suit::Hearts),
            card(Rank::Eight, Suit::Diamonds),
            card(Rank::Eight, Suit::Spades),
        ];
        let context = TurnContext {
            seat: 0,
            hand: &hand,
            opponents: vec![OpponentHand {
                seat: 1,
                hand_size: 2,
                active: true,
                pass_ceilings: PassCeilings::default(),
            }],
            unseen_cards: vec![card(Rank::King, Suit::Clubs), card(Rank::Ace, Suit::Spades)],
            own_pass_ceilings: PassCeilings::default(),
            current_combo: None,
        };
        let legal = vec![
            play(vec![card(Rank::Two, Suit::Diamonds)]),
            play(vec![card(Rank::Two, Suit::Hearts)]),
            play(vec![
                card(Rank::Two, Suit::Diamonds),
                card(Rank::Two, Suit::Hearts),
            ]),
            play(vec![card(Rank::Three, Suit::Hearts)]),
            play(vec![card(Rank::Four, Suit::Diamonds)]),
            play(vec![card(Rank::Four, Suit::Hearts)]),
            play(vec![
                card(Rank::Four, Suit::Diamonds),
                card(Rank::Four, Suit::Hearts),
            ]),
            play(vec![card(Rank::Eight, Suit::Diamonds)]),
            play(vec![card(Rank::Eight, Suit::Spades)]),
            play(vec![
                card(Rank::Eight, Suit::Diamonds),
                card(Rank::Eight, Suit::Spades),
            ]),
        ];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        // Plain LowestLegal always prefers the smallest legal combo
        // size, so among every size-1 option (one from each rank group,
        // plus the lone 3H) it leads the globally lowest-ranked single --
        // 2 of Diamonds -- immediately breaking up the pair of 2s rather
        // than leading it whole. Worse than just leading 3H first, and
        // exactly the kind of lead order the worked example shows loses
        // the race to finish.
        assert_eq!(
            LowestLegal.choose_play(&legal, DuplicateRule::FirstDealtWins, &context, &mut rng),
            play(vec![card(Rank::Two, Suit::Diamonds)])
        );

        // With bully enabled, Adaptive instead leads the cheapest whole
        // same-rank group (the pair of 2s), saving the single for last.
        let adaptive = Adaptive::new(AdaptiveConfig {
            counting: false,
            denial: DenialMode::Off,
            deception_rate: 0.0,
            tempo: false,
            bully: true,
        });
        assert_eq!(
            adaptive.choose_play(&legal, DuplicateRule::FirstDealtWins, &context, &mut rng),
            play(vec![
                card(Rank::Two, Suit::Diamonds),
                card(Rank::Two, Suit::Hearts)
            ])
        );
    }

    #[test]
    fn bully_only_runs_to_completion_at_every_table_size() {
        for player_count in [3u8, 4, 5, 6] {
            let cfgs = configs(0..20, player_count);
            let config = AdaptiveConfig {
                counting: false,
                denial: DenialMode::Off,
                deception_rate: 0.0,
                tempo: false,
                bully: true,
            };
            let adaptive: Arc<dyn Strategy> = Arc::new(Adaptive::new(config));
            let strategies = vec![adaptive; usize::from(player_count)];
            let results = run_batch(&cfgs, &strategies);
            assert_eq!(results.len(), 20, "player_count {player_count}");
        }
    }
}
