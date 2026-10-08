//! The settings of a training run, saved with it so a run can be resumed
//! (and later understood) without remembering the command line.

use engine::{DeckVariant, DuplicateRule, ExchangeRule, PassRule};
use neat::NeatConfig;
use serde::{Deserialize, Serialize};

use super::evaluate::TableSpec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeckChoice {
    Single,
    Double,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DuplicateChoice {
    FirstDealtWins,
    LastDealtWins,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainConfig {
    /// Seeds everything: the initial population, every evolution step and
    /// every evaluation match.
    pub seed: u64,
    pub player_count: u8,
    pub deck: DeckChoice,
    pub duplicate_rule: DuplicateChoice,
    /// What a pass means for the rest of the trick. Runs written before
    /// the rule existed carry no value and were played under `free`.
    #[serde(default = "legacy_pass_rule")]
    pub pass_rule: PassRule,
    /// Whether the lower role of an exchange pair must give its highest
    /// cards. Runs written before the rule existed were played under `free`.
    #[serde(default = "legacy_exchange_rule")]
    pub exchange_rule: ExchangeRule,
    /// Rounds per evaluation match (role carry-over between rounds).
    pub rounds_per_match: usize,
    /// Matches each genome plays per generation. Every genome plays the
    /// same matches (same deals, seats and opponents).
    pub matches_per_genome: usize,
    /// Matches in each fresh-seed re-evaluation of the generation's
    /// champion, against the mixed pool and against each opponent alone.
    pub reeval_matches: usize,
    /// Total generations to run (a resumed run continues up to this).
    pub generations: u32,
    pub neat: NeatConfig,
    /// The opponent pool as the `--strategy`-style specs that built it,
    /// so a resume can rebuild exactly the same pool.
    pub opponent_specs: Vec<String>,
    /// How many of the genomes with the best *training* fitness are
    /// re-evaluated on the fixed matches to pick the generation's
    /// champion (1 = trust the training fitness). Training fitness is
    /// noisy enough that its best genome is often not the strongest one.
    #[serde(default = "one")]
    pub champion_candidates: usize,
    /// Frozen past champions kept as extra opponents in the training pool
    /// (0 = none), so the population is not tuned only against the fixed
    /// opponents.
    #[serde(default)]
    pub hall_of_fame_size: usize,
    /// A generation's champion joins the hall of fame every this many
    /// generations (the oldest member leaves when it is full).
    #[serde(default = "five")]
    pub hall_of_fame_interval: u32,
}

/// The rule every run before the pass rule was introduced was played under.
fn legacy_pass_rule() -> PassRule {
    PassRule::Free
}

/// The exchange rule every run before the rule existed was played under.
fn legacy_exchange_rule() -> ExchangeRule {
    ExchangeRule::Free
}

fn one() -> usize {
    1
}

fn five() -> u32 {
    5
}

impl TrainConfig {
    pub(crate) fn table(&self) -> TableSpec {
        TableSpec {
            player_count: self.player_count,
            deck_variant: match self.deck {
                DeckChoice::Single => DeckVariant::Single,
                DeckChoice::Double => DeckVariant::Double,
            },
            duplicate_rule: match self.duplicate_rule {
                DuplicateChoice::FirstDealtWins => DuplicateRule::FirstDealtWins,
                DuplicateChoice::LastDealtWins => DuplicateRule::LastDealtWins,
            },
            pass_rule: self.pass_rule,
            exchange_rule: self.exchange_rule,
            rounds: self.rounds_per_match,
        }
    }

    /// # Errors
    ///
    /// Returns a message naming the first invalid setting.
    pub fn validate(&self) -> Result<(), String> {
        if !(3..=6).contains(&self.player_count) {
            return Err(format!(
                "player_count {} is not a table size (3-6)",
                self.player_count
            ));
        }
        for (name, value) in [
            ("rounds_per_match", self.rounds_per_match),
            ("matches_per_genome", self.matches_per_genome),
            ("reeval_matches", self.reeval_matches),
        ] {
            if value == 0 {
                return Err(format!("{name} must be at least 1"));
            }
        }
        if self.generations == 0 {
            return Err("generations must be at least 1".into());
        }
        if self.opponent_specs.is_empty() {
            return Err("the opponent pool is empty".into());
        }
        if self.champion_candidates == 0 || self.champion_candidates > self.neat.population_size {
            return Err(format!(
                "champion_candidates must be between 1 and the population size ({})",
                self.neat.population_size
            ));
        }
        if self.hall_of_fame_interval == 0 {
            return Err("hall_of_fame_interval must be at least 1".into());
        }
        self.neat.validate().map_err(|e| e.to_string())
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    pub fn sample() -> TrainConfig {
        TrainConfig {
            seed: 1,
            player_count: 4,
            deck: DeckChoice::Single,
            duplicate_rule: DuplicateChoice::FirstDealtWins,
            pass_rule: PassRule::default(),
            exchange_rule: ExchangeRule::default(),
            rounds_per_match: 4,
            matches_per_genome: 6,
            reeval_matches: 8,
            generations: 3,
            neat: NeatConfig {
                population_size: 12,
                ..NeatConfig::default()
            },
            opponent_specs: vec!["lowest-legal".into()],
            champion_candidates: 1,
            hall_of_fame_size: 0,
            hall_of_fame_interval: 5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::sample;
    use super::*;

    #[test]
    fn the_sample_config_is_valid_and_round_trips_through_json() {
        let config = sample();
        assert_eq!(config.validate(), Ok(()));
        let restored: TrainConfig =
            serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
        assert_eq!(restored, config);
    }

    #[test]
    fn each_invalid_setting_is_named() {
        let cases: Vec<(TrainConfig, &str)> = vec![
            (
                TrainConfig {
                    player_count: 2,
                    pass_rule: engine::PassRule::default(),
                    exchange_rule: engine::ExchangeRule::default(),
                    ..sample()
                },
                "player_count",
            ),
            (
                TrainConfig {
                    rounds_per_match: 0,
                    pass_rule: engine::PassRule::default(),
                    exchange_rule: engine::ExchangeRule::default(),
                    ..sample()
                },
                "rounds_per_match",
            ),
            (
                TrainConfig {
                    matches_per_genome: 0,
                    pass_rule: engine::PassRule::default(),
                    exchange_rule: engine::ExchangeRule::default(),
                    ..sample()
                },
                "matches_per_genome",
            ),
            (
                TrainConfig {
                    reeval_matches: 0,
                    pass_rule: engine::PassRule::default(),
                    exchange_rule: engine::ExchangeRule::default(),
                    ..sample()
                },
                "reeval_matches",
            ),
            (
                TrainConfig {
                    generations: 0,
                    pass_rule: engine::PassRule::default(),
                    exchange_rule: engine::ExchangeRule::default(),
                    ..sample()
                },
                "generations",
            ),
            (
                TrainConfig {
                    opponent_specs: vec![],
                    pass_rule: engine::PassRule::default(),
                    exchange_rule: engine::ExchangeRule::default(),
                    ..sample()
                },
                "pool is empty",
            ),
            (
                TrainConfig {
                    champion_candidates: 0,
                    pass_rule: engine::PassRule::default(),
                    exchange_rule: engine::ExchangeRule::default(),
                    ..sample()
                },
                "champion_candidates",
            ),
            (
                TrainConfig {
                    champion_candidates: 99,
                    pass_rule: engine::PassRule::default(),
                    exchange_rule: engine::ExchangeRule::default(),
                    ..sample()
                },
                "champion_candidates",
            ),
            (
                TrainConfig {
                    hall_of_fame_interval: 0,
                    pass_rule: engine::PassRule::default(),
                    exchange_rule: engine::ExchangeRule::default(),
                    ..sample()
                },
                "hall_of_fame_interval",
            ),
        ];
        for (config, expected) in cases {
            let error = config.validate().unwrap_err();
            assert!(error.contains(expected), "{expected}: {error}");
        }
    }

    #[test]
    fn a_config_saved_before_these_options_existed_still_loads_with_their_defaults() {
        let mut value = serde_json::to_value(sample()).unwrap();
        let object = value.as_object_mut().unwrap();
        for key in [
            "champion_candidates",
            "hall_of_fame_size",
            "hall_of_fame_interval",
        ] {
            object.remove(key);
        }
        let old: TrainConfig = serde_json::from_value(value).unwrap();
        assert_eq!(
            (
                old.champion_candidates,
                old.hall_of_fame_size,
                old.hall_of_fame_interval
            ),
            (1, 0, 5)
        );
    }

    #[test]
    fn table_converts_to_engine_types() {
        let table = TrainConfig {
            deck: DeckChoice::Double,
            duplicate_rule: DuplicateChoice::LastDealtWins,
            pass_rule: engine::PassRule::default(),
            exchange_rule: engine::ExchangeRule::default(),
            ..sample()
        }
        .table();
        assert_eq!(table.deck_variant, DeckVariant::Double);
        assert_eq!(table.duplicate_rule, DuplicateRule::LastDealtWins);
        assert_eq!(table.rounds, 4);
    }

    #[test]
    fn a_config_written_before_the_pass_rule_existed_reads_as_free() {
        let mut value = serde_json::to_value(sample()).unwrap();
        value.as_object_mut().unwrap().remove("pass_rule");
        let old: TrainConfig = serde_json::from_value(value).unwrap();
        assert_eq!(old.pass_rule, PassRule::Free);
        // A new config states its rule and round-trips.
        let mut config = sample();
        config.pass_rule = PassRule::Final;
        let again: TrainConfig =
            serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
        assert_eq!(again.pass_rule, PassRule::Final);
        assert_eq!(
            sample().pass_rule,
            PassRule::Final,
            "new runs use the rules of the game"
        );
    }

    #[test]
    fn a_config_written_before_the_exchange_rule_existed_reads_as_free() {
        let mut value = serde_json::to_value(sample()).unwrap();
        value.as_object_mut().unwrap().remove("exchange_rule");
        let old: TrainConfig = serde_json::from_value(value).unwrap();
        assert_eq!(old.exchange_rule, ExchangeRule::Free);
        assert_eq!(
            sample().exchange_rule,
            ExchangeRule::Forced,
            "new runs use the rules of the game"
        );
        let again: TrainConfig =
            serde_json::from_str(&serde_json::to_string(&sample()).unwrap()).unwrap();
        assert_eq!(again.exchange_rule, ExchangeRule::Forced);
    }
}
