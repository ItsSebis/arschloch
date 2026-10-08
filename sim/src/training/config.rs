//! The settings of a training run, saved with it so a run can be resumed
//! (and later understood) without remembering the command line.

use engine::{DeckVariant, DuplicateRule};
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
            rounds_per_match: 4,
            matches_per_genome: 6,
            reeval_matches: 8,
            generations: 3,
            neat: NeatConfig {
                population_size: 12,
                ..NeatConfig::default()
            },
            opponent_specs: vec!["lowest-legal".into()],
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
                    ..sample()
                },
                "player_count",
            ),
            (
                TrainConfig {
                    rounds_per_match: 0,
                    ..sample()
                },
                "rounds_per_match",
            ),
            (
                TrainConfig {
                    matches_per_genome: 0,
                    ..sample()
                },
                "matches_per_genome",
            ),
            (
                TrainConfig {
                    reeval_matches: 0,
                    ..sample()
                },
                "reeval_matches",
            ),
            (
                TrainConfig {
                    generations: 0,
                    ..sample()
                },
                "generations",
            ),
            (
                TrainConfig {
                    opponent_specs: vec![],
                    ..sample()
                },
                "pool is empty",
            ),
        ];
        for (config, expected) in cases {
            let error = config.validate().unwrap_err();
            assert!(error.contains(expected), "{expected}: {error}");
        }
    }

    #[test]
    fn table_converts_to_engine_types() {
        let table = TrainConfig {
            deck: DeckChoice::Double,
            duplicate_rule: DuplicateChoice::LastDealtWins,
            ..sample()
        }
        .table();
        assert_eq!(table.deck_variant, DeckVariant::Double);
        assert_eq!(table.duplicate_rule, DuplicateRule::LastDealtWins);
        assert_eq!(table.rounds, 4);
    }
}
