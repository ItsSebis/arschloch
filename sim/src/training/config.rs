//! The settings of a training run, saved with it so a run can be resumed
//! (and later understood) without remembering the command line.

use engine::{DeckVariant, DuplicateRule, ExchangeRule, PassRule};
use neat::NeatConfig;
use serde::{Deserialize, Serialize};

use super::evaluate::TableSpec;
use super::fitness::FitnessKind;
use crate::FeatureSet;

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
    /// Weight `w` of the luck-adjusted term in the training fitness:
    /// `(1 - w) * mean_role_score + w * luck_adjusted_score` (see
    /// `skill_term`). 0 (the default) is the plain mean role score and
    /// leaves a run bit-for-bit as it was before the option existed.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub skill_weight: f64,
    /// Which features the evolved networks see (a nested prefix of the
    /// full list, so the network's input count is `features.len()`). Old
    /// runs carry no value and used the 20 base features; the default is
    /// not written, so base runs keep their config files byte for byte.
    #[serde(default, skip_serializing_if = "is_base")]
    pub features: FeatureSet,
    /// What the evolution maximises (see `FitnessKind`): the mean role
    /// score (the default, not written to the config file) or a variant
    /// that rewards consistency or flattens the top. Training fitness,
    /// champion choice, re-evaluation, the held-out confirmation and the
    /// `best.json` choice all use it. Incompatible with `skill_weight > 0`.
    #[serde(default, skip_serializing_if = "FitnessKind::is_mean")]
    pub fitness: FitnessKind,
    /// Racing (noise-aware selection): the top fraction of the population,
    /// by fitness, that is evaluated on `racing_matches` extra matches
    /// before selection. 0 (the default, not written) is off; it must be
    /// set together with `racing_matches`.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub racing_top: f64,
    /// Extra matches per raced genome (0 = racing off).
    #[serde(default, skip_serializing_if = "is_zero_count")]
    pub racing_matches: usize,
    /// Table sizes to train on. Empty (the default, not written) means just
    /// `player_count`. With `decks` the settings are the product, table
    /// size major (see `TrainConfig::tables`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub player_counts: Vec<u8>,
    /// Decks to train on. Empty (the default, not written) means just `deck`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub decks: Vec<DeckChoice>,
}

#[allow(clippy::trivially_copy_pass_by_ref)] // serde's signature
fn is_zero_count(value: &usize) -> bool {
    *value == 0
}

#[allow(clippy::trivially_copy_pass_by_ref)] // serde's signature
fn is_base(value: &FeatureSet) -> bool {
    *value == FeatureSet::Base
}

#[allow(clippy::trivially_copy_pass_by_ref)] // serde's signature
fn is_zero(value: &f64) -> bool {
    value.to_bits() == 0.0f64.to_bits()
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
    fn table_for(&self, player_count: u8, deck: DeckChoice) -> TableSpec {
        TableSpec {
            player_count,
            deck_variant: match deck {
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

    /// The first setting's table (the only one unless several settings are
    /// configured).
    pub(crate) fn table(&self) -> TableSpec {
        self.tables().remove(0)
    }

    /// Every training setting: the product of the table sizes and decks, in
    /// table-size-major order (`player_counts` is the outer loop, `decks`
    /// the inner one, each in the order given). Match `i` of an evaluation
    /// is played at setting `i % tables().len()`. Without the lists it is
    /// just `player_count` and `deck`.
    #[must_use]
    pub fn tables(&self) -> Vec<TableSpec> {
        let counts: &[u8] = if self.player_counts.is_empty() {
            std::slice::from_ref(&self.player_count)
        } else {
            &self.player_counts
        };
        let decks: &[DeckChoice] = if self.decks.is_empty() {
            std::slice::from_ref(&self.deck)
        } else {
            &self.decks
        };
        counts
            .iter()
            .flat_map(|&count| decks.iter().map(move |&deck| self.table_for(count, deck)))
            .collect()
    }

    /// How many raced genomes a generation has: `ceil(racing_top *
    /// population)`, at least 1 (0 when racing is off).
    #[must_use]
    pub fn raced_count(&self) -> usize {
        if self.racing_matches == 0 {
            return 0;
        }
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let count = (self.racing_top * self.neat.population_size as f64).ceil() as usize;
        count.clamp(1, self.neat.population_size)
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
        if !(0.0..=1.0).contains(&self.skill_weight) {
            return Err(format!(
                "skill_weight {} is not between 0 and 1",
                self.skill_weight
            ));
        }
        self.validate_phase13()?;
        self.neat.validate().map_err(|e| e.to_string())
    }

    /// The settings added in Phase 13: fitness, racing, several settings.
    fn validate_phase13(&self) -> Result<(), String> {
        self.fitness.validate()?;
        if !self.fitness.is_mean() && self.skill_weight > 0.0 {
            return Err(format!(
                "fitness {} cannot be combined with skill_weight {}: the luck-adjusted term is defined for the mean role score only",
                self.fitness, self.skill_weight
            ));
        }
        if !(0.0..=1.0).contains(&self.racing_top) {
            return Err(format!(
                "racing_top {} is not between 0 and 1",
                self.racing_top
            ));
        }
        if (self.racing_top > 0.0) != (self.racing_matches > 0) {
            return Err(
                "racing needs both racing_top (a fraction in (0, 1]) and racing_matches (at least 1); set both or neither"
                    .into(),
            );
        }
        if self.racing_matches > 0 && self.skill_weight > 0.0 {
            return Err("racing cannot be combined with skill_weight > 0".into());
        }
        for (i, count) in self.player_counts.iter().enumerate() {
            if !(3..=6).contains(count) {
                return Err(format!("player_counts: {count} is not a table size (3-6)"));
            }
            if self.player_counts[..i].contains(count) {
                return Err(format!("player_counts lists {count} twice"));
            }
        }
        for (i, deck) in self.decks.iter().enumerate() {
            if self.decks[..i].contains(deck) {
                return Err(format!("decks lists {deck:?} twice"));
            }
        }
        let settings = self.tables().len();
        if settings > 1 && self.skill_weight > 0.0 {
            return Err(
                "several settings (player_counts, decks) cannot be combined with skill_weight > 0"
                    .into(),
            );
        }
        for (name, value) in [
            ("matches_per_genome", self.matches_per_genome),
            ("reeval_matches", self.reeval_matches),
        ] {
            if value < settings {
                return Err(format!(
                    "{name} ({value}) is less than the number of settings ({settings}): every setting must be played at least once"
                ));
            }
        }
        Ok(())
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
            skill_weight: 0.0,
            features: FeatureSet::Base,
            fitness: FitnessKind::Mean,
            racing_top: 0.0,
            racing_matches: 0,
            player_counts: Vec::new(),
            decks: Vec::new(),
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
    fn skill_weight_defaults_to_zero_validates_and_old_configs_load() {
        assert!(sample().skill_weight.abs() < f64::EPSILON);
        for ok in [0.0, 0.5, 1.0] {
            assert_eq!(
                TrainConfig {
                    skill_weight: ok,
                    ..sample()
                }
                .validate(),
                Ok(())
            );
        }
        for bad in [-0.1, 1.01, f64::NAN, f64::INFINITY] {
            let error = TrainConfig {
                skill_weight: bad,
                ..sample()
            }
            .validate()
            .unwrap_err();
            assert!(error.contains("skill_weight"), "{bad}: {error}");
        }
        // Zero is not written, so a default run's files are unchanged.
        let text = serde_json::to_string(&sample()).unwrap();
        assert!(!text.contains("skill_weight"), "{text}");
        let old: TrainConfig = serde_json::from_str(&text).unwrap();
        assert!(old.skill_weight.abs() < f64::EPSILON);
        // A non-zero weight round-trips.
        let config = TrainConfig {
            skill_weight: 0.5,
            ..sample()
        };
        let again: TrainConfig =
            serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
        assert!((again.skill_weight - 0.5).abs() < f64::EPSILON);
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

    #[test]
    fn the_phase_13_options_default_to_off_are_not_written_and_old_configs_load() {
        let config = sample();
        assert_eq!(config.fitness, FitnessKind::Mean);
        assert!(config.racing_top.abs() < f64::EPSILON && config.racing_matches == 0);
        assert!(config.player_counts.is_empty() && config.decks.is_empty());
        let text = serde_json::to_string(&config).unwrap();
        for key in [
            "fitness",
            "racing_top",
            "racing_matches",
            "player_counts",
            "decks",
        ] {
            assert!(!text.contains(key), "{key} written for a default: {text}");
        }
        let old: TrainConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(old, config);
        assert_eq!(old.tables().len(), 1);
        assert_eq!(old.raced_count(), 0);
        // Set options round-trip and are written.
        let set = TrainConfig {
            fitness: FitnessKind::Risk { lambda: 0.5 },
            racing_top: 0.25,
            racing_matches: 7,
            player_counts: vec![3, 5],
            decks: vec![DeckChoice::Single, DeckChoice::Double],
            ..sample()
        };
        let text = serde_json::to_string(&set).unwrap();
        for key in [
            "fitness",
            "racing_top",
            "racing_matches",
            "player_counts",
            "decks",
        ] {
            assert!(text.contains(key), "{key}: {text}");
        }
        assert_eq!(serde_json::from_str::<TrainConfig>(&text).unwrap(), set);
        assert_eq!(set.validate(), Ok(()));
    }

    #[test]
    fn settings_are_the_product_of_table_sizes_and_decks_table_size_first() {
        let config = TrainConfig {
            player_counts: vec![5, 3],
            decks: vec![DeckChoice::Double, DeckChoice::Single],
            ..sample()
        };
        let labels: Vec<String> = config.tables().iter().map(TableSpec::label).collect();
        assert_eq!(labels, ["5p-double", "5p-single", "3p-double", "3p-single"]);
        // Only one list: the other falls back to the single value.
        let counts_only = TrainConfig {
            player_counts: vec![3, 4, 6],
            ..sample()
        };
        let labels: Vec<String> = counts_only.tables().iter().map(TableSpec::label).collect();
        assert_eq!(labels, ["3p-single", "4p-single", "6p-single"]);
        let decks_only = TrainConfig {
            decks: vec![DeckChoice::Double, DeckChoice::Single],
            ..sample()
        };
        let labels: Vec<String> = decks_only.tables().iter().map(TableSpec::label).collect();
        assert_eq!(labels, ["4p-double", "4p-single"]);
        assert_eq!(sample().tables().len(), 1);
        assert_eq!(sample().table().player_count, 4);
    }

    #[test]
    #[allow(clippy::too_many_lines)] // a table of cases
    fn invalid_phase_13_settings_are_named() {
        let cases: Vec<(TrainConfig, &str)> = vec![
            (
                TrainConfig {
                    fitness: FitnessKind::Risk { lambda: -1.0 },
                    ..sample()
                },
                "lambda",
            ),
            (
                TrainConfig {
                    fitness: FitnessKind::Tanh { kappa: 0.0 },
                    ..sample()
                },
                "kappa",
            ),
            (
                TrainConfig {
                    fitness: FitnessKind::Tanh { kappa: 4.0 },
                    skill_weight: 0.5,
                    ..sample()
                },
                "skill_weight",
            ),
            (
                TrainConfig {
                    racing_top: 0.5,
                    ..sample()
                },
                "both racing_top",
            ),
            (
                TrainConfig {
                    racing_matches: 5,
                    ..sample()
                },
                "both racing_top",
            ),
            (
                TrainConfig {
                    racing_top: 1.5,
                    racing_matches: 5,
                    ..sample()
                },
                "racing_top",
            ),
            (
                TrainConfig {
                    racing_top: 0.5,
                    racing_matches: 5,
                    skill_weight: 0.5,
                    ..sample()
                },
                "racing",
            ),
            (
                TrainConfig {
                    player_counts: vec![3, 7],
                    ..sample()
                },
                "player_counts",
            ),
            (
                TrainConfig {
                    player_counts: vec![3, 3],
                    ..sample()
                },
                "twice",
            ),
            (
                TrainConfig {
                    decks: vec![DeckChoice::Double, DeckChoice::Double],
                    ..sample()
                },
                "twice",
            ),
            (
                TrainConfig {
                    player_counts: vec![3, 4],
                    skill_weight: 0.5,
                    ..sample()
                },
                "settings",
            ),
            (
                TrainConfig {
                    player_counts: vec![3, 4, 5, 6],
                    matches_per_genome: 3,
                    ..sample()
                },
                "matches_per_genome",
            ),
            (
                TrainConfig {
                    player_counts: vec![3, 4, 5, 6],
                    reeval_matches: 3,
                    ..sample()
                },
                "reeval_matches",
            ),
        ];
        for (config, expected) in cases {
            let error = config.validate().unwrap_err();
            assert!(error.contains(expected), "{expected}: {error}");
        }
    }

    #[test]
    fn the_raced_count_is_the_rounded_up_fraction_of_the_population() {
        let racing = |top: f64, population: usize| TrainConfig {
            racing_top: top,
            racing_matches: 4,
            neat: NeatConfig {
                population_size: population,
                ..NeatConfig::default()
            },
            ..sample()
        };
        assert_eq!(racing(0.25, 12).raced_count(), 3);
        assert_eq!(racing(0.2, 12).raced_count(), 3);
        assert_eq!(racing(0.01, 12).raced_count(), 1);
        assert_eq!(racing(1.0, 12).raced_count(), 12);
    }
}
