//! Which features a genome was trained with: the v1 base set and the
//! optional groups appended after it.
//!
//! The full list `FEATURE_NAMES_V2` is the v1 names unchanged, followed by
//! the `trick`, `hidden` and `table` groups. A genome sees a *prefix* of
//! it (`FeatureSet`): its file records exactly those names and its network
//! takes exactly that many inputs, so every v1 genome stays valid and
//! plays as it always did.

use serde::{Deserialize, Serialize};

use crate::strategy::ContextNeeds;

/// Every feature, in input order: the 20 of `FEATURE_NAMES`, then the
/// `trick`, `hidden` and `table` groups (see `FeatureGroups`).
pub const FEATURE_NAMES_V2: [&str; 31] = [
    // base
    "is_pass",
    "combo_size",
    "top_strength",
    "hand_below_top",
    "uses_top_group",
    "splits_group",
    "empties_hand",
    "hand_left_after",
    "is_leading",
    "table_combo_size",
    "hand_size",
    "grouped_fraction",
    "singleton_fraction",
    "active_opponents",
    "min_opponent_hand",
    "mean_opponent_hand",
    "opponent_close",
    "unseen_outranking",
    "unseen_beaters",
    "opponents_locked_out",
    // trick
    "opponents_in_trick",
    "fraction_in_trick",
    "all_in_trick_close",
    "min_hand_in_trick",
    "last_to_act_in_trick",
    // hidden
    "p_win_trick",
    "expected_beaters",
    "my_unbeatable_fraction",
    // table
    "table_players",
    "table_double_deck",
    "table_pass_final",
];

pub const FEATURE_COUNT_V2: usize = FEATURE_NAMES_V2.len();

/// How many features each group of `FEATURE_NAMES_V2` has, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureGroups {
    pub base: usize,
    pub trick: usize,
    pub hidden: usize,
    pub table: usize,
}

pub const FEATURE_GROUPS: FeatureGroups = FeatureGroups {
    base: 20,
    trick: 5,
    hidden: 3,
    table: 3,
};

/// The nested prefix of `FEATURE_NAMES_V2` a genome is trained on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FeatureSet {
    /// The 20 features of feature set v1.
    #[default]
    Base,
    /// `Base` and the `trick` group.
    Trick,
    /// `Trick` and the `hidden` group.
    Hidden,
    /// `Hidden` and the `table` group: every feature.
    All,
}

impl FeatureSet {
    pub const ALL_SETS: [Self; 4] = [Self::Base, Self::Trick, Self::Hidden, Self::All];

    /// Number of features (= network inputs).
    #[must_use]
    pub const fn len(self) -> usize {
        let g = FEATURE_GROUPS;
        match self {
            Self::Base => g.base,
            Self::Trick => g.base + g.trick,
            Self::Hidden => g.base + g.trick + g.hidden,
            Self::All => g.base + g.trick + g.hidden + g.table,
        }
    }

    /// Never empty (the base group is always there); for clippy's sake.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        false
    }

    /// The feature names, in input order.
    #[must_use]
    pub fn names(self) -> &'static [&'static str] {
        &FEATURE_NAMES_V2[..self.len()]
    }

    #[must_use]
    pub const fn has_trick(self) -> bool {
        self.len() > FEATURE_GROUPS.base
    }

    #[must_use]
    pub const fn has_hidden(self) -> bool {
        self.len() > FEATURE_GROUPS.base + FEATURE_GROUPS.trick
    }

    #[must_use]
    pub const fn has_table(self) -> bool {
        self.len() > FEATURE_GROUPS.base + FEATURE_GROUPS.trick + FEATURE_GROUPS.hidden
    }

    /// The set whose network has `inputs` inputs, if any.
    #[must_use]
    pub fn from_input_count(inputs: usize) -> Option<Self> {
        Self::ALL_SETS.into_iter().find(|set| set.len() == inputs)
    }

    /// The set whose names are exactly `names`, if any (a prefix of
    /// `FEATURE_NAMES_V2` of one of the group boundaries).
    #[must_use]
    pub fn from_names<S: AsRef<str>>(names: &[S]) -> Option<Self> {
        Self::ALL_SETS.into_iter().find(|set| {
            set.len() == names.len()
                && set
                    .names()
                    .iter()
                    .zip(names)
                    .all(|(expected, name)| *expected == name.as_ref())
        })
    }

    /// The `TurnContext` parts the features of this set read: exactly
    /// what they need, so a v1 genome pays for nothing it does not use.
    #[must_use]
    pub fn needs(self) -> ContextNeeds {
        let base = ContextNeeds::OPPONENT_PASS_CEILINGS.union(ContextNeeds::UNSEEN);
        let with_trick = if self.has_trick() {
            base.union(ContextNeeds::TRICK)
        } else {
            base
        };
        if self.has_table() {
            with_trick.union(ContextNeeds::TABLE)
        } else {
            with_trick
        }
    }
}

impl std::fmt::Display for FeatureSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Base => "base",
            Self::Trick => "trick",
            Self::Hidden => "hidden",
            Self::All => "all",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategies::neat_player::features::FEATURE_NAMES;

    #[test]
    fn the_v1_names_are_the_prefix_of_the_full_list() {
        assert_eq!(FEATURE_NAMES_V2[..FEATURE_NAMES.len()], FEATURE_NAMES);
        assert_eq!(FEATURE_GROUPS.base, FEATURE_NAMES.len());
        assert_eq!(
            FeatureSet::All.len(),
            FEATURE_COUNT_V2,
            "the groups add up to the full list"
        );
    }

    #[test]
    fn names_are_unique_and_prefixes_nest() {
        let mut seen = std::collections::HashSet::new();
        assert!(FEATURE_NAMES_V2.iter().all(|n| seen.insert(*n)));
        for pair in FeatureSet::ALL_SETS.windows(2) {
            assert!(pair[0].len() < pair[1].len());
            assert_eq!(pair[0].names(), &pair[1].names()[..pair[0].len()]);
        }
    }

    #[test]
    fn sets_are_found_by_input_count_and_by_names() {
        for set in FeatureSet::ALL_SETS {
            assert_eq!(FeatureSet::from_input_count(set.len()), Some(set));
            assert_eq!(FeatureSet::from_names(set.names()), Some(set));
        }
        assert_eq!(FeatureSet::from_input_count(21), None);
        assert_eq!(FeatureSet::from_names(&FEATURE_NAMES_V2[..21]), None);
        assert_eq!(FeatureSet::from_names::<&str>(&[]), None);
    }

    #[test]
    fn needs_follow_the_groups_and_base_is_the_v1_needs() {
        assert_eq!(
            FeatureSet::Base.needs(),
            ContextNeeds::OPPONENT_PASS_CEILINGS.union(ContextNeeds::UNSEEN)
        );
        assert!(FeatureSet::Trick.needs().contains(ContextNeeds::TRICK));
        assert!(!FeatureSet::Trick.needs().contains(ContextNeeds::TABLE));
        assert!(FeatureSet::Hidden.needs().contains(ContextNeeds::TRICK));
        assert!(!FeatureSet::Hidden.needs().contains(ContextNeeds::TABLE));
        assert!(FeatureSet::All.needs().contains(ContextNeeds::TABLE));
        assert!(!FeatureSet::Base.needs().contains(ContextNeeds::TRICK));
        assert!(!FeatureSet::Base.needs().contains(ContextNeeds::TABLE));
    }

    #[test]
    fn serde_uses_the_cli_spelling() {
        assert_eq!(
            serde_json::to_string(&FeatureSet::Base).unwrap(),
            "\"base\""
        );
        assert_eq!(
            serde_json::from_str::<FeatureSet>("\"all\"").unwrap(),
            FeatureSet::All
        );
    }
}
