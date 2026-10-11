//! The match seed sets of a run: training, selection, re-evaluation and
//! held-out. Each is a pure function of the config (and the generation).

use super::super::config::TrainConfig;
use super::super::evaluate::match_seed;

/// Training matches of `generation`: different every generation, so a
/// genome cannot be tuned to one set of deals.
pub(super) fn training_seeds(config: &TrainConfig, generation: u32) -> Vec<u64> {
    (0..config.matches_per_genome as u64)
        .map(|i| match_seed(config.seed, 2 * u64::from(generation), i))
        .collect()
}

/// The fixed set every generation's champion is compared on: the same
/// deals every time (so champions are compared like for like) and never
/// part of training.
pub(super) fn reeval_seeds(config: &TrainConfig) -> Vec<u64> {
    (0..config.reeval_matches as u64)
        .map(|i| match_seed(config.seed, u64::MAX - 1, i))
        .collect()
}

/// The set the champion *candidates* are ranked on. It is separate from
/// `reeval_seeds` so that the champion's reported score is measured on
/// matches that played no part in choosing it (a score measured on the
/// selection matches would be inflated by the selection).
pub(super) fn selection_seeds(config: &TrainConfig) -> Vec<u64> {
    (0..config.reeval_matches as u64)
        .map(|i| match_seed(config.seed, u64::MAX - 4, i))
        .collect()
}

/// A second fixed set, used only to confirm a new best champion: it was
/// not used to pick it, so its score is not inflated by the selection.
pub(super) fn heldout_seeds(config: &TrainConfig) -> Vec<u64> {
    (0..config.reeval_matches as u64 * 2)
        .map(|i| match_seed(config.seed, u64::MAX - 2, i))
        .collect()
}
