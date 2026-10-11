mod adaptive;
mod card_counter;
mod endgame_denial;
mod greedy_highest;
mod hold_back_pairs;
mod lowest_legal;
mod neat_player;
mod random_legal;

use engine::{Card, DuplicateRule, Move};

pub use adaptive::{Adaptive, AdaptiveConfig, DenialMode};
pub use card_counter::CardCounter;
pub use endgame_denial::EndgameDenial;
pub use greedy_highest::GreedyHighest;
pub use hold_back_pairs::HoldBackPairs;
pub use lowest_legal::LowestLegal;
pub use neat_player::{
    GenomeFile, GenomeFileError, NeatStrategy, ScoredCandidate, TurnSummary, FEATURE_COUNT,
    FEATURE_NAMES, FEATURE_SET_VERSION, FORMAT_VERSION,
};
pub use random_legal::RandomLegal;

/// The highest of the `unseen` cards under `Card::compare`, or `None` when
/// none is left (then nothing can beat any play). Shared by `CardCounter`
/// and `Adaptive`'s safety proof.
pub(crate) fn highest_unseen(unseen: &[Card], duplicate_rule: DuplicateRule) -> Option<Card> {
    unseen
        .iter()
        .copied()
        .max_by(|a, b| a.compare(b, duplicate_rule))
}

/// The lowest play among `moves`: fewest cards first, then the lowest top
/// card; the first of equals. `None` without a play.
pub(crate) fn lowest_play<'a>(
    moves: impl IntoIterator<Item = &'a Move>,
    duplicate_rule: DuplicateRule,
) -> Option<&'a Move> {
    moves
        .into_iter()
        .filter_map(|mv| match mv {
            Move::Play(combo) => Some((combo.size(), combo.top_card(duplicate_rule), mv)),
            Move::Pass => None,
        })
        .min_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.compare(&b.1, duplicate_rule))
        })
        .map(|(_, _, mv)| mv)
}
