mod adaptive;
mod card_counter;
mod endgame_denial;
mod greedy_highest;
mod hold_back_pairs;
mod lowest_legal;
mod random_legal;

use engine::{Card, DuplicateRule};

pub use adaptive::{Adaptive, AdaptiveConfig, DenialMode};
pub use card_counter::CardCounter;
pub use endgame_denial::EndgameDenial;
pub use greedy_highest::GreedyHighest;
pub use hold_back_pairs::HoldBackPairs;
pub use lowest_legal::LowestLegal;
pub use random_legal::RandomLegal;

/// The naive "give up your highest `count` cards" behavior from Phase 1
/// (`engine::exchange`'s old default), reused by strategies that have no
/// stronger opinion about which cards to give up.
pub(crate) fn take_highest_naive(
    hand: &[Card],
    count: usize,
    duplicate_rule: DuplicateRule,
) -> Vec<Card> {
    let mut sorted = hand.to_vec();
    sorted.sort_by(|a, b| a.compare(b, duplicate_rule));
    sorted.split_off(sorted.len() - count)
}
