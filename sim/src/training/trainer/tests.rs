//! Unit tests of the trainer's seed sets and ranking helpers.

use super::super::config::test_support::sample;
use super::seeds::*;
use super::stats::top_indices;

#[test]
fn top_indices_orders_best_first_with_ties_going_to_the_lower_index() {
    let values = [0.5, 0.9, 0.5, -1.0, 0.9];
    assert_eq!(top_indices(&values, 3), vec![1, 4, 0]);
    assert_eq!(top_indices(&values, 1), vec![1]);
    assert_eq!(top_indices(&values, 99), vec![1, 4, 0, 2, 3]);
    assert_eq!(top_indices(&[], 3), [] as [usize; 0]);
}

#[test]
fn champion_selection_has_its_own_seed_stream() {
    // The candidates are ranked on `selection_seeds`; the champion's
    // reported score is then measured on `reeval_seeds`. If they shared
    // matches, the reported score would be inflated by the very
    // selection that chose the champion.
    let config = sample();
    let selection = selection_seeds(&config);
    assert_eq!(selection.len(), config.reeval_matches);
    assert_eq!(
        selection,
        selection_seeds(&config),
        "fixed, not per generation"
    );
    for seed in &selection {
        assert!(!reeval_seeds(&config).contains(seed));
        assert!(!heldout_seeds(&config).contains(seed));
        for generation in 0..200 {
            assert!(!training_seeds(&config, generation).contains(seed));
        }
    }
}

#[test]
fn champions_are_compared_on_one_fixed_seed_set_that_training_never_uses() {
    let config = sample();
    let fixed = reeval_seeds(&config);
    let held_out = heldout_seeds(&config);
    assert_eq!(fixed.len(), config.reeval_matches);
    assert_eq!(held_out.len(), config.reeval_matches * 2);
    // The same every generation (paired comparison between champions).
    assert_eq!(fixed, reeval_seeds(&config));
    // Disjoint from each other and from every generation's training matches.
    for seed in fixed.iter().chain(&held_out) {
        assert_eq!(
            fixed.iter().chain(&held_out).filter(|s| *s == seed).count(),
            1,
            "{seed}"
        );
        for generation in 0..200 {
            assert!(!training_seeds(&config, generation).contains(seed));
        }
    }
}
