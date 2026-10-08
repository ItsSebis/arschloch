//! The network's view of a decision: one fixed-size feature vector per
//! candidate move (see docs/superpowers/specs/2026-10-08-neat-engine-design.md,
//! section 4).
//!
//! Every value is a deterministic function of `(candidate, TurnContext,
//! duplicate_rule)` and lies roughly in `[0, 1]` (hand sizes can exceed 1
//! in the double deck). The set and order below are a *contract* with
//! saved genomes: changing either invalidates every trained genome, so
//! `FEATURE_NAMES` is stored in each genome file and checked on load.

use std::cmp::Ordering;

use engine::{rank_groups, Card, DuplicateRule, Move, Rank};

use crate::strategy::{OpponentHand, TurnContext};

pub const FEATURE_NAMES: [&str; 20] = [
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
];

pub const FEATURE_COUNT: usize = FEATURE_NAMES.len();

/// Bumped whenever a feature's formula, scale or meaning changes without
/// its name changing (for example `HAND_SCALE`, `CLOSE_HAND`, the opponent
/// divisor): saved genomes were trained against the old values and would
/// load cleanly but misplay. Renaming, adding, removing or reordering
/// features is already caught by the names stored in each genome file.
pub const FEATURE_SET_VERSION: u32 = 1;

/// Hand sizes are scaled by a typical single-deck hand (52 / 4).
const HAND_SCALE: f64 = 13.0;
/// The largest combo any seat can field (double deck, 8 of a rank).
const COMBO_SCALE: f64 = 8.0;
const RANK_COUNT: f64 = 13.0;
/// An opponent this close to going out makes the round urgent.
const CLOSE_HAND: usize = 2;

#[allow(clippy::cast_precision_loss)] // counts here are tiny (< 2^52)
fn count(n: usize) -> f64 {
    n as f64
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    count(numerator) / count(denominator.max(1))
}

fn flag(condition: bool) -> f64 {
    f64::from(u8::from(condition))
}

/// Position of `card` in the 52-card order (rank first, then suit), in
/// `[0, 1]`. Ignores the double-deck duplicate tiebreak, which has no
/// strength meaning of its own.
fn strength(card: Card) -> f64 {
    f64::from(card.rank as u8 * 4 + card.suit as u8) / 51.0
}

/// Everything about the turn that does not depend on the candidate move,
/// computed once and reused for every candidate.
pub struct TurnSummary<'a> {
    duplicate_rule: DuplicateRule,
    hand: &'a [Card],
    unseen: &'a [Card],
    unseen_groups: Vec<Vec<Card>>,
    hand_group_sizes: Vec<(Rank, usize)>,
    max_hand_rank: Option<Rank>,
    active_opponents: Vec<&'a OpponentHand>,
    state: [f64; FEATURE_COUNT],
}

impl<'a> TurnSummary<'a> {
    #[must_use]
    pub fn new(context: &'a TurnContext<'a>, duplicate_rule: DuplicateRule) -> Self {
        let hand = context.hand;
        let groups = rank_groups(hand);
        let hand_group_sizes: Vec<(Rank, usize)> =
            groups.iter().map(|g| (g[0].rank, g.len())).collect();
        let active_opponents: Vec<&OpponentHand> =
            context.opponents.iter().filter(|o| o.active).collect();
        let hands: Vec<usize> = active_opponents.iter().map(|o| o.hand_size).collect();

        let mut state = [0.0; FEATURE_COUNT];
        state[8] = flag(context.current_combo.is_none());
        state[9] = context
            .current_combo
            .map_or(0.0, |c| count(c.size()) / COMBO_SCALE);
        state[10] = count(hand.len()) / HAND_SCALE;
        state[11] = ratio(
            groups.iter().filter(|g| g.len() >= 2).map(Vec::len).sum(),
            hand.len(),
        );
        state[12] = ratio(groups.iter().filter(|g| g.len() == 1).count(), hand.len());
        state[13] = count(active_opponents.len()) / 5.0;
        state[14] = hands.iter().min().map_or(0.0, |&h| count(h) / HAND_SCALE);
        state[15] = if hands.is_empty() {
            0.0
        } else {
            count(hands.iter().sum()) / count(hands.len()) / HAND_SCALE
        };
        state[16] = flag(hands.iter().any(|&h| h <= CLOSE_HAND));

        Self {
            duplicate_rule,
            hand,
            unseen: &context.unseen_cards,
            unseen_groups: rank_groups(&context.unseen_cards),
            max_hand_rank: hand.iter().map(|c| c.rank).max(),
            hand_group_sizes,
            active_opponents,
            state,
        }
    }

    /// The feature vector for playing or passing `candidate` this turn.
    #[must_use]
    pub fn features(&self, candidate: &Move) -> [f64; FEATURE_COUNT] {
        let mut f = self.state;
        match candidate {
            Move::Pass => {
                f[0] = 1.0;
                f[7] = 1.0;
            }
            Move::Play(combo) => {
                let rule = self.duplicate_rule;
                let top = combo.top_card(rule);
                let size = combo.size();
                f[1] = count(size) / COMBO_SCALE;
                f[2] = strength(top);
                f[3] = ratio(
                    self.hand
                        .iter()
                        .filter(|c| c.compare(&top, rule) == Ordering::Less)
                        .count(),
                    self.hand.len(),
                );
                f[4] = flag(self.max_hand_rank == Some(top.rank));
                let group_size = self
                    .hand_group_sizes
                    .iter()
                    .find(|(rank, _)| *rank == top.rank)
                    .map_or(size, |&(_, n)| n);
                f[5] = flag(group_size > size);
                f[6] = flag(size >= self.hand.len());
                f[7] = ratio(self.hand.len().saturating_sub(size), self.hand.len());
                f[17] = ratio(
                    self.unseen
                        .iter()
                        .filter(|c| c.compare(&top, rule) == Ordering::Greater)
                        .count(),
                    self.unseen.len(),
                );
                let beating_ranks = self
                    .unseen_groups
                    .iter()
                    .filter(|group| {
                        group.len() >= size
                            && group
                                .iter()
                                .any(|c| c.compare(&top, rule) == Ordering::Greater)
                    })
                    .count();
                f[18] = count(beating_ranks) / RANK_COUNT;
                f[19] = ratio(
                    self.active_opponents
                        .iter()
                        .filter(|o| o.pass_ceilings.cannot_beat(size, top, rule))
                        .count(),
                    self.active_opponents.len(),
                );
            }
        }
        f
    }
}

#[cfg(test)]
mod tests {
    use engine::{Combo, Rank, Suit};

    use super::*;
    use crate::hand_reading::{read_pass_ceilings, PassCeilings};

    const RULE: DuplicateRule = DuplicateRule::FirstDealtWins;

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit, 0)
    }

    fn play(cards: &[Card]) -> Move {
        Move::Play(Combo::new(cards.to_vec()).unwrap())
    }

    fn opponent(
        seat: u8,
        hand_size: usize,
        active: bool,
        pass_ceilings: PassCeilings,
    ) -> OpponentHand {
        OpponentHand {
            seat,
            hand_size,
            active,
            pass_ceilings,
        }
    }

    /// Hand: 3d 3h 9s Ac. Unseen: 4d 9c Kh Kd Ks.
    struct Scenario {
        hand: Vec<Card>,
        unseen: Vec<Card>,
        opponents: Vec<OpponentHand>,
    }

    fn scenario() -> Scenario {
        Scenario {
            hand: vec![
                card(Rank::Three, Suit::Diamonds),
                card(Rank::Three, Suit::Hearts),
                card(Rank::Nine, Suit::Spades),
                card(Rank::Ace, Suit::Clubs),
            ],
            unseen: vec![
                card(Rank::Four, Suit::Diamonds),
                card(Rank::Nine, Suit::Clubs),
                card(Rank::King, Suit::Hearts),
                card(Rank::King, Suit::Diamonds),
                card(Rank::King, Suit::Spades),
            ],
            opponents: vec![
                opponent(1, 3, true, PassCeilings::default()),
                opponent(2, 2, true, PassCeilings::default()),
                opponent(3, 0, false, PassCeilings::default()),
            ],
        }
    }

    fn context<'a>(s: &'a Scenario, table: Option<&'a Combo>) -> TurnContext<'a> {
        TurnContext {
            seat: 0,
            hand: &s.hand,
            opponents: s.opponents.clone(),
            unseen_cards: s.unseen.clone(),
            own_pass_ceilings: PassCeilings::default(),
            current_combo: table,
        }
    }

    fn approx(actual: f64, expected: f64, name: &str) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "{name}: {actual} vs {expected}"
        );
    }

    #[test]
    fn names_are_unique_and_match_the_count() {
        assert_eq!(FEATURE_NAMES.len(), FEATURE_COUNT);
        let mut sorted = FEATURE_NAMES.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), FEATURE_COUNT);
    }

    #[test]
    fn state_features_describe_the_table_and_opponents() {
        let s = scenario();
        let ctx = context(&s, None);
        let f = TurnSummary::new(&ctx, RULE).features(&play(&[card(Rank::Nine, Suit::Spades)]));
        approx(f[8], 1.0, "is_leading");
        approx(f[9], 0.0, "table_combo_size");
        approx(f[10], 4.0 / 13.0, "hand_size");
        approx(f[11], 2.0 / 4.0, "grouped_fraction (3d,3h)");
        approx(f[12], 2.0 / 4.0, "singleton_fraction (9s, Ac)");
        approx(f[13], 2.0 / 5.0, "active_opponents (inactive one ignored)");
        approx(f[14], 2.0 / 13.0, "min_opponent_hand");
        approx(f[15], 2.5 / 13.0, "mean_opponent_hand");
        approx(f[16], 1.0, "opponent_close (a seat holds 2 cards)");
    }

    #[test]
    fn following_a_combo_sets_the_table_features() {
        let s = scenario();
        let on_table = Combo::new(vec![card(Rank::Five, Suit::Hearts)]).unwrap();
        let ctx = context(&s, Some(&on_table));
        let f = TurnSummary::new(&ctx, RULE).features(&Move::Pass);
        approx(f[8], 0.0, "is_leading");
        approx(f[9], 1.0 / 8.0, "table_combo_size");
    }

    #[test]
    fn passing_has_only_state_features_and_the_pass_flag() {
        let s = scenario();
        let ctx = context(&s, None);
        let summary = TurnSummary::new(&ctx, RULE);
        let f = summary.features(&Move::Pass);
        approx(f[0], 1.0, "is_pass");
        approx(f[7], 1.0, "hand_left_after (hand unchanged)");
        for index in [1, 2, 3, 4, 5, 6, 17, 18, 19] {
            approx(f[index], 0.0, FEATURE_NAMES[index]);
        }
    }

    #[test]
    fn playing_the_hands_top_single_sets_the_play_features() {
        let s = scenario();
        let ctx = context(&s, None);
        let f = TurnSummary::new(&ctx, RULE).features(&play(&[card(Rank::Ace, Suit::Clubs)]));
        approx(f[0], 0.0, "is_pass");
        approx(f[1], 1.0 / 8.0, "combo_size");
        approx(f[2], 1.0, "top_strength (Ac is the strongest card)");
        approx(f[3], 3.0 / 4.0, "hand_below_top");
        approx(f[4], 1.0, "uses_top_group");
        approx(f[5], 0.0, "splits_group");
        approx(f[6], 0.0, "empties_hand");
        approx(f[7], 3.0 / 4.0, "hand_left_after");
        approx(f[17], 0.0, "unseen_outranking (nothing beats Ac)");
        approx(f[18], 0.0, "unseen_beaters");
    }

    #[test]
    fn splitting_a_pair_and_emptying_the_hand_are_flagged() {
        let s = scenario();
        let ctx = context(&s, None);
        let summary = TurnSummary::new(&ctx, RULE);
        let split = summary.features(&play(&[card(Rank::Three, Suit::Hearts)]));
        approx(split[5], 1.0, "splits_group (a single from the 3s pair)");
        let whole = summary.features(&play(&[
            card(Rank::Three, Suit::Diamonds),
            card(Rank::Three, Suit::Hearts),
        ]));
        approx(whole[5], 0.0, "splits_group (the whole pair)");
        approx(whole[1], 2.0 / 8.0, "combo_size");

        let last_card = Scenario {
            hand: vec![card(Rank::Two, Suit::Clubs)],
            unseen: Vec::new(),
            opponents: vec![opponent(1, 5, true, PassCeilings::default())],
        };
        let ctx = context(&last_card, None);
        let f = TurnSummary::new(&ctx, RULE).features(&play(&[card(Rank::Two, Suit::Clubs)]));
        approx(f[6], 1.0, "empties_hand");
        approx(f[7], 0.0, "hand_left_after");
        approx(f[17], 0.0, "unseen_outranking with nothing unseen");
    }

    #[test]
    fn unseen_features_count_cards_and_ranks_that_beat_the_play() {
        let s = scenario();
        let ctx = context(&s, None);
        let summary = TurnSummary::new(&ctx, RULE);
        // Playing 3d: every unseen card (5) outranks it; the four
        // distinct unseen ranks (4, 9, K) -> 3 ranks can beat a single.
        let low = summary.features(&play(&[card(Rank::Three, Suit::Diamonds)]));
        approx(low[17], 1.0, "unseen_outranking");
        approx(low[18], 3.0 / 13.0, "unseen_beaters (4, 9, K)");
        // Playing 9s: 9c (higher suit) and the three kings beat it, 4d does not.
        let nine = summary.features(&play(&[card(Rank::Nine, Suit::Spades)]));
        approx(nine[17], 4.0 / 5.0, "unseen_outranking");
        approx(nine[18], 2.0 / 13.0, "unseen_beaters (9c, K)");
        // A pair can only be beaten by a rank with 2+ unseen cards: just kings.
        let pair = summary.features(&play(&[
            card(Rank::Three, Suit::Diamonds),
            card(Rank::Three, Suit::Hearts),
        ]));
        approx(pair[18], 1.0 / 13.0, "unseen_beaters for a pair");
    }

    #[test]
    fn opponents_locked_out_counts_seats_whose_passes_prove_they_cannot_beat() {
        let mut s = scenario();
        // Seat 1 passed against the King of Hearts: it cannot beat any
        // single at or above that card. Seat 2 has said nothing.
        let king = Combo::new(vec![card(Rank::King, Suit::Hearts)]).unwrap();
        let ceilings = read_pass_ceilings(4, &[], &[(1, king, 0)], RULE);
        s.opponents[0].pass_ceilings = ceilings[1];
        let ctx = context(&s, None);
        let summary = TurnSummary::new(&ctx, RULE);
        let ace = summary.features(&play(&[card(Rank::Ace, Suit::Clubs)]));
        approx(
            ace[19],
            1.0 / 2.0,
            "one of two active opponents is locked out",
        );
        let nine = summary.features(&play(&[card(Rank::Nine, Suit::Spades)]));
        approx(
            nine[19],
            0.0,
            "9s is below the passed King, so nobody is locked out",
        );
    }

    #[test]
    fn no_active_opponents_gives_zero_not_nan() {
        let mut s = scenario();
        s.opponents.clear();
        let ctx = context(&s, None);
        let f = TurnSummary::new(&ctx, RULE).features(&play(&[card(Rank::Ace, Suit::Clubs)]));
        assert!(f.iter().all(|v| v.is_finite()));
        approx(f[13], 0.0, "active_opponents");
        approx(f[14], 0.0, "min_opponent_hand");
        approx(f[19], 0.0, "opponents_locked_out");
    }

    #[test]
    fn strength_follows_the_house_card_order() {
        let low = card(Rank::Two, Suit::Diamonds);
        let same_rank_higher_suit = card(Rank::Two, Suit::Clubs);
        let next_rank = card(Rank::Three, Suit::Diamonds);
        assert!(strength(low) < strength(same_rank_higher_suit));
        assert!(strength(same_rank_higher_suit) < strength(next_rank));
        approx(strength(low), 0.0, "weakest");
        approx(strength(card(Rank::Ace, Suit::Clubs)), 1.0, "strongest");
    }
}
