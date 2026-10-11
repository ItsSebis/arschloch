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

use engine::{Card, DuplicateRule, Move};

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

use crate::hand_features::strength;

/// The 52 rank/suit "slots" (`rank * 4 + suit`, the order `Card::compare`
/// uses before its duplicate tiebreak) a set of cards occupies, as bit
/// masks, plus the number of cards per rank. A real deck puts at most two
/// cards in a slot (the double deck); `overfull` marks a set that does not
/// and is then answered by scanning the cards instead.
struct SlotSet {
    /// Bit `s`: at least one card in slot `s`.
    first: u64,
    /// Bit `s`: at least two cards in slot `s`.
    second: u64,
    /// `deal_index` of the first two cards seen in each slot.
    deal_index: [[u8; 2]; 52],
    per_rank: [u8; 13],
    overfull: bool,
}

fn slot_of(card: Card) -> usize {
    usize::from(card.rank as u8) * 4 + usize::from(card.suit as u8)
}

impl SlotSet {
    fn new(cards: &[Card]) -> Self {
        let mut set = Self {
            first: 0,
            second: 0,
            deal_index: [[0; 2]; 52],
            per_rank: [0; 13],
            overfull: false,
        };
        for &card in cards {
            let slot = slot_of(card);
            let bit = 1u64 << slot;
            set.per_rank[card.rank as usize] += 1;
            if set.first & bit == 0 {
                set.first |= bit;
                set.deal_index[slot][0] = card.deal_index;
            } else if set.second & bit == 0 {
                set.second |= bit;
                set.deal_index[slot][1] = card.deal_index;
            } else {
                set.overfull = true;
            }
        }
        set
    }

    /// Cards in slots strictly above `slot`.
    fn above(&self, slot: usize) -> usize {
        let shift = slot + 1;
        ((self.first >> shift).count_ones() + (self.second >> shift).count_ones()) as usize
    }

    /// Cards in slots strictly below `slot`.
    fn below(&self, slot: usize) -> usize {
        let mask = (1u64 << slot) - 1;
        ((self.first & mask).count_ones() + (self.second & mask).count_ones()) as usize
    }

    /// Cards in `top`'s own slot that `Card::compare` puts at `want`
    /// relative to `top` (decided by the duplicate tiebreak).
    fn same_slot(&self, top: Card, rule: DuplicateRule, want: Ordering) -> usize {
        let slot = slot_of(top);
        let bit = 1u64 << slot;
        let copies = usize::from(self.first & bit != 0) + usize::from(self.second & bit != 0);
        let probe = |index: u8| Card::new(top.rank, top.suit, index);
        self.deal_index[slot][..copies]
            .iter()
            .filter(|&&index| probe(index).compare(&top, rule) == want)
            .count()
    }

    /// Whether some card of `top`'s rank is above `top` in `Card::compare`.
    fn rank_mate_above(&self, top: Card, rule: DuplicateRule) -> bool {
        let slot = slot_of(top);
        let higher_suits = 3 - (slot & 3);
        let mask = (1u64 << higher_suits) - 1;
        let shifted = slot + 1;
        ((self.first >> shifted) & mask) != 0 || self.same_slot(top, rule, Ordering::Greater) > 0
    }
}

/// Everything about the turn that does not depend on the candidate move,
/// computed once (without allocating) and reused for every candidate.
/// Per-candidate answers are lookups in rank counts and slot masks; the
/// formulas are exactly those of the scan-based implementation this
/// replaced (kept as a test oracle), so the feature values are
/// bit-identical.
pub struct TurnSummary<'a> {
    duplicate_rule: DuplicateRule,
    hand: &'a [Card],
    unseen: &'a [Card],
    opponents: &'a [OpponentHand],
    active_opponents: usize,
    hand_set: SlotSet,
    unseen_set: SlotSet,
    /// Bit `r` of entry `s`: some rank `r` has at least `s` unseen cards.
    unseen_ranks_with: [u16; MAX_COMBO + 1],
    max_hand_rank: Option<usize>,
    state: [f64; FEATURE_COUNT],
}

/// The largest combo size a candidate can have.
const MAX_COMBO: usize = 8;

impl<'a> TurnSummary<'a> {
    #[must_use]
    pub fn new(context: &'a TurnContext<'a>, duplicate_rule: DuplicateRule) -> Self {
        let hand = context.hand;
        let hand_set = SlotSet::new(hand);
        let unseen_set = SlotSet::new(&context.unseen_cards);

        let (mut grouped, mut singletons) = (0usize, 0usize);
        for &n in &hand_set.per_rank {
            match n {
                0 => {}
                1 => singletons += 1,
                n => grouped += usize::from(n),
            }
        }
        let max_hand_rank = hand_set.per_rank.iter().rposition(|&n| n > 0);

        let mut unseen_ranks_with = [0u16; MAX_COMBO + 1];
        for (rank, &n) in unseen_set.per_rank.iter().enumerate() {
            for with in &mut unseen_ranks_with[1..=usize::from(n).min(MAX_COMBO)] {
                *with |= 1 << rank;
            }
        }

        let mut active_opponents = 0usize;
        let (mut hand_min, mut hand_sum) = (usize::MAX, 0usize);
        let mut close = false;
        for o in context.opponents.iter().filter(|o| o.active) {
            active_opponents += 1;
            hand_min = hand_min.min(o.hand_size);
            hand_sum += o.hand_size;
            close |= o.hand_size <= CLOSE_HAND;
        }

        let mut state = [0.0; FEATURE_COUNT];
        state[8] = flag(context.current_combo.is_none());
        state[9] = context
            .current_combo
            .map_or(0.0, |c| count(c.size()) / COMBO_SCALE);
        state[10] = count(hand.len()) / HAND_SCALE;
        state[11] = ratio(grouped, hand.len());
        state[12] = ratio(singletons, hand.len());
        state[13] = count(active_opponents) / 5.0;
        state[14] = if active_opponents == 0 {
            0.0
        } else {
            count(hand_min) / HAND_SCALE
        };
        state[15] = if active_opponents == 0 {
            0.0
        } else {
            count(hand_sum) / count(active_opponents) / HAND_SCALE
        };
        state[16] = flag(close);

        Self {
            duplicate_rule,
            hand,
            unseen: &context.unseen_cards,
            opponents: &context.opponents,
            active_opponents,
            hand_set,
            unseen_set,
            unseen_ranks_with,
            max_hand_rank,
            state,
        }
    }

    /// Hand cards below `top` in `Card::compare`.
    fn hand_below(&self, top: Card) -> usize {
        let rule = self.duplicate_rule;
        if self.hand_set.overfull {
            return self
                .hand
                .iter()
                .filter(|c| c.compare(&top, rule) == Ordering::Less)
                .count();
        }
        self.hand_set.below(slot_of(top)) + self.hand_set.same_slot(top, rule, Ordering::Less)
    }

    /// Unseen cards above `top` in `Card::compare`.
    fn unseen_above(&self, top: Card) -> usize {
        let rule = self.duplicate_rule;
        if self.unseen_set.overfull {
            return self
                .unseen
                .iter()
                .filter(|c| c.compare(&top, rule) == Ordering::Greater)
                .count();
        }
        self.unseen_set.above(slot_of(top))
            + self.unseen_set.same_slot(top, rule, Ordering::Greater)
    }

    /// Ranks with at least `size` unseen cards, some of which beat `top`.
    fn unseen_beating_ranks(&self, top: Card, size: usize) -> usize {
        let rule = self.duplicate_rule;
        let with_size = self.unseen_ranks_with[size.min(MAX_COMBO)];
        let above = (u32::from(with_size) >> (top.rank as usize + 1)).count_ones() as usize;
        let own_rank = usize::from(
            usize::from(self.unseen_set.per_rank[top.rank as usize]) >= size
                && if self.unseen_set.overfull {
                    self.unseen
                        .iter()
                        .any(|c| c.rank == top.rank && c.compare(&top, rule) == Ordering::Greater)
                } else {
                    self.unseen_set.rank_mate_above(top, rule)
                },
        );
        above + own_rank
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
                f[3] = ratio(self.hand_below(top), self.hand.len());
                f[4] = flag(self.max_hand_rank == Some(top.rank as usize));
                let group_size = match usize::from(self.hand_set.per_rank[top.rank as usize]) {
                    0 => size,
                    n => n,
                };
                f[5] = flag(group_size > size);
                f[6] = flag(size >= self.hand.len());
                f[7] = ratio(self.hand.len().saturating_sub(size), self.hand.len());
                f[17] = ratio(self.unseen_above(top), self.unseen.len());
                f[18] = count(self.unseen_beating_ranks(top, size)) / RANK_COUNT;
                f[19] = ratio(
                    self.opponents
                        .iter()
                        .filter(|o| o.active && o.pass_ceilings.cannot_beat(size, top, rule))
                        .count(),
                    self.active_opponents,
                );
            }
        }
        f
    }
}

/// The straightforward implementation the fast `TurnSummary` replaced
/// (scans of the hand and unseen cards per candidate, `rank_groups`),
/// kept as the oracle the fast one is tested against.
#[cfg(test)]
pub(super) mod reference {
    use std::cmp::Ordering;

    use engine::{rank_groups, Card, DuplicateRule, Move, Rank};

    use super::{
        count, flag, ratio, strength, CLOSE_HAND, COMBO_SCALE, FEATURE_COUNT, HAND_SCALE,
        RANK_COUNT,
    };
    use crate::strategy::{OpponentHand, TurnContext};

    /// Everything about the turn that does not depend on the candidate move,
    /// computed once and reused for every candidate.
    pub(crate) struct ReferenceTurnSummary<'a> {
        duplicate_rule: DuplicateRule,
        hand: &'a [Card],
        unseen: &'a [Card],
        unseen_groups: Vec<Vec<Card>>,
        hand_group_sizes: Vec<(Rank, usize)>,
        max_hand_rank: Option<Rank>,
        active_opponents: Vec<&'a OpponentHand>,
        state: [f64; FEATURE_COUNT],
    }

    impl<'a> ReferenceTurnSummary<'a> {
        pub(crate) fn new(context: &'a TurnContext<'a>, duplicate_rule: DuplicateRule) -> Self {
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
        pub(crate) fn features(&self, candidate: &Move) -> [f64; FEATURE_COUNT] {
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
}

#[cfg(test)]
mod tests;
