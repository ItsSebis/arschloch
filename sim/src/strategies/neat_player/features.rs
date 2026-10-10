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

    fn bits(features: &[f64; FEATURE_COUNT]) -> Vec<u64> {
        features.iter().map(|v| v.to_bits()).collect()
    }

    /// The fast summary gives bit-identical feature vectors to the scan
    /// implementation it replaced, for every legal move of random states
    /// (3-6 players, both decks and duplicate rules, leading and
    /// following, pass ceilings present).
    #[test]
    fn the_fast_summary_matches_the_reference_on_random_states() {
        use crate::hand_reading::PassTracker;
        use crate::match_runner::turn_context_for;
        use crate::strategy::ContextNeeds;
        use crate::test_support::for_each_state;

        use super::reference::ReferenceTurnSummary;

        let (mut states, mut with_ceilings, mut following, mut vectors) = (0, 0, 0, 0);
        let mut tracker = None;
        for_each_state(5, 40, |state, _| {
            let round = state.round;
            if round.play_history().is_empty() && round.pass_history().is_empty() {
                tracker = Some(PassTracker::new(usize::from(state.players), state.rule));
            }
            let seat = round.seat_to_move().unwrap();
            let context = turn_context_for(
                round,
                seat,
                state.players,
                state.round_deck,
                tracker.as_mut().unwrap(),
                ContextNeeds::ALL,
            );
            let fast = TurnSummary::new(&context, state.rule);
            let reference = ReferenceTurnSummary::new(&context, state.rule);
            for candidate in &round.legal_moves() {
                assert_eq!(
                    bits(&fast.features(candidate)),
                    bits(&reference.features(candidate)),
                    "{candidate:?} in {context_hand:?}",
                    context_hand = context.hand
                );
                vectors += 1;
            }
            states += 1;
            following += usize::from(context.current_combo.is_some());
            with_ceilings += usize::from(
                context
                    .opponents
                    .iter()
                    .any(|o| o.pass_ceilings != PassCeilings::default()),
            );
        });
        assert!(states >= 10_000, "only {states} states");
        assert!(
            following > 1000 && with_ceilings > 1000,
            "{following} {with_ceilings}"
        );
        assert!(vectors > 50_000);
    }

    /// Contexts that are not a physical deck (three or more cards in one
    /// slot, copies sharing a `deal_index`) take the scanning fallbacks and
    /// must still agree with the reference.
    #[test]
    fn the_fast_summary_matches_the_reference_on_unphysical_contexts() {
        use rand::{RngExt, SeedableRng};

        use super::reference::ReferenceTurnSummary;

        const RANKS: [Rank; 13] = [
            Rank::Two,
            Rank::Three,
            Rank::Four,
            Rank::Five,
            Rank::Six,
            Rank::Seven,
            Rank::Eight,
            Rank::Nine,
            Rank::Ten,
            Rank::Jack,
            Rank::Queen,
            Rank::King,
            Rank::Ace,
        ];
        const SUITS: [Suit; 4] = [Suit::Diamonds, Suit::Hearts, Suit::Spades, Suit::Clubs];
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let random_card = |rng: &mut rand::rngs::StdRng, low_ranks: usize| {
            Card::new(
                RANKS[rng.random_range(0..low_ranks)],
                SUITS[rng.random_range(0..4)],
                rng.random_range(0..3),
            )
        };
        for _ in 0..3000 {
            let spread = rng.random_range(1..=13);
            let hand: Vec<Card> = (0..rng.random_range(0..12))
                .map(|_| random_card(&mut rng, spread))
                .collect();
            let unseen: Vec<Card> = (0..rng.random_range(0..40))
                .map(|_| random_card(&mut rng, spread))
                .collect();
            let opponents: Vec<OpponentHand> = (0..rng.random_range(0..6))
                .map(|seat| {
                    let passed = Combo::new(vec![random_card(&mut rng, 13)]).unwrap();
                    OpponentHand {
                        seat,
                        hand_size: rng.random_range(0..8),
                        active: rng.random_bool(0.7),
                        pass_ceilings: read_pass_ceilings(
                            1,
                            &[],
                            &[(0, passed, 0)],
                            DuplicateRule::LastDealtWins,
                        )[0],
                    }
                })
                .collect();
            let table = Combo::new(vec![random_card(&mut rng, 13)]).unwrap();
            let context = TurnContext {
                seat: 0,
                hand: &hand,
                opponents,
                unseen_cards: unseen,
                own_pass_ceilings: PassCeilings::default(),
                current_combo: rng.random_bool(0.5).then_some(&table),
            };
            for rule in [DuplicateRule::FirstDealtWins, DuplicateRule::LastDealtWins] {
                let fast = TurnSummary::new(&context, rule);
                let reference = ReferenceTurnSummary::new(&context, rule);
                let rank = RANKS[rng.random_range(0..13)];
                let size = rng.random_range(1..=4);
                let cards: Vec<Card> = (0..size)
                    .map(|_| Card::new(rank, SUITS[rng.random_range(0..4)], rng.random_range(0..3)))
                    .collect();
                for candidate in [Move::Pass, Move::Play(Combo::new(cards).unwrap())] {
                    assert_eq!(
                        bits(&fast.features(&candidate)),
                        bits(&reference.features(&candidate)),
                        "{candidate:?}, hand {hand:?}"
                    );
                }
            }
        }
    }
}
