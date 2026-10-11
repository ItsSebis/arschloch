//! The feature groups appended to the v1 set: `trick` (who is still in
//! the trick), `hidden` (a probabilistic reading of the unseen cards) and
//! `table` (player count, deck, pass rule). Names and the nested prefixes
//! are in `feature_set`; every value is an exact deterministic f64
//! function of the context (no randomness) and lies roughly in `[0, 1]`.

use engine::{Card, DeckVariant, Move, PassRule};

use super::{count, flag, TurnSummary, CLOSE_HAND, FEATURE_COUNT_V2, HAND_SCALE, MAX_COMBO};
use crate::strategies::neat_player::feature_set::{FeatureSet, FEATURE_GROUPS};
use crate::strategy::TurnContext;

/// Cards in one deck: the scale of `expected_beaters`.
const DECK_SCALE: f64 = 52.0;
/// Index of the first `trick` feature.
const TRICK_START: usize = FEATURE_GROUPS.base;
/// Index of the first `table` feature.
const TABLE_START: usize = FEATURE_GROUPS.base + FEATURE_GROUPS.trick + FEATURE_GROUPS.hidden;

/// Fills the candidate-independent features of the `trick` and `table`
/// groups into `state` (those of groups `set` does not have stay 0).
///
/// `trick` (`in_trick` is "active and, under pass rule `final`, not yet
/// passed"; under `free` every active seat is in the trick):
/// - `opponents_in_trick`: opponents in the trick / 5.
/// - `fraction_in_trick`: that count / active opponents (0 if none).
/// - `all_in_trick_close`: 1 if some opponent is in the trick and every
///   one of them holds at most `CLOSE_HAND` cards.
/// - `min_hand_in_trick`: the smallest hand among them / `HAND_SCALE`
///   (0 if none).
/// - `last_to_act_in_trick`: 1 if no opponent in the trick still acts
///   after this seat before the trick would come round to the seat that
///   holds the table combo (every opponent in the trick, when this seat
///   leads; seats are compared in play order starting after this seat).
///
/// `table`: `table_players` = (seats - 3) / 3, `table_double_deck` and
/// `table_pass_final` as 0/1.
pub(super) fn fill_state(
    state: &mut [f64; FEATURE_COUNT_V2],
    context: &TurnContext<'_>,
    set: FeatureSet,
    active_opponents: usize,
) {
    if set.has_trick() {
        let mut in_trick = 0usize;
        let mut min_hand = usize::MAX;
        let mut all_close = true;
        let mut acts_after_me = false;
        // Play order after `seat`: seats above it first, then the wrap.
        let order = |seat: u8| (seat < context.seat, seat);
        for o in context.opponents.iter().filter(|o| o.active && o.in_trick) {
            in_trick += 1;
            min_hand = min_hand.min(o.hand_size);
            all_close &= o.hand_size <= CLOSE_HAND;
            acts_after_me |= context
                .combo_holder
                .is_none_or(|holder| order(o.seat) < order(holder));
        }
        state[TRICK_START] = count(in_trick) / 5.0;
        state[TRICK_START + 1] = if active_opponents == 0 {
            0.0
        } else {
            count(in_trick) / count(active_opponents)
        };
        state[TRICK_START + 2] = flag(in_trick > 0 && all_close);
        state[TRICK_START + 3] = if in_trick == 0 {
            0.0
        } else {
            count(min_hand) / HAND_SCALE
        };
        state[TRICK_START + 4] = flag(!acts_after_me);
    }
    if set.has_table() {
        let table = context.table;
        state[TABLE_START] = count(usize::from(table.player_count).saturating_sub(3)) / 3.0;
        state[TABLE_START + 1] = flag(table.deck == DeckVariant::Double);
        state[TABLE_START + 2] = flag(table.pass_rule == PassRule::Final);
    }
}

/// Per-turn facts of the `hidden` group that do not depend on the
/// candidate.
pub(super) struct HiddenState {
    /// Unseen cards (every card some other active seat holds).
    unseen: usize,
    /// Per rank, the cards of my hand in it that no unseen play can beat
    /// (all the rank's cards, or none): see `my_unbeatable_fraction`.
    unbeatable: [u8; 13],
    unbeatable_total: usize,
}

impl HiddenState {
    pub(super) fn new(summary: &TurnSummary<'_>) -> Self {
        let rule = summary.duplicate_rule;
        let mut hand_top = [None::<Card>; 13];
        for &card in summary.hand {
            let slot = &mut hand_top[card.rank as usize];
            if slot.is_none_or(|top| card.compare(&top, rule).is_gt()) {
                *slot = Some(card);
            }
        }
        let mut unbeatable = [0u8; 13];
        for (rank, top) in hand_top.iter().enumerate() {
            let n = summary.hand_set.per_rank[rank];
            if let Some(top) = top {
                if summary.group_is_unbeatable(*top, usize::from(n)) {
                    unbeatable[rank] = n;
                }
            }
        }
        let unbeatable_total = unbeatable.iter().map(|&n| usize::from(n)).sum();
        Self {
            unseen: summary.unseen.len(),
            unbeatable,
            unbeatable_total,
        }
    }

    /// `[p_win_trick, expected_beaters, my_unbeatable_fraction]` for
    /// `candidate`.
    ///
    /// For a play of size `s` topped by `t`, the *beating groups* are the
    /// ranks with `u >= s` unseen cards that can beat `t`: ranks above `t`
    /// (any `s` of their cards beat it), and `t`'s own rank when some unseen
    /// card of it is above `t` (higher suit, or the duplicate tiebreak),
    /// `b` of its `u` cards. Opponent `i` holds `h` of the `U` unseen cards
    /// (a uniformly random subset). It beats the play through a group when
    /// it holds `k >= s` of the group's `u` cards, at least one of them a
    /// beater (`b = u` for ranks above `t`): the chance is the exact
    /// hypergeometric
    /// `sum_{k>=s} (C(u,k) - C(u-b,k)) * C(U-u, h-k) / C(U,h)`.
    /// - `s = 1`: any one beater card wins, so the beaters are pooled and
    ///   `P_i = 1 - prod_{j<h} (U - m - j) / (U - j)` with `m` the number
    ///   of beater cards; this is exact for one opponent.
    /// - `s >= 2`: `P_i = 1 - prod_g (1 - P_g)`, treating the groups as
    ///   independent (they are slightly negatively dependent: cards held
    ///   for one group are not available for another). A pass ceiling that
    ///   proves the opponent cannot beat the play sets `P_i = 0`.
    ///
    /// Across opponents the hands are also treated as independent, although
    /// they partition the unseen cards. The binomials come from a Pascal
    /// table of f64 sums, summed in a fixed order: deterministic.
    /// - `p_win_trick` = `prod (1 - P_i)` over the opponents in the trick
    ///   (1 when none is; 0 for a pass).
    /// - `expected_beaters` = sum over those not locked out of the expected
    ///   number of cards the opponent holds in beating groups: `h * m / U`
    ///   for `s = 1`, `sum_g sum_{k>=s} k * (C(u,k) - C(u-b,k)) *
    ///   C(U-u, h-k) / C(U,h)` for `s >= 2` (only the cards of groups it
    ///   holds in force count), divided by `DECK_SCALE` (0 for a pass): at
    ///   most the beater-group cards, so within `[0, 1]` in a single deck
    ///   and up to about 2 in a double one.
    /// - `my_unbeatable_fraction`: of the cards left in my hand after the
    ///   candidate, the share lying in a rank group that no unseen group
    ///   of at least its size can beat (its remaining cards played
    ///   together, topped by its highest); 0 for an empty hand.
    pub(super) fn features(&self, summary: &TurnSummary<'_>, candidate: &Move) -> [f64; 3] {
        let rule = summary.duplicate_rule;
        match candidate {
            Move::Pass => [
                0.0,
                0.0,
                fraction(self.unbeatable_total, summary.hand.len()),
            ],
            Move::Play(combo) => {
                let top = combo.top_card(rule);
                let size = combo.size();
                let mask = summary.unseen_beating_rank_mask(top, size);
                let mut groups = [(0usize, 0usize); 13];
                let mut group_count = 0;
                for rank in (0..13).filter(|&rank| mask >> rank & 1 == 1) {
                    let cards = usize::from(summary.unseen_set.per_rank[rank]);
                    let beating = if rank == top.rank as usize {
                        summary
                            .unseen
                            .iter()
                            .filter(|c| c.rank == top.rank && c.compare(&top, rule).is_gt())
                            .count()
                    } else {
                        cards
                    };
                    groups[group_count] = (cards, beating);
                    group_count += 1;
                }
                let groups = &groups[..group_count];
                let pooled: usize = groups.iter().map(|&(_, beating)| beating).sum();
                let mut p_win = 1.0;
                let mut expected = 0.0;
                for o in summary.opponents.iter().filter(|o| o.active && o.in_trick) {
                    if o.pass_ceilings.cannot_beat(size, top, rule) {
                        continue;
                    }
                    let hand = o.hand_size;
                    let (p_beat, cards) = if size == 1 {
                        let held = if self.unseen > 0 {
                            count(hand) * count(pooled) / count(self.unseen)
                        } else {
                            0.0
                        };
                        (beat_probability(self.unseen, pooled, hand), held)
                    } else {
                        let (mut miss, mut held) = (1.0, 0.0);
                        for &(cards, beating) in groups {
                            let (p, e) = group_tail(self.unseen, cards, beating, hand, size);
                            miss *= 1.0 - p;
                            held += e;
                        }
                        ((1.0 - miss).clamp(0.0, 1.0), held)
                    };
                    p_win *= 1.0 - p_beat;
                    expected += cards;
                }

                let rank = top.rank as usize;
                let remaining = usize::from(summary.hand_set.per_rank[rank]).saturating_sub(size);
                let remaining_top = summary
                    .hand
                    .iter()
                    .filter(|c| c.rank == top.rank && !combo.cards().contains(c))
                    .copied()
                    .reduce(|a, b| if b.compare(&a, rule).is_gt() { b } else { a });
                let rank_unbeatable = match remaining_top {
                    Some(card) if summary.group_is_unbeatable(card, remaining) => remaining,
                    _ => 0,
                };
                let kept =
                    self.unbeatable_total - usize::from(self.unbeatable[rank]) + rank_unbeatable;
                [
                    p_win,
                    expected / DECK_SCALE,
                    fraction(kept, summary.hand.len().saturating_sub(size)),
                ]
            }
        }
    }
}

fn fraction(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        0.0
    } else {
        count(part) / count(whole)
    }
}

/// The chance that a hand of `hand` cards drawn without replacement from
/// `unseen` cards, `beaters` of which are beaters, holds at least one:
/// `1 - prod_{j<hand} (unseen - beaters - j) / (unseen - j)`, clamped to
/// `[0, 1]`, 0 without beaters or cards, 1 once the hand cannot avoid them.
pub(super) fn beat_probability(unseen: usize, beaters: usize, hand: usize) -> f64 {
    if beaters == 0 || hand == 0 {
        return 0.0;
    }
    let mut miss = 1.0;
    for j in 0..hand {
        let left = unseen.saturating_sub(j);
        if left <= beaters {
            return 1.0;
        }
        miss *= count(left - beaters) / count(left);
    }
    (1.0 - miss).clamp(0.0, 1.0)
}

/// `C(n, k)` as f64 for `n, k < 128` (0 for `k > n`), from a Pascal table.
fn binomial(n: usize, k: usize) -> f64 {
    static TABLE: std::sync::OnceLock<Vec<[f64; 128]>> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut rows = vec![[0.0f64; 128]; 128];
        for n in 0..128 {
            rows[n][0] = 1.0;
            for k in 1..=n {
                rows[n][k] = rows[n - 1][k - 1] + if k < n { rows[n - 1][k] } else { 0.0 };
            }
        }
        rows
    });
    if n < 128 && k < 128 {
        table[n][k]
    } else {
        0.0
    }
}

/// For one rank group of `cards` unseen cards of which `beating` can beat
/// the play, and an opponent holding `hand` of the `unseen` cards: the
/// probability that it holds at least `size` of the group's cards including
/// a beating one, and the expected number of the group's cards it holds in
/// that case (`sum_k k * P(k cards held and a beater among them)`).
pub(super) fn group_tail(
    unseen: usize,
    cards: usize,
    beating: usize,
    hand: usize,
    size: usize,
) -> (f64, f64) {
    let hand = hand.min(unseen);
    let total = binomial(unseen, hand);
    if total == 0.0 {
        return (0.0, 0.0);
    }
    let (mut probability, mut held) = (0.0, 0.0);
    for k in size..=cards.min(hand) {
        let with_beater = binomial(cards, k) - binomial(cards - beating, k);
        let weight = with_beater * binomial(unseen - cards, hand - k) / total;
        probability += weight;
        held += count(k) * weight;
    }
    (probability.clamp(0.0, 1.0), held)
}

impl TurnSummary<'_> {
    /// Whether no unseen group of at least `size` cards holds a card
    /// above `top` (a play of `size` cards topped by `top` cannot be
    /// beaten).
    fn group_is_unbeatable(&self, top: Card, size: usize) -> bool {
        self.unseen_beating_rank_mask(top, size.min(MAX_COMBO)) == 0
    }
}
