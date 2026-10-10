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
    /// For a play of size `s` topped by `t`, `m` is the number of unseen
    /// cards lying in rank groups of at least `s` unseen cards that hold a
    /// card above `t`: the cards that could take part in a beating play.
    /// This counts cards, not combinations, so it is exact for singles and
    /// an upper bound on "holds a beater" for larger plays (a pair needs
    /// two of the group's cards, not one). Opponent `i`, holding `h` of
    /// the `U` unseen cards, is then taken to hold at least one of the `m`
    /// as `P_i = 1 - prod_{j<h} (U - m - j) / (U - j)` (sampling without
    /// replacement), and 0 if its pass ceiling proves it cannot beat the
    /// play. The opponents' hands are treated as independent, although
    /// they partition the unseen cards.
    /// - `p_win_trick` = `prod (1 - P_i)` over the opponents in the trick
    ///   (1 when none is; 0 for a pass).
    /// - `expected_beaters` = sum over them (not locked out) of
    ///   `h * m / U`, the expected number of such cards in the hand,
    ///   divided by `DECK_SCALE` (0 for a pass): at most `m`, so within
    ///   `[0, 1]` in a single deck and up to about 2 in a double one.
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
                let beaters: usize = (0..13)
                    .filter(|&rank| mask >> rank & 1 == 1)
                    .map(|rank| usize::from(summary.unseen_set.per_rank[rank]))
                    .sum();
                let mut p_win = 1.0;
                let mut expected = 0.0;
                for o in summary.opponents.iter().filter(|o| o.active && o.in_trick) {
                    if o.pass_ceilings.cannot_beat(size, top, rule) {
                        continue;
                    }
                    p_win *= 1.0 - beat_probability(self.unseen, beaters, o.hand_size);
                    if self.unseen > 0 {
                        expected += count(o.hand_size) * count(beaters) / count(self.unseen);
                    }
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

impl TurnSummary<'_> {
    /// Whether no unseen group of at least `size` cards holds a card
    /// above `top` (a play of `size` cards topped by `top` cannot be
    /// beaten).
    fn group_is_unbeatable(&self, top: Card, size: usize) -> bool {
        self.unseen_beating_rank_mask(top, size.min(MAX_COMBO)) == 0
    }
}
