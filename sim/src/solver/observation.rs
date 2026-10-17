//! What one seat knows at its turn, as the solver needs it.

use engine::{Card, Combo, DuplicateRule, PassRule, Round, SeatId};

use crate::hand_reading::PassCeilings;
use crate::strategy::TurnContext;

/// A seat's view of the round. Every `Vec` is indexed by seat and includes
/// the seat itself.
#[derive(Debug, Clone)]
pub struct Observation {
    pub duplicate_rule: DuplicateRule,
    pub seat: SeatId,
    pub player_count: u8,
    pub hand: Vec<Card>,
    pub hand_sizes: Vec<usize>,
    /// Active and not passed in the current trick (own seat: `true`).
    pub in_trick: Vec<bool>,
    pub ceilings: Vec<PassCeilings>,
    /// Every card some other seat holds right now.
    pub unseen: Vec<Card>,
    /// Holder and combo of the trick's current play; `None` when this seat leads.
    pub on_table: Option<(SeatId, Combo)>,
}

impl Observation {
    /// `None` when the context cannot be solved from: a pass rule other than
    /// `Final`, or context parts that were not requested or are inconsistent.
    #[must_use]
    pub fn from_context(context: &TurnContext<'_>, duplicate_rule: DuplicateRule) -> Option<Self> {
        let table = context.table;
        let n = usize::from(table.player_count);
        if !(3..=6).contains(&n) || table.pass_rule != PassRule::Final {
            return None;
        }
        if context.opponents.len() + 1 != n {
            return None;
        }
        let opponents_total: usize = context.opponents.iter().map(|o| o.hand_size).sum();
        if context.unseen_cards.len() != opponents_total {
            return None;
        }
        if context.combo_holder.is_some() != context.current_combo.is_some() {
            return None;
        }
        let mut hand_sizes = vec![0usize; n];
        let mut in_trick = vec![false; n];
        let mut ceilings = vec![PassCeilings::default(); n];
        hand_sizes[usize::from(context.seat)] = context.hand.len();
        in_trick[usize::from(context.seat)] = true;
        for o in &context.opponents {
            let s = usize::from(o.seat);
            if s >= n || s == usize::from(context.seat) {
                return None;
            }
            hand_sizes[s] = o.hand_size;
            in_trick[s] = o.in_trick;
            ceilings[s] = o.pass_ceilings;
        }
        Some(Self {
            duplicate_rule,
            seat: context.seat,
            player_count: table.player_count,
            hand: context.hand.to_vec(),
            hand_sizes,
            in_trick,
            ceilings,
            unseen: context.unseen_cards.clone(),
            on_table: context.combo_holder.zip(context.current_combo.copied()),
        })
    }

    /// The same view with every pass ceiling forgotten.
    #[must_use]
    pub fn without_ceilings(&self) -> Self {
        Self {
            ceilings: vec![PassCeilings::default(); self.ceilings.len()],
            ..self.clone()
        }
    }

    /// Every card still in some hand: our own and the unseen ones.
    #[must_use]
    pub fn cards_in_play(&self) -> Vec<Card> {
        let mut cards = self.hand.clone();
        cards.extend_from_slice(&self.unseen);
        cards
    }

    /// The round with `deal` (hands by seat) in place, or `None` if the
    /// position is inconsistent (see [`Round::from_position`]).
    #[must_use]
    pub fn round_for(&self, deal: &[Vec<Card>]) -> Option<Round> {
        let passed: Vec<bool> = (0..usize::from(self.player_count))
            .map(|s| !self.in_trick[s] && self.hand_sizes[s] > 0)
            .collect();
        Round::from_position(
            deal.to_vec(),
            self.duplicate_rule,
            self.seat,
            self.on_table,
            &passed,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand_reading::PassTracker;
    use crate::match_runner::turn_context_for;
    use crate::strategy::ContextNeeds;
    use crate::test_support::for_each_state;
    use engine::{Card, PassRule, Round, SeatId};
    use rand::{RngExt, SeedableRng};

    /// Plays the round on with random moves and returns the finishing order.
    fn finish(round: &mut Round, seed: u64) -> Vec<SeatId> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        while !round.is_complete() {
            let seat = round.seat_to_move().unwrap();
            let moves = round.legal_moves();
            let mv = moves[rng.random_range(0..moves.len())];
            round.submit_move(seat, mv).unwrap();
        }
        round.finishing_order().to_vec()
    }

    #[test]
    fn a_rebuilt_round_behaves_like_the_live_round() {
        let mut checked = 0;
        for_each_state(11, 2, |state, rng| {
            let round = state.round;
            if round.pass_rule() != PassRule::Final {
                return;
            }
            let seat = round.seat_to_move().unwrap();
            let mut tracker = PassTracker::new(usize::from(state.players), state.rule);
            let context = turn_context_for(
                round,
                seat,
                state.players,
                state.round_deck,
                &mut tracker,
                ContextNeeds::ALL,
            );
            let observation = Observation::from_context(&context, state.rule).unwrap();
            let hands: Vec<Vec<Card>> =
                (0..state.players).map(|s| round.hand(s).to_vec()).collect();
            let mut rebuilt = observation.round_for(&hands).unwrap();

            assert_eq!(rebuilt.seat_to_move(), round.seat_to_move());
            assert_eq!(rebuilt.legal_moves(), round.legal_moves());
            assert_eq!(rebuilt.current_combo(), round.current_combo());

            let seed = rng.random_range(0..u64::MAX);
            let out_before = round.finishing_order().len();
            let mut live = round.clone();
            let live_order = finish(&mut live, seed);
            let rebuilt_order = finish(&mut rebuilt, seed);
            // The seats still in when the position was taken finish in the same
            // order; seats already out come first in the rebuilt round.
            assert_eq!(
                live_order[out_before..],
                rebuilt_order[observation
                    .hand_sizes
                    .iter()
                    .filter(|&&size| size == 0)
                    .count()..]
            );
            checked += 1;
        });
        assert!(checked > 100, "only {checked} states were checked");
    }

    #[test]
    fn from_context_rejects_a_free_pass_rule_and_unrequested_context() {
        for_each_state(5, 1, |state, _| {
            let round = state.round;
            let seat = round.seat_to_move().unwrap();
            let mut tracker = PassTracker::new(usize::from(state.players), state.rule);
            let mut context = turn_context_for(
                round,
                seat,
                state.players,
                state.round_deck,
                &mut tracker,
                ContextNeeds::ALL,
            );
            context.table.pass_rule = PassRule::Free;
            assert!(Observation::from_context(&context, state.rule).is_none());
            let bare = turn_context_for(
                round,
                seat,
                state.players,
                state.round_deck,
                &mut tracker,
                ContextNeeds::NONE,
            );
            assert!(Observation::from_context(&bare, state.rule).is_none());
        });
    }
}
