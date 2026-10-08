//! Turn-order and pass-counting bookkeeping for a single trick, kept
//! deliberately unaware of cards or combos so it's simple to reason
//! about in isolation. `round.rs` owns hand contents and combo legality;
//! this type only tracks whose turn it is and when a trick ends. See
//! docs/RULES.md, "Playing a Round".

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::SeatId;

/// What a pass means for the rest of the trick.
///
/// Under `Final` (the rules of the game, and the default) a seat that has
/// passed is out of the trick: it is skipped until the trick ends and cannot
/// play when the turn comes round again. Under `Free` (how the simulator
/// behaved before the rule was fixed, kept so earlier results can be
/// reproduced) a pass is only a decline of the current play: the seat may
/// still play the next time its turn comes, after someone has played again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PassRule {
    Free,
    #[default]
    Final,
}

impl fmt::Display for PassRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Free => "free",
            Self::Final => "final",
        })
    }
}

impl FromStr for PassRule {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        match text.trim().to_ascii_lowercase().as_str() {
            "free" => Ok(Self::Free),
            "final" => Ok(Self::Final),
            other => Err(format!(
                "unknown pass rule `{other}`; expected free or final"
            )),
        }
    }
}

/// One trick's turn-taking state: who led it, whose turn it is now, who
/// currently holds the winning play (if anyone has played yet), and how
/// many consecutive passes are needed to resolve it.
#[derive(Debug)]
pub(crate) struct Trick {
    leader: SeatId,
    turn: SeatId,
    winner: Option<SeatId>,
    passes_since_last_play: u8,
    required_passes: u8,
    rule: PassRule,
    /// Seats that have passed this trick (only used under `Final`).
    passed: [bool; 6],
}

impl Trick {
    pub(crate) fn new(leader: SeatId, rule: PassRule) -> Self {
        Self {
            leader,
            turn: leader,
            winner: None,
            passes_since_last_play: 0,
            required_passes: 0,
            rule,
            passed: [false; 6],
        }
    }

    /// The seats that may still act after `seat` under `Final`: active,
    /// not passed, and not the current winner. `None` of them means the
    /// trick is over.
    fn next_eligible_after(&self, seat: SeatId, active: &[bool]) -> Option<SeatId> {
        let n = active.len();
        let winner = self.winner.map(usize::from);
        (1..=n)
            .map(|offset| (usize::from(seat) + offset) % n)
            .find(|&s| active[s] && !self.passed[s] && Some(s) != winner)
            .map(|s| SeatId::try_from(s).expect("seats are at most 6"))
    }

    /// Who leads when the trick ends: the winner, or the next active seat
    /// if the winner emptied their hand.
    fn new_leader(&self, active: &[bool]) -> SeatId {
        let winner = self
            .winner
            .expect("a trick only ends after a play established a winner");
        if active[usize::from(winner)] {
            winner
        } else {
            next_active_seat_after(winner, active)
        }
    }

    pub(crate) fn leader(&self) -> SeatId {
        self.leader
    }

    pub(crate) fn turn(&self) -> SeatId {
        self.turn
    }

    /// Whether anyone has played a combo yet this trick (the leader must
    /// play; every later active seat may play or pass).
    pub(crate) fn has_current_play(&self) -> bool {
        self.winner.is_some()
    }

    /// Records that `seat` (the current `turn`) played a combo. `active`
    /// reflects hand-emptiness immediately after this play (so if `seat`
    /// just emptied their hand, `active[seat]` is already `false`).
    ///
    /// Returns the new leader if no other seat can still act (only under
    /// `Final`: every other active seat has already passed), which ends
    /// the trick at once.
    pub(crate) fn record_play(&mut self, seat: SeatId, active: &[bool]) -> Option<SeatId> {
        self.winner = Some(seat);
        self.passes_since_last_play = 0;
        if self.rule == PassRule::Final {
            return match self.next_eligible_after(seat, active) {
                Some(next) => {
                    self.turn = next;
                    None
                }
                None => Some(self.new_leader(active)),
            };
        }
        let active_count = u8::try_from(active.iter().filter(|&&a| a).count())
            .expect("table sizes are capped at 6 seats");
        let winner_still_active = active[usize::from(seat)];
        self.required_passes = active_count - u8::from(winner_still_active);
        self.turn = next_active_seat_after(seat, active);
        None
    }

    /// Records that `seat` (the current `turn`) passed. `active` is
    /// unaffected by a pass, so it's the same snapshot in effect since
    /// the last play. Returns `Some(new_leader)` if this pass resolves
    /// the trick (every other active seat has now passed since the last
    /// play), else `None` (and `turn` has advanced).
    pub(crate) fn record_pass(&mut self, seat: SeatId, active: &[bool]) -> Option<SeatId> {
        if self.rule == PassRule::Final {
            self.passed[usize::from(seat)] = true;
            return match self.next_eligible_after(seat, active) {
                Some(next) => {
                    self.turn = next;
                    None
                }
                None => Some(self.new_leader(active)),
            };
        }
        self.passes_since_last_play += 1;
        if self.passes_since_last_play < self.required_passes {
            self.turn = next_active_seat_after(seat, active);
            return None;
        }
        let winner = self
            .winner
            .expect("record_pass is only reachable after a play established required_passes > 0");
        let new_leader = if active[usize::from(winner)] {
            winner
        } else {
            next_active_seat_after(winner, active)
        };
        Some(new_leader)
    }
}

fn next_active_seat_after(seat: SeatId, active: &[bool]) -> SeatId {
    let n = active.len();
    let start = usize::from(seat);
    for offset in 1..=n {
        let candidate = (start + offset) % n;
        if active[candidate] {
            return SeatId::try_from(candidate)
                .expect("candidate is a valid index into active, which is at most 6 seats");
        }
    }
    unreachable!(
        "next_active_seat_after requires at least one active seat; \
         callers must ensure the round isn't already complete"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_trick_starts_with_leader_to_move_and_no_current_play() {
        let trick = Trick::new(2, PassRule::Free);
        assert_eq!(trick.leader(), 2);
        assert_eq!(trick.turn(), 2);
        assert!(!trick.has_current_play());
    }

    #[test]
    fn record_play_sets_winner_and_required_passes_for_remaining_active_seats() {
        let mut trick = Trick::new(0, PassRule::Free);
        let active = [true, true, true, true];
        trick.record_play(0, &active);
        assert!(trick.has_current_play());
        assert_eq!(trick.turn(), 1);
        assert_eq!(trick.record_pass(1, &active), None);
        assert_eq!(trick.record_pass(2, &active), None);
        assert_eq!(trick.record_pass(3, &active), Some(0));
    }

    #[test]
    fn a_beating_play_mid_trick_resets_required_passes() {
        let mut trick = Trick::new(0, PassRule::Free);
        let active = [true, true, true, true];
        trick.record_play(0, &active);
        assert_eq!(trick.record_pass(1, &active), None);
        trick.record_play(2, &active); // seat 2 beats seat 0's combo
        assert_eq!(trick.turn(), 3);
        // All 3 other active seats (3, 0, 1) must pass again to resolve,
        // even though seat 1 already passed on the previous play.
        assert_eq!(trick.record_pass(3, &active), None);
        assert_eq!(trick.record_pass(0, &active), None);
        assert_eq!(trick.record_pass(1, &active), Some(2));
    }

    #[test]
    fn everyone_but_the_leader_passing_returns_the_leader_as_new_leader() {
        let mut trick = Trick::new(1, PassRule::Free);
        let active = [true, true, true];
        trick.record_play(1, &active);
        assert_eq!(trick.record_pass(2, &active), None);
        assert_eq!(trick.record_pass(0, &active), Some(1));
    }

    #[test]
    fn winner_emptying_their_hand_hands_the_lead_to_the_next_active_seat() {
        let mut trick = Trick::new(0, PassRule::Free);
        // Seat 0 plays and empties their hand.
        let active_after_play = [false, true, true];
        trick.record_play(0, &active_after_play);
        // Only 2 active seats remain (1 and 2); both must pass to resolve.
        assert_eq!(trick.record_pass(1, &active_after_play), None);
        assert_eq!(trick.record_pass(2, &active_after_play), Some(1));
    }
}
