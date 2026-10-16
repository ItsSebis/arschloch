//! Per-move expected outcome over all consistent deals.

use engine::Move;

use super::deals::for_each_deal;
use super::observation::Observation;
use super::search::Search;
use super::SolverConfig;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveStats {
    pub mv: Move,
    /// Mean 0-based finishing place of the acting seat over all deals.
    pub mean_place: f64,
    pub best_place: u8,
    pub worst_place: u8,
    /// Share of the deals in which this move is (one of) the best.
    pub best_share: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoveReport {
    /// One entry per legal move, in the engine's move order.
    pub moves: Vec<MoveStats>,
    pub deals: u64,
    pub nodes: u64,
    pub player_count: u8,
}

impl MoveReport {
    /// The move with the lowest mean place (the first one on ties).
    ///
    /// # Panics
    ///
    /// Panics on an empty report (`analyze` never returns one).
    #[must_use]
    pub fn best(&self) -> &MoveStats {
        let mut best = &self.moves[0];
        for stats in &self.moves[1..] {
            if stats.mean_place < best.mean_place {
                best = stats;
            }
        }
        best
    }

    /// `1 - 2 * place / (n - 1)`: +1 first out, -1 last.
    #[must_use]
    pub fn score(&self, stats: &MoveStats) -> f64 {
        1.0 - 2.0 * stats.mean_place / f64::from(self.player_count - 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveError {
    TooManyUnseen,
    BudgetExceeded,
    NoConsistentDeal,
    InvalidPosition,
}

/// Solves the observed position exactly for every consistent deal. The
/// places are absolute: `Round::from_position` enters the seats already out
/// first in the finishing order, so the seats still playing take the places
/// after them.
///
/// # Errors
///
/// See [`SolveError`].
#[allow(clippy::cast_precision_loss)] // deal counts are far below 2^52
pub fn analyze(observation: &Observation, config: &SolverConfig) -> Result<MoveReport, SolveError> {
    if observation.unseen.len() > config.max_unseen {
        return Err(SolveError::TooManyUnseen);
    }
    let seat = usize::from(observation.seat);
    let mut search = Search::new(config.node_budget);
    let mut moves: Vec<Move> = Vec::new();
    let mut sums: Vec<f64> = Vec::new();
    let mut best_places: Vec<u8> = Vec::new();
    let mut worst_places: Vec<u8> = Vec::new();
    let mut best_counts: Vec<u64> = Vec::new();
    let mut error = None;

    let deals = for_each_deal(observation, |deal| {
        let Some(round) = observation.round_for(deal) else {
            error = Some(SolveError::InvalidPosition);
            return false;
        };
        let Ok(values) = search.root_values(&round) else {
            error = Some(SolveError::BudgetExceeded);
            return false;
        };
        if moves.is_empty() {
            moves = values.iter().map(|(mv, _)| *mv).collect();
            sums = vec![0.0; moves.len()];
            best_places = vec![u8::MAX; moves.len()];
            worst_places = vec![0; moves.len()];
            best_counts = vec![0; moves.len()];
        }
        let lowest = values.iter().map(|(_, p)| p[seat]).min().unwrap_or(0);
        for (i, (_, places)) in values.iter().enumerate() {
            let place = places[seat];
            sums[i] += f64::from(place);
            best_places[i] = best_places[i].min(place);
            worst_places[i] = worst_places[i].max(place);
            if place == lowest {
                best_counts[i] += 1;
            }
        }
        true
    });

    if let Some(error) = error {
        return Err(error);
    }
    if deals == 0 {
        return Err(SolveError::NoConsistentDeal);
    }
    let denominator = deals as f64;
    Ok(MoveReport {
        moves: moves
            .iter()
            .enumerate()
            .map(|(i, &mv)| MoveStats {
                mv,
                mean_place: sums[i] / denominator,
                best_place: best_places[i],
                worst_place: worst_places[i],
                best_share: best_counts[i] as f64 / denominator,
            })
            .collect(),
        deals,
        nodes: search.nodes_used(),
        player_count: observation.player_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand_reading::{read_pass_ceilings, PassCeilings};
    use crate::solver::{Observation, SolverConfig};
    use engine::{Card, Combo, DuplicateRule, Move, Rank, Suit};

    fn c(rank: Rank, suit: Suit, i: u8) -> Card {
        Card::new(rank, suit, i)
    }

    fn obs(
        hand: Vec<Card>,
        sizes: Vec<usize>,
        unseen: Vec<Card>,
        on_table: Option<(u8, Combo)>,
    ) -> Observation {
        let n = sizes.len();
        let mut hand_sizes = sizes;
        hand_sizes[0] = hand.len();
        Observation {
            duplicate_rule: DuplicateRule::FirstDealtWins,
            seat: 0,
            player_count: u8::try_from(n).unwrap(),
            hand,
            in_trick: vec![true; n],
            ceilings: vec![PassCeilings::default(); n],
            hand_sizes,
            unseen,
            on_table,
        }
    }

    #[test]
    fn with_no_hidden_choice_the_report_is_one_exact_line() {
        // Seat 0 holds only an Ace and leads: it goes out first whatever the others hold.
        let o = obs(
            vec![c(Rank::Ace, Suit::Clubs, 0)],
            vec![1, 1, 1],
            vec![
                c(Rank::Three, Suit::Clubs, 1),
                c(Rank::Four, Suit::Clubs, 2),
            ],
            None,
        );
        let r = analyze(&o, &SolverConfig::default()).unwrap();
        assert_eq!(r.deals, 2);
        assert_eq!(r.moves.len(), 1);
        assert!((r.moves[0].mean_place - 0.0).abs() < 1e-12);
        assert!((r.score(&r.moves[0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn too_many_unseen_cards_are_refused() {
        let unseen: Vec<Card> = [
            Rank::Three,
            Rank::Four,
            Rank::Five,
            Rank::Six,
            Rank::Seven,
            Rank::Eight,
            Rank::Nine,
        ]
        .iter()
        .enumerate()
        .map(|(i, &r)| c(r, Suit::Clubs, u8::try_from(i).unwrap() + 1))
        .collect();
        let o = obs(
            vec![c(Rank::Ace, Suit::Clubs, 0)],
            vec![1, 4, 3],
            unseen,
            None,
        );
        let config = SolverConfig {
            max_unseen: 6,
            ..SolverConfig::default()
        };
        assert!(matches!(
            analyze(&o, &config),
            Err(SolveError::TooManyUnseen)
        ));
    }

    #[test]
    fn contradictory_ceilings_are_reported() {
        let three = Combo::new(vec![c(Rank::Three, Suit::Hearts, 9)]).unwrap();
        let mut o = obs(
            vec![c(Rank::Two, Suit::Clubs, 0)],
            vec![1, 1, 1],
            vec![
                c(Rank::Four, Suit::Clubs, 1),
                c(Rank::Eight, Suit::Hearts, 2),
            ],
            None,
        );
        o.ceilings = read_pass_ceilings(
            3,
            &[],
            &[(1, three, 0), (2, three, 0)],
            DuplicateRule::FirstDealtWins,
        );
        assert!(matches!(
            analyze(&o, &SolverConfig::default()),
            Err(SolveError::NoConsistentDeal)
        ));
    }

    #[test]
    fn a_seat_that_is_already_out_shifts_everyones_place() {
        // 4 seats; seat 3 is out. Seat 0 leads its only card, an Ace: it is first
        // among the three seats still playing, i.e. place index 1 overall.
        let mut o = obs(
            vec![c(Rank::Ace, Suit::Clubs, 0)],
            vec![1, 1, 1, 0],
            vec![
                c(Rank::Three, Suit::Clubs, 1),
                c(Rank::Four, Suit::Clubs, 2),
            ],
            None,
        );
        o.in_trick = vec![true, true, true, false];
        let r = analyze(&o, &SolverConfig::default()).unwrap();
        assert!((r.moves[0].mean_place - 1.0).abs() < 1e-12);
        assert_eq!(r.moves[0].best_place, 1);
    }

    #[test]
    fn the_best_move_is_never_worse_than_any_other_move() {
        use crate::hand_reading::PassTracker;
        use crate::match_runner::turn_context_for;
        use crate::strategy::ContextNeeds;
        use crate::test_support::for_each_state;
        let mut checked = 0;
        for_each_state(33, 2, |state, _| {
            let round = state.round;
            if round.pass_rule() != engine::PassRule::Final {
                return;
            }
            let seat = round.seat_to_move().unwrap();
            let mut tracker = PassTracker::new(usize::from(state.players), state.rule);
            let ctx = turn_context_for(
                round,
                seat,
                state.players,
                state.round_deck,
                &mut tracker,
                ContextNeeds::ALL,
            );
            let Some(o) = Observation::from_context(&ctx, state.rule) else {
                return;
            };
            let config = SolverConfig {
                max_unseen: 5,
                node_budget: 300_000,
            };
            let Ok(report) = analyze(&o, &config) else {
                return;
            };
            let best = report.best();
            assert!(report
                .moves
                .iter()
                .all(|m| best.mean_place <= m.mean_place + 1e-12));
            let legal = round.legal_moves();
            assert_eq!(
                report.moves.iter().map(|m| m.mv).collect::<Vec<Move>>(),
                legal,
                "the solver searches exactly the engine's legal moves"
            );
            checked += 1;
        });
        assert!(checked > 5, "only {checked} positions were solvable");
    }
}
