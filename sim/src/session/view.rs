//! What the human may see and do next: legal moves, model advice and the view.

use super::{
    role_score, turn_context_for, views, Advice, Card, Combo, DuplicateRule, FinalView, IdMap,
    Move, NeatStrategy, Phase, Playable, Round, SeatView, Session, SessionError, Strategy,
    TableView, View,
};

impl Session {
    /// The moves `Round::legal_moves` lists for the human (canonical
    /// weakest/strongest subsets); empty unless it is their turn.
    #[must_use]
    pub fn legal_moves(&self) -> Vec<Move> {
        if self.playing_human().is_err() {
            return Vec::new();
        }
        self.round
            .as_ref()
            .map(Round::legal_moves)
            .unwrap_or_default()
    }

    /// Every move `Round::legal_moves` lists for the human, ranked by what
    /// `advisor` (a trained model) thinks of it, best first. Only the
    /// human's own options and the model's scores are returned: the model
    /// sees exactly what the human sees.
    ///
    /// # Errors
    ///
    /// `SessionError::WrongPhase` / `NotYourTurn` unless the human is to move.
    ///
    /// # Panics
    ///
    /// Only if an internal invariant is broken (a bug), never for bad input.
    pub fn advice(&self, advisor: &NeatStrategy) -> Result<Vec<Advice>, SessionError> {
        self.playing_human()?;
        let round = self.round.as_ref().expect("checked by playing_human");
        let legal = round.legal_moves();
        let mut tracker = self.tracker.clone();
        let context = turn_context_for(
            round,
            self.human(),
            self.config.player_count,
            &self.round_deck,
            &mut tracker,
            advisor.needs(),
        );
        let mut scored = advisor.score_candidates(&legal, self.config.duplicate_rule, &context);
        // Best first; equal scores keep the engine's listing order.
        scored.sort_by(|a, b| b.raw_score.total_cmp(&a.raw_score));
        Ok(scored
            .into_iter()
            .map(|candidate| match candidate.candidate {
                Move::Pass => Advice {
                    cards: Vec::new(),
                    is_pass: true,
                    raw_score: candidate.raw_score,
                },
                Move::Play(combo) => Advice {
                    cards: views(&self.ids, combo.cards()),
                    is_pass: false,
                    raw_score: candidate.raw_score,
                },
            })
            .collect())
    }

    // -------------------------------------------------------------- view

    pub(super) fn final_view(&self) -> Option<FinalView> {
        (self.phase == Phase::MatchOver).then(|| {
            let roles = self.human_roles();
            #[allow(clippy::cast_precision_loss)] // a handful of rounds
            let score = roles
                .iter()
                .map(|&role| role_score(role, self.config.player_count))
                .sum::<f64>()
                / roles.len().max(1) as f64;
            FinalView { roles, score }
        })
    }

    /// Everything the human may know right now.
    #[must_use]
    pub fn view(&self) -> View {
        let human = self.human();
        let rule = self.config.duplicate_rule;
        let names = self.seat_names();
        let in_exchange = self.phase == Phase::Exchange;
        // During the exchange the previous round is over and the next has not
        // started: no table, no places, nobody to move.
        let round = if in_exchange {
            None
        } else {
            self.round.as_ref()
        };

        let mut hand: Vec<Card> = if in_exchange {
            self.pending
                .as_ref()
                .map(|p| p.hands[usize::from(human)].clone())
                .unwrap_or_default()
        } else {
            round.map(|r| r.hand(human).to_vec()).unwrap_or_default()
        };
        hand.sort_by(|a, b| a.compare(b, rule));

        let places: Vec<Option<usize>> = (0..self.config.player_count)
            .map(|seat| {
                round.and_then(|r| {
                    r.finishing_order()
                        .iter()
                        .position(|&s| s == seat)
                        .map(|p| p + 1)
                })
            })
            .collect();
        let seats = (0..self.config.player_count)
            .map(|seat| SeatView {
                seat,
                name: names[usize::from(seat)].clone(),
                is_human: seat == human,
                hand_size: if in_exchange {
                    self.pending
                        .as_ref()
                        .map_or(0, |p| p.hands[usize::from(seat)].len())
                } else {
                    round.map_or(0, |r| r.hand_size(seat))
                },
                active: !in_exchange && round.is_some_and(|r| r.is_active(seat)),
                role: self
                    .previous_roles
                    .as_ref()
                    .map(|roles| roles[usize::from(seat)]),
                place: places[usize::from(seat)],
                passed: self.phase == Phase::Playing && round.is_some_and(|r| r.has_passed(seat)),
            })
            .collect();

        let to_move = if self.phase == Phase::Playing {
            round.and_then(Round::seat_to_move)
        } else {
            None
        };
        let table = round.and_then(|r| {
            r.current_combo().and_then(|combo| {
                r.play_history().last().map(|(seat, _)| TableView {
                    seat: *seat,
                    cards: views(&self.ids, combo.cards()),
                })
            })
        });
        let must_lead = self.phase == Phase::Playing
            && to_move == Some(human)
            && round.is_some_and(|r| r.current_combo().is_none());
        let playable = if to_move == Some(human) {
            round.map_or_else(Vec::new, |r| {
                playable_ranks(&self.ids, &hand, r.current_combo(), rule)
            })
        } else {
            Vec::new()
        };
        let final_result = self.final_view();
        View {
            phase: self.phase,
            round: if matches!(self.phase, Phase::RoundOver | Phase::MatchOver) {
                self.finished_rounds
            } else {
                self.finished_rounds + 1
            },
            rounds: self.config.rounds,
            player_count: self.config.player_count,
            pass_rule: self.config.pass_rule,
            exchange_rule: self.config.exchange_rule,
            human_seat: human,
            hand: views(&self.ids, &hand),
            seats,
            table,
            to_move,
            must_lead,
            playable,
            give_count: self.pending.as_ref().map_or(0, |p| p.give_count),
            roles_history: self.roles_history.clone(),
            event_count: self.events.len(),
            final_result,
        }
    }
}

/// The ranks the human can play now. A rank offers a size when its
/// strongest subset of that size is playable (always on a lead; against a
/// table combo only the same size, and it must beat it).
pub(super) fn playable_ranks(
    ids: &IdMap,
    hand: &[Card],
    current: Option<&Combo>,
    rule: DuplicateRule,
) -> Vec<Playable> {
    engine::rank_groups(hand)
        .into_iter()
        .filter_map(|mut group| {
            group.sort_by(|a, b| a.compare(b, rule));
            let sizes: Vec<usize> = (1..=group.len())
                .filter(|&size| match current {
                    None => true,
                    Some(table) => {
                        size == table.size()
                            && Combo::new(group[group.len() - size..].to_vec())
                                .is_some_and(|strongest| strongest.beats(table, rule))
                    }
                })
                .collect();
            (!sizes.is_empty()).then(|| Playable {
                rank: group[0].rank as u8,
                card_ids: group
                    .iter()
                    .map(|c| ids.to_client[usize::from(c.deal_index)])
                    .collect(),
                sizes,
            })
        })
        .collect()
}

pub(super) fn explain(error: &engine::MoveError) -> String {
    use engine::MoveError;
    match error {
        MoveError::RoundAlreadyComplete => "the round is over".into(),
        MoveError::NotYourTurn { .. } => "it is not your turn".into(),
        MoveError::CannotPassOnLead => "you lead this trick, so you must play".into(),
        MoveError::CardNotInHand(_) => "you do not hold one of those cards".into(),
        MoveError::ComboDoesNotBeat => {
            "those cards do not beat the cards on the table (same number of cards, higher rank)"
                .into()
        }
    }
}
