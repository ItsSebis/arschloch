//! The match flow of a session: dealing a round, the exchange, the AI seats
//! playing until the human must act, and the end of a round.

use super::{
    assign_roles, deal, exchange_counts_for_player_count, exchange_with_rule, lowest_card_holder,
    roles_for_player_count, standard_deck, turn_context_for, views, Card, ExchangePair,
    ExchangeRule, GameEvent, Move, PassTracker, PendingExchange, Phase, Role, Round, SeatId,
    Session, SliceRandom,
};

impl Session {
    pub(super) fn start_round(&mut self) {
        let config = self.config.clone();
        let mut deck = standard_deck(config.deck_variant);
        deck.shuffle(&mut self.rng);
        for (index, card) in deck.iter_mut().enumerate() {
            card.deal_index = u8::try_from(index).expect("deck sizes (52/104) fit in u8");
        }
        let hands = deal(deck, config.player_count)
            .expect("standard_deck always yields enough cards for a supported player count");
        if let (Some(roles), Some(arschloch)) =
            (self.previous_roles.clone(), self.previous_arschloch)
        {
            let give_count = self.human_give_count(&roles);
            if give_count > 0 {
                self.pending = Some(PendingExchange {
                    hands,
                    roles,
                    give_count,
                });
                self.phase = Phase::Exchange;
            } else {
                self.finish_exchange(hands, &roles, None, arschloch);
            }
        } else {
            let leader = lowest_card_holder(&hands, config.duplicate_rule)
                .expect("a freshly dealt hand set is never empty");
            self.begin_play(hands, leader);
        }
    }

    /// How many cards the human must choose to give (0 when they are not
    /// a lower role, or their pair exchanges nothing).
    pub(super) fn human_give_count(&self, roles: &[Role]) -> usize {
        // Under the forced rule there is nothing to choose.
        if self.config.exchange_rule == ExchangeRule::Forced {
            return 0;
        }
        let player_count = self.config.player_count;
        let (Some(order), Some(counts)) = (
            roles_for_player_count(player_count),
            exchange_counts_for_player_count(player_count),
        ) else {
            return 0;
        };
        let mine = roles[usize::from(self.human())];
        counts
            .iter()
            .enumerate()
            .find(|&(i, &count)| count > 0 && order[order.len() - 1 - i] == mine)
            .map_or(0, |(_, &count)| usize::from(count))
    }

    /// Runs the exchange (the human's selection, when they give cards, is
    /// `human_selection`) and starts play with the Arschloch leading.
    pub(super) fn finish_exchange(
        &mut self,
        mut hands: Vec<Vec<Card>>,
        roles: &[Role],
        human_selection: Option<Vec<Card>>,
        leader: SeatId,
    ) {
        let rule = self.config.duplicate_rule;
        let human = usize::from(self.human());
        let before = hands[human].clone();
        let ai = &self.ai;
        let rng = &mut self.rng;
        let mut selection = human_selection.clone();
        exchange_with_rule(
            &mut hands,
            roles,
            rule,
            self.config.exchange_rule,
            |seat, hand, count, rule| match &ai[seat] {
                Some(ai) => ai.strategy.choose_exchange_cards(hand, count, rule, rng),
                None => selection
                    .take()
                    .expect("the human's selection was collected before the exchange"),
            },
        )
        .expect(
            "previous_roles always came from assign_roles for this player_count, and every \
             selection is validated before the exchange",
        );
        self.record_exchange(roles, &before, &hands[human], human_selection);
        self.begin_play(hands, leader);
    }

    pub(super) fn record_exchange(
        &mut self,
        roles: &[Role],
        before: &[Card],
        after: &[Card],
        human_selection: Option<Vec<Card>>,
    ) {
        let player_count = self.config.player_count;
        let order = roles_for_player_count(player_count).expect("supported size");
        let counts = exchange_counts_for_player_count(player_count).expect("supported size");
        let seat_of = |role: Role| -> SeatId {
            SeatId::try_from(
                roles
                    .iter()
                    .position(|&r| r == role)
                    .expect("every role is present"),
            )
            .expect("seat fits")
        };
        let pairs: Vec<ExchangePair> = counts
            .iter()
            .enumerate()
            .filter(|&(_, &count)| count > 0)
            .map(|(i, &count)| ExchangePair {
                from: seat_of(order[order.len() - 1 - i]),
                to: seat_of(order[i]),
                count: usize::from(count),
            })
            .collect();
        let involved = pairs
            .iter()
            .any(|p| p.from == self.human() || p.to == self.human());
        self.events.push(GameEvent::Exchange { pairs });
        if involved {
            let gave = match human_selection {
                Some(selection) => views(&self.ids, &selection),
                // No selection: the exchange was forced (a lower role gives its
                // highest cards) or the human was the higher role (hands back
                // its lowest): in both cases whatever left their hand.
                None => views(
                    &self.ids,
                    &before
                        .iter()
                        .copied()
                        .filter(|c| !after.contains(c))
                        .collect::<Vec<_>>(),
                ),
            };
            let received = views(
                &self.ids,
                &after
                    .iter()
                    .copied()
                    .filter(|c| !before.contains(c))
                    .collect::<Vec<_>>(),
            );
            self.events
                .push(GameEvent::ExchangeYours { gave, received });
        }
    }

    pub(super) fn begin_play(&mut self, hands: Vec<Vec<Card>>, leader: SeatId) {
        self.round_deck = hands.iter().flatten().copied().collect();
        let hand_sizes = hands.iter().map(Vec::len).collect();
        self.round = Some(
            Round::with_pass_rule(
                hands,
                self.config.duplicate_rule,
                self.config.pass_rule,
                leader,
            )
            .expect("player_count/leader are always valid for a supported table size"),
        );
        self.tracker = PassTracker::new(
            usize::from(self.config.player_count),
            self.config.duplicate_rule,
        );
        self.events.push(GameEvent::RoundStart {
            round: self.finished_rounds + 1,
            roles: self.previous_roles.clone(),
            leader,
            hand_sizes,
        });
        self.phase = Phase::Playing;
        self.advance();
    }

    // ------------------------------------------------------------- moves

    /// Plays the AI seats until the human must act or the round ends.
    pub(super) fn advance(&mut self) {
        loop {
            let (seat, complete) = {
                let round = self.round.as_ref().expect("a round is in progress");
                (round.seat_to_move(), round.is_complete())
            };
            if complete {
                self.end_round();
                return;
            }
            let seat = seat.expect("an incomplete round has a seat to move");
            if seat == self.human() {
                self.phase = Phase::Playing;
                return;
            }
            let chosen = {
                let round = self.round.as_ref().expect("a round is in progress");
                let legal = round.legal_moves();
                let strategy = &self.ai[usize::from(seat)]
                    .as_ref()
                    .expect("only the human seat has no strategy")
                    .strategy;
                let context = turn_context_for(
                    round,
                    seat,
                    self.config.player_count,
                    &self.round_deck,
                    &mut self.tracker,
                    strategy.needs(),
                );
                strategy.choose_play(&legal, self.config.duplicate_rule, &context, &mut self.rng)
            };
            self.apply(seat, &chosen)
                .expect("strategies only choose from the moves engine just reported as legal");
        }
    }

    /// Submits `mv` for `seat` and logs it.
    pub(super) fn apply(&mut self, seat: SeatId, mv: &Move) -> Result<(), engine::MoveError> {
        let round = self.round.as_mut().expect("a round is in progress");
        let had_combo = round.current_combo().is_some();
        let finished_before = round.finishing_order().len();
        round.submit_move(seat, *mv)?;
        match mv {
            Move::Pass => self.events.push(GameEvent::Pass { seat }),
            Move::Play(combo) => {
                let hand_left = round.hand_size(seat);
                self.events.push(GameEvent::Play {
                    seat,
                    cards: views(&self.ids, combo.cards()),
                    hand_left,
                });
            }
        }
        if round.finishing_order().len() > finished_before {
            self.events.push(GameEvent::Finished {
                seat,
                place: round.finishing_order().len(),
            });
        }
        if had_combo && round.current_combo().is_none() {
            if let Some(leader) = round.seat_to_move() {
                self.events.push(GameEvent::TrickEnd { leader });
            }
        }
        Ok(())
    }

    pub(super) fn end_round(&mut self) {
        let round = self.round.as_ref().expect("a round is in progress");
        let finishing_order = round.finishing_order().to_vec();
        let roles = assign_roles(&finishing_order, self.config.player_count)
            .expect("finishing_order is always a valid permutation for a supported player count");
        self.previous_arschloch = finishing_order.last().copied();
        self.roles_history.push(roles.clone());
        self.events.push(GameEvent::RoundEnd {
            finishing_order,
            roles: roles.clone(),
        });
        self.previous_roles = Some(roles);
        self.finished_rounds += 1;
        if self.finished_rounds >= self.config.rounds {
            self.events.push(GameEvent::MatchEnd);
            self.phase = Phase::MatchOver;
        } else {
            self.phase = Phase::RoundOver;
        }
    }
}
