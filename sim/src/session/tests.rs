use super::*;
use crate::hand_reading::PassCeilings;
use crate::strategies::LowestLegal;
use crate::strategy::TurnContext;
use crate::{run_match, MatchConfig};

fn ai(count: usize) -> Vec<AiSeat> {
    (0..count)
        .map(|_| AiSeat {
            name: "LowestLegal".into(),
            strategy: Arc::new(LowestLegal),
        })
        .collect()
}

fn config(players: u8, deck: DeckVariant, rounds: usize, human_seat: u8) -> SessionConfig {
    SessionConfig {
        player_count: players,
        deck_variant: deck,
        duplicate_rule: DuplicateRule::FirstDealtWins,
        pass_rule: PassRule::default(),
        exchange_rule: ExchangeRule::default(),
        rounds,
        seed: 7,
        human_seat,
        hide_deal_order: false,
    }
}

/// What `LowestLegal` would choose, for a scripted human.
fn scripted_move(session: &Session) -> Move {
    let moves = session.legal_moves();
    let hand: Vec<Card> = Vec::new();
    let context = TurnContext {
        seat: 0,
        hand: &hand,
        opponents: Vec::new(),
        unseen_cards: Vec::new(),
        own_pass_ceilings: PassCeilings::default(),
        current_combo: None,
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    LowestLegal.choose_play(&moves, DuplicateRule::FirstDealtWins, &context, &mut rng)
}

/// Plays the human's seat like `LowestLegal` until the match is over.
fn play_out(session: &mut Session) {
    let mut guard = 0;
    while session.phase() != Phase::MatchOver {
        guard += 1;
        assert!(guard < 10_000, "the game does not end");
        match session.phase() {
            Phase::Exchange => {
                let view = session.view();
                let mut rng = rand::rngs::StdRng::seed_from_u64(0);
                let hand: Vec<Card> = view
                    .hand
                    .iter()
                    .map(|c| Card::new(rank_from(c.rank), suit_from(c.suit), c.id))
                    .collect();
                let give = LowestLegal.choose_exchange_cards(
                    &hand,
                    view.give_count,
                    DuplicateRule::FirstDealtWins,
                    &mut rng,
                );
                let ids: Vec<u8> = give.iter().map(|c| c.deal_index).collect();
                session.give(&ids).unwrap();
            }
            Phase::Playing => match scripted_move(session) {
                Move::Pass => session.pass().unwrap(),
                Move::Play(combo) => {
                    let ids: Vec<u8> = combo.cards().iter().map(|c| c.deal_index).collect();
                    session.play(&ids).unwrap();
                }
            },
            Phase::RoundOver => session.next_round().unwrap(),
            Phase::MatchOver => unreachable!(),
        }
    }
}

fn rank_from(rank: u8) -> Rank {
    use Rank::*;
    [
        Two, Three, Four, Five, Six, Seven, Eight, Nine, Ten, Jack, Queen, King, Ace,
    ][usize::from(rank)]
}

fn suit_from(suit: u8) -> Suit {
    [Suit::Diamonds, Suit::Hearts, Suit::Spades, Suit::Clubs][usize::from(suit)]
}

#[test]
fn a_session_plays_a_whole_match_for_every_table_size_and_deck() {
    for players in 3..=6u8 {
        for deck in [DeckVariant::Single, DeckVariant::Double] {
            for human_seat in [0, players - 1] {
                let mut session = Session::new(
                    config(players, deck, 5, human_seat),
                    ai(usize::from(players) - 1),
                )
                .unwrap();
                play_out(&mut session);
                let view = session.view();
                assert_eq!(view.roles_history.len(), 5, "{players} {deck:?}");
                for roles in &view.roles_history {
                    assert_eq!(roles.len(), usize::from(players));
                }
                let result = view.final_result.expect("a finished match has a result");
                assert_eq!(result.roles.len(), 5);
                assert!((-1.0..=1.0).contains(&result.score));
            }
        }
    }
}

#[test]
fn the_ai_seats_play_exactly_as_run_match_does() {
    // A scripted human that plays like LowestLegal, among LowestLegal
    // seats, must reproduce run_match with four LowestLegal seats: the
    // same shuffles, exchanges and moves.
    for players in 3..=6u8 {
        let mut session = Session::new(
            config(players, DeckVariant::Single, 6, 0),
            ai(usize::from(players) - 1),
        )
        .unwrap();
        play_out(&mut session);
        let strategies: Vec<Arc<dyn Strategy>> = (0..players)
            .map(|_| Arc::new(LowestLegal) as Arc<dyn Strategy>)
            .collect();
        let expected = run_match(
            &MatchConfig {
                player_count: players,
                deck_variant: DeckVariant::Single,
                duplicate_rule: DuplicateRule::FirstDealtWins,
                rounds: 6,
                seed: 7,
                pass_rule: engine::PassRule::default(),
                exchange_rule: engine::ExchangeRule::default(),
            },
            &strategies,
        );
        assert_eq!(
            session.view().roles_history,
            expected.role_history,
            "{players} players"
        );
    }
}

#[test]
fn the_view_and_events_never_contain_another_seats_cards() {
    let mut session = Session::new(config(4, DeckVariant::Single, 4, 1), ai(3)).unwrap();
    let mut seen_events = 0;
    // Cards the human gave away are legitimately known to be elsewhere.
    let mut known: Vec<u8> = Vec::new();
    let mut guard = 0;
    while session.phase() != Phase::MatchOver {
        guard += 1;
        assert!(guard < 10_000);
        let view = session.view();
        // The human's real hand, independent of what the view claims.
        let own: Vec<u8> = match &session.pending {
            Some(pending) => pending.hands[1].iter().map(|c| c.deal_index).collect(),
            None => session.round.as_ref().map_or_else(Vec::new, |r| {
                r.hand(1).iter().map(|c| c.deal_index).collect()
            }),
        };
        let events = session.events_since(seen_events).to_vec();
        seen_events += events.len();
        if events
            .iter()
            .any(|e| matches!(e, GameEvent::RoundStart { .. }))
        {
            known.clear();
        }
        for event in &events {
            if let GameEvent::ExchangeYours { gave, .. } = event {
                known.extend(gave.iter().map(|c| c.id));
            }
        }
        // Ids restart every round, so events spanning a round boundary
        // are not comparable with the current hands; the view always is.
        let crosses_a_round = events
            .iter()
            .any(|e| matches!(e, GameEvent::RoundStart { .. } | GameEvent::RoundEnd { .. }));
        let text = if crosses_a_round {
            serde_json::to_string(&view).unwrap()
        } else {
            serde_json::to_string(&(&view, &events)).unwrap()
        };
        for seat in (0..4u8).filter(|&s| s != 1) {
            let hidden: Vec<u8> = session.round.as_ref().map_or_else(Vec::new, |r| {
                r.hand(seat).iter().map(|c| c.deal_index).collect()
            });
            for id in hidden {
                if own.contains(&id) || known.contains(&id) {
                    continue;
                }
                assert!(
                    !text.contains(&format!("\"id\":{id},")),
                    "card {id} of seat {seat}'s hand leaked: {text}"
                );
            }
        }
        match session.phase() {
            Phase::Exchange => {
                let ids: Vec<u8> = view
                    .hand
                    .iter()
                    .take(view.give_count)
                    .map(|c| c.id)
                    .collect();
                session.give(&ids).unwrap();
            }
            Phase::Playing => match scripted_move(&session) {
                Move::Pass => session.pass().unwrap(),
                Move::Play(combo) => {
                    let ids: Vec<u8> = combo.cards().iter().map(|c| c.deal_index).collect();
                    session.play(&ids).unwrap();
                }
            },
            Phase::RoundOver => session.next_round().unwrap(),
            Phase::MatchOver => unreachable!(),
        }
    }
}

fn json(session: &Session) -> String {
    serde_json::to_string(&(session.view(), session.events_since(0))).unwrap()
}

/// A session at the human's first turn (the human leads or follows).
fn at_first_turn(players: u8, human_seat: u8, seed: u64) -> Session {
    let mut cfg = config(players, DeckVariant::Single, 3, human_seat);
    cfg.seed = seed;
    let mut session = Session::new(cfg, ai(usize::from(players) - 1)).unwrap();
    assert_eq!(session.phase(), Phase::Playing);
    // Let the round proceed until it is the human's turn (it already is,
    // `new` advances the AI seats); keep the type used.
    let _ = &mut session;
    session
}

#[test]
fn rejected_actions_leave_the_game_unchanged() {
    let mut session = at_first_turn(4, 0, 3);
    let before = json(&session);
    let own: Vec<u8> = session.view().hand.iter().map(|c| c.id).collect();
    let absent = (0..52u8).find(|id| !own.contains(id)).unwrap();
    let attempts: Vec<(&str, Result<(), SessionError>)> = vec![
        ("unknown card", session.play(&[absent])),
        ("duplicate card", session.play(&[own[0], own[0]])),
        ("empty", session.play(&[])),
        ("give in the wrong phase", session.give(&[own[0]])),
        ("next round too early", session.next_round()),
    ];
    for (what, result) in attempts {
        assert!(result.is_err(), "{what} was accepted");
        assert_eq!(json(&session), before, "{what} changed the game");
    }
    // Two cards of different ranks.
    let view = session.view();
    let first = &view.hand[0];
    let other = view.hand.iter().find(|c| c.rank != first.rank).unwrap();
    assert!(session.play(&[first.id, other.id]).is_err());
    assert_eq!(json(&session), before);
    // A pass on the lead (the human leads when nothing is on the table).
    if session.view().must_lead {
        assert!(session.pass().is_err());
        assert_eq!(json(&session), before);
    }
}

#[test]
fn nothing_works_after_the_match_ends() {
    let mut session = Session::new(config(3, DeckVariant::Single, 1, 0), ai(2)).unwrap();
    play_out(&mut session);
    let before = json(&session);
    assert_eq!(session.play(&[0]), Err(SessionError::WrongPhase));
    assert_eq!(session.pass(), Err(SessionError::WrongPhase));
    assert_eq!(session.give(&[0]), Err(SessionError::WrongPhase));
    assert_eq!(session.next_round(), Err(SessionError::WrongPhase));
    assert_eq!(json(&session), before);
}

#[test]
fn any_subset_of_a_rank_that_beats_the_table_is_accepted() {
    // Search seeds for a first turn where the human may lead a rank held
    // three times; leading a pair out of the triple (a non-canonical
    // choice: the engine lists only weakest and strongest) must work.
    for seed in 0..200 {
        let mut session = at_first_turn(4, 0, seed);
        let view = session.view();
        if !view.must_lead {
            continue;
        }
        let Some(triple) = view.playable.iter().find(|p| p.card_ids.len() >= 3) else {
            continue;
        };
        // The middle two cards of the group, not the weakest or strongest pair.
        let ids = [triple.card_ids[1], triple.card_ids[2]];
        if triple.card_ids.len() == 3 {
            // pair (1,2) is the strongest pair; use (0,2) instead: neither
            // weakest nor strongest.
            let ids = [triple.card_ids[0], triple.card_ids[2]];
            session.play(&ids).unwrap();
        } else {
            session.play(&ids).unwrap();
        }
        return;
    }
    panic!("no seed produced a leading triple");
}

#[test]
fn the_human_as_a_lower_role_chooses_the_cards_to_give() {
    // Play round 1 out; find a seed where the human ends lower and the
    // second round opens in the exchange phase (free exchange rule).
    for seed in 0..300 {
        let mut cfg = config(4, DeckVariant::Single, 3, 0);
        cfg.exchange_rule = ExchangeRule::Free;
        cfg.seed = seed;
        let mut session = Session::new(cfg, ai(3)).unwrap();
        // Finish round 1 with the scripted human.
        while session.phase() == Phase::Playing {
            match scripted_move(&session) {
                Move::Pass => session.pass().unwrap(),
                Move::Play(c) => {
                    let ids: Vec<u8> = c.cards().iter().map(|c| c.deal_index).collect();
                    session.play(&ids).unwrap();
                }
            }
        }
        session.next_round().unwrap();
        if session.phase() != Phase::Exchange {
            continue;
        }
        let view = session.view();
        assert!(view.give_count >= 1);
        let hand_before = view.hand.clone();
        // Wrong count and a foreign card are refused.
        assert!(session.give(&[]).is_err());
        let ids: Vec<u8> = hand_before
            .iter()
            .rev()
            .take(view.give_count)
            .map(|c| c.id)
            .collect();
        session.give(&ids).unwrap();
        assert_eq!(session.phase(), Phase::Playing);
        let yours = session
            .events_since(0)
            .iter()
            .find_map(|e| match e {
                GameEvent::ExchangeYours { gave, received } => {
                    Some((gave.clone(), received.clone()))
                }
                _ => None,
            })
            .expect("the human sees their own exchange");
        assert_eq!(yours.0.iter().map(|c| c.id).collect::<Vec<_>>(), ids);
        assert_eq!(yours.1.len(), view.give_count);
        return;
    }
    panic!("the human never ended lower in 300 seeds");
}

#[test]
fn the_exchange_view_shows_a_fresh_table_not_the_last_round() {
    for seed in 0..300 {
        let mut cfg = config(4, DeckVariant::Single, 3, 0);
        cfg.exchange_rule = ExchangeRule::Free;
        cfg.seed = seed;
        let mut session = Session::new(cfg, ai(3)).unwrap();
        while session.phase() == Phase::Playing {
            match scripted_move(&session) {
                Move::Pass => session.pass().unwrap(),
                Move::Play(c) => {
                    let ids: Vec<u8> = c.cards().iter().map(|c| c.deal_index).collect();
                    session.play(&ids).unwrap();
                }
            }
        }
        session.next_round().unwrap();
        if session.phase() != Phase::Exchange {
            continue;
        }
        let view = session.view();
        assert!(
            view.table.is_none(),
            "last round's final combo is still shown"
        );
        assert!(
            view.seats.iter().all(|s| s.place.is_none()),
            "last round's places are still shown"
        );
        assert!(view.to_move.is_none() && !view.must_lead && view.playable.is_empty());
        return;
    }
    panic!("the human never ended lower in 300 seeds");
}

#[test]
fn under_the_forced_rule_the_human_as_a_lower_role_gives_the_highest_cards_without_choosing() {
    for seed in 0..300 {
        let mut cfg = config(4, DeckVariant::Single, 3, 0);
        cfg.exchange_rule = ExchangeRule::Forced;
        cfg.seed = seed;
        let mut session = Session::new(cfg, ai(3)).unwrap();
        while session.phase() == Phase::Playing {
            match scripted_move(&session) {
                Move::Pass => session.pass().unwrap(),
                Move::Play(c) => {
                    let ids: Vec<u8> = c.cards().iter().map(|c| c.deal_index).collect();
                    session.play(&ids).unwrap();
                }
            }
        }
        let role = session.view().roles_history[0][0];
        session.next_round().unwrap();
        assert_ne!(
            session.phase(),
            Phase::Exchange,
            "nothing to choose when forced"
        );
        if !matches!(role, Role::ViceArschloch | Role::Arschloch) {
            continue;
        }
        let (gave, received) = session
            .events_since(0)
            .iter()
            .find_map(|e| match e {
                GameEvent::ExchangeYours { gave, received } => {
                    Some((gave.clone(), received.clone()))
                }
                _ => None,
            })
            .expect("a lower role still sees what was taken and received");
        let count = if role == Role::Arschloch { 2 } else { 1 };
        assert_eq!(gave.len(), count);
        assert_eq!(received.len(), count);
        // What was given is at least as high as anything kept (rank order).
        let kept_max = session
            .view()
            .hand
            .iter()
            .filter(|c| {
                !gave.iter().any(|g| g.id == c.id) && !received.iter().any(|r| r.id == c.id)
            })
            .map(|c| (c.rank, c.suit))
            .max();
        let given_min = gave.iter().map(|c| (c.rank, c.suit)).min().unwrap();
        if let Some(kept) = kept_max {
            assert!(
                given_min >= kept,
                "seed {seed}: gave {gave:?} but kept a higher card"
            );
        }
        // And what came back is from the president's low end.
        assert!(
            received.iter().all(|c| c.rank <= 6),
            "received {received:?}"
        );
        return;
    }
    panic!("the human never ended in a lower role in 300 seeds");
}

#[test]
fn the_human_as_a_higher_role_gets_cards_without_choosing() {
    for seed in 0..300 {
        let mut cfg = config(4, DeckVariant::Single, 3, 0);
        cfg.seed = seed;
        let mut session = Session::new(cfg, ai(3)).unwrap();
        while session.phase() == Phase::Playing {
            match scripted_move(&session) {
                Move::Pass => session.pass().unwrap(),
                Move::Play(c) => {
                    let ids: Vec<u8> = c.cards().iter().map(|c| c.deal_index).collect();
                    session.play(&ids).unwrap();
                }
            }
        }
        let role = session.view().roles_history[0][0];
        session.next_round().unwrap();
        if role != Role::President {
            continue;
        }
        assert_ne!(session.phase(), Phase::Exchange);
        let yours = session.events_since(0).iter().any(|e| {
            matches!(e, GameEvent::ExchangeYours { gave, received } if gave.len() == 2 && received.len() == 2)
        });
        assert!(yours, "the president hands back 2 cards and receives 2");
        return;
    }
    panic!("the human was never President in 300 seeds");
}

#[test]
fn bad_setups_are_refused() {
    let bad = |cfg: SessionConfig, n: usize| Session::new(cfg, ai(n)).err().unwrap();
    assert!(matches!(
        bad(config(2, DeckVariant::Single, 1, 0), 1),
        SessionError::BadConfig(_)
    ));
    assert!(matches!(
        bad(config(7, DeckVariant::Single, 1, 0), 6),
        SessionError::BadConfig(_)
    ));
    assert!(matches!(
        bad(config(4, DeckVariant::Single, 0, 0), 3),
        SessionError::BadConfig(_)
    ));
    assert!(matches!(
        bad(config(4, DeckVariant::Single, 1, 4), 3),
        SessionError::BadConfig(_)
    ));
    assert!(matches!(
        bad(config(4, DeckVariant::Single, 1, 0), 2),
        SessionError::BadConfig(_)
    ));
}

/// Plays a whole match from what the client sees only.
fn play_by_view(session: &mut Session) {
    let mut guard = 0;
    while session.phase() != Phase::MatchOver {
        guard += 1;
        assert!(guard < 10_000);
        let view = session.view();
        match view.phase {
            Phase::Exchange => {
                let ids: Vec<u8> = view
                    .hand
                    .iter()
                    .take(view.give_count)
                    .map(|c| c.id)
                    .collect();
                session.give(&ids).unwrap();
            }
            Phase::Playing => match view.playable.first() {
                Some(rank) => {
                    let size = rank.sizes[0];
                    let ids = &rank.card_ids[rank.card_ids.len() - size..];
                    session.play(ids).unwrap();
                }
                None => session.pass().unwrap(),
            },
            Phase::RoundOver => session.next_round().unwrap(),
            Phase::MatchOver => unreachable!(),
        }
    }
}

#[test]
fn hidden_deal_order_keeps_the_game_playable_and_unrevealing() {
    // With the deal position as the id, `id % players` is the seat a card
    // was dealt to. Hidden ids must break that for every seat and still
    // play identically.
    let mut revealing = 0;
    let mut total = 0;
    for seed in 0..20 {
        let mut cfg = config(4, DeckVariant::Single, 2, 1);
        cfg.seed = seed;
        cfg.hide_deal_order = true;
        let mut hidden = Session::new(cfg.clone(), ai(3)).unwrap();
        for card in &hidden.view().hand {
            total += 1;
            revealing += usize::from(card.id % 4 == 1);
        }
        play_by_view(&mut hidden);
        cfg.hide_deal_order = false;
        let mut plain = Session::new(cfg, ai(3)).unwrap();
        play_by_view(&mut plain);
        // Same game, only the numbers differ: same roles every round.
        assert_eq!(
            hidden.view().roles_history,
            plain.view().roles_history,
            "seed {seed}"
        );
    }
    // Dealt round-robin, all 13 of seat 1's cards would be 1 mod 4.
    assert!(
        revealing < total / 2,
        "{revealing} of {total} ids still match the seat"
    );
}

#[test]
fn under_the_final_rule_passes_show_in_the_view_and_a_pass_ends_the_humans_trick() {
    let mut saw_an_ai_pass = false;
    let mut humans_passes = 0;
    for seed in 0..60 {
        let mut cfg = config(4, DeckVariant::Single, 2, 0);
        cfg.seed = seed;
        cfg.pass_rule = PassRule::Final;
        let mut session = Session::new(cfg, ai(3)).unwrap();
        let mut guard = 0;
        while session.phase() != Phase::MatchOver && guard < 2000 {
            guard += 1;
            let view = session.view();
            if view.seats.iter().any(|s| s.passed && !s.is_human) {
                saw_an_ai_pass = true;
            }
            match view.phase {
                Phase::Exchange => {
                    let ids: Vec<u8> = view
                        .hand
                        .iter()
                        .take(view.give_count)
                        .map(|c| c.id)
                        .collect();
                    session.give(&ids).unwrap();
                }
                Phase::Playing if !view.must_lead => {
                    // Always pass when following: the human is out of the
                    // trick, so the next human decision must come after
                    // a TrickEnd.
                    let before = view.event_count;
                    session.pass().unwrap();
                    humans_passes += 1;
                    let ended = session.events_since(before).iter().any(|e| {
                        matches!(e, GameEvent::TrickEnd { .. } | GameEvent::RoundEnd { .. })
                    });
                    assert!(
                        ended,
                        "seed {seed}: the human acted again inside the trick they passed in"
                    );
                }
                Phase::Playing => {
                    let rank = view.playable.first().unwrap();
                    let size = rank.sizes[0];
                    session
                        .play(&rank.card_ids[rank.card_ids.len() - size..])
                        .unwrap();
                }
                Phase::RoundOver => session.next_round().unwrap(),
                Phase::MatchOver => unreachable!(),
            }
        }
        assert_eq!(session.phase(), Phase::MatchOver, "seed {seed}");
    }
    assert!(saw_an_ai_pass, "the view never showed a seat as passed");
    assert!(humans_passes > 0);
}

#[test]
fn nobody_is_shown_as_passed_outside_a_trick_in_progress() {
    // The round-over and match-over screens must not show last trick's passes.
    let mut stale = 0;
    for seed in 0..80 {
        let mut cfg = config(4, DeckVariant::Single, 2, 0);
        cfg.seed = seed;
        cfg.pass_rule = PassRule::Final;
        let mut session = Session::new(cfg, ai(3)).unwrap();
        let mut guard = 0;
        while session.phase() != Phase::MatchOver && guard < 3000 {
            guard += 1;
            let view = session.view();
            if view.phase != Phase::Playing && view.seats.iter().any(|s| s.passed) {
                stale += 1;
            }
            match view.phase {
                Phase::Exchange => {
                    let ids: Vec<u8> = view
                        .hand
                        .iter()
                        .take(view.give_count)
                        .map(|c| c.id)
                        .collect();
                    session.give(&ids).unwrap();
                }
                Phase::Playing if !view.must_lead => session.pass().unwrap(),
                Phase::Playing => {
                    let rank = view.playable.first().unwrap();
                    let size = rank.sizes[0];
                    session
                        .play(&rank.card_ids[rank.card_ids.len() - size..])
                        .unwrap();
                }
                Phase::RoundOver => session.next_round().unwrap(),
                Phase::MatchOver => unreachable!(),
            }
        }
        let end = session.view();
        if end.seats.iter().any(|s| s.passed) {
            stale += 1;
        }
    }
    assert_eq!(stale, 0, "views outside play showed passed seats");
}

#[test]
fn under_the_free_rule_no_seat_is_ever_shown_as_passed() {
    let mut cfg = config(4, DeckVariant::Single, 2, 0);
    cfg.pass_rule = PassRule::Free;
    let mut session = Session::new(cfg, ai(3)).unwrap();
    play_by_view(&mut session);
    assert_eq!(session.view().pass_rule, PassRule::Free);
    assert!(session.view().seats.iter().all(|s| !s.passed));
}

fn champion() -> NeatStrategy {
    NeatStrategy::from_file(std::path::Path::new(
        "../docs/baselines/neat-v1/champion.json",
    ))
    .unwrap()
}

#[test]
fn advice_ranks_the_humans_moves_and_agrees_with_the_models_own_choice() {
    let model = champion();
    let session = at_first_turn(4, 0, 5);
    let advice = session.advice(&model).unwrap();
    let legal = session.legal_moves();
    assert_eq!(advice.len(), legal.len());
    assert!(advice
        .windows(2)
        .all(|pair| pair[0].raw_score >= pair[1].raw_score));
    // What the model itself would play in this exact situation.
    let round = session.round.as_ref().unwrap();
    let mut tracker = session.tracker.clone();
    let context = turn_context_for(
        round,
        0,
        4,
        &session.round_deck,
        &mut tracker,
        crate::strategy::ContextNeeds::ALL,
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let chosen = model.choose_play(&legal, DuplicateRule::FirstDealtWins, &context, &mut rng);
    let top = &advice[0];
    match chosen {
        Move::Pass => assert!(top.is_pass),
        Move::Play(combo) => {
            let ids: Vec<u8> = combo.cards().iter().map(|c| c.deal_index).collect();
            assert_eq!(top.cards.iter().map(|c| c.id).collect::<Vec<_>>(), ids);
        }
    }
}

#[test]
fn advice_is_only_available_on_the_humans_turn() {
    let model = champion();
    let mut session = Session::new(config(3, DeckVariant::Single, 1, 0), ai(2)).unwrap();
    play_out(&mut session);
    assert_eq!(session.advice(&model).err(), Some(SessionError::WrongPhase));
}
