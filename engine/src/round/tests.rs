use super::*;
use crate::card::{Rank, Suit};

fn card(rank: Rank, suit: Suit) -> Card {
    Card::new(rank, suit, 0)
}

fn combo(cards: Vec<Card>) -> Combo {
    Combo::new(cards).unwrap()
}

#[test]
fn new_round_rejects_unsupported_player_counts() {
    assert!(Round::new(
        vec![vec![card(Rank::Two, Suit::Clubs)]; 2],
        DuplicateRule::FirstDealtWins,
        0
    )
    .is_none());
    assert!(Round::new(
        vec![vec![card(Rank::Two, Suit::Clubs)]; 7],
        DuplicateRule::FirstDealtWins,
        0
    )
    .is_none());
}

#[test]
fn new_round_rejects_out_of_range_leader() {
    let hands = vec![vec![card(Rank::Two, Suit::Clubs)]; 3];
    assert!(Round::new(hands, DuplicateRule::FirstDealtWins, 3).is_none());
}

#[test]
fn new_round_rejects_an_empty_starting_hand() {
    let hands = vec![
        vec![card(Rank::Two, Suit::Clubs)],
        vec![],
        vec![card(Rank::Three, Suit::Clubs)],
    ];
    assert!(Round::new(hands, DuplicateRule::FirstDealtWins, 0).is_none());
}

#[test]
fn leader_must_play_and_cannot_pass() {
    let hands = vec![
        vec![card(Rank::Two, Suit::Clubs)],
        vec![card(Rank::Three, Suit::Clubs)],
        vec![card(Rank::Four, Suit::Clubs)],
    ];
    let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
    assert_eq!(
        round.submit_move(0, Move::Pass),
        Err(MoveError::CannotPassOnLead)
    );
}

#[test]
fn rejects_a_move_from_the_wrong_seat() {
    let hands = vec![
        vec![card(Rank::Two, Suit::Clubs)],
        vec![card(Rank::Three, Suit::Clubs)],
        vec![card(Rank::Four, Suit::Clubs)],
    ];
    let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
    let play = Move::Play(combo(vec![card(Rank::Three, Suit::Clubs)]));
    assert_eq!(
        round.submit_move(1, play),
        Err(MoveError::NotYourTurn { expected: 0 })
    );
}

#[test]
fn rejects_a_combo_with_a_card_not_in_hand() {
    let hands = vec![
        vec![card(Rank::Two, Suit::Clubs)],
        vec![card(Rank::Three, Suit::Clubs)],
        vec![card(Rank::Four, Suit::Clubs)],
    ];
    let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
    let play = Move::Play(combo(vec![card(Rank::Nine, Suit::Clubs)]));
    assert_eq!(
        round.submit_move(0, play),
        Err(MoveError::CardNotInHand(card(Rank::Nine, Suit::Clubs)))
    );
}

#[test]
fn rejects_a_combo_that_does_not_beat_the_current_combo() {
    let hands = vec![
        vec![card(Rank::Eight, Suit::Clubs)],
        vec![card(Rank::Seven, Suit::Clubs)],
        vec![card(Rank::Nine, Suit::Clubs)],
    ];
    let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
    round
        .submit_move(0, Move::Play(combo(vec![card(Rank::Eight, Suit::Clubs)])))
        .unwrap();
    let play = Move::Play(combo(vec![card(Rank::Seven, Suit::Clubs)]));
    assert_eq!(round.submit_move(1, play), Err(MoveError::ComboDoesNotBeat));
}

#[test]
fn everyone_passing_returns_the_lead_to_the_original_leader() {
    let hands = vec![
        vec![card(Rank::Eight, Suit::Clubs), card(Rank::Ten, Suit::Clubs)],
        vec![card(Rank::Seven, Suit::Clubs), card(Rank::Six, Suit::Clubs)],
        vec![card(Rank::Nine, Suit::Clubs), card(Rank::Five, Suit::Clubs)],
    ];
    let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
    round
        .submit_move(0, Move::Play(combo(vec![card(Rank::Eight, Suit::Clubs)])))
        .unwrap();
    round.submit_move(1, Move::Pass).unwrap();
    round.submit_move(2, Move::Pass).unwrap();
    assert_eq!(round.seat_to_move(), Some(0));
    assert_eq!(round.current_combo(), None);
    assert_eq!(round.current_trick_leader(), 0);
}

#[test]
fn a_seat_emptying_its_hand_mid_round_is_recorded_and_skipped() {
    // 3 players, each starts with 1 card so seat 0 empties
    // immediately on its lead.
    let hands = vec![
        vec![card(Rank::Two, Suit::Clubs)],
        vec![card(Rank::Three, Suit::Clubs)],
        vec![card(Rank::Four, Suit::Clubs)],
    ];
    let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
    round
        .submit_move(0, Move::Play(combo(vec![card(Rank::Two, Suit::Clubs)])))
        .unwrap();
    assert_eq!(round.finishing_order(), &[0]);
    assert!(!round.is_active(0));
    assert_eq!(round.seat_to_move(), Some(1));

    round
        .submit_move(1, Move::Play(combo(vec![card(Rank::Three, Suit::Clubs)])))
        .unwrap();
    // Only seat 2 remains active: the round auto-completes.
    assert!(round.is_complete());
    assert_eq!(round.finishing_order(), &[0, 1, 2]);
    assert_eq!(round.seat_to_move(), None);
}

#[test]
fn rejects_a_combo_that_lists_the_same_physical_card_twice() {
    let original_hand = vec![
        card(Rank::Seven, Suit::Clubs),
        card(Rank::Seven, Suit::Diamonds),
    ];
    let hands = vec![
        original_hand.clone(),
        vec![card(Rank::Eight, Suit::Clubs)],
        vec![card(Rank::Nine, Suit::Clubs)],
    ];
    let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
    let play = Move::Play(combo(vec![
        card(Rank::Seven, Suit::Clubs),
        card(Rank::Seven, Suit::Clubs),
    ]));
    assert_eq!(
        round.submit_move(0, play),
        Err(MoveError::CardNotInHand(card(Rank::Seven, Suit::Clubs)))
    );
    assert_eq!(round.hand(0), original_hand.as_slice());
}

#[test]
fn legal_moves_is_empty_once_the_round_is_complete() {
    let hands = vec![
        vec![card(Rank::Two, Suit::Clubs)],
        vec![card(Rank::Three, Suit::Clubs)],
        vec![card(Rank::Four, Suit::Clubs)],
    ];
    let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
    round
        .submit_move(0, Move::Play(combo(vec![card(Rank::Two, Suit::Clubs)])))
        .unwrap();
    round
        .submit_move(1, Move::Play(combo(vec![card(Rank::Three, Suit::Clubs)])))
        .unwrap();
    assert!(round.is_complete());
    assert_eq!(round.legal_moves(), Vec::new());
}

#[test]
fn legal_moves_matches_the_leaders_hand_when_no_combo_is_on_the_table() {
    let hands = vec![
        vec![
            card(Rank::Two, Suit::Clubs),
            card(Rank::Two, Suit::Diamonds),
        ],
        vec![card(Rank::Three, Suit::Clubs)],
        vec![card(Rank::Four, Suit::Clubs)],
    ];
    let round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
    let moves = round.legal_moves();
    assert_ne!(moves, []);
    assert!(!moves.contains(&Move::Pass));
}

#[test]
fn accepts_a_legal_two_card_combo_of_distinct_cards() {
    let hands = vec![
        vec![
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Seven, Suit::Diamonds),
        ],
        vec![card(Rank::Eight, Suit::Clubs)],
        vec![card(Rank::Nine, Suit::Clubs)],
    ];
    let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
    let play = Move::Play(combo(vec![
        card(Rank::Seven, Suit::Clubs),
        card(Rank::Seven, Suit::Diamonds),
    ]));
    assert_eq!(round.submit_move(0, play), Ok(()));
    assert_eq!(round.hand(0), []);
}

#[test]
fn play_history_records_plays_in_order_and_omits_passes() {
    let hands = vec![
        vec![card(Rank::Eight, Suit::Clubs), card(Rank::Ten, Suit::Clubs)],
        vec![card(Rank::Seven, Suit::Clubs), card(Rank::Six, Suit::Clubs)],
        vec![card(Rank::Nine, Suit::Clubs), card(Rank::Five, Suit::Clubs)],
    ];
    let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
    assert_eq!(round.play_history(), &[]);

    let first_play = combo(vec![card(Rank::Eight, Suit::Clubs)]);
    round.submit_move(0, Move::Play(first_play)).unwrap();
    assert_eq!(round.play_history(), &[(0, first_play)]);

    round.submit_move(1, Move::Pass).unwrap();
    assert_eq!(
        round.play_history(),
        &[(0, first_play)],
        "a pass must not appear in play_history"
    );

    let second_play = combo(vec![card(Rank::Nine, Suit::Clubs)]);
    round.submit_move(2, Move::Play(second_play)).unwrap();
    assert_eq!(round.play_history(), &[(0, first_play), (2, second_play)]);
}

#[test]
fn hand_size_matches_hand_len_and_decreases_after_a_play() {
    let hands = vec![
        vec![card(Rank::Eight, Suit::Clubs), card(Rank::Ten, Suit::Clubs)],
        vec![card(Rank::Seven, Suit::Clubs)],
        vec![card(Rank::Nine, Suit::Clubs)],
    ];
    let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
    assert_eq!(round.hand_size(0), 2);
    assert_eq!(round.hand_size(0), round.hand(0).len());

    round
        .submit_move(0, Move::Play(combo(vec![card(Rank::Eight, Suit::Clubs)])))
        .unwrap();
    assert_eq!(round.hand_size(0), 1);
}

#[test]
fn pass_history_records_the_combo_and_play_count_at_pass_time() {
    let hands = vec![
        vec![card(Rank::Eight, Suit::Clubs), card(Rank::Ten, Suit::Clubs)],
        vec![card(Rank::Seven, Suit::Clubs), card(Rank::Six, Suit::Clubs)],
        vec![card(Rank::Nine, Suit::Clubs), card(Rank::Five, Suit::Clubs)],
    ];
    let mut round =
        Round::with_pass_rule(hands, DuplicateRule::FirstDealtWins, PassRule::Free, 0).unwrap();
    assert_eq!(round.pass_history(), &[]);

    let lead = combo(vec![card(Rank::Eight, Suit::Clubs)]);
    round.submit_move(0, Move::Play(lead)).unwrap(); // play_history now len 1
    round.submit_move(1, Move::Pass).unwrap();
    assert_eq!(round.pass_history(), &[(1, lead, 1)]);

    let beat = combo(vec![card(Rank::Nine, Suit::Clubs)]);
    round.submit_move(2, Move::Play(beat)).unwrap(); // play_history now len 2
                                                     // Trick resolves (both non-leaders acted); seat 0 leads again with
                                                     // its remaining card.
    let second_lead = combo(vec![card(Rank::Ten, Suit::Clubs)]);
    round.submit_move(0, Move::Play(second_lead)).unwrap(); // len 3
    round.submit_move(1, Move::Pass).unwrap();
    assert_eq!(
        round.pass_history(),
        &[(1, lead, 1), (1, second_lead, 3)],
        "each pass is tagged with play_history().len() at that exact moment"
    );
}

// ---- the pass rule -------------------------------------------------

fn single_hands(ranks: &[&[Rank]]) -> Vec<Vec<Card>> {
    ranks
        .iter()
        .enumerate()
        .map(|(seat, hand)| {
            hand.iter()
                .enumerate()
                .map(|(i, &rank)| {
                    Card::new(rank, Suit::Clubs, u8::try_from(seat * 10 + i).unwrap())
                })
                .collect()
        })
        .collect()
}

fn play(round: &mut Round, seat: SeatId, rank: Rank) {
    let card = round
        .hand(seat)
        .iter()
        .copied()
        .find(|c| c.rank == rank)
        .expect("the seat holds that rank");
    round
        .submit_move(seat, Move::Play(combo(vec![card])))
        .unwrap();
}

#[test]
fn under_the_final_rule_a_seat_that_passed_is_skipped_for_the_rest_of_the_trick() {
    use Rank::*;
    // 4 seats: 0 leads 4, 1 passes (out), 2 plays 6, 3 passes (out),
    // 0 plays 9 -> turn must go to 2 (1 and 3 are out), not to 1.
    let hands = single_hands(&[
        &[Four, Nine, Ace],
        &[Five, Seven],
        &[Six, Eight],
        &[Seven, Ten],
    ]);
    let mut round =
        Round::with_pass_rule(hands, DuplicateRule::FirstDealtWins, PassRule::Final, 0).unwrap();
    play(&mut round, 0, Four);
    round.submit_move(1, Move::Pass).unwrap();
    play(&mut round, 2, Six);
    round.submit_move(3, Move::Pass).unwrap();
    assert_eq!(round.seat_to_move(), Some(0));
    play(&mut round, 0, Nine);
    assert_eq!(
        round.seat_to_move(),
        Some(2),
        "seats 1 and 3 passed and are out"
    );
    assert_eq!(
        round.submit_move(1, Move::Pass),
        Err(MoveError::NotYourTurn { expected: 2 })
    );
    // Seat 2 passes: nobody is left but seat 0, who wins and leads.
    round.submit_move(2, Move::Pass).unwrap();
    assert_eq!(round.current_combo(), None);
    assert_eq!(round.seat_to_move(), Some(0));
    assert!(!round.has_passed(1), "a new trick clears the passes");
}

#[test]
fn the_free_rule_lets_a_passed_seat_play_again_in_the_same_trick() {
    use Rank::*;
    let hands = single_hands(&[
        &[Four, Nine, Ace],
        &[Five, Seven],
        &[Six, Eight],
        &[Seven, Ten],
    ]);
    let mut round =
        Round::with_pass_rule(hands, DuplicateRule::FirstDealtWins, PassRule::Free, 0).unwrap();
    play(&mut round, 0, Four);
    round.submit_move(1, Move::Pass).unwrap();
    play(&mut round, 2, Six);
    round.submit_move(3, Move::Pass).unwrap();
    play(&mut round, 0, Nine);
    assert_eq!(round.seat_to_move(), Some(1), "free: seat 1 is back in");
    assert!(!round.has_passed(1), "free never marks a seat as out");
    // Seat 1 can act again (here it passes once more) and play goes on.
    round.submit_move(1, Move::Pass).unwrap();
    assert_eq!(round.seat_to_move(), Some(2));
}

#[test]
fn a_passed_seat_acts_again_in_the_next_trick() {
    use Rank::*;
    let hands = single_hands(&[&[Four, Nine], &[Five, Seven], &[Six, Eight]]);
    let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, 0).unwrap();
    assert_eq!(round.pass_rule(), PassRule::Final, "the default rule");
    play(&mut round, 0, Four);
    round.submit_move(1, Move::Pass).unwrap();
    round.submit_move(2, Move::Pass).unwrap();
    // Trick over: seat 0 leads again, and seat 1 may play in the next one.
    assert_eq!(round.seat_to_move(), Some(0));
    play(&mut round, 0, Nine);
    assert_eq!(round.seat_to_move(), Some(1));
    round.submit_move(1, Move::Pass).unwrap();
}

#[test]
fn when_the_winner_goes_out_the_lead_passes_to_the_next_active_seat_under_final() {
    use Rank::*;
    // Seat 0 leads its last card (and is out); 1 passes; 2 beats it with
    // its last card (and is out, as the trick's winner); 3 passes. The
    // trick ends and the next active seat after the winner leads.
    let hands = single_hands(&[&[Four], &[Five, Seven], &[Six], &[Eight, Ten]]);
    let mut round =
        Round::with_pass_rule(hands, DuplicateRule::FirstDealtWins, PassRule::Final, 0).unwrap();
    play(&mut round, 0, Four); // seat 0 is out
    round.submit_move(1, Move::Pass).unwrap();
    play(&mut round, 2, Six); // seat 2 is out, winner of the trick
    assert_eq!(round.seat_to_move(), Some(3));
    round.submit_move(3, Move::Pass).unwrap();
    // 1 and 3 passed, 0 and 2 are out: the trick is over and the next
    // active seat after the winner (2) leads: that is seat 3.
    assert_eq!(round.current_combo(), None);
    assert_eq!(round.seat_to_move(), Some(3));
}

/// Plays random legal rounds with a lot of passing, for every table
/// size, both decks and both rules: every round must end, every seat
/// must appear once in the finishing order, and (under `Final`) nobody is
/// ever asked to move after passing in the same trick.
#[test]
fn random_rounds_always_finish_under_both_rules() {
    use crate::card::DeckVariant;
    use crate::deal::{deal, lowest_card_holder};
    use crate::deck::standard_deck;
    use rand::seq::SliceRandom;
    use rand::{RngExt, SeedableRng};

    let mut rng = rand::rngs::StdRng::seed_from_u64(99);
    for rule in [PassRule::Free, PassRule::Final] {
        for variant in [DeckVariant::Single, DeckVariant::Double] {
            for players in 3..=6u8 {
                for _ in 0..150 {
                    let mut deck = standard_deck(variant);
                    deck.shuffle(&mut rng);
                    for (i, c) in deck.iter_mut().enumerate() {
                        c.deal_index = u8::try_from(i).unwrap();
                    }
                    let hands = deal(deck, players).unwrap();
                    let leader = lowest_card_holder(&hands, DuplicateRule::FirstDealtWins).unwrap();
                    let mut round =
                        Round::with_pass_rule(hands, DuplicateRule::FirstDealtWins, rule, leader)
                            .unwrap();
                    let mut passed_in_trick: Vec<SeatId> = Vec::new();
                    let mut steps = 0;
                    while let Some(seat) = round.seat_to_move() {
                        steps += 1;
                        assert!(steps < 10_000, "a round never ended ({rule:?})");
                        if round.current_combo().is_none() {
                            passed_in_trick.clear();
                        }
                        if rule == PassRule::Final {
                            assert!(
                                !passed_in_trick.contains(&seat),
                                "seat {seat} acts again after passing in the same trick"
                            );
                        }
                        let moves = round.legal_moves();
                        let pass = moves.contains(&Move::Pass) && rng.random_bool(0.5);
                        let mv = if pass {
                            Move::Pass
                        } else {
                            moves[rng.random_range(0..moves.len())]
                        };
                        if mv == Move::Pass {
                            passed_in_trick.push(seat);
                        }
                        round.submit_move(seat, mv).unwrap();
                    }
                    let mut order = round.finishing_order().to_vec();
                    order.sort_unstable();
                    let expected: Vec<SeatId> = (0..players).collect();
                    assert_eq!(order, expected, "{rule:?} {variant:?} {players}p");
                }
            }
        }
    }
}
#[test]
fn a_cloned_round_continues_identically() {
    use rand::{RngExt, SeedableRng};
    for rule in [PassRule::default(), PassRule::Final] {
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let mut deck = crate::deck::standard_deck(crate::card::DeckVariant::Single);
        for (index, c) in deck.iter_mut().enumerate() {
            c.deal_index = u8::try_from(index).unwrap();
        }
        let hands = crate::deal::deal(deck, 4).unwrap();
        let mut original =
            Round::with_pass_rule(hands, DuplicateRule::FirstDealtWins, rule, 0).unwrap();
        // Advance a few moves, then clone mid-round.
        for _ in 0..6 {
            let seat = original.seat_to_move().unwrap();
            let moves = original.legal_moves();
            let mv = moves[rng.random_range(0..moves.len())];
            original.submit_move(seat, mv).unwrap();
        }
        let mut copy = original.clone();
        while let Some(seat) = original.seat_to_move() {
            assert_eq!(copy.seat_to_move(), Some(seat));
            let moves = original.legal_moves();
            assert_eq!(moves, copy.legal_moves());
            let mv = moves[rng.random_range(0..moves.len())];
            original.submit_move(seat, mv).unwrap();
            copy.submit_move(seat, mv).unwrap();
        }
        assert!(copy.is_complete());
        assert_eq!(original.finishing_order(), copy.finishing_order());
        assert_eq!(original.play_history(), copy.play_history());
    }
}
