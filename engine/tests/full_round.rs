//! Scripted, fully deterministic integration tests exercising the whole
//! Phase 1 pipeline together: deal -> round play -> finishing order ->
//! role assignment -> exchange for the next round. See docs/RULES.md.

use engine::{
    assign_roles, deal, exchange, lowest_card_holder, Card, Combo, DuplicateRule, Move, Rank, Role,
    Round, SeatId, Suit,
};

fn clubs(rank: Rank) -> Card {
    Card::new(rank, Suit::Clubs, 0)
}

fn diamonds(rank: Rank) -> Card {
    Card::new(rank, Suit::Diamonds, 0)
}

fn single(card: Card) -> Combo {
    Combo::new(vec![card]).unwrap()
}

#[test]
fn four_player_round_deal_through_exchange() {
    // --- Deal ---
    let deck = vec![
        clubs(Rank::Two),
        clubs(Rank::Three),
        clubs(Rank::Four),
        clubs(Rank::Five),
        clubs(Rank::Six),
        clubs(Rank::Seven),
        clubs(Rank::Eight),
        clubs(Rank::Nine),
    ];
    let hands = deal(deck, 4).expect("8 cards deal evenly into 4 hands of 2");
    assert_eq!(hands[0], vec![clubs(Rank::Two), clubs(Rank::Six)]);
    assert_eq!(hands[1], vec![clubs(Rank::Three), clubs(Rank::Seven)]);
    assert_eq!(hands[2], vec![clubs(Rank::Four), clubs(Rank::Eight)]);
    assert_eq!(hands[3], vec![clubs(Rank::Five), clubs(Rank::Nine)]);

    let leader =
        lowest_card_holder(&hands, DuplicateRule::FirstDealtWins).expect("every hand has cards");
    assert_eq!(
        leader, 0,
        "seat 0 holds the globally lowest card (Two of Clubs)"
    );

    let mut round = Round::new(hands, DuplicateRule::FirstDealtWins, leader)
        .expect("4 hands, valid leader, no empty hands");

    // --- Trick 1: an escalating single-card war, then everyone passes
    //     even though seat 0 could beat seat 3's Five with its own Six
    //     (passing is always legal, not just a last resort) ---
    round
        .submit_move(0, Move::Play(single(clubs(Rank::Two))))
        .unwrap();
    round
        .submit_move(1, Move::Play(single(clubs(Rank::Three))))
        .unwrap();
    round
        .submit_move(2, Move::Play(single(clubs(Rank::Four))))
        .unwrap();
    round
        .submit_move(3, Move::Play(single(clubs(Rank::Five))))
        .unwrap();
    round.submit_move(0, Move::Pass).unwrap(); // voluntary pass, holds a beating Six
    round.submit_move(1, Move::Pass).unwrap();
    round.submit_move(2, Move::Pass).unwrap();
    assert_eq!(
        round.seat_to_move(),
        Some(3),
        "seat 3 won trick 1 and leads trick 2"
    );

    // --- Trick 2: seat 3 leads its last card and empties its hand while
    //     leading. Per the confirmed house rule, leadership then passes
    //     to the next active seat in turn order (seat 0), not staying
    //     with the now-finished seat 3. ---
    round
        .submit_move(3, Move::Play(single(clubs(Rank::Nine))))
        .unwrap();
    round.submit_move(0, Move::Pass).unwrap();
    round.submit_move(1, Move::Pass).unwrap();
    round.submit_move(2, Move::Pass).unwrap();
    assert_eq!(round.finishing_order(), &[3]);
    assert_eq!(
        round.seat_to_move(),
        Some(0),
        "seat 3 emptied its hand leading trick 2; seat 0 (next active) leads trick 3"
    );

    // --- Trick 3: seat 0 leads its last card and empties its hand ---
    round
        .submit_move(0, Move::Play(single(clubs(Rank::Six))))
        .unwrap();
    round.submit_move(1, Move::Pass).unwrap();
    round.submit_move(2, Move::Pass).unwrap();
    assert_eq!(round.finishing_order(), &[3, 0]);
    assert_eq!(round.seat_to_move(), Some(1));

    // --- Trick 4: seat 1 leads its last card; only seat 2 remains, so
    //     the round completes immediately without needing seat 2 to
    //     explicitly pass on an unbeatable card. ---
    round
        .submit_move(1, Move::Play(single(clubs(Rank::Seven))))
        .unwrap();
    assert!(round.is_complete());
    assert_eq!(round.finishing_order(), &[3, 0, 1, 2]);
    assert_eq!(round.seat_to_move(), None);

    // --- Role assignment and exchange for the next round ---
    assert_role_assignment_and_exchange(round.finishing_order());
}

fn assert_role_assignment_and_exchange(finishing_order: &[SeatId]) {
    // --- Role assignment ---
    let role_by_seat =
        assign_roles(finishing_order, 4).expect("a complete 4-player finishing order");
    assert_eq!(role_by_seat[3], Role::President);
    assert_eq!(role_by_seat[0], Role::Vize);
    assert_eq!(role_by_seat[1], Role::ViceArschloch);
    assert_eq!(role_by_seat[2], Role::Arschloch);

    // --- Exchange for the next round, on a fresh (unrelated) deal ---
    let mut next_hands = vec![
        vec![
            diamonds(Rank::Ten),
            diamonds(Rank::Jack),
            diamonds(Rank::Queen),
        ], // seat 0, Vize
        vec![
            diamonds(Rank::Two),
            diamonds(Rank::Three),
            diamonds(Rank::Four),
        ], // seat 1, ViceArschloch
        vec![
            diamonds(Rank::Five),
            diamonds(Rank::Six),
            diamonds(Rank::Seven),
        ], // seat 2, Arschloch
        vec![
            diamonds(Rank::Eight),
            diamonds(Rank::Nine),
            diamonds(Rank::King),
        ], // seat 3, President
    ];
    exchange(
        &mut next_hands,
        &role_by_seat,
        DuplicateRule::FirstDealtWins,
    )
    .expect("valid 4-player exchange");

    // President (seat 3) <-> Arschloch (seat 2) swap 2 cards; Vize (seat
    // 0) <-> ViceArschloch (seat 1) swap 1 card.
    assert_eq!(
        next_hands[0],
        vec![
            diamonds(Rank::Jack),
            diamonds(Rank::Queen),
            diamonds(Rank::Four)
        ]
    );
    assert_eq!(
        next_hands[1],
        vec![
            diamonds(Rank::Two),
            diamonds(Rank::Three),
            diamonds(Rank::Ten)
        ]
    );
    assert_eq!(
        next_hands[2],
        vec![
            diamonds(Rank::Five),
            diamonds(Rank::Eight),
            diamonds(Rank::Nine)
        ]
    );
    assert_eq!(
        next_hands[3],
        vec![
            diamonds(Rank::King),
            diamonds(Rank::Six),
            diamonds(Rank::Seven)
        ]
    );
}
