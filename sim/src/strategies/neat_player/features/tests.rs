use engine::{Combo, Rank, Suit};

use super::*;
use crate::hand_reading::{read_pass_ceilings, PassCeilings};

const RULE: DuplicateRule = DuplicateRule::FirstDealtWins;

fn card(rank: Rank, suit: Suit) -> Card {
    Card::new(rank, suit, 0)
}

fn play(cards: &[Card]) -> Move {
    Move::Play(Combo::new(cards.to_vec()).unwrap())
}

fn opponent(seat: u8, hand_size: usize, active: bool, pass_ceilings: PassCeilings) -> OpponentHand {
    OpponentHand {
        seat,
        hand_size,
        active,
        pass_ceilings,
    }
}

/// Hand: 3d 3h 9s Ac. Unseen: 4d 9c Kh Kd Ks.
struct Scenario {
    hand: Vec<Card>,
    unseen: Vec<Card>,
    opponents: Vec<OpponentHand>,
}

fn scenario() -> Scenario {
    Scenario {
        hand: vec![
            card(Rank::Three, Suit::Diamonds),
            card(Rank::Three, Suit::Hearts),
            card(Rank::Nine, Suit::Spades),
            card(Rank::Ace, Suit::Clubs),
        ],
        unseen: vec![
            card(Rank::Four, Suit::Diamonds),
            card(Rank::Nine, Suit::Clubs),
            card(Rank::King, Suit::Hearts),
            card(Rank::King, Suit::Diamonds),
            card(Rank::King, Suit::Spades),
        ],
        opponents: vec![
            opponent(1, 3, true, PassCeilings::default()),
            opponent(2, 2, true, PassCeilings::default()),
            opponent(3, 0, false, PassCeilings::default()),
        ],
    }
}

fn context<'a>(s: &'a Scenario, table: Option<&'a Combo>) -> TurnContext<'a> {
    TurnContext {
        seat: 0,
        hand: &s.hand,
        opponents: s.opponents.clone(),
        unseen_cards: s.unseen.clone(),
        own_pass_ceilings: PassCeilings::default(),
        current_combo: table,
    }
}

fn approx(actual: f64, expected: f64, name: &str) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "{name}: {actual} vs {expected}"
    );
}

#[test]
fn names_are_unique_and_match_the_count() {
    assert_eq!(FEATURE_NAMES.len(), FEATURE_COUNT);
    let mut sorted = FEATURE_NAMES.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), FEATURE_COUNT);
}

#[test]
fn state_features_describe_the_table_and_opponents() {
    let s = scenario();
    let ctx = context(&s, None);
    let f = TurnSummary::new(&ctx, RULE).features(&play(&[card(Rank::Nine, Suit::Spades)]));
    approx(f[8], 1.0, "is_leading");
    approx(f[9], 0.0, "table_combo_size");
    approx(f[10], 4.0 / 13.0, "hand_size");
    approx(f[11], 2.0 / 4.0, "grouped_fraction (3d,3h)");
    approx(f[12], 2.0 / 4.0, "singleton_fraction (9s, Ac)");
    approx(f[13], 2.0 / 5.0, "active_opponents (inactive one ignored)");
    approx(f[14], 2.0 / 13.0, "min_opponent_hand");
    approx(f[15], 2.5 / 13.0, "mean_opponent_hand");
    approx(f[16], 1.0, "opponent_close (a seat holds 2 cards)");
}

#[test]
fn following_a_combo_sets_the_table_features() {
    let s = scenario();
    let on_table = Combo::new(vec![card(Rank::Five, Suit::Hearts)]).unwrap();
    let ctx = context(&s, Some(&on_table));
    let f = TurnSummary::new(&ctx, RULE).features(&Move::Pass);
    approx(f[8], 0.0, "is_leading");
    approx(f[9], 1.0 / 8.0, "table_combo_size");
}

#[test]
fn passing_has_only_state_features_and_the_pass_flag() {
    let s = scenario();
    let ctx = context(&s, None);
    let summary = TurnSummary::new(&ctx, RULE);
    let f = summary.features(&Move::Pass);
    approx(f[0], 1.0, "is_pass");
    approx(f[7], 1.0, "hand_left_after (hand unchanged)");
    for index in [1, 2, 3, 4, 5, 6, 17, 18, 19] {
        approx(f[index], 0.0, FEATURE_NAMES[index]);
    }
}

#[test]
fn playing_the_hands_top_single_sets_the_play_features() {
    let s = scenario();
    let ctx = context(&s, None);
    let f = TurnSummary::new(&ctx, RULE).features(&play(&[card(Rank::Ace, Suit::Clubs)]));
    approx(f[0], 0.0, "is_pass");
    approx(f[1], 1.0 / 8.0, "combo_size");
    approx(f[2], 1.0, "top_strength (Ac is the strongest card)");
    approx(f[3], 3.0 / 4.0, "hand_below_top");
    approx(f[4], 1.0, "uses_top_group");
    approx(f[5], 0.0, "splits_group");
    approx(f[6], 0.0, "empties_hand");
    approx(f[7], 3.0 / 4.0, "hand_left_after");
    approx(f[17], 0.0, "unseen_outranking (nothing beats Ac)");
    approx(f[18], 0.0, "unseen_beaters");
}

#[test]
fn splitting_a_pair_and_emptying_the_hand_are_flagged() {
    let s = scenario();
    let ctx = context(&s, None);
    let summary = TurnSummary::new(&ctx, RULE);
    let split = summary.features(&play(&[card(Rank::Three, Suit::Hearts)]));
    approx(split[5], 1.0, "splits_group (a single from the 3s pair)");
    let whole = summary.features(&play(&[
        card(Rank::Three, Suit::Diamonds),
        card(Rank::Three, Suit::Hearts),
    ]));
    approx(whole[5], 0.0, "splits_group (the whole pair)");
    approx(whole[1], 2.0 / 8.0, "combo_size");

    let last_card = Scenario {
        hand: vec![card(Rank::Two, Suit::Clubs)],
        unseen: Vec::new(),
        opponents: vec![opponent(1, 5, true, PassCeilings::default())],
    };
    let ctx = context(&last_card, None);
    let f = TurnSummary::new(&ctx, RULE).features(&play(&[card(Rank::Two, Suit::Clubs)]));
    approx(f[6], 1.0, "empties_hand");
    approx(f[7], 0.0, "hand_left_after");
    approx(f[17], 0.0, "unseen_outranking with nothing unseen");
}

#[test]
fn unseen_features_count_cards_and_ranks_that_beat_the_play() {
    let s = scenario();
    let ctx = context(&s, None);
    let summary = TurnSummary::new(&ctx, RULE);
    // Playing 3d: every unseen card (5) outranks it; the four
    // distinct unseen ranks (4, 9, K) -> 3 ranks can beat a single.
    let low = summary.features(&play(&[card(Rank::Three, Suit::Diamonds)]));
    approx(low[17], 1.0, "unseen_outranking");
    approx(low[18], 3.0 / 13.0, "unseen_beaters (4, 9, K)");
    // Playing 9s: 9c (higher suit) and the three kings beat it, 4d does not.
    let nine = summary.features(&play(&[card(Rank::Nine, Suit::Spades)]));
    approx(nine[17], 4.0 / 5.0, "unseen_outranking");
    approx(nine[18], 2.0 / 13.0, "unseen_beaters (9c, K)");
    // A pair can only be beaten by a rank with 2+ unseen cards: just kings.
    let pair = summary.features(&play(&[
        card(Rank::Three, Suit::Diamonds),
        card(Rank::Three, Suit::Hearts),
    ]));
    approx(pair[18], 1.0 / 13.0, "unseen_beaters for a pair");
}

#[test]
fn opponents_locked_out_counts_seats_whose_passes_prove_they_cannot_beat() {
    let mut s = scenario();
    // Seat 1 passed against the King of Hearts: it cannot beat any
    // single at or above that card. Seat 2 has said nothing.
    let king = Combo::new(vec![card(Rank::King, Suit::Hearts)]).unwrap();
    let ceilings = read_pass_ceilings(4, &[], &[(1, king, 0)], RULE);
    s.opponents[0].pass_ceilings = ceilings[1];
    let ctx = context(&s, None);
    let summary = TurnSummary::new(&ctx, RULE);
    let ace = summary.features(&play(&[card(Rank::Ace, Suit::Clubs)]));
    approx(
        ace[19],
        1.0 / 2.0,
        "one of two active opponents is locked out",
    );
    let nine = summary.features(&play(&[card(Rank::Nine, Suit::Spades)]));
    approx(
        nine[19],
        0.0,
        "9s is below the passed King, so nobody is locked out",
    );
}

#[test]
fn no_active_opponents_gives_zero_not_nan() {
    let mut s = scenario();
    s.opponents.clear();
    let ctx = context(&s, None);
    let f = TurnSummary::new(&ctx, RULE).features(&play(&[card(Rank::Ace, Suit::Clubs)]));
    assert!(f.iter().all(|v| v.is_finite()));
    approx(f[13], 0.0, "active_opponents");
    approx(f[14], 0.0, "min_opponent_hand");
    approx(f[19], 0.0, "opponents_locked_out");
}

#[test]
fn strength_follows_the_house_card_order() {
    let low = card(Rank::Two, Suit::Diamonds);
    let same_rank_higher_suit = card(Rank::Two, Suit::Clubs);
    let next_rank = card(Rank::Three, Suit::Diamonds);
    assert!(strength(low) < strength(same_rank_higher_suit));
    assert!(strength(same_rank_higher_suit) < strength(next_rank));
    approx(strength(low), 0.0, "weakest");
    approx(strength(card(Rank::Ace, Suit::Clubs)), 1.0, "strongest");
}

fn bits(features: &[f64; FEATURE_COUNT]) -> Vec<u64> {
    features.iter().map(|v| v.to_bits()).collect()
}

/// The fast summary gives bit-identical feature vectors to the scan
/// implementation it replaced, for every legal move of random states
/// (3-6 players, both decks and duplicate rules, leading and
/// following, pass ceilings present).
#[test]
fn the_fast_summary_matches_the_reference_on_random_states() {
    use crate::hand_reading::PassTracker;
    use crate::match_runner::turn_context_for;
    use crate::strategy::ContextNeeds;
    use crate::test_support::for_each_state;

    use super::reference::ReferenceTurnSummary;

    let (mut states, mut with_ceilings, mut following, mut vectors) = (0, 0, 0, 0);
    let mut tracker = None;
    for_each_state(5, 40, |state, _| {
        let round = state.round;
        if round.play_history().is_empty() && round.pass_history().is_empty() {
            tracker = Some(PassTracker::new(usize::from(state.players), state.rule));
        }
        let seat = round.seat_to_move().unwrap();
        let context = turn_context_for(
            round,
            seat,
            state.players,
            state.round_deck,
            tracker.as_mut().unwrap(),
            ContextNeeds::ALL,
        );
        let fast = TurnSummary::new(&context, state.rule);
        let reference = ReferenceTurnSummary::new(&context, state.rule);
        for candidate in &round.legal_moves() {
            assert_eq!(
                bits(&fast.features(candidate)),
                bits(&reference.features(candidate)),
                "{candidate:?} in {context_hand:?}",
                context_hand = context.hand
            );
            vectors += 1;
        }
        states += 1;
        following += usize::from(context.current_combo.is_some());
        with_ceilings += usize::from(
            context
                .opponents
                .iter()
                .any(|o| o.pass_ceilings != PassCeilings::default()),
        );
    });
    assert!(states >= 10_000, "only {states} states");
    assert!(
        following > 1000 && with_ceilings > 1000,
        "{following} {with_ceilings}"
    );
    assert!(vectors > 50_000);
}

/// Contexts that are not a physical deck (three or more cards in one
/// slot, copies sharing a `deal_index`) take the scanning fallbacks and
/// must still agree with the reference.
#[test]
fn the_fast_summary_matches_the_reference_on_unphysical_contexts() {
    use rand::{RngExt, SeedableRng};

    use super::reference::ReferenceTurnSummary;

    const RANKS: [Rank; 13] = [
        Rank::Two,
        Rank::Three,
        Rank::Four,
        Rank::Five,
        Rank::Six,
        Rank::Seven,
        Rank::Eight,
        Rank::Nine,
        Rank::Ten,
        Rank::Jack,
        Rank::Queen,
        Rank::King,
        Rank::Ace,
    ];
    const SUITS: [Suit; 4] = [Suit::Diamonds, Suit::Hearts, Suit::Spades, Suit::Clubs];
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    let random_card = |rng: &mut rand::rngs::StdRng, low_ranks: usize| {
        Card::new(
            RANKS[rng.random_range(0..low_ranks)],
            SUITS[rng.random_range(0..4)],
            rng.random_range(0..3),
        )
    };
    for _ in 0..3000 {
        let spread = rng.random_range(1..=13);
        let hand: Vec<Card> = (0..rng.random_range(0..12))
            .map(|_| random_card(&mut rng, spread))
            .collect();
        let unseen: Vec<Card> = (0..rng.random_range(0..40))
            .map(|_| random_card(&mut rng, spread))
            .collect();
        let opponents: Vec<OpponentHand> = (0..rng.random_range(0..6))
            .map(|seat| {
                let passed = Combo::new(vec![random_card(&mut rng, 13)]).unwrap();
                OpponentHand {
                    seat,
                    hand_size: rng.random_range(0..8),
                    active: rng.random_bool(0.7),
                    pass_ceilings: read_pass_ceilings(
                        1,
                        &[],
                        &[(0, passed, 0)],
                        DuplicateRule::LastDealtWins,
                    )[0],
                }
            })
            .collect();
        let table = Combo::new(vec![random_card(&mut rng, 13)]).unwrap();
        let context = TurnContext {
            seat: 0,
            hand: &hand,
            opponents,
            unseen_cards: unseen,
            own_pass_ceilings: PassCeilings::default(),
            current_combo: rng.random_bool(0.5).then_some(&table),
        };
        for rule in [DuplicateRule::FirstDealtWins, DuplicateRule::LastDealtWins] {
            let fast = TurnSummary::new(&context, rule);
            let reference = ReferenceTurnSummary::new(&context, rule);
            let rank = RANKS[rng.random_range(0..13)];
            let size = rng.random_range(1..=4);
            let cards: Vec<Card> = (0..size)
                .map(|_| Card::new(rank, SUITS[rng.random_range(0..4)], rng.random_range(0..3)))
                .collect();
            for candidate in [Move::Pass, Move::Play(Combo::new(cards).unwrap())] {
                assert_eq!(
                    bits(&fast.features(&candidate)),
                    bits(&reference.features(&candidate)),
                    "{candidate:?}, hand {hand:?}"
                );
            }
        }
    }
}
