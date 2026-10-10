//! Tests of the `trick`, `hidden` and `table` feature groups.

use engine::{Combo, DeckVariant, PassRule, Rank, Suit};

use super::*;
use crate::hand_reading::{read_pass_ceilings, PassCeilings};
use crate::strategies::neat_player::feature_set::FEATURE_NAMES_V2;
use crate::strategy::TableInfo;

const RULE: DuplicateRule = DuplicateRule::FirstDealtWins;

fn card(rank: Rank, suit: Suit) -> Card {
    Card::new(rank, suit, 0)
}

fn card_n(rank: Rank, suit: Suit, deal_index: u8) -> Card {
    Card::new(rank, suit, deal_index)
}

fn play(cards: &[Card]) -> Move {
    Move::Play(Combo::new(cards.to_vec()).unwrap())
}

fn opp(seat: u8, hand_size: usize, in_trick: bool) -> OpponentHand {
    OpponentHand {
        seat,
        hand_size,
        active: hand_size > 0,
        in_trick,
        pass_ceilings: PassCeilings::default(),
    }
}

struct Spec {
    seat: u8,
    hand: Vec<Card>,
    unseen: Vec<Card>,
    opponents: Vec<OpponentHand>,
    holder: Option<u8>,
    table: TableInfo,
}

impl Spec {
    fn new(hand: Vec<Card>, unseen: Vec<Card>, opponents: Vec<OpponentHand>) -> Self {
        Self {
            seat: 0,
            hand,
            unseen,
            opponents,
            holder: None,
            table: TableInfo {
                player_count: 4,
                deck: DeckVariant::Single,
                pass_rule: PassRule::Final,
            },
        }
    }

    /// The value of the feature `name` for `candidate`, from the full set.
    fn feature(&self, name: &str, candidate: &Move) -> f64 {
        self.features(candidate)[FEATURE_NAMES_V2.iter().position(|n| *n == name).unwrap()]
    }

    fn features(&self, candidate: &Move) -> [f64; FEATURE_COUNT_V2] {
        let context = TurnContext {
            seat: self.seat,
            hand: &self.hand,
            opponents: self.opponents.clone(),
            unseen_cards: self.unseen.clone(),
            own_pass_ceilings: PassCeilings::default(),
            current_combo: None,
            combo_holder: self.holder,
            table: self.table,
        };
        TurnSummary::with_features(&context, RULE, FeatureSet::All).features(candidate)
    }
}

fn approx(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "{what}: {actual} vs {expected}"
    );
}

const PASS: Move = Move::Pass;

fn bits(features: &[f64]) -> Vec<u64> {
    features.iter().map(|v| v.to_bits()).collect()
}

// ------------------------------------------------------------------ trick

#[test]
fn the_trick_group_counts_the_opponents_still_in_the_trick() {
    // Seat 1 to act with seats 2 (5 cards), 3 (7, passed) and 0 (2, holds the
    // table combo) around the table: play order after seat 1 is 2, 3, 0.
    let mut spec = Spec::new(
        vec![card(Rank::Nine, Suit::Spades)],
        vec![],
        vec![opp(0, 2, true), opp(2, 5, true), opp(3, 7, false)],
    );
    spec.seat = 1;
    spec.holder = Some(0);
    let any = play(&[card(Rank::Nine, Suit::Spades)]);
    approx(spec.feature("opponents_in_trick", &any), 2.0 / 5.0, "count");
    approx(
        spec.feature("fraction_in_trick", &any),
        2.0 / 3.0,
        "fraction",
    );
    approx(
        spec.feature("all_in_trick_close", &any),
        0.0,
        "5 cards is not close",
    );
    approx(
        spec.feature("min_hand_in_trick", &any),
        2.0 / 13.0,
        "min hand",
    );
    // Seat 2 acts after me before the trick returns to the holder (seat 0).
    approx(
        spec.feature("last_to_act_in_trick", &any),
        0.0,
        "seat 2 still to act",
    );

    // Seat 2 has passed: only the holder (2 cards) is left in the trick and
    // nobody acts between me and the holder.
    spec.opponents[1].in_trick = false;
    approx(spec.feature("opponents_in_trick", &any), 1.0 / 5.0, "count");
    approx(
        spec.feature("fraction_in_trick", &any),
        1.0 / 3.0,
        "fraction",
    );
    approx(
        spec.feature("all_in_trick_close", &any),
        1.0,
        "holder has 2 cards",
    );
    approx(
        spec.feature("min_hand_in_trick", &any),
        2.0 / 13.0,
        "min hand",
    );
    approx(
        spec.feature("last_to_act_in_trick", &any),
        1.0,
        "last to act",
    );

    // Leading a fresh trick: everyone active acts after me.
    spec.holder = None;
    spec.opponents = vec![opp(0, 2, true), opp(2, 1, true), opp(3, 7, true)];
    approx(spec.feature("opponents_in_trick", &any), 3.0 / 5.0, "count");
    approx(spec.feature("fraction_in_trick", &any), 1.0, "fraction");
    approx(
        spec.feature("all_in_trick_close", &any),
        0.0,
        "seat 3 has 7",
    );
    approx(
        spec.feature("min_hand_in_trick", &any),
        1.0 / 13.0,
        "min hand",
    );
    approx(
        spec.feature("last_to_act_in_trick", &any),
        0.0,
        "all to act",
    );

    // Nobody in the trick (out of cards, or all passed).
    spec.opponents = vec![opp(0, 0, false), opp(2, 3, false), opp(3, 4, false)];
    for name in [
        "opponents_in_trick",
        "fraction_in_trick",
        "all_in_trick_close",
        "min_hand_in_trick",
    ] {
        approx(spec.feature(name, &any), 0.0, name);
    }
    approx(
        spec.feature("last_to_act_in_trick", &any),
        1.0,
        "nobody left",
    );
    // No opponent at all.
    spec.opponents.clear();
    approx(spec.feature("fraction_in_trick", &any), 0.0, "no opponents");
    approx(spec.feature("last_to_act_in_trick", &any), 1.0, "alone");
}

#[test]
fn the_trick_features_do_not_depend_on_the_candidate() {
    let mut spec = Spec::new(
        vec![
            card(Rank::Nine, Suit::Spades),
            card(Rank::Four, Suit::Hearts),
        ],
        vec![card(Rank::King, Suit::Hearts)],
        vec![opp(1, 1, true), opp(2, 3, true)],
    );
    spec.holder = Some(2);
    let a = spec.features(&PASS);
    let b = spec.features(&play(&[card(Rank::Four, Suit::Hearts)]));
    assert_eq!(a[20..25], b[20..25]);
    assert_eq!(a[28..31], b[28..31]);
}

/// `in_trick` is read off the real round: a pass takes a seat out of the
/// trick under `final` but not under `free`, and `combo_holder` is the seat
/// that played the combo on the table.
#[test]
fn in_trick_follows_the_pass_rule_of_a_real_round() {
    use crate::hand_reading::PassTracker;
    use crate::match_runner::turn_context_for;
    use crate::strategy::ContextNeeds;
    use engine::Round;

    let deck_cards = [
        card_n(Rank::Three, Suit::Diamonds, 0),
        card_n(Rank::Nine, Suit::Spades, 1),
        card_n(Rank::Four, Suit::Diamonds, 2),
        card_n(Rank::King, Suit::Clubs, 3),
        card_n(Rank::Five, Suit::Diamonds, 4),
        card_n(Rank::Ace, Suit::Clubs, 5),
    ];
    let deck = deck_cards.to_vec();
    for rule in [PassRule::Final, PassRule::Free] {
        let hands = vec![
            vec![deck_cards[0], deck_cards[1]],
            vec![deck_cards[2], deck_cards[3]],
            vec![deck_cards[4], deck_cards[5]],
        ];
        let mut round = Round::with_pass_rule(hands, RULE, rule, 0).unwrap();
        round.submit_move(0, play(&[deck_cards[0]])).unwrap();
        let mut tracker = PassTracker::new(3, RULE);
        let context_of = |round: &Round, seat: u8, tracker: &mut PassTracker| {
            let context = turn_context_for(round, seat, 3, &deck, tracker, ContextNeeds::ALL);
            (
                context
                    .opponents
                    .iter()
                    .map(|o| o.in_trick)
                    .collect::<Vec<_>>(),
                context.combo_holder,
                context.table,
            )
        };

        // Seat 1 to move; nobody has passed.
        let (in_trick, holder, table) = context_of(&round, 1, &mut tracker);
        assert_eq!(in_trick, vec![true, true], "{rule}");
        assert_eq!(holder, Some(0));
        assert_eq!(
            table,
            TableInfo {
                player_count: 3,
                deck: DeckVariant::Single,
                pass_rule: rule
            }
        );

        // Seat 1 passes; seat 2 sees it.
        round.submit_move(1, PASS).unwrap();
        let (in_trick, holder, _) = context_of(&round, 2, &mut tracker);
        // Opponents of seat 2 are seats 0 and 1.
        let seat_1_in = rule == PassRule::Free;
        assert_eq!(in_trick, vec![true, seat_1_in], "{rule}");
        assert_eq!(holder, Some(0));
    }
}

#[test]
fn a_leader_sees_no_holder_and_the_unrequested_parts_stay_default() {
    use crate::hand_reading::PassTracker;
    use crate::match_runner::turn_context_for;
    use crate::strategy::ContextNeeds;
    use engine::Round;

    let hands = vec![
        vec![card_n(Rank::Three, Suit::Diamonds, 0)],
        vec![card_n(Rank::Four, Suit::Diamonds, 1)],
        vec![card_n(Rank::Five, Suit::Diamonds, 2)],
    ];
    let round = Round::with_pass_rule(hands, RULE, PassRule::Final, 0).unwrap();
    let mut tracker = PassTracker::new(3, RULE);
    let full = turn_context_for(&round, 0, 3, &[], &mut tracker, ContextNeeds::ALL);
    assert_eq!(full.combo_holder, None);
    assert!(full.opponents.iter().all(|o| o.in_trick));
    let bare = turn_context_for(&round, 0, 3, &[], &mut tracker, ContextNeeds::OPPONENTS);
    assert!(
        bare.opponents.iter().all(|o| !o.in_trick),
        "TRICK not asked"
    );
    assert_eq!(bare.table, TableInfo::default());
}

// ------------------------------------------------------------------ table

#[test]
fn the_table_group_reads_player_count_deck_and_pass_rule() {
    let mut spec = Spec::new(vec![card(Rank::Nine, Suit::Spades)], vec![], vec![]);
    let any = PASS;
    for (players, double, free, expected_players) in [
        (3u8, false, false, 0.0),
        (4, false, true, 1.0 / 3.0),
        (5, true, false, 2.0 / 3.0),
        (6, true, true, 1.0),
    ] {
        spec.table = TableInfo {
            player_count: players,
            deck: if double {
                DeckVariant::Double
            } else {
                DeckVariant::Single
            },
            pass_rule: if free {
                PassRule::Free
            } else {
                PassRule::Final
            },
        };
        approx(
            spec.feature("table_players", &any),
            expected_players,
            "players",
        );
        approx(
            spec.feature("table_double_deck", &any),
            f64::from(u8::from(double)),
            "deck",
        );
        approx(
            spec.feature("table_pass_final", &any),
            f64::from(u8::from(!free)),
            "rule",
        );
    }
}

// ----------------------------------------------------------------- hidden

/// Hand 3d Kh Ac; unseen Ks Ad 4c 5d 6d (U = 5); seats 1 and 2 hold two
/// cards each and are in the trick, seat 3 holds one and has passed.
fn hidden_spec() -> Spec {
    Spec::new(
        vec![
            card(Rank::Three, Suit::Diamonds),
            card(Rank::King, Suit::Hearts),
            card(Rank::Ace, Suit::Clubs),
        ],
        vec![
            card(Rank::King, Suit::Spades),
            card(Rank::Ace, Suit::Diamonds),
            card(Rank::Four, Suit::Clubs),
            card(Rank::Five, Suit::Diamonds),
            card(Rank::Six, Suit::Diamonds),
        ],
        vec![opp(1, 2, true), opp(2, 2, true), opp(3, 1, false)],
    )
}

#[test]
fn p_win_trick_expected_beaters_and_unbeatable_fraction_by_hand() {
    let spec = hidden_spec();
    // Playing Kh: beaters are Ks (same rank, higher suit) and Ad, so m = 2.
    // P(an opponent holding 2 of the 5 has one) = 1 - (3/5)(2/4) = 0.7;
    // both fail with 0.3 * 0.3.
    let kh = play(&[card(Rank::King, Suit::Hearts)]);
    approx(spec.feature("p_win_trick", &kh), 0.09, "p_win Kh");
    // Expected beater cards: 2 * 2 / 5 per opponent in the trick, / 13.
    approx(
        spec.feature("expected_beaters", &kh),
        1.6 / 52.0,
        "beaters Kh",
    );
    // Left in hand: 3d (beatable) and Ac (the unseen Ad is below it).
    approx(
        spec.feature("my_unbeatable_fraction", &kh),
        0.5,
        "unbeatable Kh",
    );

    // Ac cannot be beaten: nothing above, and Ad is the lower Ace.
    let ac = play(&[card(Rank::Ace, Suit::Clubs)]);
    approx(spec.feature("p_win_trick", &ac), 1.0, "p_win Ac");
    approx(spec.feature("expected_beaters", &ac), 0.0, "beaters Ac");
    // Left: 3d and Kh, both beatable.
    approx(
        spec.feature("my_unbeatable_fraction", &ac),
        0.0,
        "unbeatable Ac",
    );

    // A pass wins nothing and beats nothing; the hand is unchanged: only Ac
    // of 3d, Kh, Ac is unbeatable.
    approx(spec.feature("p_win_trick", &PASS), 0.0, "p_win pass");
    approx(spec.feature("expected_beaters", &PASS), 0.0, "beaters pass");
    approx(
        spec.feature("my_unbeatable_fraction", &PASS),
        1.0 / 3.0,
        "unbeatable pass",
    );

    // Playing 3d: beaters are every unseen card above it (all five); the
    // opponents' hands of 2 out of 5 cannot avoid... they can't: m = U.
    let d3 = play(&[card(Rank::Three, Suit::Diamonds)]);
    approx(spec.feature("p_win_trick", &d3), 0.0, "p_win 3d");
    approx(
        spec.feature("expected_beaters", &d3),
        (2.0 * 5.0 / 5.0 * 2.0) / 52.0,
        "beaters 3d",
    );
    // Left: Kh (Ks above) and Ac (unbeatable).
    approx(
        spec.feature("my_unbeatable_fraction", &d3),
        0.5,
        "unbeatable 3d",
    );
}

#[test]
fn an_opponent_locked_out_by_a_pass_ceiling_cannot_beat() {
    let mut spec = hidden_spec();
    // Seat 2 passed on a Queen of Diamonds earlier: it holds nothing above
    // a Queen, so Ks and Ad are not in its hand.
    spec.opponents[1].pass_ceilings = read_pass_ceilings(
        1,
        &[],
        &[(
            0,
            Combo::new(vec![card(Rank::Queen, Suit::Diamonds)]).unwrap(),
            0,
        )],
        RULE,
    )[0];
    let kh = play(&[card(Rank::King, Suit::Hearts)]);
    // Only seat 1 can still beat: 1 - 0.7.
    approx(
        spec.feature("p_win_trick", &kh),
        0.3,
        "p_win with a lock-out",
    );
    approx(
        spec.feature("expected_beaters", &kh),
        0.8 / 52.0,
        "beaters with a lock-out",
    );
    // Below the ceiling the lock-out proves nothing.
    let low = play(&[card(Rank::Three, Suit::Diamonds)]);
    approx(
        spec.feature("p_win_trick", &low),
        0.0,
        "p_win below the ceiling",
    );
    // Both locked out: the trick is won for sure.
    spec.opponents[0].pass_ceilings = spec.opponents[1].pass_ceilings;
    approx(spec.feature("p_win_trick", &kh), 1.0, "both locked out");
    approx(
        spec.feature("expected_beaters", &kh),
        0.0,
        "no beaters expected",
    );
}

#[test]
fn opponents_that_passed_or_left_do_not_count() {
    let mut spec = hidden_spec();
    spec.opponents = vec![opp(1, 2, false), opp(2, 0, false), opp(3, 3, false)];
    let kh = play(&[card(Rank::King, Suit::Hearts)]);
    approx(
        spec.feature("p_win_trick", &kh),
        1.0,
        "nobody left in the trick",
    );
    approx(
        spec.feature("expected_beaters", &kh),
        0.0,
        "no beaters expected",
    );
    spec.opponents = vec![opp(1, 2, true), opp(2, 3, false)];
    approx(
        spec.feature("p_win_trick", &kh),
        0.3,
        "one opponent in the trick",
    );
}

#[test]
fn an_empty_or_tiny_unseen_set_is_handled() {
    // Nothing unseen: nobody can beat, everything of mine is unbeatable.
    let mut spec = hidden_spec();
    spec.unseen.clear();
    spec.opponents = vec![opp(1, 2, true)];
    let kh = play(&[card(Rank::King, Suit::Hearts)]);
    approx(spec.feature("p_win_trick", &kh), 1.0, "empty unseen");
    approx(spec.feature("expected_beaters", &kh), 0.0, "empty unseen");
    approx(
        spec.feature("my_unbeatable_fraction", &kh),
        1.0,
        "empty unseen",
    );
    approx(
        spec.feature("my_unbeatable_fraction", &PASS),
        1.0,
        "empty unseen pass",
    );

    // The opponent holds every unseen card and all of them beat: certain.
    let mut spec = Spec::new(
        vec![card(Rank::Three, Suit::Diamonds)],
        vec![
            card(Rank::Ace, Suit::Diamonds),
            card(Rank::King, Suit::Diamonds),
        ],
        vec![opp(1, 2, true)],
    );
    let d3 = play(&[card(Rank::Three, Suit::Diamonds)]);
    approx(spec.feature("p_win_trick", &d3), 0.0, "U = 2, h = 2");
    approx(
        spec.feature("expected_beaters", &d3),
        2.0 * 2.0 / 2.0 / 52.0,
        "U = 2, h = 2",
    );
    // Playing the last card leaves an empty hand: fraction 0, not NaN.
    approx(
        spec.feature("my_unbeatable_fraction", &d3),
        0.0,
        "empty hand",
    );
    // One unseen card, one beater, a hand of one: certain loss.
    spec.unseen.truncate(1);
    spec.opponents = vec![opp(1, 1, true)];
    approx(spec.feature("p_win_trick", &d3), 0.0, "U = 1");
    // An opponent with an empty hand never beats (and is not in the trick).
    spec.opponents = vec![opp(1, 0, true)];
    approx(spec.feature("p_win_trick", &d3), 1.0, "h = 0");
}

#[test]
fn pairs_need_two_cards_of_a_group_and_double_decks_count_copies() {
    // Hand 9d 9h 3c; unseen Ks Kd Ad 4c 5d: only the Kings can beat a pair
    // of Nines (the Ace is a lone card), so m = 2 although 3 cards beat a
    // single Nine.
    let spec = Spec::new(
        vec![
            card(Rank::Nine, Suit::Diamonds),
            card(Rank::Nine, Suit::Hearts),
            card(Rank::Three, Suit::Clubs),
        ],
        vec![
            card(Rank::King, Suit::Spades),
            card(Rank::King, Suit::Diamonds),
            card(Rank::Ace, Suit::Diamonds),
            card(Rank::Four, Suit::Clubs),
            card(Rank::Five, Suit::Diamonds),
        ],
        vec![opp(1, 2, true), opp(2, 3, false)],
    );
    let pair = play(&[
        card(Rank::Nine, Suit::Diamonds),
        card(Rank::Nine, Suit::Hearts),
    ]);
    approx(spec.feature("p_win_trick", &pair), 0.3, "pair p_win");
    approx(
        spec.feature("expected_beaters", &pair),
        2.0 * 2.0 / 5.0 / 52.0,
        "pair beaters",
    );
    let single = play(&[card(Rank::Nine, Suit::Hearts)]);
    // Single: m = 3 (Ks Kd Ad); P = 1 - (2/5)(1/4) = 0.9.
    approx(spec.feature("p_win_trick", &single), 0.1, "single p_win");
    // Left after the pair: 3c, beatable.
    approx(
        spec.feature("my_unbeatable_fraction", &pair),
        0.0,
        "pair unbeatable",
    );

    // Double deck: two copies of the Ace of Diamonds are unseen; the group
    // of 2 beats a pair and a single alike (m = 2), and the table group says so.
    let mut double = Spec::new(
        vec![
            card_n(Rank::Nine, Suit::Diamonds, 0),
            card_n(Rank::Nine, Suit::Hearts, 1),
        ],
        vec![
            card_n(Rank::Ace, Suit::Diamonds, 2),
            card_n(Rank::Ace, Suit::Diamonds, 3),
            card_n(Rank::Four, Suit::Clubs, 4),
        ],
        vec![opp(1, 3, true)],
    );
    double.table.deck = DeckVariant::Double;
    approx(
        double.feature("table_double_deck", &PASS),
        1.0,
        "double deck",
    );
    let pair = play(&[
        card_n(Rank::Nine, Suit::Diamonds, 0),
        card_n(Rank::Nine, Suit::Hearts, 1),
    ]);
    // One opponent holds all three unseen cards: m = 2 of 3, h = 3.
    approx(
        double.feature("p_win_trick", &pair),
        0.0,
        "all unseen in one hand",
    );
    approx(
        double.feature("expected_beaters", &pair),
        3.0 * 2.0 / 3.0 / 52.0,
        "double beaters",
    );
    // The Nines are beatable by the Aces; an Ace pair of mine would not be.
    approx(
        double.feature("my_unbeatable_fraction", &PASS),
        0.0,
        "pair of Nines",
    );
}

// ---------------------------------------------------------- brute force

/// Whether a hand holds a play of `size` cards that beats a play topped by
/// `top`: some rank with at least `size` cards whose best card is above it.
fn can_beat(hand: &[Card], size: usize, top: Card) -> bool {
    engine::rank_groups(hand).iter().any(|group| {
        group.len() >= size
            && group
                .iter()
                .any(|c| c.compare(&top, RULE) == Ordering::Greater)
    })
}

/// Every way to deal `unseen` into hands of the given sizes (the sizes must
/// add up to the number of cards), as `(hands)`.
fn all_deals(unseen: &[Card], sizes: &[usize]) -> Vec<Vec<Vec<Card>>> {
    fn rec(
        rest: &[Card],
        sizes: &[usize],
        current: &mut Vec<Vec<Card>>,
        out: &mut Vec<Vec<Vec<Card>>>,
    ) {
        let Some((&size, later)) = sizes.split_first() else {
            out.push(current.clone());
            return;
        };
        // Choose `size` of `rest` by index masks.
        for mask in 0u32..(1 << rest.len()) {
            if mask.count_ones() as usize != size {
                continue;
            }
            let (mut taken, mut left) = (Vec::new(), Vec::new());
            for (i, &c) in rest.iter().enumerate() {
                if mask >> i & 1 == 1 {
                    taken.push(c);
                } else {
                    left.push(c);
                }
            }
            current.push(taken);
            rec(&left, later, current, out);
            current.pop();
        }
    }
    let mut out = Vec::new();
    rec(unseen, sizes, &mut Vec::new(), &mut out);
    out
}

/// The exact probability, over all equally likely deals of `unseen` into
/// hands of `sizes`, that none of the hands at `in_trick` can beat `top`.
fn exact_p_win(
    unseen: &[Card],
    sizes: &[usize],
    in_trick: &[usize],
    size: usize,
    top: Card,
) -> f64 {
    let deals = all_deals(unseen, sizes);
    let wins = deals
        .iter()
        .filter(|deal| in_trick.iter().all(|&i| !can_beat(&deal[i], size, top)))
        .count();
    #[allow(clippy::cast_precision_loss)]
    {
        wins as f64 / deals.len() as f64
    }
}

#[test]
fn p_win_trick_is_exact_for_singles_against_one_opponent() {
    // For a play of one card the formula counts exactly the cards that beat
    // it, and a single hand drawn without replacement is exactly the
    // hypergeometric draw it models.
    let unseen = vec![
        card(Rank::King, Suit::Spades),
        card(Rank::Ace, Suit::Diamonds),
        card(Rank::Four, Suit::Clubs),
        card(Rank::Five, Suit::Diamonds),
        card(Rank::Six, Suit::Diamonds),
    ];
    for hand_size in 1..=3 {
        for top in [
            card(Rank::Three, Suit::Diamonds),
            card(Rank::Five, Suit::Hearts),
            card(Rank::King, Suit::Hearts),
            card(Rank::Ace, Suit::Clubs),
        ] {
            // Seat 1 holds `hand_size` cards; the rest is with a passed seat.
            let spec = Spec::new(
                vec![top],
                unseen.clone(),
                vec![opp(1, hand_size, true), opp(2, 5 - hand_size, false)],
            );
            let exact = exact_p_win(&unseen, &[hand_size, 5 - hand_size], &[0], 1, top);
            approx(
                spec.feature("p_win_trick", &play(&[top])),
                exact,
                &format!("h = {hand_size}, top {top:?}"),
            );
        }
    }
}

#[test]
fn p_win_trick_is_an_approximation_for_pairs_and_for_several_opponents() {
    // Pairs: the formula counts the cards of groups that could beat, not the
    // combinations. Unseen Ks Kd Ad 4c 5d against a pair of Nines, seat 1
    // holding two cards: it beats only with both Kings, probability
    // 1 / C(5, 2) = 0.1, so it fails with 0.9; the formula's m = 2 says 0.3.
    let unseen = vec![
        card(Rank::King, Suit::Spades),
        card(Rank::King, Suit::Diamonds),
        card(Rank::Ace, Suit::Diamonds),
        card(Rank::Four, Suit::Clubs),
        card(Rank::Five, Suit::Diamonds),
    ];
    let nine_h = card(Rank::Nine, Suit::Hearts);
    let exact = exact_p_win(&unseen, &[2, 3], &[0], 2, nine_h);
    approx(exact, 0.9, "exact pair");
    let spec = Spec::new(
        vec![card(Rank::Nine, Suit::Diamonds), nine_h],
        unseen.clone(),
        vec![opp(1, 2, true), opp(2, 3, false)],
    );
    let pair = play(&[card(Rank::Nine, Suit::Diamonds), nine_h]);
    approx(spec.feature("p_win_trick", &pair), 0.3, "formula pair");
    assert!(
        spec.feature("p_win_trick", &pair) < exact,
        "for pairs the formula is pessimistic"
    );

    // Two opponents (2 + 2 cards, one card with a passed seat): the hands are
    // not independent. Beating Kh needs Ks or Ad; with a third seat holding
    // one card they cannot both be out of the trick hands, so the exact
    // chance of winning is 0, the independent product says 0.09.
    let unseen = vec![
        card(Rank::King, Suit::Spades),
        card(Rank::Ace, Suit::Diamonds),
        card(Rank::Four, Suit::Clubs),
        card(Rank::Five, Suit::Diamonds),
        card(Rank::Six, Suit::Diamonds),
    ];
    let kh = card(Rank::King, Suit::Hearts);
    let exact = exact_p_win(&unseen, &[2, 2, 1], &[0, 1], 1, kh);
    let spec = Spec::new(
        vec![kh],
        unseen,
        vec![opp(1, 2, true), opp(2, 2, true), opp(3, 1, false)],
    );
    approx(exact, 0.0, "exact two opponents");
    approx(
        spec.feature("p_win_trick", &play(&[kh])),
        0.09,
        "formula two opponents",
    );
}

// ---------------------------------------------------------------- oracle

/// The plain-loop reading of the `hidden` group (rank groups and scans).
fn reference_hidden(context: &TurnContext<'_>, candidate: &Move, rule: DuplicateRule) -> [f64; 3] {
    let unseen_groups = engine::rank_groups(&context.unseen_cards);
    let beats = |size: usize, top: Card| -> usize {
        unseen_groups
            .iter()
            .filter(|g| g.len() >= size && g.iter().any(|c| c.compare(&top, rule).is_gt()))
            .map(Vec::len)
            .sum()
    };
    let unbeatable_fraction = |remaining: &[Card]| -> f64 {
        let mut kept = 0;
        for group in engine::rank_groups(remaining) {
            let top = group
                .iter()
                .copied()
                .reduce(|a, b| if b.compare(&a, rule).is_gt() { b } else { a })
                .unwrap();
            if beats(group.len(), top) == 0 {
                kept += group.len();
            }
        }
        if remaining.is_empty() {
            0.0
        } else {
            count(kept) / count(remaining.len())
        }
    };
    let total = context.unseen_cards.len();
    match candidate {
        Move::Pass => [0.0, 0.0, unbeatable_fraction(context.hand)],
        Move::Play(combo) => {
            let top = combo.top_card(rule);
            let size = combo.size();
            let m = beats(size, top);
            let (mut p_win, mut expected) = (1.0, 0.0);
            for o in context.opponents.iter().filter(|o| o.active && o.in_trick) {
                if o.pass_ceilings.cannot_beat(size, top, rule) {
                    continue;
                }
                let mut miss = 1.0;
                let mut certain = false;
                for j in 0..o.hand_size {
                    if total.saturating_sub(j) <= m {
                        certain = true;
                        break;
                    }
                    miss *= count(total - j - m) / count(total - j);
                }
                let p_beat = if m == 0 || o.hand_size == 0 {
                    0.0
                } else if certain {
                    1.0
                } else {
                    1.0 - miss
                };
                p_win *= 1.0 - p_beat;
                if total > 0 {
                    expected += count(o.hand_size) * count(m) / count(total);
                }
            }
            let remaining: Vec<Card> = context
                .hand
                .iter()
                .filter(|c| !combo.cards().contains(c))
                .copied()
                .collect();
            [p_win, expected / 52.0, unbeatable_fraction(&remaining)]
        }
    }
}

/// On random states of real rounds the fast hidden group equals the plain
/// reading, every new feature is finite and in `[0, 3]` (hand sizes scaled by
/// a single-deck hand pass 1 in a double deck), the groups
/// are nested prefixes, and a context built for a set's own `needs` gives
/// the same features as the full one.
#[test]
fn the_new_groups_match_the_plain_reading_on_random_states() {
    use crate::hand_reading::PassTracker;
    use crate::match_runner::turn_context_for;
    use crate::strategy::ContextNeeds;
    use crate::test_support::for_each_state;

    let mut trackers: Vec<PassTracker> = Vec::new();
    let (mut vectors, mut informative) = (0usize, 0usize);
    let mut seen_trick_flags = [false; 2];
    for_each_state(17, 30, |state, _| {
        let round = state.round;
        if round.play_history().is_empty() && round.pass_history().is_empty() {
            trackers = FeatureSet::ALL_SETS
                .iter()
                .map(|_| PassTracker::new(usize::from(state.players), state.rule))
                .collect();
        }
        let seat = round.seat_to_move().unwrap();
        let mut full_tracker = PassTracker::new(usize::from(state.players), state.rule);
        let full = turn_context_for(
            round,
            seat,
            state.players,
            state.round_deck,
            &mut full_tracker,
            ContextNeeds::ALL,
        );
        let all = TurnSummary::with_features(&full, state.rule, FeatureSet::All);
        let legal = round.legal_moves();
        for (set, tracker) in FeatureSet::ALL_SETS.into_iter().zip(trackers.iter_mut()) {
            let reduced = turn_context_for(
                round,
                seat,
                state.players,
                state.round_deck,
                tracker,
                set.needs(),
            );
            let summary = TurnSummary::with_features(&reduced, state.rule, set);
            for candidate in &legal {
                let from_full = all.features(candidate);
                let from_needs = summary.features(candidate);
                assert_eq!(
                    bits(&from_full[..set.len()]),
                    bits(&from_needs[..set.len()]),
                    "{set}: a context built from needs() differs from the full one"
                );
                assert!(from_needs[set.len()..].iter().all(|&v| v == 0.0));
            }
        }
        for candidate in &legal {
            let features = all.features(candidate);
            let hidden = reference_hidden(&full, candidate, state.rule);
            for (k, expected) in hidden.iter().enumerate() {
                let actual = features[25 + k];
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "hidden[{k}]: {actual} vs {expected} for {candidate:?}"
                );
            }
            for (i, &v) in features.iter().enumerate() {
                assert!(v.is_finite(), "{} not finite", FEATURE_NAMES_V2[i]);
                if i >= 20 {
                    assert!(
                        (0.0..=3.0).contains(&v),
                        "{} = {v} out of range",
                        FEATURE_NAMES_V2[i]
                    );
                }
            }
            informative += usize::from(features[25] > 0.0 && features[25] < 1.0);
            seen_trick_flags[usize::from(features[24] > 0.5)] = true;
            vectors += 1;
        }
    });
    assert!(vectors > 50_000, "only {vectors}");
    assert!(
        informative > 1000,
        "p_win_trick was rarely strictly between 0 and 1"
    );
    assert_eq!(
        seen_trick_flags,
        [true, true],
        "last_to_act takes both values"
    );
}

#[test]
fn the_beat_probability_guards_degenerate_inputs() {
    use super::extra::beat_probability;
    approx(beat_probability(10, 0, 4), 0.0, "no beaters");
    approx(beat_probability(10, 3, 0), 0.0, "empty hand");
    approx(beat_probability(0, 0, 0), 0.0, "nothing at all");
    approx(beat_probability(4, 4, 2), 1.0, "all beaters");
    approx(beat_probability(5, 2, 4), 1.0, "cannot avoid the beaters");
    approx(
        beat_probability(3, 3, 7),
        1.0,
        "a hand larger than the unseen set",
    );
    approx(beat_probability(5, 2, 2), 0.7, "hand-computed");
    approx(
        beat_probability(52, 1, 13),
        0.25,
        "one beater in a quarter-sized hand",
    );
}
