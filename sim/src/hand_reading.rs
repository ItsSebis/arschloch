//! Pass-based hand reading (docs/ROADMAP.md, Phase 7).
//!
//! A seat that passed against a size-`s` combo topped by `c` held no
//! size-`s` combo topped above `c` at that moment; hands only shrink
//! (no draw pile), so that stays true for the rest of the round. By
//! upward closure it also holds no *larger* combo topped above `c` (any
//! such combo contains a size-`s` sub-combo with the same top card), so
//! a seat's ceiling at size `s` is the lowest top card it has passed
//! against at any size `<= s`. Ceilings are `Card`s, not `Rank`s:
//! passing on 9♣ says nothing about 9♦ — the same lesson `CardCounter`
//! learned the hard way (see its own doc comment).
//!
//! **Refutation.** Not every pass is honest — `HoldBackPairs` passes on
//! a single when every beating play would split a pair, `RandomLegal`
//! passes at random, and `Adaptive`'s deception modifier passes on
//! purpose. A pass is never recorded (the reverse sweep checks
//! refutation before ever writing a ceiling, so a refuted pass is
//! simply skipped) if the same seat later plays a combo of the same or
//! larger size topped above the passed-on card — proof it held a
//! beater at the time. An honest pass can never be refuted, since hands
//! only shrink; this only ever removes false information, never true
//! information.

use std::cmp::Ordering;

use engine::{Card, Combo, DuplicateRule, SeatId};

/// The largest combo any seat can field: 8 copies of one rank in the
/// double deck.
pub const MAX_COMBO_SIZE: usize = 8;

/// Per-size "can't beat" facts about one seat, derived from its
/// unrefuted passes. `by_size[s - 1]` is the lowest top card passed
/// against at exactly size `s`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PassCeilings {
    by_size: [Option<Card>; MAX_COMBO_SIZE],
}

impl PassCeilings {
    /// The lowest top card this seat is known unable to beat at
    /// `size`, applying upward closure. `None`: nothing is known at
    /// this size.
    #[must_use]
    pub fn ceiling(&self, size: usize, duplicate_rule: DuplicateRule) -> Option<Card> {
        self.by_size
            .iter()
            .take(size)
            .flatten()
            .copied()
            .min_by(|a, b| a.compare(b, duplicate_rule))
    }

    /// Whether this seat is known unable to beat a size-`size` combo
    /// topped by `top`.
    ///
    /// A pass against ceiling `c` means the seat held nothing that beats
    /// `c`. For any `top` with `top >= c` under `Card::compare` (a real
    /// transitive total order over rank, then suit, then the
    /// duplicate-deal tiebreak), if the seat held something beating
    /// `top` it would also beat `c` by transitivity — contradicting the
    /// pass. So "cannot beat `c`" soundly extends to every `top >= c`,
    /// including same-rank higher-suit cards: `Card::compare`'s suit
    /// tiebreak (this game's `Suit` order is Diamonds < Hearts < Spades
    /// < Clubs) is a real difference in how hard two same-rank cards are
    /// to beat, not an arbitrary one — `Combo::beats` uses it to decide
    /// legality, so a King of Clubs genuinely is harder to beat than a
    /// King of Hearts (only an Ace beats it). A pass on the King of
    /// Hearts therefore also covers the King of Clubs, but says nothing
    /// about the King of Diamonds (see the module doc's 9♣-vs-9♦
    /// example).
    #[must_use]
    pub fn cannot_beat(&self, size: usize, top: Card, duplicate_rule: DuplicateRule) -> bool {
        self.ceiling(size, duplicate_rule)
            .is_some_and(|c| top.compare(&c, duplicate_rule) != Ordering::Less)
    }
}

/// Reduces a round's play/pass history to one `PassCeilings` per seat,
/// in a single reverse sweep: walking backward, `highest_later[seat]`
/// at index `s - 1` is the highest top card `seat` played at size
/// `>= s` *after* the pass currently being examined — exactly the
/// refutation test.
///
/// `pass_history` entries are `(seat, combo, plays_before)` exactly as
/// `engine::Round::pass_history()` returns them (`plays_before` is
/// `play_history.len()` at the moment of that pass).
#[must_use]
pub fn read_pass_ceilings(
    player_count: usize,
    play_history: &[(SeatId, Combo)],
    pass_history: &[(SeatId, Combo, usize)],
    duplicate_rule: DuplicateRule,
) -> Vec<PassCeilings> {
    let mut ceilings = vec![PassCeilings::default(); player_count];
    let mut highest_later = vec![[None::<Card>; MAX_COMBO_SIZE]; player_count];
    let mut play_index = play_history.len();

    for (seat, combo, plays_before) in pass_history.iter().rev() {
        while play_index > *plays_before {
            play_index -= 1;
            let (play_seat, play_combo) = &play_history[play_index];
            let top = play_combo.top_card(duplicate_rule);
            for slot in &mut highest_later[usize::from(*play_seat)][..play_combo.size()] {
                if slot.is_none_or(|h| top.compare(&h, duplicate_rule) == Ordering::Greater) {
                    *slot = Some(top);
                }
            }
        }

        let seat_idx = usize::from(*seat);
        let size = combo.size();
        let passed_top = combo.top_card(duplicate_rule);
        let refuted = highest_later[seat_idx][size - 1]
            .is_some_and(|h| h.compare(&passed_top, duplicate_rule) == Ordering::Greater);
        if !refuted {
            let slot = &mut ceilings[seat_idx].by_size[size - 1];
            if slot.is_none_or(|c| passed_top.compare(&c, duplicate_rule) == Ordering::Less) {
                *slot = Some(passed_top);
            }
        }
    }
    ceilings
}

/// `read_pass_ceilings`, computed incrementally: the match loop reads
/// the ceilings before every turn, and re-sweeping the whole history each
/// time made that the largest cost of simulating a game. The tracker
/// ingests each play and pass once, in the order they happened, keeping
/// per seat only the passes not (yet) refuted; a play by a seat refutes
/// its earlier passes at the same or a smaller size that it topped.
/// Always equal to `read_pass_ceilings` on the same history (tested move
/// by move on random rounds).
#[derive(Debug, Clone)]
pub struct PassTracker {
    duplicate_rule: DuplicateRule,
    plays_seen: usize,
    passes_seen: usize,
    /// Per seat: `(size, top card)` of every pass not refuted so far.
    open: Vec<Vec<(usize, Card)>>,
    ceilings: Vec<PassCeilings>,
}

impl PassTracker {
    #[must_use]
    pub fn new(player_count: usize, duplicate_rule: DuplicateRule) -> Self {
        Self {
            duplicate_rule,
            plays_seen: 0,
            passes_seen: 0,
            open: vec![Vec::new(); player_count],
            ceilings: vec![PassCeilings::default(); player_count],
        }
    }

    /// Takes in whatever was added to the round's histories since the
    /// last call (both slices are the round's complete histories).
    pub fn update(
        &mut self,
        play_history: &[(SeatId, Combo)],
        pass_history: &[(SeatId, Combo, usize)],
    ) {
        let rule = self.duplicate_rule;
        loop {
            // A pass tagged `plays_before` happened after that many plays
            // and before the next one.
            let next_pass = pass_history
                .get(self.passes_seen)
                .filter(|(_, _, plays_before)| *plays_before <= self.plays_seen);
            if let Some((seat, combo, _)) = next_pass {
                let seat = usize::from(*seat);
                self.open[seat].push((combo.size(), combo.top_card(rule)));
                self.refresh(seat);
                self.passes_seen += 1;
            } else if let Some((seat, combo)) = play_history.get(self.plays_seen) {
                let (size, top) = (combo.size(), combo.top_card(rule));
                let seat = usize::from(*seat);
                let before = self.open[seat].len();
                self.open[seat].retain(|&(pass_size, pass_top)| {
                    !(pass_size <= size && top.compare(&pass_top, rule) == Ordering::Greater)
                });
                if self.open[seat].len() != before {
                    self.refresh(seat);
                }
                self.plays_seen += 1;
            } else {
                break;
            }
        }
    }

    fn refresh(&mut self, seat: usize) {
        let rule = self.duplicate_rule;
        let mut ceilings = PassCeilings::default();
        for &(size, top) in &self.open[seat] {
            let slot = &mut ceilings.by_size[size - 1];
            if slot.is_none_or(|c| top.compare(&c, rule) == Ordering::Less) {
                *slot = Some(top);
            }
        }
        self.ceilings[seat] = ceilings;
    }

    /// One `PassCeilings` per seat, as of the histories last `update`d.
    #[must_use]
    pub fn ceilings(&self) -> &[PassCeilings] {
        &self.ceilings
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::{Rank, Suit};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit, 0)
    }

    fn combo(cards: Vec<Card>) -> Combo {
        Combo::new(cards).unwrap()
    }

    /// Plays whole rounds with random legal moves (so many passes, and
    /// passes that later turn out dishonest) and checks, after every
    /// move, that the incremental tracker equals the reverse sweep.
    #[test]
    fn the_tracker_equals_the_reverse_sweep_after_every_move() {
        use engine::{deal, lowest_card_holder, standard_deck, DeckVariant, Move, Round};
        use rand::seq::SliceRandom;
        use rand::{RngExt, SeedableRng};

        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        for (variant, rule) in [
            (DeckVariant::Single, DuplicateRule::FirstDealtWins),
            (DeckVariant::Double, DuplicateRule::LastDealtWins),
        ] {
            for players in 3..=6u8 {
                for _ in 0..40 {
                    let mut deck = standard_deck(variant);
                    deck.shuffle(&mut rng);
                    for (i, c) in deck.iter_mut().enumerate() {
                        c.deal_index = u8::try_from(i).unwrap();
                    }
                    let hands = deal(deck, players).unwrap();
                    let leader = lowest_card_holder(&hands, rule).unwrap();
                    let mut round = Round::new(hands, rule, leader).unwrap();
                    let mut tracker = PassTracker::new(usize::from(players), rule);
                    while !round.is_complete() {
                        tracker.update(round.play_history(), round.pass_history());
                        let expected = read_pass_ceilings(
                            usize::from(players),
                            round.play_history(),
                            round.pass_history(),
                            rule,
                        );
                        assert_eq!(tracker.ceilings(), &expected[..]);
                        let seat = round.seat_to_move().unwrap();
                        let moves = round.legal_moves();
                        // Passing a lot makes refutable passes common.
                        let pass = moves.contains(&Move::Pass) && rng.random_bool(0.5);
                        let chosen = if pass {
                            Move::Pass
                        } else {
                            moves[rng.random_range(0..moves.len())]
                        };
                        round.submit_move(seat, chosen).unwrap();
                    }
                }
            }
        }
    }

    #[test]
    fn a_pass_at_size_one_bounds_size_two_as_well() {
        let pass_history = vec![(0u8, combo(vec![card(Rank::Nine, Suit::Diamonds)]), 0)];
        let ceilings = read_pass_ceilings(2, &[], &pass_history, DuplicateRule::FirstDealtWins);
        let nine = card(Rank::Nine, Suit::Diamonds);
        assert_eq!(
            ceilings[0].ceiling(1, DuplicateRule::FirstDealtWins),
            Some(nine)
        );
        assert_eq!(
            ceilings[0].ceiling(2, DuplicateRule::FirstDealtWins),
            Some(nine)
        );
    }

    #[test]
    fn a_pass_at_size_two_says_nothing_about_size_one() {
        let pair = combo(vec![
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Seven, Suit::Diamonds),
        ]);
        let pass_history = vec![(0u8, pair, 0)];
        let ceilings = read_pass_ceilings(2, &[], &pass_history, DuplicateRule::FirstDealtWins);
        assert_eq!(ceilings[0].ceiling(1, DuplicateRule::FirstDealtWins), None);
    }

    #[test]
    fn suit_precision_a_pass_covers_higher_suits_but_not_lower_suits_at_the_same_rank() {
        // engine's Suit order: Diamonds < Hearts < Spades < Clubs.
        let pass_history = vec![(0u8, combo(vec![card(Rank::King, Suit::Hearts)]), 0)];
        let ceilings = read_pass_ceilings(2, &[], &pass_history, DuplicateRule::FirstDealtWins);
        let king_hearts = card(Rank::King, Suit::Hearts);
        let king_clubs = card(Rank::King, Suit::Clubs);
        let king_diamonds = card(Rank::King, Suit::Diamonds);
        assert!(
            ceilings[0].cannot_beat(1, king_hearts, DuplicateRule::FirstDealtWins),
            "a pass covers itself"
        );
        assert!(
            ceilings[0].cannot_beat(1, king_clubs, DuplicateRule::FirstDealtWins),
            "Clubs is the higher suit, so King-of-Clubs >= King-of-Hearts and is covered"
        );
        assert!(
            !ceilings[0].cannot_beat(1, king_diamonds, DuplicateRule::FirstDealtWins),
            "Diamonds is the LOWER suit, so this must NOT be covered \
             (a Rank-keyed-only ceiling would wrongly say it is)"
        );
    }

    #[test]
    fn a_later_play_above_the_passed_card_refutes_the_pass() {
        let nine = combo(vec![card(Rank::Nine, Suit::Diamonds)]);
        let king = combo(vec![card(Rank::King, Suit::Clubs)]);
        let pass_history = vec![(0u8, nine, 0)]; // plays_before: 0
        let play_history_refuting = vec![(0u8, king)]; // this play is index 0, so plays_before(0) <= 0 means it happened AFTER
        let ceilings = read_pass_ceilings(
            2,
            &play_history_refuting,
            &pass_history,
            DuplicateRule::FirstDealtWins,
        );
        assert_eq!(
            ceilings[0].ceiling(1, DuplicateRule::FirstDealtWins),
            None,
            "seat 0 later played a King, proving the earlier pass was not honest"
        );
    }

    #[test]
    fn a_play_recorded_before_the_pass_does_not_refute_it() {
        let king = combo(vec![card(Rank::King, Suit::Clubs)]);
        let nine = combo(vec![card(Rank::Nine, Suit::Diamonds)]);
        let play_history = vec![(0u8, king)]; // index 0, happened before
        let pass_history = vec![(0u8, nine, 1)]; // plays_before: 1 (i.e. after that one play)
        let ceilings = read_pass_ceilings(
            2,
            &play_history,
            &pass_history,
            DuplicateRule::FirstDealtWins,
        );
        assert_eq!(
            ceilings[0].ceiling(1, DuplicateRule::FirstDealtWins),
            Some(card(Rank::Nine, Suit::Diamonds)),
            "the King was played BEFORE this pass, so it doesn't refute it"
        );
    }

    #[test]
    fn a_later_play_at_a_smaller_size_does_not_refute_a_larger_pass() {
        let pair = combo(vec![
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Seven, Suit::Diamonds),
        ]);
        let king_single = combo(vec![card(Rank::King, Suit::Clubs)]);
        let pass_history = vec![(0u8, pair, 0)]; // pass at size 2, plays_before: 0
        let play_history = vec![(0u8, king_single)]; // size-1 play, index 0, happened after
        let ceilings = read_pass_ceilings(
            2,
            &play_history,
            &pass_history,
            DuplicateRule::FirstDealtWins,
        );
        assert_eq!(
            ceilings[0].ceiling(2, DuplicateRule::FirstDealtWins),
            Some(pair.top_card(DuplicateRule::FirstDealtWins)),
            "a size-1 play, even above the passed rank, cannot refute a size-2 pass"
        );
    }

    #[test]
    fn a_later_play_by_a_different_seat_does_not_refute_this_seats_pass() {
        let nine = combo(vec![card(Rank::Nine, Suit::Diamonds)]);
        let king = combo(vec![card(Rank::King, Suit::Clubs)]);
        let pass_history = vec![(0u8, nine, 0)];
        let play_history = vec![(1u8, king)]; // seat 1 plays, not seat 0
        let ceilings = read_pass_ceilings(
            2,
            &play_history,
            &pass_history,
            DuplicateRule::FirstDealtWins,
        );
        assert_eq!(
            ceilings[0].ceiling(1, DuplicateRule::FirstDealtWins),
            Some(card(Rank::Nine, Suit::Diamonds)),
            "seat 1's play cannot refute seat 0's pass"
        );
    }

    #[test]
    fn a_later_play_lower_than_the_passed_card_does_not_refute_the_pass() {
        let king = combo(vec![card(Rank::King, Suit::Clubs)]);
        let nine = combo(vec![card(Rank::Nine, Suit::Diamonds)]);
        let pass_history = vec![(0u8, king, 0)];
        let play_history = vec![(0u8, nine)]; // lower than the passed King, happened after
        let ceilings = read_pass_ceilings(
            2,
            &play_history,
            &pass_history,
            DuplicateRule::FirstDealtWins,
        );
        assert_eq!(
            ceilings[0].ceiling(1, DuplicateRule::FirstDealtWins),
            Some(card(Rank::King, Suit::Clubs)),
            "a later play lower than the passed card cannot refute the pass"
        );
    }

    #[test]
    fn ceilings_are_tracked_independently_per_seat() {
        let pass_history = vec![
            (0u8, combo(vec![card(Rank::Nine, Suit::Diamonds)]), 0),
            (1u8, combo(vec![card(Rank::Six, Suit::Diamonds)]), 0),
        ];
        let ceilings = read_pass_ceilings(2, &[], &pass_history, DuplicateRule::FirstDealtWins);
        assert_eq!(
            ceilings[0].ceiling(1, DuplicateRule::FirstDealtWins),
            Some(card(Rank::Nine, Suit::Diamonds))
        );
        assert_eq!(
            ceilings[1].ceiling(1, DuplicateRule::FirstDealtWins),
            Some(card(Rank::Six, Suit::Diamonds))
        );
    }
}
