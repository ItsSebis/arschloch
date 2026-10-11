//! Shared test helper: random but legal game states, taken from rounds
//! played with random moves (lots of passes, so pass ceilings appear).

use engine::{
    deal, lowest_card_holder, standard_deck, Card, DeckVariant, DuplicateRule, Move, Round,
};
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};

/// One decision point of a random round.
pub(crate) struct State<'a> {
    pub round: &'a Round,
    /// Every card dealt this round, numbered by `deal_index`.
    pub round_deck: &'a [Card],
    pub rule: DuplicateRule,
    pub players: u8,
}

/// Plays `rounds` random rounds for every combination of 3-6 players,
/// both decks and both duplicate rules, calling `visit` before each move.
pub(crate) fn for_each_state(
    seed: u64,
    rounds: usize,
    mut visit: impl FnMut(&State<'_>, &mut rand::rngs::StdRng),
) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    for variant in [DeckVariant::Single, DeckVariant::Double] {
        for rule in [DuplicateRule::FirstDealtWins, DuplicateRule::LastDealtWins] {
            for players in 3..=6u8 {
                for _ in 0..rounds {
                    let mut deck = standard_deck(variant);
                    deck.shuffle(&mut rng);
                    for (i, c) in deck.iter_mut().enumerate() {
                        c.deal_index = u8::try_from(i).unwrap();
                    }
                    let hands = deal(deck, players).unwrap();
                    let round_deck: Vec<Card> = hands.iter().flatten().copied().collect();
                    let leader = lowest_card_holder(&hands, rule).unwrap();
                    let mut round = Round::new(hands, rule, leader).unwrap();
                    while !round.is_complete() {
                        visit(
                            &State {
                                round: &round,
                                round_deck: &round_deck,
                                rule,
                                players,
                            },
                            &mut rng,
                        );
                        let seat = round.seat_to_move().unwrap();
                        let moves = round.legal_moves();
                        let chosen = if moves.contains(&Move::Pass) && rng.random_bool(0.4) {
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
}
