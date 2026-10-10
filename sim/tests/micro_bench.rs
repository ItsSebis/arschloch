//! Micro-benchmarks, ignored by default (they measure, they do not assert):
//!
//! ```text
//! cargo test --release -p sim --test micro_bench -- --ignored --nocapture --test-threads=1
//! ```
//!
//! They print nanoseconds per call for `Round::legal_moves` and for
//! `TurnSummary::new` plus `features` on fixed, deterministic states (a
//! seeded shuffle, so every run times the same hands). Compare the numbers
//! before and after a change to the hot path; only the same machine in a
//! quiet state is comparable. `docs/baselines/perf/README.md` has the
//! recorded numbers.

#![allow(clippy::cast_precision_loss)] // counts are far below 2^52

use std::hint::black_box;
use std::time::Instant;

use engine::{deal, standard_deck, Card, DeckVariant, DuplicateRule, Move, Round, SeatId};
use sim::strategies::TurnSummary;
use sim::{OpponentHand, PassCeilings, TurnContext};

const RULE: DuplicateRule = DuplicateRule::FirstDealtWins;
const STATES_PER_SETUP: u64 = 24;

/// xorshift64*: a fixed, dependency-free source of a reproducible shuffle.
fn next(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

fn shuffled(variant: DeckVariant, seed: u64) -> Vec<Card> {
    let mut deck = standard_deck(variant);
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    for i in (1..deck.len()).rev() {
        let j = usize::try_from(next(&mut state) % (i as u64 + 1)).unwrap();
        deck.swap(i, j);
    }
    deck
}

/// A round at the first turn (the seat leads) or one move later (the next
/// seat follows the lowest single or the first play on offer).
fn round(variant: DeckVariant, players: u8, seed: u64, following: bool) -> Round {
    let hands = deal(shuffled(variant, seed), players).unwrap();
    let mut round = Round::new(hands, RULE, 0).unwrap();
    if following {
        let seat = round.seat_to_move().unwrap();
        let lead = round
            .legal_moves()
            .into_iter()
            .find(|m| matches!(m, Move::Play(_)))
            .unwrap();
        round.submit_move(seat, lead).unwrap();
    }
    round
}

struct Setup {
    label: &'static str,
    variant: DeckVariant,
    players: u8,
    following: bool,
}

const SETUPS: [Setup; 4] = [
    Setup {
        label: "single deck, 4 players, leading",
        variant: DeckVariant::Single,
        players: 4,
        following: false,
    },
    Setup {
        label: "single deck, 4 players, following",
        variant: DeckVariant::Single,
        players: 4,
        following: true,
    },
    Setup {
        label: "double deck, 5 players, leading",
        variant: DeckVariant::Double,
        players: 5,
        following: false,
    },
    Setup {
        label: "double deck, 5 players, following",
        variant: DeckVariant::Double,
        players: 5,
        following: true,
    },
];

fn rounds(setup: &Setup) -> Vec<Round> {
    (0..STATES_PER_SETUP)
        .map(|seed| round(setup.variant, setup.players, seed, setup.following))
        .collect()
}

/// Runs `body` over `states` until about 0.4 s have passed, then returns ns
/// per call (a call is one pass over one state's worth of work).
fn ns_per_call<T>(states: &[T], mut body: impl FnMut(&T) -> usize) -> (f64, f64) {
    let mut sink = 0usize;
    for state in states {
        sink = sink.wrapping_add(body(state)); // warm-up
    }
    let started = Instant::now();
    let mut calls = 0u64;
    while started.elapsed().as_secs_f64() < 0.4 {
        for state in states {
            sink = sink.wrapping_add(body(black_box(state)));
            calls += 1;
        }
    }
    black_box(sink);
    let ns = started.elapsed().as_secs_f64() * 1e9 / calls as f64;
    (ns, calls as f64)
}

#[test]
#[ignore = "micro-benchmark: run with --ignored --nocapture"]
fn legal_moves_ns_per_call() {
    println!("\nRound::legal_moves (mean over {STATES_PER_SETUP} fixed states)");
    for setup in &SETUPS {
        let states = rounds(setup);
        let moves = states.iter().map(|r| r.legal_moves().len()).sum::<usize>() as f64
            / states.len() as f64;
        let (ns, _) = ns_per_call(&states, |round| round.legal_moves().len());
        println!(
            "  {:<36} {ns:>9.0} ns/call  ({moves:.1} moves, hand {})",
            setup.label,
            states[0].hand(states[0].seat_to_move().unwrap()).len()
        );
    }
}

struct Turn {
    seat: SeatId,
    round: Round,
    unseen: Vec<Card>,
    opponents: Vec<OpponentHand>,
}

impl Turn {
    fn new(round: Round, count: u8) -> Self {
        let seat = round.seat_to_move().unwrap();
        let unseen = (0..count)
            .filter(|&s| s != seat)
            .flat_map(|s| round.hand(s).to_vec())
            .collect();
        let opponents = (0..count)
            .filter(|&s| s != seat)
            .map(|s| OpponentHand {
                seat: s,
                hand_size: round.hand_size(s),
                active: true,
                pass_ceilings: PassCeilings::default(),
            })
            .collect();
        Self {
            seat,
            round,
            unseen,
            opponents,
        }
    }

    fn context(&self) -> TurnContext<'_> {
        TurnContext {
            seat: self.seat,
            hand: self.round.hand(self.seat),
            opponents: self.opponents.clone(),
            unseen_cards: self.unseen.clone(),
            own_pass_ceilings: PassCeilings::default(),
            current_combo: self.round.current_combo(),
        }
    }
}

#[test]
#[ignore = "micro-benchmark: run with --ignored --nocapture"]
fn turn_summary_ns_per_call() {
    println!("\nTurnSummary::new, and features() per legal move (mean over {STATES_PER_SETUP} fixed states)");
    for setup in &SETUPS {
        let turns: Vec<Turn> = rounds(setup)
            .into_iter()
            .map(|r| Turn::new(r, setup.players))
            .collect();
        let (new_ns, _) = ns_per_call(&turns, |turn| {
            let context = turn.context();
            let summary = TurnSummary::new(&context, RULE);
            black_box(&summary);
            0
        });
        let (all_ns, _) = ns_per_call(&turns, |turn| {
            let context = turn.context();
            let summary = TurnSummary::new(&context, RULE);
            let moves = turn.round.legal_moves();
            moves.iter().map(|m| summary.features(m).len()).sum()
        });
        let (legal_ns, _) = ns_per_call(&turns, |turn| turn.round.legal_moves().len());
        println!(
            "  {:<36} new {new_ns:>7.0} ns   new + legal_moves + features of every move {all_ns:>8.0} ns (legal_moves alone {legal_ns:.0} ns)",
            setup.label
        );
    }
}
