# The endgame solver

A player that does not guess: once few cards are unseen it considers **every way the unseen cards could be
split between the other players** and picks the move with the best expected finishing place. Design:
`docs/superpowers/specs/2026-10-16-endgame-solver-design.md`; measurements: `docs/baselines/solver/README.md`.

## What it computes, and what it does not

For one seat at its turn, the solver knows its own hand, every card played so far, each opponent's hand size and
what the passes revealed (`sim::hand_reading`: an opponent that passed on a 9 holds nothing above it). It lists
every split of the unseen cards that fits this, solves each split exactly (every seat takes the move that gives
*itself* the best place, "max^n"), and averages per move. The result is the best move **in expectation over what
the opponents could hold**. It is not a proof against one specific hidden hand, and it does not bluff or hide
information. It supports `--pass-rule final` (the rules of the game) only; under `free` the strategy plays its
fallback.

## As a player: `endgame-solver`

`--strategy endgame-solver` (or `endgame-solver:OPTIONS`, comma separated) works wherever a strategy spec does:

| Option | Default | Meaning |
|---|---|---|
| `max-unseen=N` | 6 | solve only when at most N cards are unseen; before that the fallback plays |
| `budget=N` | 2000000 | search nodes allowed per decision; beyond it the fallback plays |
| `fallback=NAME` | `adaptive` | `lowest-legal`, `greedy-highest`, `hold-back-pairs`, `card-counter`, `endgame-denial` or `adaptive` |

It reports as `EndgameSolver(<fallback>,T=<max-unseen>)`. The exchange is always the fallback's choice.

## Explaining a situation: `cli solve FILE`

`cli solve [--max-unseen N] [--budget N] FILE` reads a position and prints every legal move with its expected
place. The file has one `key: value` per line (`#` starts a comment):

```
players: 4              # 3-6
deck: single            # single | double
duplicate: first        # first | last: which of two identical cards wins; default first
seat: 0                 # the seat to move
hand: 5C 9D KH          # our cards (2-10, J, Q, K, A, then D H S C)
sizes: 3 2 2 2          # cards held by every seat in seat order (0 = already out)
table: lead             # `lead`, or the combo on the table and who holds it: pair 8H 8S by 2
passed: 3               # seats that passed in this trick (only with a combo on the table); every seat that acts
                        # between the holder and us must be listed here (or be out)
played: 2C 2H ...       # every card played so far this round, not counting `table`
```

The unseen cards are the deck minus `hand`, `played` and `table`; their number must equal the other seats' hand
sizes added up (otherwise the command says so). Identical cards of a double deck are numbered in the order
listed. `docs/examples/solve-endgame.txt` is a worked position; its output:

```
seat 0 to move, hand: 5C 9D KH
deals considered: 90 (367127 search nodes)
move                   exp.place    score best/worst   best in
5C                          1.00    +1.00      1/1        100%
9D                          1.00    +1.00      1/1        100%
KH                          2.40    +0.07      2/3          0%
```

`exp.place` is the expected finishing place (1 = first out) over all consistent deals, `score` the same on the
+1 (first) to -1 (last) scale of the statistics, `best/worst` the best and worst place over the deals, `best in`
the share of deals in which the move is (one of) the best. Here the solver keeps the unbeatable `KH` and leads
a low card: leading `KH` first wins the trick but gives the lead away at the wrong moment.

## Where do other strategies go wrong: `cli solver-gap`

`cli solver-gap --strategy SPEC ... [--player-count N] [--deck single|double] [--games N] [--seed N] [--max-unseen N]
[--budget N]` plays single rounds with the given strategies (one per seat, rotated each game) and, at every
decision the solver can solve, compares the move taken with the solver's best. For each strategy it prints
how many decisions were solved, how many differ, the places lost per decision, and the costliest decisions in words,
for example `seat 2, hand [10S 4H 3C], leading, 5 unseen: Adaptive(counting,reading) plays 10S, solver prefers 3C (costs 2.00 places)`.
Those are the lessons; `docs/baselines/solver/README.md` has the first results.

## Limits

- Cost grows quickly with the number of unseen cards; `max-unseen` and `budget` bound it. Measured defaults: 6 unseen cards, 2M nodes per decision (T=8 already takes minutes per 200 games). At T=6 the solver plays exactly about 11% of the decisions of a 4-player single-deck round.
- It beats the strategy it falls back to clearly (+0.14 score head to head) but not the trained neat champion, whose strength lies in the part of the round the solver does not play; see the measurements.
- The unseen cards are weighted equally; hand reading only removes impossible splits. Pass ceilings are used as hard
  facts: a seat that passed on purpose while holding a beating card (`adaptive`'s deception, `hold-back-pairs`) makes
  the real split look impossible until that seat plays above its ceiling. The solver then solves the wrong splits, or
  finds none and plays its fallback; this is counted in `SolverStats::fell_back_other`. A switch to ignore ceilings
  (the spec's `use_pass_ceilings`) is not built.
- A double deck has identical cards that are told apart by deal order, which a written-down position cannot show:
  `cli solve` numbers your copies first and says so when it matters.
- `endgame-solver:max-unseen=N` with a large N is bounded by the node budget per decision, not by N; the enumeration
  of splits prunes contradictory pass ceilings early.
- A later phase (the spec's phase B) will sample splits to search the early game as well.
