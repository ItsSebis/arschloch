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

## As a toggle on any strategy: `SPEC+solver`

Add `+solver` to any strategy spec and that strategy plays as before until at most `N` cards are unseen (default 6),
then the solver plays the rest of the round: `adaptive+solver`, `adaptive:reading,tempo+solver:8`,
`neat:docs/baselines/neat-v2/champion.json+solver`, `lowest-legal+solver:max-unseen=4,budget=500000`. `+solver:N` is
`max-unseen=N`; `budget=N` is the only other option (the wrapped strategy is the fallback). It works wherever a strategy
spec does: `--strategy`, `cli evaluate`, `cli play`, `train --opponent`, `train --auto --opponent`, the strategy rows of
the desktop window (a tick box per seat) and the web play page (a "solver" box per opponent; the server takes ids such as
`lowest-legal+solver` or `model:champion-v2+solver:4`). It reports as `<name>+Solver(T=N)`.

- **Before the hand-off** the wrapped strategy decides every move, forced ones too, with the same random stream as
  alone: with `max-unseen=0` a wrapped strategy plays move for move like the bare one (tested, 3 strategies including
  one that draws random numbers on every decision).
- **The exchange** (which cards a loser gives) is always the wrapped strategy's.
- **Per-round state** needs nothing extra: the pass tracker (what the passes revealed) lives in the match runner and
  catches up on the whole round's history at each request, so the wrapped strategy and the solver both see every
  event, also those that happened while the other one was playing. A wrapped strategy asks for the full context
  (`ContextNeeds::ALL`), which costs a little time in the early game (see the measurements).
- **`--pass-rule free`**: the solver supports `final` only, so under `free` the wrapper never engages and the wrapped
  strategy plays the whole round (tested: identical to the bare strategy).
- Over budget or no consistent split of the unseen cards: the wrapped strategy plays that decision, as with
  `endgame-solver`.
- `endgame-solver[:OPTIONS]` is unchanged (same output byte for byte); `endgame-solver+solver` and `X+solver+solver`
  are errors.
- **Training pools:** `train --opponent adaptive+solver` works (and `neat:PATH+solver` is frozen into the run
  directory like a plain genome). The solver makes the opponent's endgame exact but slow: see
  `docs/baselines/solver-toggle/README.md` for the cost per match. Training a *genome* with the solver finishing its
  own rounds is not built (a genome is one of the `neat` strategies, the toggle is only on opponents and in play).

## The early game: the sampler (phase B), `samples=N`

Both forms above also take the options of the early-game sampler. With `samples=N` the solver does not hand the early
game to the fallback: at each decision with more than `max-unseen` unseen cards it **samples** N splits of the unseen
cards that fit what the seat knows (hand sizes, and the pass ceilings unless `ceilings=off`), searches each split to a
bounded depth with max^n (every seat takes its own best place), values the positions at the depth limit with a **leaf
evaluator**, and plays the move with the best mean place over the samples ("perfect-information Monte Carlo"). Once at
most `max-unseen` cards are unseen, the exact solver plays as before. Design:
`docs/superpowers/specs/2026-10-17-solver-phase-b-design.md`; measurements: `docs/baselines/solver-phase-b/README.md`.

| Option | Default | Meaning |
|---|---|---|
| `samples=N` | off | sampled splits per early-game decision; without it (or `samples=0`) nothing changes |
| `depth=D` | 1 | plies searched in each split (1 to 4); 1 = each own move, then the leaf. Forced moves do not count |
| `leaf=E` | `adaptive` | `adaptive`: play the round out with `adaptive` in every seat; `cards`: a static guess from the cards left (no play-out, cheap, crude); `neat:PATH`: play the round out with a trained genome in every seat |
| `rollouts=N` | 20000 | leaf evaluations allowed per decision; sampling stops after the sample that crosses it |
| `ceilings=M` | `hard` | pass ceilings: `hard` (a split must respect them), `soft` (respect them while some split does, else ignore them), `off` (ignore them). Also applies to the exact endgame |

Examples: `endgame-solver:samples=16`, `adaptive+solver:samples=16,depth=2`, the sampler alone without the exact
endgame `adaptive+solver:max-unseen=0,samples=16`, and the champion searching with its own network at the leaves
`neat:docs/baselines/neat-v2/champion.json+solver:samples=8,leaf=neat:docs/baselines/neat-v2/champion.json`. The name
shows the settings: `Adaptive(counting,reading)+Solver(T=6,S=16,D=1,adaptive)`.

- **Deterministic:** when the sampler plays a decision it draws one number from the match's random stream and derives
  every sample and play-out from it, so a seed gives the same match at any thread count. Without `samples` nothing is
  drawn and the solver plays byte for byte as before.
- **Sampling:** each unseen card goes to a seat with probability proportional to the room left in its hand: without
  ceilings exactly a uniform random split; with ceilings close to uniform over the consistent splits, not exactly.
- **Ceilings that no split fits** (a seat that passed while holding a beating card): `hard` skips the sample, and if no
  sample fits, the fallback plays the decision; `soft` then samples without ceilings.
- **Not built:** `--pass-rule free` (the sampler rebuilds a `final`-rule round like the exact solver; under `free` the
  fallback plays every move), and play-outs with different strategies per seat.
- **Cost:** about `samples x legal moves x replies^(depth-1)` play-outs per decision: see the measurements before using
  it in a training pool.

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
- It beats the strategy it falls back to clearly (+0.14 score head to head) but not the trained neat champion, whose strength lies in the part of the round the solver does not play; see the measurements. The better use is the toggle: the champion with `+solver` beats three plain champions by +0.083 (`docs/baselines/solver-toggle/README.md`).
- The sampler (phase B) changes that (`docs/baselines/solver-phase-b/README.md`): `adaptive+solver:samples=16` beats
  three `adaptive` by +0.84 (the exact endgame alone: +0.13), but with `adaptive` play-outs it still loses to the
  champion. With the champion as the play-out policy and the ceilings off,
  `neat:docs/baselines/neat-v2/champion.json+solver:samples=16,leaf=neat:docs/baselines/neat-v2/champion.json,ceilings=off`
  beats three champions by +0.153 +-0.011 and wins the mixed table. Depth 2-3 and `leaf=cards` are worse than depth 1
  with a play-out; more samples help (+0.50 at 4, +0.96 at 64 against `adaptive`).
- The unseen cards are weighted equally; hand reading only removes impossible splits. Pass ceilings are used as hard
  facts by default: a seat that passed on purpose while holding a beating card (`adaptive`'s deception,
  `hold-back-pairs`) makes the real split look impossible until that seat plays above its ceiling. The solver then
  solves the wrong splits, or finds none and plays its fallback; this is counted in `SolverStats::fell_back_other`.
  `ceilings=soft` or `ceilings=off` (the spec's `use_pass_ceilings`) changes that; measured in
  `docs/baselines/solver-phase-b/README.md`.
- A double deck has identical cards that are told apart by deal order, which a written-down position cannot show:
  `cli solve` numbers your copies first and says so when it matters.
- `endgame-solver:max-unseen=N` with a large N is bounded by the node budget per decision, not by N; the enumeration
  of splits prunes contradictory pass ceilings early.
