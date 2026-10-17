# Endgame solver phase B, the early-game sampler: measurements

Binary: release build of the commit that added `samples=N` (spec:
`docs/superpowers/specs/2026-10-17-solver-phase-b-design.md`, user guide: `docs/SOLVER.md`). Every table: 4 players,
single deck, `--pass-rule final`, forced exchange, `--matches 1200 --rounds 5 --seed 7 --skill-score duplicate` (300
duplicate groups, the luck-free head to head of `docs/baselines/solver/README.md`), 4 threads (the three rows with the
neat leaf and `ceilings=off`: 5 threads). One seat plays the configuration under test, the others what the table names; the
duplicate deals rotate everybody through every seat. Score: +1 best .. -1 worst, +-SE. "Gain" is the tested seat's
score minus the other seats' in the same table (its SE is about the tested seat's SE). Wall times were taken on a shared
8-core machine with another long job running, so they are rough. `run.sh` reproduces everything; the summaries of these runs are in `summaries/`.

Notation: `S` samples per decision, `D` depth, leaf `adaptive` (play the round out with `adaptive` in every seat),
`cards` (static guess), `neat` (play out with the neat-v2 champion in every seat). `T=0` = the sampler alone (no exact
endgame), `T=6` = sampler plus the exact endgame of phase A ("solver-B").

## The sampler against three `adaptive`

`adaptive+solver:max-unseen=0,samples=S[,depth=D][,leaf=E]` in one seat, plain `adaptive` in three.

| S | D | leaf | sampler | adaptive (each) | gain | wall time |
|---|---|---|---|---|---|---|
| 4 | 1 | adaptive | +0.371 +-0.008 | -0.124 | +0.495 | 9 s |
| 8 | 1 | adaptive | +0.529 +-0.007 | -0.176 | +0.705 | 20 s |
| 16 | 1 | adaptive | +0.628 +-0.007 | -0.209 | **+0.837** | 41 s |
| 32 | 1 | adaptive | +0.683 +-0.007 | -0.228 | +0.911 | 78 s |
| 64 | 1 | adaptive | +0.719 +-0.007 | -0.240 | **+0.959** | 159 s |
| 16 | 2 | adaptive | +0.360 +-0.009 | -0.120 | +0.480 | 212 s |
| 16 | 3 | adaptive | +0.219 +-0.010 | -0.073 | +0.292 | 1106 s |
| 16 | 1 | cards | -0.379 +-0.011 | +0.126 | **-0.505** | 2 s |
| 16 | 3 | cards | -0.137 +-0.010 | +0.046 | -0.183 | 47 s |
| 16 | 1 | neat | +0.638 +-0.007 | -0.213 | +0.851 | 71 s |
| 16, T=6 (solver-B) | 1 | adaptive | +0.626 +-0.007 | -0.209 | +0.835 | 632 s |

For comparison, the phase A toggle (`adaptive+solver`, exact endgame only) gains +0.133 in the same table
(`docs/baselines/solver-toggle/README.md`); plain `adaptive` against itself is 0 by construction.

Reading:

- **The sampler is far stronger than the strategy it wraps**: +0.84 with 16 samples, six times the exact endgame's
  gain, at about 1 ms per decision. More samples keep helping with diminishing returns (+0.50, +0.71, +0.84, +0.91, +0.96
  for 4 to 64).
- **Depth hurts.** Depth 2 and 3 score well below depth 1 at many times the cost. A plausible reason (not tested
  separately): in a sampled deal every opponent sees all hands, so a deeper max^n assumes ever better replies than
  real opponents make (strategy fusion grows with depth).
- **The static leaf is useless**: `leaf=cards` plays clearly worse than plain `adaptive`. The play-out is what makes
  the sampler work; the `neat` play-out is as good as `adaptive`'s against `adaptive` (+0.85 against +0.84).
- **The exact endgame adds nothing measurable on top** (+0.835 with T=6 against +0.837 without) and costs 15 times the
  time, so with the sampler on, `max-unseen=0` is the cheaper choice.
- The sampler passes on purpose much more than any heuristic (voluntary pass rate about 67-75% of its decisions with a
  pass available, `adaptive` 0%), the same tendency the exact solver showed in the endgame.

## Against the neat-v2 champion

One seat as named, three champions (`neat:docs/baselines/neat-v2/champion.json`).

| Tested seat | ceilings | tested | champion (each) | gain | wall time |
|---|---|---|---|---|---|
| plain `adaptive` | | -0.565 +-0.012 | +0.188 | -0.753 | under 1 s |
| `adaptive+solver` (phase A, T=6) | hard | -0.496 +-0.011 | +0.165 | -0.661 | 160 s |
| sampler alone, S=16, adaptive leaf | hard | -0.351 +-0.011 | +0.117 | -0.468 | 22 s |
| solver-B, S=16, adaptive leaf | hard | -0.271 +-0.010 | +0.090 | -0.361 | 541 s |
| solver-B, S=16, adaptive leaf | soft | -0.200 +-0.010 | +0.067 | -0.267 | 1000 s |
| solver-B, S=16, adaptive leaf | off | -0.174 +-0.010 | +0.058 | -0.232 | 1048 s |
| champion + solver-B, S=16, adaptive leaf | hard | -0.161 +-0.010 | +0.054 | -0.215 | 619 s |
| champion + solver-B, S=16, **neat leaf** | hard | +0.080 +-0.012 | -0.027 | **+0.107** | 499 s |
| solver-B, S=16, **neat leaf** | **off** | **+0.115 +-0.011** | -0.038 | **+0.153** | 829 s |

(The phase A toggle on the champion, `neat:...+solver`, gains +0.083 in this table:
`docs/baselines/solver-toggle/README.md`.)

The mixed table of `docs/baselines/solver/README.md` (one seat each):

| Tested seat | tested | champion | adaptive | lowest-legal | wall time |
|---|---|---|---|---|---|
| phase A `endgame-solver` (earlier result) | -0.076 +-0.011 | +0.628 | -0.234 | -0.318 | |
| solver-B, S=16, adaptive leaf, hard | +0.202 +-0.010 | **+0.496** +-0.010 | -0.360 | -0.338 | 712 s |
| solver-B, S=16, neat leaf, ceilings off | **+0.422** +-0.009 | +0.379 +-0.010 | -0.436 | -0.366 | 421 s |

Reading:

- **With `adaptive` play-outs the sampler does not beat the champion.** It closes half of plain `adaptive`'s gap
  (-0.36 instead of -0.75 against three champions) and stays behind it in the mixed table (+0.20 against +0.50).
  Wrapped around the champion with `adaptive` play-outs it makes the champion *worse* (-0.215): the champion's own
  early game is better than what search over `adaptive` play-outs finds.
- **With the champion as the play-out policy and the ceilings off, it beats the champion**: +0.153 +-0.011 over three
  champions (the phase A toggle on the champion: +0.083), and +0.422 against +0.379 in the mixed table (a margin of
  about three standard errors). It is the first solver configuration that beats the neat-v2 champion head to head.
- With the forced exchange and the ceilings off the sampler answers every early decision itself, so the wrapped
  strategy only plays forced moves: `adaptive+solver:...` and `neat:...+solver:...` with the same options played
  move for move the same (both rows gave +0.115 / -0.038 exactly); the table lists it once.
- **Pass ceilings against the champion:** the champion passes on purpose (about 18% of its decisions), so hard
  ceilings make its real hand look impossible; `soft` (+0.094 better) and `off` (+0.129 better) both help, `off` most.

## Ceilings against a strategy that passes on purpose

solver-B (S=16, adaptive leaf, T=6) in one seat, three `hold-back-pairs`:

| ceilings | solver-B | hold-back-pairs (each) | gain | wall time |
|---|---|---|---|---|
| hard | +0.931 +-0.004 | -0.310 | +1.241 | 33 s |
| soft | +0.748 +-0.007 | -0.249 | +0.997 | 108 s |
| off | +0.747 +-0.007 | -0.249 | +0.996 | 97 s |

Here hard ceilings are clearly better: `hold-back-pairs` passes to keep a pair together, which still tells the truth
about its singles ("nothing above that single") often enough. Whether ceilings help depends on who passes how, so the
default stays `hard`; against the champion use `off`.

## Endgame decisions against the exact solver (`cli solver-gap`)

4 players, 1500 single-round games, seed 3, one seat each: the sampler alone (no exact endgame) with 16 and with 64
samples, `adaptive`, the champion. 70,945 decisions with a choice, 7,430 solved exactly (T=6), 546 over budget; 1414 s,
single-threaded.

| Strategy | Solved | Differ | Share | Places lost each |
|---|---|---|---|---|
| sampler, S=16 | 3336 | 186 | 5.6% | 0.039 |
| sampler, S=64 | 2548 | 167 | 6.6% | 0.050 |
| `adaptive` | 714 | 54 | 7.6% | 0.058 |
| `neat-v2` champion | 832 | 82 | 9.9% | 0.073 |

The sampler loses fewer places per endgame decision than `adaptive` and the champion, but it is not exact; its
costliest mistakes are all passes where the solver plays a card (for example: hand 8H JC 2C on a 5C, 3 unseen: the
sampler passes, the solver plays 8H, 1.5 places). S=64 is not better than S=16 here; the two seats see different
positions (the sampler seats reach far more solvable endgames, since they play stronger), so the rows are not a
controlled comparison of the sample count.

## Not measured

Other table sizes and the double deck; `rollouts` limits (no run reached the default 20,000 at depth 1); the NEAT leaf
at more samples; `--pass-rule free` (not supported).
