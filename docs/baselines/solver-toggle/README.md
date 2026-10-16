# Endgame solver as a toggle (`SPEC+solver`): measurements

Binary: release build of the commit that added `+solver`. 4 players, single deck, `--pass-rule final`, forced exchange,
`--matches 1200 --rounds 5 --seed 7 --skill-score duplicate` (300 duplicate groups; the luck-free head to head of
`docs/baselines/solver/README.md`), 4 threads. One seat plays `SPEC+solver:6`, the other three play plain `SPEC`; the
duplicate deals rotate everybody through every seat. `run.sh` reproduces it; the full summaries are the `*-T6.txt` files.
Score: +1 best .. -1 worst, +-SE. The gain is the toggled seat's score minus the plain seats' score in the same table (its
SE combines the two and ignores their small correlation, so it is slightly too large).

| SPEC | with `+solver:6` | plain (each of 3) | gain from the toggle | wall time |
|---|---|---|---|---|
| `adaptive` (= `Adaptive(counting,reading)`) | +0.100 +-0.006 | -0.033 +-0.002 | **+0.133 +-0.006** | 208 s |
| `endgame-denial` | +0.095 +-0.007 | -0.032 +-0.002 | **+0.127 +-0.007** | 229 s |
| `hold-back-pairs` | +0.085 +-0.005 | -0.028 +-0.002 | **+0.113 +-0.005** | 59 s |
| `lowest-legal` | +0.136 +-0.007 | -0.045 +-0.002 | **+0.181 +-0.007** | 202 s |
| `neat:` v2 champion | +0.062 +-0.006 | -0.021 +-0.002 | **+0.083 +-0.006** | 388 s |

Reading:

- Every strategy gains, by 13 to 30 standard errors, so the toggle is a real improvement for all five, including the
  trained champion. The weaker the strategy's own endgame, the larger the gain (`lowest-legal` most, the champion least),
  as `cli solver-gap` predicted (the share of solved decisions where the strategy deviates: 10-15%).
- The champion gains least but it does gain: with the solver on the last cards it beats three plain copies of itself by
  +0.083. The earlier table (solver+`adaptive` against the champion, `docs/baselines/solver/README.md`) only said that
  the *solver player* is far weaker than the champion, because it falls back to `adaptive` in 89% of the decisions;
  the champion keeping its own early game and the solver finishing is the stronger combination.
- `adaptive+solver` here (+0.133 over plain adaptive) agrees with the older `endgame-solver` table (+0.140): the same
  player, since `endgame-solver` is `adaptive+solver` with a different name.
- Comparisons are within one table. The toggled seats' scores are not comparable across tables (different opponents);
  the gains are.
- Threshold 8 was not measured: it costs about 7 times more than 6 (see `docs/baselines/solver/README.md`: 5 minutes
  per 200 single-round games) and over budget on a tenth of the solved decisions.

## What the toggle changes in how the strategy plays

The voluntary pass rate of the toggled seat rises (adaptive 0% to 6.5%, `lowest-legal` 0% to 9.2%, endgame-denial 0% to
6.9%, champion 19.8% to 25.5%; `hold-back-pairs` already passes on purpose, 65% to 67%): the solver passes when keeping
a card back gives a better expected place, which the heuristics never do or do rarely. See `*-T6.txt`.

## Cost

One toggled seat, 240 matches x 5 rounds, 4 threads, `adaptive` against three `adaptive` (the plain table takes 0.03 s):

| `max-unseen` | wall time | factor over plain |
|---|---|---|
| 2 | 0.21 s | about 7 |
| 4 | 6.5 s | about 200 |
| 6 | 42 s | about 1400 |

Wrapping with `max-unseen=0` costs about 15% (the strategy then asks for the full turn context): 0.16 s against 0.14 s
for 1200 x 5 rounds. **Training pools:** `train --opponent adaptive+solver:2` (and `neat:PATH+solver:2`, frozen into
the run directory) was run end to end and resumed; a default-sized generation (150 genomes x 100 matches x 8 rounds)
with a threshold-6 solver in one opponent seat would take hours instead of seconds, so for training use a threshold of
2 to 4, or evaluate the finished genome against solver opponents afterwards. Training a genome *with* the solver
finishing its own rounds changes what the fitness measures (the genome would no longer be judged on its endgame) and is
not built; it remains an open experiment.

## Regression

Without the toggle nothing changes: a 4-player run of `endgame-solver`, `adaptive:reading,tempo`, the champion and
`hold-back-pairs` (400 matches x 3 rounds, `--skill-score both`) gives a byte-identical summary and results file from the
binary before and after, and every checksum of `docs/baselines/perf/bench.sh` is unchanged. A wrapped strategy with
`max-unseen=0`, and any wrapped strategy under `--pass-rule free`, plays move for move like the bare one (unit tests).
