# Baselines under the pass rule

Phase 14 made "a pass ends your part in the trick" (`--pass-rule final`) the rule
of the game and the default; the baselines recorded before it were measured
under the older behaviour (`--pass-rule free`). This directory re-measures them
under the new rule with exactly the same seeds, match counts and round counts, so
the two sets compare directly.

- `hand-written/`: `docs/baselines/pre-neat/run.sh` with `PASS_RULE=final`
  (36 files; compare with `docs/baselines/pre-neat/summaries`).
- `neat-v1-champion/`: `docs/baselines/neat-v1/run.sh` with `PASS_RULE=final`
  (28 files; compare with `docs/baselines/neat-v1/summaries`). The v1 champion
  was trained under `free`.
- The retrained champion is in `docs/baselines/neat-v2`.

## What changed, and why so little

The rule only matters when a player passes **voluntarily**, that is, while holding
a play that beats the table. Passing because nothing beats the table is
unaffected: the table only ever gets higher and hands only shrink, so a seat that
could not beat the table once never can later. Most hand-written strategies pass
only when they have to (`Voluntary pass rate: 0.00%` in their summaries), so their
results are **identical** under both rules. Only the strategies that pass on
purpose differ:

| strategy (vs 3 `lowest-legal`, 4 players, 2000 x 10 rounds) | President | Arschloch | table-wide voluntary pass rate |
|---|---:|---:|---:|
| `hold-back-pairs`, free | 87 | 16195 | 25.66% |
| `hold-back-pairs`, final | 33 | 16785 | 27.98% |
| `random-legal`, free | 516 | 11531 | 9.05% |
| `random-legal`, final | 427 | 11135 | 10.88% |
| adaptive with deception 0.2, free | 6798 | 3252 | 1.10% |
| adaptive with deception 0.2, final | 6789 | 3202 | 1.46% |

(The pass-rate column is the whole table's rate, as the summaries print it; the
named strategy's own rate is higher.)

Files that differ between `pre-neat/summaries` and `hand-written/` (ignoring the
header line): `random-legal`, `hold-back-pairs` and the deception variant at 3-6
players, and the 5- and 6-player mixed fields (which include `hold-back-pairs`).
The 14 differing files are exactly the ones containing a strategy that passes
voluntarily; every other file is byte-identical.

## The v1 champion under both rules (`cli evaluate`, 800 matches x 8 rounds)

| opponent set | free | final |
|---|---:|---:|
| mixed (all opponents) | +0.677 ±0.008 | +0.651 ±0.009 |
| `lowest-legal` | +0.619 ±0.009 | +0.601 ±0.010 |

The champion plays the evolved networks' voluntary passes too, so it is a little
weaker when passing is final (it gives up chances to play later in the trick).
It is still far ahead of every hand-written strategy.
