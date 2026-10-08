# NEAT v2 baseline (pass rule `final`)

> `run.sh` defaults to `EXCHANGE_RULE=free` (how the summaries here were recorded);
> `../current-rules/neat-v2-champion` is the same measurement under the forced
> exchange. The champion itself is unchanged by that rule (a retrain under it is
> byte-identical, see `../current-rules/README.md`), so it is the champion built into
> `cli play`.

The champion retrained under the rules of the game (`--pass-rule final`, a pass
ends your part in the trick) with exactly the neat-v1 recipe. `neat-v1` remains
the reference for the older rule (`free`).

- `champion.json`: the genome (plays with
  `--strategy neat:docs/baselines/neat-v2/champion.json`; also the champion that
  is built into `cli play`).
- `run.sh`: the same measurements as `neat-v1/run.sh`, defaulting to the new
  rule; `summaries/` holds the output (28 files). Seeded and independent of the
  thread count.

## How it was trained

```bash
target/release/cli train --out runs/v2 --seed 1 --population 150 \
  --generations 120 --matches-per-genome 80 --reeval-matches 200 \
  --rounds 8 --champion-candidates 5 --weight-power 0.2 --pass-rule final \
  --exchange-rule free --quiet
```

4 players, single deck, opponent pool `lowest-legal`, `endgame-denial`,
`adaptive:reading,tempo,bully`. 120 generations took 6:51 on a 4-core laptop
(about 30k rounds/s). `champion.json` is the run's `best.json` (generation 51; 3
hidden nodes, 24 enabled connections). Its score against the mixed training pool
on held-out matches: **+0.608 ±0.013** under `final`.

## v1 against v2 (`cli evaluate`, 800 matches x 8 rounds per cell, seed 99)

| rule | v1 mixed | v2 mixed |
|---|---:|---:|
| final | +0.651 ±0.009 | +0.663 ±0.008 |
| free | +0.677 ±0.008 | +0.669 ±0.009 |

At 4 players the retrained champion is not measurably different from v1 under
either rule (the differences are within one or two standard errors). The mixed
rows for 3-6 players (600 matches x 8 rounds per cell: `summaries/evaluate_*p.txt` here against
`../pass-final/neat-v1-champion/evaluate_*p.txt` for v1, both under `final`):

| players | v1 | v2 |
|---|---:|---:|
| 3 | +0.553 | +0.578 |
| 4 | +0.651 | +0.665 |
| 5 | +0.647 | +0.643 |
| 6 | +0.593 | +0.580 |

Why the rule matters so little: it only changes games where somebody passes while
holding a beating play, which the evolved players do occasionally and most
hand-written strategies never do (see `../pass-final/README.md`). A model that
could see who is still in the trick might exploit the rule more; that is a Phase
13 idea (an extra input), not part of this baseline.
