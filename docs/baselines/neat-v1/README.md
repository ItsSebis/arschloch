# NEAT v1 baseline

The first evolved champion, committed so later work can be compared
against it the way `docs/baselines/pre-neat` records the hand-written
strategies. Nothing here changes the pre-NEAT numbers.

- `champion.json`: the genome (a `GenomeFile`; plays with
  `--strategy neat:docs/baselines/neat-v1/champion.json`).
- `run.sh`: re-measures it with the pre-NEAT protocol (seed 1000, 2000
  matches x 10 rounds) at 3-6 players, plus `cli evaluate` on a seed
  stream training never used. Output is seeded and independent of
  `--threads`; `summaries/` holds the recorded output.
- `experiments.md` and `experiments/`: how the learning defaults were
  chosen.

## How it was trained

```bash
target/release/cli train --out runs/v1 --seed 1 --population 150 \
  --generations 120 --matches-per-genome 80 --reeval-matches 200 \
  --rounds 8 --champion-candidates 5 --weight-power 0.2 --quiet
```

4 players, single deck, opponent pool `lowest-legal`, `endgame-denial`,
`adaptive:reading,tempo,bully`. 120 generations took 11.5 minutes on 8
cores. `champion.json` is the run's `best.json` (generation 69; 5 hidden nodes and
33 enabled connections). Held-out score
against the mixed training pool: **+0.624 +- 0.013**. Re-scoring the top 5
genomes on the fixed matches picked a genome other than the training-best
one in 75 of 120 generations, which is how noisy a generation's training
fitness is.

## What it does (`summaries/evaluate_*p.txt`, 600 matches x 8 rounds per cell)

Mean finishing-role score, +1 always President to -1 always last, one
champion against copies of one opponent. Trained on 4 players, evaluated
at every table size without retraining:

| opponent | 3 players | 4 players | 5 players | 6 players |
|---|---|---|---|---|
| LowestLegal | +0.468 | +0.630 | +0.545 | +0.421 |
| EndgameDenial | +0.534 | +0.692 | +0.643 | +0.586 |
| Adaptive(reading,tempo,bully) | +0.468 | +0.609 | +0.592 | +0.555 |
| HoldBackPairs (never trained against) | +0.911 | +0.917 | +0.881 | +0.838 |
| GreedyHighest (never trained against) | +0.960 | +0.939 | +0.927 | +0.895 |
| RandomLegal (never trained against) | +0.960 | +0.914 | +0.826 | +0.744 |
| all of the above mixed | +0.639 | +0.708 | +0.674 | +0.596 |

Standard errors are 0.003-0.016; see the `evaluate_*p.txt` files.

## Against the pre-NEAT baseline (`summaries/champion-vs-*`)

4 players, 2000 matches x 10 rounds (20000 rounds), one champion against
three copies of one opponent. A player at random would be President in
5000 rounds and last in 5000.

| opponent (x3) | champion President | champion last | opponent President | opponent last |
|---|---|---|---|---|
| LowestLegal | 11477 (57%) | 414 (2%) | 8523 (total, 3 seats) | 19586 (total, 3 seats) |
| EndgameDenial | 12512 (63%) | 255 (1%) | 7488 (total) | 19745 (total) |
| Adaptive(reading,tempo,bully) | 11326 (57%) | 452 (2%) | 8674 (total) | 19548 (total) |

For scale, in `docs/baselines/pre-neat` the best hand-written strategy,
`adaptive:reading,tempo,bully`, against three `LowestLegal` players was
President in 6641 rounds (33%) and last in 3507 (18%).

The champion passes voluntarily (a legal play existed) in about 8-10% of
its turns; every hand-written strategy except `HoldBackPairs` and
`RandomLegal` never does.

## Caveats

- One champion from one seed, trained against one pool. It says what
  this approach reaches, not an upper bound; `experiments.md` shows
  seed-to-seed spread of about 0.01-0.03 on these scores.
- It was trained at 4 players. The 3-, 5- and 6-player rows are
  generalization, and its edge shrinks at 6 players against
  `LowestLegal` (+0.421).
- Opponent tables here are identical copies, so no table-position bias
  enters (see the caveats in `docs/baselines/pre-neat/README.md`).
- Scores of the pre-NEAT strategies and of this champion come from
  different table compositions; the rows above only compare each against
  the same opponents where stated.
