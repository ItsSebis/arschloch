# NEAT v1 baseline

The first evolved champion, committed so later work can be compared
against it the way `docs/baselines/pre-neat` records the hand-written
strategies. Nothing here changes the pre-NEAT numbers.

- `champion.json`: the genome (a `GenomeFile`; plays with
  `--strategy neat:docs/baselines/neat-v1/champion.json`).
- `run.sh`: re-measures it with the pre-NEAT protocol (seed 1000, 2000
  matches x 10 rounds) at 3-6 players, the design spec's mixed-field lineup
  over four seed bases, and `cli evaluate` on a seed stream training never
  used. Output is seeded and independent of `--threads`; `summaries/` holds
  the recorded output (28 files).
- `experiments.md` and `experiments/`: how the learning defaults were
  chosen.

## How it was trained

```bash
target/release/cli train --out runs/v1 --seed 1 --population 150 \
  --generations 120 --matches-per-genome 80 --reeval-matches 200 \
  --rounds 8 --champion-candidates 5 --weight-power 0.2 --quiet
```

4 players, single deck, opponent pool `lowest-legal`, `endgame-denial`,
`adaptive:reading,tempo,bully`. 120 generations took about 11 minutes on 8
cores. `champion.json` is the run's `best.json` (generation 14; 2 hidden
nodes and 21 enabled connections). Its score against the mixed training
pool on held-out matches (which did not choose it): **+0.619 +- 0.013**.
Re-scoring the top 5 genomes picked a genome other than the training-best
one in 79 of 120 generations, which is how noisy a generation's training
fitness is. (An earlier version of this baseline was trained before
candidates were ranked on their own matches; see `experiments.md`.)

## What it does (`summaries/evaluate_*p.txt`, 600 matches x 8 rounds per cell)

Mean finishing-role score, +1 always President to -1 always last, one
champion against copies of one opponent. Trained on 4 players, evaluated at
every table size without retraining. "Never seen" opponents are not in the
training pool; the two adaptive variants are strong, the last three weak.

| opponent | 3 players | 4 players | 5 players | 6 players |
|---|---|---|---|---|
| LowestLegal | +0.430 | +0.620 | +0.556 | +0.443 |
| EndgameDenial | +0.515 | +0.695 | +0.653 | +0.596 |
| Adaptive(reading,tempo,bully) | +0.464 | +0.601 | +0.595 | +0.570 |
| Adaptive(counting,reading) (never seen, strong) | +0.513 | +0.687 | +0.660 | +0.594 |
| Adaptive(reading,deception=0.2,tempo,bully) (never seen, strong) | +0.407 | +0.568 | +0.554 | +0.541 |
| HoldBackPairs (never seen, weak) | +0.911 | +0.914 | +0.881 | +0.838 |
| GreedyHighest (never seen, weak) | +0.945 | +0.930 | +0.930 | +0.896 |
| RandomLegal (never seen, weak) | +0.948 | +0.905 | +0.833 | +0.758 |
| all eight, mixed | +0.573 | +0.673 | +0.648 | +0.609 |

Standard errors are 0.003-0.015; see the `evaluate_*p.txt` files. Against
the strong opponents it never saw, the champion scores about as well as
against the ones it trained on.

## Against the hand-written strategies in the pre-NEAT protocol

4 players, seed 1000, 2000 matches x 10 rounds (20000 rounds), the same
protocol and lineups as `docs/baselines/pre-neat/run.sh`
(`summaries/champion-vs-*`). One champion against three copies of one
opponent; a player at random would be President in 5000 rounds (25%) and
last in 5000.

| opponent (x3) | champion President | champion last |
|---|---|---|
| LowestLegal | 11302 (56.5%) | 431 (2.2%) |
| EndgameDenial | 12557 (62.8%) | 244 (1.2%) |
| Adaptive(reading,tempo,bully) | 11256 (56.3%) | 517 (2.6%) |

(`card-counter` gives exactly the `LowestLegal` row: it plays identically
to `lowest-legal`, see `docs/baselines/pre-neat/README.md`.)

The like-for-like pre-NEAT row is the best hand-written strategy against
three `LowestLegal` players, same protocol: `adaptive:reading,tempo,bully`
was President in 6641 rounds (33%) and last in 3507 (18%); this champion
against the same three `LowestLegal` players is President in 56.5% and last
in 2.2%.

## The spec's mixed field (section 9)

The champion, `card-counter`, `endgame-denial` and
`adaptive:reading,tempo,bully` at one 4-player table, four independent seed
bases (`summaries/mixed-field_4p_seed*.txt`):

| seed base | champion President | champion last | Adaptive President | Adaptive last |
|---|---|---|---|---|
| 1000 | 11120 (55.6%) | 561 (2.8%) | 3834 (19.2%) | 5317 (26.6%) |
| 2000 | 11098 (55.5%) | 575 (2.9%) | 3771 (18.9%) | 5407 (27.0%) |
| 3000 | 11102 (55.5%) | 568 (2.8%) | 3734 (18.7%) | 5306 (26.5%) |
| 4000 | 11266 (56.3%) | 577 (2.9%) | 3644 (18.2%) | 5320 (26.6%) |

The champion is President in about 56% of rounds at a table where chance
gives 25%. Mixed fields carry a table-position bias in this simulator (see
`docs/baselines/pre-neat/README.md`); the champion's margin is far larger.

## How it plays differently

Of the champion's own passes, 15-77% (depending on table size and
opponent; 57% at 4 players against `LowestLegal`) were voluntary, meaning a
legal play existed and it declined it, holding cards back. The
hand-written strategies other than `hold-back-pairs` and `random-legal`
never do. (The pooled figure printed under "Voluntary pass rate" in the
summaries is across all four seats' passes, so it is lower.)

## Caveats

- One champion from one seed, trained against one pool. It says what this
  approach reaches, not an upper bound; `experiments.md` shows seed-to-seed
  spread of about 0.01-0.03 on these scores.
- It was trained at 4 players. The 3-, 5- and 6-player rows are
  generalization, and its edge shrinks at 6 players against
  `LowestLegal` (+0.443).
- The "never seen" opponents are variants and weaker members of the same
  family of hand-written strategies; this says nothing about how it would
  do against a human or a differently designed opponent.
- Scores in `evaluate_*p.txt` (a -1..+1 mean role score) and the role
  counts above (a count of rounds) are two views of the same games.
