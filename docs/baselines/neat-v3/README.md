# NEAT v3 baseline (`hidden` features, 300 generations)

`champion.json` is the champion shipped with the release (built into `cli play` as
`model:champion-v3`, packed as `champion.json` in the release archives). neat-v2 stays
in the repo (`../neat-v2/`, built in as `model:champion-v2`) for comparison.

**It is a `hidden` genome (28 inputs): it plays in every Rust tool of this repo, but NOT in the
casino bot** (the TypeScript port knows feature set v1 only, open work). neat-v1 and neat-v2
are `base` genomes and still run there.

## Recipe

`cli train` with the neat-v2 recipe (population 150, 80 or 200 matches x 8 rounds per genome,
re-evaluation 200 matches, 5 champion candidates, weight power 0.2, pass rule `final`, forced
exchange, 4 players, single deck, pool `lowest-legal`, `endgame-denial`,
`adaptive:reading,tempo,bully`) and the new default `--features hidden` (not spelled out;
`train-config.json` is the run's `config.json`). `train.sh` is the driver that produced every
candidate (4 threads). The champion is the best.json of the run with seed 32:

```bash
cli train --out runs/v3 --seed 32 --population 150 --generations 300 \
  --matches-per-genome 200 --reeval-matches 200 --rounds 8 --champion-candidates 5 \
  --weight-power 0.2 --pass-rule final --exchange-rule forced --threads 4 --quiet
```

It took 25 minutes (a 120-generation, 80-match run takes 5 to 7 minutes). Best generation 247;
15 hidden nodes, 71 enabled connections. Its own held-out training score against the pool is
+0.734 +- 0.010 (400 matches), which is optimistic: it is the best of the champion candidates.

## Candidates and selection (seed stream 99)

Eleven candidates: 8 short runs (120 generations x 80 matches, seeds 31-38) and 3 long runs
(300 generations x 200 matches, seeds 31-33; the roadmap asked whether longer runs help).
Each was evaluated on seed stream 99, 2000 matches x 10 rounds, 4 players, single deck, forced
exchange, final pass rule: once against the frozen battery, once head to head against three
copies of the neat-v2 champion (score 0 = even). Raw output: `selection/`.

| candidate | battery (mixed) | head to head vs neat-v2 |
|---|---:|---:|
| short31 | +0.707 +- 0.005 | +0.139 +- 0.008 |
| short32 | +0.683 | +0.015 |
| short33 | +0.691 | +0.049 |
| short34 | +0.675 | +0.038 |
| short35 | +0.666 | +0.002 |
| short36 | +0.684 | +0.022 |
| short37 | +0.715 | +0.072 |
| short38 | +0.684 | +0.026 |
| long31 | +0.695 | +0.085 |
| **long32** | **+0.739 +- 0.004** | **+0.171 +- 0.008** |
| long33 | +0.743 +- 0.004 | +0.117 +- 0.008 |

Means: short runs battery +0.688, head to head +0.045; long runs battery +0.726, head to head
+0.124. With three long runs against eight short ones, longer runs look clearly better (the
long run with the worst result, long31, still beats seven of eight short ones head to head),
but it is three seeds, not a tested claim, and a long run costs about four times as much.
**Selection rule, fixed before looking: the best head-to-head score against neat-v2 on seed
stream 99** -> long32 (it also has the second best battery score).

## Confirmation on a different seed stream (seed 777777)

Selected on stream 99 out of 11 candidates, so its stream-99 numbers are inflated by the choice
(winner's curse). Confirmed on a stream no selection used: seed 777777, 4000 matches x 10
rounds, same rules. Raw output: `confirmation/`.

| test | neat-v3 (long32) | neat-v2 | difference |
|---|---:|---:|---|
| battery, mixed over 8 opponents | +0.745 +- 0.003 | +0.677 +- 0.003 | +0.068, 16 SE (unpaired) |
| head to head, 1 neat-v3 vs 3 neat-v2 (0 = even) | +0.164 +- 0.006 | | 27 SE from even |

neat-v3 is ahead of neat-v2 against every opponent of the battery (for example
LowestLegal +0.735 vs +0.620, Adaptive(reading,tempo,bully) +0.699 vs +0.605; the weak
opponents HoldBackPairs, GreedyHighest and RandomLegal are near the ceiling for both). It beats
the 2-SE bar by a wide margin, so it replaces neat-v2 as the shipped champion.

## Caveats

- One table size (4 players, single deck), one opponent set, the same pool it was trained on
  plus the battery; 3, 5 and 6 players were not measured for v3.
- The confirmation compares one selected run with the old champion; it says nothing about the
  average long run (see the means above for that, from the selection stream).
- Against the old champion it is stronger because of both the inputs and the longer run; this
  experiment did not separate the two (the short `hidden` runs alone average +0.045 head to head).
- Casino bot: cannot load it (see top).

## Real-scenario check (release binary, default settings)

`cli train --population 60 --generations 6 ... --seed 7` (no `--features`) wrote `"features": "hidden"` to
`config.json`; `cli evaluate --genome <that run> --genome docs/baselines/neat-v3/champion.json` (300 matches x 6
rounds, seed 5) printed mixed +0.607 +- 0.019 for the 6-generation run and +0.762 +- 0.012 for neat-v3; a 500-match
table of neat-v3 against two LowestLegal and an adaptive player gave it 2372 first places out of 4000 rounds; and
`cli play` served `model:champion-v3` and `model:champion-v2` as opponents, a 3-player game started and the "Ask a model"
advice endpoint answered for both models (a hidden genome included).
