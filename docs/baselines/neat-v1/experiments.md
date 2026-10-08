# Phase 10e learning-option experiments

Question: do the new learning options (top-k champion selection, a hall of
fame, a smaller weight-perturbation power) produce better champions, and
do champions trained against three fixed opponents generalize to opponents
they never saw?

**Setup.** Population 100, 60 generations, 40 matches x 6 rounds per
genome, 100 re-evaluation matches, 4 players, training pool
`lowest-legal`, `endgame-denial`, `adaptive:reading,tempo,bully`. Each
run's `best.json` is then scored by `cli evaluate` (600 matches x 8 rounds
per cell, seed 99, a stream training never uses) against the three
training opponents ("trained-3") and against `hold-back-pairs`,
`greedy-highest` and `random-legal`, which training never saw
("unseen-3"). Scores are mean finishing-role scores, +1 always President
to -1 always last. Reproduce with `experiments/run_experiments.sh`.

## Five configurations, three seeds each

| configuration | trained-3 | unseen-3 | mixed |
|---|---|---|---|
| baseline (1 candidate, no hall, power 0.5) | +0.597 | +0.907 | +0.672 |
| top-5 champion selection | +0.619 | +0.904 | +0.684 |
| hall of fame (3 members, every 5 generations) | +0.586 | +0.915 | +0.671 |
| weight power 0.2 | +0.614 | +0.921 | +0.703 |
| weight power 0.1 | +0.603 | +0.918 | +0.699 |

Seed-to-seed spread (one standard deviation) of a single configuration on
trained-3 is about 0.012 to 0.03, so differences of 0.01 to 0.02 between
rows are within noise.

## Baseline against the two best options combined, five seeds

| configuration | trained-3 | unseen-3 | mixed |
|---|---|---|---|
| baseline | +0.603 (sd 0.012) | +0.907 (sd 0.017) | +0.679 (sd 0.017) |
| top-5 selection + weight power 0.2 | +0.619 (sd 0.022) | +0.912 (sd 0.023) | +0.697 (sd 0.022) |

Paired difference on trained-3 (same seeds): **+0.016, standard error
0.012** (per seed +0.048, -0.007, +0.032, +0.025, -0.016).

## What this does and does not show

- **Generalization is not a problem here.** Every configuration scores
  +0.90 to +0.92 against opponents it never trained against, far above
  its score against the trained opponents; there is no sign of overfitting
  to the three-opponent pool.
- **The learning options help a little, not conclusively.** The combined
  configuration is better by about 0.016 (1.3 standard errors) on the
  trained opponents, 0.018 in mixed tables and 0.005 on unseen ones. It was
  adopted as the default because top-k selection is principled (the
  training fitness of a generation's best genome is dominated by noise and
  re-scoring a few candidates on fixed matches removes that), costs about
  12% more time per generation, and was never worse; but five seeds do not
  prove it.
- **The hall of fame showed no benefit** against this pool and stays off by
  default (`--hall-of-fame N` enables it). It may matter against pools that
  include evolved opponents.
- Every champion here already beats the strongest hand-written strategy
  (`adaptive:reading,tempo,bully`) by about +0.55 to +0.6 in a table of
  one champion and three copies of it.
