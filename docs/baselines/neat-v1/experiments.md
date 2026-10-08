# Phase 10e learning-option experiments

Question: do the new learning options (top-k champion selection, a hall of
fame, a smaller weight-perturbation power) produce better champions, and
do champions trained against three fixed opponents hold up against
opponents they never trained against?

**Setup.** Population 100, 60 generations, 40 matches x 6 rounds per
genome, 100 re-evaluation matches, 4 players, training pool
`lowest-legal`, `endgame-denial`, `adaptive:reading,tempo,bully`. Each
run's `best.json` is scored by `cli evaluate` (600 matches x 8 rounds per
cell, seed 99, a stream training never uses) against eight opponents:

- **trained-3**: the three training opponents;
- **stronger-unseen-2**: `adaptive:counting,reading` and
  `adaptive:reading,deception=0.2,tempo,bully`, strong strategies (variants
  of the best hand-written one) that training never saw;
- **weak-unseen-3**: `hold-back-pairs`, `greedy-highest`, `random-legal`,
  never seen but weak, so beating them says little about overfitting;
- **mixed-8**: tables mixing all eight.

Scores are mean finishing-role scores, +1 always President to -1 always
last. Reproduce with `experiments/run_experiments.sh`. (These runs were
re-measured after a fix to champion selection, see below; configurations
with one candidate are unaffected by it and were only re-evaluated.)

## Six configurations (mean over seeds)

| configuration | seeds | trained-3 | stronger-unseen-2 | weak-unseen-3 | mixed-8 |
|---|---|---|---|---|---|
| baseline (1 candidate, no hall, power 0.5) | 5 | +0.603 | +0.593 | +0.907 | +0.639 |
| top-5 champion selection | 3 | +0.615 | +0.603 | +0.913 | +0.657 |
| hall of fame (3 members, every 5 generations) | 3 | +0.586 | +0.575 | +0.915 | +0.625 |
| weight power 0.2 | 3 | +0.614 | +0.601 | +0.921 | +0.657 |
| weight power 0.1 | 3 | +0.603 | +0.590 | +0.918 | +0.643 |
| top-5 selection + weight power 0.2 | 5 | +0.619 | +0.606 | +0.912 | +0.650 |

For reference, the 3-seed baseline (seeds 1-3, same seeds as the 3-seed
rows) is +0.597 | +0.591 | +0.907 | +0.637. The seed-to-seed standard deviation of a single
configuration on trained-3 is about 0.012 to 0.03, so differences of
0.01 to 0.02 between rows are within noise.

## Paired differences against the baseline (same seeds)

| comparison | seeds | trained-3 | stronger-unseen-2 | weak-unseen-3 | mixed-8 |
|---|---|---|---|---|---|
| top-5 + power 0.2 minus baseline | 5 | +0.016 (se 0.012) | +0.012 (se 0.012) | +0.005 (se 0.014) | +0.011 (se 0.015) |
| top-5 alone minus baseline | 3 | +0.018 (se 0.016) | +0.012 (se 0.017) | +0.007 (se 0.014) | +0.020 (se 0.007) |
| power 0.2 alone minus baseline | 3 | +0.017 (se 0.012) | +0.010 (se 0.015) | +0.014 (se 0.005) | +0.021 (se 0.010) |

Per seed, the combined option's difference on trained-3 is
+0.048, -0.007, +0.032, +0.025, -0.016: two of five seeds are negative.

## What this does and does not show

- **No sign of overfitting to the training pool.** On the two strong
  opponents training never saw, champions score about +0.59 to +0.61,
  the same as on the three opponents they trained against (+0.60 to
  +0.62). (The +0.91 against the weak unseen opponents is a sanity floor,
  not evidence: those strategies are weak.)
- **The learning options help a little, not conclusively.** The combined
  configuration is better by +0.016 on the trained opponents
  (about 1.4 standard errors) and by a similar,
  equally uncertain amount elsewhere; two of five seeds are negative.
  The two options are not separated: power 0.2 alone and top-5 alone each
  show about +0.017, and together they show +0.016, which is not additive.
  They were adopted as the defaults because they are cheap (top-5 costs
  about 10% more matches per generation) and principled, not because five
  seeds prove them.
- **What top-k selection really changes.** Evolution runs on *training*
  fitness, so top-k does not change which genomes breed; it changes which
  genome is recorded as the generation's champion (and so `best.json`, the
  dashboard and the hall of fame). Its value is a more reliable recorded
  champion, because the training fitness of a generation's best genome is
  dominated by noise: in the v1 run the fixed matches picked a genome other
  than the training best in most generations.
- **Selection fix.** The first version ranked the candidates and reported
  the champion's score on the same fixed matches, which inflated that score
  (v1: +0.667 reported against +0.624 on held-out matches). Candidates are
  now ranked on their own matches; the remaining gap between the fixed-match
  score and the held-out score of a new best (about +0.03 in the v1 run) is
  the winner's curse of taking a maximum over generations, which is why a new
  best is also confirmed on held-out matches and the summary reports that.
- **The hall of fame showed no benefit** against this pool and stays off by
  default (`--hall-of-fame N` enables it). It may matter against pools that
  include evolved opponents. Note that the training-fitness curves jump when
  the hall admits a member, because the opponent mix changes.
- Every champion already beats the strongest hand-written strategy
  (`adaptive:reading,tempo,bully`) in a table of one champion and three
  copies of it, by about +0.6.
