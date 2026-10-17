# Phase 13b: confirming `--features hidden` with fresh seeds

Open work item 5 (ROADMAP). Question: is the Phase 13 result "hidden beats base" a lucky draw?

## Design (no scale-down)

Measured first: 10 generations of the full recipe took 17 s at 4 threads (about 3.5 minutes for
120). That is far cheaper than the "days" the full confirmation was expected to cost, so the
**full neat-v2 recipe was run unchanged**: population 150, 120 generations, 80 matches x 8
rounds, re-evaluation 200, 5 champion candidates, weight power 0.2, pass rule `final`, forced
exchange. Arms A0 (`--features base`) and A2 (`--features hidden`), **fresh seeds 11-20**
(Phase 13 used 1-10), via the unmodified `docs/baselines/phase-13/run.sh` (THREADS=4, build of
the open-work-v0.7 HEAD snapshot). Held-out evaluation with the run.sh eval step: seed 99,
2000 matches x 10 rounds, 4 players, single deck, once against the frozen opponent battery and
once head-to-head against the committed neat-v2 champion. Wall clock: about 1 hour for 20
trainings plus evaluation. The committed champion and all defaults are untouched.

## Result

| test | A2 hidden | A0 base | difference | Welch t (df) | verdict | seeds favouring hidden |
|---|---:|---:|---:|---:|---|---:|
| opponent battery (mixed) | +0.691 +- 0.003 | +0.672 +- 0.003 | +0.019 | +4.40 (17.8) | BETTER | 10/10 |
| head-to-head vs neat-v2 champion | +0.044 +- 0.011 | -0.003 +- 0.009 | +0.047 | +3.26 (16.9) | BETTER | 8/10 |

(Phase 13 seeds 1-10 for comparison: battery +0.009, t = 1.87, not significant; head-to-head
+0.048, t = 3.68.) The head-to-head effect reproduces almost exactly (+0.047 vs +0.048); the
battery effect is now significant and larger (+0.019). Pooling both batches (20 seeds per arm)
gives about +0.014 on the battery and +0.048 head-to-head; I did not run a pooled test.

Per-seed table:

| seed | battery A0 | battery A2 | diff | h2h A0 | h2h A2 | diff |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | +0.670 | +0.706 | +0.036 | -0.021 | +0.067 | +0.089 |
| 12 | +0.681 | +0.686 | +0.005 | +0.005 | +0.027 | +0.022 |
| 13 | +0.670 | +0.684 | +0.015 | +0.048 | -0.007 | -0.055 |
| 14 | +0.673 | +0.681 | +0.007 | -0.033 | +0.048 | +0.081 |
| 15 | +0.674 | +0.678 | +0.004 | +0.003 | +0.082 | +0.080 |
| 16 | +0.664 | +0.692 | +0.028 | +0.008 | -0.011 | -0.019 |
| 17 | +0.664 | +0.690 | +0.027 | +0.004 | +0.026 | +0.021 |
| 18 | +0.657 | +0.695 | +0.038 | -0.038 | +0.049 | +0.087 |
| 19 | +0.693 | +0.694 | +0.001 | +0.026 | +0.099 | +0.074 |
| 20 | +0.674 | +0.704 | +0.029 | -0.031 | +0.059 | +0.089 |

Hidden better on 10/10 seeds (battery), 8/10 seeds (head-to-head vs champion).

Training diagnostics (from `analyse.py`): stall generation of A2 runs 74-119 (A0 21-119), winner's
curse gap +0.030 (A2) vs +0.026 (A0), hidden nodes 8 vs 6 on average. Full analyse.py output
below and in `analysis.md`.

## Caveats

- One table size only (4 players, single deck), as in Phase 13; M1/other sizes untested.
- Same evaluation seed (99) and same champion as Phase 13; not an independent opponent set. The
  champion was trained with base features, which could favour hidden slightly less or more; not
  checked.
- Effect on the battery is small in absolute terms (+0.019 on a saturated scale).
- The 120-generation recipe is the confirmed one. "Longer runs" (300 generations, 200 matches)
  were not tested here and stay open.
- Casino bot (TypeScript, outside this repo) still knows only `base` (20 inputs): a `hidden`
  genome cannot be played there until the inputs are ported.

## Recommendation

Hidden inputs are confirmed on 10 fresh seeds with the full recipe (BETTER on both tests,
|t| > 3). That meets the bar for making `hidden` the default `--features` for new training in this
repo. I did NOT change the default (out of scope; it changes genome width for every default
command and baseline). Decision for the owner.

## Raw analyse.py output

# Phase 13 results (p13)

Caution: with 3 seeds per arm the tests have only 2-4 degrees of freedom, so conclusions are weak; the contested arms are rerun with 5 seeds (SEEDS env of run.sh).

| arm | setting | seeds | held-out mixed (seed mean ± SE) | per seed | per-round role score (mean, sd) | vs neat-v2 champion | stall gen | curse gap | hidden nodes |
|---|---|---:|---:|---|---|---:|---|---:|---:|
| A0 | 4p-single | 10 | +0.672 ± 0.003 | s11:+0.670, s12:+0.681, s13:+0.670, s14:+0.673, s15:+0.674, s16:+0.664, s17:+0.664, s18:+0.657, s19:+0.693, s20:+0.674 | +0.672, 0.446 | -0.003 ± 0.009 | 119, 117, 97, 73, 21, 119, 91, 76, 104, 119 | +0.026 | 6 |
| A2 | 4p-single | 10 | +0.691 ± 0.003 | s11:+0.706, s12:+0.686, s13:+0.684, s14:+0.681, s15:+0.678, s16:+0.692, s17:+0.690, s18:+0.695, s19:+0.694, s20:+0.704 | +0.691, 0.432 | +0.044 ± 0.011 | 119, 119, 107, 119, 80, 74, 108, 119, 105, 90 | +0.030 | 8 |

## Comparisons (per setting, never across table sizes)

| arm | setting | control | difference | Welch t | df | crit (95%) | verdict | z (2-SE rule) |
|---|---|---|---:|---:|---:|---:|---|---:|
| A2 | 4p-single | A0 | +0.019 | +4.40 | 17.8 | 2.13 | BETTER | +4.4 |

## Head-to-head against the committed neat-v2 champion (per-seed scores, same rule)

The default battery saturates against weak opponents; the champion is the discriminating test.

| arm | setting | control | difference | Welch t | df | crit (95%) | verdict | z (2-SE rule) |
|---|---|---|---:|---:|---:|---:|---|---:|
| A2 | 4p-single | A0 | +0.047 | +3.26 | 16.9 | 2.13 | BETTER | +3.3 |

Rule: BETTER/WORSE only when |t| exceeds the two-sided 95% critical value of Student's t at the Welch-Satterthwaite degrees of freedom (the table lookup rounds the df down, i.e. conservatively); otherwise no difference. B1 is compared against C2 (same evaluation budget) and A2. The z of the 2-SE rule is for reference only.
