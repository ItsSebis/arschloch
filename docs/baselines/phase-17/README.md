# Phase 17 real-data check: `cli analyse-run`

The analysis tool (`docs/superpowers/specs/2026-10-11-phase-17-run-analysis-design.md`,
numbers explained in `docs/STATISTICS.md`, usage in `docs/TRAINING.md` "Analysing a
run") run on real runs that existed before it: the four long runs of the
`runs/PC` folder (loose `.jsonl` files) and the two ten-seed sets of Phase 13
(`docs/baselines/phase-13/README.md`: A0 = the 20 original inputs, `base`; A2 =
`--features hidden`). Nothing was trained for this check, and no number below
is typed by hand: they are the tool's output, pasted. The raw runs are local
(`runs/PC` is not in the repository, the Phase 13 sets are in
`~/.cache/phase13`), so this file is the record.

## Commands

```bash
CARGO_TARGET_DIR=/home/sebi/.cache/target-p17docs cargo build --release -p cli
cli=/home/sebi/.cache/target-p17docs/release/cli
$cli analyse-run runs/PC --json pc.json                     # a folder of loose .jsonl files
$cli analyse-run /home/sebi/.cache/phase13/A0 --json a0.json   # a set dir: ten runs, seed1..seed10
$cli analyse-run /home/sebi/.cache/phase13/A2 --json a2.json
```

## (a) The four long runs in `runs/PC`

Each `.jsonl` file is one run, labelled by its file name. `firstLongRun` has no
`run_end` line (it was stopped), so it has no best generation, no final scores
and no winner's-curse gap; everything else is computed from its generations.

```
== comparison ==
run             gens  best gen  stall  final re-eval  last-third /100  curse gap         verdict
3rd-veryLong    300   30        58     0.415          -0.001           +0.044 (+2.7 SE)  noise_limited,pool_limited,converged,stop_early
4th-ontopOf3rd  98    79        97     0.080          +0.028           +0.039 (+2.2 SE)  still_improving,noise_limited,pool_limited,growth_without_gain
firstLongRun    76    -         23     -              -0.002           -                 growth_without_gain,stop_early
secondLongRun   50    45        25     0.444          +0.177           +0.050 (+3.3 SE)  noise_limited,pool_limited,stop_early
```

The full report of the longest run (300 generations):

```
== 3rd-veryLong ==
  generations                          300
  best generation                      30
  final re-evaluation                  0.415 +- 0.023
  held-out                             0.371 +- 0.016
  stall generation                     58 (smoothed peak 0.386)
  noise                                one evaluation +- 0.024, generation-to-generation sd 0.023
  trend, whole run /100 gen            +0.004 [+0.002, +0.006]
  trend, last third /100 gen           -0.001 [-0.011, +0.010]
  winner's curse (re-eval - held-out)  +0.044 (+2.7 SE)
  train fitness - re-eval (mean)       +0.036
  hidden nodes / connections           0 -> 13 / 21 -> 86
  species / threshold                  1 -> 7 / 0.50 -> 0.30
  top fitness bin (last third)         65%

Per opponent (champion score):
opponent                                   class   first third  last third  /100 gen
Neat(0-best)                               flat    0.281        0.288       +0.003 [-0.000, +0.007]
Neat(1-best5P)                             rising  0.635        0.639       +0.003 [+0.001, +0.006]
Adaptive(reading,deception=1,tempo,bully)  rising  0.621        0.628       +0.003 [+0.001, +0.006]
LowestLegal                                flat    0.458        0.467       +0.005 [-0.005, +0.014]

Verdict:
  [noise_limited] progress is below the noise (last-third trend -0.0005 [-0.0106, +0.0095] per 100 generations, change sd 0.0228 vs single se 0.0237, re-evaluation minus held-out 2.68 held-out se): more --matches-per-genome, the duplicate-deal skill fitness, several seeds and choose the winner on fresh matches
  [pool_limited] the pool no longer teaches against the flat opponents (rising: Neat(1-best5P), Adaptive(reading,deception=1,tempo,bully); flat: Neat(0-best), LowestLegal): harder or additional opponents
  [converged] the population has converged (65% of the genomes in the top fitness bin over the last third): raise mutation rates / species target
  [stop_early] the last 241 generations added nothing (stall generation 58 of 299): stop earlier
```

**Reading.**
- `3rd-veryLong`: the best generation is 30 of 300 and the smoothed curve is
  within one single-evaluation standard error of its peak from generation 58
  on; the last third has no trend (-0.001, interval from -0.011 to +0.010 per
  100 generations). The best generation looked +0.044 (2.7 held-out standard
  errors) better on the matches it was chosen on than on fresh ones, so most of
  its lead is selection luck. One evaluation is +- 0.024 and the curve moves
  by about 0.023 from generation to generation: the run is noise-limited, and
  the last 241 generations added nothing the measurement can see. The
  population sits 65% in the top fitness bin (converged). The two opponents
  called rising have slopes of +0.003 per 100 generations: significant on paper,
  worth nothing in practice (see the caveats).
- `4th-ontopOf3rd` (continued from the third): the stall generation is the last
  one, so the rule calls it still improving, but the last-third interval
  (-0.113 to +0.168) is wide and contains 0, and the held-out score is 0.041,
  barely above 0: it is also noise-limited (curse +2.2 SE).
- `firstLongRun` and `secondLongRun`: the first plateaus around generation 23 of
  76, the second around 25 of 50 (with a +3.3 SE curse and a wide last-third
  interval): both stopped improving in the first half.
- `secondLongRun` lists `Neat(2-best)` and `Neat(3-best)` with identical
  numbers; the two opponent models are evidently the same genome. The tool
  reports what the events say.

## (b) Phase 13, ten seeds each: A0 (`base`) and A2 (`--features hidden`)

120 generations per run. A0:

```
== comparison ==
run     gens  best gen  stall  final re-eval  last-third /100  curse gap         verdict
seed1   120   51        109    0.639          -0.040           +0.031 (+2.4 SE)  no_clear_limit
seed10  120   105       84     0.642          +0.022           +0.052 (+4.1 SE)  noise_limited,growth_without_gain
seed2   120   92        119    0.655          -0.033           +0.039 (+2.9 SE)  still_improving,noise_limited,growth_without_gain
seed3   120   60        119    0.647          -0.055           +0.024 (+1.8 SE)  still_improving
seed4   120   24        119    0.608          -0.033           -0.003 (-0.3 SE)  still_improving,growth_without_gain
seed5   120   83        119    0.649          -0.145           +0.007 (+0.6 SE)  still_improving
seed6   120   104       92     0.649          +0.062           +0.026 (+2.0 SE)  still_improving
seed7   120   95        90     0.648          +0.026           +0.036 (+2.8 SE)  noise_limited,growth_without_gain
seed8   120   113       97     0.667          +0.076           +0.066 (+4.9 SE)  still_improving
seed9   120   65        114    0.652          +0.050           +0.013 (+1.1 SE)  still_improving
```

A2:

```
== comparison ==
run     gens  best gen  stall  final re-eval  last-third /100  curse gap         verdict
seed1   120   117       93     0.653          +0.075           +0.019 (+1.5 SE)  still_improving
seed10  120   116       92     0.685          +0.045           +0.022 (+2.1 SE)  still_improving
seed2   120   112       98     0.684          +0.078           +0.009 (+0.8 SE)  still_improving
seed3   120   100       119    0.683          -0.056           +0.041 (+3.6 SE)  still_improving,converged
seed4   120   81        119    0.634          -0.013           -0.003 (-0.2 SE)  still_improving
seed5   120   93        119    0.664          -0.070           +0.004 (+0.4 SE)  still_improving
seed6   120   104       110    0.670          +0.174           +0.015 (+1.4 SE)  still_improving
seed7   120   117       92     0.682          +0.054           +0.028 (+2.4 SE)  still_improving
seed8   120   59        119    0.670          +0.017           +0.048 (+3.8 SE)  still_improving,noise_limited,growth_without_gain
seed9   120   107       101    0.690          +0.074           +0.015 (+1.4 SE)  still_improving
```

One A0 run in full (`seed5`, whose last third is clearly falling):

```
== seed5 ==
  generations                          120
  best generation                      83
  final re-evaluation                  0.649 +- 0.016
  held-out                             0.642 +- 0.012
  stall generation                     119 (smoothed peak 0.644)
  noise                                one evaluation +- 0.019, generation-to-generation sd 0.023
  trend, whole run /100 gen            +0.113 [+0.090, +0.137]
  trend, last third /100 gen           -0.145 [-0.191, -0.099]
  winner's curse (re-eval - held-out)  +0.007 (+0.6 SE)
  train fitness - re-eval (mean)       +0.017
  hidden nodes / connections           0 -> 12 / 21 -> 51
  species / threshold                  1 -> 7 / 0.50 -> 0.40
  top fitness bin (last third)         35%
  seconds per stage: training 160.5s, selection 12.8s, reproduction 0.1s, re-eval mixed 10.0s, re-eval per opponent 10.0s, hall of fame 0.0s, confirmation 0.8s, decision sample 0.0s, files 1.3s

Per opponent (champion score):
opponent                       class   first third  last third  /100 gen
LowestLegal                    rising  0.515        0.600       +0.112 [+0.087, +0.137]
EndgameDenial                  rising  0.588        0.643       +0.076 [+0.058, +0.094]
Adaptive(reading,tempo,bully)  rising  0.501        0.586       +0.111 [+0.089, +0.134]

Verdict:
  [still_improving] still improving at the end (stall generation 119 of 119, last-third trend -0.1450 [-0.1915, -0.0985] per 100 generations): train longer (--resume/--from)
```

**Reading.** All twenty runs have a best generation (all ended normally) and
a positive-or-zero curse in all but two runs (seed4 in both sets, -0.3 and -0.2
SE): the best generation is usually partly luck, between +0.4 and +4.9 held-out
standard errors in the others. The final re-evaluation is higher for A2 (0.634
to 0.690) than for A0 (0.608 to 0.667), which is what the Phase 13 head-to-head
measured (+0.048): the analysis does not measure that, it looks at the
shape of each curve. Seven of the ten A0 runs and all ten A2 runs carry `still_improving`; the
stall generation is 84 or later (of 119) in all of them, so 120 generations
is not enough to see the plateau in either set. `noise_limited` appears in
three A0 runs (seed10, seed2, seed7) and one A2 run (seed8), `growth_without_gain`
in four A0 runs and one A2 run, and `converged` in one A2 run (seed3).

## Caveats

- The verdict rules are heuristics with fixed thresholds (stall within 95% /
  60% of the run, change sd 2x the single evaluation error, curse 1 held-out
  standard error, 50% of genomes in the top bin), not tests. A run just inside
  or outside a threshold gets a different verdict for a small reason.
- The trend interval treats consecutive generations as independent. They are
  not (the champion of one generation descends from the one before), so
  intervals are too narrow and tiny slopes such as +0.003 per 100 generations
  come out "rising". Judge a trend by its size against the noise, not by its
  class alone.
- When the last smoothed value falls out of the one-standard-error band of its
  peak, the stall index is the last generation. A run that peaked and is now
  falling can then show `still_improving` (several A0 and A2 seeds with a
  negative last-third trend do); read the trend and the curves with the
  verdict. This is reported to the author of the analysis as a possible rule
  fix, not corrected here.
- Short runs have wide intervals: 50 to 120 generations give last-third
  intervals that contain 0 whether or not there is progress.
- "Seeds" in A0/A2 are single runs; the tool analyses each run alone, the
  comparison table is only a side-by-side listing, not a statistical comparison
  of the two sets.
