# Phase 17 real-data check: `cli analyse-run`

The analysis tool (`docs/superpowers/specs/2026-10-11-phase-17-run-analysis-design.md`,
numbers explained in `docs/STATISTICS.md`, usage in `docs/TRAINING.md` "Analysing a
run") run on real runs that existed before it: the four long runs of the
`runs/PC` folder (loose `.jsonl` files) and the two ten-seed sets of Phase 13
(`docs/baselines/phase-13/README.md`: A0 = the 20 original inputs, `base`; A2 =
`--features hidden`). Nothing was trained for this check, and no number below
is typed by hand: they are the tool's output, pasted (this file was regenerated
after the review fixes of the verdict rules: a falling curve is no longer
"still improving", intervals are corrected for autocorrelation, opponent
classes use the last third, the curse arm of `noise_limited` uses combined
standard errors). The raw runs are local (`runs/PC` is not in the repository,
the Phase 13 sets are in `~/.cache/phase13`), so this file is the record; the
committed tests cover the same logic on a 12-generation excerpt of A0 seed 5
(`sim/src/training/analysis/fixtures/`) and, where the runs are on disk, on
all ten A0 seeds against `docs/baselines/phase-13/analyse.py`.

## Commands

```bash
CARGO_TARGET_DIR=/home/sebi/.cache/target-p17fix cargo build --release -p cli
cli=/home/sebi/.cache/target-p17fix/release/cli
$cli analyse-run runs/PC --json pc.json                        # a folder of loose .jsonl files
$cli analyse-run /home/sebi/.cache/phase13/A0 --json a0.json   # a set dir: ten runs, seed1..seed10
$cli analyse-run /home/sebi/.cache/phase13/A2 --json a2.json
```

## How often each verdict fires

Counted from the three `--json` files (a run can carry several verdicts, so
the columns add up to more than the number of runs; no run ended with
`no_clear_limit`).

| verdict | runs/PC (4 runs) | Phase 13 A0 (10) | Phase 13 A2 (10) | all 24 |
|---|---|---|---|---|
| `still_improving` | 1 | 3 | 6 | 10 |
| `declining` | 1 | 4 | 3 | 8 |
| `noise_limited` | 0 | 0 | 2 | 2 |
| `pool_limited` | 0 | 1 | 0 | 1 |
| `model_limited` | 3 | 4 | 1 | 8 |
| `growth_without_gain` | 2 | 5 | 2 | 9 |
| `converged` | 1 | 0 | 1 | 2 |
| `stop_early` | 2 | 0 | 0 | 2 |
| `no_clear_limit` | 0 | 0 | 0 | 0 |

## (a) The four long runs in `runs/PC`

Each `.jsonl` file is one run, labelled by its file name. `firstLongRun` has no
`run_end` line (it was stopped), so it has no best generation, no final scores
and no winner's-curse gap; everything else is computed from its generations.
`4th-ontopOf3rd` continues the third run: its generations are numbered up to
99 although the file holds 98 of them, which is why the stall and peak are
reported as generation numbers.

```
== comparison ==
run             gens  best gen  stall gen  final re-eval  last-third /100  curse gap                  verdict
3rd-veryLong    300   30        58         0.415          -0.001           +0.044 (+1.5 combined SE)  model_limited,growth_without_gain,converged,stop_early
4th-ontopOf3rd  98    79        99         0.080          +0.028           +0.039 (+1.3 combined SE)  declining,model_limited
firstLongRun    76    -         23         -              -0.002           -                          model_limited,growth_without_gain,stop_early
secondLongRun   50    45        25         0.444          +0.177           +0.050 (+1.9 combined SE)  still_improving
```

Two full reports (the longest run, and the continuation):

```
== 3rd-veryLong ==
  generations                          300
  best generation                      30
  final re-evaluation                  0.415 +- 0.023
  held-out                             0.371 +- 0.016
  stall generation                     58 (last generation 299), smoothed peak 0.386 at generation 283
  noise                                one evaluation +- 0.024, generation-to-generation sd 0.023
  trend, whole run /100 gen            +0.004 [+0.001, +0.007]
  trend, last third /100 gen           -0.001 [-0.011, +0.010]
  winner's curse (re-eval - held-out)  +0.044 (+1.5 combined SE)
  train fitness - re-eval (mean)       +0.036
  hidden nodes / connections           0 -> 13 / 21 -> 86
  species / threshold                  1 -> 7 / 0.50 -> 0.30
  top fitness bin (last third)         65%

Per opponent (champion score; class from the last third):
opponent                                   class  first third  last third  last third /100 gen      whole run /100 gen
Neat(0-best)                               flat   0.281        0.288       +0.003 [-0.012, +0.017]  +0.003 [-0.001, +0.007]
Neat(1-best5P)                             flat   0.635        0.639       +0.011 [-0.004, +0.025]  +0.003 [-0.000, +0.007]
Adaptive(reading,deception=1,tempo,bully)  flat   0.621        0.628       +0.004 [-0.008, +0.016]  +0.003 [+0.000, +0.006]
LowestLegal                                flat   0.458        0.467       -0.008 [-0.077, +0.061]  +0.005 [-0.007, +0.017]

Verdict:
  [model_limited] flat everywhere above the noise (last-third trend -0.0005 [-0.0111, +0.0100] per 100 generations, no opponent rising): new inputs (--features hidden) or a different fitness
  [growth_without_gain] the network keeps growing without gain (mean hidden nodes 12.3 over the first 3 generations of the last third -> 15.7 over its last 3; last-third trend -0.0005 [-0.0111, +0.0100] per 100 generations): stagnation limit / complexity penalty
  [converged] the population has converged (65% of the genomes in the top fitness bin over the last third): raise mutation rates / species target
  [stop_early] after generation 58 the smoothed score stayed within one noise unit (0.024) of its peak 0.386: the last 241 generations (up to generation 299) added nothing visible; stop earlier
```

```
== 4th-ontopOf3rd ==
  generations                          98
  best generation                      79
  final re-evaluation                  0.080 +- 0.024
  held-out                             0.041 +- 0.017
  stall generation                     99 (last generation 99), smoothed peak 0.062 at generation 85, final smoothed value below the peak band
  noise                                one evaluation +- 0.028, generation-to-generation sd 0.023
  trend, whole run /100 gen            +0.049 [+0.012, +0.086]
  trend, last third /100 gen           +0.028 [-0.309, +0.365]
  winner's curse (re-eval - held-out)  +0.039 (+1.3 combined SE)
  train fitness - re-eval (mean)       +0.040
  hidden nodes / connections           22 -> 17 / 93 -> 106
  species / threshold                  12 -> 8 / 0.25 -> 0.30
  top fitness bin (last third)         41%

Per opponent (champion score; class from the last third):
opponent          class    first third  last third  last third /100 gen      whole run /100 gen
Neat(0-best)      falling  0.043        0.037       -0.131 [-0.248, -0.013]  -0.014 [-0.035, +0.006]
Neat(1-best)      flat     0.022        0.086       +0.027 [-0.529, +0.582]  +0.086 [+0.028, +0.144]
Neat(2-gen-0299)  flat     0.064        0.123       +0.047 [-0.419, +0.512]  +0.083 [+0.034, +0.133]

Verdict:
  [declining] the smoothed score peaked at generation 85 (0.062), now 0.015 (last-third trend +0.0278 [-0.3094, +0.3649] per 100 generations): the final champion is not the best one; use the best generation's genome / stop earlier
  [model_limited] flat everywhere above the noise (last-third trend +0.0278 [-0.3094, +0.3649] per 100 generations, no opponent rising): new inputs (--features hidden) or a different fitness
```

**Reading.**
- `3rd-veryLong`: the best generation is 30 of 300 and the smoothed curve is
  within one single-evaluation standard error of its peak from generation 58
  on (the peak itself, 0.386, is at generation 283); the last third has no trend
  (-0.001, interval from -0.011 to +0.010 per 100 generations) and no opponent's
  last-third trend is rising. The best generation looked +0.044 better on the
  matches it was chosen on than on fresh ones, which is +1.5 standard errors of
  that difference: below the 2 the verdict needs, so with the corrected rule
  the run is `model_limited`, not `noise_limited` as the first version of the
  tool said (it measured the gap in held-out standard errors, +2.7, which
  selection bias makes large almost always). The change standard deviation
  (0.023) is no more than twice one evaluation (0.024), so the noise arm does
  not fire either. The population sits 65% in the top fitness bin (converged),
  the network kept growing (mean hidden nodes 12.3 -> 15.7 over the last
  third), and the last 241 generations added nothing the measurement can see.
- `4th-ontopOf3rd` (continued from the third): the smoothed score peaked at
  generation 85 (0.062) and ends at 0.015, below the band: `declining`, with
  the best generation 79 the better genome to use. The last-third interval
  (-0.309 to +0.365) is very wide and contains 0 (the autocorrelation
  correction widened it; the first version showed -0.113 to +0.168), so
  `model_limited` also fires. The whole-run trend is clearly positive
  (+0.049 [+0.012, +0.086]): the run improved early and then fell back. It was
  called `still_improving` before the fix, because its stall index is the
  last generation by the fallback.
- `firstLongRun` plateaus around generation 23 of 76 (`stop_early`), with no
  opponent rising in the last third (`model_limited`) and growing networks.
- `secondLongRun` (50 generations) is `still_improving` only: its smoothed peak
  is at generation 48 of 49 and the last-third slope is +0.177 per 100
  generations, but with a wide interval that contains 0, so this is the rule
  that the peak at the end triggers, on a short run. Its best generation is 45.
- `secondLongRun` lists `Neat(2-best)` and `Neat(3-best)` with identical
  numbers; the two opponent models are evidently the same genome. The tool
  reports what the events say.

## (b) Phase 13, ten seeds each: A0 (`base`) and A2 (`--features hidden`)

120 generations per run. A0:

```
== comparison ==
run     gens  best gen  stall gen  final re-eval  last-third /100  curse gap                  verdict
seed1   120   51        109        0.639          -0.040           +0.031 (+1.4 combined SE)  model_limited,growth_without_gain
seed10  120   105       84         0.642          +0.022           +0.052 (+2.5 combined SE)  still_improving
seed2   120   92        119        0.655          -0.033           +0.039 (+1.8 combined SE)  declining,model_limited,growth_without_gain
seed3   120   60        119        0.647          -0.055           +0.024 (+1.1 combined SE)  declining
seed4   120   24        119        0.608          -0.033           -0.003 (-0.2 combined SE)  declining,model_limited,growth_without_gain
seed5   120   83        119        0.649          -0.145           +0.007 (+0.3 combined SE)  declining
seed6   120   104       92         0.649          +0.062           +0.026 (+1.2 combined SE)  pool_limited,growth_without_gain
seed7   120   95        90         0.648          +0.026           +0.036 (+1.6 combined SE)  model_limited,growth_without_gain
seed8   120   113       97         0.667          +0.076           +0.066 (+3.0 combined SE)  still_improving
seed9   120   65        114        0.652          +0.050           +0.013 (+0.6 combined SE)  still_improving
```

A2:

```
== comparison ==
run     gens  best gen  stall gen  final re-eval  last-third /100  curse gap                  verdict
seed1   120   117       93         0.653          +0.075           +0.019 (+0.8 combined SE)  still_improving
seed10  120   116       92         0.685          +0.045           +0.022 (+1.2 combined SE)  still_improving
seed2   120   112       98         0.684          +0.078           +0.009 (+0.5 combined SE)  still_improving
seed3   120   100       119        0.683          -0.056           +0.041 (+2.1 combined SE)  declining,noise_limited,growth_without_gain,converged
seed4   120   81        119        0.634          -0.013           -0.003 (-0.1 combined SE)  declining,model_limited
seed5   120   93        119        0.664          -0.070           +0.004 (+0.2 combined SE)  declining
seed6   120   104       110        0.670          +0.174           +0.015 (+0.8 combined SE)  still_improving
seed7   120   117       92         0.682          +0.054           +0.028 (+1.3 combined SE)  still_improving
seed8   120   59        119        0.670          +0.017           +0.048 (+2.3 combined SE)  noise_limited,growth_without_gain
seed9   120   107       101        0.690          +0.074           +0.015 (+0.8 combined SE)  still_improving
```

Two A0 runs in full: `seed1` (a plateau that has just started to fall) and
`seed5` (whose last third is clearly falling):

```
== seed1 ==
  generations                          120
  best generation                      51
  final re-evaluation                  0.639 +- 0.017
  held-out                             0.608 +- 0.013
  stall generation                     109 (last generation 119), smoothed peak 0.621 at generation 97
  noise                                one evaluation +- 0.019, generation-to-generation sd 0.026
  trend, whole run /100 gen            +0.088 [+0.051, +0.126]
  trend, last third /100 gen           -0.040 [-0.082, +0.003]
  winner's curse (re-eval - held-out)  +0.031 (+1.4 combined SE)
  train fitness - re-eval (mean)       +0.014
  hidden nodes / connections           0 -> 8 / 21 -> 46
  species / threshold                  1 -> 7 / 0.50 -> 0.25
  top fitness bin (last third)         34%
  seconds per stage: training 211.3s, selection 18.1s, reproduction 0.1s, re-eval mixed 14.1s, re-eval per opponent 14.2s, hall of fame 0.0s, confirmation 0.6s, decision sample 0.0s, files 1.4s

Per opponent (champion score; class from the last third):
opponent                       class    first third  last third  last third /100 gen      whole run /100 gen
LowestLegal                    flat     0.549        0.609       -0.027 [-0.089, +0.034]  +0.078 [+0.040, +0.116]
EndgameDenial                  falling  0.589        0.663       -0.051 [-0.082, -0.021]  +0.094 [+0.061, +0.128]
Adaptive(reading,tempo,bully)  falling  0.503        0.572       -0.042 [-0.075, -0.008]  +0.086 [+0.040, +0.131]

Verdict:
  [model_limited] flat everywhere above the noise (last-third trend -0.0395 [-0.0820, +0.0030] per 100 generations, no opponent rising): new inputs (--features hidden) or a different fitness
  [growth_without_gain] the network keeps growing without gain (mean hidden nodes 5.0 over the first 3 generations of the last third -> 10.3 over its last 3; last-third trend -0.0395 [-0.0820, +0.0030] per 100 generations): stagnation limit / complexity penalty
```

```
== seed5 ==
  generations                          120
  best generation                      83
  final re-evaluation                  0.649 +- 0.016
  held-out                             0.642 +- 0.012
  stall generation                     119 (last generation 119), smoothed peak 0.644 at generation 86, final smoothed value below the peak band
  noise                                one evaluation +- 0.019, generation-to-generation sd 0.023
  trend, whole run /100 gen            +0.113 [+0.038, +0.189]
  trend, last third /100 gen           -0.145 [-0.197, -0.093]
  winner's curse (re-eval - held-out)  +0.007 (+0.3 combined SE)
  train fitness - re-eval (mean)       +0.017
  hidden nodes / connections           0 -> 12 / 21 -> 51
  species / threshold                  1 -> 7 / 0.50 -> 0.40
  top fitness bin (last third)         35%
  seconds per stage: training 160.5s, selection 12.8s, reproduction 0.1s, re-eval mixed 10.0s, re-eval per opponent 10.0s, hall of fame 0.0s, confirmation 0.8s, decision sample 0.0s, files 1.3s

Per opponent (champion score; class from the last third):
opponent                       class    first third  last third  last third /100 gen      whole run /100 gen
LowestLegal                    falling  0.515        0.600       -0.142 [-0.253, -0.031]  +0.112 [+0.038, +0.187]
EndgameDenial                  falling  0.588        0.643       -0.119 [-0.176, -0.062]  +0.076 [+0.020, +0.133]
Adaptive(reading,tempo,bully)  falling  0.501        0.586       -0.160 [-0.211, -0.108]  +0.111 [+0.041, +0.182]

Verdict:
  [declining] the smoothed score peaked at generation 86 (0.644), now 0.592 (last-third trend -0.1450 [-0.1968, -0.0932] per 100 generations): the final champion is not the best one; use the best generation's genome / stop earlier
```

**Reading.** The curves split into three kinds. (1) Runs that peaked and fell
back: `declining` in four A0 runs (seeds 2, 3, 4, 5) and three A2 runs (3, 4, 5).
These are the runs the first version of the tool called `still_improving`
because the fallback of the stall rule put the stall at the last generation;
the corrected rule names what happened (the final champion is not the best
one; for A0 seed5 the smoothed score peaked at generation 86 and the last
third falls by -0.145 per 100 generations, interval [-0.197, -0.093]).
(2) Runs still rising at the end: `still_improving` in three A0 runs (seeds 8,
9, 10) and six A2 runs (1, 2, 6, 7, 9, 10); 120 generations is not enough to
see their plateau, and more of the A2 runs are in this class than of the A0
runs (6 of 10 against 3 of 10). (3) Plateaus with a flat last third:
`model_limited` (four A0 runs, one A2 run: flat, no opponent rising, no clear
noise signal), `noise_limited` (two A2 runs, seeds 3 and 8: the curse gap is
+2.1 and +2.3 combined standard errors), `pool_limited` once (A0 seed6: one
opponent rising and two flat in the last third, champion trend flat). `growth_without_gain` is the
most frequent secondary verdict (9 of 24 runs). No run was `stop_early`
in the Phase 13 sets: either the best generation is later than the stall or
the stall is beyond 60% of the run. The curse gaps are positive in 21 of the 23
runs (seed4 of both A0 and A2 are the exceptions, -0.2 and -0.1 combined SE),
between +0.2 and +3.0 combined standard errors. The final re-evaluation is higher for A2 (0.634
to 0.690) than for A0 (0.608 to 0.667), which is what the Phase 13 head-to-head
measured: the analysis does not measure that, it looks at the shape of each
curve.

## Caveats

- The verdict rules are heuristics with fixed thresholds (peak within 5% of
  the end, stall within 60% of the run, change sd 2x the single evaluation
  error, curse 2 combined standard errors, 50% of genomes in the top bin),
  not tests. A run just inside or outside a threshold gets a different verdict
  for a small reason; the change from the first version (`still_improving` in
  18 of 24 runs then, 10 now) shows how much a rule definition moves the
  counts.
- The trend interval is corrected for lag-1 autocorrelation of the residuals
  (standard error times sqrt((1 + r) / (1 - r)), capped at 5). That is the
  correction of a first-order autoregressive process, estimated from the same
  few points, so it is approximate; it still can be too narrow for strongly
  structured curves, and r estimated from 40 points is itself noisy. Judge a
  trend by its size against the noise, not by its class alone.
- `declining` uses the same one-single-evaluation-error band as the stall rule:
  an end that is lower than the peak by less than that is not reported, and
  one that is lower by more may still be a dip of the last ten generations.
- `model_limited` is the residual explanation (flat, nothing rising, not
  noisy by the two thresholds). It fires on 8 of 24 runs, including A0 seed1
  where two opponents' last-third trends are falling: "flat everywhere" there
  means the champion's own trend interval contains 0, and the label is a hint,
  not a diagnosis.
- Short runs have wide intervals: 50 to 120 generations give last-third
  intervals that contain 0 whether or not there is progress.
- "Seeds" in A0/A2 are single runs; the tool analyses each run alone, the
  comparison table is only a side-by-side listing, not a statistical comparison
  of the two sets.
