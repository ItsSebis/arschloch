# Phase 13 baseline: NEAT inputs, fitness variants and one model for every table

Design: `docs/superpowers/specs/2026-10-10-phase-13-neat-design.md`. Harness:
`run.sh` (trains and evaluates every arm, resumable) and `analyse.py` (tables and
verdicts, `--selftest`). Measured on a laptop (8 threads, optimised build from
Phase 19): one 120-generation run of the neat-v2 recipe takes about 3 minutes.

## What was compared

Every arm trains with the neat-v2 recipe (population 150, 120 generations, 80
matches x 8 rounds per genome, re-evaluation 200, 5 champion candidates, weight
power 0.2, pass rule `final`, forced exchange) plus the arm's flags, then each
genome is scored on held-out matches (seed 99, 2000 matches x 10 rounds): once on
the frozen opponent battery, once head-to-head against the committed neat-v2
champion (the discriminating test: the battery saturates). An arm counts as
better or worse **only** when a Welch t-test between the per-seed scores of the
arm and its control is significant at the two-sided 95% level; otherwise "no
difference". Seeds: 10 for A0 and A2, 5 for B1/B2/B3/C2, 3 for A1, S3, S5D, M1.

| arm | flags | question |
|---|---|---|
| A0 | `--features base` | control: the 20 original inputs |
| A1 | `--features trick` | does knowing who is still in the trick help? |
| A2 | `--features hidden` | plus hidden-hand prediction inputs |
| B1 | A2 + racing (top 20%, 100 extra matches) | noise-aware selection |
| C2 | A2 with 100 matches per genome | matched-evaluation-budget control for B1 |
| B2 | A2 + `--fitness risk:0.5` | stable rank (mean minus 0.5 x spread) |
| B3 | A2 + `--fitness tanh:4` | concave: Vize almost as good as President |
| M1 | `--features all`, sizes 3-6, both decks | one model for every table |
| S3, S5D | base, 3 players single / 5 players double | specialist controls for M1 |

Step 0: the Phase 19 build reproduces `docs/baselines/neat-v2/champion.json` byte
for byte.

## Verdicts in plain language

1. **Hidden-hand inputs (`--features hidden`) are the one clear win.** Head-to-head
   against the neat-v2 champion A2 scores +0.048 better than A0 (Welch t = 3.68
   at 18 degrees of freedom, BETTER). On the plain opponent battery the gain is
   +0.009 (t = 1.87, not significant): that battery is nearly saturated, so a
   small edge hides there. A1 (trick only) is indistinguishable from A0 on
   three seeds (too few seeds to say much).
2. **Fitness variants and racing change nothing measurable.** B1, B2, B3 and C2
   all land within about one standard error of A2 on both tests (|t| <= 1.7).
   B2 (risk 0.5) is nominally highest head-to-head (+0.014) but not
   significantly. They stay opt-in. Racing's expected benefit (less winner's
   curse) is visible only as a smaller curse gap (B1 +0.007 against +0.020 for
   A2), not as a stronger champion.
3. **One model for every table works, with one exception.** M1 matches the
   specialists at 5 players double (+0.004, no difference) and at 4 players single
   (+0.011 against A0, no difference), but is clearly worse at 3 players single
   head-to-head (-0.088, t = -3.85, WORSE; on the battery -0.036 is not
   significant). Use one model where convenience matters; train a 3-player
   specialist where 3-player strength matters.
4. **The runs are not capacity-limited.** Most runs have their best generation
   at 90-119 of 120 (no stall), and the winner's-curse gap is 0.005-0.03, so
   longer runs and more matches per genome are the cheapest next experiments
   (see the conclusion).

Nothing here changes a default: `--features` stays `base` so every existing
command, genome file and baseline behaves exactly as before. Genome files are
prefix-compatible, so a `hidden` genome carries the base weights as its prefix.

## Full tables (output of `analyse.py /home/sebi/.cache/phase13`)

# Phase 13 results (/home/sebi/.cache/phase13)

Caution: with 3 seeds per arm the tests have only 2-4 degrees of freedom, so conclusions are weak; the contested arms are rerun with 5 seeds (SEEDS env of run.sh).

| arm | setting | seeds | held-out mixed (seed mean ± SE) | per seed | per-round role score (mean, sd) | vs neat-v2 champion | stall gen | curse gap | hidden nodes |
|---|---|---:|---:|---|---|---:|---|---:|---:|
| A0 | 4p-single | 10 | +0.673 ± 0.004 | s1:+0.674, s2:+0.677, s3:+0.684, s4:+0.653, s5:+0.681, s6:+0.678, s7:+0.674, s8:+0.679, s9:+0.681, s10:+0.652 | +0.673, 0.450 | -0.015 ± 0.009 | 109, 119, 119, 119, 119, 92, 90, 97, 114, 84 | +0.029 | 7 |
| A1 | 4p-single | 3 | +0.671 ± 0.013 | s1:+0.644, s2:+0.685, s3:+0.685 | +0.671, 0.452 | -0.030 ± 0.029 | 119, 119, 119 | +0.015 | 7 |
| A2 | 4p-single | 10 | +0.683 ± 0.003 | s1:+0.674, s2:+0.694, s3:+0.693, s4:+0.663, s5:+0.677, s6:+0.684, s7:+0.684, s8:+0.672, s9:+0.696, s10:+0.688 | +0.683, 0.434 | +0.033 ± 0.009 | 93, 98, 119, 119, 119, 110, 92, 119, 101, 92 | +0.020 | 7 |
| B1 | 4p-single | 5 | +0.692 ± 0.009 | s1:+0.712, s2:+0.672, s3:+0.680, s4:+0.714, s5:+0.681 | +0.692, 0.436 | +0.032 ± 0.026 | 113, 119, 102, 111, 100 | +0.007 | 5 |
| B2 | 4p-single | 5 | +0.692 ± 0.006 | s1:+0.705, s2:+0.701, s3:+0.695, s4:+0.685, s5:+0.672 | +0.692, 0.430 | +0.047 ± 0.017 | 119, 105, 91, 119, 65 | +0.009 | 6 |
| B3 | 4p-single | 5 | +0.673 ± 0.006 | s1:+0.661, s2:+0.674, s3:+0.683, s4:+0.660, s5:+0.687 | +0.673, 0.422 | +0.037 ± 0.012 | 115, 119, 119, 119, 90 | +0.005 | 7 |
| C2 | 4p-single | 5 | +0.692 ± 0.004 | s1:+0.678, s2:+0.703, s3:+0.686, s4:+0.693, s5:+0.699 | +0.692, 0.426 | +0.037 ± 0.011 | 84, 119, 97, 119, 117 | +0.023 | 6 |
| M1 | 3p-single | 3 | +0.646 ± 0.012 | s1:+0.659, s2:+0.621, s3:+0.658 | +0.646, 0.542 | -0.039 ± 0.012 | 119, 115, 95 | +0.027 | 8 |
| M1 | 3p-double | 3 | +0.736 ± 0.015 | s1:+0.723, s2:+0.766, s3:+0.721 | +0.736, 0.481 | -0.051 ± 0.107 | 119, 115, 95 | +0.027 | 8 |
| M1 | 4p-single | 3 | +0.685 ± 0.016 | s1:+0.712, s2:+0.657, s3:+0.685 | +0.685, 0.434 | +0.053 ± 0.044 | 119, 115, 95 | +0.027 | 8 |
| M1 | 4p-double | 3 | +0.839 ± 0.004 | s1:+0.846, s2:+0.837, s3:+0.833 | +0.839, 0.319 | -0.015 ± 0.027 | 119, 115, 95 | +0.027 | 8 |
| M1 | 5p-single | 3 | +0.692 ± 0.017 | s1:+0.722, s2:+0.663, s3:+0.691 | +0.692, 0.409 | +0.107 ± 0.049 | 119, 115, 95 | +0.027 | 8 |
| M1 | 5p-double | 3 | +0.852 ± 0.009 | s1:+0.870, s2:+0.840, s3:+0.844 | +0.852, 0.285 | +0.074 ± 0.057 | 119, 115, 95 | +0.027 | 8 |
| M1 | 6p-single | 3 | +0.667 ± 0.017 | s1:+0.700, s2:+0.644, s3:+0.656 | +0.667, 0.400 | +0.142 ± 0.055 | 119, 115, 95 | +0.027 | 8 |
| M1 | 6p-double | 3 | +0.847 ± 0.008 | s1:+0.863, s2:+0.840, s3:+0.838 | +0.847, 0.266 | +0.133 ± 0.074 | 119, 115, 95 | +0.027 | 8 |
| S3 | 3p-single | 3 | +0.682 ± 0.012 | s1:+0.673, s2:+0.668, s3:+0.705 | +0.682, 0.541 | +0.050 ± 0.020 | 119, 111, 99 | +0.019 | 7 |
| S5D | 5p-double | 3 | +0.847 ± 0.017 | s1:+0.821, s2:+0.878, s3:+0.843 | +0.847, 0.296 | +0.021 ± 0.035 | 119, 119, 116 | +0.011 | 8 |

## Comparisons (per setting, never across table sizes)

| arm | setting | control | difference | Welch t | df | crit (95%) | verdict | z (2-SE rule) |
|---|---|---|---:|---:|---:|---:|---|---:|
| A1 | 4p-single | A0 | -0.002 | -0.13 | 2.3 | 4.30 | no difference | -0.1 |
| A2 | 4p-single | A0 | +0.009 | +1.87 | 18.0 | 2.13 | no difference | +1.9 |
| B1 | 4p-single | C2 | +0.000 | +0.02 | 5.9 | 2.57 | no difference | +0.0 |
| B1 | 4p-single | A2 | +0.009 | +1.00 | 5.3 | 2.57 | no difference | +1.0 |
| B2 | 4p-single | A2 | +0.009 | +1.32 | 6.7 | 2.45 | no difference | +1.3 |
| B3 | 4p-single | A2 | -0.010 | -1.49 | 7.2 | 2.36 | no difference | -1.5 |
| C2 | 4p-single | A2 | +0.009 | +1.65 | 8.9 | 2.31 | no difference | +1.6 |
| M1 | 3p-single | S3 | -0.036 | -2.15 | 4.0 | 3.18 | no difference | -2.2 |
| M1 | 3p-double | - | - | - | - | - | no specialist: reported without verdict | - |
| M1 | 4p-single | A0 | +0.011 | +0.69 | 2.2 | 4.30 | no difference | +0.7 |
| M1 | 4p-double | - | - | - | - | - | no specialist: reported without verdict | - |
| M1 | 5p-single | - | - | - | - | - | no specialist: reported without verdict | - |
| M1 | 5p-double | S5D | +0.004 | +0.24 | 3.1 | 3.18 | no difference | +0.2 |
| M1 | 6p-single | - | - | - | - | - | no specialist: reported without verdict | - |
| M1 | 6p-double | - | - | - | - | - | no specialist: reported without verdict | - |

## Head-to-head against the committed neat-v2 champion (per-seed scores, same rule)

The default battery saturates against weak opponents; the champion is the discriminating test.

| arm | setting | control | difference | Welch t | df | crit (95%) | verdict | z (2-SE rule) |
|---|---|---|---:|---:|---:|---:|---|---:|
| A1 | 4p-single | A0 | -0.016 | -0.51 | 2.4 | 4.30 | no difference | -0.5 |
| A2 | 4p-single | A0 | +0.048 | +3.68 | 18.0 | 2.13 | BETTER | +3.7 |
| B1 | 4p-single | C2 | -0.005 | -0.16 | 5.3 | 2.57 | no difference | -0.2 |
| B1 | 4p-single | A2 | -0.001 | -0.02 | 5.0 | 2.78 | no difference | -0.0 |
| B2 | 4p-single | A2 | +0.014 | +0.72 | 6.2 | 2.45 | no difference | +0.7 |
| B3 | 4p-single | A2 | +0.004 | +0.26 | 8.2 | 2.31 | no difference | +0.3 |
| C2 | 4p-single | A2 | +0.004 | +0.28 | 9.5 | 2.26 | no difference | +0.3 |
| M1 | 3p-single | S3 | -0.088 | -3.85 | 3.2 | 3.18 | WORSE | -3.8 |
| M1 | 4p-single | A0 | +0.068 | +1.53 | 2.2 | 4.30 | no difference | +1.5 |
| M1 | 5p-double | S5D | +0.053 | +0.80 | 3.3 | 3.18 | no difference | +0.8 |

Rule: BETTER/WORSE only when |t| exceeds the two-sided 95% critical value of Student's t at the Welch-Satterthwaite degrees of freedom (the table lookup rounds the df down, i.e. conservatively); otherwise no difference. B1 is compared against C2 (same evaluation budget) and A2. The z of the 2-SE rule is for reference only.

## Conclusion: what to try first, and how to train most effectively

**First things to try (desktop, in this order):**

1. `git pull`, `cargo build --release -p cli` (use a build directory on a disk
   with space, not the small `/tmp`), then confirm the build with
   a one-minute smoke run (`docs/TRAINING.md`, Quick start).
2. **Reproduce the main result with more seeds** (about 3 minutes per run on the
   8-thread laptop, so 10 seeds are about half an hour; a desktop with more
   cores is faster, results do not depend on the thread count):
   `OUT=runs/p13 CLI=target/release/cli SEEDS="11 12 13 14 15 16 17 18 19 20" docs/baselines/phase-13/run.sh all A2`
   then `docs/baselines/phase-13/analyse.py runs/p13`. Seeds 1-10 are already in
   this README; fresh seeds confirm that `--features hidden` is not a lucky draw.
3. **Train your real model with `--features hidden`**, the neat-v2 recipe, and
   several seeds (`--runs N`, or the loop above). Evolution is noisy (a single
   evaluation has a standard error of 0.023-0.034, the best generation is
   inflated by about 0.02-0.03), so *the seed is a lottery ticket*: train 5-10,
   then pick the winner by a **head-to-head against the champion on fresh
   seeds** (`cli evaluate --opponent neat:docs/baselines/neat-v2/champion.json
   --seed 99`), not by the training score. Do not choose by the score you
   trained on.
4. **Go longer, not fancier.** The fitness variants (risk, tanh) and racing did
   not beat plain `mean`, but 6 of 10 A2 runs only stalled after generation 100 (4 never did) so there is room past
   120 generations. Try `--generations 300` on the best seed (warm start with
   `--from` or `--resume`) before anything else. Raise `--matches-per-genome`
   (80 -> 200) when the curve is flat but noisy; cost per generation grows
   linearly.
5. **Wide model:** if you want one model for every table, run
   `--features all --player-counts 3,4,5,6 --decks single,double` (M1). It is as
   good as the specialists except at 3 players, where a 3-player specialist
   (`--player-count 3`) is clearly better.
6. Hardware note: training uses the CPU only. The RTX 4070 is not used until
   Phase 20 exists; a faster CPU (more cores) is the only speed-up today.

**Most effective training recipe today:** `--features hidden`, default
`--fitness mean`, no racing, pop 150, 120+ generations, 80 matches x 8 rounds,
pass rule `final`, forced exchange, 5-10 seeds, choose by head-to-head on fresh
seeds, then lengthen the best one.

**The casino port:** the TypeScript bot in the casino repo only knows feature
set v1 (`base`, 20 inputs). A `hidden` or `all` genome needs the new inputs
ported; until then use `base` genomes there.

## Review of the roadmap order

- **Phase 19 before Phase 13 was the right order.** It cut CPU time 3-4x with
  byte-identical results, which is what made 10-seed experiments (about 50
  trainings) affordable, and it proved the whole pipeline reproduces the
  neat-v2 champion exactly. Doing 13 first would have meant a third of the
  evidence at triple the cost.
- **Next: Phase 17 (run analysis), before 16 and 18.** It is cheap (the
  `analyse.py` script already does most of it), and both the advisor and
  self-play need its stall, noise and winner's-curse numbers. The Phase 13
  results say the same thing again: differences are small against noise, so
  tools that quantify noise pay off first.
- **Then Phase 18 (advisor) in a reduced form:** the result "hidden inputs, mean
  fitness, many seeds, choose by head-to-head" is already a recipe; an
  advisor that automates seeds, selection and the longer run is more valuable
  than new options.
- **Phase 16 (self-play) stays low.** Nothing here suggests the pool limits
  progress; runs are noise- and information-limited.
- **Phase 20 (GPU) only after a G0 spike and only if longer runs and more
  matches per genome help** (step 4 above is the cheap test). If 300
  generations at 200 matches beats 120 at 80 clearly, a 10x simulator is worth
  the 5 months in the evenings; if not, it is not.
- **Cheap, high value before any of them:** extend the casino port to
  feature set v2 (see above), and consider making `hidden` the default
  `--features` after step 2 confirms the result with 10 fresh seeds.
