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
