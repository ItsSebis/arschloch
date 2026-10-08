# Baselines under the current rules (pass rule `final`, exchange rule `forced`)

Phase 15 made the forced exchange (the lower role of an exchange pair gives its
highest cards, no choice) the rule of the game and the default, on top of Phase
14's `final` pass rule. This directory re-measures with both defaults, with the
same seeds, match counts and round counts as the earlier baselines:

- `hand-written/`: `docs/baselines/pre-neat/run.sh` with `PASS_RULE=final
  EXCHANGE_RULE=forced` (36 files).
- `neat-v2-champion/`: `docs/baselines/neat-v2/run.sh` with `EXCHANGE_RULE=forced`
  (28 files).

Where the earlier baselines stand (all reproducible with the scripts' env):

| directory | pass rule | exchange rule |
|---|---|---|
| `../pre-neat`, `../neat-v1` | free | free |
| `../pass-final`, `../neat-v2` | final | free |
| this directory | final | forced |

## What the forced exchange changes

Only strategies that would have given something other than their highest cards:
`random-legal` (random cards) and `hold-back-pairs` (keeps pairs together). Every
other strategy, including the evolved players and `adaptive`, already gave its
highest cards, so their games are **identical** under both rules. Of the 36 files
here, 10 differ from `../pass-final/hand-written` (ignoring the header line):
`random-legal` and `hold-back-pairs` at 3-6 players and the 5- and 6-player mixed
fields.

| 4 players, 1 strategy vs 3 `lowest-legal`, 2000 x 10 rounds | President | Arschloch |
|---|---:|---:|
| `random-legal`, free exchange | 427 | 11135 |
| `random-legal`, forced | 259 | 12367 |
| `hold-back-pairs`, free exchange | 33 | 16785 |
| `hold-back-pairs`, forced | 36 | 16842 |

A random player that has to give its best cards does worse; a hold-back player
barely changes.

## The retrained champion

The training pool (`lowest-legal`, `endgame-denial`, `adaptive:reading,tempo,bully`)
contains no strategy the rule changes, so training under the new default is the
same run: a retrain with exactly the neat-v2 recipe and default rules produced a
**byte-identical** `best.json` (generation 51, +0.608 ±0.013 held out).
`../neat-v2/champion.json` is therefore already the champion trained under the
current rules. Its evaluation battery does include `random-legal` and
`hold-back-pairs`, so those rows move (`cli evaluate`, 600 matches x 8 rounds,
mixed row):

| players | exchange free | exchange forced |
|---|---:|---:|
| 3 | +0.578 | +0.587 |
| 4 | +0.665 | +0.678 |
| 5 | +0.643 | +0.657 |
| 6 | +0.580 | +0.607 |
