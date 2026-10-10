# Training a model

How to evolve an Arschloch-playing neural network with `cli train`, watch it
learn, and compare the result with older models and with the hand-written
strategies. Design background is in
`docs/superpowers/specs/2026-10-08-neat-engine-design.md`.

## The idea in four sentences

A *genome* describes a small neural network that scores every legal move
(including passing) from 20 facts about the situation; the player plays the
highest score. Training keeps a *population* of genomes (150 by default).
Every generation each genome plays the same matches against opponents, the
best ones breed (with mutation and crossover) and the next generation takes
their place. The score you see everywhere is the mean finishing role, from
**+1** (always President) to **-1** (always last); **0** is an even match.

## Quick start

```bash
cargo build --release -p cli
target/release/cli train --out runs/first --serve
```

`--out` is a new directory for this run (it is never overwritten). `--serve`
prints `dashboard: http://127.0.0.1:8080/`; open it in a browser to watch
the charts and the evolving network. With the defaults (population 150,
100 generations, 100 matches per genome) a run takes roughly 10 minutes on
8 cores.

When it ends you have `runs/first/best.json`, the best champion found. It
plays like any other strategy:

```bash
target/release/cli --player-count 4 --matches 1000 \
  --strategy neat:runs/first/best.json --strategy lowest-legal \
  --strategy lowest-legal --strategy lowest-legal
```

## Reading the terminal

One row per generation:

```
  gen    best    mean  champion (fresh)  spc  nodes/conn   o1     o2     o3    rounds/s      ETA
   42  +0.412  +0.188   +0.397 ±0.021     9    3/25     +0.61  +0.30  +0.52       18k   0:41:10 *
```

| column | meaning |
|---|---|
| `best`, `mean` | the best and the average genome of this generation on its *training* matches (the best is inflated: it is the luckiest of many) |
| `champion (fresh)` | the generation's chosen champion on a fixed set of other matches, with its standard error; this is the curve to watch |
| `spc` | how many species the population is split into (about 8 is normal) |
| `nodes/conn` | hidden nodes / enabled connections of the champion (complexity) |
| `o1`, `o2`, ... | the champion against tables made only of opponent 1, 2, ...; the legend above the table names them |
| `hof` | with `--hall-of-fame`: the champion against its recent ancestors (near 0 means about even) |
| `rounds/s`, `ETA` | throughput and time remaining |
| `*` | a new best champion (confirmed on separate held-out matches) |

At the end it prints the best champion's generation and its **held-out**
score, which is the number to quote.

## Watching in the browser

- `--serve [PORT]` on `cli train` (default port 8080) shows the run live.
  The server lives in the training process, so it keeps running after the
  run ends until you press Ctrl-C.
- `cli watch runs/first [--port N]` shows any run directory, running in
  another process or finished. Use it to look at old runs.
- It listens on `127.0.0.1` only. To view a remote run:
  `ssh -L 8080:127.0.0.1:8080 host`.

The page shows progress and headline numbers, fitness curves, the champion
against each opponent, species, the population's fitness spread, complexity,
the champion's network (slider over generations) and a decision inspector
that replays real decisions of each new best champion through the network.

## What a run leaves behind

```
runs/first/
  config.json        the settings of the run
  events.jsonl       one JSON line per generation (everything the terminal and browser show)
  checkpoint.json    the resume point, replaced after every generation
  best.json          the best champion so far (a playable genome file)
  gen-0000.json ...  every generation's champion
  decisions/         12 recorded real decisions of each new best champion
  opponents/         frozen copies of any neat:PATH opponents the run trains against
```

A run is reproducible: the same settings and `--seed` give the same result,
independent of `--threads`.

Each generation line in `events.jsonl` also has an optional `timings` object:
the wall-clock seconds of every stage, to see where a slow run spends its
time (it differs from run to run, like the other wall-clock fields
`elapsed_secs`, `generation_secs` and `rounds_per_sec`). The stages are:
`training_evaluation` (all genomes play their matches), `champion_selection`,
`speciation_and_reproduction`, `reevaluation_mixed`,
`reevaluation_per_opponent`, `hall_of_fame` (the champion against the hall,
plus updating it), `confirmation` and `decision_sample` (both 0 unless the
generation has a new best) and `checkpoint_and_files` (champion, best and
decision files; the event line and the checkpoint are written afterwards and
are in no stage). `reevaluation_mixed`, `reevaluation_per_opponent` and
`hall_of_fame` run at the same time, so their times overlap and only the
longest of them adds to the generation's wall time. Events from older runs have no `timings`; the terminal table
does not show them. `docs/baselines/perf/scaling.sh` summarises them per
thread count.

## Choosing settings

| option | default | what it does / when to change it |
|---|---|---|
| `--player-count` | 4 | table size 3-6. Train the size you care about; a genome trained at 4 still beats every opponent at 3, 5 and 6 (see `docs/baselines/neat-v1`), but is not tuned for them |
| `--population` | 150 | more genomes explore more; 60-100 is fine for quick tests |
| `--generations` | 100 | most progress happens in the first 20-30; later generations mostly add complexity |
| `--matches-per-genome` | 100 | more matches = less noisy fitness but slower. Cost per generation is about population x matches x rounds |
| `--rounds` | 8 | rounds per match (roles carry over between rounds) |
| `--reeval-matches` | 200 | matches used to score each champion; the held-out confirmation uses twice as many |
| `--opponent SPEC` | `lowest-legal`, `endgame-denial`, `adaptive:reading,tempo,bully` | the training pool, in `--strategy` syntax; repeat for several. A harder pool gives a stronger player but slower early progress |
| `--champion-candidates` | 5 | the champion is the best of the top N genomes (by training fitness) re-scored on other matches; 1 trusts the noisy training fitness |
| `--weight-power` | 0.2 | size of weight mutations; smaller is gentler fine-tuning |
| `--hall-of-fame N` | 0 (off) | frozen past champions join the training opponents (`--hall-interval K`: every K generations). No measured benefit against the default pool; worth trying when the pool includes evolved opponents |
| `--fitness-skill-weight W` | 0 (off) | experimental, 0 to 1: trains on `(1 - W) x mean role score + W x a luck-adjusted round-1 score` (see below). Fixed for the run (`--resume` cannot change it) |
| `--features base\|trick\|hidden\|all` | base | which inputs the networks see (see "Feature sets" below). Fixed for the run (`--resume` cannot change it); a warm start (`--from`) needs the source's value |
| `--fitness mean\|risk:L\|tanh:K` | mean | what evolution maximises (see "Fitness variants" below). Fixed for the run; cannot be combined with `--fitness-skill-weight` |
| `--racing-top F --racing-matches K` | off | noise-aware selection: the top fraction F of each generation is evaluated on K extra matches before selection (see "Racing" below). Both together; fixed for the run |
| `--player-counts 3,4,5,6`, `--decks single,double` | off | train one model on several table sizes and decks (see "One model for several tables" below). Replace `--player-count` / `--deck-variant`, which they exclude; fixed for the run |
| `--target-species` | 8 | how many species the speciation steers toward |
| `--deck-variant`, `--duplicate-rule` | single, first-dealt-wins | table rules |
| `--exchange-rule` | forced | `forced`: the lower role of an exchange pair must give its highest cards (the rules of the game); `free`: it may choose (the older behaviour). Evolved players give their highest cards either way, so it only changes what the opponents do. Like `--pass-rule` it cannot change on `--resume`; runs written before it existed resume as `free` |
| `--pass-rule` | final | `final`: a pass ends your part in the trick (the rules of the game); `free`: the old behaviour. Genomes trained under one rule play the other legally but are tuned to their own; runs written before the rule existed resume as `free` (`--resume` cannot change it) |
| `--seed` | 0 | change it to get an independent run |
| `--from RUN_DIR` | off | start from the final population of an earlier run instead of a random one (see below); the population size is the earlier run's, so `--population` cannot be given |
| `--runs N` | 1 | train N independent runs one after another as a set (see below) |
| `--threads` | all cores | rayon thread count (results do not depend on it) |
| `--quiet` | off | print only the final summary |

Cost: roughly 30,000-33,000 rounds per second on a 4-core / 8-thread laptop
(i5-11300H); a Ryzen 7 7800X3D should be about two to three times faster
(an estimate, not a measurement: `--generations 5 --quiet` on your machine
prints the real `rounds/s`). A default run (population 150, 100 generations,
100 matches x 8 rounds) takes roughly 7 minutes on the laptop. Before
Phase 10f the same laptop did 18,000-23,000 rounds/s; see
`docs/baselines/perf/README.md` for what changed and the numbers.

Recipes:

```bash
# smoke test (about a minute)
target/release/cli train --out runs/test --population 60 --generations 30 \
  --matches-per-genome 40 --reeval-matches 100 --rounds 6

# the setup of the committed baseline champion (about 11 minutes)
target/release/cli train --out runs/v1 --seed 1 --population 150 \
  --generations 120 --matches-per-genome 80 --reeval-matches 200 \
  --rounds 8 --champion-candidates 5 --weight-power 0.2 --quiet

# train for 6 players against a harder pool
target/release/cli train --out runs/six --player-count 6 \
  --opponent endgame-denial --opponent adaptive:reading,tempo,bully \
  --opponent adaptive:counting,reading
```

## The luck-adjusted fitness term (experimental)

`--fitness-skill-weight W` (default 0) blends a second term into the training
fitness: `fitness = (1 - W) x mean_role_score + W x luck_adjusted_score`. The
mean role score is the usual one, over all rounds of all matches. The
luck-adjusted score of a genome uses **round 1 only**: in each match its
round-1 role score minus `beta . (hand features - mean hand features)`,
averaged over the matches. The six hand features are the ones of the
statistics estimator (`docs/STATISTICS.md`). `beta` is one least-squares fit
(with an intercept) over the round-1 data of **all genomes of the
generation**, which all play the same deals, so it describes how hand strength
affects the result at this table and a genome cannot explain away its own
result. With `W = 0` nothing extra is computed and a run is byte-for-byte what
it was before the option existed (no new event fields, same files).

With `W > 0` the generation's `fitness` carries a `skill_term` (mean and
standard deviation of the term over the genomes) and the champion carries its
own `skill_term` (both additive, absent otherwise). The `fitness` numbers and
`train_fitness` are then the blend. Everything that judges a champion keeps
using the plain mean role score: the choice among the top candidates, the
fixed re-evaluation, the held-out confirmation and `best.json`. The term only
steers selection pressure inside a generation.

**Honest caveat: what it can and cannot do.** Every genome of a generation
plays the same matches, so every genome has the same deals and the same
feature rows, and the mean of `beta . (features - mean features)` over those
matches is exactly zero. The adjusted score of a genome therefore equals its
plain round-1 mean score, and `W > 0` can only change which genomes are
selected by weighting round 1 more heavily than the later rounds; it does not
(and cannot) cancel deal luck between genomes, which common random numbers
already do. What the adjustment reduces is the spread of the per-match values.
In a first measurement (population 60, 30 generations, 40 matches, 6 rounds,
seeds 11 and 12, `W` = 0 vs 0.5) the held-out champion scores were 0.525 vs
0.521 and 0.495 vs 0.547 (standard error about 0.03 each), so no effect is
visible beyond noise. Whether any weight helps is an open Phase 13 experiment;
treat the option as a way to run it, not as a recommended setting.

## Feature sets (`--features`)

A network's inputs are the features of one candidate move. The full list
(`FEATURE_NAMES_V2` in `sim/src/strategies/neat_player/features.rs`) is the 20
features of set v1 (`base`, unchanged: same names, order and formulas) followed
by three groups; `--features` picks a nested prefix, so each choice is a clean
ablation with an honest genome size:

| `--features` | inputs | adds |
|---|---|---|
| `base` (default) | 20 | the v1 features |
| `trick` | 25 | `opponents_in_trick`, `fraction_in_trick`, `all_in_trick_close`, `min_hand_in_trick`, `last_to_act_in_trick` |
| `hidden` | 28 | `p_win_trick`, `expected_beaters`, `my_unbeatable_fraction` |
| `all` | 31 | `table_players`, `table_double_deck`, `table_pass_final` |

*Trick.* An opponent is "in the trick" while it is active and, under pass rule
`final`, has not passed in it (under `free` every active seat counts).
`opponents_in_trick` is their number / 5, `fraction_in_trick` that number over
all active opponents, `all_in_trick_close` is 1 when there is at least one and
all of them hold at most 2 cards, `min_hand_in_trick` the smallest of their hands
/ 13, and `last_to_act_in_trick` is 1 when no opponent in the trick acts after
you before the trick comes back to the seat holding the table combo (all of them
act after you when you lead).

*Hidden.* For a candidate play of size s topped by card t, with U unseen cards,
the beating groups are the unseen ranks with u >= s cards that can beat t
(ranks above t; t's own rank when some unseen card of it is above t in suit or
duplicate order, b of its u cards). An opponent holding h of the U unseen cards
beats the play through a group when it holds k >= s of its cards including a
beater, probability `sum_{k>=s} (C(u,k) - C(u-b,k)) C(U-u,h-k) / C(U,h)` (exact for
one opponent and one group). For s = 1 the beater cards are pooled
(`1 - prod_{j<h} (U - m - j) / (U - j)`, m = number of beater cards), also
exact. For s >= 2 the groups are combined as independent
(`1 - prod_g (1 - P_g)`, slightly optimistic for the opponent, e.g. 0.81 against
an exact 0.80 won for a pair against two beating pairs), and a pass ceiling that
proves the opponent cannot beat the play gives 0. `p_win_trick` is the
product of the complements over the opponents in the trick (the hands are also
treated as independent, though they partition the unseen cards; 1 when nobody is
in the trick, 0 for a pass), `expected_beaters` the expected number of cards each
such opponent holds in beating groups (for s >= 2 only groups it holds in force
count), summed and divided by 52, and `my_unbeatable_fraction` the share of the
cards left in your hand (after the candidate) in a rank group that no unseen group
of at least its size can beat. All are exact deterministic functions of the cards
counted so far, no randomness.

*Table.* `table_players` = (seats - 3) / 3, `table_double_deck` and
`table_pass_final` are 0/1. They only vary when one model trains on several
tables.

A genome file records the feature names it was trained with, and it loads when
those names are a prefix of the full list ending at a group boundary (20, 25, 28
or 31) and the genome has exactly that many inputs; anything else (other names,
other order, a prefix that stops inside a group, 20 names with 22 inputs) is
refused with both lists named. Every older genome therefore loads and plays
unchanged, and `--features base` runs reproduce earlier runs byte for byte
(`config.json` only mentions `features` when it is not `base`). `NeatStrategy`
builds only the inputs it was trained with and asks the match loop only for the
context parts they read, so a `base` model costs what it always did. The casino
TypeScript port implements feature set v1 only: models with `trick`, `hidden` or
`all` features load in the Rust tools but not in the casino until its port is
extended.

## Fitness variants, racing and several tables (Phase 13 options)

These three options are experiments: whether any of them produces a stronger or
more robust player than the default is the open question of Phase 13, and this
page does not claim they do. With none of them given a run is byte for byte what
it was before they existed (`config.json` and the events only mention an option
that is on, and the seed streams of the existing matches are untouched), so every
older run directory still loads and resumes.

### Fitness variants (`--fitness`)

Every variant is a function of the per-round role scores `s` of the candidate
over the rounds of an evaluation (+1 President .. -1 last, linear in the place,
the same scale at every table size):

| `--fitness` | formula |
|---|---|
| `mean` (default) | the mean of `s` (the mean of the match means, as before) |
| `risk:L` (`L >= 0`) | `mean(s) - L x sd(s)`, `sd` the population standard deviation of `s` over all the rounds of the evaluation, `sqrt(sum(s^2)/n - mean^2)`. Rewards a player that finishes in the same place every round over one that alternates President and last |
| `tanh:K` (`K > 0`) | the mean of `g(s) = tanh(K s) / tanh(K)`. `g(1) = 1`, `g(-1) = -1`, and `g` flattens towards the top: with `K = 4` a Vize (`s = 1/3` at four seats) is worth about 0.87 of a President, so the network is not pushed to gamble for the top place |

The training fitness, the choice among `--champion-candidates`, the champion's
re-evaluation, the held-out confirmation and the choice of `best.json` all use the
configured fitness. Events always carry `mean` and `std_error` as before; with a
non-mean fitness every score also carries `spread` (the `sd` above) and `fitness`
(the configured scalar), and the terminal row adds `sd`. A mean run's events carry
neither (adding them would change its bytes). `cli evaluate` always reports
`spread` in its JSON cells. `--fitness` cannot be combined with
`--fitness-skill-weight`: the luck-adjusted term is defined for the mean role score.

### Racing (`--racing-top F --racing-matches K`)

The training fitness of the genomes that look best is inflated by luck (the
winner's curse), and selection and the champion choice then act on that inflated
number. After the normal evaluation of a generation, the top `ceil(F x population)`
genomes by fitness play `K` extra matches. The extra matches use a fresh seed
stream (new every generation, and the same deals, seats, tables and opponents for
every raced genome, so they are still common random numbers), and each raced
genome's fitness becomes the fitness of all its `m + K` matches together
(`m` = `--matches-per-genome`) before speciation, selection and the champion choice.
Genomes that were not raced keep their fitness.

Pooling is done on the moments, not on the scalars: an evaluation keeps `n`, the
sum of `s`, the sum of `s^2` and the sum of `g(s)`; the two evaluations' sums are
added and the configured fitness is taken of the totals. For `mean` this is
exactly the match-weighted mean `(m x old + K x extra) / (m + K)`; for `tanh` the
same weighted mean of the two `tanh` fitnesses; for `risk` it is *not* the
weighted mean of the two risk values (the standard deviation of the pooled rounds
is not the average of two standard deviations), which is why the sums are kept.
Each generation's event has a `racing` object: `raced`, `extra_matches`,
`mean_before`, `mean_after` and `mean_shift` of the raced genomes' fitness, so the
correction is visible (a negative shift means the leaders were lucky). Cost: `raced
x K` extra matches per generation, counted in `rounds_evaluated`. Racing cannot be
combined with `--fitness-skill-weight`.

### One model for several tables (`--player-counts`, `--decks`)

`--player-counts 3,4,5,6` and/or `--decks single,double` (comma lists, validated)
train one model on every combination of the lists, table size first: with
`3,5` and `single,double` the settings are `3p-single, 3p-double, 5p-single,
5p-double`. Match `i` of an evaluation is played at setting `i mod (number of
settings)`, in seat `i mod (players of that table)`, the same for every genome, so
common random numbers hold. The same cycle is used by the re-evaluations (mixed,
per opponent, hall of fame), the candidate selection and the held-out
confirmation, each with its own seeds. The fitness is the configured scalar
averaged over the matches (role scores are already normalised per table size).
The recorded decisions of a new best come from the first setting. The opponents
are the same strategy objects at every table size (the hand-written ones and
`neat:` opponents with the base features are table-size independent).

The lists replace `--player-count` and `--deck-variant`; giving both is an error. A
list with one entry is the single flag (and writes the same `config.json`). Both
`--matches-per-genome` and `--reeval-matches` must be at least the number of
settings, so every setting is played. A run with several settings should use
`--features all`: only the `table` feature group lets the network see the table
size and deck, and with another feature set it plays every setting the same way
(`cli train` prints a warning, not an error). Several settings cannot be combined
with `--fitness-skill-weight`. With more than one setting every score in the events
also has `by_setting`: one entry per setting with `label` (such as `4p-single`),
`matches`, `mean`, `spread`, `fitness` and `placements` at that table size
(the top-level `placements` add the counts of equal place indices, so read the
per-setting ones). `cli evaluate --player-counts 3,4,5,6 --decks single,double`
evaluates every setting from the same seeds and prints one block per setting; its
JSON keeps the first setting's `results` and adds `settings` with all of them.

## Building on an earlier run

`--from RUN_DIR` starts a *new* run whose first population is the final
population of an earlier run, instead of random genomes:

```bash
# 20 generations against the default pool, then 10 more against a harder one
target/release/cli train --out runs/a --seed 1 --generations 20
target/release/cli train --out runs/b --from runs/a --seed 2 --generations 10 \
  --opponent endgame-denial --opponent adaptive:counting,reading
```

The new run has its own settings, seed and generation counter (it counts
from 0; `events.jsonl` records a `warm_started_from` entry),
and the earlier run is only read. Because the genomes are kept as they are,
the population size is the earlier run's. A source trained on another feature
set cannot be used: the command says so and leaves nothing behind. To continue
the *same* run with its own settings, use `--resume` instead.

## Sets of runs and the time left

`--runs N` trains N independent runs one after another. `--out` then holds
`run-01`, `run-02`, ... (run k uses seed `SEED + k - 1`) and a `set.json`
that records how long the finished runs took:

```bash
target/release/cli train --out runs/set --runs 5 --serve
```

Every row ends with `set H:MM:SS`, the estimate for the whole set: the
current run's own ETA plus the remaining runs at the pace of the finished ones
(before the first run ends, at this run's projected time). The dashboard
shows the same on its Generation card (`run ETA ... · set ... · run 2/5`) and
follows the set from run to run on its own. A measured example: three runs of
12 generations, predicted 58 s at the first generation, took 55 s.
`--runs 1` is an ordinary run. `--from` combined with `--runs` warm-starts
every run from the same source.

A killed set continues with `--resume` on the set directory: finished runs
are left alone, the run in progress is resumed from its checkpoint, and runs
that had not started are started with the first run's settings. (`--generations`
sets the total for the run in progress and for the runs not yet started.)

## Stopping and resuming

The checkpoint is written after every generation and replaced atomically, so
you can press Ctrl-C or kill the process at any time. Continue with:

```bash
target/release/cli train --out runs/first --resume --generations 200
```

`--generations` is the new total to run up to. Only `--generations`,
`--threads`, `--quiet` and `--serve` may accompany `--resume`: everything else
is read from the checkpoint. A resumed run ends exactly where an uninterrupted
one would have.

## Comparing models and strategies

`cli evaluate` scores genomes against opponents on the same games for every
genome, using matches training never used:

```bash
# an older champion against the current one, against the default battery
target/release/cli evaluate \
  --genome runs/old/best.json --genome runs/first/best.json

# any generation's champion, against chosen opponents
target/release/cli evaluate --genome runs/first/gen-0030.json \
  --opponent lowest-legal --opponent adaptive:reading,tempo,bully

# a model against another model
target/release/cli evaluate --genome runs/first/best.json \
  --opponent neat:docs/baselines/neat-v1/champion.json
```

It prints one row per opponent plus a mixed row, one column per genome
(`+0.620 ±0.011`), and `--json PATH` also writes the numbers. Options:
`--player-count`, `--matches` (default 400 per cell), `--rounds`, `--seed`.
The default battery is the three training opponents, two stronger adaptive
variants, and three weak strategies.

How to read scores:
- `0` is even; `+0.5` means usually near the top; against the strongest
  hand-written strategy the committed baseline champion scores about `+0.6`
  at 4 players (President in 56% of rounds where chance gives 25%).
- Look at the standard error: differences smaller than about two standard
  errors, or smaller than the seed-to-seed spread (about 0.01-0.03), are
  noise. Compare models with at least a few training seeds before believing
  a small difference.
- `docs/baselines/neat-v1/` holds a committed reference champion and its
  recorded comparison with the hand-written strategies at 3-6 players;
  `docs/baselines/pre-neat/` holds the hand-written strategies' own
  baseline. Run `docs/baselines/neat-v1/run.sh` to repeat the comparison for
  any genome (set `GENOME=...`).

## Training against your own models

`--opponent neat:PATH` trains against a previous champion. The run copies the
file into `runs/NAME/opponents/` and resumes from that copy, so changing or
deleting the original later cannot silently change the run.

## Troubleshooting

- **The champion curve is flat after about 20-30 generations.** That is
  normal for this setup. More generations alone rarely help; try a harder
  `--opponent` pool, another `--seed`, or more `--matches-per-genome`.
- **Scores jump around.** Fitness is noisy (card luck). Raise
  `--matches-per-genome` or `--reeval-matches`, and compare several seeds.
- **`dashboard` says the port is in use.** Pick another with `--serve 8081`.
- **"already holds a run".** A new run never overwrites a directory; use a new
  `--out` or `--resume`.
- **A genome file is refused ("does not fit this build").** Genomes record the
  feature set they were trained on; a build that changed the features cannot
  load them. Retrain.
- **Comparing results between table sizes or opponents.** Scores depend on the
  table; only compare numbers measured the same way. In mixed tables of
  different hand-written strategies, results also depend on who sits where
  (see `docs/baselines/pre-neat/README.md`).
- **`card-counter` as an opponent.** It plays identically to `lowest-legal`,
  so it adds nothing to a pool.

## Playing against a model yourself

Phase 11 of the roadmap adds a browser game where you play against one or
several trained models. Until then, a genome plays in simulations with
`--strategy neat:PATH`.
