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
| `--target-species` | 8 | how many species the speciation steers toward |
| `--deck-variant`, `--duplicate-rule` | single, first-dealt-wins | table rules |
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
