# NEAT engine design (Phase 10)

Date: 2026-10-08. Status: proposed, not yet implemented.

## 1. Goal and non-goals

**Goal.** Evolve neural-network players with NEAT (NeuroEvolution of
Augmenting Topologies), generation by generation, and measure them
against the existing hand-written strategies under the same statistics
pipeline.

**Non-goals (v1).**
- No changes to `engine` rules, deck variants or duplicate rules.
- No recurrent connections or memory; the network is a pure feedforward
  function of the current `TurnContext` (the game is closed-deck with
  exact unseen-card knowledge, so most of what memory would provide is
  already in the context).
- No change to the existing strategies. They stay as live opponents.

## 2. Preserving the current version

- Git tag `baseline-pre-neat` marks the last commit before NEAT work.
- `docs/baselines/pre-neat/` holds frozen result summaries for every
  strategy (including `Adaptive` with the strongest modifier sets) at
  3-6 players, with the script and seeds that produced them.
- Rule: NEAT work is additive. Baseline strategies are never edited;
  if one must change, create a new tag and a new baseline directory.
- Because they remain in the crate, the baselines are also live
  opponents for training and evaluation, not just history.

## 3. Architecture

```
engine  <-  sim  <-  web  <-  cli (train / watch / simulate)
                ^
neat (generic NEAT, no game knowledge)
```

- New crate **`neat`**: genome, innovation tracker, speciation,
  crossover, mutation, reproduction, feedforward network compilation,
  JSON (de)serialization. Depends only on `rand` and `serde`; knows
  nothing about cards. Unit-testable in isolation.
- **`sim`** gains `NeatStrategy` and a feature extractor
  (`sim/src/strategies/neat.rs`). It depends on `neat` and implements
  the existing `Strategy` trait, holding an immutable compiled network
  (`Send + Sync`, no interior mutability), so it works unchanged with
  `run_match` / `run_batch` / `aggregate`.
- **`web`** (currently a stub) gains the dashboard server and pages
  (section 7a) in phase 10d; it reads run directories and depends on
  `sim`/`neat` types only for decoding genomes and events.
- **`cli`** gains a `train` subcommand, `watch`, `--serve`, and a
  `neat:<genome.json>` strategy spec. Dependency direction stays
  one-way (`cli -> web -> sim -> engine`, `sim -> neat`).

## 4. Playing: scoring variable move sets

The number of legal moves changes every turn, so the network does not
output "a move". It scores one candidate at a time:

```
for each move in legal_moves:
    score = net(features(move, TurnContext))   // one scalar output
play argmax(score)    // ties: lowest combo, then first listed
```

`Move::Pass` is a candidate when legal, flagged by an `is_pass` input.
The strategy never leaves `legal_moves`, so illegal plays are
impossible by construction.

**Candidate features** (all scaled to roughly `[0, 1]`):
- move: `is_pass`, combo size / 4 (8 for double deck), top-card rank
  relative to hand (rank position among distinct ranks held),
  whether the play uses the hand's highest rank, cards left after play
  / starting hand size, whether the play empties the hand;
- table: `is_leading` (no current combo), size of current combo;
- self: hand size, fraction of the hand in same-rank groups of size
  >= 2, number of singleton rank groups;
- opponents: min and mean active opponent hand size, count of active
  opponents, whether any opponent is within 2 cards of finishing;
- information: fraction of `unseen_cards` that outrank the play's top
  card, fraction that could beat it at the same combo size, number of
  active opponents whose pass ceiling is below the play (from
  `TurnContext::opponents[].pass_ceilings`), own pass ceiling signal;
- role: *deferred to v2.* `TurnContext` does not carry the acting
  seat's role, and adding it touches every strategy's context
  construction; it only matters once the exchange step is evolved too.

The exact set is fixed in the implementation plan; the invariants are
that features are deterministic functions of `(move, TurnContext,
duplicate_rule)` and that the input count is a compile-time constant
stored in each genome file (so old genomes are rejected cleanly when the
feature set changes).

**Exchange.** `choose_exchange_cards` initially reuses
`take_highest_naive`. A v2 option scores each held card with the same
network (card-only features) and gives up the top `count`.

## 5. NEAT algorithm specifics

- Genome: node genes (input, bias, output, hidden) and connection genes
  (in, out, weight, enabled, innovation number). Inputs and bias are
  fixed; one output node; hidden nodes appear via mutation. Minimal
  start topology (inputs fully connected to the output) with small
  random weights.
- Mutations: perturb/replace weights, add connection (reject cycles to
  keep nets feedforward), add node (split a connection), toggle
  enable. Rates are config values with NEAT-paper-style defaults.
- Innovation numbers come from a per-run tracker so identical
  structural mutations within a generation share an id.
- Crossover aligns genes by innovation number; excess/disjoint genes
  come from the fitter parent; a gene disabled in either parent is
  disabled in the child with 75% probability.
- Speciation by compatibility distance
  `c1*E/N + c2*D/N + c3*Wbar`, with a threshold adapted to hold the
  species count near a target. Explicit fitness sharing within species.
- Reproduction: per-species offspring quotas from shared fitness,
  top-fraction survival, champion of each large species copied
  unchanged, species culled after `stagnation_limit` generations
  without improvement (the global best is protected).
- Activation: `tanh` for hidden and output nodes (output is only
  compared, never interpreted absolutely).
- Everything is driven by one `StdRng` seeded from the run seed.

## 6. Fitness and evaluation

**Episode.** One genome plays a match (`rounds` rounds, role carry-over)
against opponents drawn from an opponent pool, using `run_match`.

**Score per round** from finishing position (derived from
`MatchResult.role_history`): linear from +1 (President) to -1 (the
last-place role), averaged over rounds and over seat rotations (as
`run_batch` does) so the dealing seat bias cancels.

**Noise control.** Card luck dominates a single match, so:
- *Common random numbers*: within a generation every genome is
  evaluated on the same list of match seeds, so differences between
  genomes reflect the genome, not the shuffles.
- Matches per genome (default ~200 matches x 10 rounds) grow over
  generations; the champion is re-evaluated on a fresh seed set before
  being recorded as best, to avoid winner's-curse selection.
- Ranking uses a mean with its standard error; ties inside the error
  band prefer the smaller genome (a mild complexity pressure).

**Opponent pool.**
1. Fixed baselines: `LowestLegal`, `CardCounter`, `EndgameDenial`,
   `Adaptive(reading,tempo,bully)` (the strongest at baseline time).
2. Hall of fame: champion genomes from earlier generations (frozen
   `NeatStrategy` instances), giving co-evolutionary pressure and
   protection against overfitting to fixed opponents.
3. Curriculum: begin with weaker baselines, add stronger ones and the
   hall of fame as the champion's score against the current pool
   passes a threshold.

Seats not held by candidates are sampled from the pool per match
(seeded, so reproducible). Training is per table size (3, 4, 5, 6);
a genome's features include active-opponent counts, so one genome can
be tried across sizes, which is evaluated as an experiment rather than
assumed.

## 7. Training CLI

`cli train` (a clap subcommand; the existing simulate invocation stays
as is):

```
cli train --player-count 4 --population 150 --generations 200 \
    --matches-per-genome 200 --rounds 10 --seed 1 \
    --opponent lowest-legal --opponent card-counter \
    --opponent adaptive:reading,tempo,bully \
    --out runs/neat-4p-seed1/
```

Per generation it writes `gen-NNNN.json` (champion genome) and a line
to `events.jsonl` (all statistics, see 7a), keeps `best.json`, prints a
progress line to the terminal, and, with `--serve`, updates the browser
dashboard; `--resume <dir>` continues from the latest checkpoint,
reproducing the same run as an uninterrupted one. A trained genome is
used anywhere a strategy spec is accepted:
`--strategy neat:runs/neat-4p-seed1/best.json`, so all existing
output and statistics apply unchanged.

## 7a. Live monitoring

Training runs for a long time, so progress must be visible while it
runs, never only afterwards. The trainer does not print or serve
anything itself: it calls a `TrainObserver` trait (`on_generation`,
`on_eval_progress`, `on_finish`) with plain-data events. Observers are
independent and can be combined:

1. **Terminal observer** (always on unless `--quiet`). One line per
   generation to stdout, plain text so it survives piping and logging:

   ```
   gen   best    mean    champ(re-eval)   spc  nodes/conn  vs LL   vs CC   vs AD   r/s   ETA
    42  +0.412  +0.188  +0.397 ±0.021     11   7 / 23     +0.61   +0.30   -0.05   84k   0:41:10
   ```

   Columns: best and mean fitness, the champion's fresh-seed
   re-evaluation with its standard error, species count, champion
   genome size, the champion's score against each fixed baseline
   (`LowestLegal`, `CardCounter`, `Adaptive`), evaluation throughput
   (rounds/second) and an ETA. On a TTY, a single refreshed stderr line
   shows within-generation progress (`genome 87/150`). A `new best`
   marker flags generations that improve the global best.
2. **Event log** (always written). `events.jsonl` in the run directory,
   one JSON object per generation (schema below). It is the single
   source of truth for every other view and makes a run replayable and
   comparable after the fact. Checkpoints (`gen-NNNN.json`, `best.json`)
   hold genomes; the event log holds statistics.
3. **Browser dashboard** (`--serve [PORT]`, default 8080, or
   `cli watch <run-dir>` to follow a run from another process or look at
   a finished one). A small HTTP server (no async runtime; a blocking
   server such as `tiny_http`) on a background thread. It serves a
   static single-page app embedded in the binary (`include_str!`,
   vanilla JS + inline SVG, no build step, no CDN so it works offline)
   and a JSON API:
   - `GET /api/events?since=<gen>` returns events newer than `<gen>`
     (the page polls every second; polling is chosen over WebSockets/SSE
     because it is trivially robust and needs no async code);
   - `GET /api/genome/<gen>` returns that generation's champion genome;
   - `GET /api/state` returns run config and current progress.
   The server only reads the event log and checkpoints, so a slow or
   disconnected browser can never block or slow training, and closing
   the tab loses nothing.

**Event schema** (one per generation; additive changes only, with a
`schema_version` field):
- run/progress: generation, elapsed, rounds evaluated, rounds/second;
- fitness: best, mean, median, min, standard deviation, a 10-bucket
  histogram, champion fresh-seed re-evaluation with standard error;
- vs baselines: champion's score and role distribution against each
  fixed opponent (re-using `statistics::aggregate`), plus voluntary
  pass rate and President-retention rate, so the Phase 4 statistics
  appear for the champion live;
- species: count, per-species size, age, best fitness, stagnation
  counter, current compatibility threshold;
- complexity: nodes/connections (min, mean, max, champion), enabled
  ratio, total innovations so far, mutation counts by kind that
  produced improvements;
- champion: reference to its genome checkpoint.

**Dashboard views**:
- *Overview*: progress bar, ETA, run config, and a new-best log.
- *Fitness*: best/mean/median curves with the champion's error band,
  and the population histogram as a heat strip over generations.
- *Versus baselines*: the champion's score against each pre-NEAT
  baseline over time, with the success thresholds from section 9 drawn
  as reference lines, so progress against the goal is visible directly.
- *Species*: stacked area of species sizes over generations (the
  standard NEAT "speciation swim-lane"), plus a table with age, best
  fitness and stagnation.
- *Complexity*: nodes/connections over time for champion and
  population.
- *Network*: the champion's topology as a layered SVG graph (inputs left,
  output right, hidden nodes placed by depth), with named input nodes
  (the feature names from section 4), edge width = |weight|, colour =
  sign, disabled connections dashed or hidden by a toggle. A generation
  scrubber replays how the topology grew; hovering a node or edge shows
  its id, innovation number and weight.
- *Decision inspector*: pick a recorded sample decision (the trainer
  stores a handful of champion decisions per generation: the candidate
  moves, their feature vectors, and the scores). The network view then
  shows the activation of every node for the selected candidate, so a
  viewer can see why the champion chose one move over another. This is
  computed in the browser from the genome and the stored features (a
  ~50-line feedforward evaluator in JS), not by the server.

All values shown must come from the event log or checkpoints, so
terminal, dashboard and offline analysis can never disagree.

Relation to the web interface (Phase 11, which follows this phase): the
dashboard is the first part of the `web` crate, not a throwaway. The
server, the embedded single-page app and the JSON API live in `web`;
`cli` depends on `web` to provide `--serve` and `watch`. Phase 11 then
adds the simulation-statistics pages (role outcomes, variance/luck
indicators, strategy comparison, evolved players next to the
hand-written ones) as further routes and pages in the same app, reusing
its chart, layout and polling code. To keep that extension cheap, the
dashboard code is organised as a generic shell (routing, polling,
chart helpers, themed layout) plus a `neat` page module; nothing in the
shell may assume NEAT-specific data.

## 7b. As built in Phase 10c

- **Where:** evaluation, run files, events and the loop live in
  `sim/src/training/`; `cli train` only parses arguments and prints
  (`cli/src/train*.rs`). Run directory: `config.json`, `events.jsonl`,
  `checkpoint.json` (replaced atomically after every generation),
  `best.json`, `gen-NNNN.json`.
- **Evaluation:** a genome's score is its mean finishing-role score
  (+1 President .. -1 last) over `matches_per_genome` matches. Match
  seeds derive from `(run seed, generation, index)`, so every genome in
  a generation plays the same deals, seats and opponents. The candidate's
  seat rotates with the match index, but opponents are *sampled* from the
  pool per match, not rotated: table position relative to the other
  players is a large effect (a clone of `LowestLegal` and `LowestLegal`
  exchange their results exactly when their seats swap), and cyclic
  rotation keeps neighbour order fixed.
- **Re-evaluation and the believable score:** each generation's champion
  is replayed against the mixed pool and against every opponent alone on
  one *fixed* set of matches (the same deals every generation, never used
  in training), so champions of different generations are compared like
  for like; the event carries the score, its standard error and the
  placement counts. `best.json` follows this score. Because picking the
  maximum over many generations inflates it, a new best is also scored on
  a second fixed *held-out* set that did not pick it; that held-out score
  is what the summary reports (in a 120-generation run the selection
  score overstated the champion by about 0.03-0.04).
- **Frozen opponents:** `--opponent neat:PATH` files are copied into
  `<run>/opponents/N-<stem>.json` and the run records
  `neat:@opponents/...`, so a resume cannot silently face a different
  player and two files with the same name stay distinct.
- **Stagnation:** a species is exempt from culling if it holds the best
  fitness ever recorded *or* this generation's champion (fitness is
  noisy; a stagnant species can hold the current best genome).
- **Crash safety:** the run directory holds a checkpoint from the moment
  it is created (a run killed in generation 0 is resumable), and a
  checkpoint records the feature count and feature-set version, so a
  resume from a build with different features is refused before the log
  is touched.
- **Default pool:** `lowest-legal`, `endgame-denial`,
  `adaptive:reading,tempo,bully`. `CardCounter` is omitted: it plays
  identically to `LowestLegal` in every table tried (0 of 200 all-same
  tables differ), so it adds nothing.
- **Deviations from section 7a, deliberately:** the event carries
  per-opponent placement counts rather than the full Phase 4 statistics
  (voluntary-pass and retention rates); the dashboard (10d) may add them.
  No decision samples are stored yet (10d's inspector).
- **Speciation:** the default compatibility threshold is 0.5 (not 3.0):
  random initial weights put genomes about 0.3 apart.
- **Resume:** the population state (including its xoshiro256++ generator)
  is checkpointed; resuming equals never stopping, also after SIGKILL.
  A crash between a generation's event and its checkpoint is repaired by
  trimming the duplicate event on resume.
- **Measured:** about 23k rounds/s on 4 cores in release; a 30-generation
  run (population 80, 50 matches x 6 rounds) takes about 35 s and its
  champion beats three copies of each default opponent with a score of
  about +0.55 to +0.6 (President in roughly half of all rounds),
  confirmed through the ordinary simulator.

## 7c. As built in Phase 10d

- **Where:** the server and page live in the `web` crate (std-only HTTP,
  no async runtime, no new dependencies); the page is embedded in the
  binary (`web/assets/`, vanilla JS and inline SVG, no CDN, light and
  dark themes). `cli train --serve [PORT]` runs it inside the training
  process (and keeps serving after the run ends, because the server lives
  there); `cli watch RUN_DIR [--port N]` is a standalone viewer for a run
  in another process or a finished one. It binds to `127.0.0.1` only
  (use an SSH tunnel to view a remote run), refuses requests whose `Host`
  is not `127.0.0.1`/`localhost`/`::1` (DNS rebinding), and only reads the
  run directory.
- **API:** `GET /api/state`, `/api/events?since=N` (with an `epoch`
  identifying the log: unique per server process and bumped when the log
  is rewritten by a resume, so a client never mixes two runs' data), `/api/genome/N|best`,
  `/api/decisions/N`. The event reader consumes only appended bytes,
  waits for a partial last line, and skips lines that do not parse.
- **Views:** progress and headline numbers; fitness (selection best,
  population mean, median, the champion on fixed matches with its
  standard-error band, held-out dots); the champion against each opponent
  alone; species as stacked areas; the population fitness spread as a
  heat strip; complexity; the champion's network with a generation
  slider, best/latest buttons and a disabled-connections toggle; and the
  decision inspector. The page polls once a second but redraws only when
  something changed (an idle tab does no DOM work and keeps the user's
  hover).
- **Decision inspector:** for every new best champion the trainer records
  12 real decisions from one held-out match (`decisions/gen-NNNN.json`:
  hand, table, every legal move with its feature vector and score). The
  browser replays the network on a chosen candidate, colours every node
  by its activation and checks its own replay against the recorded Rust
  scores (a real run replayed 700+ candidates with a maximum difference
  of 0).
- **Tests:** Rust tests for the event reader, routes and a real socket;
  Node tests for the helpers; a `cargo test` harness that runs the Node
  tests (skipped loudly without Node) including a replay of a real run's
  decisions; and Task 6's real-browser checks, including killing and
  resuming a run while the page is open.
- **Not built:** the Phase 4 statistics per champion (voluntary-pass and
  retention rates), mutation counts by kind, and a WebSocket/SSE channel
  (polling once a second is enough and trivially robust).

## 7d. As built in Phase 10e

- **Learning options** (`cli train`): `--champion-candidates N` re-scores
  the N genomes with the best training fitness on the fixed matches and
  takes the best as the generation's champion (default 5; the event's
  `training_rank` says where it ranked by training fitness: in the v1 run
  it was not the training best in 75 of 120 generations);
  `--hall-of-fame N --hall-interval K` keeps frozen past champions as extra
  training opponents (default off; persisted in the checkpoint, resumable;
  per-opponent reporting stays on the fixed pool and a `hall_score`
  reports the champion against the hall); `--weight-power P` is the size of
  a weight perturbation (default 0.2).
- **`cli evaluate`**: the baseline-comparison protocol (section 9) as a
  command: genomes against any opponents on a seed stream training never
  uses, one row per opponent plus a mixed row, optional JSON. Genomes can
  face each other with `neat:PATH` opponents.
- **Evidence:** `docs/baselines/neat-v1/experiments.md`: five
  configurations over three seeds and a 5-seed comparison of the combined
  options. All configurations generalize to opponents never trained
  against (+0.90 to +0.92); the combined options are better by about 0.016
  (1.3 standard errors) and became the defaults; the hall of fame showed no
  benefit against this pool.
- **Baseline comparison** (`docs/baselines/neat-v1/`): the first
  committed champion and the recorded comparison with the pre-NEAT
  strategies at 3-6 players (success criteria of section 9 met: it beats
  `lowest-legal`, `endgame-denial` and `adaptive:reading,tempo,bully` with
  differences far outside the seed-to-seed spread).
- **Not built:** a weak-to-strong opponent curriculum and matches growing
  over generations (the champion already beats every opponent, so neither
  has a measured need), complexity pressure in ties, the role feature and
  evolving the exchange step.

## 8. Determinism and performance

- A run is fully determined by `(config, seed)`: per-genome evaluation
  seeds derive from `(run seed, generation, genome index)`, evolution
  uses a single reproduction RNG, and results are independent of thread
  count (same property `run_batch` already has).
- Genomes are evaluated in parallel with rayon; each evaluation is
  independent and immutable.
- `turn_context_for` (`sim/src/match_runner.rs`) rebuilds the context
  every turn, including an O(deck) unseen-card scan. Measure first;
  only if it dominates training time, add a cheaper incremental path
  without changing what strategies see. Compiled networks should be
  evaluated with preallocated buffers to avoid per-move allocation.

## 9. Baseline comparison protocol

- Champions are evaluated with the exact lineups, seeds, rounds and
  matches used in `docs/baselines/pre-neat/run.sh`, with the champion
  taking the contender seat, and reported through the existing
  `statistics::aggregate` output.
- Success criteria, in order: (1) beats `lowest-legal` and
  `random-legal` field-wide; (2) beats `card-counter`; (3) stretch:
  at least matches `Adaptive(reading,tempo,bully)` in the mixed field.
  "Beats" means a higher President rate and a lower last-place rate
  with differences larger than the seed-to-seed spread across at least
  four independent seed bases (as done for Phases 8-9).
- Control experiment: evolve only the numeric knobs of `AdaptiveConfig`
  (e.g. deception rate) with the same loop. If the network cannot beat
  tuned heuristics, that is a finding about the feature set, not a
  failure of the engine.

## 10. Testing

- `neat` crate: innovation ids are stable within a generation; crossover
  of valid genomes is valid; add-connection never creates a cycle; the
  compiled network matches a hand-computed small example; genome JSON
  round-trips; speciation assigns every genome to exactly one species.
- Determinism: same seed and config give an identical generation-N
  champion, regardless of thread count; resume equals uninterrupted.
- `NeatStrategy`: always returns a move contained in `legal_moves`
  (property-style test over many random states), same result for
  identical context; genome with mismatched input count is rejected.
- Smoke test: tiny population, 3 generations, finishes, writes
  checkpoints, and fitness is finite.
- Monitoring: the terminal line renderer is a pure function of an event
  (snapshot-tested); every event line in `events.jsonl` parses against
  the schema; the HTTP API is tested against a recorded run directory
  (`/api/events?since=` returns only newer events, `/api/genome/<gen>`
  round-trips); the server serving a run never changes its training
  results (same seed, with and without `--serve`, same champion).
- `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace` stay clean (existing phase
  gate from `docs/CODING_GUIDELINES.md`).

## 11. Phasing

- **10a** `neat` crate core, fully tested, no game dependency.
- **10b** `NeatStrategy`, feature extractor, `neat:<path>` spec.
- **10c** Fitness evaluation, `train` subcommand, checkpoint/resume,
  `TrainObserver`, per-generation terminal output and `events.jsonl`
  (monitoring in the terminal is part of the first usable trainer, not
  an afterthought).
- **10d** Browser dashboard: JSON API, fitness/baseline/species/
  complexity charts, network view, generation scrubber, decision
  inspector, `--serve` and `cli watch`.
- **10e** Opponent pool/hall of fame/curriculum, tuning, comparison
  against the pre-NEAT baseline, results written up.

## 12. Risks and open questions

- **Noisy fitness** may stall selection; mitigations are common random
  numbers, growing evaluation budgets and champion re-evaluation.
- **Overfitting** to fixed opponents; mitigated by the hall of fame.
- **Feature expressiveness**: `Adaptive`'s edge comes from hand reading
  and denial/tempo/bully logic; if the features do not expose enough of
  that, the network is capped near `CardCounter`. Feature additions
  invalidate old genomes (versioned input count).
- **Dashboard scope creep**: the dashboard is a viewer over a stable
  event schema; new charts must not require trainer changes beyond
  adding fields to the event log.
- **Compute**: millions of rounds per generation; to be confirmed by
  profiling in 10c before optimizing.
- **Open**: whether to train one genome for all table sizes or one per
  size (default: one per size, cross-size transfer evaluated later);
  whether the exchange step should be evolved in v2.
