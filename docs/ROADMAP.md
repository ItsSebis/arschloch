# Roadmap

Phase numbers are stable names (specs, commits and code comments refer to them), not
an order. **Done** phases come first, with what each one built; below them, under
"Open work", is everything that was not done, in the order to do it. The table is the
overview of the done phases.

Each phase should leave the workspace building, tested, and documented before the
next one starts. Phases are scoped to be reviewable in a single sitting; a phase that
grows beyond that should be split.

| Phase | Title | Status |
|---|---|---|
| 0 | Foundations | done |
| 1 | Round & game state machine | done |
| 2 | Strategies & simulation runner | done |
| 3 | CLI | done |
| 4 | Deeper statistics | done |
| 5 | Smart exchange | done |
| 6 | Strategic context, card counting, and endgame denial | done |
| 7 | Adaptive strategy | done |
| 8 | Trick-lead tempo | done |
| 9 | Lead-order bullying | done |
| 10 | NEAT training | done |
| 11 | Web interface: play against the models | done |
| 12 | Richer scoring and statistics | done |
| 13 | NEAT: better play and wider training | done |
| 14 | Rule change: a pass ends your part in the trick | done |
| 15 | Rule change: the exchange gives your highest cards | done |
| 17 | Run analysis tool (CLI and web) | done |
| 19 | CPU performance and simplification | done |
| 21 | Endgame solver and determinized search | done |
| 22 | Desktop GUI to configure and run simulations | done |
| 11b | Web views for simulation results | done |
| 18 | Training advisor: pick the opponents and settings for you | done |

**Open work, in working order** (details below the done phases):

1. ~~Pack the desktop window into the release archives (Phase 22)~~ built; the Windows and macOS windows still have to be started once by hand
2. ~~Desktop window: file pickers, progress, a solver editor, a training form (Phase 22)~~ built; a few pieces left, see the section
3. ~~Results page: the luck-adjusted estimate, comparing two files (Phase 11b)~~ built
4. Training advisor, the rest: `train --auto`, scoring against genomes, early stop, a web form (Phase 18)
5. Settle the `hidden` input set with fresh seeds and longer runs; the casino bot's port (Phase 13)
6. Simplification left over from Phase 19 (`session.rs`, legacy rule paths, file sizes)
7. Endgame solver as a toggle on every strategy (Phase 21)
8. Endgame solver, phase B: the early game (Phase 21)
9. Phase 16, self-play and co-evolutionary training (an idea)
10. Phase 20, the GPU port (the largest and least certain)

# Done

## Phase 0 — Foundations (done)

- Cargo workspace scaffold: `engine`, `sim`, `cli`, `web` (placeholder)
  crates, workspace-level lint/format config.
- `engine`: `Suit`, `Rank`, `Card` (with rank/suit/duplicate-tiebreak
  ordering), `DeckVariant`, `DuplicateRule`, `Role` + the per-player-count
  role and exchange-count tables, `Combo` with a legal-play checker
  (singles and same-rank sets only — no straights/bombs yet).
- Unit tests for all of the above.
- `docs/` written: `RULES.md`, `ARCHITECTURE.md`, `ROADMAP.md`,
  `CODING_GUIDELINES.md`.
- Explicitly **not** in Phase 0: the trick-taking loop, round/game state
  machine, card exchange execution, strategies, simulation runner, CLI.

## Phase 1 — Round & game state machine (done)

- `engine`: full trick loop (lead, follow/pass, trick resolution, hand
  exhaustion, finishing order), round-end role assignment, and the mandatory
  best/worst card exchange (naive tie-break — just take the top/bottom N
  cards by the total order, no strategic selection yet).
- Scripted integration tests that play a full round move-by-move and
  assert final roles/hands match expectations.

## Phase 2 — Strategies & simulation runner (done)

- `sim`: `Strategy` trait, 2-3 baseline strategies (lowest-legal,
  random-legal, greedy-highest).
- Multi-threaded match runner (`rayon`), `MatchResult`/`Statistics` types,
  `serde_json` output.
- Integration tests asserting statistical invariants over small batches.

## Phase 3 — CLI (done)

- `cli` crate: `clap`-based configuration (player count, deck variant,
  duplicate rule, match count, seat strategies, thread count).
- JSON results file + human-readable summary table on stdout.
- End-to-end smoke tests.

## Phase 4 — Deeper statistics (done)

- Role-sustainment tracking: does a strategy that becomes President tend
  to stay President across repeated matches with role carry-over, or is it
  closer to random?
- Luck-vs-skill signal: outcome variance across repeated matches with
  identical strategy assignments but different shuffles.
- Strategy diversification: pass-when-a-beating-play-exists frequency, and
  at least one strategy that deliberately does this (e.g. sometimes holds
  back a matching set rather than always playing greedily) to compare
  against always-play-if-possible baselines.

## Phase 5 — Smart exchange (done)

- Replace Phase 1's naive top/bottom-N tie-break with strategy-aware card
  selection for `choose_exchange_cards` — e.g. prefer to give away an
  isolated high card over one that's part of a pair/triple the strategy
  would rather keep, when multiple cards tie for "the Nth highest".

## Phase 6 — Strategic context, card counting, and endgame denial (done)

- `engine`: `Round` gains a play-history accessor (every combo played so
  far this round, tagged with the seat that played it) and a per-seat
  hand-size accessor — the one piece of information about *other* seats
  this genre treats as public.
- `sim`: `Strategy::choose_play` gains a `TurnContext` parameter built
  fresh each turn: this seat's own hand, every opponent's hand size (and
  whether it's still active), and the exact multiset of unseen cards —
  deterministic, since this is a closed-deck game with no draw pile.
- Two new strategies built on `TurnContext`: `CardCounter`, which plays
  a legal combo while some unseen card could still beat it, and holds
  it back once no unseen card can; and `EndgameDenial`, which switches
  to aggressive, control-retaining play whenever an active opponent's
  hand size is low enough to be close to finishing, to deny them an easy
  trick. Kept as two separate strategies rather than merged into one so
  Phase 4's existing statistics can show which technique — counting or
  denial — actually helps.

## Phase 7 — Adaptive strategy (done)

- `engine`: `Round` gains a pass-history accessor alongside Phase 6's
  play history (every pass this round, tagged with the seat, the combo
  it declined to beat, and the play count at that moment) — the raw
  material for hand reading.
- `sim`: a new `hand_reading` module reduces a round's play/pass
  history into per-seat "pass ceilings" (the lowest top card a seat is
  known unable to beat, per combo size), with unrefuted-pass logic so a
  seat that later plays above a card it passed on doesn't leave a false
  ceiling behind. `TurnContext` gains these ceilings (both the acting
  seat's own and every opponent's) plus the combo currently on the
  table, and `Strategy::name()` changes from `&'static str` to `&str`
  so a configurable strategy's name can reflect its actual
  configuration.
- One new strategy, `Adaptive`: `LowestLegal` (or `CardCounter`, if
  card-counting is enabled) as its base play selection, with two more
  independently-toggleable modifiers layered on top — endgame denial
  (either a plain hand-size trigger matching `EndgameDenial`, or the
  same trigger sharpened by hand reading to spend the *lowest* card
  that provably locks a close-to-finishing opponent out, rather than a
  blanket highest-card push) and deception (occasionally bluff-passing
  on a seat's only beating rank, to plant a false pass ceiling in an
  opponent's hand reading). Each modifier reuses the existing
  `CardCounter`/`GreedyHighest`/`LowestLegal` strategies by direct
  delegation rather than duplicating their logic, so Phase 4's
  statistics can show which modifier combination actually beats plain
  `LowestLegal` — the strategy that beat every existing strategy in an
  empirical batch run.
- `cli`: `--strategy` becomes a per-seat spec grammar, `FIXED |
  "adaptive" | "adaptive:" OPTIONS`, so a batch run can seat any mix of
  the six fixed strategies and independently-configured `Adaptive`
  instances (e.g. `adaptive:reading,deception=0.2`) in the same table.

## Phase 8 — Trick-lead tempo (done)

- A fourth `Adaptive` modifier, `tempo`: finishing a round is a race to
  be the one who *leads* a trick while holding exactly one card
  (leading your last card finishes you unconditionally; being stuck
  *responding* with your last card means someone already claimed that
  position). Once this seat's own hand is down to 2 cards, `tempo`
  spends the cheapest legal play that's provably safe against the
  *whole* active field — reusing the same proof `denial` uses, now
  extracted into a shared `adaptive::safety` module — instead of the
  base strategy's cheapest-legal instinct, to seize the next trick's
  lead rather than risk losing that race to an opponent's
  re-escalation.
- Empirically validated (not just argued): across four independent
  seed bases and all four supported table sizes, `close=2` gave a
  consistent ~4-6% relative edge over plain `lowest-legal` in a clean
  head-to-head, with monotonic dominance across *every* role (higher
  President/Vize/Offizier rates, lower Dummkopf/ViceArschloch/Arschloch
  rates simultaneously) — the first modifier in this project validated
  to beat `lowest-legal` outright rather than trade placement for
  safety. The threshold is deliberately a fixed constant, not a
  configurable `tempo=<n>` option: `close=1` was statistically a wash
  and `close=3` and up actively lost ground, so this sweet spot doesn't
  extend safely in either direction.
- `cli`: `adaptive:tempo` (combinable with every other modifier, e.g.
  `adaptive:reading,deception=0.2,tempo`).

## Phase 9 — Lead-order bullying (done)

- A fifth `Adaptive` modifier, `bully`: finishing a round on your very
  last card locks in your placement unconditionally, the same mechanic
  `tempo` relies on. A hand shaped mostly as same-rank groups (pairs,
  triples, ...) plus one or a few leftover singles can turn that into a
  free win *if* every group is led before any single — opponents who
  can't match a group are forced to pass, handing the lead straight
  back, while leading a vulnerable single too early risks losing the
  lead (and every unplayed group's leverage) to whoever beats it. While
  leading, with this seat's hand shaped mostly as same-rank groups
  (`engine::rank_groups` groups of size >= 2 at least as numerous as
  singleton groups), `bully` leads the cheapest *whole* same-rank group
  instead of the base strategy's own cheapest-legal instinct — careful
  to never split a larger group into a smaller lead, since
  `engine::legal_moves` offers every subset size of a held group while
  leading, not just the smallest and the whole thing.
- Empirically validated (not just argued): in a clean head-to-head
  against plain `lowest-legal` across four independent seed bases and
  all four table sizes, `bully` gave a large, monotonic edge across
  *every* role (higher President/Vize/Offizier rates, lower Dummkopf/
  ViceArschloch/Arschloch rates simultaneously) — markedly bigger than
  `tempo`'s own ~4-6% edge. The design originally included a third
  gate, an opponent-proximity threshold (by analogy to `denial`/
  `tempo`), but sweeping that threshold showed the edge only grew as
  the gate was relaxed, plateauing once it exceeded every realistic
  hand size rather than reversing — the gate was never protecting
  against a downside, so it was removed entirely rather than shipped as
  a no-op constant. `bully` also stacks cleanly with `tempo` (their
  trigger windows don't overlap), but its marginal contribution
  vanishes once `denial`'s `reading` mode is also enabled — `denial`
  runs earlier in the pipeline and is a strictly broader trigger
  (fires on both leading and following), so it already claims most of
  the same opportunities once active.
- `cli`: `adaptive:bully` (combinable with every other modifier, e.g.
  `adaptive:tempo,bully`).

## Phase 10 — NEAT training (done)

- Evolve feedforward neural-network players with NEAT and compare them
  against the hand-written strategies. Full design:
  `docs/superpowers/specs/2026-10-08-neat-engine-design.md`.
- The pre-NEAT state is preserved as git tag `baseline-pre-neat` plus
  frozen result summaries in `docs/baselines/pre-neat/`; existing
  strategies are not modified by this phase.
- 10a (done): generic `neat` crate. 10b (done): `NeatStrategy` (scores each legal
  move) and the `neat:<genome.json>` spec. 10c (done): fitness evaluation and
  a `train` subcommand with checkpoint/resume, a per-generation
  terminal progress line and a JSONL event log. 10d (done): live browser
  dashboard, the first part of the `web` crate, which Phase 11 extends
  (fitness/species/complexity charts, champion network visualization,
  decision inspector). 10e (done): opponent pool, hall of
  fame, tuning and baseline comparison. 10f (done): warm start from an
  earlier run (`--from`), sets of runs with a set-wide ETA (`--runs`), and a
  CPU speedup of about 1.5x with measurements that rule out GPU offload of the
  network (see the GPU idea below).

## Phase 11 — Web interface: play against the models (done)

**Status:** built: `cli play` ("play against the models"; see `docs/PLAYING.md`) with
a pure `sim::session` game session, a POST-capable local HTTP layer, model advice and
personal records. The run-analysis page is Phase 17 (done). The simulation-statistics
overview views, originally listed here, are now Phase 11b; the training-advisor form is
part of Phase 18. Both stay in the browser (see Phase 22 for why).

- Extends the `web` crate that Phase 10d starts with the NEAT training
  dashboard (HTTP server, embedded single-page app, JSON API), rather
  than building a second frontend.
- Reads (or live-drives) `sim`'s existing output types — no changes to
  `engine`/`sim`'s public shape were needed.
- **Play against the models:** a user can sit at a table in the browser
  and compete against one or several trained models (genome files from
  `cli train`, for example `docs/baselines/neat-v1/champion.json`), and
  optionally against hand-written strategies, at any table size 3-6. The
  server holds the game state and asks a `NeatStrategy` for its move
  through the same `Strategy` interface the simulator uses; the page shows
  the human's hand, the table and the other seats' hand sizes, and offers
  the legal moves. Results (roles per round) are kept so a human's record
  against each model can be viewed and compared with the models' own
  scores.

## Phase 12 — Richer scoring and statistics (done)

**Status:** built (design: `docs/superpowers/specs/2026-10-09-phase-12-statistics-skill-score-design.md`).
Every statistic is catalogued once in `sim::stats_catalog` (meaning, formula, how to
read it, caveats, JSON path) and the catalogue generates `docs/STATISTICS.md`, the
`--explain` text, the `statistics_catalog` key of `results.json` and
`GET /api/statistics-catalog` for the later website page. New numbers: average rank and
mean role score per strategy and per seat, rank distribution, the opponent-adjusted
Bradley-Terry strength rating (bootstrap errors), Wilson intervals, useful passes
(`--useful-passes K`, counterfactual rollouts), and the luck/skill statistic
(`--skill-score off|estimate|duplicate|both`: duplicate deals as ground truth, a cheap
hand-feature regression as estimator, compared in `both`). The training hook `--fitness-skill-weight` is built but off by default; whether it helps is a Phase 13 experiment. Findings: duplicate deals leave the point estimate of a strategy's score
unchanged (only its error shrinks); deal luck is only about 0-50% of the single-match
variance (the champion about 6% over six rounds, weaker strategies up to 50%), because
carried-over roles dominate later rounds; the estimator costs no extra time and agrees
with the duplicate result within about 1 standard error at 4 players (it removes less of
the luck at 5 players with the double deck); `card-counter` plays identically to
`lowest-legal` in every setup tested (its "precious card" branch never fires).

Original plan, kept for reference:

From `docs/Notes.md`. Today the CLI summary reports mainly President
retention, and training scores a player by its mean finishing role
(+1 .. -1). Planned:

- **More numbers in the CLI summary and the output JSON:** average finishing
  rank per strategy (and per seat), and **useful passes** (a pass that was
  followed by a better outcome than playing would have given: for example a
  pass after which the player later led a trick it would have lost by playing
  early, or a pass that denied an opponent; the exact definition is the first
  task of the phase and is documented in `RULES.md`/the statistics docs).
  Existing keys stay; new ones are additive (`docs/BUILDING.md` documents the
  JSON).
- **Scores that adapt to how the other players react:** the Notes ask for a
  score that takes the opponents' behaviour into account instead of one fixed
  role scale. First step is to settle what this means (candidates: weight a
  result by the strength of the table, which the existing mixed-table
  position bias already shows matters; or report a score relative to what each
  opponent set achieves against a fixed reference). Decide with the user before
  building; until then nothing changes in `role_score`.
- Show the same numbers in the dashboards (training dashboard cards, the play
  page record) once they exist in the output.

## Phase 13 — NEAT: better play and wider training (done)

**Status:** built and measured (`docs/baselines/phase-13/README.md`). Result:
`--features hidden` (hidden-hand inputs) is significantly stronger than the
20 original inputs head-to-head against the neat-v2 champion (+0.048, Welch
t = 3.68, 10 seeds each); the fitness variants (`risk`, `tanh`) and racing
show no measurable gain and stay opt-in; one model for all tables
(`--features all`, sizes 3-6, both decks) matches the specialists except at 3
players. No default changed (`--features` is still `base`); making `hidden`
the default waits for a confirmation run with fresh seeds (open work, "Open work").
The casino repo's TypeScript NEAT bot only implements feature set v1, so `hidden`
and `all` genomes need its feature port extended first (also open work).

From `docs/Notes.md`. These change what the evolved players are rewarded for
and what they can see, so every item needs the Phase 10 discipline: a spec
note, a frozen comparison against `docs/baselines/neat-v1`, and a measured
result before it becomes the default.

- **Who is still in the trick:** under the pass rule `final` a seat that has
  passed is out until the trick ends. The neat inputs do not yet say who that is
  (`FEATURE_SET_VERSION` bump), and the strategies' `TurnContext` could carry it.
- **Predict the other players' cards:** nudge the network towards modelling
  hidden hands. Options to try, cheapest first: add inputs derived from the
  known information (an estimate of each opponent's likely holdings from the
  unseen cards and their pass history, building on `PassCeilings`), then, if
  that helps, an auxiliary training objective (a small prediction head scored
  against the real hands, used only during training). Measure against the
  current 20 inputs; a new feature set bumps `FEATURE_SET_VERSION`, which
  retires old genome files, so batch such changes.
- **Stable rank over rushing for President:** "retain the highest possible
  rank, but settle on a lower one if the cards are bad; do not rush for
  President, stay safe on Vize." Concretely a fitness variant that rewards
  consistency (for example mean role score minus a penalty on the variance of
  the roles, or a concave utility that values Vize almost as much as
  President), selectable like the other training options and compared with the
  default fitness on the same held-out matches. Report both mean score and
  spread so the trade-off is visible.
- **One model for several table sizes and both decks:** train a genome that
  adapts to the player count and the deck variant instead of one model per
  setting. Add inputs for the player count (and the deck variant) and let
  training draw the table size and deck per match from a configured set
  (`--player-counts 3,4,5,6`, `--decks single,double`); fitness is then
  averaged over the settings with per-setting columns in the terminal and
  the dashboard, and `cli evaluate` reports every setting. A first experiment
  is whether a single genome loses to the specialised ones; the baselines for
  3-6 players already exist in `docs/baselines/neat-v1`.

## Phase 14 — Rule change: a pass ends your part in the trick (done)

From `docs/Notes.md` ("once you pass in a trick you are out and cannot rejoin the
trick"). `docs/RULES.md` now says a pass is final for the trick, and that is the
default everywhere (`--pass-rule final`); the older behaviour, where a passed player
could still play later in the same trick, is kept as `--pass-rule free` so earlier
results can be reproduced.

- Engine: `engine::PassRule` in the trick bookkeeping (`Round::with_pass_rule`;
  `Round::new` plays by the rules of the game), tested with scripted tricks, a
  random-round property test for 3-6 seats and both decks, and a mutation check.
- The rule is a field of `MatchConfig`, `TableSpec`, `TrainConfig` and the play
  session, and an option of every CLI mode and the play page. Old run directories
  carry no rule and resume as `free`; `--resume` cannot change it.
- Re-baselined: the pre-neat and neat-v1 scripts now state `--pass-rule free`
  explicitly and still reproduce their committed summaries (apart from the header
  line); `docs/baselines/pass-final` re-measures both under the new rule, and
  `docs/baselines/neat-v2` is the champion retrained under it (built into
  `cli play`).
- Finding: the rule only changes games where someone passes while holding a beating
  play. Most hand-written strategies never do (their results are byte-identical
  under both rules); the evolved players do occasionally. The retrained champion is
  not measurably stronger than v1. `PassCeilings` hand reading is unchanged: a pass
  still proves the seat held no beater, and a later play in a later trick still
  refutes a dishonest pass.
- Done in Phase 13 (`--features trick`): neat inputs for "who is still in this trick", which the new
  rule makes meaningful.

## Phase 15 — Rule change: the exchange gives your highest cards (done)

The lower role of an exchange pair must hand over its highest cards (no choice),
as in the real game; `docs/RULES.md` says so and it is the default everywhere
(`--exchange-rule forced`). The older behaviour, where the lower role or its
strategy chose any cards, stays as `--exchange-rule free` (a game modifier, and
how earlier results were measured). Carried like the pass rule: `engine::
ExchangeRule` with `exchange_with_rule`, fields of the match, training,
evaluation and play-session configs and of the play records, options of every CLI
mode and the play page (under `forced` the human is never asked to choose and
only sees what was taken and received); old runs, configs and records read as
`free`. Finding: only `random-legal` and `hold-back-pairs` ever gave other
cards, so every other strategy plays identical games under both rules and the
retrained champion is byte-identical (`docs/baselines/current-rules`).

## Phase 17 — Run analysis tool (CLI and web) (done)

**Status:** built (design: `docs/superpowers/specs/2026-10-11-phase-17-run-analysis-design.md`;
PR #10). `cli analyse-run` (several paths, set dirs, folders of loose `.jsonl`, `--json`)
and the web page `/analysis.html` (served by `cli watch` and `train --serve`)
report stall generation, trends, noise, winner's-curse gaps, per-opponent
trends, growth, convergence, species, racing and timings, and a verdict with
suggestions; the logic is `sim::training::analysis`, shared by both. Every number is in
the statistics catalogue (`Scope::RunAnalysis`, `docs/STATISTICS.md`); usage and
how to read each verdict in `docs/TRAINING.md` ("Analysing a run"); the
real-data check (the four `runs/PC` runs and the ten-seed Phase 13 sets) in
`docs/baselines/phase-17/README.md`. The verdicts are heuristics with fixed
thresholds; Phase 18 consumes the same analysis. The text below is the
original plan.

Born from the first stagnation analysis of the long runs in `runs/PC` (done by
hand with a throwaway script). Turn that into a tool so every run can answer
"where did it stop improving, and why?" in one command.

- **`cli analyse-run <events.jsonl | run dir> [more...]`**: accepts several
  runs at once and labels them by file or directory name. Reports per run:
  - the **stall generation** (smoothed champion re-evaluation within one
    standard error of its peak) and the **trend** (slope per 100 generations
    with a 95% interval, overall and for the last part of the run);
  - the **noise level** (standard error of a single evaluation, standard
    deviation of the generation-to-generation change of the champion's score);
  - the **winner's-curse gap** (best generation's re-evaluation against its
    held-out score, and training best against champion level), so a lucky "best
    generation" is not mistaken for progress;
  - **per-opponent trends** (flat everywhere = information/noise limited; flat
    against some and rising against others = pool limited);
  - **growth without gain** (hidden nodes and connections against score),
    **population convergence** (share of genomes in the top fitness bin), species
    counts and the compatibility threshold;
  - a plain-language **verdict with suggestions** (more matches per genome or the
    duplicate-deal fitness when noise dominates, new inputs when the model stalls
    everywhere, harder opponents when only the pool is exhausted, stop early
    when the slope is flat).
  - `--json` output for other tools; the numbers are catalogued and explained
    like the Phase 12 statistics (`docs/STATISTICS.md`).
- **Web interface (the Phase 11 dashboard work):** the same analysis as a page
  of the web interface: pick one or several run directories or imported
  `events.jsonl` files, see the curves with the stall point, noise band,
  winner's-curse markers and per-opponent lines, the verdict text, and compare
  runs side by side. It reads the same JSON as the CLI; no new simulation.

## Phase 19 — CPU performance and simplification (done)

**Status:** the speed work is built and merged (PR #6, released in v0.5.0); measurements in
`docs/baselines/perf/README.md`. Its simplification part is only partly done: what is left is open work (item 6).

The cheaper road to faster training, done before any GPU work (Phase 20), and
useful on its own. Reference points measured on the owner's Ryzen 7800X3D (8
cores / 16 threads): the long runs in `runs/PC` reach about 39,000 rounds/s
(4 players, single deck, hand-written opponents) and about 11,000 rounds/s
(5 players, double deck, two neat and one adaptive opponent), barely above the
i5-11300H laptop's 30,000 rounds/s, so either the heavier strategies cost a lot
per round or the 16 threads do not scale. Finding out which is the first task.

- **Measure first:** a per-generation timing breakdown in the training events
  (training evaluation, champion candidates, re-evaluation, opponent scores,
  speciation, checkpoint and file writes) and a thread-scaling check
  (`--threads 1, 2, 4, 8, 16` on the same run); micro-benchmarks per strategy
  (one `bench.sh` row each for card-counter, endgame-denial, adaptive with
  reading, neat) next to the existing ones. Record in `docs/baselines/perf`.
- **Cheap CPU wins from `docs/baselines/perf/README.md`, "What is left":**
  `Round::legal_moves` allocations (a `Vec` per candidate combo and helper
  vectors per rank group), `TurnSummary::new` building `Vec<Vec<Card>>` twice per
  decision, `Round::submit_move` cloning the hand to validate. Reusable scratch
  buffers and fixed-size, bitmask-based hands where the profile says so.
- **Heavier strategies:** the per-turn cost of the adaptive modifiers (pass
  reading, safety proofs), card counting and neat feature building, which the
  profile will rank; the same data structures feed Phase 20.
- **Training loop:** remove serial sections and contention found by the timing
  breakdown (for example per-generation file writes, species distance
  computation, opponent re-evaluation granularity).
- **Simplification pass:** split files over the ~400-line guideline, remove or
  isolate code that only exists to reproduce very old results where it is not
  needed, merge duplicated logic across strategies, refresh outdated comments;
  nothing is removed that a committed baseline still needs.
- **Findings of the exploration (code reading, not yet measured):**
  - **Why 16 threads do not scale:** `sim::training::evaluate` is serial per
    candidate; rayon is used only across genomes, champion candidates and
    opponents. Training runs its genomes in 10 chunks, each its own `par_iter`
    with a barrier (15 tasks per chunk at population 150, 6 at population 60),
    and the champion selection (5 tasks), the mixed re-evaluation (1 thread), the
    per-opponent evaluation (3 threads), the hall of fame (1 thread), the
    confirmation (1 thread, on a new best) and the decision sample (1 thread)
    run with little or no parallelism. Roughly 12% of the rounds run on 1-5
    threads, which can double the wall time. Fix, results identical: parallelise
    `evaluate` over matches (order-preserving collect, sequential sum), run the
    re-evaluation stages in one `rayon::scope`, replace the chunk barriers by an
    atomic progress counter; also try mimalloc/jemalloc (no global allocator is
    set; many small allocations on 16 threads).
  - **Per-turn allocations (about 60-90 per turn, roughly 2 µs per turn):**
    `legal_moves` (sorted hand copy, a `Vec` per rank group, size lists,
    subset vectors, and `Combo` is a heap `Vec<Card>`); `turn_context_for`
    builds opponents, a 256-byte seen array, the unseen list and updates the pass
    tracker for every seat although four strategies never read the context and
    `EndgameDenial` needs only hand sizes; `TurnSummary::new` builds
    `Vec<Vec<Card>>` groups twice, and each candidate scans the whole hand and
    unseen list (features 3, 17, 18); `Round::validate_play` clones the hand and
    `active_mask` allocates per move. Fixes: inline `Combo` (`[Card; 8]` plus
    length), `legal_moves_into` with a reused buffer and no Vecs, a
    `Strategy::needs()` flag set for a lazy context (pure data, safe), per-rank
    counts and `u128` masks for unseen cards, a prefix-min in the pass
    tracker. All must keep the move order identical (Pass, ascending rank,
    ascending size, lowest before highest), which neat's tie-breaks rely on.
  - **Simplification:** `session.rs` (1035 production lines) duplicates the
    deal/exchange/role loop of `run_match`; `trainer.rs` and `useful_passes.rs`
    exceed the size guideline; duplicated helpers (`min_by` on (size, top card)
    in four strategies, `highest_unseen` twice, `strength()` twice); about 37
    "Phase N" mentions in `sim` and `engine` comments to move into docs; the
    legacy `--pass-rule free` / `--exchange-rule free` paths (about 115 lines plus
    flag plumbing in about 25 places and `Strategy::choose_exchange_cards` in 9
    strategies) could move behind a `legacy-rules` feature, only if the old
    baselines are still wanted.
- **Order of work:** (1) per-stage timers in the generation events and a
  thread-scaling run (`THREADS=1,4,8,16`); (2) `evaluate` over matches, the
  re-evaluation scope, atomic progress; (3) a global allocator trial;
  (4) stack-based `active_mask`, no-clone `validate_play`; (5) inline `Combo` and
  allocation-free `legal_moves`; (6) `Strategy::needs()` and the lazy context;
  (7) `TurnSummary` on counts and masks; (8) prefix-min and the duplicated
  helpers; (9) file splits and legacy-rule isolation. New benchmarks first: a
  5-player double-deck case with two neat players, adaptive and lowest-legal
  (the 11k rounds/s setup, missing from `bench.sh`), a training thread-scaling
  case at population 150, and micro-benchmarks of `legal_moves` and
  `TurnSummary::new`.
- **Acceptance:** every speed change must leave the `bench.sh` checksums and the
  reference batch outputs byte-identical (the rule of `docs/baselines/perf`); the
  phase ends with a before/after table and the new rounds/s numbers for the two
  real training setups above.

## Phase 21 — Endgame solver and determinized search (done)

**Status:** phase A built and measured (spec: `docs/superpowers/specs/2026-10-16-endgame-solver-design.md`,
plan: `docs/superpowers/plans/2026-10-16-endgame-solver.md`, user guide `docs/SOLVER.md`, measurements
`docs/baselines/solver/README.md`). Result: the solver clearly beats the strategy it falls back to (+0.14 score head to
head, luck-free) but plays exactly only the last ~11% of the decisions, so it does not catch the neat-v2 champion
(+0.63 against -0.08 in the same table). The hand-off threshold defaults to 6 unseen cards (T=8 is too costly). A
position memo made the search about 95x cheaper. The early-game sampler (phase B) that would close the gap is open
work, see "Open work".

Goal: find the best overall strategy for real games, and learn from it. A
search-based player that, from one seat's view, considers every distribution of
the unseen cards the other players could hold and picks the move with the best
expected finishing position (max^n, rational opponents).

- **Built:** `Round::from_position` in `engine` (a pure
  addition), an exact `EndgameSolver` strategy that solves once few cards are unseen
  and defers to a fallback before that, `endgame-solver[:OPTIONS]` as a strategy spec,
  `cli solve` (explain a written-down situation move by move) and `cli solver-gap`
  (where do Adaptive, endgame-denial and the NEAT champions disagree with the solver,
  and what does it cost), a measured hand-off threshold and a luck-free head-to-head.
- Limits recorded in the spec: expected value over consistent deals, not a proof
  against a specific hidden hand; `PassRule::Final` only; no bluffing.

## Phase 22 — Desktop GUI to configure and run simulations (done)

**Status:** built (`gui` crate, binary `arschloch-gui`; user guide `docs/GUI.md`). A native egui window that builds the
arguments of a `cli` run from a form with every option, runs the real `cli` as a child process, streams its output into
an activity log (also written to `arschloch-gui.log`), cancels it, saves and loads presets, runs any `cli` command,
and starts `cli play` / `cli watch` and opens the browser for the parts that stay on the web (play against the
models, dashboards, run analysis, and the result charts of Phase 11b). Behaviour cannot drift from the command line:
the window has no simulation code, and a test runs the real `cli` with the arguments the form builds.

It is not in the workspace's `default-members` because it pulls in a windowing stack (about 300 more crates in
`Cargo.lock`); CI's `--workspace` runs include it.

**Packaging (open-work item 1, built):** `arschloch-gui` sits next to `cli` in the release archives:
`.github/workflows/release.yml` builds `-p cli -p gui` for both macOS targets and packs it with `docs/GUI.md`;
`docs/RELEASING.md` has the Linux and the `cargo-xwin` Windows commands and the archive layout. Verified: the Linux
release build and the window started from an extracted archive folder (it found the `cli` next to it); the Windows
cross build (`cargo xwin build --release --target x86_64-pc-windows-msvc -p cli -p gui`) links and `arschloch-gui.exe`
is a GUI-subsystem PE32+ (no console behind it; its `cli.exe` children start without a console window via
`CREATE_NO_WINDOW`). **Not verified:** nobody has started the window on Windows or macOS (no machine, no Wine here), and
the macOS workflow step is untested until a release runs it.

**Window features (open-work item 2, built):** a built-in file picker (`gui/src/browse.rs`; a folder list with up,
enter, an extension filter and a typed name for saving: `rfd` was rejected because its Linux backends need GTK system
packages or a D-Bus portal stack, which cannot be checked on CI's three systems from here) on the output file, the
preset, the genome (`neat:PATH`) of a seat, the run folder of `cli watch` and of the training; a **progress bar**
(`cli train` prints `progress: FRACTION TEXT` lines on stderr when `ARSCHLOCH_PROGRESS` is set, which the window sets;
the output is unchanged without it); a **Solver** tab (paint the cards of a position, the other seats' counts, the
table holder and passes; the real `cli solve` runs on the generated position file and the ranked moves are shown as a
table); a **Training** tab (the advisor form, its measurements, the recommended `cli train` command split into
editable fields with unknown flags passed through, a Start button). The logic (listing, progress lines, position text,
report and recommendation parsing) is unit-tested, and tests run the real `cli solve`, `train-advisor` and `train` on
what the window builds; the window was started under the real Wayland session on Linux and its Solver and Training
tabs screenshotted. No clicking was possible (no X server tool such as `xvfb` or `xdotool` installed), so button
handling itself is untested.
What is still open: a progress hook for a plain simulation (`sim::run_batch_with` is a single `par_iter`; its bar only
says "running"), a solver editor for double-deck positions (needs a count per face), and clicking through the window on
Windows and macOS.

## Phase 11b — Web views for simulation results (done)

**Status:** built. `cli view results.json` (`web::ResultsApp`, `web/assets/results*.{html,js}`) serves a page for the
results file of a batch: the strategies best first with average place, mean score and strength rating (each with its
standard error) and the share of rounds in every place as a stacked bar, the luck-adjusted score of a duplicate-deal
run, and the explanation of every number from the statistics catalogue stored in the file. Reload the page after a new
run. It reads the existing output only; the logic is a pure library tested in Node, the handler in Rust. The GUI
(Phase 22) starts it for the last simulation. The luck-adjusted estimate (with its agreement with the duplicate score when both ran) is drawn, `cli view A.json B.json` (or a file picked in the page, read in the browser) puts two files side by side per strategy with the difference (± its combined standard error) and a verdict, clear or within noise (no z value is shown; the rating table gets no verdict, and a missing standard error gives no verdict), and only the explanations of what the page shows are listed (not the run-analysis statistics).

## Phase 18 — Training advisor: pick the opponents and settings for you (done, reduced)

**Status:** built (reduced): `cli train-advisor` (`cli/src/advisor.rs`, user guide in `docs/TRAINING.md`, "Choosing
settings"). It probes the machine with a reference player (`adaptive`) against the candidate opponents, drops opponents
with no headroom (the reference wins almost every game, or almost never), sizes matches per genome from the measured
per-match noise (a standard error of about 0.02 on one fitness measurement), sizes population and generations from the
measured speed and the time budget (presets `quick`, `thorough`, `overnight`; a plan that cannot fit 20 generations is
warned about), and prints each choice with its reason plus a `cli train` command that `cli train`'s own parser is
tested to accept. The rest of the original idea is open work, see "Open work".

# Open work, in working order

What the done phases above did not do, plus the two phases that were never started (8 and 9). The order is by size and
risk: small polish first, the training workflow next, decisions that need measuring after that, the research and the
GPU port last; items do not depend on each other unless said. Phase numbers in brackets say where the idea came from.

## 1. Pack the desktop window into the release archives (Phase 22)

Built, see Phase 22 ("Packaging"). Left: start the Windows and the macOS window once from an extracted archive and run a
few matches, the solver and the advisor (listed in `docs/RELEASING.md`); the macOS workflow step runs for the first
time with the next release.

## 2. Desktop window, what it lacks (Phase 22)

Built, see Phase 22 ("Window features"). Left, each small: a progress hook for a plain simulation (an atomic counter
in `sim::run_batch_with`, shown through `progress:` lines; must not change the seat rotation index or any output),
a double-deck position editor (a count per face instead of a per-card paint), and a native file dialog if a CI
check ever shows `rfd` builds on all three systems without extra packages (the built-in picker works everywhere).

## 3. Results page, what it lacks (Phase 11b)

Built, see Phase 11b. Not done: comparing results files of different kinds (an `evaluate` file against a `cli` run) and
more than two files.

## 4. Training advisor, the rest (Phase 18)

`cli train --auto` (run the recommended settings and write them into the run's config instead of printing a command);
score the candidate opponents against a **starting genome**, a **models folder** and the **hall of fame** of earlier
runs instead of the stand-in reference player (the advisor judges against `adaptive`, which is far stronger than an early
network); use the **duplicate-deal skill score** where it cuts the noise (Phase 12); **advise or stop a running training**
from the Phase 17 analysis once the slope is flat; and a **"New training" form** in the web interface with the
recommended values and the reasons next to them. Depends on Phases 12, 13 and 17 (all done). Also worth fixing while
there: the speed is measured over all candidates, not the chosen pool.

## 5. Settle the `hidden` input set, and the casino bot (Phase 13)

Phase 13 found `--features hidden` significantly better than `base` head to head (+0.048, t = 3.68, 10 seeds) but kept
`base` as the default until a **confirmation with fresh seeds** (11 to 20, `docs/baselines/phase-13/run.sh all A2`) and
**longer runs** (300 generations, more matches per genome; the advisor of item 4 sizes them). If it holds, make `hidden`
the default and retrain the committed champion. The casino repo's TypeScript NEAT bot implements feature set v1 only, so
its feature port has to learn the new inputs before a `hidden` or `all` genome can play there (outside this repository).
Needs hours of compute; the result decides a default, so it is measured, not assumed.

## 6. Simplification left over from Phase 19

Done in Phase 19: the speed work and its measurements. Not done: `sim/src/session.rs` (1038 lines) still repeats the
deal/exchange/role loop of `run_match`; `trainer.rs` (455) and `useful_passes.rs` (408) are over the 400-line guideline;
duplicated helpers across strategies (`min_by` on size and top card, `highest_unseen`, `strength()`); and the legacy
`--pass-rule free` / `--exchange-rule free` paths (flag plumbing in about 25 places, `choose_exchange_cards` in 9
strategies) could move behind a `legacy-rules` feature, **only if** the old baselines are no longer wanted. Every change
must leave the `docs/baselines/perf/bench.sh` checksums and the reference batch outputs byte-identical. Small leftovers
of Phases 21 to 22: `StrategyName` in `sim` duplicates the strategy list of `cli`, `gap.rs` repeats the deal setup of
`match_runner`, the card-bit helper exists twice.

## 7. Endgame solver as a toggle on every strategy (Phase 21)

`endgame-solver` is a strategy of its own, but it is only useful in the endgame: it plays exactly only the last ~11% of
the decisions and otherwise defers to a fallback. Turn that around: an **endgame solver toggle** on every strategy
(Adaptive, endgame-denial, the NEAT champions, the simple ones). The player plays with its chosen strategy as before;
once the hand-off threshold is reached (unseen cards at or below the limit, 6 by default) the solver takes over and
solves the rest of the round. A strategy spec option (for example `adaptive+solver` or `:solver=6`), the same switch in
the GUI, the web play page and `cli train`/`evaluate` pools; the existing `EndgameSolver` fallback becomes the generic
wrapper. Measure with the luck-free head to head: each strategy with and without the toggle, and whether training a
NEAT genome with the solver finishing its rounds changes what it learns.

## 8. Endgame solver, phase B: the early game (Phase 21)

The solver plays only the last ~11% of the decisions exactly, which is why the neat champion still beats it by a wide
margin (+0.63 against -0.08). Phase B: **sample** consistent deals of the unseen cards instead of enumerating them, search
to a **depth** with max^n and a **leaf evaluator** (the NEAT network or the Adaptive heuristics are candidates, which is
itself an experiment), average per move. It reuses `Observation`, `deals` and `search`; the hand-off threshold and the
samples per decision are the numbers to measure (`cli solver-gap` and the luck-free head to head of
`docs/baselines/solver/README.md` are the yardsticks). Also open there: the `use_pass_ceilings` switch to measure how
much dishonest passes hurt (pass ceilings are hard constraints today), and `--pass-rule free`. Gets its own spec first;
the design outline is in `docs/superpowers/specs/2026-10-16-endgame-solver-design.md`.

## 9. Phase 16 — Self-play and co-evolutionary training (idea)

**Status:** not started, on purpose (decision of the overnight run of 2026-10-16): it is a research question that needs
long training runs and a judgement of the results, not a feature to write unattended. The tools to judge it exist (Phase
12 skill scores, Phase 17 run analysis, Phase 18 advisor for the settings, Phase 21 solver as an independent yardstick
for the endgame).

Raised while planning Phase 12; deliberately not built yet. Today genomes train
against a fixed opponent pool, so the opposition never gets stronger than the
pool. Idea: train populations against each other and a hall of fame of their
own past champions, optionally mixed with fixed strategies.

- Relative fitness from the Phase 12 statistics: duplicate-deal skill scores
  (so card luck does not swamp a zero-sum comparison) and the opponent-adjusted
  strength rating (a result against a stronger field counts for more).
- A fixed **anchor benchmark**: every N generations the current best is also
  scored against frozen baselines (`docs/baselines/neat-v2` champion and the
  hand-written strategies), so progress is absolute even though self-play
  fitness is only relative.
- Known risks to design for: cycling (A beats B beats C beats A) and forgetting
  old tricks; the hall of fame and the anchors are the standard guard. The first
  run is an experiment compared against `neat-v2`, not a new default.
- Evidence so far (from the stagnation analysis of the long runs, see Phase 17):
  the stall seen in the long runs looked noise- and information-limited by hand analysis (the Phase 17 tool is more cautious: `docs/baselines/phase-17/README.md`) rather
  than opponent-limited, so this phase ranks below Phase 13's new inputs.

## 10. Phase 20 — GPU simulation port (RTX 4070) (planned, not started)

**Status:** not started, and not startable on the development machine of 2026-10-16: there is no NVIDIA driver or CUDA
toolchain there (the RTX 4070 is on the owner's desktop). The first step is the G0 spike in the schedule below, measured
on that machine; Phase 19's numbers and the Phase 18 advisor's measured rounds per second are the CPU reference to beat.

Go/no-go, schedule and design for running the simulation on the GPU. Motivation
(from the stagnation analysis, Phase 17): the long runs look noise-limited by hand analysis (the Phase 17 tool's verdicts on the same runs are mixed, see `docs/baselines/phase-17/README.md`), and a
10-30x faster simulator would allow many more matches per genome (standard error
down 3-5x) and bigger populations in the same wall time. The measured facts in
`docs/baselines/perf/README.md` still hold: the network is only a few percent of
the time, so only a port of the **whole simulation** (dealing, exchange, legal
moves, the round loop, strategy features, network evaluation) pays off, one
match per GPU thread, and the CPU implementation stays the reference.

**Target hardware and tooling.** NVIDIA RTX 4070 (5,888 CUDA cores, 12 GB).
Recommended stack: a Rust host with CUDA C++ kernels loaded at run time through
`cudarc` (NVRTC, no `nvcc` needed to build the default workspace), behind a cargo
feature `gpu` and a `--backend cpu|gpu` flag (default `cpu`; automatic fall back
when no device is found). A portable `wgpu`/WGSL version is the fallback if CUDA
tooling proves too heavy; it lacks 64-bit integers, which the 52/104-card
bitmasks would need emulating, and is expected to be 1.5-2x slower. Decided at
the end of stage G0.

**Exactness.** The roadmap's acceptance rule (bit-for-bit equal to the CPU
simulator) needs the same random numbers on both sides. The CPU uses `StdRng`
(ChaCha12), costly to port. Plan: add a counter-based generator (Philox4x32)
as an extra `--rng philox` mode of the CPU simulator, implemented identically in
Rust and CUDA; ordinary runs keep `StdRng` and all committed baselines stay
reproducible, and GPU/CPU parity is then tested exactly in Philox mode.

**Schedule.** Working days for one developer working focused with AI assistance;
calendar time is longer when done in the evenings (multiply by about 2). Start
after Phase 13 has fixed the feature set (a new feature set changes stage G3),
but G0 can run earlier because it uses the current 20 features.

| Stage | Content | Days | Gate |
|---|---|---:|---|
| G0 Spike | `lowest-legal` x4 plus one neat player, 4 players, single deck, no exchange: dealing, legal moves, trick loop, network, measured rounds/s on the 4070 against the CPU at 16 threads after Phase 19 | 2-3 | go if >= 8x; conditional at 4-8x (decide with the profile); stop below 4x. Also decides CUDA vs wgpu. |
| G1 Foundations | host/device data layout (hands as bitmasks / per-card counts for the double deck), Philox mode on the CPU, buffer upload of genomes (flattened network), seed scheme equal to `match_seed` | 4-5 | Philox CPU mode reproducible and unit-tested |
| G2 Round engine | deal, forced exchange, canonical legal-move generation, pass rule `final`, role assignment, all table sizes 3-6, both decks and duplicate rules | 6-8 | exact parity: identical finishing orders and role histories on >= 100k seeded matches with scripted strategies |
| G3 Strategies | `lowest-legal`, `greedy-highest`, `random-legal`, `hold-back-pairs`, `card-counter`, `endgame-denial`, neat feature extraction (20 features, `PassTracker` equivalent) and network scoring | 6-8 | exact parity on decisions, reusing the CPU fixtures of `neat_equivalence`; checksum equality of batch outputs |
| G4 Adaptive | `adaptive` modifiers (reading, tempo, bully, deception) used in the training pools | 5-7 | exact parity; optional: skipped if pools can avoid them (documented limitation) |
| G5 Training backend | `cli train --backend gpu`: population fitness, champion candidates and re-evaluation on the GPU, generational loop (speciation, crossover, checkpoints, events, dashboard) unchanged on the CPU | 5-7 | the same run on `cpu` (Philox mode) and `gpu` produces identical event streams for a small configuration; real run >= 8x faster per generation |
| G6 Evaluation and statistics | `cli evaluate`, batch runs, duplicate-deal analysis and useful-pass rollouts on the GPU | 4-5 | parity of the reported statistics |
| G7 Optimisation | occupancy and register pressure, divergence (group matches by strategy mix), launch batching, Nsight profiling | 5-7 | target: >= 10x rounds/s on the two real training setups of Phase 19 |
| G8 Release | `docs/GPU.md`, CPU-only default build and CI unchanged, GPU parity tests marked as optional (need a device), build and run instructions for Windows/Linux CUDA | 3 | CI green without a GPU |

Core path to a useful result (G0-G3 without adaptive plus G5) is about 23-31
working days; everything about 40-51 working days (8-10 weeks full time, or
about 5 months in the evenings). Break-even on time saved alone is poor (a long
run takes about 1.5 h today), so the case rests on what the speed enables: more
matches per genome, larger populations, Phase 16 self-play and Phase 18's
advisor probing many settings quickly.

**Order relative to other phases:** 12 (statistics, in progress) -> 19 (CPU
speed and simplification) -> 13 (NEAT inputs and fitness) -> 17/18 (analysis and
advisor) -> G0 spike with the real Phase 19 numbers -> go/no-go -> 20 in full.

**Risks.** Divergence between the matches of a warp (different games branch on
every decision): mitigated by sorting/grouping matches and by checking the
spike's number before investing; per-thread state size (double deck, 5-6
players, histories for pass reading) causing register spilling: layout work in
G1/G7; feature drift between CPU and GPU when Phase 13 adds features: every
feature has a fixture test in both implementations and a single written
specification table; reproducing `StdRng` instead of Philox if old-seed parity
on the GPU is ever required (a day or two extra, not planned).

## Parked — deferred rule variants

Not scheduled, but recorded so they aren't lost (see `RULES.md`,
"Deferred / Out-of-scope Rules"): bombs/revolution, straights, jokers,
rule-violation auto-Arschloch penalty, ace-as-last-card auto-Arschloch.
Any of these could become their own phase if desired later.
