# Roadmap

Each phase should leave the workspace building, tested, and documented
before the next one starts. Phases are scoped to be reviewable in a single
sitting; a phase that grows beyond that should be split.

## Phase 0 — Foundations

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

## Phase 1 — Round & game state machine

- `engine`: full trick loop (lead, follow/pass, trick resolution, hand
  exhaustion, finishing order), round-end role assignment, and the mandatory
  best/worst card exchange (naive tie-break — just take the top/bottom N
  cards by the total order, no strategic selection yet).
- Scripted integration tests that play a full round move-by-move and
  assert final roles/hands match expectations.

## Phase 2 — Strategies & simulation runner

- `sim`: `Strategy` trait, 2-3 baseline strategies (lowest-legal,
  random-legal, greedy-highest).
- Multi-threaded match runner (`rayon`), `MatchResult`/`Statistics` types,
  `serde_json` output.
- Integration tests asserting statistical invariants over small batches.

## Phase 3 — CLI

- `cli` crate: `clap`-based configuration (player count, deck variant,
  duplicate rule, match count, seat strategies, thread count).
- JSON results file + human-readable summary table on stdout.
- End-to-end smoke tests.

## Phase 4 — Deeper statistics

- Role-sustainment tracking: does a strategy that becomes President tend
  to stay President across repeated matches with role carry-over, or is it
  closer to random?
- Luck-vs-skill signal: outcome variance across repeated matches with
  identical strategy assignments but different shuffles.
- Strategy diversification: pass-when-a-beating-play-exists frequency, and
  at least one strategy that deliberately does this (e.g. sometimes holds
  back a matching set rather than always playing greedily) to compare
  against always-play-if-possible baselines.

## Phase 5 — Smart exchange

- Replace Phase 1's naive top/bottom-N tie-break with strategy-aware card
  selection for `choose_exchange_cards` — e.g. prefer to give away an
  isolated high card over one that's part of a pair/triple the strategy
  would rather keep, when multiple cards tie for "the Nth highest".

## Phase 6 — Strategic context, card counting, and endgame denial

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

## Phase 7 — Adaptive strategy

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

## Phase 8 — Trick-lead tempo

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

## Phase 9 — Lead-order bullying

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

## Phase 10 — NEAT training

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

## Phase 11 — Web interface

**Status:** the interactive part is built: `cli play` (Phase 11, "play against the
models"; see `docs/PLAYING.md`) with a pure `sim::session` game session, a POST-capable
local HTTP layer, model advice and personal records. Still open: the
simulation-statistics overview views below, the run-analysis page (Phase 17)
and the training-advisor form (Phase 18).

- Extends the `web` crate that Phase 10d starts with the NEAT training
  dashboard (HTTP server, embedded single-page app, JSON API), rather
  than building a second frontend. Adds the simulation views: a small
  web frontend to view running simulations and the Phase 4 statistics
  as an overview — role outcomes by strategy, variance/luck indicators,
  strategy comparison charts — and can show evolved players next to the
  hand-written strategies.
- Reads (or live-drives) `sim`'s existing output types — no changes to
  `engine`/`sim`'s public shape should be needed going in.
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

## Phase 12 — Richer scoring and statistics

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

## Phase 13 — NEAT: better play and wider training

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
- Not done (Phase 13): neat inputs for "who is still in this trick", which the new
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

## Phase 16 — Self-play and co-evolutionary training (idea)

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
  the stall seen in the long runs looks noise- and information-limited rather
  than opponent-limited, so this phase ranks below Phase 13's new inputs.

## Phase 17 — Run analysis tool (CLI and web)

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
  Listed under Phase 11's still-open web items.

## Phase 18 — Training advisor: pick the opponents and settings for you

Choosing the opponent pool and settings by hand (population, matches per
genome, rounds, hall of fame, target species, opponents) is trial and error;
the long runs show it matters (see Phase 17). A simple option that picks them.

- **`cli train --auto`** (and a `cli train-advisor` that only prints the
  recommendation): from the table (player count, deck, rules), an optional
  starting genome or model folder, and a time budget (`--time-budget 30m`),
  run a short probe and choose:
  - the **opponent pool**: score the candidates (hand-written strategies, models
    found in a models directory, the hall of fame of earlier runs) against the
    starting genome, drop opponents that are too weak to teach anything (the
    candidate already wins almost every game) or too strong to give signal, and
    pick a mix with usable headroom; report the per-opponent score and standard
    error behind each choice;
  - **matches per genome** from the measured per-match variance, so the standard
    error of a fitness measurement is small enough to see the improvements the
    run is expected to make (using the duplicate-deal skill score where it
    reduces the noise, Phase 12);
  - population size and generations that fit the time budget on this machine
    (measured rounds per second), rounds per match, hall-of-fame size and
    interval, species target and threshold defaults;
  - presets on top: `quick` (a few minutes, a sanity run), `thorough` and
    `overnight`.
- It prints the chosen settings and the reason for each in plain language, writes
  them to the run's config, and they can all be overridden by flags. During the
  run the Phase 17 analysis can stop it early or recommend a change once the
  slope is flat.
- **Web interface:** a "New training" form with the recommended values filled in
  and the reasons next to them, plus the live recommendation panel; extends the
  Phase 11 dashboard.
- Depends on Phase 12 (statistics, skill score), Phase 13 (inputs and fitness
  variants to choose between) and Phase 17 (the analysis).

## Phase 19 — CPU performance and simplification

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

## Phase 20 — GPU simulation port (RTX 4070)

Go/no-go, schedule and design for running the simulation on the GPU. Motivation
(from the stagnation analysis, Phase 17): the long runs are noise-limited, and a
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
