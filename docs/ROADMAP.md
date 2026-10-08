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
simulation-statistics overview views below.

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

## Phase 14 — Rule change: a pass ends your part in the trick

From `docs/Notes.md` ("once you pass in a trick you are out and cannot rejoin
the trick"). This is a **rule variant**, not a tweak: today `docs/RULES.md`
says passing is always legal and a player who passed may play again when the
turn comes round to them in the same trick. Changing it affects the engine, the
strategies and every number measured so far, so it is its own phase:

- Make it a selectable rule (`--pass-rule free|final`, default `free` so all
  existing results and genome files stay valid), implemented in
  `engine::trick` (a passed seat is skipped for the rest of the trick and the
  trick ends when only the last player to play remains) and in
  `Round::legal_moves`, with RULES.md updated and the state-machine tests
  extended (including the double deck and 3-6 seats).
- Strategies and hand reading: under `final` a pass is far stronger
  information (the seat can no longer play a beater later), and `PassCeilings`'
  refutation logic becomes unnecessary; the strategies' behaviour under the new
  rule needs checking, not just compiling.
- Re-baseline: run the pre-NEAT baseline and the neat-v1 comparison under
  `final`, retrain (the evolved players were trained under `free` and will not
  be optimal), and record both rule sets side by side. The web play page and
  `cli play` gain the rule as a setup option.

## Idea — GPU (CUDA / ROCm) training

Measured in Phase 12 (details in `docs/baselines/perf/README.md`): the
network is not where training time goes. One move scored costs 46 ns on the
committed champion; a game spends about 54% of its time building the
strategy's view of the table and 30% enumerating legal moves (both before
the Phase 12 speedup), and the network is about 2% of a training table
and 6% even when all four seats are neat players. Offloading only the
network to a GPU (CUDA, ROCm/HIP or a portable layer such as wgpu) can
therefore save at most a few percent, and moving data to the GPU and back for
a handful of 20-number vectors per decision would cost more than that.

A real speedup means porting the simulation itself to the GPU: dealing, legal
move generation, the round state machine, feature extraction and the network,
with fixed-size card and hand representations, no allocation and one match per
GPU thread. That is a separate large project; it also has to reproduce the CPU
simulator exactly (a bit-for-bit comparison on thousands of seeds is the
acceptance test), and it only pays off at populations and match counts far
beyond what the CPU runs need (about 30k rounds/s on a laptop, more on a
desktop). Go / no-go: start it only if training runs that matter take hours
on the best CPU available, and after the cheaper CPU wins listed in the perf
README are exhausted.

## Parked — deferred rule variants

Not scheduled, but recorded so they aren't lost (see `RULES.md`,
"Deferred / Out-of-scope Rules"): bombs/revolution, straights, jokers,
rule-violation auto-Arschloch penalty, ace-as-last-card auto-Arschloch.
Any of these could become their own phase if desired later.
