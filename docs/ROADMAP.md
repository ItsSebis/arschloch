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
  fame, tuning and baseline comparison.

## Phase 11 — Web interface

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

## Parked — deferred rule variants

Not scheduled, but recorded so they aren't lost (see `RULES.md`,
"Deferred / Out-of-scope Rules"): bombs/revolution, straights, jokers,
rule-violation auto-Arschloch penalty, ace-as-last-card auto-Arschloch.
Any of these could become their own phase if desired later.
