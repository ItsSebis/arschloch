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

## Phase 2 — Strategies & simulation runner (current)

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

## Phase 6 — Web interface

- A small web frontend (technology choice deferred to when this phase
  starts) to view running simulations and the Phase 4 statistics as an
  overview: role outcomes by strategy, variance/luck indicators, strategy
  comparison charts.
- Reads (or live-drives) `sim`'s existing output types — no changes to
  `engine`/`sim`'s public shape should be needed going in.

## Parked — deferred rule variants

Not scheduled, but recorded so they aren't lost (see `RULES.md`,
"Deferred / Out-of-scope Rules"): bombs/revolution, straights, jokers,
rule-violation auto-Arschloch penalty, ace-as-last-card auto-Arschloch.
Any of these could become their own phase if desired later.
