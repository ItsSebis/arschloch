# Arschloch Multi-threaded Simulator — Design

Date: 2026-09-26
Status: Approved (chat), implementation starting with Phase 0.

## Problem

Build a Rust simulator for the card game Arschloch that can run many
matches in parallel across different AI strategies, and eventually
surface statistics (role sustainment, luck vs. skill, strategy
diversification) through a web interface. Rules are based on the German
Wikipedia article plus explicit house-rule decisions made during
brainstorming (see `docs/RULES.md` for the full, authoritative ruleset —
this document focuses on the *system* design, not the game rules
themselves).

## Goals

- A correct, well-tested rules engine covering 3/4/6-player tables (5 as
  well), single or double deck, with the project's specific suit-ranking
  and exchange-count house rules.
- Multi-threaded batch simulation of many matches with pluggable
  strategies, producing statistics that can answer: how much does role
  outcome depend on strategy vs. luck? How does passing-when-beatable
  (strategic diversification) affect outcomes?
- A codebase kept small and readable on purpose (see
  `docs/CODING_GUIDELINES.md`), so each roadmap phase is a manageable,
  reviewable increment (see `docs/ROADMAP.md`).
- A web interface to view running simulations and statistics, added once
  the engine/statistics foundation is solid (not immediately).

## Non-goals (for now)

- Bombs/revolution, straights, jokers, and other rule variants not
  explicitly requested (recorded in `docs/RULES.md` as deferred).
- Any UI before the roadmap's web-interface phase.
- Networked/interactive human play — this is a simulator for AI
  strategies, not a playable game client.

## Approach considered

1. **Single crate monolith**, split into modules, refactored into a
   workspace later when the web interface is needed. Rejected: the
   engine/sim/cli separation is cheap to set up now and avoids a
   disruptive mid-project refactor; "later" refactors like this tend to
   get deferred indefinitely.
2. **Workspace with `engine`/`sim`/`cli`/`web`-placeholder from day one**
   (chosen). Each crate has one clear responsibility and a one-way
   dependency graph (`cli`/`web` → `sim` → `engine`), so `engine` can be
   fully unit-tested with no notion of threads or I/O, and the eventual
   web crate is additive rather than a rewrite.
3. **Workspace with strategies as their own crate from day one.**
   Rejected for now: with only 2-3 baseline strategies planned for Phase
   2, a dedicated crate is premature; strategies live as a module in `sim`
   and can be extracted later if the set grows large enough to warrant it.

## Design

### Domain model (`engine`)

- `Suit` (`Diamonds < Hearts < Spades < Clubs`), `Rank` (`2..Ace`, poker
  order), `Card { rank, suit, deal_index }` — `deal_index` is only
  consulted when rank and suit are both equal (double-deck duplicates),
  per the configurable `DuplicateRule`.
- `DeckVariant` (`Single` | `Double`), `DuplicateRule` (`FirstDealtWins` |
  `LastDealtWins`) — match-level configuration affecting card comparison.
- `Combo` — N cards of equal rank; legality check compares size and
  top-card ordering against the current table state.
- `Role` — per-player-count enum/table (3/4/5/6 variants) plus the
  exchange-count table, encoded as data (see `RULES.md` for the exact
  numbers, including the inferred-and-flagged 3-player case).
- `Round`/`Game` — state machine: deal → (exchange if round > 1) → trick
  loop → finishing order → role assignment.

`engine` only validates/applies moves and reports legal moves from a
state; it has no notion of "which move to choose."

### Strategy & simulation (`sim`, Phase 2+)

- `Strategy` trait with `choose_play` and `choose_exchange_cards`.
- `rayon`-based `par_iter` over independent match configs — no shared
  mutable state between matches, so no locking is needed.
- `MatchResult` and aggregated `Statistics`, serialized via `serde_json`
  so the future web phase can consume the same format without engine/sim
  changes.

### CLI (Phase 3+)

`clap`-based configuration → `sim` invocation → JSON + human-readable
output.

### Web (Phase 6, deferred)

Placeholder crate only; reads `sim`'s existing output types when built.

## Testing

- `engine`: unit tests per module, including scripted full-round
  playthroughs.
- `sim`: batch integration tests asserting statistical invariants.
- `cli`: smoke tests on fixed args.

See `docs/CODING_GUIDELINES.md` for the concrete lint/format/test gate
each phase must pass (`cargo fmt --check`, `cargo clippy --workspace
--all-targets -- -D warnings`, `cargo test --workspace`).

## Open questions / risks

- The 3-player exchange count (President ↔ Arschloch: 1 card) was
  inferred by extrapolating the pattern in the 4/5/6-player answers, not
  given explicitly. Flagged in `RULES.md`; worth a real-play sanity check
  before trusting 3-player simulation output.
- The double-deck `DuplicateRule` naming (`FirstDealtWins` /
  `LastDealtWins`) is a placeholder; may want friendlier names once it's
  actually implemented and used in CLI flags.

## Rollout

Implemented phase-by-phase per `docs/ROADMAP.md`, starting with Phase 0
(this session): workspace scaffold, `engine` domain types and legal-play
checker, unit tests, and this docs set.
