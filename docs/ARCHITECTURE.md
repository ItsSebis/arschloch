# Architecture

## Workspace layout

A Cargo workspace with one crate per concern, so each stays small,
independently testable, and dependency-light:

```
arschloch/
├── Cargo.toml            # workspace root
├── engine/                # pure game rules, no I/O, no threading
├── sim/                   # simulation runner, strategies, statistics
├── cli/                   # thin binary wiring config -> sim -> output
├── web/                   # placeholder, built in a later roadmap phase
└── docs/
```

Dependency direction is strictly one-way: `cli` and `web` depend on `sim`,
`sim` depends on `engine`. `engine` depends on nothing project-internal.
This means engine correctness can be fully verified with unit tests that
have no notion of threads, randomness, or CLI args.

### `engine`

The rules from `RULES.md`, encoded as types and pure functions:

- `Suit`, `Rank`, `Card` — comparable value types. `Card::compare` (not a
  global `Ord` impl, since ordering depends on the match's
  `DuplicateRule`) encodes rank-then-suit-then-duplicate-tiebreak exactly
  as documented.
- `DeckVariant` (`Single` / `Double`) and `DuplicateRule`
  (`FirstDealtWins` / `LastDealtWins`) — match configuration that affects
  card comparison.
- `Combo` — a validated "N cards of equal rank" play, plus the legality
  check: does a candidate combo beat the current table state (same size,
  strictly higher top card — rank, then suit, then the duplicate-tiebreak
  rule)?
- `Role` and the per-player-count role table + exchange-count table from
  `RULES.md`, as data (e.g. a `const` table indexed by player count),
  not branching logic scattered through the codebase.
- `Round` — a single round's trick-taking state machine (trick loop,
  finishing order, role assignment via `assign_roles`), plus the
  standalone `deal` and `exchange` functions. These are composable
  single-round primitives: `Round` does not loop multiple rounds
  together. Looping them into a full match (deal → exchange using the
  previous round's roles → play a round → assign new roles → repeat) is
  `sim`'s Phase 2 match-runner responsibility, not `engine`'s.

`engine` has no concept of "strategy" or "which move to pick" — it only
knows how to validate and apply a move it's given.
`Round::submit_move` judges a single candidate move (play or pass) a
caller proposes against the current state, and applies it if legal.
`Round::legal_moves()` additionally enumerates every legal move from the
current state (Phase 2), used by `sim`'s strategies to choose among.
Decision-making itself still lives in `sim` — `engine` only reports what's
legal, never which move to prefer.

### `sim`

- `Strategy` trait: `choose_play(&self, legal_moves, duplicate_rule,
  rng) -> Move`. `legal_moves` (from `engine::Round::legal_moves`) already
  encodes every card a candidate move would use, so no separate `hand`
  parameter is needed; `rng` is threaded through explicitly per call
  (rather than owned by the strategy) so a single `Arc<dyn Strategy>` can
  be shared read-only across parallel matches while staying fully
  deterministic per match seed. Three baseline implementations this
  phase: `LowestLegal`, `RandomLegal`, `GreedyHighest`.
  `choose_exchange_cards` doesn't exist yet — `exchange()`'s naive
  top/bottom-N tie-break (Phase 1) is applied directly by the match
  runner; strategy-aware exchange selection is Phase 5's job
  ("smart exchange"), not introduced early as speculative generality.
- A match runner that drives `engine`'s state machine to completion using
  each seat's `Strategy`.
- Multi-threading via `rayon`: independent matches have no shared mutable
  state, so a batch of simulated matches is a `par_iter` map over match
  seeds/configs, with no locks needed.
- `MatchResult` (per-round role history, strategy tag per seat, trick/pass
  counts) and `Statistics` (aggregated across a batch, Phase 2): role
  counts by strategy name, and a pooled voluntary-pass rate (passes
  submitted while a beating play was also legal, summed across every
  seat and match — not yet broken out per strategy). Outcome variance
  (a proxy for luck vs. skill) and a per-strategy diversification signal
  are Phase 4's job, not built yet. Serialized with `serde`/`serde_json`
  so a later web phase can consume the same output format.

### `cli`

Parses run configuration (player count, deck variant, duplicate rule,
number of matches, which strategies occupy which seats, thread count) via
`clap`, calls into `sim`, and writes both a JSON results file and a
human-readable summary table to stdout.

### `web` (future phase)

Not built yet. Reserved so that when the roadmap reaches the web-interface
phase, it's a new crate reading `sim`'s existing `Statistics`/`MatchResult`
JSON output (or driving `sim` live), rather than a refactor of the crates
above.

## Threading model

`rayon`'s data-parallel `par_iter` over a list of independent match
configurations. Each match is simulated start-to-finish on one thread with
no cross-match communication; results are collected and reduced after the
parallel step. This avoids needing any synchronization primitives in
`engine` or in per-match logic — the only shared state is the final
result-reduction step, which uses plain iterator combinators.

## Testing strategy

- `engine`: unit tests per module (card ordering incl. duplicate-tiebreak,
  combo legality, role/exchange tables per player count, full-round
  scripted playthroughs).
- `sim`: integration tests running small batches (e.g. 100 matches) with
  deterministic seeds and asserting statistical invariants (e.g. win counts
  sum to batch size, every finishing order is a valid permutation).
- `cli`: a couple of end-to-end smoke tests invoking the binary with fixed
  args and asserting the output parses as valid JSON.
