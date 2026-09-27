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
  `sim`'s Phase 2 match-runner responsibility, not `engine`'s. `Round`
  also exposes `play_history` (every combo played so far this round,
  tagged with the seat that played it; passes are omitted, since a pass
  never removes a card from a hand) and `hand_size` (a seat's hand size
  without exposing its contents) — Phase 6's strategic-context
  primitives, used to build `sim::TurnContext`. `Round` also exposes
  `pass_history` (every pass this round, in order, tagged with the seat,
  the combo it declined to beat, and `play_history().len()` at that
  moment) — Phase 7's addition, the raw material `sim::hand_reading`
  reduces into pass ceilings.

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
  context, rng) -> Move`. `legal_moves` (from `engine::Round::legal_moves`)
  already encodes every card a candidate move would use, so no separate
  `hand` parameter is needed; `rng` is threaded through explicitly per
  call (rather than owned by the strategy) so a single `Arc<dyn
  Strategy>` can be shared read-only across parallel matches while
  staying fully deterministic per match seed. `context` is a
  `TurnContext` (Phase 6, "strategic context"), built fresh each turn by
  the match runner: this seat's own hand, every opponent's hand size
  (and whether it's still active), and the exact multiset of unseen
  cards — deterministic, since this is a closed-deck game with no draw
  pile. Phase 7 extends `TurnContext` with `own_pass_ceilings` (this
  seat's own pass ceilings, as read by anyone else) and each opponent's
  `pass_ceilings`, plus `current_combo` (the combo on the table, or
  `None` if this seat must lead). `Strategy::name()` also changes from
  `&'static str` to `&str` in Phase 7, so a configurable strategy's name
  can reflect its actual configuration. Three baseline implementations
  from Phase 2: `LowestLegal`, `RandomLegal`, `GreedyHighest`. A fourth,
  `HoldBackPairs` (Phase 4), plays identically while leading but
  deliberately passes rather than split up a same-rank reserve while
  following, as a diversification comparison point against the three
  always-play baselines. Two more strategies (Phase 6) use `TurnContext`
  directly: `CardCounter` plays a legal combo while some unseen card
  could still beat it, and holds it back once no unseen card can;
  `EndgameDenial` switches to aggressive, control-retaining play
  whenever an active opponent's hand size is low enough to be close to
  finishing, to deny them an easy trick — otherwise it conserves like
  `LowestLegal`.
  `Strategy::choose_exchange_cards` (Phase 5,
  "smart exchange") now exists and is implemented by all strategies;
  `engine::exchange`'s naive top-N tie-break has been
  replaced by `engine::exchange_with_selection`, which lets each low
  seat's strategy choose which cards it gives up, validated by
  `engine` rather than merely tie-broken. `HoldBackPairs` avoids
  splitting a same-rank reserve here the same way it does while
  playing, per `docs/ROADMAP.md`'s Phase 5 entry.
  `LowestLegal`/`GreedyHighest` reproduce the old naive highest-N
  selection; `RandomLegal` now gives up `count` uniformly random cards
  instead — a real behavior change from before Phase 5, when
  `engine::exchange` was strategy-agnostic and applied the same naive
  top-N selection no matter which strategy occupied the seat.
  `engine::rank_groups` is now public.
- `hand_reading` (Phase 7): a pure reduction of a round's play/pass
  history into one `PassCeilings` per seat — the lowest top card that
  seat is known unable to beat, per combo size, derived from its
  unrefuted passes (a pass is dropped if that seat later plays a combo
  of the same or larger size topped above the passed-on card, proof it
  held a beater at the time — a bluff or a forced pass like
  `HoldBackPairs`'s is never mistaken for true information). This is
  what powers `TurnContext`'s pass ceilings; the match runner computes
  it fresh each turn from `Round::play_history`/`pass_history`.
  `Adaptive` (Phase 7): one configurable strategy whose base play
  selection is `LowestLegal` (or `CardCounter`, if its `counting`
  modifier is enabled), with two more independently-toggleable
  modifiers layered on top: endgame denial (either a plain hand-size
  trigger reproducing `EndgameDenial` exactly, or that same trigger
  sharpened by pass-based hand reading to spend the *lowest* card that
  provably locks a close-to-finishing opponent out, rather than a
  blanket `GreedyHighest` push) and deception (occasionally
  bluff-passing on a seat's only beating rank, to plant a false pass
  ceiling in an opponent's hand reading). Every modifier is implemented
  by direct delegation to the existing `CardCounter`/`GreedyHighest`/
  `LowestLegal` strategies rather than duplicating their logic, and a
  fully disabled configuration (`AdaptiveConfig::NONE`) never draws from
  `rng`, making `Adaptive`'s output byte-identical to plain
  `LowestLegal` for the same seed. `AdaptiveConfig` has a
  `Display`/`FromStr` grammar (a comma-separated option list — see
  `docs/BUILDING.md` for the full grammar) so `cli` can parse a
  per-seat spec from the command line and `Strategy::name()` can render
  a configuration back out (e.g. `Adaptive(reading,deception=0.2)`).
- A match runner that drives `engine`'s state machine to completion using
  each seat's `Strategy`.
- Multi-threading via `rayon`: independent matches have no shared mutable
  state, so a batch of simulated matches is a `par_iter` map over match
  seeds/configs, with no locks needed.
- `MatchResult` (per-round role history, strategy tag per seat, per-seat
  pass counts) and `Statistics` (aggregated across a batch): role counts
  by strategy name, a pooled and per-strategy voluntary-pass rate (passes
  submitted while a beating play was also legal — Phase 4, "strategy
  diversification"), per-strategy role-sustainment counts (how often a
  role is still held the very next round — Phase 4, "role-sustainment
  tracking"), and a per-strategy first-round placement variance across
  matches with an identical seating but a different shuffle (Phase 4,
  "luck-vs-skill signal"). Serialized with `serde`/`serde_json` so a
  later web phase can consume the same output format.

### `cli`

Parses run configuration (player count, deck variant, duplicate rule,
number of matches, rounds per match, which strategies occupy which
seats, thread count, and a base seed each match's seed derives from) via
`clap`, calls into `sim::run_batch`/`sim::aggregate`, and writes both a
JSON results file (every `MatchResult` plus the aggregated `Statistics`,
`sim`'s existing types with no new schema) and a human-readable summary
table to stdout.

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
