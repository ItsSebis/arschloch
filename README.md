# Arschloch Simulator

A multi-threaded simulator for *Arschloch* ("Asshole"), the German
shedding-type card game — play out thousands of matches between simple
AI strategies and see who tends to end up President, and why.

## Quickstart

```bash
cargo build --release
cargo run -p cli -- \
  --player-count 4 \
  --matches 1000 \
  --rounds 10 \
  --strategy lowest-legal \
  --strategy greedy-highest \
  --strategy random-legal \
  --strategy hold-back-pairs \
  --seed 1 \
  --output results.json
```

This simulates 1000 independent matches (10 rounds each, with role
carry-over between rounds within a match) at a 4-player table, one seat
per named strategy, and produces two things:

- `results.json` — every match's full result plus aggregated statistics.
- A human-readable summary table printed to stdout.

See `docs/BUILDING.md` for the full flag reference (deck variant,
duplicate-card rule, thread count, and cross-compiling a Windows `.exe`).

## What you get

### The strategies

- `lowest-legal` — always plays the smallest, lowest-ranked legal combo;
  the most conservative baseline.
- `greedy-highest` — always plays the largest, highest-ranked legal
  combo; the most aggressive baseline.
- `random-legal` — picks uniformly among every legal move, including
  passing when it doesn't have to.
- `hold-back-pairs` — plays like `lowest-legal`, except while following:
  if every beating play would break up cards of a rank it holds more of
  than this play needs, it passes instead, to keep that reserve intact
  for a later lead.

### The stdout summary

- **Role counts by strategy** — across every round simulated, how often
  each strategy ended up in each role (President down to Arschloch). This
  is the headline number: which strategy tends to win?
- **Voluntary pass rate (pooled, then per strategy)** — how often a seat
  passed even though it had a legal play available. A strategy with a
  much higher rate than the always-play baselines is deliberately trading
  away an immediate play for something else (see `hold-back-pairs`
  above); a rate near zero means "always play if it can."
- **President retention** — of the times a strategy became President,
  what fraction stayed President the very next round? This only means
  something for matches with more than one round, since roles only carry
  over round-to-round within a single match. A rate far above chance
  suggests the role — and the winner's-advantage card exchange that comes
  with it — is genuinely self-reinforcing for that strategy; a rate close
  to how often it becomes President at all suggests it's closer to
  random.
- **First-round placement variance** — how much a strategy's very first
  round's outcome (before any role carry-over) swings between matches
  that seated it identically (same seat, same opponents) but shuffled the
  deck differently. Low variance means the shuffle barely matters for
  this strategy (its play style dominates the outcome — a "skill"
  signal); high variance means the shuffle matters a lot (a "luck"
  signal). This needs a big enough batch to repeat a seating at least
  twice — see `docs/BUILDING.md` for the exact threshold — otherwise it
  prints "(not enough data)".

### The JSON file

`results.json` has two top-level keys: `matches` (one entry per simulated
match) and `statistics` (the aggregated numbers behind the stdout table).
Both are `sim`'s own Rust types, serialized as-is — see the doc comments
on `sim::MatchResult` and `sim::Statistics` in `sim/src/` for the exact
field names and meanings.

## Learn more

| Doc | What's in it |
|---|---|
| [`docs/RULES.md`](docs/RULES.md) | The authoritative game rules this simulator implements, including house-rule decisions. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | How the `engine`/`sim`/`cli`/`web` crates fit together. |
| [`docs/BUILDING.md`](docs/BUILDING.md) | Full build, test, and CLI flag reference; cross-compiling a Windows `.exe`. |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | What's built, what's planned, what's explicitly out of scope. |
| [`docs/CODING_GUIDELINES.md`](docs/CODING_GUIDELINES.md) | Conventions for contributing. |
