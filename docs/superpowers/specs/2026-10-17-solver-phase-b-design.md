# Endgame solver, phase B: a determinized sampler for the early game (Phase 21, open item 8)

Status: written 2026-10-17 for the open-work run (branch `open-work-v0.7`), headless: decisions below are mine and
documented, not confirmed in conversation. Builds on phase A
(`docs/superpowers/specs/2026-10-16-endgame-solver-design.md`, "Revisions" sections) and the `+solver` toggle
(`docs/baselines/solver-toggle/README.md`).

## Goal

The exact solver plays only the last ~11% of a round's decisions. Phase B plays the rest: at a decision with too many
unseen cards to enumerate, **sample** `samples` deals of the unseen cards consistent with what the seat knows, search
each deal to a bounded **depth** with max^n, value the positions at the depth limit with a **leaf evaluator**, and take
the move with the best mean place over the samples ("perfect-information Monte Carlo", PIMC). Measure honestly whether
that beats the plain fallback, `adaptive` and the neat-v2 champion, alone and combined with the exact endgame.

## Facts that shape the design (from phase A and the toggle)

- `Observation::from_context` already holds everything a seat knows (own hand, sizes, who is in the trick, pass
  ceilings, unseen cards, the table); `Observation::round_for(deal)` rebuilds a `Final`-rule `Round` for one deal.
- `deals::for_each_deal` enumerates; for 30+ unseen cards that is hopeless, so sampling is new code (`sample.rs`) that
  applies the same two constraints: per-seat hand sizes and pass ceilings (size-1 ceiling as a card filter while
  assigning, the rank-group check on the finished deal, exactly as `deals.rs` does).
- A rebuilt `Round` has an empty history (it starts at the position). Rollouts from it must therefore build their own
  `TurnContext`: the cards of the round are exactly those in the hands at the rebuild (earlier plays are gone and
  irrelevant), and the trick holder comes from `Round::combo_holder`, not from the play history.
- Cost basis: a plain 4-player round takes about 0.1 ms per thread; a seat has about 11 decisions with a choice per round
  (68,538 decisions in 1500 games x 4 seats, `docs/baselines/solver/README.md`). With rollouts at the leaves, one
  decision costs about `samples x moves x replies^(depth-1)` rollouts of up to a round each.
- `--pass-rule free`: the rebuilt round is `Final`-only (phase A revision) and `TurnContext` does not carry the per-trick
  counters `Free` needs. The sampler has the same limit; under `free` the fallback plays. Documented, not built.

## Design

### Where it plugs in: options on the existing solver, not a new strategy

One wrapper already decides "solver or fallback" per decision (`EndgameSolver`): `endgame-solver[:OPTIONS]` (fallback =
a named heuristic) and `SPEC+solver[:OPTIONS]` (fallback = any spec). Phase B adds options to both, so every place that
takes a spec (cli, evaluate, play, train pools, GUI, web ids) gets it without new plumbing:

| Option | Default | Meaning |
|---|---|---|
| `samples=N` | 0 (off) | sample N deals per early-game decision; 0 keeps phase A behaviour byte for byte |
| `depth=D` | 1 | plies searched with max^n per sampled deal (1 = our move only, then the leaf) |
| `leaf=E` | `adaptive` | leaf evaluator: `adaptive` (roll the round out with `Adaptive` for every seat), `cards` (static: fewer cards and more top cards is better, no rollout), `neat:PATH` (roll out with a NEAT genome for every seat; the experiment) |
| `ceilings=M` | `hard` | pass ceilings when sampling: `hard` (a deal must respect them; none found = fallback), `soft` (respect them, but if no consistent deal is found, sample without them), `off` (ignore them). Applies to the exact endgame too (its enumeration) |
| `rollouts=N` | 20000 | budget: leaf evaluations per decision; a sample that would cross it is abandoned mid-search and sampling stops (the first sample is always completed, so a decision costs at most `max(N, first sample)` leaves) |

Examples: `endgame-solver:samples=16`, `adaptive+solver:samples=16,depth=2`,
`neat:docs/baselines/neat-v2/champion.json+solver:samples=8,leaf=neat:docs/baselines/neat-v2/champion.json`, and the
sampler alone without the exact endgame: `adaptive+solver:max-unseen=0,samples=16`. `+solver:N` keeps meaning
`max-unseen=N`. Order per decision: at most `max-unseen` unseen = exact (phase A, unchanged); otherwise, if `samples>0`,
the sampler; otherwise (or when the sampler cannot answer) the fallback.

Rejected alternative: a separate `+sampler` suffix. Combining it with `+solver` needs suffix ordering rules and a second
wrapper layer whose `needs`/forced-move handling must match the first; options on the one wrapper cost nothing extra.

Names: unchanged without `samples` (`EndgameSolver(adaptive,T=6)`, `X+Solver(T=6)`); with the sampler the configuration
is appended: `EndgameSolver(adaptive,T=6,S=16,D=1,adaptive)`, `X+Solver(T=6,S=16,D=1,adaptive)` (the leaf `neat:PATH`
shows as `neat(<file stem>)`; `ceilings` and `rollouts` show only when not default). `ceilings=off|soft` on a phase-A-only
spec appends `,ceilings=off` likewise.

### Sampling (`sim/src/solver/sample.rs`)

`sample_deal(observation, use_ceilings, rng) -> Option<Vec<Vec<Card>>>`: shuffle the unseen cards, then assign them
most-constrained first (sorted from the highest card down, since size-1 ceilings exclude high cards), each to a seat
drawn with probability proportional to its remaining capacity among the seats whose size-1 ceiling the card does not
beat. Without ceilings this gives exactly the uniform random split; with them it is close to uniform over the consistent
deals but not exactly (documented). A dead end (a card no seat can take) or a deal that fails the rank-group check is
rejected and retried, up to 64 attempts per sample; if none succeed the decision has no consistent deal.

### Search of one sampled deal (`sim/src/solver/sampler.rs`)

Depth-limited max^n on `Round` clones (no memo: at depth 1-3 positions rarely repeat). Values are `[f32; 6]` expected
places; each seat to move picks the move with its own lowest value. Ties (equal value to the seat) are broken by an explicit rule, the same at the root and inside the search: a play beats `Pass` (a pass is chosen only when strictly better), among plays the one with the lower top card wins (strong cards are kept for later), equal strength keeps engine order. Phase A's "first in engine order" would send every tie to `Pass`, which the engine lists first; a probe found 5527 of 13418 pass choices to be pure tie-breaks. The exact solver (`report.best()`) keeps its own rule and its outputs are unchanged. A finished
round is valued by its finishing order; at the depth limit the leaf evaluator values it.

### Leaf evaluators (`sim/src/solver/leaf.rs`), pluggable

`trait LeafEvaluator: Send + Sync { fn places(&self, round: &Round, cards: &[Card], rule, rng) -> [f32; 6] }` where
`cards` is every card in the hands at the rebuild (the "deck" of the rest of the round).

- `Rollout(Arc<dyn Strategy>)`: plays the round to its end with the strategy in every seat (forced moves submitted
  directly) and returns the finishing places. Its `TurnContext` comes from `match_runner::turn_context_for` with
  `cards` as the round deck and a fresh pass tracker per rollout, with `combo_holder` patched from the round (the
  history starts at the rebuild). `leaf=adaptive` is `Rollout(Adaptive::default())`; `leaf=neat:PATH` is
  `Rollout(NeatStrategy)`.
- `CardCount`: static, no rollout. Seats still in are ranked by `cards left - 0.5 x cards that no unseen card of the
  same size can beat` (lower is better), ties shared; seats out keep their places. Cheap, crude, there to show whether
  rollouts are worth their cost and to allow larger depths.

### Determinism and parallel safety

The strategy object is shared between threads and holds no per-decision state (statistics behind the existing mutex).
When the sampler engages it draws **one** `u64` from the decision's `rng`; sample `i` seeds its own `StdRng` from
`mix(seed ^ i)` for the deal and the rollouts of all moves of that sample use the same rollout seed (common random
numbers, so moves are compared on equal luck). Identical seeds give identical matches at any thread count. Phase A
paths draw nothing, so specs without `samples` keep their exact random streams and outputs.

### Statistics

`SolverStats` gains `sampled` (decisions the sampler answered), `sample_fell_back` (no consistent deal or unsupported),
`rollouts` (leaf evaluations). Not printed anywhere today, used by tests and the measurement notes.

## Measurement protocol

All runs: release build, 4 players, single deck, `--pass-rule final`, 4 threads, wall time recorded per run.

1. **Cost probe:** time per decision for samples in {4, 16, 32} x depth {1, 2} x leaf {adaptive, cards} on a short run.
2. **Yardstick 1, `cli solver-gap`** with the sampler alone in the endgame (`adaptive+solver:max-unseen=0,samples=N`):
   places lost per solved decision against the exact solver, compared with plain `adaptive` (0.067). Cheap and exact,
   but it scores the sampler only on endgame positions.
3. **Yardstick 2, luck-free head to head** (`--skill-score duplicate`, `--rounds 5`, seed 7, matches a multiple of 4,
   as many as the budget allows; the phase A tables used 1200):
   - solver-B (`adaptive+solver:samples=...`, exact endgame on) against 3x `adaptive`; the gain over plain adaptive
     compares with the phase A toggle's +0.133.
   - the sampler without the exact endgame (`max-unseen=0`) against 3x `adaptive`.
   - solver-B against the neat-v2 champion: one solver-B seat, three champion seats, and the phase A table's mixed
     table (champion, adaptive, lowest-legal).
   - the champion with the sampler (`neat:...+solver:samples=...`) against three plain champions (does search on top of
     the champion help?).
   - `ceilings=hard` vs `off` vs `soft` for one configuration, to measure how much dishonest passes hurt.
   - modest sweep of samples and depth where budget allows.
4. Report negative results plainly; no default changes (the sampler stays opt-in, `samples=0`).

## Testing

- `sample_deal`: hand sizes respected, every unseen card dealt exactly once, ceilings respected (`hard`), ignored (`off`),
  contradictory ceilings give `None` with `hard`, a deal with `soft`; without ceilings the distribution of one card's
  seat matches the capacity proportions (chi-square-free check: counts within a tolerance on 4000 draws); deterministic
  for a seed.
- Sampler search: depth-limited value equals the exact search's on positions small enough to finish within the depth
  (leaf never reached); the chosen move is always legal; at depth 1 with a rollout leaf the value of each root move is
  that rollout's outcome.
- Rollout: completes from rebuilt positions with a combo on the table and seats already out; returns a full set of places.
- Strategy: same seed, same match result at 1 and 4 threads (via `run_batch`); `samples=0` plays move for move like
  phase A (and `max-unseen=0,samples=0` like the bare strategy, already tested); under `free` the sampler never engages.
- Parsing: new options, errors for bad values, names; `+solver:8` unchanged.
- Regression: reference batch outputs, `docs/baselines/perf/bench.sh` checksums and the phase A `endgame_solver_batch`
  goldens unchanged.

## Non-goals and known weaknesses

- **Exchange holdings are not used.** A seat knows which cards it gave and received in the exchange (they are in a
  specific hand until played), but `Observation`/`TurnContext` do not carry them, so the sampler deals those cards at
  random like any other unseen card. Pinning them needs the exchange record in the context.
- **`ceilings=hard` can fail:** `sample_deal` gives up after 64 attempts (`NoConsistentDeal`), the sample is skipped and
  if every sample fails the fallback plays the decision. `ceilings=soft` switches the ceilings off for the *remaining*
  samples of that decision after the first failed one (so a decision can mix constrained and unconstrained samples).
- The rollout `PassTracker` starts empty at each rebuild: the play-out policy does not know the passes seen before the
  decision (only `combo_holder` is patched in).
- With the sampler on, the wrapped strategy no longer "sees every decision" (phase A's wording): it plays forced moves
  and the fallbacks only, so its random stream differs from a run without the sampler.
- Ties are broken by the rule above, a heuristic: it prefers weak plays over `Pass`; it does not look at what the move
  gives up beyond the top card.

Strategy fusion and no information hiding (as phase A, now larger because positions are far from the end); the
sampling is near-uniform, not exactly uniform, under ceilings; opponents in rollouts all play the same policy; no
`--pass-rule free`; no training of a genome against or with the sampler.

## Revisions while building and measuring (2026-10-17)

- **Cheaper than estimated.** A depth-1 decision with 16 samples and `adaptive` play-outs costs about 1 ms of one
  thread; 1200 matches x 5 rounds with one sampler seat take 41 s at 4 threads, so every table could use the phase A
  size (1200 matches, 300 duplicate groups). Depth 3 is the exception (1106 s).
- **Depth hurts, and the static leaf is useless.** Depth 2 and 3 score lower than depth 1 (plausibly strategy fusion:
  in a sampled deal every opponent sees all hands, so deeper max^n assumes better replies than real opponents make; not
  tested separately); `leaf=cards` plays worse than plain `adaptive`. Recommended: depth 1 with a play-out leaf.
- **The leaf policy decides whether it beats the champion.** With `adaptive` play-outs it does not; with the champion as
  the play-out policy and `ceilings=off` it beats three champions by +0.153 and wins the mixed table.
- **`ceilings` matters against the champion**, which passes on purpose: `off` and `soft` beat `hard` there; against
  `hold-back-pairs`, `hard` is best. The default stays `hard` (no default changes without a decision).
- Everything else as designed; numbers in `docs/baselines/solver-phase-b/README.md`.
