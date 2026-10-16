# Endgame solver and determinized search (proposed Phase 21)

Status: design approved in conversation on 2026-10-16 (phase A and the learning outputs); phase B is outlined only and gets its own spec once phase A has measurements.

## Goal

Find the best overall strategy to apply in real games, and learn from it. Concretely: a search-based player that, from one seat's point of view, considers every distribution of the unseen cards the other players could hold and picks the move with the best expected finishing position. Two outputs matter equally: (1) the strategy itself (a strong player and a benchmark ceiling for the heuristic and NEAT players), (2) an analysis that explains *why* a move is best, so a human can learn from it.

Scope: **phase A** = exact endgame solver (this spec, in full). **Phase B** = general depth-limited sampler for the early game (outline, later spec).

## Why determinization, and what "best" means here

The game has hidden hands, so it is not chess: there is no single position to run minimax on. For a seat to move we know its own hand, every card played so far, each opponent's hand size and (via `hand_reading`) per-opponent pass ceilings. The unseen cards (`TurnContext::unseen_cards`) are split among the opponents in an unknown way. A *deal* is one such split consistent with the known hand sizes and pass ceilings.

For a fixed deal the game is a perfect-information game that can be solved exactly. The solver does that for every deal (phase A, small endgames) or for a sample of deals (phase B), then picks the move with the best **expected finishing position over the deals**. That is the best move in expectation over what the opponents could hold, not a proof against a specific hidden hand. Two known weaknesses are accepted and documented: *strategy fusion* (each deal is solved as if the searching seat knew the opponents' cards, so moves that only work by luck are slightly overrated) and *no bluffing or information hiding* (the search does not value hiding its own hand). Phase A weakens neither much, because endgames are small.

## Facts from the code (v0.5.0, main at 0bf7562)

- `engine::Round` is `Clone` (Phase 19), with `legal_moves()` and `submit_move(seat, mv)`; `Move` is `Copy`. It also carries `play_history` and `pass_history`, which grow with the game; a search position should not pay for that on every fork (see Components, `SearchState`).
- The pass rule `Final` (default): a seat that passes is out of the trick until it ends. `Free` exists only to reproduce old results. The solver supports both by deferring to `Round` for legality and turn order (it never reimplements trick logic).
- `Strategy::choose_play(legal_moves, duplicate_rule, context, rng)` with `TurnContext { seat, hand, opponents (hand_size, active, pass_ceilings, in_trick), unseen_cards, own_pass_ceilings, current_combo, .. }`, built lazily through `Strategy::needs`. The solver requests `ContextNeeds::ALL`.
- The double deck makes cards repeat (`Card` has a `deal_index`; `Card::compare` and `rank_groups` treat equal rank/suit duplicates by the `DuplicateRule`). Enumerating deals must count *distinguishable* hands, not permutations of identical cards.
- Strategies live in `sim/src/strategies/` and are registered in `sim/src/strategies/mod.rs` and the CLI strategy names (`cli/src/args`).

## Design: phase A, `EndgameSolver`

### Behaviour

`EndgameSolver { fallback: Arc<dyn Strategy>, config: SolverConfig }` implements `Strategy`:

1. `choose_play`: if `unseen_cards.len() <= config.max_unseen` (and the estimated work is below `config.node_budget`), solve; otherwise (or if the budget is exceeded mid-solve) return `fallback.choose_play(...)`. A fallback is always available, so the solver can never fail a move. The budget fallback is counted and reported (see Statistics).
2. `choose_exchange_cards` and `needs`: delegate to the fallback (the solver only changes play; `needs` returns `ALL`).
3. `name()` reflects the configuration, e.g. `endgame-solver(adaptive,T=8,max^n)`.

### Components (all in `sim/src/solver/`, strategy wrapper in `sim/src/strategies/endgame_solver.rs`)

- **`deals.rs` — deal enumeration.** `enumerate_deals(unseen: &[Card], opponents: &[OpponentHand], duplicate_rule) -> impl Iterator<Item = Deal>`. Groups unseen cards by identical `(rank, suit)` multiplicity so that equivalent splits are generated once, and weights each distinct deal by its multiplicity (the number of ordered ways it arises) so the average over deals is the true expectation under a uniform random split. Filters by the opponents' pass ceilings: a deal is dropped if an opponent holds a combo its ceiling says it cannot have. Hand sizes come from `OpponentHand::hand_size`. Provides `count_deals(..)` for the budget estimate without enumerating.
- **`search.rs` — exact search for one deal.** Input: a `Round` rebuilt from the real position with the sampled opponent hands. Output: for each legal move of the acting seat, the vector of final finishing positions (or role scores) of all seats. Algorithm: max^n. Each node's mover picks the move maximizing *its own* component of the outcome vector (ties broken by a fixed, documented rule: prefer the lower total of cards given up, then lexicographic move order, so results are deterministic). Alpha-beta is not valid for n > 2 players in general; shallow pruning (a mover cannot do better than "finish first now") is the only pruning used. A transposition table keyed on `(sorted hands, current combo, trick leader/turn, passed set)` memoizes node values; it is per deal and dropped afterwards (measure first, keep only if it pays).
- **`opponent_model.rs` — how opponents choose.** `OpponentModel::MaxN` (default; opponents are rational for their own finishing position) and `OpponentModel::Policy(Arc<dyn Strategy>)` (every opponent plays a fixed strategy, e.g. `lowest-legal` or `adaptive`; the tree then has branching only on the solver's own seat, so it is far cheaper). Running both on the same positions shows how much the rationality assumption matters, which is itself a lesson. Under `Policy`, the fixed strategy's `choose_play` is called with a `TurnContext` built for that opponent from the sampled deal.
- **`value.rs` — leaf value.** The round is solved to its end, so the value of a leaf is the finishing position. `Outcome::position` (1..=n) is the primary objective; the role score (the repo's normalised +1..-1 scale from Phase 12/13) is the reported value, so results are comparable to everything else in the repo. A later option may fold in the next round's exchange consequences; not in phase A.
- **`SearchState`.** The search does not clone `Round` per node. It uses a small private mutable state (hands, current combo, trick state, finished order) seeded from the real `Round`'s public accessors, with make/unmake moves, and uses `Round` itself only (a) to generate legal moves through the same function the engine uses, so legality can never diverge, and (b) in tests as the reference implementation. If `engine` lacks a public legal-moves-from-parts function, the minimal change is to expose one, covered by the existing legal-move tests. **Decision to confirm in implementation planning:** if a make/unmake state turns out to be more code than it is worth, fall back to cloning `Round` and measure; the interface above does not change.
- **`report.rs` — move analysis.** `analyze(position, config) -> MoveReport`: per legal move, the expected score, the spread over deals, the share of deals where it is the best move, and the deals (as readable hands) where it is best and worst. This is the "why this move" data; the CLI and later the web view render it.

### Config (`SolverConfig`)

`max_unseen` (hand-off threshold `T`, default chosen by measurement, see Open points), `node_budget` (nodes per decision; exceeding it falls back), `opponent_model` (`MaxN` | `Policy(name)`), `use_pass_ceilings` (default true; false makes a pure "all deals" baseline for measuring what hand reading is worth), `fallback` (strategy name, default `adaptive`).

### CLI

- A strategy name `endgame-solver` with options `--solver-max-unseen`, `--solver-opponents max-n|<strategy>`, `--solver-fallback <strategy>`, usable anywhere a strategy name is (simulate, evaluate, play).
- `cli solve`: reads a position (own hand, current combo or "leading", played cards or the unseen set, opponent hand sizes and optional pass ceilings, pass rule, deck) and prints the `MoveReport` as text. Also reads a position from a `cli play` save or an events log if one is already available; otherwise a small documented text format. This command is the learning tool.
- `cli solver-gap`: runs the benchmark below.

### Learning outputs (the part that serves "learn from this")

1. **`cli solve`** as above: for a real game situation, the ranked moves with expected score and the reasons (which opponent holdings make each move win or lose).
2. **Disagreement benchmark (`solver-gap`).** Over N simulated games, in every position where the solver engages, record the move chosen by each comparison strategy (`adaptive`, the NEAT champions, `endgame-denial`, `lowest-legal`) next to the solver's best move, and the expected-score loss of the strategy's move. Output: per strategy the share of endgame decisions that differ and the average cost; plus the most costly disagreements as readable positions. Those positions are the lessons ("when X, play Y, not Z").
3. **Head-to-head.** `EndgameSolver(adaptive)` against plain `adaptive` and against the neat-v2 champion with the duplicate-deal skill score (Phase 12), so luck is removed from the comparison.

### Statistics

Per run: share of decisions where the solver engaged, share that fell back on budget, deals enumerated and nodes per decision (mean, p95, max), time per decision. These go to the summary so a threshold `T` can be chosen from data, not guessed.

### Testing

- Unit: `enumerate_deals` counts and multiplicity weights against brute-force enumeration on tiny double-deck examples; pass-ceiling filtering; determinism of tie-breaks.
- Unit: `search` on hand-built endgames with a known optimum (last two cards, forced wins, a position where passing is correct, a trick the leader cannot lose); max^n against an independent brute-force reference on random tiny positions for 3 and 4 seats, both pass rules, both decks.
- Property: the chosen move's expected score is never lower than any other legal move's on the same position; the solver never returns an illegal move (checked against `Round::submit_move`).
- Consistency: with `OpponentModel::Policy(lowest-legal)` and a position where every deal is identical (no hidden cards), the solver's best move equals a direct simulation of each move against that policy.
- Integration: a full match with the solver as one seat completes, with identical results for identical seeds (determinism, including under the match runner's parallelism); fallback is used above the threshold and above the budget.
- Defaults unchanged: all existing strategies, goldens and bench checksums are untouched (the solver is opt-in; the one engine change, if any, is a pure addition).

### Open points (resolved by measurement in the plan's first task, not by guessing)

- Maximum workable `T` per table size and deck, from the node/time statistics, and whether the transposition table is needed.
- Whether max^n or the `Policy` model is the better default for *winning*, from the head-to-head. The spec default is `MaxN`; the measurement may change the *recommended* setting, not the interface.
- Whether tie-breaking by rank order or by "fewer cards given up" matters for results.

## Design outline: phase B, `DeterminizedSearch` (later)

For positions above the endgame threshold: sample `n_samples` deals weighted by pass-ceiling consistency (reusing `deals.rs`, sampling instead of enumerating), run depth-limited max^n (or `Policy`) on each with the phase A search plus a cutoff, and a leaf evaluator (cards left, ranks held, tempo; the NEAT network or `Adaptive` heuristics are candidates, which is itself an experiment). Average per move across deals. It gets its own spec after phase A, using phase A's measurements (branching factor, nodes per second) to set realistic `n_samples` and depth. It also gets the experiment protocol discipline of Phase 10/13 (frozen comparison against `docs/baselines`, measured result before any default changes).

## Non-goals

No changes to the rules, to NEAT training, or to existing strategies' behaviour. No GPU port (Phase 20 is a separate idea). No claim of game-theoretic optimality: the result is "best in expectation over consistent deals under the stated opponent model".

## Verification summary

Brute-force cross-checks of enumeration and search on tiny positions; legality delegated to the engine; determinism under the match runner; fallback paths tested; measured thresholds and a documented head-to-head and disagreement benchmark before the phase is called done; CI on three OSes; docs updated (`docs/ROADMAP.md` entry for this phase, `docs/ARCHITECTURE.md` section, `cli solve` usage in `docs/PLAYING.md` or a new short doc).

## Revisions after reading the code for the plan (2026-10-16)

These replace the matching statements above where they differ.

- **Engine change after all, a pure addition.** `Round` can only be built from a fresh deal, but the solver starts from the middle of a trick. `Round::from_position` (and a crate-private `Trick::from_parts`) rebuild a `Final`-rule round from what a seat observes: hands, who is to move, the combo on the table and its holder, who has passed. No existing behaviour changes. Clone-based search then needs no separate make/unmake state: it clones `Round` per node (measured in the plan, replaced only if too slow).
- **`PassRule::Final` only.** `Free` needs per-trick counters (`passes_since_last_play`, `required_passes`) that `TurnContext` does not carry, so under `Free` the solver always defers to the fallback. `Free` exists only to reproduce old results.
- **Finishing order of seats already out is not observable**, and does not matter: the objective is linear (`role_score = 1 - 2 * place / (n - 1)`), so the seats already out only shift every remaining seat's place by the same constant. The solver counts seats already out and adds that offset.
- **Duplicate cards are not interchangeable.** `Card::compare` orders identical rank+suit cards by `deal_index`, so every unseen card is distinct and the deal enumeration needs no multiplicity weights; each consistent deal counts once.
- **Moves are the engine's canonical moves** (`legal_moves`: per rank and size, only the weakest and strongest sub-combo). The solver searches exactly those, so it can never choose a move the match runner would not offer.
- **Tie-break:** the first best move in the engine's move order (deterministic), not "fewest cards given up".
- **Opponent policies** (`OpponentModel::Policy`) see the round state only: hand, opponents' hand sizes and trick state, and the unseen set derived from the deal; no pass ceilings, and a fixed RNG seed per call, so only deterministic policies are meaningful.
- **`cli solve` takes no pass ceilings in v1**, only the pass state of the current trick; the ceilings come from the strategy path (`TurnContext`).
- **Measurement of `T` happens after the strategy exists** (plan Task 8), not in the first task; the default `max_unseen` starts at 6 until then.
