# Endgame solver phase B (determinized sampler) Implementation Plan

> **Executed 2026-10-17.** The code is authoritative; what changed on the way is in the spec's last section. Cut: the
> GUI/web form controls for the new options (the spec string works there), other table sizes and the double deck.

**Spec:** `docs/superpowers/specs/2026-10-17-solver-phase-b-design.md`. Roadmap: open item 8.

**Goal:** an early-game sampler (PIMC: sample consistent deals, depth-limited max^n, pluggable leaf evaluator) as new
options of the existing solver wrapper, plus the `ceilings` switch, measured with the two yardsticks.

**Constraints:** additive only; `endgame-solver[:OLD OPTIONS]` and `SPEC+solver[:OLD OPTIONS]` byte-identical (no rng
draws, same names); reference batch outputs and `docs/baselines/perf/bench.sh` checksums unchanged; deterministic and
thread-count independent; at most 4 threads (`cargo -j4`, `--threads 4`), one cargo job at a time; never `git add -A`
(another agent writes `docs/baselines/phase-13b/` in this worktree).

## Tasks

1. **Config types** (`sim/src/solver/mod.rs`, `config.rs`)
   - `CeilingMode { Hard, Soft, Off }` (Copy, default Hard) as a new `SolverConfig::ceilings` field; fix the literal
     constructors (`cli/src/solve.rs`, `cli/src/solver_gap.rs`, `gap.rs`, tests) with `..Default::default()`.
   - `SamplerConfig { samples, depth, leaf: LeafSpec, rollouts }` and `LeafSpec { Adaptive, Cards, Neat { path, genome } }`
     (equality by path); `SolverSpec` gains `sampler: Option<SamplerConfig>` (drops `Copy`/`Eq`).
   - `parse_solver_toggle` returns `SolverToggle { solver, sampler }`; one shared option parser for both spec forms
     (`max-unseen`, `budget`, `samples`, `depth`, `leaf`, `ceilings`, `rollouts`; `fallback` only for `endgame-solver`).
     `samples=0` = no sampler; `depth` 1..=4; `leaf=neat:PATH` loads the genome at parse time.
   - Tests: parse round trips, errors, `+solver:8`, old option strings give the old values and `sampler: None`.
2. **Ceilings in the exact path**: `analyze` with `Off` clears the observation's ceilings; with `Soft` retries without
   them on `NoConsistentDeal`. Test: the contradictory-ceilings position solves under `Soft`/`Off`, errors under `Hard`.
3. **Sampling** (`sim/src/solver/sample.rs`): `sample_deal(&Observation, use_ceilings, &mut rng)` as in the spec. Tests
   from the spec (sizes, every card once, ceilings hard/ignored, contradictory -> None, near-uniform marginal without
   ceilings, deterministic).
4. **Leaf evaluators** (`sim/src/solver/leaf.rs`): `LeafEvaluator` trait, `Rollout`, `CardCount`. Rollout builds the
   context with `turn_context_for(round, seat, n, cards, tracker, policy.needs())` and patches `combo_holder` from
   `Round::combo_holder`. Tests: completes from rebuilt positions with a combo on the table and seats out; full place
   set; `CardCount` ranks a seat with fewer cards first.
5. **Sampler search** (`sim/src/solver/sampler.rs`): `sample_analyze(&Observation, &SamplerConfig, leaf, seed) ->
   Result<MoveReport-like, SolveError>`: per sample a deal, `round_for`, per root move depth-limited max^n (`[f32; 6]`),
   mean own place per move, budget in leaf evaluations. Tests: depth large enough to finish = exact search's values on
   small positions; best move legal and in engine order; deterministic per seed; budget stops after >= 1 sample.
6. **Strategy** (`sim/src/strategies/endgame_solver.rs`): `with_sampler(Option<SamplerConfig>)`; decision order exact ->
   sampler -> fallback; one `rng.next_u64()` only when the sampler engages; names per spec; `SolverStats` gains
   `sampled`, `sample_fell_back`, `rollouts`. Tests: `samples=0` move-for-move like before; free rule -> never samples;
   `sim/tests/endgame_solver_batch.rs` gains a sampler batch identical at 1 and 4 threads.
7. **Wiring**: `cli/src/args.rs` (`WithSolver` carries the toggle, `Solver` the spec with sampler), `web/src/play.rs`
   (`resolve` passes the sampler on), cli help text, arg tests.
8. **Gates**: fmt, `cargo test --workspace -j4`, clippy 1.99 as CI; reference outputs: run `bench.sh` checksums and the
   phase A regression run (`endgame-solver`, `adaptive:reading,tempo`, champion, `hold-back-pairs`, 400x3,
   `--skill-score both`) before/after, `cmp` the summaries.
9. **Measurements** (copy the release `cli` to the scratchpad first; 4 threads; record wall time), per the spec's
   protocol; write `docs/baselines/solver-phase-b/README.md` (+ `run.sh`, summaries).
10. **Docs**: `docs/SOLVER.md` (options, limits), ROADMAP item 8 + Phase 21 text, README strategy list, spec revisions
    section, ARCHITECTURE module map.
11. **Commit** (own paths only) with the session's attribution lines.

## Acceptance

- Old specs byte-identical (regression run + bench checksums); new tests green; clippy clean.
- A head-to-head table with numbers and wall time for: solver-B vs 3x adaptive, sampler alone vs 3x adaptive, solver-B
  vs champions, champion+sampler vs champions, ceilings hard/soft/off, a samples/depth sweep; solver-gap rows.
- Negative results stated as such.

## Revision (after re-reading the plan)

- **Cost risk.** Rollout leaves at depth 2 multiply the cost by the reply count (~6-10). Budget is ~2-3 h at 4 threads.
  Decision: measure the cost probe (task 9 first step) before choosing match counts; head to heads at depth 2 and NEAT
  leaves get fewer matches (e.g. 400 = 100 duplicate groups) and say so with their larger standard errors. The NEAT leaf
  may be too slow to measure in a full table; then it gets a solver-gap row and a short table only.
- **Scope cut candidates** (cut first if time runs short, list as open work): `ceilings=soft` in the exact path (keep
  `off`/`hard`), the `cards` static leaf beyond a smoke measurement, GUI/web form controls for the new options (the
  spec string works there already, no new widgets).
- **Determinism trap:** the phase A toggle path must not draw from `rng` when `samples=0`; covered by the existing
  "max-unseen=0 plays like the bare strategy" test plus a new "samples=0 like before" test and the regression run.
- **Rebuilt-round trap:** `turn_context_for` takes the trick holder from `play_history().last()`, empty right after a
  rebuild; the rollout must patch it (test with a combo on the table at the rebuild).
- **Sampling bias:** capacity-proportional assignment with filtered seats is not exactly uniform under ceilings;
  documented, not fixed (exact uniformity would need rejection sampling, which strong ceilings make arbitrarily slow).
- **Memory:** the exact search allocates a 40 MB memo per decision in flight; the sampler does not use it (no memo), so
  4 threads stay small.
- **`--pass-rule free`:** not built (same reason as phase A); documented.
