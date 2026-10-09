# Phase 12 implementation plan

Design: `docs/superpowers/specs/2026-10-09-phase-12-statistics-skill-score-design.md`.

## Execution (the user's standing workflow)
1. Worktree + branch `phase-12-statistics` off `main` in `~/Documents/Code/arschloch` (the user's checkout has uncommitted `results.json`/`.gitignore`: never touch them; `docs/Notes.md` stays theirs). Write and commit the spec and plan docs first.
2. Subagents on matching models, each task TDD with the repo's guidelines (pedantic clippy, tests next to code, `cargo fmt`):
   - sonnet 1: catalogue module + generated `docs/STATISTICS.md` + staleness test + JSON `statistics_catalog` + `--explain` + web route (A).
   - sonnet 2 (parallel, separate files): `extended_stats` (avg rank, mean score, Wilson, strength rating + bootstrap) (B1, B2, B4, B5) and CLI summary/evaluate columns.
   - sonnet 3: `RunOptions`/`run_match_with`/`play_out` refactor + duplicate batches + estimator + `linalg` + comparison report (C); first verify the refactor reproduces the committed baseline summaries byte for byte (`docs/baselines/*/run.sh` style spot checks, and `sim/tests/small_batch.rs`).
   - sonnet 4 (after 3): useful passes with `Round: Clone` and rollouts (B3), then the training `skill_weight` hook.
   - haiku: roadmap/Phase 16, BUILDING/ARCHITECTURE doc edits, mechanical test fixes.
   - opus: full review (determinism/reproducibility of old runs, statistics correctness: SE formulas, BT convergence/identifiability, duplicate-group invariants like zero-sum within a group, estimator leakage, clippy cleanliness) and fixes.
3. Real-case checks (not only unit tests): release-build runs at 3, 4, 5, 6 players, single and double deck, with the hand-written strategies plus `neat-v2/champion.json`: (a) an ordinary run with default flags is byte-identical to the pre-change output apart from the added `extended`/`statistics_catalog` keys; (b) `--skill-score both` on e.g. 4 players x 2000 matches: duplicate skill SE is far below the plain SE (report M and the luck share), estimator vs duplicate comparison printed, timings of off vs estimate vs duplicate vs both recorded in the spec (decides the default); (c) sanity: a strategy against copies of itself has skill ~0 and the zero-sum group invariant holds; the strong strategies outrank random in the strength rating; (d) useful-passes on a sample run gives plausible numbers (hold-back-pairs > lowest-legal, which never passes voluntarily = 0); (e) the dashboard `web` route returns the catalogue.
4. `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`; push the branch and open a PR (this repo's history is PRs #1/#3/#4) instead of pushing main; CI green on all three OSes; mention results in the PR body. Kill nothing lingering (no servers needed except the short dashboard route check).
5. Then: plan and execute Phase 13 (separate plan round; first item likely "who is still in the trick" input, then stable-rank fitness, hidden-hand prediction, one model for several table sizes/decks), all measured against `docs/baselines/neat-v2` with the new statistics.

