# Phase 17: run analysis tool (design)

Approved plan: `~/.claude/plans/modular-toasting-starfish.md` (summarised here so it lives in the repo).

## Goal
`cli analyse-run <events.jsonl | run dir | set dir> [more...]` and a web page answer, for every training run, "where did it stop improving, and why?" from the data the trainer already writes (`events.jsonl`, schema v1, `sim/src/training/events.rs`). One analysis core, two front ends; no new simulation, no change to training.

## Core (`sim/src/training/analysis/`)
Pure functions plus one tolerant loader.

- `load_runs(path) -> Vec<RunData>`: `events.jsonl` file, run dir, or set dir (one `RunData` per run subdirectory, labelled by directory name; a single file is labelled by its file stem or its directory name). Parses lines with `serde_json` into `sim::training::Event`; unparseable lines are counted and skipped (old runs, truncated last line). A resumed run has several `run_start` events: keep the generation events by generation number (a later event for the same generation replaces the earlier one). Empty or event-less input is an error naming the path.
- `analyse(&RunData) -> RunAnalysis` (`serde::Serialize`; this struct is the `--json` and web payload). All numbers are on the champion's fixed-set re-evaluation score (`champion.reeval.mean`, its `std_error`), the same score the stall analysis of `docs/baselines/phase-13/analyse.py` uses.

| field | definition |
|---|---|
| `generations`, `best_generation`, `final_reeval` | counts and `run_end` data when present |
| `smoothed` | trailing mean of `reeval.mean` over the last 10 generations (shorter at the start) |
| `stall_generation` | first generation index from which every smoothed value is >= peak smoothed - mean `reeval.std_error` (rule of `analyse.py`, `stall_generation`) |
| `noise.single_se` | mean of `reeval.std_error` over generations (one evaluation) |
| `noise.change_sd` | standard deviation of the generation-to-generation change of `reeval.mean` |
| `trend.overall`, `trend.last_third` | OLS slope of `reeval.mean` on generation, reported per 100 generations, with a 95% interval (normal quantile 1.96 is acceptable when n >= 30, otherwise a t quantile table) |
| `curse.reeval_minus_heldout` | `run_end.best_reeval.mean - best_heldout.mean` (None without `run_end`) and in units of `heldout.std_error` |
| `curse.train_minus_reeval` | mean over generations of `champion.train_fitness - champion.reeval.mean` |
| `opponents[]` | per opponent: first-third mean, last-third mean, slope per 100 generations with 95% interval, class `rising`/`flat`/`falling` (interval strictly above / contains / strictly below 0) |
| `growth` | hidden nodes and enabled connections at the first and last generation; `without_gain` = nodes grew in the last third while the last-third trend interval contains 0 |
| `population` | share of genomes in the top fitness-histogram bin (mean over the last third), species count (first/last), compatibility threshold (first/last) |
| `racing`, `timings` | totals when the run used them (extra matches share, seconds per stage); optional |
| `verdict` | list of `{code, message}` from the rules below, most important first |

### Verdict rules (every message quotes its numbers)
1. `still_improving`: stall generation >= 95% of the last generation index, or last-third trend interval strictly above 0 -> "still improving at the end: train longer (`--resume`/`--from`)".
2. `noise_limited`: last-third interval contains 0 and (`change_sd` >= 2 x `single_se` or `curse.reeval_minus_heldout` >= 1 held-out standard error) -> "progress is below the noise: more `--matches-per-genome`, the duplicate-deal skill fitness, several seeds and choose the winner on fresh matches".
3. `pool_limited`: overall flat last-third trend while at least one opponent is `rising` and at least one is `flat` -> "the pool no longer teaches against the flat opponents: harder or additional opponents".
4. `model_limited`: flat last-third trend, no opponent rising, not `noise_limited` -> "flat everywhere above the noise: new inputs (`--features hidden`) or a different fitness".
5. `growth_without_gain`: `growth.without_gain` -> "the network keeps growing without gain: stagnation limit / complexity penalty".
6. `converged`: top-bin share >= 0.5 -> "the population has converged: raise mutation rates / species target".
7. `stop_early`: stall generation <= 60% of the last generation index and rule 1 does not apply -> "the last N generations added nothing: stop earlier".
Rules are independent; several can fire. If none fires the verdict is "no clear limit found".

## CLI (`cli/src/analyse_run.rs`)
Follows `cli/src/watch.rs`: clap `AnalyseRunArgs`, `run(args)`, anyhow. Positional paths (several runs allowed, labelled), `--json PATH` (like `evaluate`), text report per run (the table above, then the verdict). Several runs additionally get a side-by-side table.

## Web (`web/`)
`/analysis.html`, `/analysis.js`, `/api/analysis` (payload = `Vec<{label, analysis, series}>`, `series` = per-generation arrays for the charts), served from the dashboard's root (run dir, set dir, or a directory of runs); `?path=` picks a file/dir under the root only (hostile paths rejected, as in the existing tests). Charts: champion curve with smoothed line, stall marker, noise band, curse markers; per-opponent lines; verdict text; run comparison. Vanilla ES modules and the hand-written SVG helpers of `web/assets/lib.js`.

## Catalogue and docs
`Scope::RunAnalysis` in `sim/src/stats_catalog.rs`, one `StatInfo` per table row with `json_path` into the analysis JSON; `docs/STATISTICS.md` regenerated by `cli stats-doc`; section in `docs/TRAINING.md`; ROADMAP Phase 17 marked done; `docs/baselines/phase-17/README.md` records the analysis of `runs/PC/*.jsonl` and the ten-seed Phase 13 runs (real-data check).

## Tests
Unit tests with hand-computed fixtures per row and per verdict rule; a golden check that stall generation and curse gap on the Phase 13 A0 seeds reproduce `analyse.py`; CLI smoke tests on real run dirs (trained in the test, as `train_smoke.rs`) including old-format, resumed, empty and corrupt inputs; routes tests and `node --test` JS tests for the page helpers. No platform-dependent float goldens (CI runs three OSes).

## Out of scope
Live recommendations/early stopping during training (Phase 18), any change to training or defaults, releases.
