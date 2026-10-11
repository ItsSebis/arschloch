# Phase 17: run analysis tool (design)

Approved plan: `~/.claude/plans/modular-toasting-starfish.md` (summarised here so it lives in the repo).

## Goal
`cli analyse-run <events.jsonl | run dir | set dir> [more...]` and a web page answer, for every training run, "where did it stop improving, and why?" from the data the trainer already writes (`events.jsonl`, schema v1, `sim/src/training/events.rs`). One analysis core, two front ends; no new simulation, no change to training.

## Core (`sim/src/training/analysis/`)
Pure functions plus one tolerant loader.

- `load_runs(path) -> Loaded { runs: Vec<RunData>, skipped_runs: Vec<String> }`: `events.jsonl` file, run dir, or set dir (one `RunData` per run subdirectory or loose `*.jsonl`, labelled by directory name or file stem; an `events.jsonl` is labelled by its directory). Lines are parsed with `serde_json` into a lean analysis-only struct (`lean.rs`: only the fields the analysis reads, everything but the champion's re-evaluation defaulted), so event files from older versions with missing fields still load; unparseable lines are counted and skipped (truncated last line, other record types). A resumed run has several `run_start` events: the generation events are kept by generation number (a later event for the same generation replaces the earlier one), the LAST `run_start` is `start` (its configuration gives the racing `extra_share`, its `resumed_from_generation` marks a resume), `starts` counts them, and a `run_end` followed by a later `run_start` is dropped (`end` is reset on every `run_start`). In a set directory a run that cannot be read is left out and listed in `skipped_runs` (the CLI prints a warning, the web reply carries it); it is an error only when no run remains. Empty or event-less input is an error naming the path. `load_runs_within(path, Some(root))` additionally canonicalises every run directory and events file (symlinks followed), drops those that resolve outside `root`, and writes error text with paths relative to `root` (never absolute) for the web.
- `qualify_labels(&mut runs, force)`: when labels collide, or `force` (the CLI has several path arguments), every label gets the name of its parent directory in front (`A0/seed1`); labels still equal after that are numbered (`(2)`). CLI table, `--json` and web use it.
- `analyse(&RunData) -> RunAnalysis` (`serde::Serialize`; this struct is the `--json` and web payload). All numbers are on the champion's fixed-set re-evaluation score (`champion.reeval.mean`, its `std_error`), the same score the stall analysis of `docs/baselines/phase-13/analyse.py` uses.

| field | definition |
|---|---|
| `generations`, `last_generation`, `best_generation`, `final_reeval` | number of generation events, the NUMBER of the last generation (a run need not start at 0 or be gap free), and `run_end` data when present |
| `smoothed` | trailing mean of `reeval.mean` over the last 10 generations (shorter at the start) |
| `peak_smoothed`, `peak_index`, `peak_generation`, `final_smoothed` | the smoothed maximum (first one), its position in the generations and its generation number; the smoothed value at the last generation |
| `stall_index`, `stall_generation`, `stall_fallback` | first position from which every smoothed value is >= peak smoothed - mean `reeval.std_error` (rule of `analyse.py`, `stall_generation`, whose number equals `stall_index`); `stall_generation` is the generation NUMBER at that position; `stall_fallback` is true when no such position exists (the final smoothed value is itself below the band) and `stall_index` is the last position, as in `analyse.py` |
| `noise.single_se` | mean of `reeval.std_error` over generations (one evaluation) |
| `noise.change_sd` | standard deviation of the generation-to-generation change of `reeval.mean` (needs two changes, so three generations) |
| `trend.overall`, `trend.last_third` | OLS slope of `reeval.mean` on generation number, reported per 100 generations, with a 95% interval: slope +- t * se, t for n - 2 degrees of freedom (table up to 30, then linear interpolation in 1/df between 30, 40, 60, 120 and 1.96 at infinity: no switch to 1.96 at n >= 30), and se multiplied by sqrt((1 + r) / (1 - r)), capped at 5, when the lag-1 autocorrelation r of the residuals is positive (consecutive generations are not independent; the correction is approximate) |
| `curse.reeval_minus_heldout` | `run_end.best_reeval.mean - best_heldout.mean` (None without `run_end`), in units of `heldout.std_error` (`in_heldout_se`) and of `sqrt(se_reeval^2 + se_heldout^2)` (`in_combined_se`, the error of the difference; the one the verdict uses) |
| `curse.train_minus_reeval` | mean over generations of `champion.train_fitness - champion.reeval.mean` |
| `opponents[]` | per opponent: first-third and last-third mean (the same windows as the champion's trend; None when the window has no data), `last_third_slope` and `overall_slope` per 100 generations with 95% intervals, class `rising`/`flat`/`falling` of the LAST-THIRD slope (interval strictly above / contains / strictly below 0), so `pool_limited` is true for what was measured |
| `growth` | hidden nodes and enabled connections at the first and last generation; `third_start_nodes` / `third_end_nodes` = mean hidden-node count of the first / last `third_window` generations of the last third (`third_window` = the third's length div 2, between 1 and 3); `without_gain` = `third_end_nodes > third_start_nodes` while the last-third trend interval contains 0 |
| `population` | share of genomes in the top fitness-histogram bin (mean over the last third), species count (first/last), compatibility threshold (first/last) |
| `racing`, `timings` | totals when the run used them (extra matches share, seconds per stage); optional |
| `verdict` | list of `{code, message}` from the rules below, most important first |

### Verdict rules (every message quotes its numbers)
Definitions: the last-third interval is *flat* if it contains 0, *rising* if strictly above 0, *falling* if strictly below 0. `declining` = last-third interval falling, or `stall_fallback` (the final smoothed value is below peak - single_se).

1. `still_improving`: (the smoothed PEAK is at an index >= 95% of the last index, or the last-third interval is rising) and the run is not declining -> "still improving at the end: train longer (`--resume`/`--from`)". The stall position is not used: a falling curve stalls "at the end" by the fallback of `analyse.py`.
1b. `declining`: -> "the smoothed score peaked at generation G (x), now y: the final champion is not the best one; use the best generation's genome / stop earlier".
2. `noise_limited`: not `still_improving`, last-third interval flat, `single_se > 0`, and (`change_sd > 2 x single_se`, strictly, or the curse gap is >= 2 in units of `sqrt(se_reeval^2 + se_heldout^2)` -- it was ">= 1 held-out standard error", which selection bias makes nearly always true) -> "the last-third trend (x [lo, hi]) is not distinguishable from zero given the noise: more `--matches-per-genome`, the duplicate-deal skill fitness, several seeds and choose the winner on fresh matches".
3. `pool_limited`: not `still_improving`, flat last-third trend while at least one opponent is `rising` and at least one is `flat` over the last third -> "harder or additional opponents".
4. `model_limited`: not `still_improving`, flat last-third trend, no opponent rising, not `noise_limited` -> "no progress left in the last third that the noise does not explain: new inputs (`--features hidden`) or a different fitness".
5. `growth_without_gain`: not `still_improving` and `growth.without_gain` (quotes the two node means) -> "stagnation limit / complexity penalty".
6. `converged`: top-bin share >= 0.5 -> "the population has converged: raise mutation rates / species target".
7. `stop_early`: not `still_improving`, stall position <= 60% of the last position, last-third point estimate not above 0 and `best_generation` not later than the stall generation -> "after generation G the smoothed score stayed within one noise unit (x) of its peak y: the last N generations (up to generation L) added nothing visible; stop earlier" (N and L are generation numbers).
Precedence: `still_improving` shuts out rules 2 to 5 and 7; a declining run is never `still_improving`. Otherwise rules are independent and several can fire. If none fires the verdict is "no clear limit found".

## CLI (`cli/src/analyse_run.rs`)
Follows `cli/src/watch.rs`: clap `AnalyseRunArgs`, `run(args)`, anyhow. Positional paths (several runs allowed; with several paths, or equal names, labelled `parent/name`), `--json PATH` (like `evaluate`), text report per run (the table above, then the verdict). Several runs additionally get a side-by-side table. Runs of a set that could not be read are named in a warning on stderr; errors name the path once; a failed `--json` write says that the report was printed but the file was not written.

## Web (`web/`)
`/analysis.html`, `/analysis.js`, `/api/analysis` (payload = `{runs: [{label, analysis, series}], skipped_runs: [text]}`, `series` = per-generation arrays for the charts incl. `heldout_mean`), served from the dashboard's root (run dir, set dir, or a directory of runs); `?path=` picks a file/dir under the root only (hostile paths rejected, as in the existing tests). Errors: 400 for refused paths, 404 for a missing path or no events, 500 for an I/O error; bodies name paths relative to the root only. Charts: champion curve with smoothed line, stall marker (at the stall generation number), noise band, held-out markers (the winner's curse) from `series.heldout_mean`; per-opponent lines; verdict text; run comparison. Vanilla ES modules and the hand-written SVG helpers of `web/assets/lib.js`.

## Catalogue and docs
`Scope::RunAnalysis` in `sim/src/stats_catalog.rs`, one `StatInfo` per table row with `json_path` into the analysis JSON; `docs/STATISTICS.md` regenerated by `cli stats-doc`; section in `docs/TRAINING.md`; ROADMAP Phase 17 marked done; `docs/baselines/phase-17/README.md` records the analysis of `runs/PC/*.jsonl` and the ten-seed Phase 13 runs (real-data check).

## Tests
Unit tests with hand-computed fixtures per row and per verdict rule; golden checks that stall index/generation and curse gap reproduce `analyse.py` on the Phase 13 A0 seeds (skipped where the runs are not on disk) and on a committed 12-generation excerpt of A0 seed 5 (`sim/src/training/analysis/fixtures/`, always run; numbers computed with the python script), fixtures with generation numbers that do not start at 0 and have gaps, and a check that the real runs that stalled at the end are not `still_improving`; CLI smoke tests on real run dirs (trained in the test, as `train_smoke.rs`) including old-format, resumed, empty and corrupt inputs; routes tests and `node --test` JS tests for the page helpers. No platform-dependent float goldens (CI runs three OSes).

## Out of scope
Live recommendations/early stopping during training (Phase 18), any change to training or defaults, releases.
