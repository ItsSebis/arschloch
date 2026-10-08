# Phase 12: warm start, run sets with ETA, speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (the user chose native execution). Steps use checkbox (`- [ ]`) syntax.

**Goal:** (a) start a new run from an earlier run's population, (b) run a *set* of runs and show the time left for the whole set in the terminal and the dashboard, (c) make training faster on the CPU and record, with measurements, why the GPU is not the next step.

**Architecture:** (a) `Population::warm_start` keeps genomes/innovations/species of a restored state but resets generation, best and RNG; `Trainer::new_from` builds on it; `cli train --from RUN`. (b) A `set.json` file in the set directory (`sim::training::set`) records finished runs; a pure `set_eta` function (Rust, mirrored in `web/assets/lib.js`) computes the remaining time; `cli train --runs N` runs `OUT/run-01..N`; the dashboard follows the current run of a set. (c) Measure first, change only what measurements justify, prove results bit-identical after every change.

**Tech Stack:** Rust workspace (engine, sim, neat, cli, web), vanilla JS dashboard, node tests through `cargo test`.

**Spec:** `docs/superpowers/specs/2026-10-08-neat-engine-design.md` (add section 7e as built).

## Interpretation notes (decided, ledgered)

- "ETA ... for the current training of set runs": the per-run ETA already exists in the terminal (`ETA` column) and the dashboard (Generation card). New is the ETA for a *set* of runs, so there is a set concept (`--runs N`).
- "Build on top of earlier generations, else just resume": `--from RUN_DIR` starts a *new* run (new settings allowed) from the final population of an earlier run. If that run cannot be used (feature set differs, no checkpoint) the command fails with a message naming `--resume` as the way to continue the old run unchanged.
- "GPU acceleration": measured share of a game spent in the network is small (see Task 9), so offloading it cannot give a worthwhile speedup; porting the whole simulator to a GPU is a separate large project. This phase delivers the measurements, the CPU speedups they justify, and updates the roadmap with go/no-go numbers. It does not add GPU code.

## Global Constraints

- Same seed and settings give the same run, independent of threads (existing guarantee). Every speed change must leave `cli` output bit-identical (checked by `docs/baselines/perf/bench.sh` checksums and the existing tests).
- Resume equals never stopping, also for sets.
- `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings` (also with `cargo +1.99.0`), `cargo test --workspace` pass; judge by exit status, never through a pipe.
- Event schema changes are additive (`#[serde(default)]`); `SCHEMA_VERSION` stays 1.
- A bad option must leave nothing behind in `--out` (existing rule).
- Windows and macOS must pass CI: no hard-coded `/` in assertions, no unix-only behavior without a Windows branch.

## Review Focus

1. `--from` with a differently sized or missing/corrupt source: clear error, nothing created in `--out`.
2. `--from` source still running or killed mid-run: its last checkpoint is used; never modified.
3. `--runs 1` must behave exactly like today (no `run-01` subdirectory, no `set.json`).
4. Resuming a set killed between runs (checkpoint of run k+1 absent) and mid-run; ETA after resume stays sane (no division by zero, no negative).
5. Dashboard when `set.json` appears/changes while a browser is open: the epoch bump makes the client reload; no stale charts from the previous run.
6. ETA with zero finished runs and zero generations done: shows a dash, not `NaN`/`Infinity`.
7. Speedups: identical results across `--threads 1` and many threads still hold.

---

## Part A: warm start

### Task 1: `Population::warm_start`

**Files:** Modify `neat/src/population.rs`. Test in the same file's `tests` module.

**Interfaces:**
- Produces: `Population::warm_start(state: PopulationState, config: NeatConfig, seed: u64) -> Result<Population, NeatError>`; requires `config.population_size == state.genomes.len()`; result has `generation() == 0`, `best() == None`, genomes/species/tracker/threshold from `state`, RNG `seed_from_u64(seed)`.

- [ ] **Step 1: failing tests** (in `population.rs` tests; use the existing helpers that build a small evolved population — look at how `snapshot_then_restore...` tests create one):

```rust
#[test]
fn a_warm_start_keeps_the_genomes_but_restarts_the_counters() {
    let mut pop = evolved_population(3); // existing helper or build: new + 3 advance steps with fitness
    let state = pop.snapshot();
    let genomes_before: Vec<_> = pop.genomes().to_vec();
    let config = NeatConfig { population_size: genomes_before.len(), ..NeatConfig::default() };
    let warm = Population::warm_start(state, config, 99).unwrap();
    assert_eq!(warm.generation(), 0);
    assert!(warm.best().is_none());
    assert_eq!(warm.genomes(), &genomes_before[..]);
}

#[test]
fn a_warm_start_with_another_population_size_is_refused() {
    let pop = Population::new(3, NeatConfig { population_size: 10, ..NeatConfig::default() }, 1).unwrap();
    let config = NeatConfig { population_size: 11, ..NeatConfig::default() };
    assert!(matches!(Population::warm_start(pop.snapshot(), config, 1), Err(NeatError::InvalidConfig(_))));
}

#[test]
fn a_warm_start_is_deterministic_in_its_seed() {
    // two warm starts from the same state and seed advance identically
}
```

- [ ] **Step 2:** `cargo test -p neat warm_start` → FAIL (no such function).
- [ ] **Step 3: implement** next to `restore`: validate config, size check, same input-count check as `restore`, build with `generation: 0, best: None, fitness: None, rng: Xoshiro256PlusPlus::seed_from_u64(seed)`, `config` the new one, rest from state.
- [ ] **Step 4:** tests pass; whole `neat` suite green.
- [ ] **Step 5:** commit `neat: warm_start a population from a snapshot`.

### Task 2: `Trainer::new_from` and `RunStart.warm_started_from`

**Files:** Modify `sim/src/training/{trainer,events,run_dir}.rs`; tests in `sim/tests/training_run.rs`.

**Interfaces:**
- Consumes: `Population::warm_start`, `RunDir::read_checkpoint`.
- Produces: `Trainer::new_from(config: TrainConfig, opponents: Vec<Opponent>, dir: &Path, source: &Path) -> Result<Self, TrainError>`; the source checkpoint's `population_size` must equal `config.neat.population_size` (else `TrainError::Mismatch` naming both numbers); feature mismatch -> `TrainError::Mismatch` mentioning `--resume`; `RunStart.warm_started_from: Option<String>` (`#[serde(default)]`), `Some("<source dir display>")`.
- The source directory is only read, never written.

- [ ] **Step 1: failing tests** in `sim/tests/training_run.rs` (follow the file's existing small-config helper):
  1. `a_warm_started_run_begins_from_the_sources_final_population`: run source for 3 generations; `new_from` with the same config (new out dir, different seed) and 2 generations; assert generation-0 `fitness.mean` of the new run is greater than generation-0 `fitness.mean` of a cold run with the same config/seed (use a fixed seed pair; if flaky, compare to the source's generation-0 mean instead: warm mean > source gen-0 mean), the new run's `RunStart.warm_started_from` is set, its events start at generation 0, and the source dir's `checkpoint.json` bytes are unchanged.
  2. `a_warm_start_from_a_different_population_size_is_refused_and_leaves_nothing`: `new_from` with a larger population -> `Err(TrainError::Mismatch(_))`, and the target directory does not exist or is empty.
  3. `a_warm_start_from_a_missing_run_is_a_clear_error`.
- [ ] **Step 2:** run them, expect FAIL (no `new_from`).
- [ ] **Step 3: implement.** Validate the source *before* `RunDir::create_new` so a refusal leaves nothing. `Trainer { .. }` same as `new` but `population` from `warm_start`, plus a field `warm_started_from: Option<String>` copied into `RunStart` (and into the checkpoint's data only through RunStart; resumed runs read it from nothing — keep it only in the event).
- [ ] **Step 4:** the three tests plus the whole `sim` suite pass.
- [ ] **Step 5:** commit `sim: start a run from an earlier run's population`.

### Task 3: `cli train --from`

**Files:** Modify `cli/src/{train_args,train,train_output}.rs`; tests in `cli/tests/train_smoke.rs` and the args unit tests.

**Interfaces:**
- `--from RUN_DIR` (`Option<PathBuf>`), conflicts with `--resume` and `--population` (the size comes from the source; the error text says so). Everything else (opponents, matches, seed, ...) is taken from the new command line as for a fresh run. The banner prints `warm start: RUN_DIR (its final population of N genomes)`.

- [ ] **Step 1: failing tests:** (args) `from_conflicts_with_resume_and_population`; (smoke) `a_run_can_build_on_an_earlier_run`: small train A (40 pop, 4 gens), then `train --from A --out B --generations 2 ...` succeeds, `B/events.jsonl` first line has `warm_started_from`, banner contains `warm start`; and `a_bad_from_leaves_nothing_behind` (nonexistent source -> failure, `--out` not created).
- [ ] **Step 2:** FAIL.
- [ ] **Step 3: implement:** in `train.rs` new-run branch: if `from` given, read the source checkpoint to get its population size, set `args.population` accordingly *before* `new_config`/`validate`, call `Trainer::new_from`. Keep the "validate before freezing" order so a bad option leaves nothing behind.
- [ ] **Step 4:** pass, then **real scenario** (record in ledger): train A (pop 100, 20 gens, 40 matches, 6 rounds), evaluate its best; run `--from A` for 10 generations against a harder pool with a new seed; compare generation-0 champion of B to A's final champion with `cli evaluate`; the warm run must start near A's strength, a cold run with the same settings must start near 0. Also kill `--from` run with SIGKILL and `--resume` it.
- [ ] **Step 5:** commit `cli: train --from builds on an earlier run`.

---

## Part B: sets of runs and ETA

### Task 4: `set_eta` and the set file

**Files:** Create `sim/src/training/set.rs`; modify `sim/src/training/mod.rs`; tests inline.

**Interfaces:**
- `pub struct SetFile { pub total_runs: u32, pub finished_secs: Vec<f64>, pub current_run: u32 }` (serde), `SetFile::{write(dir), read(dir) -> Option<SetFile>}` (atomic temp+fsync+rename like the checkpoint; `read` returns `Err` for corrupt files, `Ok(None)` when absent), `pub fn run_dir_name(index: u32) -> String` (`run-01`...).
- `pub fn set_eta(finished_secs: &[f64], total_runs: u32, current_run_eta: Option<f64>, current_run_elapsed: f64) -> Option<f64>`:
  - `None` if `current_run_eta` is `None`.
  - remaining runs after the current = `total_runs - finished.len() - 1` (saturating).
  - per-run time = mean of `finished_secs` if any, else `current_run_elapsed + current_run_eta`.
  - result = `current_run_eta + remaining * per_run`.

- [ ] **Step 1: failing tests:** no finished runs, first run half done (elapsed 100, eta 100, 3 total) -> `Some(100 + 2*200)=500`; two finished (60, 80), third of 5 with eta 30 -> `30 + 2*70 = 170`; last run -> just its own eta; `current_run_eta: None` -> `None`; more finished than total -> no negative (saturates); write/read round trip; corrupt file is an error; absent is `Ok(None)`.
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** PASS.
- [ ] **Step 5:** commit `sim: set file and set_eta for runs in a set`.

### Task 5: `cli train --runs N`

**Files:** Modify `cli/src/{train_args,train,train_output}.rs`; tests `cli/tests/train_smoke.rs`.

**Interfaces:**
- `--runs N` (default 1, range 1..=1000). `N == 1`: exactly today's behavior. `N > 1`: `--out` is the set directory; run k (1-based) lives in `OUT/run-KK` with seed `seed + (k-1)`; `OUT/set.json` is written before each run starts and after each finishes (`finished_secs` gets the run's wall time). `--serve` serves the set directory.
- `--resume` on a set directory: finished runs are skipped, an in-progress run is resumed from its checkpoint, later runs are started fresh. `--resume` conflicts unchanged otherwise; a set resume reads settings from the first run's `config.json` (every run shares them except the seed).
- `--from` with `--runs`: every run warm-starts from the same source.
- Terminal: a line `=== run 2/5 (seed 1) ===` before each run, and in a set the row's ETA cell is followed by `set 0:34:10` (`--:--:--` when unknown); the final summary after the last run prints one line per run (best generation, held-out score) and the set's total time.

- [ ] **Step 1: failing tests** (smoke, small settings): `a_set_runs_every_run_in_its_own_directory` (`--runs 2`: `run-01` and `run-02` each with `best.json`, `set.json` has two finished times, seeds differ so `events.jsonl` first generations differ); `runs_1_is_exactly_a_normal_run` (no `run-01`, no `set.json`); `a_killed_set_resumes_where_it_stopped` (start `--runs 2`, kill after run 1 finished and run 2 began (poll for `run-02/checkpoint.json`), `--resume`, end state has both runs complete and run-01's files unchanged); `a_bad_option_leaves_nothing_behind_in_a_set`; terminal check: stdout contains `run 2/2` and `set `.
- [ ] **Step 2:** FAIL. **Step 3:** implement: a `run_set` driver in `cli/src/train.rs` looping over runs, reusing the single-run function; `TerminalObserver::with_set(SetContext { index, total, finished: Vec<f64> })` computing the cell via `set_eta` from the run's own `eta()` and `elapsed_secs` of the event.
- [ ] **Step 4:** PASS and the whole workspace. Then **real scenario:** `--runs 3` with small settings, read the ETA column, compare the predicted set ETA at the first generation of run 2 with the actual remaining time (record the error in the ledger; expect within ~20%).
- [ ] **Step 5:** commit `cli: train --runs N runs a set with a set ETA`.

### Task 6: dashboard follows a set and shows the set ETA

**Files:** Modify `web/src/{event_index,routes,server,lib}.rs` as needed, `web/assets/{app.js,lib.js,index.html,style.css}`, tests `web/src/routes.rs`, `web/tests/lib.test.mjs`, `cli/tests/dashboard_smoke.rs`.

**Interfaces:**
- If `DIR/set.json` exists, the run shown is `DIR/<run_dir_name(current_run)>`; `App::run_dir()` returns that path (every route that reads run files uses it). When it changes, `EventIndex::set_path` resets the index and bumps the epoch (clients already reload on an epoch change).
- `/api/state` gets `set: null | { total_runs, current_run, finished_secs }`.
- `lib.js`: `export function setEta(finishedSecs, totalRuns, currentRunEta, currentRunElapsed)` identical to the Rust function (same test vectors as Task 4). The Generation card shows `ETA <run> · set <set>` and a "Run k of N" line; the page title/pill is unchanged for single runs.

- [ ] **Step 1: failing tests:** Rust: `state_reports_the_set_and_the_current_run` (fixture dir with `set.json` and `run-02` events); `the_current_run_changing_resets_the_index_and_bumps_the_epoch`; a plain dir still has `set: null`. JS: the Task 4 vectors for `setEta`, and a Node test that `renderSummary`-level helper (extract `summaryKpis(state)` if needed) shows `set` text only when a set is present. Smoke: `cli train --runs 2 --serve` style test using the existing dashboard smoke helpers: `/api/state` shows `set.total_runs == 2`.
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** PASS (`cargo test -p web`, node tests run via cargo). **Step 5:** real scenario with the browser: `cli train --runs 2 --serve ...`, open with the Chrome tools, screenshot the card; verify it switches to run 2 when run 1 ends and the charts reset. Commit `web: follow a set of runs and show its ETA`.

---

## Part C: speed and the GPU question

### Task 7: reproducible benchmark and the time split

**Files:** Create `docs/baselines/perf/bench.sh`, `docs/baselines/perf/README.md`.

- [ ] **Step 1:** `bench.sh` (bash, `LC_ALL=C`, `set -euo pipefail`) times, single-threaded and with all threads, (a) 3000 matches x 8 rounds of 4x `lowest-legal`, (b) the same with 4x `neat:champion.json`, (c) a mixed table (1 neat + 3 hand-written), (d) a short `train` run (pop 60, 4 generations, 40 matches, 6 rounds, seed 7) printing the final `rounds/s`; it also prints a checksum (`sha256sum`) of the `--json` output of (a)-(c) so any change in results is visible. The first run's checksums are committed in the README as the reference.
- [ ] **Step 2:** run it, record the numbers (machine, date) in the README: this is the baseline. Also record the split: network cost versus the rest, using (a) vs (b) and a throwaway measurement of the network alone (microbenchmark inside a `#[ignore]` test: 1e6 `Network::score` calls on the champion). The throwaway code is not committed.
- [ ] **Step 3:** commit `perf: benchmark script and baseline`.

### Task 8: CPU hot-path improvements the measurements justify

**Files:** whatever Task 7 points at; likely `sim/src/strategies/neat_player/{mod,features}.rs`, `sim/src/match_runner.rs` (`turn_context_for`), `engine/src/legal_moves.rs`.

- [ ] **Step 1:** use `perf` if permitted, otherwise add temporary counters/timers in a scratch copy (never committed) to rank the costs: legal-move generation, `turn_context_for` (unseen cards, pass ceilings), `TurnSummary::new` allocations, per-candidate features, network. Ledger the ranking.
- [ ] **Step 2 (per optimization, each its own commit):** change one thing (typical candidates: avoid `Vec<Vec<Card>>` in `rank_groups` use inside `TurnSummary`, reuse the activation scratch buffer across candidates, avoid cloning `opponents`/`Vec` per turn, `SmallVec`-style fixed arrays for per-rank counts), run `cargo test --workspace` (exit status), run `bench.sh` and require **identical checksums** and a faster time. A change that does not measurably help is reverted, not kept.
- [ ] **Step 3:** record the new numbers in the perf README and `docs/TRAINING.md`'s cost paragraph (keep claims to what was measured).

### Task 9: GPU findings in the roadmap and docs

**Files:** Modify `docs/ROADMAP.md` (the GPU idea section), `docs/TRAINING.md` (one paragraph), spec section 7e.

- [ ] **Step 1:** replace the GPU idea's speculation with the measured numbers from Task 7 (share of time in the network, best-case speedup by Amdahl's law for offloading it, why transfer latency makes the real result worse) and a go/no-go list for a full simulator port (what must be ported: deal, legal moves, round state machine, features, network; expected gain only at populations far above 150 x 100 matches; correctness guard: bit-for-bit comparison with the CPU simulator on thousands of seeds).
- [ ] **Step 2:** write spec section 7e covering warm start, sets and the ETA formula, and the performance work. Update `docs/TRAINING.md` (`--from`, `--runs`, set ETA, how a set resume works) and add `--from`/`--runs` rows to the settings table. Verify every command in the doc by running it.
- [ ] **Step 3:** commit `docs: warm start, sets, speed measurements, GPU findings`.

### Task 10: whole-phase verification and review

- [ ] `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings` (stable and `+1.99.0`), `cargo test --workspace --no-fail-fast`, all by exit status.
- [ ] Real run on this machine: `--runs 3 --from` of an earlier run with `--serve`, watched in the browser; record outputs in the ledger.
- [ ] Fresh opus reviewer on the whole branch with the Review Focus list; one fix pass, each fix RED then GREEN; minors to the ledger.
- [ ] Push the branch, open a PR, wait for CI on all three systems and fix what it finds.
