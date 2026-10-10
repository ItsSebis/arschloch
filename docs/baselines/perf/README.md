# Simulator performance baseline

`bench.sh` times the simulator and prints a checksum of every result next to
each time. A speed change must leave every checksum unchanged (the
simulator's results are bit-for-bit the same); only the seconds may move.
Run it from the repository root after `cargo build --release -p cli`.

Measured on an Intel Core i5-11300H laptop (4 cores / 8 threads), 2026-10-08,
3000 matches x 8 rounds, 4 players, wall-clock seconds:

| workload (1 thread)            | before | after | checksum (both) |
|--------------------------------|-------:|------:|-----------------|
| 4x lowest-legal                |  2.88s | 1.88s | `e686c2a079c1b476` |
| 4x neat (champion)             |  5.23s | 4.31s | `f25c9146e47d417f` |
| mixed (1 neat + 3 classic)     |  3.60s | 2.62s | `ce79c92cd31c0ecc` |
| train, pop 60 x 4 gen (best.json) | 11.06s | 7.87s | `85d7c401da1bb195` |

All cores: 4x lowest-legal 0.68s -> 0.46s, mixed 0.84s -> 0.62s. Training
throughput on all threads rose from about 18-23k to about 30-33k rounds/s
(population 150, 100 matches x 8 rounds).

## Where the time goes

Measured with temporary timers around each step of the match loop (the
machine does not allow `perf` or `gdb -p`), single thread, 4x lowest-legal,
before the change:

| step | share |
|------|------:|
| building the strategy's view of the table (`turn_context_for`) | 54% |
| enumerating legal moves (`Round::legal_moves`) | 30% |
| submitting the move and everything else | ~11% |
| the strategy's own decision | 3% |

The view-building was the cost: the pass ceilings were rebuilt from the whole
round history, and the unseen cards by repeated removal, before every turn.
`PassTracker` (incremental, equal to the old function move by move on random
rounds, see its test) and a per-card bitmask fixed both.

The neural network itself is small: 46 ns per move scored on the committed
champion, 5.9 million scored moves per 3000 matches of four neat players,
so about 0.27 s of 4.31 s (6%) in the worst case, and about 2% in a training
table that has one neat player. Feature extraction around it costs several
times more than the network. See the GPU idea in `docs/ROADMAP.md`.

## What is left (not done)

- `Round::legal_moves` allocates a `Vec` per candidate combo and several
  helper vectors per rank group: the next largest cost.
- `TurnSummary::new` builds `Vec<Vec<Card>>` groups twice per decision.
- `Round::submit_move` clones the hand to validate a play.

## Phase 19 baseline (before any Phase 19 speed change)

Measured 2026-10-10 on the same i5-11300H laptop (4 cores / 8 threads) with
nothing else running, each timing run at least twice, the faster shown. The
CPU governor was `powersave` (intel_pstate, turbo on) and the machine ran
about 1.4x slower than on 2026-10-08 (the table above: 1.88s for 4x
lowest-legal), so compare only numbers measured in the same session. Wall
seconds, 3000 matches x 8 rounds, 4 players unless noted; checksums are the
same as above (and unchanged by the timing instrumentation).

| workload (1 thread unless noted) | seconds | checksum |
|----------------------------------|--------:|----------|
| 4x lowest-legal                  |  2.79 | `e686c2a079c1b476` |
| 4x neat (v1 champion)            |  6.22 | `f25c9146e47d417f` |
| mixed (1 neat + 3 classic)       |  3.89 | `ce79c92cd31c0ecc` |
| all threads: 4x lowest-legal / 4x neat / mixed | 0.66 / 1.36 / 0.86 | same |
| train, pop 60 x 4 gen (best.json) | 11.46 | `85d7c401da1bb195` |
| new: 5 players, double deck, 2 neat (v2 champion) + adaptive:reading,deception=1,tempo,bully + 2 lowest-legal, 1000 matches | 4.63 | `9fe3bd9172cca244` |
| new: 4x card-counter             |  3.25 | `1ded4d66fcea8ac8` |
| new: 4x endgame-denial           |  2.86 | `11aab2a4d00bfd38` |
| new: 4x adaptive:reading,tempo,bully | 2.63 | `2c54b4c39e8d7a04` |

In rounds per second per thread: the 5-player double-deck row plays 8,000
rounds in 4.63s (1.7k rounds/s, two neat seats), 4x card-counter 24,000
rounds in 3.25s (7.4k rounds/s), 4x lowest-legal 24,000 rounds in 2.79s
(8.6k rounds/s).

### After step 4 (engine hot path)

Measured in one session (before and after built and run back to back, same
laptop, same `powersave` governor caveat: compare only within this table).
Checksums are identical on every row and `check12.sh` prints REFERENCE
IDENTICAL. What changed: `Combo` is an inline `[Card; 8]` plus a length
(`Copy`, no heap), `Round::active_mask` and `validate_play` no longer
allocate or clone the hand, and `legal_moves` is one pass over a stack copy of
the hand sorted by `Card::compare` (rank groups are contiguous slices, the
lowest and highest subsets are windows, the lowest beating subset is an index
search), written into a reusable buffer (`Round::legal_moves_into`, used by
`play_out` and the `useful_passes` loop). The old implementation is kept as
the test-only oracle `legal_moves_reference`, compared with the new one on
more than 20,000 random hands and tables, order included.

| workload (1 thread) | before | after | checksum (both) |
|---------------------|-------:|------:|-----------------|
| 4x lowest-legal | 2.86 s | 1.13 s | `e686c2a079c1b476` |
| 4x neat | 6.26 s | 3.83 s | `f25c9146e47d417f` |
| mixed (1 neat + 3 classic) | 3.97 s | 1.97 s | `ce79c92cd31c0ecc` |
| 5p double deck, 2 neat + adaptive + 2 lowest-legal, 1000 matches | 4.62 s | 2.57 s | `9fe3bd9172cca244` |
| 4x card-counter | 3.22 s | 1.40 s | `1ded4d66fcea8ac8` |
| 4x endgame-denial | 2.94 s | 1.16 s | `11aab2a4d00bfd38` |
| 4x adaptive:reading,tempo,bully | 2.67 s | 1.16 s | `2c54b4c39e8d7a04` |
| all threads: lowest-legal / neat / mixed | 0.68 / 1.35 / 0.87 s | 0.27 / 0.87 / 0.46 s | same |
| train, pop 60 x 4 gen | 12.16 s | 5.68 s | `85d7c401da1bb195` |

`Round::legal_moves` in the micro-benchmark (ns per call, the wrapper that
still allocates its one result `Vec`):

| setup | before | after |
|-------|-------:|------:|
| single deck, 4 players, leading | 1499 | 265 |
| single deck, 4 players, following | 968 | 215 |
| double deck, 5 players, leading | 2762 | 539 |
| double deck, 5 players, following | 1380 | 407 |

`TurnSummary::new` is unchanged work (1.1-1.7 us), now the largest single
item of a turn for the classic strategies.

### Training thread scaling (`docs/baselines/perf/scaling.sh`)

`THREADS_LIST="1 2 4 8" docs/baselines/perf/scaling.sh`: population 150, 100
matches per genome, 8 rounds, 4 players, single deck, default hand-written
pool, 6 generations, seed 7. Medians over the 6 generations (second run
within 1% of the first). `best.json` identical for every thread count (OK).

| threads | s / generation | rounds/s | speed-up | training_evaluation | champion_selection | reevaluation_mixed | reevaluation_per_opponent | confirmation | rest |
|--------:|---------------:|---------:|---------:|--------------------:|-------------------:|-------------------:|--------------------------:|-------------:|-----:|
| 1 | 19.28 |  7,074 | 1.00x | 16.93 | 1.08 | 0.26 | 0.68 | 0.47 | < 0.01 |
| 2 | 10.92 | 12,603 | 1.77x |  9.09 | 0.65 | 0.26 | 0.44 | 0.50 | < 0.01 |
| 4 |  6.11 | 22,518 | 3.16x |  4.77 | 0.45 | 0.25 | 0.24 | 0.46 | < 0.01 |
| 8 |  5.33 | 25,816 | 3.62x |  3.95 | 0.34 | 0.27 | 0.25 | 0.53 | < 0.01 |

Stage seconds are medians per generation (`speciation_and_reproduction`,
`hall_of_fame`, `decision_sample`, `checkpoint_and_files` are all below 10 ms;
the pool has no hall of fame by default). Reading the table: the evaluation
of the population scales (4.3x on 4 cores / 8 threads, which is about what
SMT gives), but the stages that are serial or barely parallel (mixed
re-evaluation: 1 task, confirmation: 1 task, per-opponent: 3 tasks, champion
selection: 5 tasks) cost about 1.3 s of the 5.3 s generation at 8 threads
(25%) and do not shrink at all from 4 to 8 threads; at 16 threads they
would be about 40% of the generation. See the Phase 19 notes in
`docs/ROADMAP.md`.

### After step 2 (training-loop parallelism)

Same script and settings, same laptop, measured in a session that ran about
6% slower single-threaded (1 thread: 20.5 s against 19.3 s above), so the
speed-up column is against this table's own 1-thread row. `best.json` is
identical for every thread count and equal to the baseline's checksum run
(`bench.sh` training row `85d7c401da1bb195`, all other rows unchanged).

| threads | s / generation | rounds/s | speed-up | training_evaluation | champion_selection | reevaluation_mixed | reevaluation_per_opponent | confirmation | rest |
|--------:|---------------:|---------:|---------:|--------------------:|-------------------:|-------------------:|--------------------------:|-------------:|-----:|
| 1 | 20.50 |  6,681 | 1.00x | 17.97 | 1.21 | 0.24 | 0.72 | 0.48 | < 0.01 |
| 2 | 10.43 | 13,087 | 1.96x |  9.17 | 0.61 | 0.24 | 0.49 | 0.24 | < 0.01 |
| 4 |  5.27 | 25,648 | 3.89x |  4.67 | 0.31 | 0.25 | 0.25 | 0.12 | < 0.01 |
| 8 |  4.46 | 30,690 | 4.60x |  3.92 | 0.25 | 0.20 | 0.20 | 0.10 | < 0.01 |

Compared with the baseline at 8 threads: 5.33 s -> 4.46 s per generation
(-16%, 25.8k -> 30.7k rounds/s; 4 -> 8 threads now gains 18% instead of 13%).
The stage times of the three re-evaluations (mixed, per-opponent, hall) are
each stage's own wall time while the stages run concurrently, so they overlap
and their sum is larger than their share of the generation. Where the time
is now: the training evaluation is 88% of a generation at 8 threads and
scales like the hardware (4 cores, SMT: about 4.5x); champion selection
(5 candidates, matches in parallel) and the re-evaluations (about 0.2 s,
run concurrently) plus the confirmation (0.1 s, only on a new best, still
after the mixed re-evaluation) remain as the tail that does not shrink with
more threads: 0.55 s of 4.46 s at 8 threads (12%). Each is a short
fork-join with a straggler (the slowest match) at its end; the confirmation
could be started speculatively next to the re-evaluations but would then be
wasted on every generation without a new best. Further serial work:
`TurnSummary`/move generation per match (step 4/5) is per-thread, so it
helps all thread counts.

### Micro-benchmarks

`cargo test --release -p sim --test micro_bench -- --ignored --nocapture
--test-threads=1` prints nanoseconds per call on 24 fixed, seeded states per
setup (deterministic hands; "following" is the next seat after the leader
played its lowest play). Faster of two runs:

| setup | `Round::legal_moves` | moves | `TurnSummary::new` (+ context clone) | new + legal_moves + `features` of every move |
|-------|---------------------:|------:|-------------------------------------:|---------------------------------------------:|
| single deck, 4 players, leading   | 1396 ns | 17.1 | 1561 ns |  5902 ns |
| single deck, 4 players, following |  964 ns | 12.1 | 1537 ns |  4245 ns |
| double deck, 5 players, leading   | 2764 ns | 30.2 | 2485 ns | 11153 ns |
| double deck, 5 players, following | 1339 ns | 16.5 | 2478 ns |  6914 ns |

`TurnSummary::new` costs as much as move generation, and scoring every move
costs several times both.
