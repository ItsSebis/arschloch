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
