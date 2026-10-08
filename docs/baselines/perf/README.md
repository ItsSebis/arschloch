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
