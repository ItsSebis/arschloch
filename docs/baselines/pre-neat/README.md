# Pre-NEAT baseline

Frozen results for every hand-written strategy as of git tag
`baseline-pre-neat` (commit `e28209e`), recorded before any NEAT work
(see `docs/superpowers/specs/2026-10-08-neat-engine-design.md`). Evolved
players are compared against these numbers and against the live
strategies, which the NEAT work never modifies.

- `run.sh` regenerates `summaries/` (single deck, first-dealt-wins,
  2000 matches x 10 rounds, seed 1000, 3-6 players; ~40 s).
- `summaries/vs-lowest-legal_<spec>_<n>p.txt`: one contender seat
  against `n-1` `lowest-legal` seats.
- `summaries/mixed-field_<n>p.txt`: a mixed field of the strongest
  strategies.
- Output is seeded and independent of `--threads`; re-running `run.sh`
  on the tag must reproduce `summaries/` byte-for-byte apart from the
  header's thread count.

**Caveats found while building Phase 10c.** (1) `CardCounter` plays
identically to `LowestLegal` (0 of 200 all-same tables differ), so the
differences between their rows in `mixed-field_*` are *not* skill. (2)
`run_batch` rotates seats cyclically, which keeps neighbour order, so
mixed-field rows carry a table-position bias (swapping two identical
players' seats swaps their results). The `vs-lowest-legal_*` files, where
one contender faces identical opponents, are the cleaner comparison.

If a baseline strategy ever has to change, create a new tag and a new
`docs/baselines/<name>/` instead of overwriting this one.
