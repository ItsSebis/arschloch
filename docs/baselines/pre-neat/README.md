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

If a baseline strategy ever has to change, create a new tag and a new
`docs/baselines/<name>/` instead of overwriting this one.
