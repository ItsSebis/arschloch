#!/usr/bin/env bash
# Compares the committed champion with the hand-written strategies at tables
# of 3-6 players, with the same seeds, match counts and round counts as
# docs/baselines/pre-neat/run.sh, so the results are directly comparable.
# Usage (from the repository root, after `cargo build --release -p cli`):
#   docs/baselines/neat-v1/run.sh [outdir]
# Output is seeded and independent of the thread count.
set -euo pipefail
CLI=${CLI:-target/release/cli}
GENOME=${GENOME:-docs/baselines/neat-v1/champion.json}
OUT=${1:-docs/baselines/neat-v1/summaries}
SEED=1000
MATCHES=2000
ROUNDS=10
THREADS=${THREADS:-4}
mkdir -p "$OUT"
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

for n in 3 4 5 6; do
  # One champion against n-1 copies of each opponent. (The champion sits in
  # every seat in turn: sim rotates the seating across the batch.)
  for opponent in lowest-legal endgame-denial "adaptive:reading,tempo,bully"; do
    specs=("neat:$GENOME")
    for ((i = 1; i < n; i++)); do specs+=("$opponent"); done
    args=()
    for spec in "${specs[@]}"; do args+=(--strategy "$spec"); done
    "$CLI" --player-count "$n" --matches "$MATCHES" --rounds "$ROUNDS" \
      --seed "$SEED" --threads "$THREADS" --output "$TMP" "${args[@]}" \
      > "$OUT/champion-vs-${opponent//[:,=]/_}_${n}p.txt"
  done
  # The same champion on a seed stream training never used, against the
  # whole battery including opponents it never trained against.
  "$CLI" evaluate --genome "$GENOME" --player-count "$n" --matches 600 --seed 99 \
    --threads "$THREADS" --json "$OUT/evaluate_${n}p.json" > "$OUT/evaluate_${n}p.txt"
done
