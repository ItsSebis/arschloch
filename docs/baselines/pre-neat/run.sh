#!/usr/bin/env bash
# Re-records the pre-NEAT baseline (git tag `baseline-pre-neat`).
# Usage: docs/baselines/pre-neat/run.sh [outdir]   (run from the repo root,
# ideally on a checkout of the tag). Output is the human-readable stdout
# summary of each run; the bulky per-match JSON is discarded.
set -euo pipefail

OUT="${1:-docs/baselines/pre-neat/summaries}"
SEED=1000
MATCHES=2000
ROUNDS=10
THREADS=4
mkdir -p "$OUT"
cargo build --release -p cli >/dev/null
CLI=target/release/cli
TMP="$(mktemp)"; trap 'rm -f "$TMP"' EXIT

# Strategy specs measured against a field of `lowest-legal` seats.
CONTENDERS=(
  random-legal greedy-highest hold-back-pairs card-counter endgame-denial
  adaptive "adaptive:reading,tempo,bully" "adaptive:reading,deception=0.2,tempo,bully"
)
# Mixed-field order (first N used at an N-seat table).
FIELD=(lowest-legal card-counter endgame-denial "adaptive:reading,tempo,bully" hold-back-pairs greedy-highest)

run() { # name player_count spec...
  local name="$1" n="$2"; shift 2
  local args=()
  for s in "$@"; do args+=(--strategy "$s"); done
  "$CLI" --player-count "$n" --matches "$MATCHES" --rounds "$ROUNDS" \
    --seed "$SEED" --threads "$THREADS" --output "$TMP" "${args[@]}" \
    > "$OUT/${name}_${n}p.txt"
}

for n in 3 4 5 6; do
  for c in "${CONTENDERS[@]}"; do
    specs=("$c"); for ((i = 1; i < n; i++)); do specs+=(lowest-legal); done
    run "vs-lowest-legal_${c//[:,=]/_}" "$n" "${specs[@]}"
  done
  run "mixed-field" "$n" "${FIELD[@]:0:$n}"
done
