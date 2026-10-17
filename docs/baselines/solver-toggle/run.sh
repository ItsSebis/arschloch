#!/usr/bin/env bash
# Luck-free head to head of every strategy with the endgame solver toggle against three copies of itself without it
# (docs/baselines/solver/README.md describes the method). Usage, from the repository root after
# `cargo build --release -p cli`:   docs/baselines/solver-toggle/run.sh [OUT_DIR] [THRESHOLD]
# Writes one summary per table to OUT_DIR (default docs/baselines/solver-toggle); results.json files go to a temp dir.
set -euo pipefail
export LC_ALL=C
CLI=${CLI:-target/release/cli}
OUT=${1:-docs/baselines/solver-toggle}
T=${2:-6}
THREADS=${THREADS:-4}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT"
for spec in adaptive endgame-denial hold-back-pairs lowest-legal neat:docs/baselines/neat-v2/champion.json; do
  name=$(basename "${spec#neat:}" .json)
  name=${name/champion/neat-v2-champion}
  start=$(date +%s)
  "$CLI" --player-count 4 --matches 1200 --rounds 5 --seed 7 --skill-score duplicate --threads "$THREADS" \
    --output "$TMP/r.json" \
    --strategy "$spec+solver:$T" --strategy "$spec" --strategy "$spec" --strategy "$spec" \
    | sed '1d' > "$OUT/$name-T$T.txt"
  echo "$name T=$T: $(( $(date +%s) - start )) s"
done
