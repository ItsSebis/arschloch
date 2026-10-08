#!/usr/bin/env bash
# Reproduces the Phase 10e learning-option experiments (see ../experiments.md).
# Usage (from the repository root, after `cargo build --release -p cli`):
#   docs/baselines/neat-v1/experiments/run_experiments.sh [outdir]
# Needs roughly 40 minutes on 8 cores: 5 configurations x 3 seeds, then 2
# more baseline seeds and 5 seeds of the combined configuration, each a
# 60-generation training run followed by a held-out evaluation.
set -euo pipefail
CLI=${CLI:-target/release/cli}
OUT=${1:-/tmp/neat-experiments}
mkdir -p "$OUT"
# Run before the pass rule existed: `free` is what they were measured under.
PASS_RULE=${PASS_RULE:-free}
EXCHANGE_RULE=${EXCHANGE_RULE:-free}
COMMON=(--pass-rule "$PASS_RULE" --exchange-rule "$EXCHANGE_RULE" --population 100 --generations 60 --matches-per-genome 40 --reeval-matches 100 --rounds 6 --quiet)

run() { # name seed extra-args...
  local name=$1 seed=$2; shift 2
  local dir=$OUT/${name}_$seed
  [ -f "$dir/best.json" ] && return
  rm -rf "$dir"
  # Explicit flags: the experiment must not depend on today's defaults.
  "$CLI" train --out "$dir" --seed "$seed" "${COMMON[@]}" "$@" > "$dir.log" 2>&1
  # A battery that includes opponents training never saw, on a seed stream
  # training never uses.
  "$CLI" evaluate --genome "$dir/best.json" --matches 600 --seed 99 --pass-rule "$PASS_RULE" --exchange-rule "$EXCHANGE_RULE" --json "$dir.eval.json" > "$dir.eval.txt"
  echo "done $name seed $seed"
}

for seed in 1 2 3 4 5; do
  run base  "$seed" --champion-candidates 1 --hall-of-fame 0 --weight-power 0.5
  run combo "$seed" --champion-candidates 5 --hall-of-fame 0 --weight-power 0.2
done
for seed in 1 2 3; do
  run topk "$seed" --champion-candidates 5 --hall-of-fame 0 --weight-power 0.5
  run hof  "$seed" --champion-candidates 1 --hall-of-fame 3 --hall-interval 5 --weight-power 0.5
  run wp02 "$seed" --champion-candidates 1 --hall-of-fame 0 --weight-power 0.2
  run wp01 "$seed" --champion-candidates 1 --hall-of-fame 0 --weight-power 0.1
done
python3 "$(dirname "$0")/analyze.py" "$OUT"
