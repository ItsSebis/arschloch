#!/usr/bin/env bash
# The Phase 13 experiment matrix (docs/superpowers/specs/2026-10-10-phase-13-neat-design.md,
# section D5). Every arm trains with the neat-v2 recipe (docs/baselines/neat-v2/README.md)
# plus the arm's options, for several seeds, then `cli evaluate` scores every genome on
# held-out matches (a seed stream training never uses). Resumable: finished runs are skipped.
#
# Usage (repository root, after `cargo build --release -p cli`):
#   docs/baselines/phase-13/run.sh train A0 A1 A2     # train the named arms
#   docs/baselines/phase-13/run.sh eval  A0 A1 A2     # evaluate the named arms
#   docs/baselines/phase-13/run.sh all   A0 A1 A2     # both
#   docs/baselines/phase-13/analyse.py [OUT]          # tables and verdicts
#
# Arms (an arm is just the extra `cli train` flags):
#   A0  --features base            control: the 20 original inputs
#   A1  --features trick           + who is still in the trick
#   A2  --features hidden          + hidden-hand prediction (includes trick)
#   B1  racing on $BEST            extra matches for the top genomes (noise-aware selection)
#   B2  risk:0.5 on $BEST          fitness = mean - 0.5 x spread (stable rank)
#   B3  tanh:4 on $BEST            concave fitness (Vize almost as good as President)
#   C2  $BEST, 100 matches/genome  matched-budget control for B1 (B1 costs ~25% more evaluation
#                                  per generation than A0/A2: compare B1 against C2, and also A2)
#   M1  all features, sizes 3-6, both decks (one model for every table)
#   S3, S5D  specialist controls for M1 (3 players single deck, 5 players double deck, base features)
# BEST is the feature set of the better of A0/A2 (default hidden); set BEST=base to stack on A0.
#
# Environment: CLI, OUT (default runs/phase13), SEEDS (default "1 2 3"), THREADS (0 = all),
# GENS/POP/MATCHES/ROUNDS/REEVAL (default: the neat-v2 recipe), EVAL_MATCHES (default 2000),
# EVAL_ROUNDS (default 10), EVAL_SEED (default 99). ARM_MATCHES[arm] overrides --matches-per-genome
# for one arm (default $MATCHES; C2 uses 100).
set -euo pipefail
CLI=${CLI:-target/release/cli}
OUT=${OUT:-runs/phase13}
SEEDS=${SEEDS:-"1 2 3"}
THREADS=${THREADS:-0}
GENS=${GENS:-120}
POP=${POP:-150}
MATCHES=${MATCHES:-80}
ROUNDS=${ROUNDS:-8}
REEVAL=${REEVAL:-200}
EVAL_MATCHES=${EVAL_MATCHES:-2000}
EVAL_ROUNDS=${EVAL_ROUNDS:-10}
EVAL_SEED=${EVAL_SEED:-99}
BEST=${BEST:-hidden}
CHAMPION=${CHAMPION:-docs/baselines/neat-v2/champion.json}
RULES=(--pass-rule final --exchange-rule forced)

declare -A ARM_ARGS=(
  [A0]="--features base"
  [A1]="--features trick"
  [A2]="--features hidden"
  [B1]="--features $BEST --racing-top 0.2 --racing-matches 100"
  [B2]="--features $BEST --fitness risk:0.5"
  [B3]="--features $BEST --fitness tanh:4"
  [C2]="--features $BEST"
  [M1]="--features all --player-counts 3,4,5,6 --decks single,double"
  [S3]="--features base --player-count 3"
  [S5D]="--features base --player-count 5 --deck-variant double"
)

# Per-arm matches per genome (default $MATCHES). C2 matches B1's evaluation budget.
declare -A ARM_MATCHES=(
  [C2]=100
)

train_arm() { # arm seed
  local arm=$1 seed=$2 dir=$OUT/$1/seed$2
  if [ -f "$dir/best.json" ] && grep -q '"type":"run_end"' "$dir/events.jsonl" 2>/dev/null; then
    echo "skip $arm seed $seed (done)"; return
  fi
  rm -rf "$dir"; mkdir -p "$OUT/$arm"
  echo "train $arm seed $seed: ${ARM_ARGS[$arm]}"
  # shellcheck disable=SC2086
  "$CLI" train --out "$dir" --seed "$seed" --population "$POP" --generations "$GENS" \
    --matches-per-genome "${ARM_MATCHES[$arm]:-$MATCHES}" --reeval-matches "$REEVAL" --rounds "$ROUNDS" \
    --champion-candidates 5 --weight-power 0.2 "${RULES[@]}" --threads "$THREADS" --quiet \
    ${ARM_ARGS[$arm]}
}

eval_arm() { # arm
  local arm=$1 files=() seed counts=4 deck=single
  for seed in $SEEDS; do files+=(--genome "$OUT/$arm/seed$seed/best.json"); done
  case $arm in S3) counts=3 ;; S5D) counts=5; deck=double ;; esac
  if [ -f "$OUT/$arm/eval.json" ] && [ -f "$OUT/$arm/h2h.json" ]; then echo "skip eval $arm (done)"; return; fi
  echo "eval $arm"
  local table=(--player-count "$counts" --deck-variant "$deck")
  if [ "$arm" = M1 ]; then table=(--player-counts 3,4,5,6 --decks single,double); fi
  # (1) the frozen default battery (hand-written strategies incl. ones training never saw) and its mixed row
  "$CLI" evaluate "${files[@]}" "${table[@]}" \
    --matches "$EVAL_MATCHES" --rounds "$EVAL_ROUNDS" --seed "$EVAL_SEED" "${RULES[@]}" \
    --threads "$THREADS" --json "$OUT/$arm/eval.json" > "$OUT/$arm/eval.txt"
  # (2) head-to-head against the committed neat-v2 champion (the strongest known opponent)
  "$CLI" evaluate "${files[@]}" "${table[@]}" --opponent "neat:$CHAMPION" \
    --matches "$EVAL_MATCHES" --rounds "$EVAL_ROUNDS" --seed "$EVAL_SEED" "${RULES[@]}" \
    --threads "$THREADS" --json "$OUT/$arm/h2h.json" > "$OUT/$arm/h2h.txt"
}

mode=${1:?usage: run.sh train|eval|all ARM...}
shift
for arm in "$@"; do
  [ -n "${ARM_ARGS[$arm]:-}" ] || { echo "unknown arm $arm" >&2; exit 2; }
  if [ "$mode" = train ] || [ "$mode" = all ]; then
    for seed in $SEEDS; do train_arm "$arm" "$seed"; done
  fi
  if [ "$mode" = eval ] || [ "$mode" = all ]; then eval_arm "$arm"; fi
done
