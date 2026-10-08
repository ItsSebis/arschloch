#!/usr/bin/env bash
# Compares the committed champion with the hand-written strategies at tables
# of 3-6 players, with the same seeds, match counts and round counts as
# docs/baselines/pre-neat/run.sh, so the results are directly comparable.
# Usage (from the repository root, after `cargo build --release -p cli`):
#   docs/baselines/neat-v2/run.sh [outdir]
# Output is seeded and independent of the thread count.
set -euo pipefail
CLI=${CLI:-target/release/cli}
GENOME=${GENOME:-docs/baselines/neat-v2/champion.json}
OUT=${1:-docs/baselines/neat-v2/summaries}
SEED=1000
MATCHES=2000
ROUNDS=10
THREADS=${THREADS:-4}
# Measured under the rules of the game (a pass ends your part in the trick).
PASS_RULE=${PASS_RULE:-final}
# The exchange rule was introduced after these were recorded; `free` is what they
# were measured under (set EXCHANGE_RULE=forced for the rules of the game).
EXCHANGE_RULE=${EXCHANGE_RULE:-free}
mkdir -p "$OUT"
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

for n in 3 4 5 6; do
  # One champion against n-1 copies of each opponent. (The champion sits in
  # every seat in turn: sim rotates the seating across the batch.)
  for opponent in lowest-legal card-counter endgame-denial "adaptive:reading,tempo,bully"; do
    specs=("neat:$GENOME")
    for ((i = 1; i < n; i++)); do specs+=("$opponent"); done
    args=()
    for spec in "${specs[@]}"; do args+=(--strategy "$spec"); done
    "$CLI" --player-count "$n" --matches "$MATCHES" --rounds "$ROUNDS" \
      --seed "$SEED" --pass-rule "$PASS_RULE" --exchange-rule "$EXCHANGE_RULE" --threads "$THREADS" --output "$TMP" "${args[@]}" \
      > "$OUT/champion-vs-${opponent//[:,=]/_}_${n}p.txt"
  done
  # The same champion on a seed stream training never used, against the
  # whole battery including opponents it never trained against.
  "$CLI" evaluate --genome "$GENOME" --player-count "$n" --matches 600 --seed 99 \
    --pass-rule "$PASS_RULE" --exchange-rule "$EXCHANGE_RULE" --threads "$THREADS" --json "$OUT/evaluate_${n}p.json" > "$OUT/evaluate_${n}p.txt"
done

# The design spec's mixed-field lineup (section 9): the champion and three
# different hand-written strategies at one table, over four independent seed
# bases. (Seat rotation in the simulator keeps neighbour order, so a mixed
# field carries a table-position bias; the champion's margin here is far
# larger than that effect, see docs/baselines/pre-neat/README.md.)
for base in 1000 2000 3000 4000; do
  "$CLI" --player-count 4 --matches "$MATCHES" --rounds "$ROUNDS" \
    --seed "$base" --pass-rule "$PASS_RULE" --exchange-rule "$EXCHANGE_RULE" --threads "$THREADS" --output "$TMP" \
    --strategy "neat:$GENOME" --strategy card-counter --strategy endgame-denial \
    --strategy "adaptive:reading,tempo,bully" > "$OUT/mixed-field_4p_seed${base}.txt"
done
