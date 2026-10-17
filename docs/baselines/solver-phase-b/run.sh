#!/usr/bin/env bash
# Reproduces docs/baselines/solver-phase-b/README.md (repository root, after
# `cargo build --release -p cli`). Every row: 4 players, single deck, final pass
# rule, forced exchange, 1200 matches x 5 rounds, seed 7, duplicate skill score.
# Summaries go to $OUT (default /tmp/solver-phase-b). THREADS defaults to 4.
set -u
export LC_ALL=C
CLI=${CLI:-target/release/cli}
OUT=${OUT:-/tmp/solver-phase-b}
THREADS=${THREADS:-4}
C=neat:docs/baselines/neat-v2/champion.json
A=adaptive
mkdir -p "$OUT"
h2h() { # name, then one spec per seat
  local name=$1 start
  shift
  local args=()
  for spec in "$@"; do args+=(--strategy "$spec"); done
  start=$(date +%s)
  "$CLI" --player-count 4 --matches 1200 --rounds 5 --seed 7 --skill-score duplicate \
    --threads "$THREADS" --output "$OUT/$name.json" "${args[@]}" > "$OUT/$name.txt"
  echo "$name $(( $(date +%s) - start )) s"
}
# The sampler alone (no exact endgame) against three adaptive.
for s in 4 8 16 32 64; do h2h sweep-s$s "$A+solver:max-unseen=0,samples=$s" $A $A $A; done
h2h sweep-s16-d2 "$A+solver:max-unseen=0,samples=16,depth=2" $A $A $A
h2h sweep-s16-d3 "$A+solver:max-unseen=0,samples=16,depth=3" $A $A $A
h2h sweep-s16-cards "$A+solver:max-unseen=0,samples=16,leaf=cards" $A $A $A
h2h sweep-s16-d3-cards "$A+solver:max-unseen=0,samples=16,depth=3,leaf=cards" $A $A $A
h2h sweep-s16-neat "$A+solver:max-unseen=0,samples=16,leaf=$C" $A $A $A
# Sampler plus exact endgame (T=6).
h2h solverB-vs-adaptive "$A+solver:samples=16" $A $A $A
# Against three neat-v2 champions.
h2h plain-adaptive-vs-champ $A $C $C $C
h2h solverA-vs-champ "$A+solver" $C $C $C
h2h sampler-vs-champ "$A+solver:max-unseen=0,samples=16" $C $C $C
h2h solverB-vs-champ "$A+solver:samples=16" $C $C $C
for m in soft off; do h2h ceil-$m-vs-champ "$A+solver:samples=16,ceilings=$m" $C $C $C; done
h2h neatleaf-off-vs-champ "$A+solver:samples=16,leaf=$C,ceilings=off" $C $C $C
# The champion with the sampler.
h2h champ-sampler "$C+solver:samples=16" $C $C $C
h2h champ-sampler-neatleaf "$C+solver:samples=16,leaf=$C" $C $C $C
h2h champ-neatleaf-off "$C+solver:samples=16,leaf=$C,ceilings=off" $C $C $C
# The mixed table of docs/baselines/solver/README.md.
h2h mixed-solverB "$A+solver:samples=16" $C $A lowest-legal
h2h mixed-neatleaf-off "$A+solver:samples=16,leaf=$C,ceilings=off" $C $A lowest-legal
# Ceilings against a strategy that passes on purpose.
for m in hard soft off; do
  h2h ceil-$m-vs-hbp "$A+solver:samples=16,ceilings=$m" hold-back-pairs hold-back-pairs hold-back-pairs
done
# Endgame decisions against the exact solver (single-threaded).
"$CLI" solver-gap --player-count 4 --games 1500 --seed 3 \
  --strategy "$A+solver:max-unseen=0,samples=16" --strategy $A \
  --strategy "$A+solver:max-unseen=0,samples=64" --strategy $C > "$OUT/gap.txt"
