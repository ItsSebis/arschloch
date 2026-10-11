#!/usr/bin/env bash
# Times the simulator and prints a checksum of every result, so a speed
# change that also changes a result is caught. Usage (repository root, after
# `cargo build --release -p cli`):
#   docs/baselines/perf/bench.sh
# The statistics sections appended by Phase 12 (from the line 'Average place (1 = best)')
# are not part of the checksum, so the checksums of the committed baselines stay valid.
# Times are wall-clock seconds of the single-threaded run (THREADS=1 default)
# and of all cores; checksums must stay identical across code changes.
set -euo pipefail
export LC_ALL=C
CLI=${CLI:-target/release/cli}
GENOME=${GENOME:-docs/baselines/neat-v1/champion.json}
THREADS=${THREADS:-1}
MATCHES=${MATCHES:-3000}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

timed() { # label, then the command; prints "label seconds checksum"
  local label=$1
  shift
  local start end
  start=$(date +%s.%N)
  "$@" > "$TMP/out.txt"
  end=$(date +%s.%N)
  printf '%-28s %7.2fs  %s\n' "$label" "$(echo "$end - $start" | bc -l)" \
    "$(tail -n +2 "$TMP/out.txt" | sed '/^Average place (1 = best)/,$d' | sed '$d' | sha256sum | cut -c1-16)"
}

table() { # threads, strategy specs...
  local threads=$1
  shift
  local args=()
  for spec in "$@"; do args+=(--strategy "$spec"); done
  "$CLI" --player-count 4 --matches "$MATCHES" --rounds 8 --seed 42 --pass-rule "${PASS_RULE:-free}" --exchange-rule "${EXCHANGE_RULE:-free}" --output "$TMP/result.json" \
    --threads "$threads" "${args[@]}"
}

N="neat:$GENOME"
for threads in "$THREADS" 0; do
  echo "== threads: $([ "$threads" = 0 ] && echo all || echo "$threads")"
  timed "4x lowest-legal" table "$threads" lowest-legal lowest-legal lowest-legal lowest-legal
  timed "4x neat" table "$threads" "$N" "$N" "$N" "$N"
  timed "mixed (1 neat + 3 classic)" table "$threads" "$N" lowest-legal endgame-denial adaptive:reading,tempo,bully
done

# Rows added in Phase 19 (the rows above are unchanged). Single thread, same
# checksum scheme. The first block is the setup of the slowest real training
# runs (5 players, double deck: 2 neat, 1 adaptive, 2 lowest-legal seats); it plays
# MATCHES5 matches (default 1000: a 5-player double-deck match is several
# times dearer than the 4-player ones above). The second block times the
# heavier classic strategies on their own (4 players, MATCHES matches).
MATCHES5=${MATCHES5:-1000}
N2=${GENOME2:-docs/baselines/neat-v2/champion.json}
echo "== phase 19 rows (threads: 1; 5 players double deck: $MATCHES5 matches)"
table5() {
  local args=()
  for spec in "$@"; do args+=(--strategy "$spec"); done
  "$CLI" --player-count 5 --deck-variant double --matches "$MATCHES5" --rounds 8 --seed 42 --pass-rule "${PASS_RULE:-free}" --exchange-rule "${EXCHANGE_RULE:-free}" --output "$TMP/result.json" \
    --threads 1 "${args[@]}"
}
timed "5p double 2 neat+adapt+2 LL" table5 "neat:$N2" "neat:$N2" "adaptive:reading,deception=1,tempo,bully" lowest-legal lowest-legal
timed "4x card-counter" table 1 card-counter card-counter card-counter card-counter
timed "4x endgame-denial" table 1 endgame-denial endgame-denial endgame-denial endgame-denial
timed "4x adaptive:reading,tempo,bully" table 1 adaptive:reading,tempo,bully adaptive:reading,tempo,bully adaptive:reading,tempo,bully adaptive:reading,tempo,bully

echo "== training (pop 60, 4 generations, 40 matches, 6 rounds, seed 7)"
rm -rf "$TMP/run"
start=$(date +%s.%N)
"$CLI" train --out "$TMP/run" --population 60 --generations 4 --matches-per-genome 40 \
  --reeval-matches 60 --rounds 6 --seed 7 --pass-rule "${PASS_RULE:-free}" --exchange-rule "${EXCHANGE_RULE:-free}" --threads "$THREADS" --quiet > /dev/null
end=$(date +%s.%N)
printf '%-28s %7.2fs  %s\n' "train (best.json)" "$(echo "$end - $start" | bc -l)" \
  "$(sha256sum < "$TMP/run/best.json" | cut -c1-16)"
