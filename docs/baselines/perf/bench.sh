#!/usr/bin/env bash
# Times the simulator and prints a checksum of every result, so a speed
# change that also changes a result is caught. Usage (repository root, after
# `cargo build --release -p cli`):
#   docs/baselines/perf/bench.sh
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
    "$(tail -n +2 "$TMP/out.txt" | sha256sum | cut -c1-16)"
}

table() { # threads, strategy specs...
  local threads=$1
  shift
  local args=()
  for spec in "$@"; do args+=(--strategy "$spec"); done
  "$CLI" --player-count 4 --matches "$MATCHES" --rounds 8 --seed 42 --output "$TMP/result.json" \
    --threads "$threads" "${args[@]}"
}

N="neat:$GENOME"
for threads in "$THREADS" 0; do
  echo "== threads: $([ "$threads" = 0 ] && echo all || echo "$threads")"
  timed "4x lowest-legal" table "$threads" lowest-legal lowest-legal lowest-legal lowest-legal
  timed "4x neat" table "$threads" "$N" "$N" "$N" "$N"
  timed "mixed (1 neat + 3 classic)" table "$threads" "$N" lowest-legal endgame-denial adaptive:reading,tempo,bully
done

echo "== training (pop 60, 4 generations, 40 matches, 6 rounds, seed 7)"
rm -rf "$TMP/run"
start=$(date +%s.%N)
"$CLI" train --out "$TMP/run" --population 60 --generations 4 --matches-per-genome 40 \
  --reeval-matches 60 --rounds 6 --seed 7 --threads "$THREADS" --quiet > /dev/null
end=$(date +%s.%N)
printf '%-28s %7.2fs  %s\n' "train (best.json)" "$(echo "$end - $start" | bc -l)" \
  "$(sha256sum < "$TMP/run/best.json" | cut -c1-16)"
