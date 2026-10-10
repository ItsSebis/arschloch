#!/usr/bin/env bash
# Thread-scaling check of `cli train`. For each thread count it trains the
# same small run (population 150, 100 matches per genome, 8 rounds, 4 players,
# single deck, the default hand-written opponent pool) for a few generations
# into a temporary directory and prints
#   - the median wall seconds per generation and the median rounds/s,
#   - the median seconds of every stage from events.jsonl (the `timings`
#     object of the generation events, see docs/TRAINING.md),
# then checks that best.json is byte-identical for every thread count
# (results do not depend on --threads): OK or FAIL.
# Usage (repository root, after `cargo build --release -p cli`):
#   docs/baselines/perf/scaling.sh
#   THREADS_LIST="1 4 16" GENERATIONS=10 docs/baselines/perf/scaling.sh
# Needs python3 for the medians (without it only the wall time and the
# checksum check are printed). Thread counts above the number of logical
# CPUs (nproc) are skipped with a note; set ALLOW_OVERSUBSCRIBE=1 to keep them.
set -euo pipefail
export LC_ALL=C
CLI=${CLI:-${CARGO_TARGET_DIR:-target}/release/cli}
GENERATIONS=${GENERATIONS:-6}
POPULATION=${POPULATION:-150}
THREADS_LIST=${THREADS_LIST:-"1 2 4 8 16"}
SEED=${SEED:-7}
CPUS=$(nproc 2>/dev/null || echo 1)

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "population $POPULATION, 100 matches/genome, 8 rounds, 4 players, $GENERATIONS generations, seed $SEED; $CPUS logical CPUs"
checksums=()
for threads in $THREADS_LIST; do
  if [ "$threads" -gt "$CPUS" ] && [ "${ALLOW_OVERSUBSCRIBE:-0}" != 1 ]; then
    echo "-- threads $threads: skipped (more than the $CPUS logical CPUs; ALLOW_OVERSUBSCRIBE=1 runs it anyway)"
    continue
  fi
  run="$TMP/t$threads"
  start=$(date +%s.%N)
  "$CLI" train --out "$run" --player-count 4 --population "$POPULATION" --generations "$GENERATIONS" \
    --matches-per-genome 100 --rounds 8 --seed "$SEED" --threads "$threads" --quiet > /dev/null
  end=$(date +%s.%N)
  wall=$(echo "$end - $start" | bc -l)
  sum=$(sha256sum < "$run/best.json" | cut -c1-16)
  checksums+=("$sum")
  printf -- '-- threads %s: total wall %.2fs, best.json %s\n' "$threads" "$wall" "$sum"
  if command -v python3 > /dev/null; then
    python3 -I - "$run/events.jsonl" <<'PY'
import json, statistics, sys
gens = [e for e in map(json.loads, open(sys.argv[1])) if e["type"] == "generation"]
med = lambda xs: statistics.median(xs)
print(f"   per generation (median of {len(gens)}): {med([g['generation_secs'] for g in gens]):.3f} s, "
      f"{med([g['rounds_per_sec'] for g in gens]):,.0f} rounds/s")
stages = {}
for g in gens:
    for name, value in (g.get("timings") or {}).items():
        stages.setdefault(name, []).append(value)
total = sum(med(v) for v in stages.values()) or 1.0
for name, values in stages.items():
    m = med(values)
    print(f"   {name:<30} {m:8.3f} s  {100 * m / total:5.1f}%")
PY
  fi
done

if [ "${#checksums[@]}" -gt 1 ]; then
  if [ "$(printf '%s\n' "${checksums[@]}" | sort -u | wc -l)" = 1 ]; then
    echo "thread-count independence: OK (best.json identical for every thread count)"
  else
    echo "thread-count independence: FAIL (best.json differs: ${checksums[*]})"
    exit 1
  fi
else
  echo "thread-count independence: not checked (fewer than two thread counts ran)"
fi
