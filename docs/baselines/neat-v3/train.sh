#!/usr/bin/env bash
# neat-v3 candidates: neat-v2 recipe with the new default feature set (hidden).
set -uo pipefail
CLI=/tmp/claude-1000/-home-sebi-Documents-Code-arschloch/d4dabbea-31e7-4303-a456-53cbaba01a2e/scratchpad/cli-train
cd /home/sebi/Documents/Code/arschloch-wt-open
OUT=$HOME/.cache/phase13c; mkdir -p $OUT
CH=docs/baselines/neat-v2/champion.json
RULES="--pass-rule final --exchange-rule forced"
train() { # name seed gens matches
  local d=$OUT/$1; [ -f $d/best.json ] && grep -q run_end $d/events.jsonl 2>/dev/null && return
  rm -rf $d
  $CLI train --out $d --seed $2 --population 150 --generations $3 --matches-per-genome $4 --reeval-matches 200 --rounds 8 \
    --champion-candidates 5 --weight-power 0.2 $RULES --threads 4 --quiet
}
for s in 31 32 33 34 35 36 37 38; do train short$s $s 120 80; done
for s in 31 32 33; do train long$s $s 300 200; done
# selection stream: seed 99, 2000 matches x 10 rounds
for n in short31 short32 short33 short34 short35 short36 short37 short38 long31 long32 long33; do
  [ -f $OUT/$n/sel-h2h.json ] && continue
  $CLI evaluate --genome $OUT/$n/best.json --player-count 4 --matches 2000 --rounds 10 --seed 99 $RULES --threads 4 --json $OUT/$n/sel-eval.json > $OUT/$n/sel-eval.txt
  $CLI evaluate --genome $OUT/$n/best.json --player-count 4 --opponent neat:$CH --matches 2000 --rounds 10 --seed 99 $RULES --threads 4 --json $OUT/$n/sel-h2h.json > $OUT/$n/sel-h2h.txt
done
echo ALLDONE > $OUT/ALLDONE
