"""Summarises the experiment runs: mean score of each configuration's best
champion against the opponents it trained on, strong opponents it never saw
and weak ones it never saw, from the `*.eval.json` files written by
run_experiments.sh."""
import json
import os
import statistics as st
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/neat-experiments"
TRAINED = ["LowestLegal", "EndgameDenial", "Adaptive(reading,tempo,bully)"]
STRONG = ["Adaptive(counting,reading)", "Adaptive(reading,deception=0.2,tempo,bully)"]
WEAK = ["HoldBackPairs", "GreedyHighest", "RandomLegal"]
LABELS = {
    "base": "baseline (1 candidate, no hall, power 0.5)",
    "topk": "top-5 champion selection",
    "hof": "hall of fame (3, every 5)",
    "wp02": "weight power 0.2",
    "wp01": "weight power 0.1",
    "combo": "top-5 selection + power 0.2",
}


def cells(name, seed):
    path = f"{OUT}/{name}_{seed}.eval.json"
    if not os.path.exists(path):
        return None
    return {c["opponent"]: c["mean"] for c in json.load(open(path))["results"][0]["cells"]}


def summary(name, seed):
    c = cells(name, seed)
    return (
        st.mean(c[n] for n in TRAINED),
        st.mean(c[n] for n in STRONG),
        st.mean(c[n] for n in WEAK),
        c["mixed (all opponents)"],
    )


for name, label in LABELS.items():
    rows = [summary(name, s) for s in range(1, 6) if cells(name, s)]
    if not rows:
        continue
    parts = []
    for i, what in enumerate(("trained-3", "stronger-unseen-2", "weak-unseen-3", "mixed-8")):
        values = [r[i] for r in rows]
        spread = st.stdev(values) if len(values) > 1 else 0.0
        parts.append(f"{what} {st.mean(values):+.3f} (sd {spread:.3f})")
    print(f"{label:44s} {len(rows)} seeds   " + "   ".join(parts))

paired = [
    summary("combo", s)[0] - summary("base", s)[0]
    for s in range(1, 6)
    if cells("combo", s) and cells("base", s)
]
if len(paired) > 1:
    se = st.stdev(paired) / len(paired) ** 0.5
    print(f"\npaired difference combo - base on the trained opponents: {st.mean(paired):+.3f} (standard error {se:.3f}, {len(paired)} seeds)")
