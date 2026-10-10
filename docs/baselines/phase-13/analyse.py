#!/usr/bin/env python3
"""Tables and verdicts for the Phase 13 experiment matrix (see run.sh).

Usage: docs/baselines/phase-13/analyse.py [OUT] [--control A0] [--json FILE]

For every arm directory in OUT (default runs/phase13) it reads the per-seed training events
(events.jsonl) and the held-out evaluation (eval.json written by run.sh) and prints markdown:

* held-out mixed-opponents score per seed, the seed average and its standard error (the spread
  between seeds, not the within-evaluation error), the score against the neat-v2 champion
  (the frozen head-to-head), the spread of the champion's per-round scores when the run
  recorded it;
* the stall generation of each run (the first generation from which the 10-generation smoothed
  champion re-evaluation stays within one standard error of its peak) and the winner's-curse
  gap (best generation's re-evaluation minus its held-out score);
* the verdict against the control: an arm counts as better only when its seed-average exceeds
  the control's by at least 2 standard errors of the difference (the project rule: a default
  changes only for such an arm).

Python 3 standard library only.
"""
import json
import math
import os
import statistics as st
import sys


def load_events(path):
    gens, start, end = [], None, None
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if e["type"] == "generation":
                gens.append(e)
            elif e["type"] == "run_start":
                start = e
            elif e["type"] == "run_end":
                end = e
    return start, gens, end


def smooth(x, w):
    return [sum(x[max(0, i - w + 1): i + 1]) / len(x[max(0, i - w + 1): i + 1]) for i in range(len(x))]


def stall_generation(gens):
    """First generation from which the smoothed champion score stays within 1 SE of its peak."""
    if not gens:
        return None
    re = [g["champion"]["reeval"]["mean"] for g in gens]
    se = st.mean(g["champion"]["reeval"]["std_error"] for g in gens)
    s = smooth(re, 10)
    peak = max(s)
    for i in range(len(s)):
        if all(v >= peak - se for v in s[i:]):
            return i
    return len(s) - 1


def mean_se(values):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = st.mean(values)
    return m, (st.stdev(values) / math.sqrt(n) if n > 1 else float("nan"))


def eval_cells(path):
    """{genome stem: {opponent: cell}} from an `evaluate --json` file (per-setting results if present)."""
    with open(path) as f:
        doc = json.load(f)
    out = {}
    for r in doc["results"]:
        out[r["genome"]] = {c["opponent"]: c for c in r["cells"]}
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = args[0] if args else "runs/phase13"
    control = "A0"
    json_out = None
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--control":
            control = argv[i + 1]
        if a == "--json":
            json_out = argv[i + 1]
    arms = sorted(d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d)))
    rows, summary = [], {}
    for arm in arms:
        seeds = sorted(d for d in os.listdir(os.path.join(out, arm)) if d.startswith("seed"))
        per_seed_mixed, per_seed_h2h, stalls, gaps, hidden = [], [], [], [], []
        ev = None
        evp = os.path.join(out, arm, "eval.json")
        if os.path.exists(evp):
            ev = eval_cells(evp)
        for s in seeds:
            ep = os.path.join(out, arm, s, "events.jsonl")
            if not os.path.exists(ep):
                continue
            start, gens, end = load_events(ep)
            sg = stall_generation(gens)
            if sg is not None:
                stalls.append(sg)
            if end and end.get("best_heldout"):
                gaps.append(end["best_reeval"]["mean"] - end["best_heldout"]["mean"])
            if gens:
                hidden.append(gens[-1]["champion"].get("hidden_nodes", 0))
            if ev:
                # evaluate labels columns by the genome file stem: best, best (2), ... in the order given
                idx = seeds.index(s)
                key = list(ev.keys())[idx] if idx < len(ev) else None
                if key:
                    cells = ev[key]
                    mixed = next((c for n, c in cells.items() if n.startswith("mixed")), None)
                    h2h = next((c for n, c in cells.items() if "champion" in n.lower() or "neat" in n.lower()), None)
                    if mixed:
                        per_seed_mixed.append(mixed["mean"])
                    if h2h:
                        per_seed_h2h.append(h2h["mean"])
        m, se = mean_se(per_seed_mixed)
        h, hse = mean_se(per_seed_h2h)
        summary[arm] = dict(seeds=len(per_seed_mixed), mixed=m, se=se, per_seed=per_seed_mixed,
                            h2h=h, h2h_se=hse, stall=stalls, gap=gaps, hidden=hidden)
    c = summary.get(control)
    print(f"# Phase 13 results ({out})\n")
    print("| arm | seeds | held-out mixed (seed mean ± SE) | per seed | vs neat-v2 champion | stall gen | curse gap | hidden nodes | vs control {c} |".replace("{c}", control))
    print("|---|---:|---:|---|---:|---|---:|---:|---|")
    for arm, r in summary.items():
        verdict = "-"
        if c and arm != control and r["seeds"] and c["seeds"] and not math.isnan(r["se"]) and not math.isnan(c["se"]):
            d = r["mixed"] - c["mixed"]
            dse = math.sqrt(r["se"] ** 2 + c["se"] ** 2)
            z = d / dse if dse > 0 else float("inf")
            verdict = f"{d:+.3f} ({z:+.1f} SE) " + ("BETTER" if z >= 2 else "WORSE" if z <= -2 else "no difference")
        per = ", ".join(f"{v:+.3f}" for v in r["per_seed"])
        stall = ", ".join(str(s) for s in r["stall"]) or "-"
        gap = f"{st.mean(r['gap']):+.3f}" if r["gap"] else "-"
        hid = f"{st.mean(r['hidden']):.0f}" if r["hidden"] else "-"
        print(f"| {arm} | {r['seeds']} | {r['mixed']:+.3f} ± {r['se']:.3f} | {per} | {r['h2h']:+.3f} ± {r['h2h_se']:.3f} | {stall} | {gap} | {hid} | {verdict} |")
    print("\nRule: an arm is better only if its seed average beats the control's by >= 2 standard errors of the "
          "difference (standard errors across seeds); anything else is documented as no difference.")
    if json_out:
        with open(json_out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
