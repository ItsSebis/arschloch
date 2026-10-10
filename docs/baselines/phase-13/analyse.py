#!/usr/bin/env python3
"""Tables and verdicts for the Phase 13 experiment matrix (see run.sh).

Usage: docs/baselines/phase-13/analyse.py [OUT] [--json FILE]
       docs/baselines/phase-13/analyse.py --selftest
Environment: BEST (default hidden; like run.sh: with BEST=base the stacked arms B2, B3 and the
matched-budget control C2 are compared against A0 instead of A2).

For every arm directory in OUT (default runs/phase13) it reads the per-seed training events
(events.jsonl) and the held-out evaluations (eval.json, h2h.json written by run.sh) and prints
markdown:

* per arm and setting: the held-out mixed-opponents score per seed (seeds are taken from the
  `seedN/` in each result's `genome` path, never from the order of the results), the seed
  average with its standard error (spread between seeds), the score against the neat-v2
  champion (head-to-head), and the mean and population standard deviation of the champion's
  per-round role score computed exactly from the placement counts of the mixed cell (the role
  score is linear in the place: place p of n -> 1 - 2 (p - 1) / (n - 1)); multi-setting
  evaluations (`settings[]` of the JSON, keyed by label such as 3p-single) give one row per
  setting, and numbers are never compared across table sizes;
* the stall generation of each run and the winner's-curse gap;
* comparisons, each arm family against its own control: A1, A2 vs A0; B1 vs C2 (matched
  evaluation budget) and vs A2; B2, B3 and C2 vs the BEST control (A2, or A0 when BEST=base);
  M1 per setting vs the matching specialist (S3 for 3p-single, S5D for 5p-double, A0 for
  4p-single; other settings are reported without a verdict). Verdict: Welch's t-test between
  the two arms' per-seed scores (two-sided 95%, Welch-Satterthwaite degrees of freedom,
  conservative table lookup): BETTER / WORSE only when |t| exceeds the critical value, else
  no difference. The plain difference and the z of the 2-SE rule are printed for reference.

Python 3 standard library only.
"""
import json
import math
import os
import re
import statistics as st
import sys
import tempfile

# two-sided 95% critical values of Student's t
T_CRIT = [(1, 12.71), (2, 4.30), (3, 3.18), (4, 2.78), (5, 2.57), (6, 2.45), (7, 2.36), (8, 2.31),
          (9, 2.26), (10, 2.23), (12, 2.18), (15, 2.13), (20, 2.09), (30, 2.04)]
T_INF = 1.96

SPECIALISTS = {"3p-single": "S3", "5p-double": "S5D", "4p-single": "A0"}


def controls_for(best):
    """{arm: [controls]} for the single-setting arms (M1 is per setting, see SPECIALISTS)."""
    base = "A0" if best == "base" else "A2"
    return {"A1": ["A0"], "A2": ["A0"], "B1": ["C2", "A2"], "B2": [base], "B3": [base],
            "C2": [base]}


def t_critical(df):
    """Critical value for the largest tabulated df not above `df` (conservative); inf -> 1.96."""
    if math.isinf(df):
        return T_INF
    key = max([k for k, _ in T_CRIT if k <= max(df, 1)] or [1])
    return dict(T_CRIT)[key]


def welch(a, b):
    """(t, df) of Welch's t-test for samples a, b; None when a sample has fewer than 2 values."""
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    d = st.mean(a) - st.mean(b)
    if va + vb == 0:
        return (0.0 if d == 0 else math.copysign(math.inf, d)), float(len(a) + len(b) - 2)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return d / math.sqrt(va + vb), df


def verdict(a, b):
    """(difference, t, df, crit, word, z by the 2-SE rule) or None."""
    w = welch(a, b)
    if w is None:
        return None
    t, df = w
    crit = t_critical(df)
    word = "BETTER" if t > crit else "WORSE" if t < -crit else "no difference"
    se = math.sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))
    d = st.mean(a) - st.mean(b)
    z = d / se if se > 0 else (0.0 if d == 0 else math.copysign(math.inf, d))
    return d, t, df, crit, word, z


def role_stats(placements):
    """(mean, population sd) of the per-round role score from placement counts (index 0 = place 1)."""
    n = len(placements)
    total = sum(placements)
    if n < 2 or total == 0:
        return None
    scores = [1 - 2 * p / (n - 1) for p in range(n)]
    mean = sum(c * s for c, s in zip(placements, scores)) / total
    var = sum(c * s * s for c, s in zip(placements, scores)) / total - mean * mean
    return mean, math.sqrt(max(0.0, var))


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
    re_ = [g["champion"]["reeval"]["mean"] for g in gens]
    se = st.mean(g["champion"]["reeval"]["std_error"] for g in gens)
    s = smooth(re_, 10)
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


def seed_of(genome_path):
    m = re.search(r"(?:^|[\\/])seed(\d+)[\\/]", genome_path)
    return int(m.group(1)) if m else None


def setting_label(doc):
    return f"{doc['player_count']}p-{str(doc['deck_variant']).lower()}"


def eval_cells(path):
    """{setting label: {seed: {opponent: cell}}} from an `evaluate --json` file.

    Several settings are read from `settings[]` (labels like 3p-single); a single-setting file
    from `results`. The seed comes from the `seedN/` in each result's genome path."""
    with open(path) as f:
        doc = json.load(f)
    if doc.get("settings"):
        blocks = [(s["label"], s["results"]) for s in doc["settings"]]
    else:
        blocks = [(setting_label(doc), doc["results"])]
    out = {}
    for label, results in blocks:
        for r in results:
            seed = seed_of(r["genome"])
            if seed is None:
                print(f"warning: {path}: no seedN/ in genome path {r['genome']!r}, skipped",
                      file=sys.stderr)
                continue
            out.setdefault(label, {})[seed] = {c["opponent"]: c for c in r["cells"]}
    return out


def mixed_cell(cells):
    return next((c for n, c in cells.items() if n.startswith("mixed")), None)


def other_cell(cells):
    return next((c for n, c in cells.items() if not n.startswith("mixed")), None)


def seed_dirs(path):
    found = []
    for d in os.listdir(path):
        m = re.fullmatch(r"seed(\d+)", d)
        if m and os.path.isdir(os.path.join(path, d)):
            found.append((int(m.group(1)), d))
    return sorted(found)


def summarise_arm(out, arm):
    base = os.path.join(out, arm)
    evp = os.path.join(base, "eval.json")
    ev = eval_cells(evp) if os.path.exists(evp) else {}
    hp = os.path.join(base, "h2h.json")
    h2h = eval_cells(hp) if os.path.exists(hp) else {}
    stalls, gaps, hidden = [], [], []
    for _, d in seed_dirs(base):
        ep = os.path.join(base, d, "events.jsonl")
        if not os.path.exists(ep):
            continue
        _, gens, end = load_events(ep)
        sg = stall_generation(gens)
        if sg is not None:
            stalls.append(sg)
        if end and end.get("best_heldout"):
            gaps.append(end["best_reeval"]["mean"] - end["best_heldout"]["mean"])
        if gens:
            hidden.append(gens[-1]["champion"].get("hidden_nodes", 0))
    settings = {}
    for label, by_seed in ev.items():
        per_seed, pooled = {}, None
        for seed in sorted(by_seed):
            c = mixed_cell(by_seed[seed])
            if c:
                per_seed[seed] = c["mean"]
                pl = c["placements"]
                pooled = pl[:] if pooled is None else [a + b for a, b in zip(pooled, pl)]
        h2h_per_seed = {}
        for seed in sorted(h2h.get(label, {})):
            c = other_cell(h2h[label][seed])
            if c:
                h2h_per_seed[seed] = c["mean"]
        m, se = mean_se(list(per_seed.values()))
        hm, hse = mean_se(list(h2h_per_seed.values()))
        settings[label] = dict(seeds=sorted(per_seed), per_seed=per_seed, mixed=m, se=se, h2h=hm,
                               h2h_se=hse, h2h_per_seed=h2h_per_seed,
                               role=role_stats(pooled) if pooled else None)
    return dict(settings=settings, stall=stalls, gap=gaps, hidden=hidden)


def comparisons(summary, best, metric="per_seed"):
    """[(arm, label, control, verdict tuple or None)] in a fixed order."""
    rows = []
    ctl = controls_for(best)
    for arm in summary:
        for label, r in summary[arm]["settings"].items():
            if arm == "M1":
                controls = [SPECIALISTS[label]] if label in SPECIALISTS else []
                if not controls:
                    rows.append((arm, label, None, None))
            else:
                controls = ctl.get(arm, [])
            for c in controls:
                other = summary.get(c, {"settings": {}})["settings"].get(label)
                if other is None:
                    rows.append((arm, label, c, None))
                    continue
                rows.append((arm, label, c,
                             verdict(list(r[metric].values()), list(other[metric].values()))))
    return rows


def render(out, summary, best):
    lines = [f"# Phase 13 results ({out})", ""]
    lines.append("Caution: with 3 seeds per arm the tests have only 2-4 degrees of freedom, so "
                 "conclusions are weak; the contested arms are rerun with 5 seeds (SEEDS env of run.sh).")
    lines.append("")
    lines.append("| arm | setting | seeds | held-out mixed (seed mean ± SE) | per seed | per-round role score "
                 "(mean, sd) | vs neat-v2 champion | stall gen | curse gap | hidden nodes |")
    lines.append("|---|---|---:|---:|---|---|---:|---|---:|---:|")
    for arm, a in summary.items():
        for label, r in a["settings"].items():
            per = ", ".join(f"s{s}:{v:+.3f}" for s, v in r["per_seed"].items())
            role = f"{r['role'][0]:+.3f}, {r['role'][1]:.3f}" if r["role"] else "-"
            stall = ", ".join(str(s) for s in a["stall"]) or "-"
            gap = f"{st.mean(a['gap']):+.3f}" if a["gap"] else "-"
            hid = f"{st.mean(a['hidden']):.0f}" if a["hidden"] else "-"
            lines.append(f"| {arm} | {label} | {len(r['seeds'])} | {r['mixed']:+.3f} ± {r['se']:.3f} | {per} | "
                         f"{role} | {r['h2h']:+.3f} ± {r['h2h_se']:.3f} | {stall} | {gap} | {hid} |")
    lines += ["", "## Comparisons (per setting, never across table sizes)", "",
              "| arm | setting | control | difference | Welch t | df | crit (95%) | verdict | z (2-SE rule) |",
              "|---|---|---|---:|---:|---:|---:|---|---:|"]
    for arm, label, c, v in comparisons(summary, best):
        if c is None:
            lines.append(f"| {arm} | {label} | - | - | - | - | - | no specialist: reported without verdict | - |")
        elif v is None:
            lines.append(f"| {arm} | {label} | {c} | - | - | - | - | n/a (control missing or < 2 seeds) | - |")
        else:
            d, t, df, crit, word, z = v
            lines.append(f"| {arm} | {label} | {c} | {d:+.3f} | {t:+.2f} | {df:.1f} | {crit:.2f} | {word} | {z:+.1f} |")
    lines += ["", "## Head-to-head against the committed neat-v2 champion (per-seed scores, same rule)", "",
              "The default battery saturates against weak opponents; the champion is the discriminating test.", "",
              "| arm | setting | control | difference | Welch t | df | crit (95%) | verdict | z (2-SE rule) |",
              "|---|---|---|---:|---:|---:|---:|---|---:|"]
    for arm, label, c, v in comparisons(summary, best, "h2h_per_seed"):
        if c is None or v is None:
            continue
        d, t, df, crit, word, z = v
        lines.append(f"| {arm} | {label} | {c} | {d:+.3f} | {t:+.2f} | {df:.1f} | {crit:.2f} | {word} | {z:+.1f} |")
    lines += ["", "Rule: BETTER/WORSE only when |t| exceeds the two-sided 95% critical value of Student's t at the "
              "Welch-Satterthwaite degrees of freedom (the table lookup rounds the df down, i.e. conservatively); "
              "otherwise no difference. B1 is compared against C2 (same evaluation budget) and A2. "
              "The z of the 2-SE rule is for reference only."]
    return "\n".join(lines)


def main(argv):
    if "--selftest" in argv:
        selftest()
        return
    json_out = None
    args = []
    i = 0
    while i < len(argv):
        if argv[i] == "--json":
            json_out = argv[i + 1]
            i += 2
            continue
        args.append(argv[i])
        i += 1
    out = args[0] if args else "runs/phase13"
    best = os.environ.get("BEST", "hidden")
    arms = sorted(d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d)))
    summary = {arm: summarise_arm(out, arm) for arm in arms}
    print(render(out, summary, best))
    if json_out:
        with open(json_out, "w") as f:
            json.dump(summary, f, indent=2)


# ---------------------------------------------------------------- self test

def _cell(opp, mean, placements):
    return {"opponent": opp, "mean": mean, "std_error": 0.01, "matches": 10, "placements": placements,
            "spread": 0.0, "avg_rank": 1.0, "avg_rank_std_error": 0.0}


def _results(arm_dir, scores, n, opp):
    """Results written in REVERSE seed order, to prove the mapping is by genome path."""
    pl = {3: [4, 3, 2], 4: [5, 3, 1, 1]}.get(n, [1] * n)
    return [{"genome": f"{arm_dir}/seed{seed}/best.json",
             "cells": [_cell(opp, scores[seed] - 0.1, pl), _cell("mixed (all opponents)", scores[seed], pl)]}
            for seed in sorted(scores, reverse=True)]


def _write_eval(path, arm_dir, scores=None, label="4p-single", opp="X", multi=None):
    if multi:
        doc = {"player_count": 3, "deck_variant": "Single", "results": [],
               "settings": [{"label": lb, "player_count": int(lb[0]), "deck_variant": lb.split("-")[1].title(),
                             "results": _results(arm_dir, sc, int(lb[0]), opp)} for lb, sc in multi.items()]}
    else:
        n = int(label[0])
        doc = {"player_count": n, "deck_variant": label.split("-")[1].title(),
               "results": _results(arm_dir, scores, n, opp)}
    with open(path, "w") as f:
        json.dump(doc, f)


def selftest():
    # welch by hand: a = [1,2,3], b = [4,5,6]: means 2, 5; variances 1, 1 -> se^2 = 2/3, df = 4
    t, df = welch([1, 2, 3], [4, 5, 6])
    assert abs(t - (-3 / math.sqrt(2 / 3))) < 1e-12 and abs(df - 4.0) < 1e-12, (t, df)
    assert t_critical(4.0) == 2.78 and t_critical(4.9) == 2.78 and t_critical(11) == 2.23
    assert t_critical(1000) == 2.04 and t_critical(math.inf) == 1.96 and t_critical(0.5) == 12.71
    assert verdict([1, 2, 3], [4, 5, 6])[4] == "WORSE"      # |t| = 3.67 > 2.78
    assert verdict([1, 2, 3], [1.5, 2.5, 3.5])[4] == "no difference"
    assert welch([1.0], [1, 2]) is None
    # role stats: 4 places, counts 5,3,1,1 -> scores 1, 1/3, -1/3, -1
    mean, sd = role_stats([5, 3, 1, 1])
    exp = (5 + 3 / 3 - 1 / 3 - 1) / 10
    assert abs(mean - exp) < 1e-12
    var = (5 + 3 / 9 + 1 / 9 + 1) / 10 - exp * exp
    assert abs(sd - math.sqrt(var)) < 1e-12
    assert seed_of("/x/A0/seed10/best.json") == 10 and seed_of("/x/best.json") is None
    # synthetic run directory (seed10 included, results written in reverse seed order)
    with tempfile.TemporaryDirectory() as tmp:
        arms = {"A0": {1: .60, 2: .62, 10: .64}, "A2": {1: .65, 2: .66, 10: .67},
                "B1": {1: .70, 2: .71, 10: .72}, "C2": {1: .66, 2: .67, 10: .68},
                "S3": {1: .5, 2: .5, 10: .52}, "S5D": {1: .4, 2: .42, 10: .41}}
        for arm, sc in arms.items():
            d = os.path.join(tmp, arm)
            for s in sc:
                os.makedirs(os.path.join(d, f"seed{s}"))
            label = {"S3": "3p-single", "S5D": "5p-double"}.get(arm, "4p-single")
            _write_eval(os.path.join(d, "eval.json"), d, sc, label)
            _write_eval(os.path.join(d, "h2h.json"), d, {s: v - 0.2 for s, v in sc.items()}, label,
                        "Neat(champion)")
        m1 = os.path.join(tmp, "M1")
        multi = {"3p-single": {1: .55, 2: .56, 10: .57}, "4p-single": {1: .5, 2: .51, 10: .52},
                 "6p-double": {1: .3, 2: .3, 10: .3}}
        for s in multi["3p-single"]:
            os.makedirs(os.path.join(m1, f"seed{s}"))
        _write_eval(os.path.join(m1, "eval.json"), m1, multi=multi)
        _write_eval(os.path.join(m1, "h2h.json"), m1, multi=multi)
        summary = {arm: summarise_arm(tmp, arm) for arm in sorted(os.listdir(tmp))}
        # mapping by path: every seed (10 included) gets its own score although written in reverse
        assert summary["A0"]["settings"]["4p-single"]["per_seed"] == {1: .60, 2: .62, 10: .64}
        assert set(summary["M1"]["settings"]) == {"3p-single", "4p-single", "6p-double"}
        assert abs(summary["A0"]["settings"]["4p-single"]["h2h"] - 0.32) < 1e-12
        rows = {(a, lb, c): v for a, lb, c, v in comparisons(summary, "hidden")}
        # B1 vs C2 by hand: diff .04, both variances 1e-4 -> se = sqrt(2e-4/3), df 4, crit 2.78
        v = rows[("B1", "4p-single", "C2")]
        assert abs(v[0] - 0.04) < 1e-12 and v[4] == "BETTER"
        assert abs(v[1] - 0.04 / math.sqrt(2e-4 / 3)) < 1e-9 and abs(v[2] - 4.0) < 1e-9
        assert rows[("B1", "4p-single", "A2")][4] == "BETTER"        # diff .05, t = 6.12
        # A2 vs A0: diff .04, variances 1e-4 and 4e-4 -> t = 3.10, df = 2.94 -> crit 4.30: not enough
        v = rows[("A2", "4p-single", "A0")]
        assert abs(v[1] - 3.098) < 1e-3 and 2.9 < v[2] < 3.0 and v[3] == 4.30 and v[4] == "no difference"
        assert abs(v[5] - 3.098) < 1e-3                               # the 2-SE z would say BETTER
        # C2 vs A2: diff .01, t = 1.22
        assert rows[("C2", "4p-single", "A2")][4] == "no difference"
        assert ("A1", "4p-single", "A0") not in rows                  # no A1 arm present
        # M1: 3p-single vs S3, 4p-single vs A0, 6p-double without a specialist
        assert rows[("M1", "3p-single", "S3")][0] > 0
        assert ("M1", "4p-single", "A0") in rows
        assert rows[("M1", "6p-double", None)] is None
        assert not any(arm == "S3" for arm, _, _ in rows)             # never across table sizes
        text = render(tmp, summary, "hidden")
        assert "weak" in text and "5 seeds" in text and "no specialist" in text
        assert "s10:+0.640" in text
        assert controls_for("base")["C2"] == ["A0"] and controls_for("hidden")["B2"] == ["A2"]
    print("selftest ok")


if __name__ == "__main__":
    main(sys.argv[1:])
