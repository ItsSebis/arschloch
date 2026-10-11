// Run with: node --test web/tests
import assert from "node:assert/strict";
import test from "node:test";
import {
  analysisUrl, championSpec, chartSvg, comparisonRows, fmtScore, fmtSlope, keyRows, legend, noiseBand,
  opponentColour, opponentsSpec, stallMarker, trendColour, verdictTone,
} from "../assets/analysis_lib.js";

const run = (label = "r", stall = 2) => ({
  label,
  analysis: {
    generations: 4,
    stall_generation: stall,
    peak_smoothed: 0.3,
    final_reeval: { mean: 0.25, std_error: 0.02 },
    final_heldout: null,
    noise: { single_se: 0.05, change_sd: null },
    trend: { overall: { value: 0.1, low: -0.1, high: 0.3 }, last_third: null },
    curse: { reeval_minus_heldout: null, train_minus_reeval: 0.04 },
    skipped_lines: 0,
    opponents: [],
    verdict: [{ code: "noise_limited", message: "m" }],
  },
  series: {
    generation: [0, 1, 2, 3],
    reeval_mean: [0.1, 0.2, 0.3, 0.2],
    reeval_se: [0.05, 0.05, 0.05, 0.05],
    smoothed: [0.1, 0.15, 0.2, 0.2],
    train_fitness: [0.2, 0.3, 0.4, 0.3],
    heldout_mean: [null, null, null, null],
    opponents: [{ name: "A<b>", means: [0.1, null, 0.3, 0.4] }, { name: "B", means: [0, 0, 0, 0] }],
  },
});

test("slope and score formatting, with missing values", () => {
  assert.equal(fmtSlope({ value: 0.1, low: -0.1, high: 0.3 }, 2), "+0.10 [-0.10, +0.30]");
  assert.equal(fmtSlope(null), "n/a");
  assert.equal(fmtScore({ mean: 0.25, std_error: 0.02 }), "+0.250 ± 0.020");
  assert.equal(fmtScore(null), "–");
});

test("tones, colours and urls", () => {
  assert.equal(verdictTone("still_improving"), "info");
  assert.equal(verdictTone("converged"), "good");
  assert.equal(verdictTone("pool_limited"), "warn");
  assert.equal(trendColour("rising"), "var(--s3)");
  assert.equal(trendColour(null), "var(--s6)");
  assert.equal(opponentColour(0), "var(--s1)");
  assert.equal(opponentColour(6), "var(--s1)");
  assert.equal(analysisUrl(), "/api/analysis");
  assert.equal(analysisUrl("a b/c"), "/api/analysis?path=a%20b%2Fc");
});

test("the stall marker is clamped into the series", () => {
  const r = run();
  assert.deepEqual(stallMarker(r.series, r.analysis), { index: 2, generation: 2 });
  assert.equal(stallMarker(r.series, { stall_generation: 99 }).generation, 3);
  assert.equal(stallMarker({ generation: [] }, r.analysis), null);
});

test("the noise band is plus and minus one se around the smoothed curve", () => {
  const band = noiseBand(run().series, 0.05);
  assert.ok(Math.abs(band.hi[1] - 0.2) < 1e-12 && Math.abs(band.lo[1] - 0.1) < 1e-12);
  assert.equal(band.lo.length, 4);
});

test("key rows cover stall, noise, trends and curse, tolerating missing parts", () => {
  const rows = keyRows(run().analysis);
  const byLabel = Object.fromEntries(rows.map(([k, v]) => [k, v]));
  assert.equal(byLabel["Stall generation"], "2 of 3");
  assert.equal(byLabel["Trend, last third (per 100 gens)"], "n/a");
  assert.equal(byLabel["Curse: re-evaluation minus held-out"], "n/a");
  assert.equal(byLabel["Final held-out"], "–");
});

test("comparison rows summarise each run", () => {
  const rows = comparisonRows([run("a", 1), run("b", 3)]);
  assert.deepEqual(rows.map((r) => [r.label, r.stall, r.finalReeval, r.lastThird, r.codes]), [
    ["a", 1, 0.25, null, ["noise_limited"]],
    ["b", 3, 0.25, null, ["noise_limited"]],
  ]);
});

test("chart specs carry the curves, the band and the stall line", () => {
  const spec = championSpec(run());
  assert.deepEqual(spec.vlines, [{ x: 2, label: "stall 2" }]);
  assert.deepEqual(legend(spec).map((l) => l.name), ["re-evaluation", "smoothed", "training fitness"]);
  const opp = opponentsSpec(run());
  assert.equal(opp.series.length, 2);
  assert.equal(opp.ref, 0);
});

test("chartSvg draws paths, a stall line and escapes names; empty data is a note", () => {
  const svg = chartSvg(championSpec(run()));
  assert.ok(svg.startsWith("<svg"));
  assert.ok(svg.includes('class="stall"'));
  assert.ok(svg.includes("fill=\"var(--band)\""));
  assert.equal((svg.match(/<path d=/g) ?? []).length, 4, "band plus three lines");
  const opponents = chartSvg(opponentsSpec(run()));
  assert.ok(!opponents.includes("NaN"), "null gaps are skipped");
  assert.ok(chartSvg({ xs: [], series: [] }).includes("no data"));
  // A single generation does not divide by zero.
  const one = run();
  one.series = { ...one.series, generation: [0], smoothed: [0.1], reeval_mean: [0.1], train_fitness: [0.2], opponents: [] };
  assert.ok(!chartSvg(championSpec(one)).includes("NaN"));
});
