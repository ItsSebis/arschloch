// Run with: node --test web/tests
import assert from "node:assert/strict";
import test from "node:test";
import {
  analysisUrl, championSpec, chartSvg, compareHtml, comparisonRows, esc, fmtScore, fmtSlope, keyNumbersHtml,
  keyRows, legend, noiseBand, opponentColour, opponentRows, opponentsSpec, opponentsTableHtml, pickerHtml,
  skippedHtml, stallMarker, swatchesHtml, trendColour, verdictHtml, verdictTone,
} from "../assets/analysis_lib.js";

// Generation numbers 10, 12, 14, 16: a stall at index `stall` is generation 10 + 2 * stall.
const run = (label = "r", stall = 2) => ({
  label,
  analysis: {
    generations: 4,
    last_generation: 16,
    stall_generation: 10 + 2 * stall,
    stall_index: stall,
    stall_fallback: false,
    peak_index: 2,
    peak_generation: 14,
    final_smoothed: 0.2,
    peak_smoothed: 0.3,
    final_reeval: { mean: 0.25, std_error: 0.02 },
    final_heldout: null,
    noise: { single_se: 0.05, change_sd: null },
    trend: { overall: { value: 0.1, low: -0.1, high: 0.3 }, last_third: null },
    curse: { reeval_minus_heldout: null, train_minus_reeval: 0.04 },
    skipped_lines: 0,
    opponents: [
      { name: "A<b>", first_third_mean: 0.1, last_third_mean: 0.4, overall_slope: { value: 1, low: 0.5, high: 1.5 }, last_third_slope: { value: 0, low: -1, high: 1 }, class: "flat" },
      { name: "B", first_third_mean: null, last_third_mean: null, overall_slope: null, last_third_slope: null, class: null },
    ],
    verdict: [{ code: "noise_limited", message: "m" }],
  },
  series: {
    generation: [10, 12, 14, 16],
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
  assert.equal(verdictTone("converged"), "info", "converged is not good news");
  assert.equal(verdictTone("declining"), "warn");
  assert.equal(verdictTone("no_clear_limit"), "good");
  assert.equal(verdictTone("pool_limited"), "warn");
  assert.equal(trendColour("rising"), "var(--s3)");
  assert.equal(trendColour(null), "var(--s6)");
  assert.equal(opponentColour(0), "var(--s1)");
  assert.equal(opponentColour(6), "var(--s1)");
  assert.equal(analysisUrl(), "/api/analysis");
  assert.equal(analysisUrl("a b/c"), "/api/analysis?path=a%20b%2Fc");
});

test("the stall marker is at the stall generation NUMBER, found through the index and clamped", () => {
  const r = run();
  assert.equal(r.analysis.stall_generation, 14);
  assert.deepEqual(stallMarker(r.series, r.analysis), { index: 2, generation: 14 });
  assert.equal(stallMarker(r.series, { stall_index: 99 }).generation, 16);
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
  assert.equal(byLabel["Stall generation"], "14 (last generation 16)");
  assert.equal(byLabel["Peak (smoothed)"], "+0.300");
  assert.equal(byLabel["Trend, last third (per 100 gens)"], "n/a");
  assert.equal(byLabel["Curse: re-evaluation minus held-out"], "n/a");
  assert.equal(byLabel["Final held-out"], "–");
});

test("comparison rows summarise each run", () => {
  const rows = comparisonRows([run("a", 1), run("b", 3)]);
  assert.deepEqual(rows.map((r) => [r.label, r.stall, r.finalReeval, r.lastThird, r.codes]), [
    ["a", 12, 0.25, null, ["noise_limited"]],
    ["b", 16, 0.25, null, ["noise_limited"]],
  ], "the stall column is a generation number");
});

test("chart specs carry the curves, the band and the stall line", () => {
  const spec = championSpec(run());
  assert.deepEqual(spec.vlines, [{ x: 14, label: "stall 14" }]);
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
  assert.ok(!svg.includes("<circle"), "no held-out markers without held-out scores");
  const opponents = chartSvg(opponentsSpec(run()));
  assert.ok(!opponents.includes("NaN"), "null gaps are skipped");
  assert.ok(chartSvg({ xs: [], series: [] }).includes("no data"));
  // A single generation does not divide by zero.
  const one = run();
  one.series = { ...one.series, generation: [0], smoothed: [0.1], reeval_mean: [0.1], train_fitness: [0.2], opponents: [] };
  assert.ok(!chartSvg(championSpec(one)).includes("NaN"));
});

test("held-out scores of new-best generations are drawn as markers on the champion chart", () => {
  const r = run();
  r.series.heldout_mean = [null, 0.15, null, 0.18];
  const spec = championSpec(r);
  assert.ok(legend(spec).some((l) => l.name === "held-out (new best)"));
  const svg = chartSvg(spec);
  assert.equal((svg.match(/<circle class="marker"/g) ?? []).length, 2, "one marker per held-out score");
  assert.equal((svg.match(/<path d=/g) ?? []).length, 4, "markers are not a line");
  assert.ok(!svg.includes("NaN"));
});

test("a long series does not overflow the stack", () => {
  const n = 250000;
  const xs = Array.from({ length: n }, (_, i) => i);
  const values = xs.map((i) => Math.sin(i / 1000));
  const svg = chartSvg({ xs, series: [{ name: "s", color: "red", values, band: { lo: values.map((v) => v - 1), hi: values.map((v) => v + 1) } }], vlines: [] });
  assert.ok(svg.startsWith("<svg"));
});

test("esc escapes the five HTML characters, also in numbers and repeated", () => {
  assert.equal(esc(`<a href="x">'&'</a>`), "&lt;a href=&quot;x&quot;&gt;&#39;&amp;&#39;&lt;/a&gt;");
  assert.equal(esc(12), "12");
  assert.equal(esc(null), "null");
});

const hostile = '</td><img src=x onerror=alert(1)>"\'&';

test("run labels, opponent names, verdict text and notes are escaped in every HTML string", () => {
  const r = run(hostile);
  r.analysis.opponents[0].name = hostile;
  r.analysis.verdict = [{ code: hostile, message: hostile }];
  const pages = [
    verdictHtml(r),
    opponentsTableHtml(r),
    compareHtml([r, run("ok")]),
    pickerHtml([r]),
    swatchesHtml([{ name: hostile, color: "var(--s1)" }]),
    skippedHtml([hostile]),
    keyNumbersHtml(r),
  ];
  for (const html of pages) {
    assert.ok(!html.includes("<img"), html);
    assert.ok(!html.includes("onerror=alert(1)>"), html);
    assert.ok(html.includes("&lt;img src=x onerror=alert(1)&gt;") || !html.includes("img"), html);
  }
  assert.ok(verdictHtml(r).includes(`&lt;/td&gt;&lt;img`), "message text is shown escaped");
  assert.ok(opponentsTableHtml(r).includes("&quot;&#39;&amp;"));
  // A vertical-line label is escaped inside the SVG as well.
  const spec = championSpec(r);
  spec.vlines = [{ x: 12, label: hostile }];
  const svg = chartSvg(spec);
  assert.ok(!svg.includes("<img") && svg.includes("&lt;img"));
  assert.equal(skippedHtml([]), "");
  assert.equal(skippedHtml(undefined), "");
});

test("opponent rows show last-third and whole-run slopes and tolerate missing values", () => {
  const rows = opponentRows(run().analysis);
  assert.deepEqual(rows[0], ["A<b>", "+0.100", "+0.400", "0.000 [-1.000, +1.000]", "+1.000 [+0.500, +1.500]", "flat"]);
  assert.deepEqual(rows[1], ["B", "–", "–", "n/a", "n/a", "n/a"]);
});
