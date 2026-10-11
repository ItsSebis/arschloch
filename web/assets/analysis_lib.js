// Pure helpers for the run-analysis page: no DOM, no network, so they can be
// unit-tested in Node (see tests/analysis.test.mjs) and reused by analysis.js.
import { bandPath, extent, fmt, linePath, linearScale, niceScale } from "./lib.js";

export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/** The `/api/analysis` URL, optionally for one path under the served root. */
export function analysisUrl(path) {
  return path ? `/api/analysis?path=${encodeURIComponent(path)}` : "/api/analysis";
}

/** A slope with its interval: `+0.012 [-0.004, +0.028]`, or `n/a`. */
export function fmtSlope(slope, digits = 3) {
  if (!slope) return "n/a";
  return `${fmt(slope.value, digits, true)} [${fmt(slope.low, digits, true)}, ${fmt(slope.high, digits, true)}]`;
}

/** A score and its standard error: `+0.412 ± 0.015`, or `–`. */
export function fmtScore(score, digits = 3) {
  return score ? `${fmt(score.mean, digits, true)} ± ${fmt(score.std_error, digits)}` : "–";
}

/** Colour class of a verdict code: `good`, `warn` or `info`. */
export function verdictTone(code) {
  switch (code) {
    case "still_improving":
      return "info";
    case "converged":
    case "no_clear_limit":
      return "good";
    default:
      return "warn";
  }
}

/** CSS colour of an opponent trend class. */
export function trendColour(cls) {
  switch (cls) {
    case "rising":
      return "var(--s3)";
    case "falling":
      return "var(--s2)";
    default:
      return "var(--s6)";
  }
}

/** Stable per-opponent line colours: the palette, cycled. */
export function opponentColour(index) {
  return `var(--s${(index % 6) + 1})`;
}

/**
 * Where the stall marker goes: the generation number at the analysis' stall
 * index (clamped into the series), or null for an empty series.
 */
export function stallMarker(series, analysis) {
  const n = series.generation.length;
  if (n === 0) return null;
  const index = Math.min(Math.max(0, analysis.stall_generation), n - 1);
  return { index, generation: series.generation[index] };
}

/** The +-`se` band around the smoothed curve, as parallel arrays. */
export function noiseBand(series, se) {
  return { lo: series.smoothed.map((v) => v - se), hi: series.smoothed.map((v) => v + se) };
}

/** Rows of the key-numbers table: `[label, value, note]`. */
export function keyRows(a) {
  const gap = a.curse.reeval_minus_heldout;
  return [
    ["Generations", String(a.generations), a.skipped_lines > 0 ? `${a.skipped_lines} unreadable line(s) skipped` : ""],
    ["Stall generation", `${a.stall_generation} of ${Math.max(0, a.generations - 1)}`, "smoothed curve stays within one noise unit of its peak from here"],
    ["Peak (smoothed)", fmt(a.peak_smoothed, 3, true), ""],
    ["Final re-evaluation", fmtScore(a.final_reeval), ""],
    ["Final held-out", fmtScore(a.final_heldout), ""],
    ["Noise (single re-evaluation se)", fmt(a.noise.single_se, 4), a.noise.change_sd === null ? "" : `generation-to-generation sd ${fmt(a.noise.change_sd, 4)}`],
    ["Trend, whole run (per 100 gens)", fmtSlope(a.trend.overall), ""],
    ["Trend, last third (per 100 gens)", fmtSlope(a.trend.last_third), ""],
    ["Curse: re-evaluation minus held-out", gap ? fmt(gap.value, 3, true) : "n/a", gap && gap.in_heldout_se !== null ? `${fmt(gap.in_heldout_se, 1, true)} held-out se` : ""],
    ["Curse: training minus re-evaluation", fmt(a.curse.train_minus_reeval, 3, true), "mean over generations"],
  ];
}

/** One row per run for the comparison table (only useful for several runs). */
export function comparisonRows(runs) {
  return runs.map((r) => ({
    label: r.label,
    generations: r.analysis.generations,
    stall: r.analysis.stall_generation,
    finalReeval: r.analysis.final_reeval ? r.analysis.final_reeval.mean : null,
    lastThird: r.analysis.trend.last_third ? r.analysis.trend.last_third.value : null,
    codes: r.analysis.verdict.map((v) => v.code),
  }));
}

/**
 * Chart specs for one run. Each is `{ xs, series: [{ name, color, values,
 * band?, dashed? }], vlines: [{ x, label }], ref? }`; `chartSvg` draws them.
 */
export function championSpec(run) {
  const { series, analysis } = run;
  const marker = stallMarker(series, analysis);
  return {
    xs: series.generation,
    series: [
      { name: "noise band", color: "var(--s1)", values: [], band: noiseBand(series, analysis.noise.single_se) },
      { name: "re-evaluation", color: "var(--s6)", values: series.reeval_mean },
      { name: "smoothed", color: "var(--s1)", values: series.smoothed },
      { name: "training fitness", color: "var(--s5)", values: series.train_fitness, dashed: true },
    ],
    vlines: marker ? [{ x: marker.generation, label: `stall ${marker.generation}` }] : [],
  };
}

export function opponentsSpec(run) {
  const { series } = run;
  return {
    xs: series.generation,
    ref: 0,
    series: series.opponents.map((o, i) => ({ name: o.name, color: opponentColour(i), values: o.means })),
    vlines: [],
  };
}

const W = 900;
const M = { l: 48, r: 12, t: 10, b: 26 };

/** An SVG string for a chart spec (no DOM needed). */
export function chartSvg(spec, height = 280) {
  const { xs, series } = spec;
  if (xs.length === 0) return '<p class="muted">no data</p>';
  const values = [];
  for (const s of series) {
    values.push(...s.values.filter((v) => v !== null && v !== undefined && Number.isFinite(v)));
    if (s.band) values.push(...s.band.lo, ...s.band.hi);
  }
  if (spec.ref !== undefined) values.push(spec.ref);
  const range = extent(values);
  const y = niceScale(range.min, range.max);
  const x0 = xs[0];
  const x1 = xs[xs.length - 1];
  const sx = linearScale(x0, x1 === x0 ? x0 + 1 : x1, M.l, W - M.r);
  const sy = linearScale(y.min, y.max, height - M.b, M.t);
  const parts = [];
  for (const t of y.ticks) {
    parts.push(`<line class="grid" x1="${M.l}" x2="${W - M.r}" y1="${sy(t)}" y2="${sy(t)}"/>`);
    parts.push(`<text x="${M.l - 6}" y="${sy(t) + 4}" text-anchor="end">${fmt(t, 2)}</text>`);
  }
  const xt = x1 === x0 ? [x0] : niceScale(x0, x1, 6).ticks.filter((t) => Number.isInteger(t) && t >= x0 && t <= x1);
  for (const t of xt) parts.push(`<text x="${sx(t)}" y="${height - 8}" text-anchor="middle">${t}</text>`);
  if (spec.ref !== undefined) parts.push(`<line class="ref" x1="${M.l}" x2="${W - M.r}" y1="${sy(spec.ref)}" y2="${sy(spec.ref)}"/>`);
  for (const s of series) {
    if (s.band) {
      const hi = xs.map((x, i) => [sx(x), sy(s.band.hi[i])]);
      const lo = xs.map((x, i) => [sx(x), sy(s.band.lo[i])]);
      parts.push(`<path d="${bandPath(hi, lo)}" fill="var(--band)" stroke="none"/>`);
    }
    const pts = xs
      .map((x, i) => [x, s.values[i]])
      .filter(([, v]) => v !== null && v !== undefined && Number.isFinite(v))
      .map(([x, v]) => [sx(x), sy(v)]);
    if (pts.length > 0) {
      parts.push(`<path d="${linePath(pts)}" fill="none" stroke="${s.color}" stroke-width="1.8" ${s.dashed ? 'stroke-dasharray="5 4"' : ""}/>`);
    }
  }
  for (const v of spec.vlines ?? []) {
    parts.push(`<line class="stall" x1="${sx(v.x)}" x2="${sx(v.x)}" y1="${M.t}" y2="${height - M.b}"/>`);
    parts.push(`<text x="${sx(v.x) + 4}" y="${M.t + 10}">${esc(v.label)}</text>`);
  }
  return `<svg viewBox="0 0 ${W} ${height}" role="img">${parts.join("")}</svg>`;
}

/** Legend items (`{ name, color }`) of a spec, without the band. */
export function legend(spec) {
  return spec.series.filter((s) => !s.band).map((s) => ({ name: s.name, color: s.color }));
}
