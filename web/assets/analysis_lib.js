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

/**
 * Colour class of a verdict code: `good`, `warn` or `info`. A converged
 * population is not good news (it means little variety is left), so it is
 * neutral, like a run that is still improving.
 */
export function verdictTone(code) {
  switch (code) {
    case "still_improving":
    case "converged":
      return "info";
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
 * Where the stall marker goes: `analysis.stall_generation` is a generation
 * NUMBER (the run need not start at 0 or be gap free), `analysis.stall_index`
 * its position in the series. Returns `{ index, generation }`, or null for an
 * empty series.
 */
export function stallMarker(series, analysis) {
  const n = series.generation.length;
  if (n === 0) return null;
  const index = Math.min(Math.max(0, analysis.stall_index), n - 1);
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
    [
      "Stall generation",
      `${a.stall_generation} (last generation ${a.last_generation})`,
      a.stall_fallback ? "the run ended below its smoothed peak band" : "smoothed curve stays within one noise unit of its peak from here",
    ],
    ["Peak (smoothed)", fmt(a.peak_smoothed, 3, true), `at generation ${a.peak_generation}; final smoothed ${fmt(a.final_smoothed, 3, true)}`],
    ["Final re-evaluation", fmtScore(a.final_reeval), ""],
    ["Final held-out", fmtScore(a.final_heldout), ""],
    ["Noise (single re-evaluation se)", fmt(a.noise.single_se, 4), a.noise.change_sd === null ? "" : `generation-to-generation sd ${fmt(a.noise.change_sd, 4)}`],
    ["Trend, whole run (per 100 gens)", fmtSlope(a.trend.overall), ""],
    ["Trend, last third (per 100 gens)", fmtSlope(a.trend.last_third), ""],
    ["Curse: re-evaluation minus held-out", gap ? fmt(gap.value, 3, true) : "n/a", gap && gap.in_combined_se !== null ? `${fmt(gap.in_combined_se, 1, true)} combined se` : ""],
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

/** Opponent rows of the table: `[name, first, last, lastSlope, wholeSlope, class]` as strings. */
export function opponentRows(a) {
  return a.opponents.map((o) => [o.name, fmt(o.first_third_mean, 3, true), fmt(o.last_third_mean, 3, true), fmtSlope(o.last_third_slope), fmtSlope(o.overall_slope), o.class ?? "n/a"]);
}

// ---- HTML strings (escaped here, so the page only assigns them) ----------

export function swatchesHtml(items) {
  return items.map((i) => `<span class="item"><span class="swatch" style="background:${esc(i.color)}"></span>${esc(i.name)}</span>`).join(" ");
}

export function verdictHtml(run) {
  const items = run.analysis.verdict
    .map((v) => `<li class="verdict ${verdictTone(v.code)}"><code>${esc(v.code)}</code> ${esc(v.message)}</li>`)
    .join("");
  return `<ul class="verdicts">${items}</ul>`;
}

export function keyNumbersHtml(run) {
  const rows = keyRows(run.analysis)
    .map(([k, v, note]) => `<tr><th>${esc(k)}</th><td>${esc(v)}</td><td class="muted">${esc(note)}</td></tr>`)
    .join("");
  return `<table class="data">${rows}</table>`;
}

export function opponentsTableHtml(run) {
  const rows = opponentRows(run.analysis);
  if (rows.length === 0) return "";
  const body = rows
    .map(([name, first, last, lastSlope, wholeSlope, cls]) => `<tr><th>${esc(name)}</th><td>${esc(first)}</td><td>${esc(last)}</td><td>${esc(lastSlope)}</td><td>${esc(wholeSlope)}</td><td style="color:${trendColour(cls)}">${esc(cls)}</td></tr>`)
    .join("");
  return `<table class="data"><tr><th>opponent</th><th>first third</th><th>last third</th><th>last-third slope / 100 gens</th><th>whole-run slope / 100 gens</th><th>trend (last third)</th></tr>${body}</table>`;
}

export function compareHtml(runs) {
  const rows = comparisonRows(runs)
    .map(
      (r, i) =>
        `<tr class="pick" data-index="${i}"><th>${esc(r.label)}</th><td>${r.generations}</td><td>${r.stall}</td><td>${fmt(r.finalReeval, 3, true)}</td><td>${fmt(r.lastThird, 3, true)}</td><td>${r.codes.map((c) => `<code>${esc(c)}</code>`).join(" ")}</td></tr>`,
    )
    .join("");
  return `<table class="data"><tr><th>run</th><th>generations</th><th>stall generation</th><th>final re-eval</th><th>last-third trend</th><th>verdict</th></tr>${rows}</table>`;
}

export function pickerHtml(runs) {
  return runs.map((r, i) => `<option value="${i}">${esc(r.label)}</option>`).join("");
}

/** The note about runs of a set that could not be read, or an empty string. */
export function skippedHtml(skipped) {
  if (!skipped || skipped.length === 0) return "";
  return `<p class="muted">${skipped.length} run(s) could not be read and were left out: ${skipped.map(esc).join("; ")}</p>`;
}

/**
 * Chart specs for one run. Each is `{ xs, series: [{ name, color, values,
 * band?, dashed? }], vlines: [{ x, label }], ref? }`; `chartSvg` draws them.
 */
export function championSpec(run) {
  const { series, analysis } = run;
  const marker = stallMarker(series, analysis);
  const heldout = series.heldout_mean ?? [];
  return {
    xs: series.generation,
    series: [
      { name: "noise band", color: "var(--s1)", values: [], band: noiseBand(series, analysis.noise.single_se) },
      { name: "re-evaluation", color: "var(--s6)", values: series.reeval_mean },
      { name: "smoothed", color: "var(--s1)", values: series.smoothed },
      { name: "training fitness", color: "var(--s5)", values: series.train_fitness, dashed: true },
      // Held-out score of the generations that set a new best: the gap to the
      // re-evaluation curve at the same generation is the winner's curse.
      ...(heldout.some((v) => v !== null && v !== undefined)
        ? [{ name: "held-out (new best)", color: "var(--s3)", values: heldout, markers: true }]
        : []),
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
    for (const v of s.values) if (v !== null && v !== undefined && Number.isFinite(v)) values.push(v);
    if (s.band) {
      for (const v of s.band.lo) values.push(v);
      for (const v of s.band.hi) values.push(v);
    }
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
    if (s.markers) {
      for (const [px, py] of pts) parts.push(`<circle class="marker" cx="${px}" cy="${py}" r="4" fill="${s.color}" stroke="var(--bg, #fff)" stroke-width="1"/>`);
    } else if (pts.length > 0) {
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
