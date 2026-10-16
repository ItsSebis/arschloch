// Pure helpers for the simulation-results page: no DOM, no network, so they can be
// unit-tested in Node (see tests/results.test.mjs) and reused by results.js.
import { fmt } from "./lib.js";

export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/** A number with its standard error: `+0.628 ± 0.010`, or `–`. */
const pm = (stat, digits, signed = false) => (stat ? `${fmt(stat.value, digits, signed)} ± ${fmt(stat.std_error, digits)}` : "–");

/** Green for the first place, red for the last. */
export function placeColour(index, count) {
  const t = count > 1 ? index / (count - 1) : 0;
  return `hsl(${Math.round(120 - 120 * t)} 55% 45%)`;
}

/** One row per strategy, best mean score first. */
export function strategyRows(data) {
  const by = data?.extended?.by_strategy;
  if (!by) throw new Error("no extended.by_strategy in this file: not the results of a plain cli run?");
  return Object.entries(by)
    .map(([name, s]) => ({
      name,
      rounds: s.rounds ?? null,
      avgRank: pm(s.avg_rank, 2),
      score: pm(s.mean_role_score, 3, true),
      rating: pm(s.strength_rating, 0, true),
      places: s.rank_distribution ?? [],
      best: s.mean_role_score?.value ?? -Infinity,
    }))
    .sort((a, b) => b.best - a.best);
}

/** Luck-adjusted scores of a duplicate-deal run, best first (empty otherwise). */
export function skillRows(data) {
  const rows = data?.extended?.skill?.duplicate?.strategies ?? [];
  return rows
    .map((s) => ({
      name: s.name,
      skill: pm(s.skill_score_duplicate, 3, true),
      plain: pm(s.plain_mean_role_score, 3, true),
      luck: s.luck_share == null ? "–" : `${Math.round(100 * s.luck_share)}%`,
      best: s.skill_score_duplicate?.value ?? -Infinity,
    }))
    .sort((a, b) => b.best - a.best);
}

/** Said when the file holds a luck-adjusted estimate that this page does not draw. */
export function skillNote(data) {
  const mode = data?.extended?.skill?.mode;
  return mode === "estimate" || mode === "both"
    ? `This run also has a luck-adjusted estimate (skill score mode: ${mode}); this page shows only the exact duplicate-deal scores.`
    : "";
}

/** The share of rounds finished in each place, as one stacked bar. */
const barHtml = (places) =>
  `<div class="places">${places
    .map((p, i) => `<span class="seg" style="width:${Math.round(p * 100) / 100}%;background:${placeColour(i, places.length)}" title="place ${i + 1}: ${fmt(p, 1)}%"></span>`)
    .join("")}</div>`;

const table = (head, body) =>
  `<table class="data"><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table>`;

export function tableHtml(rows) {
  const body = rows
    .map((r) => `<tr><td>${esc(r.name)}</td><td>${esc(r.rounds ?? "–")}</td><td>${r.avgRank}</td><td>${r.score}</td><td>${r.rating}</td><td>${barHtml(r.places)}</td></tr>`)
    .join("");
  return table(["Strategy", "Rounds", "Average place", "Mean score", "Strength rating", "Places, first to last"], body);
}

export function skillHtml(rows) {
  const body = rows.map((r) => `<tr><td>${esc(r.name)}</td><td>${r.skill}</td><td>${r.plain}</td><td>${r.luck}</td></tr>`).join("");
  return table(["Strategy", "Skill score", "Plain score", "Luck share"], body);
}

/** What every statistic means, from the catalogue stored in the results file. */
export function explainHtml(catalog) {
  return (catalog ?? [])
    .map((c) => `<details><summary>${esc(c.name)}</summary><p>${esc(c.meaning)}</p><p class="muted">${esc(c.formula ?? "")}</p></details>`)
    .join("");
}
