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

/**
 * The cheap luck-adjusted estimate (`--skill-score estimate` or `both`), best first, empty
 * otherwise. It is fitted on round 1 only, so its plain column is the round-1 mean, not the
 * whole-run mean of the other tables.
 */
export function estimateRows(data) {
  const rows = data?.extended?.skill?.estimate?.strategies ?? [];
  return rows
    .map((s) => ({
      name: s.name,
      estimate: pm(s.skill_score_estimate, 3, true),
      plain: pm(s.plain_mean_role_score_round1, 3, true),
      luck: s.luck_share == null ? "–" : `${Math.round(100 * s.luck_share)}%`,
      best: s.skill_score_estimate?.value ?? -Infinity,
    }))
    .sort((a, b) => b.best - a.best);
}

/**
 * When both modes ran: the estimate next to the duplicate round-1 score of the same strategy,
 * how far apart they are in standard errors, and the file's own verdict. Null otherwise.
 */
export function agreement(data) {
  const c = data?.extended?.skill?.comparison;
  if (!c?.strategies?.length) return null;
  return {
    verdict: c.verdict ?? "",
    rows: c.strategies.map((s) => ({
      name: s.name,
      duplicate: pm(s.skill_score_duplicate_round1, 3, true),
      estimate: pm(s.skill_score_estimate, 3, true),
      gap: s.difference_in_se == null ? "–" : `${fmt(s.difference_in_se, 1, true)} SE`,
      removed: s.variance_removed_ratio == null ? "–" : `${Math.round(100 * s.variance_removed_ratio)}%`,
    })),
  };
}

// The statistics this page draws, by catalogue id. The catalogue also holds the statistics
// of `cli analyse-run`, which belong to the run-analysis page, so only these are explained.
const BASE_IDS = ["matches_played", "avg_rank", "rank_distribution", "mean_role_score", "strength_rating"];

/** The catalogue ids that explain what `data` shows, in reading order. */
export function explainIds(data) {
  const skill = data?.extended?.skill;
  const ids = [...BASE_IDS];
  if (skill?.duplicate) ids.push("skill_score_duplicate", "plain_mean_role_score");
  if (skill?.estimate) ids.push("skill_score_estimate");
  if (skill?.duplicate || skill?.estimate) ids.push("luck_share");
  if (skill?.comparison) ids.push("estimator_agreement", "difference_in_se");
  return ids;
}

/** Difference b - a of two numbers with standard errors, in combined standard errors. */
function diff(a, b, digits, signed) {
  if (!a || !b) return { a: pm(a, digits, signed), b: pm(b, digits, signed), diff: "–", z: null };
  const d = b.value - a.value;
  const se = Math.hypot(a.std_error ?? 0, b.std_error ?? 0);
  const z = se > 0 ? d / se : null;
  return {
    a: pm(a, digits, signed),
    b: pm(b, digits, signed),
    diff: `${fmt(d, digits, true)} ± ${fmt(se, digits)}`,
    z,
  };
}

/** "clear" when the difference is more than two standard errors, else "within noise". */
export const verdictOf = (z) => (z == null ? "" : Math.abs(z) > 2 ? "clear" : "within noise");

const GETTERS = {
  score: { title: "Mean score", digits: 3, signed: true, get: (s) => s.mean_role_score, higher: true },
  rank: { title: "Average place", digits: 2, signed: false, get: (s) => s.avg_rank, higher: false },
  rating: { title: "Strength rating", digits: 0, signed: true, get: (s) => s.strength_rating, higher: true },
};

/**
 * The same strategies in two results files, side by side. Rows come from `a`, then the
 * strategies only `b` has. `notes` says what does not line up (names, matches, tables).
 */
export function compareTables(a, b) {
  const byA = a?.extended?.by_strategy;
  const byB = b?.extended?.by_strategy;
  if (!byA || !byB) throw new Error("both files need extended.by_strategy: not the results of a plain cli run?");
  const names = [...Object.keys(byA), ...Object.keys(byB).filter((n) => !(n in byA))];
  const notes = [];
  const onlyA = Object.keys(byA).filter((n) => !(n in byB));
  const onlyB = Object.keys(byB).filter((n) => !(n in byA));
  if (onlyA.length) notes.push(`Only in the first file: ${onlyA.join(", ")}.`);
  if (onlyB.length) notes.push(`Only in the second file: ${onlyB.join(", ")}.`);
  if (onlyA.length || onlyB.length) notes.push("The tables differ in who sat at them, and a score depends on the opponents, so those differences are not only the strategy.");
  const ma = a.statistics?.matches_played;
  const mb = b.statistics?.matches_played;
  if (ma != null && mb != null && ma !== mb) notes.push(`The files differ in size (${ma} and ${mb} matches); the standard errors reflect that.`);
  const tables = [];
  const add = (title, digits, signed, higher, pick) => {
    const rows = names.map((name) => ({ name, ...diff(pick(a, name), pick(b, name), digits, signed), higher }));
    if (rows.some((r) => r.z != null || r.diff !== "–")) tables.push({ title, rows });
  };
  for (const g of Object.values(GETTERS)) add(g.title, g.digits, g.signed, g.higher, (d, n) => g.get(d.extended.by_strategy[n] ?? {}));
  const dup = (d, n) => d.extended?.skill?.duplicate?.strategies?.find((s) => s.name === n)?.skill_score_duplicate;
  const est = (d, n) => d.extended?.skill?.estimate?.strategies?.find((s) => s.name === n)?.skill_score_estimate;
  add("Skill score (duplicate deals)", 3, true, true, dup);
  add("Skill score (estimate, round 1)", 3, true, true, est);
  const hasDup = (d) => !!d.extended?.skill?.duplicate;
  const hasEst = (d) => !!d.extended?.skill?.estimate;
  if (hasDup(a) !== hasDup(b)) notes.push("Only one file has duplicate-deal skill scores, so that table is empty.");
  if (hasEst(a) !== hasEst(b)) notes.push("Only one file has a skill estimate, so that table is empty.");
  notes.push("Strategies are matched by name; a difference is called clear when it is more than two combined standard errors.");
  return { tables, notes };
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

export function estimateHtml(rows) {
  const body = rows.map((r) => `<tr><td>${esc(r.name)}</td><td>${r.estimate}</td><td>${r.plain}</td><td>${r.luck}</td></tr>`).join("");
  return table(["Strategy", "Skill score (estimate)", "Plain score, round 1", "Luck share"], body);
}

export function agreementHtml(ag) {
  const body = ag.rows
    .map((r) => `<tr><td>${esc(r.name)}</td><td>${r.duplicate}</td><td>${r.estimate}</td><td>${r.gap}</td><td>${r.removed}</td></tr>`)
    .join("");
  return `${table(["Strategy", "Duplicate, round 1", "Estimate", "Estimate minus duplicate", "Luck removed"], body)}<p class="muted">${esc(ag.verdict)}</p>`;
}

/** The comparison tables of `compareTables`, each difference marked clear or within noise. */
export function compareHtml(cmp) {
  const tables = cmp.tables
    .map((t) => {
      const body = t.rows
        .map((r) => {
          const v = verdictOf(r.z);
          // Mark the side that is better only when the difference is clear.
          const better = v === "clear" ? ((r.z > 0) === r.higher ? "second file better" : "first file better") : v;
          return `<tr><td>${esc(r.name)}</td><td>${r.a}</td><td>${r.b}</td><td>${r.diff}</td><td>${esc(better)}</td></tr>`;
        })
        .join("");
      return `<h3>${esc(t.title)}</h3>${table(["Strategy", "First file", "Second file", "Second minus first", "Difference"], body)}`;
    })
    .join("");
  return `${tables}${cmp.notes.map((n) => `<p class="muted">${esc(n)}</p>`).join("")}`;
}

/**
 * What the statistics mean, from the catalogue stored in the results file. With `ids` only
 * those entries, in that order; entries of the run analysis never belong to this page.
 */
export function explainHtml(catalog, ids) {
  const entries = (catalog ?? []).filter((c) => c.scope !== "run_analysis");
  const picked = ids ? ids.map((id) => entries.find((c) => c.id === id)).filter(Boolean) : entries;
  return picked
    .map((c) => `<details><summary>${esc(c.name)}</summary><p>${esc(c.meaning)}</p><p class="muted">${esc(c.formula ?? "")}</p></details>`)
    .join("");
}
