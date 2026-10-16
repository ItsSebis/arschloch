// Run with: node --test web/tests
import assert from "node:assert/strict";
import test from "node:test";
import {
  agreement, agreementHtml, compareHtml, compareTables, estimateHtml, estimateRows, explainHtml, explainIds, esc,
  placeColour, skillRows, strategyRows, tableHtml, verdictOf,
} from "../assets/results_lib.js";

const stat = (value, std_error = 0.01) => ({ value, std_error, n: 10 });
const data = {
  statistics: { matches_played: 40 },
  extended: {
    by_strategy: {
      "Neat(<champion>)": {
        avg_rank: stat(1.56), mean_role_score: stat(0.628), strength_rating: stat(193, 4), rounds: 200,
        rank_distribution: [56.5, 24.5, 13.0, 6.0],
      },
      LowestLegal: {
        avg_rank: stat(2.98), mean_role_score: stat(-0.318), strength_rating: stat(-96, 5), rounds: 200,
        rank_distribution: [13.2, 20.0, 27.0, 39.8],
      },
    },
    skill: {
      mode: "duplicate",
      duplicate: {
        strategies: [
          { name: "LowestLegal", skill_score_duplicate: stat(-0.318), plain_mean_role_score: stat(-0.31), luck_share: 0.42 },
          { name: "Neat(<champion>)", skill_score_duplicate: stat(0.628), plain_mean_role_score: stat(0.63), luck_share: 0.1 },
        ],
      },
    },
  },
  statistics_catalog: [
    { id: "avg_rank", name: "Average place", meaning: "Mean finishing place, 1 is best.", formula: "sum / rounds" },
  ],
};

test("strategies are ordered best first and numbers keep their standard errors", () => {
  const rows = strategyRows(data);
  assert.deepEqual(rows.map((r) => r.name), ["Neat(<champion>)", "LowestLegal"]);
  assert.equal(rows[0].rounds, 200);
  assert.equal(rows[0].places.length, 4);
  assert.equal(rows[0].avgRank, "1.56 ± 0.01");
  assert.equal(rows[0].score, "+0.628 ± 0.010");
  assert.equal(rows[1].score, "-0.318 ± 0.010");
});

test("the table shows every strategy, escapes names and draws one bar segment per place", () => {
  const html = tableHtml(strategyRows(data));
  assert.ok(html.includes("Neat(&lt;champion&gt;)"));
  assert.ok(!html.includes("<champion>"));
  assert.equal((html.match(/class="seg"/g) ?? []).length, 8);
  assert.ok(html.includes("width:56.5%"));
});

test("skill rows exist only when a duplicate skill score was computed", () => {
  const rows = skillRows(data);
  assert.deepEqual(rows.map((r) => r.name), ["Neat(<champion>)", "LowestLegal"], "best first");
  assert.equal(rows[0].skill, "+0.628 ± 0.010");
  assert.equal(rows[0].luck, "10%");
  assert.deepEqual(skillRows({ statistics: {}, extended: { by_strategy: {} } }), []);
});

test("the place colours run from green (first) to red (last)", () => {
  assert.notEqual(placeColour(0, 4), placeColour(3, 4));
  assert.match(placeColour(0, 4), /^hsl\(/);
  assert.match(placeColour(0, 1), /^hsl\(/, "a single place does not divide by zero");
});

test("the explanations come from the catalogue and are escaped", () => {
  const html = explainHtml(data.statistics_catalog);
  assert.ok(html.includes("Average place"));
  assert.ok(html.includes("Mean finishing place, 1 is best."));
  assert.equal(esc(`<a href="x">&</a>`), "&lt;a href=&quot;x&quot;&gt;&amp;&lt;/a&gt;");
  assert.equal(explainHtml(undefined), "");
});

test("a file that is not the results of a plain run is refused in words, not shown empty", () => {
  assert.throws(() => strategyRows({}), /by_strategy/);
  assert.throws(() => strategyRows({ extended: {} }), /not the results of a plain/);
});

test("every inserted value is escaped, and missing numbers read as a dash", () => {
  const hostile = structuredClone(data);
  hostile.extended.by_strategy.LowestLegal.rounds = "<img src=x onerror=alert(1)>";
  assert.ok(!tableHtml(strategyRows(hostile)).includes("<img"));
  const missing = structuredClone(data);
  delete missing.extended.by_strategy.LowestLegal.rounds;
  assert.ok(!tableHtml(strategyRows(missing)).includes("undefined"));
  const noLuck = structuredClone(data);
  delete noLuck.extended.skill.duplicate.strategies[0].luck_share;
  assert.ok(skillRows(noLuck).every((r) => !r.luck.includes("NaN")));
});

const withEstimate = () => {
  const d = structuredClone(data);
  d.extended.skill = {
    mode: "both",
    duplicate: d.extended.skill.duplicate,
    estimate: {
      strategies: [
        { name: "LowestLegal", skill_score_estimate: stat(-0.3), plain_mean_role_score_round1: stat(-0.33), luck_share: 0.2 },
        { name: "Neat(<champion>)", skill_score_estimate: stat(0.6), plain_mean_role_score_round1: stat(0.62), luck_share: 0.05 },
      ],
    },
    comparison: {
      strategies: [{ name: "Neat(<champion>)", skill_score_duplicate_round1: stat(0.62), skill_score_estimate: stat(0.6), difference_in_se: -0.4, variance_removed_ratio: 0.5 }],
      verdict: "estimator removes 50% <b>of</b> the luck",
    },
  };
  return d;
};

test("the estimate is drawn best first, with its round-1 plain score", () => {
  const rows = estimateRows(withEstimate());
  assert.deepEqual(rows.map((r) => r.name), ["Neat(<champion>)", "LowestLegal"]);
  assert.equal(rows[0].estimate, "+0.600 ± 0.010");
  assert.equal(rows[0].plain, "+0.620 ± 0.010");
  assert.equal(rows[0].luck, "5%");
  assert.deepEqual(estimateRows(data), [], "a duplicate-only run has no estimate");
  assert.ok(!estimateHtml(rows).includes("<champion>"));
});

test("the estimate next to the duplicate score, with the verdict escaped", () => {
  assert.equal(agreement(data), null);
  const ag = agreement(withEstimate());
  assert.equal(ag.rows[0].gap, "-0.4 SE");
  assert.equal(ag.rows[0].removed, "50%");
  const html = agreementHtml(ag);
  assert.ok(html.includes("&lt;b&gt;of"));
  assert.ok(!html.includes("<b>"));
});

test("only the explanations of what the page shows are listed, never the run analysis", () => {
  const plain = explainIds({ extended: { by_strategy: {} } });
  assert.ok(plain.includes("avg_rank") && !plain.includes("skill_score_estimate"));
  const both = explainIds(withEstimate());
  for (const id of ["skill_score_duplicate", "skill_score_estimate", "luck_share", "estimator_agreement"]) assert.ok(both.includes(id), id);
  assert.ok(!both.some((id) => id.startsWith("analysis_")));
  const catalog = [
    { id: "avg_rank", name: "Average place", scope: "per_strategy", meaning: "m", formula: "f" },
    { id: "analysis_trend", name: "Trend of a run", scope: "run_analysis", meaning: "m", formula: "f" },
    { id: "skill_score_estimate", name: "Estimate", scope: "per_strategy", meaning: "m", formula: "f" },
  ];
  const html = explainHtml(catalog, plain);
  assert.ok(html.includes("Average place"));
  assert.ok(!html.includes("Estimate"), "not in this file");
  assert.ok(!explainHtml(catalog).includes("Trend of a run"), "run-analysis entries never show");
});

test("two files are compared by strategy name with the difference in standard errors", () => {
  const b = structuredClone(data);
  b.extended.by_strategy.LowestLegal.mean_role_score = stat(-0.1, 0.01);
  b.extended.by_strategy["Neat(<champion>)"].mean_role_score = stat(0.629, 0.01);
  b.extended.by_strategy.Extra = { mean_role_score: stat(0.2) };
  b.statistics.matches_played = 80;
  const cmp = compareTables(data, b);
  const score = cmp.tables.find((t) => t.title === "Mean score");
  assert.deepEqual(score.rows.map((r) => r.name), ["Neat(<champion>)", "LowestLegal", "Extra"], "first file's order, then the extras");
  const low = score.rows.find((r) => r.name === "LowestLegal");
  assert.equal(low.diff, "+0.218 ± 0.014");
  assert.equal(verdictOf(low.z), "clear");
  assert.equal(verdictOf(score.rows[0].z), "within noise");
  assert.equal(score.rows[2].diff, "–", "present in one file only");
  assert.ok(cmp.notes.some((n) => /Only in the second file: Extra/.test(n)));
  assert.ok(cmp.notes.some((n) => /40 and 80 matches/.test(n)));
  assert.ok(!cmp.tables.some((t) => t.title.startsWith("Skill score (estimate")), "no estimate in either file");
});

test("the comparison names the better file only when the difference is clear, and respects the direction", () => {
  const b = structuredClone(data);
  b.extended.by_strategy.LowestLegal.mean_role_score = stat(-0.1, 0.01);
  b.extended.by_strategy.LowestLegal.avg_rank = stat(2.5, 0.01);
  const html = compareHtml(compareTables(data, b));
  const row = (title) => html.split(`<h3>${title}</h3>`)[1].split("</table>")[0].split("<tr>").find((r) => r.includes("LowestLegal"));
  assert.ok(row("Mean score").includes("second file better"));
  assert.ok(row("Average place").includes("second file better"), "a lower average place is better");
  assert.ok(html.includes("within noise"));
});

test("skill tables are compared only where both files have them; a one-sided file says so", () => {
  const cmp = compareTables(withEstimate(), data);
  assert.ok(cmp.tables.some((t) => t.title === "Skill score (duplicate deals)"));
  assert.ok(!cmp.tables.some((t) => t.title.startsWith("Skill score (estimate")), "second file has none");
  assert.ok(cmp.notes.some((n) => /skill estimate/.test(n)));
  const same = compareTables(withEstimate(), withEstimate());
  assert.ok(same.tables.some((t) => t.title.startsWith("Skill score (estimate")));
  assert.throws(() => compareTables(data, {}), /both files/);
});

test("an unknown standard error (null) is not 0: no plus-minus and no verdict", () => {
  const a = structuredClone(data);
  const b = structuredClone(data);
  a.extended.by_strategy.LowestLegal.mean_role_score = { value: -0.3, std_error: null };
  b.extended.by_strategy.LowestLegal.mean_role_score = { value: 0.5 };
  const row = compareTables(a, b).tables.find((t) => t.title === "Mean score").rows.find((r) => r.name === "LowestLegal");
  assert.equal(row.z, null);
  assert.equal(row.diff, "+0.800");
  assert.equal(row.a, "-0.300");
  assert.equal(verdictOf(row.z), "");
  const nan = structuredClone(a);
  nan.extended.by_strategy.LowestLegal.mean_role_score.std_error = NaN;
  assert.equal(compareTables(nan, b).tables[0].rows.find((r) => r.name === "LowestLegal").z, null);
  assert.equal(verdictOf(NaN), "");
});

test("the comparison always says what it assumes, and the rating gets no verdict", () => {
  const b = structuredClone(data);
  b.extended.by_strategy.LowestLegal.strength_rating = stat(300, 1);
  const cmp = compareTables(data, b);
  assert.ok(cmp.notes.some((n) => /relative to each file's own field.*same table size and rules/.test(n)));
  assert.ok(cmp.notes.some((n) => /independent.*by chance/.test(n)));
  const html = compareHtml(cmp);
  const rating = html.split("<h3>Strength rating</h3>")[1].split("</table>")[0];
  assert.ok(!/better|clear|within noise/.test(rating.replace(/<th>.*?<\/th>/g, "")), "no verdict for the rating");
  assert.ok(cmp.notes.every((n) => !/is empty/.test(n)));
});

test("the agreement shows n/a where the file wrote 0.0 for an undefined value", () => {
  const d = withEstimate();
  d.extended.skill.comparison.strategies[0].difference_in_se = 0;
  d.extended.skill.comparison.strategies[0].variance_removed_ratio = 0;
  const ag = agreement(d);
  assert.equal(ag.rows[0].gap, "n/a");
  assert.equal(ag.rows[0].removed, "n/a");
  assert.ok(agreementHtml(ag).includes("Share of duplicate's luck removed"));
});

test("a strategy named like an Object property is looked up as a name", () => {
  const a = structuredClone(data);
  const b = structuredClone(data);
  b.extended.by_strategy.constructor = { mean_role_score: stat(0.2) };
  const cmp = compareTables(a, b);
  const score = cmp.tables.find((t) => t.title === "Mean score");
  assert.deepEqual(score.rows.map((r) => r.name).slice(-1), ["constructor"]);
  assert.equal(score.rows.at(-1).diff, "–");
  assert.ok(cmp.notes.some((n) => /Only in the second file: constructor/.test(n)));
});
