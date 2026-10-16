// Run with: node --test web/tests
import assert from "node:assert/strict";
import test from "node:test";
import { explainHtml, esc, placeColour, skillNote, skillRows, strategyRows, tableHtml } from "../assets/results_lib.js";

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

test("a luck-adjusted estimate that is in the file but not drawn is said so", () => {
  assert.equal(skillNote(data), "", "duplicate results are all that is drawn");
  const estimate = { extended: { skill: { mode: "estimate", estimate: { strategies: [] } } } };
  assert.match(skillNote(estimate), /estimate/);
  assert.match(skillNote({ extended: { skill: { mode: "both" } } }), /estimate/);
  assert.equal(skillNote({ extended: {} }), "");
});
