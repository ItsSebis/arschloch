// Cross-language check: the browser's network replay must reproduce the
// scores the Rust implementation recorded for real decisions.
// Needs RUN_DIR (a run directory written by `cli train`); skipped otherwise.
import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import test from "node:test";
import { forwardPass } from "../assets/lib.js";

const runDir = process.env.RUN_DIR;

test("browser replay matches the recorded Rust scores", { skip: !runDir && "RUN_DIR not set" }, () => {
  const decisionDir = `${runDir}/decisions`;
  assert.ok(existsSync(decisionDir), "the run recorded decisions");
  const files = readdirSync(decisionDir).filter((f) => f.endsWith(".json"));
  assert.ok(files.length > 0, "at least one new-best champion has decisions");
  let candidates = 0;
  for (const name of files) {
    const decisions = JSON.parse(readFileSync(`${decisionDir}/${name}`, "utf8"));
    const generation = String(decisions.generation).padStart(4, "0");
    const file = JSON.parse(readFileSync(`${runDir}/gen-${generation}.json`, "utf8"));
    assert.deepEqual(decisions.feature_names, file.feature_names);
    const output = file.genome.nodes.find((n) => n.kind === "Output").id;
    for (const decision of decisions.decisions) {
      for (const candidate of decision.candidates) {
        const { values, raw } = forwardPass(file.genome, candidate.features);
        assert.ok(Math.abs(raw.get(output) - candidate.raw_score) < 1e-9, `${name}: raw score`);
        assert.ok(Math.abs(values.get(output) - candidate.activation) < 1e-9, `${name}: activation`);
        candidates += 1;
      }
    }
  }
  assert.ok(candidates > 0);
});
