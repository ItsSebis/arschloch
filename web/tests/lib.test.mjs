// Run with: node --test web/tests
import assert from "node:assert/strict";
import test from "node:test";
import {
  bandPath, divergingColor, edgeWidth, fmt, forwardPass, formatDuration, histogramOpacities,
  extent, layoutNetwork, linePath, linearScale, nearestIndex, niceScale, setEta, shouldRedraw, speciesColor, stackSpecies,
} from "../assets/lib.js";

const node = (id, kind) => ({ id, kind });
const edge = (innovation, from, to, weight, enabled = true) => ({ innovation, from, to, weight, enabled });

// 2 inputs (0, 1), bias 2, output 3, hidden 4: 0 -> 4 (2.0), 4 -> 3 (-1.5), 1 -> 3 (0.5), 2 -> 3 (0.25)
const genome = {
  num_inputs: 2,
  nodes: [node(0, "Input"), node(1, "Input"), node(2, "Bias"), node(3, "Output"), node(4, "Hidden")],
  connections: [edge(0, 0, 4, 2.0), edge(1, 4, 3, -1.5), edge(2, 1, 3, 0.5), edge(3, 2, 3, 0.25), edge(4, 0, 3, 9, false)],
};

test("fmt handles signs, rounding and non-finite values", () => {
  assert.equal(fmt(0.41234, 3, true), "+0.412");
  assert.equal(fmt(-0.05, 2, true), "-0.05");
  assert.equal(fmt(0, 2, true), "0.00");
  assert.equal(fmt(null), "–");
  assert.equal(fmt(NaN), "–");
  assert.equal(fmt(Infinity), "∞");
});

test("formatDuration matches the terminal's H:MM:SS", () => {
  assert.equal(formatDuration(0), "0:00:00");
  assert.equal(formatDuration(61.4), "0:01:01");
  assert.equal(formatDuration(3725), "1:02:05");
  assert.equal(formatDuration(-3), "0:00:00");
});

test("niceScale rounds outward to tidy ticks and survives flat data", () => {
  const s = niceScale(-0.83, 0.71);
  assert.ok(s.min <= -0.83 && s.max >= 0.71);
  assert.ok(s.ticks.length >= 3 && s.ticks.length <= 12);
  assert.deepEqual(s.ticks, [...s.ticks].sort((a, b) => a - b));
  const flat = niceScale(0.5, 0.5);
  assert.ok(flat.min < 0.5 && flat.max > 0.5);
  assert.deepEqual(niceScale(NaN, 1).ticks, [0, 1]);
});

test("linearScale maps endpoints and degenerate domains", () => {
  const f = linearScale(0, 10, 100, 200);
  assert.equal(f(0), 100);
  assert.equal(f(10), 200);
  assert.equal(f(5), 150);
  assert.equal(linearScale(3, 3, 0, 10)(3), 5);
});

test("linePath and bandPath produce SVG path data", () => {
  assert.equal(linePath([[0, 0], [10, 5]]), "M0.0,0.0 L10.0,5.0");
  const band = bandPath([[0, 0], [10, 0]], [[0, 4], [10, 4]]);
  assert.ok(band.startsWith("M0.0,0.0") && band.endsWith("Z"));
  assert.equal(bandPath([], []), "");
});

test("stackSpecies stacks by id with zeros for absent species", () => {
  const gens = [
    { generation: 0, species: [{ id: 0, size: 10 }] },
    { generation: 1, species: [{ id: 1, size: 4 }, { id: 0, size: 6 }] },
  ];
  const { ids, layers } = stackSpecies(gens);
  assert.deepEqual(ids, [0, 1]);
  assert.deepEqual(layers[0], [{ generation: 0, lower: 0, upper: 10 }, { generation: 1, lower: 0, upper: 6 }]);
  assert.deepEqual(layers[1], [{ generation: 0, lower: 10, upper: 10 }, { generation: 1, lower: 6, upper: 10 }]);
  assert.notEqual(speciesColor(1), speciesColor(2));
  assert.equal(speciesColor(5), speciesColor(5));
});

test("layoutNetwork puts inputs left, output right, hidden between", () => {
  const { nodes, edges } = layoutNetwork(genome);
  const by = Object.fromEntries(nodes.map((n) => [n.id, n]));
  assert.equal(by[0].x, 0);
  assert.equal(by[2].x, 0);
  assert.equal(by[3].x, 1);
  assert.ok(by[4].x > 0 && by[4].x < 1);
  assert.equal(edges.length, 5, "disabled connections are laid out too");
  for (const n of nodes) assert.ok(n.x >= 0 && n.x <= 1 && n.y > 0 && n.y < 1);
  // Column slots are distinct.
  assert.notEqual(by[0].y, by[1].y);
});

test("layoutNetwork handles a genome with no hidden nodes", () => {
  const minimal = {
    num_inputs: 1,
    nodes: [node(0, "Input"), node(1, "Bias"), node(2, "Output")],
    connections: [edge(0, 0, 2, 1), edge(1, 1, 2, 1)],
  };
  const by = Object.fromEntries(layoutNetwork(minimal).nodes.map((n) => [n.id, n]));
  assert.equal(by[0].x, 0);
  assert.equal(by[2].x, 1);
});

test("forwardPass matches the hand-computed network (and ignores disabled edges)", () => {
  const { values, raw } = forwardPass(genome, [0.3, -0.2]);
  const hidden = Math.tanh(2.0 * 0.3);
  const rawOut = -1.5 * hidden + 0.5 * -0.2 + 0.25 * 1; // the disabled 9.0 edge must not count
  assert.ok(Math.abs(values.get(4) - hidden) < 1e-12);
  assert.ok(Math.abs(raw.get(3) - rawOut) < 1e-12);
  assert.ok(Math.abs(values.get(3) - Math.tanh(rawOut)) < 1e-12);
  assert.equal(values.get(2), 1, "bias is always 1");
  assert.equal(values.get(0), 0.3);
  assert.ok(!raw.has(0), "inputs have no pre-activation sum");
});

test("forwardPass gives tanh(0) = 0 to a node with no enabled inputs", () => {
  const g = { ...genome, connections: genome.connections.map((c) => ({ ...c, enabled: false })) };
  const { values } = forwardPass(g, [1, 1]);
  assert.equal(values.get(3), 0);
  assert.equal(values.get(4), 0);
});

test("forwardPass refuses a cyclic genome instead of recursing forever", () => {
  const cyclic = {
    num_inputs: 1,
    nodes: [node(0, "Input"), node(1, "Bias"), node(2, "Output"), node(3, "Hidden"), node(4, "Hidden")],
    connections: [edge(0, 3, 4, 1), edge(1, 4, 3, 1), edge(2, 3, 2, 1)],
  };
  assert.throws(() => forwardPass(cyclic, [0]), /cycle/);
});

test("colour and width helpers stay in range", () => {
  assert.notEqual(divergingColor(-1), divergingColor(1));
  assert.equal(divergingColor(5), divergingColor(1), "clamped");
  assert.ok(edgeWidth(0) < edgeWidth(4) && edgeWidth(4) < edgeWidth(100));
  assert.ok(edgeWidth(1000) <= 3.7);
});

test("histogramOpacities are relative to the largest bucket", () => {
  assert.deepEqual(histogramOpacities([0, 5, 10]), [0, 0.5, 1]);
  assert.deepEqual(histogramOpacities([0, 0]), [0, 0]);
});

test("nearestIndex finds the closest generation to a hover position", () => {
  assert.equal(nearestIndex([0, 10, 20], 14), 1);
  assert.equal(nearestIndex([0, 10, 20], 100), 2);
  assert.equal(nearestIndex([], 3), 0);
});

test("extent finds the min and max without spreading huge arrays", () => {
  assert.deepEqual(extent([3, -1, 2]), { min: -1, max: 3 });
  assert.deepEqual(extent([]), { min: Infinity, max: -Infinity });
  const huge = Array.from({ length: 500000 }, (_, i) => i - 250000);
  assert.deepEqual(extent(huge), { min: -250000, max: 249999 }); // Math.min(...huge) throws a RangeError
  assert.deepEqual(extent([null, 4, undefined, 2]), { min: 2, max: 4 }, "null and undefined are skipped");
});

test("shouldRedraw redraws only when something the page shows has changed", () => {
  const idle = { first: false, newEvents: 0, epochChanged: false, finishedChanged: false, reconnected: false, runChanged: false };
  assert.equal(shouldRedraw(idle), false, "nothing new: leave the DOM (and the user's hover) alone");
  for (const key of ["first", "epochChanged", "finishedChanged", "reconnected", "runChanged"]) {
    assert.equal(shouldRedraw({ ...idle, [key]: true }), true, key);
  }
  assert.equal(shouldRedraw({ ...idle, newEvents: 1 }), true);
});

// The same vectors as sim::training::set (Rust), so terminal and page agree.
test("setEta projects the first run, then uses the finished runs' pace", () => {
  assert.equal(setEta([], 3, 100, 100), 100 + 2 * 200);
  assert.equal(setEta([60, 80], 5, 30, 10), 30 + 2 * 70);
  assert.equal(setEta([60, 80], 3, 30, 10), 30);
});

test("setEta has no answer without a current-run estimate and never goes negative", () => {
  assert.equal(setEta([], 3, null, 0), null);
  assert.equal(setEta([10], 3, undefined, 5), null);
  assert.equal(setEta([10, 10, 10], 2, 4, 1), 4);
  assert.equal(setEta([], 0, 4, 1), 4);
});
