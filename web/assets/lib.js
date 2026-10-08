// Pure helpers for the dashboard: no DOM, no network, so they can be
// unit-tested in Node (see tests/lib.test.mjs) and reused by app.js.

/** Formats a number with fixed decimals; `signed` forces a leading + for positives. */
export function fmt(value, digits = 3, signed = false) {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  if (!Number.isFinite(value)) return value > 0 ? "∞" : "-∞";
  const text = value.toFixed(digits);
  return signed && value > 0 ? "+" + text : text;
}

/** `H:MM:SS` for a duration in seconds. */
export function formatDuration(seconds) {
  const total = Math.max(0, Math.round(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

/** Rounds a data range outward to tidy axis ticks. Returns { min, max, ticks }. */
export function niceScale(min, max, targetTicks = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: 0, max: 1, ticks: [0, 1] };
  if (min === max) {
    const pad = Math.abs(min) > 0 ? Math.abs(min) * 0.1 : 0.5;
    min -= pad;
    max += pad;
  }
  const span = max - min;
  const rough = span / Math.max(1, targetTicks);
  const magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
  const residual = rough / magnitude;
  const step = (residual >= 5 ? 10 : residual >= 2 ? 5 : residual >= 1 ? 2 : 1) * magnitude;
  const niceMin = Math.floor(min / step) * step;
  const niceMax = Math.ceil(max / step) * step;
  const ticks = [];
  for (let t = niceMin; t <= niceMax + step / 2; t += step) ticks.push(Number(t.toFixed(10)));
  return { min: niceMin, max: niceMax, ticks };
}

/** Linear map from [d0, d1] to [r0, r1]. */
export function linearScale(d0, d1, r0, r1) {
  const span = d1 - d0;
  return (x) => (span === 0 ? (r0 + r1) / 2 : r0 + ((x - d0) / span) * (r1 - r0));
}

/** An SVG path through `points` ([[x, y], ...]) in pixel space. */
export function linePath(points) {
  return points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
}

/** A closed band between an upper and a lower polyline (both left to right). */
export function bandPath(upper, lower) {
  if (upper.length === 0) return "";
  const forward = upper.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`);
  const back = [...lower].reverse().map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`);
  return `M${forward.join(" L")} L${back.join(" L")} Z`;
}

/**
 * Stacks per-generation species sizes. `generations` is an array of
 * `{ generation, species: [{ id, size }] }`. Returns `{ ids, layers }`:
 * `layers[k]` is, for species `ids[k]`, an array of `{ generation, lower, upper }`
 * (cumulative counts), species ordered by id so colours and positions stay put.
 */
export function stackSpecies(generations) {
  const ids = [...new Set(generations.flatMap((g) => g.species.map((s) => s.id)))].sort((a, b) => a - b);
  const layers = ids.map(() => []);
  for (const g of generations) {
    const sizes = new Map(g.species.map((s) => [s.id, s.size]));
    let running = 0;
    ids.forEach((id, k) => {
      const size = sizes.get(id) ?? 0;
      layers[k].push({ generation: g.generation, lower: running, upper: running + size });
      running += size;
    });
  }
  return { ids, layers };
}

/** A stable categorical colour for a species id (golden-angle hue steps). */
export function speciesColor(id) {
  const hue = (id * 137.508) % 360;
  return `hsl(${hue.toFixed(0)} 60% 55%)`;
}

/**
 * Lays a genome out in columns: inputs and bias on the left, the output on
 * the right, hidden nodes in between by their longest path from an input.
 * Returns `{ nodes, edges }` with coordinates in [0, 1] (x = layer, y = slot).
 * Edges keep their genome fields. Uses *all* connections, enabled or not, so
 * the layout does not jump when a gene is toggled.
 */
export function layoutNetwork(genome) {
  const nodes = genome.nodes;
  const inputsAndBias = nodes.filter((n) => n.kind === "Input" || n.kind === "Bias");
  const output = nodes.find((n) => n.kind === "Output");
  const hidden = nodes.filter((n) => n.kind === "Hidden");
  const incoming = new Map(nodes.map((n) => [n.id, []]));
  for (const c of genome.connections) incoming.get(c.to).push(c.from);

  const depth = new Map(inputsAndBias.map((n) => [n.id, 0]));
  const resolve = (id, visiting = new Set()) => {
    if (depth.has(id)) return depth.get(id);
    if (visiting.has(id)) return 1; // cannot happen in a valid genome
    visiting.add(id);
    let best = 0;
    for (const from of incoming.get(id) ?? []) best = Math.max(best, resolve(from, visiting));
    visiting.delete(id);
    depth.set(id, best + 1);
    return best + 1;
  };
  for (const n of hidden) resolve(n.id);
  const lastLayer = Math.max(1, ...hidden.map((n) => depth.get(n.id)), 0) + (hidden.length > 0 ? 1 : 1);
  if (output) depth.set(output.id, lastLayer);

  const byLayer = new Map();
  for (const n of nodes) {
    const layer = depth.get(n.id) ?? 0;
    if (!byLayer.has(layer)) byLayer.set(layer, []);
    byLayer.get(layer).push(n);
  }
  const placed = [];
  for (const [layer, members] of byLayer) {
    members.sort((a, b) => a.id - b.id);
    members.forEach((n, i) => {
      placed.push({
        id: n.id,
        kind: n.kind,
        layer,
        x: lastLayer === 0 ? 0 : layer / lastLayer,
        y: (i + 0.5) / members.length,
      });
    });
  }
  return { nodes: placed, edges: genome.connections.map((c) => ({ ...c })) };
}

/**
 * Evaluates a genome exactly as the Rust `Network` does: inputs copied
 * through, bias = 1, every other node `tanh(sum of weight * source)` over its
 * *enabled* incoming connections. Returns `{ values, raw }` maps by node id;
 * `raw` holds the pre-`tanh` sum of every non-input node (for the output
 * node that is the score decisions are made on).
 */
export function forwardPass(genome, inputs) {
  const values = new Map();
  const raw = new Map();
  const inputNodes = genome.nodes.filter((n) => n.kind === "Input");
  inputNodes.forEach((n, i) => values.set(n.id, inputs[i]));
  for (const n of genome.nodes) if (n.kind === "Bias") values.set(n.id, 1);

  const incoming = new Map(genome.nodes.map((n) => [n.id, []]));
  for (const c of genome.connections) if (c.enabled) incoming.get(c.to).push(c);

  const visiting = new Set();
  const compute = (id) => {
    if (values.has(id)) return values.get(id);
    if (visiting.has(id)) throw new Error("cycle in genome");
    visiting.add(id);
    let sum = 0;
    for (const c of incoming.get(id)) sum += compute(c.from) * c.weight;
    visiting.delete(id);
    raw.set(id, sum);
    const value = Math.tanh(sum);
    values.set(id, value);
    return value;
  };
  for (const n of genome.nodes) compute(n.id);
  return { values, raw };
}

/** Maps a value in [-1, 1] to a diverging colour (orange negative, blue positive). */
export function divergingColor(value) {
  const v = Math.max(-1, Math.min(1, value));
  const strength = Math.abs(v);
  const hue = v >= 0 ? 215 : 28;
  return `hsl(${hue} ${Math.round(25 + 55 * strength)}% ${Math.round(92 - 42 * strength)}%)`;
}

/** Stroke width for a connection weight. */
export function edgeWidth(weight, limit = 8) {
  return 0.4 + 3.2 * Math.min(1, Math.abs(weight) / limit);
}

/** Ten-bucket histogram counts to opacities in [0, 1] relative to the largest bucket. */
export function histogramOpacities(counts) {
  const max = Math.max(1, ...counts);
  return counts.map((c) => c / max);
}

/** The index of the entry of `values` closest to `x` (for hover readouts). */
export function nearestIndex(values, x) {
  let best = 0;
  let bestDistance = Infinity;
  values.forEach((v, i) => {
    const d = Math.abs(v - x);
    if (d < bestDistance) {
      bestDistance = d;
      best = i;
    }
  });
  return best;
}
