// The training dashboard. Polls the server once a second, keeps every
// generation event in memory and redraws. Text from the server is always
// escaped; charts are inline SVG built from the pure helpers in lib.js.
import {
  bandPath, divergingColor, edgeWidth, fmt, formatDuration, forwardPass, histogramOpacities,
  extent, layoutNetwork, linePath, linearScale, nearestIndex, niceScale, shouldRedraw, speciesColor, stackSpecies,
} from "/lib.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const state = {
  events: [], runStart: null, runKey: null, rendered: false, epoch: 0, finished: false, connected: false, lastChange: Date.now(),
  follow: true, showDisabled: false, networkGeneration: null,
  genomes: new Map(), decisions: new Map(),
  decisionGeneration: null, decisionIndex: 0, candidateIndex: null,
  checks: { replays: 0, maxError: 0 },
};
window.__dashboard = state; // for debugging and the verification scripts

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}

// ------------------------------------------------------------------ polling
async function poll() {
  let flags;
  try {
    const last = state.events.length ? state.events[state.events.length - 1].generation : null;
    const [info, data] = await Promise.all([
      fetchJson("/api/state"),
      fetchJson("/api/events" + (last === null ? "" : `?since=${last}`)),
    ]);
    let incoming = data.events;
    const epochChanged = data.epoch !== state.epoch;
    if (epochChanged) {
      // This is a different log (a resume rewrote it, or a different server
      // process took over the port): everything cached is suspect.
      state.epoch = data.epoch;
      state.events = [];
      state.genomes.clear();
      state.decisions.clear();
      if (last !== null) incoming = (await fetchJson("/api/events")).events;
    }
    const byGeneration = new Map(state.events.map((e) => [e.generation, e]));
    for (const e of incoming) byGeneration.set(e.generation, e);
    state.events = [...byGeneration.values()].sort((a, b) => a.generation - b.generation);
    if (incoming.length > 0) state.lastChange = Date.now();
    const runKey = JSON.stringify(info.run_start);
    flags = {
      first: !state.rendered,
      newEvents: incoming.length,
      epochChanged,
      finishedChanged: info.finished !== state.finished,
      reconnected: !state.connected,
      runChanged: runKey !== state.runKey,
    };
    state.runKey = runKey;
    state.runStart = info.run_start;
    state.finished = info.finished;
    state.connected = true;
  } catch (error) {
    state.connected = false;
    renderStatus();
    console.warn("dashboard poll failed:", error);
    setTimeout(poll, 1000);
    return;
  }
  // A drawing bug must not masquerade as a lost connection, nor stop polling.
  try {
    if (shouldRedraw(flags)) {
      render();
      state.rendered = true;
    } else {
      renderStatus();
    }
  } catch (error) {
    console.error("dashboard render failed:", error);
    $("status").textContent = "display error (see the browser console)";
    $("status").className = "pill warn";
  }
  setTimeout(poll, 1000);
}

// ------------------------------------------------------------------ summary
function bestEvent() {
  return [...state.events].reverse().find((e) => e.champion.is_new_best) ?? null;
}

function renderStatus() {
  const pill = $("status");
  const last = state.events[state.events.length - 1];
  let text = "waiting for the first generation…";
  let cls = "pill";
  if (!state.connected) {
    text = "disconnected, retrying…";
    cls = "pill warn";
  } else if (state.finished) {
    text = "finished";
    cls = "pill done";
  } else if (last) {
    const quiet = (Date.now() - state.lastChange) / 1000;
    const limit = Math.max(30, last.generation_secs * 5);
    if (quiet > limit) {
      text = `no new generation for ${formatDuration(quiet)}`;
      cls = "pill warn";
    } else {
      text = "training";
      cls = "pill live";
    }
  }
  pill.textContent = text;
  pill.className = cls;
}

function kpi(label, value, sub = "") {
  return `<div class="kpi"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div><div class="sub">${esc(sub)}</div></div>`;
}

function renderSummary() {
  const config = state.runStart?.config;
  const last = state.events[state.events.length - 1];
  const total = config?.generations ?? null;
  const best = bestEvent();
  const recent = state.events.slice(-10);
  const average = recent.length ? recent.reduce((s, e) => s + e.generation_secs, 0) / recent.length : null;
  const remaining = total !== null && last ? Math.max(0, total - (last.generation + 1)) : null;
  const eta = average !== null && remaining !== null ? formatDuration(average * remaining) : "–";
  const cards = [
    kpi("Generation", last ? `${last.generation + 1}${total ? ` / ${total}` : ""}` : "–", last ? `ETA ${state.finished ? "done" : eta}` : ""),
    kpi("Best champion (held-out)", best?.champion.heldout ? fmt(best.champion.heldout.mean, 3, true) : "–", best?.champion.heldout ? `±${fmt(best.champion.heldout.std_error, 3)} · generation ${best.generation}` : ""),
    kpi("Latest champion", last ? fmt(last.champion.reeval.mean, 3, true) : "–", last ? `±${fmt(last.champion.reeval.std_error, 3)} on fixed matches` : ""),
    kpi("Population mean", last ? fmt(last.fitness.mean, 3, true) : "–", last ? `best ${fmt(last.fitness.best, 3, true)}` : ""),
    kpi("Species", last ? String(last.species.length) : "–", last ? `threshold ${fmt(last.compatibility_threshold, 2)}` : ""),
    kpi("Speed", last ? `${Math.round(last.rounds_per_sec).toLocaleString()} rounds/s` : "–", last ? `${formatDuration(last.elapsed_secs)} elapsed` : ""),
  ];
  $("kpis").innerHTML = cards.join("");
  $("progress-bar").style.width = total && last ? `${Math.min(100, ((last.generation + 1) / total) * 100)}%` : "0";
  if (config) {
    const names = (state.runStart.opponents ?? []).join(", ");
    $("run-info").textContent = `${config.player_count} players · ${config.deck.toLowerCase()} deck · population ${config.neat.population_size} · ${config.matches_per_genome} matches × ${config.rounds_per_match} rounds per genome · seed ${config.seed} · opponents: ${names}`;
  }
}

// ------------------------------------------------------------------ charts
const W = 900;
const MARGIN = { l: 48, r: 12, t: 10, b: 26 };

function xTicks(x0, x1) {
  if (x1 === x0) return [x0];
  return niceScale(x0, x1, 6).ticks.filter((t) => Number.isInteger(t) && t >= x0 && t <= x1);
}

function readoutHtml(items) {
  return items
    .map((i) => `<span class="item">${i.swatch ? `<span class="swatch" style="background:${i.swatch}"></span>` : ""}${esc(i.label)} <b>${esc(i.value)}</b></span>`)
    .join(" ");
}

/**
 * spec: { xs, series: [{ name, color, values, band?: { lo, hi }, markers?, dashed? }],
 *         ref?, digits?, signed?, height? }
 */
function lineChart(containerId, readoutId, spec) {
  const el = $(containerId);
  const { xs, series } = spec;
  if (xs.length === 0) {
    el.innerHTML = '<p class="muted">no data yet</p>';
    return;
  }
  const H = spec.height ?? 280;
  const values = [];
  for (const s of series) {
    for (const v of s.values) if (v !== null && v !== undefined) values.push(v);
    if (s.band) for (const v of [...s.band.lo, ...s.band.hi]) if (v !== null && v !== undefined) values.push(v);
  }
  if (spec.ref !== undefined) values.push(spec.ref);
  const range = extent(values);
  const y = niceScale(range.min, range.max);
  const x0 = xs[0];
  const x1 = xs[xs.length - 1];
  const sx = linearScale(x0, x1 === x0 ? x0 + 1 : x1, MARGIN.l, W - MARGIN.r);
  const sy = linearScale(y.min, y.max, H - MARGIN.b, MARGIN.t);
  const parts = [];
  for (const t of y.ticks) {
    parts.push(`<line class="grid" x1="${MARGIN.l}" x2="${W - MARGIN.r}" y1="${sy(t)}" y2="${sy(t)}"/>`);
    parts.push(`<text x="${MARGIN.l - 6}" y="${sy(t) + 4}" text-anchor="end">${fmt(t, 2)}</text>`);
  }
  for (const t of xTicks(x0, x1)) {
    parts.push(`<text x="${sx(t)}" y="${H - 8}" text-anchor="middle">${t}</text>`);
  }
  if (spec.ref !== undefined) parts.push(`<line class="ref" x1="${MARGIN.l}" x2="${W - MARGIN.r}" y1="${sy(spec.ref)}" y2="${sy(spec.ref)}"/>`);
  for (const s of series) {
    const pts = xs.map((x, i) => [sx(x), s.values[i] === null || s.values[i] === undefined ? null : sy(s.values[i])]).filter((p) => p[1] !== null);
    if (s.band) {
      const hi = xs.map((x, i) => [sx(x), sy(s.band.hi[i])]);
      const lo = xs.map((x, i) => [sx(x), sy(s.band.lo[i])]);
      parts.push(`<path d="${bandPath(hi, lo)}" fill="var(--band)" stroke="none"/>`);
    }
    if (s.markers) {
      for (const p of pts) parts.push(`<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="3.5" fill="${s.color}"/>`);
    } else if (pts.length > 0) {
      parts.push(`<path d="${linePath(pts)}" fill="none" stroke="${s.color}" stroke-width="1.8" ${s.dashed ? 'stroke-dasharray="5 4"' : ""}/>`);
    }
  }
  parts.push(`<line class="cursor" x1="0" x2="0" y1="${MARGIN.t}" y2="${H - MARGIN.b}" visibility="hidden"/>`);
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img">${parts.join("")}</svg>`;

  if (!readoutId) return;
  const readout = $(readoutId);
  const cursor = el.querySelector(".cursor");
  const show = (i) => {
    const items = [{ label: "generation", value: String(xs[i]) }];
    for (const s of series) {
      const v = s.values[i];
      if (v !== null && v !== undefined) items.push({ label: s.name, value: fmt(v, spec.digits ?? 3, spec.signed ?? false), swatch: s.color });
    }
    readout.innerHTML = readoutHtml(items);
  };
  show(xs.length - 1);
  const svg = el.querySelector("svg");
  svg.addEventListener("mousemove", (event) => {
    const box = svg.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * W;
    const generation = x0 + ((px - MARGIN.l) / (W - MARGIN.l - MARGIN.r)) * (x1 - x0);
    const i = nearestIndex(xs, generation);
    cursor.setAttribute("x1", sx(xs[i]));
    cursor.setAttribute("x2", sx(xs[i]));
    cursor.setAttribute("visibility", "visible");
    show(i);
  });
  svg.addEventListener("mouseleave", () => {
    cursor.setAttribute("visibility", "hidden");
    show(xs.length - 1);
  });
}

function renderFitness() {
  const events = state.events;
  const xs = events.map((e) => e.generation);
  lineChart("fitness-chart", "fitness-readout", {
    xs, signed: true,
    series: [
      { name: "selection best", color: "var(--s6)", values: events.map((e) => e.fitness.best), dashed: true },
      { name: "population mean", color: "var(--s3)", values: events.map((e) => e.fitness.mean) },
      { name: "median", color: "var(--s5)", values: events.map((e) => e.fitness.median), dashed: true },
      { name: "champion (fixed matches)", color: "var(--s1)", values: events.map((e) => e.champion.reeval.mean),
        band: { lo: events.map((e) => e.champion.reeval.mean - e.champion.reeval.std_error), hi: events.map((e) => e.champion.reeval.mean + e.champion.reeval.std_error) } },
      { name: "new best, held-out", color: "var(--s2)", values: events.map((e) => (e.champion.heldout ? e.champion.heldout.mean : null)), markers: true },
    ],
  });
}

function renderOpponents() {
  const events = state.events;
  const names = events.length ? events[events.length - 1].opponents.map((o) => o.name) : [];
  lineChart("opponents-chart", "opponents-readout", {
    xs: events.map((e) => e.generation), ref: 0, signed: true,
    series: names.map((name, k) => ({
      name, color: `var(--s${(k % 6) + 1})`,
      values: events.map((e) => e.opponents.find((o) => o.name === name)?.score.mean ?? null),
    })),
  });
}

function renderSpecies() {
  const el = $("species-chart");
  const events = state.events;
  if (events.length === 0) { el.innerHTML = '<p class="muted">no data yet</p>'; return; }
  const H = 260;
  const { ids, layers } = stackSpecies(events.map((e) => ({ generation: e.generation, species: e.species })));
  const top = Math.max(1, extent(events.map((e) => e.species.reduce((s, x) => s + x.size, 0))).max);
  const x0 = events[0].generation;
  const x1 = events[events.length - 1].generation;
  const sx = linearScale(x0, x1 === x0 ? x0 + 1 : x1, MARGIN.l, W - MARGIN.r);
  const sy = linearScale(0, top, H - MARGIN.b, MARGIN.t);
  const parts = [];
  for (const t of niceScale(0, top).ticks.filter((t) => t <= top)) {
    parts.push(`<line class="grid" x1="${MARGIN.l}" x2="${W - MARGIN.r}" y1="${sy(t)}" y2="${sy(t)}"/>`);
    parts.push(`<text x="${MARGIN.l - 6}" y="${sy(t) + 4}" text-anchor="end">${t}</text>`);
  }
  for (const t of xTicks(x0, x1)) parts.push(`<text x="${sx(t)}" y="${H - 8}" text-anchor="middle">${t}</text>`);
  layers.forEach((layer, k) => {
    const upper = layer.map((p) => [sx(p.generation), sy(p.upper)]);
    const lower = layer.map((p) => [sx(p.generation), sy(p.lower)]);
    parts.push(`<path d="${bandPath(upper, lower)}" fill="${speciesColor(ids[k])}" stroke="var(--card)" stroke-width="0.5"><title>species ${ids[k]}</title></path>`);
  });
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img">${parts.join("")}</svg>`;
  const last = events[events.length - 1];
  const sizes = [...last.species].sort((a, b) => b.size - a.size).slice(0, 4).map((s) => `#${s.id}: ${s.size}`).join(", ");
  $("species-readout").innerHTML = readoutHtml([
    { label: "generation", value: String(last.generation) },
    { label: "species", value: String(last.species.length) },
    { label: "largest", value: sizes },
    { label: "oldest", value: `${Math.max(...last.species.map((s) => s.age))} generations` },
  ]);
}

function renderSpread() {
  const el = $("spread-chart");
  const events = state.events;
  if (events.length === 0) { el.innerHTML = '<p class="muted">no data yet</p>'; return; }
  const H = 150;
  const innerW = W - MARGIN.l - MARGIN.r;
  const innerH = H - MARGIN.t - MARGIN.b;
  const column = innerW / events.length;
  const parts = [`<text x="${MARGIN.l - 6}" y="${MARGIN.t + 10}" text-anchor="end">best</text>`, `<text x="${MARGIN.l - 6}" y="${H - MARGIN.b}" text-anchor="end">low</text>`];
  events.forEach((e, i) => {
    const opacities = histogramOpacities(e.fitness.histogram);
    opacities.forEach((opacity, bucket) => {
      const y = MARGIN.t + innerH - ((bucket + 1) / opacities.length) * innerH;
      parts.push(`<rect x="${(MARGIN.l + i * column).toFixed(1)}" y="${y.toFixed(1)}" width="${(column + 0.4).toFixed(1)}" height="${(innerH / opacities.length + 0.4).toFixed(1)}" fill="var(--accent)" opacity="${opacity.toFixed(2)}"><title>generation ${e.generation}: ${e.fitness.histogram[bucket]} genomes</title></rect>`);
    });
  });
  for (const t of xTicks(events[0].generation, events[events.length - 1].generation)) {
    const i = events.findIndex((e) => e.generation === t);
    if (i >= 0) parts.push(`<text x="${MARGIN.l + (i + 0.5) * column}" y="${H - 8}" text-anchor="middle">${t}</text>`);
  }
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img">${parts.join("")}</svg>`;
}

function renderComplexity() {
  const events = state.events;
  const xs = events.map((e) => e.generation);
  lineChart("nodes-chart", "complexity-readout", {
    xs, height: 200, digits: 1,
    series: [
      { name: "champion", color: "var(--s1)", values: events.map((e) => e.champion.hidden_nodes) },
      { name: "population mean", color: "var(--s3)", values: events.map((e) => e.complexity.mean_hidden_nodes), dashed: true },
    ],
  });
  lineChart("connections-chart", null, {
    xs, height: 200, digits: 1,
    series: [
      { name: "champion", color: "var(--s1)", values: events.map((e) => e.champion.enabled_connections) },
      { name: "population mean", color: "var(--s3)", values: events.map((e) => e.complexity.mean_enabled_connections), dashed: true },
    ],
  });
}

// ------------------------------------------------------------------ network
/**
 * file: a genome file ({ feature_names, genome }).
 * options: { showDisabled, activations?: { values, raw, inputs } }
 */
function renderNetwork(container, file, options) {
  const genome = file.genome;
  const { nodes, edges } = layoutNetwork(genome);
  const tallest = Math.max(...Object.values(nodes.reduce((acc, n) => ((acc[n.layer] = (acc[n.layer] ?? 0) + 1), acc), {})));
  const H = Math.max(340, tallest * 24 + 40);
  const left = 190;
  const right = 150;
  const px = (x) => left + x * (W - left - right);
  const py = (y) => 20 + y * (H - 40);
  const position = new Map(nodes.map((n) => [n.id, { x: px(n.x), y: py(n.y) }]));
  const activations = options.activations;
  const inputIndex = new Map(genome.nodes.filter((n) => n.kind === "Input").map((n, i) => [n.id, i]));
  const parts = [];
  const sorted = [...edges].sort((a, b) => Math.abs(a.weight) - Math.abs(b.weight));
  for (const edge of sorted) {
    if (!edge.enabled && !options.showDisabled) continue;
    const a = position.get(edge.from);
    const b = position.get(edge.to);
    const color = edge.weight >= 0 ? "var(--pos)" : "var(--neg)";
    const opacity = edge.enabled ? 0.75 : 0.35;
    parts.push(`<line x1="${a.x.toFixed(1)}" y1="${a.y.toFixed(1)}" x2="${b.x.toFixed(1)}" y2="${b.y.toFixed(1)}" stroke="${color}" stroke-width="${edgeWidth(edge.weight).toFixed(2)}" opacity="${opacity}" ${edge.enabled ? "" : 'stroke-dasharray="4 3"'}><title>innovation ${edge.innovation}: ${edge.from} → ${edge.to}, weight ${fmt(edge.weight, 3, true)}${edge.enabled ? "" : " (disabled)"}</title></line>`);
  }
  for (const node of nodes) {
    const p = position.get(node.id);
    const activation = activations?.values.get(node.id);
    const fill = activation === undefined ? "var(--card)" : divergingColor(activation);
    parts.push(`<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="7" fill="${fill}" stroke="var(--muted)" stroke-width="1.2"><title>node ${node.id} (${node.kind})${activation === undefined ? "" : `, activation ${fmt(activation, 3, true)}`}</title></circle>`);
    let label = "";
    let anchor = "end";
    let dx = -12;
    if (node.kind === "Input") {
      const name = file.feature_names[inputIndex.get(node.id)] ?? `input ${node.id}`;
      label = activations ? `${name} = ${fmt(activations.inputs[inputIndex.get(node.id)], 2)}` : name;
    } else if (node.kind === "Bias") {
      label = "bias";
    } else if (node.kind === "Output") {
      anchor = "start";
      dx = 12;
      label = activations ? `score ${fmt(activations.raw.get(node.id), 3, true)}` : "score";
    } else if (activations) {
      anchor = "middle";
      dx = 0;
      label = fmt(activation, 2);
    }
    if (label) {
      const dy = node.kind === "Hidden" ? -11 : 4;
      parts.push(`<text x="${(p.x + dx).toFixed(1)}" y="${(p.y + dy).toFixed(1)}" text-anchor="${anchor}">${esc(label)}</text>`);
    }
  }
  container.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img">${parts.join("")}</svg>`;
}

async function getGenome(generation) {
  if (!state.genomes.has(generation)) state.genomes.set(generation, await fetchJson(`/api/genome/${generation}`));
  return state.genomes.get(generation);
}

async function showNetwork() {
  const slider = $("network-slider");
  const last = state.events.length ? state.events[state.events.length - 1].generation : 0;
  slider.max = String(last);
  if (state.follow || state.networkGeneration === null) state.networkGeneration = last;
  slider.value = String(state.networkGeneration);
  const generation = state.networkGeneration;
  $("network-generation").textContent = state.events.length ? `generation ${generation}` : "–";
  if (state.events.length === 0) return;
  const event = state.events.find((e) => e.generation === generation);
  try {
    const file = await getGenome(generation);
    if (generation !== state.networkGeneration) return; // the user moved on
    renderNetwork($("network"), file, { showDisabled: state.showDisabled });
    $("network-note").textContent = event
      ? `Champion of generation ${generation}: ${event.champion.hidden_nodes} hidden nodes, ${event.champion.enabled_connections} enabled connections, fixed-match score ${fmt(event.champion.reeval.mean, 3, true)}${event.champion.is_new_best ? " (new best)" : ""}.`
      : "";
  } catch (error) {
    $("network").innerHTML = `<p class="muted">no genome for generation ${generation} yet</p>`;
  }
}

// ------------------------------------------------------------------ decisions
function renderDecisionControls() {
  const generations = state.events.filter((e) => e.champion.is_new_best).map((e) => e.generation);
  const select = $("decision-generation");
  const wanted = generations.map(String).join(",");
  if (select.dataset.options !== wanted) {
    select.dataset.options = wanted;
    select.innerHTML = generations.map((g) => `<option value="${g}">generation ${g}</option>`).join("");
    if (generations.length && !generations.includes(state.decisionGeneration)) {
      state.decisionGeneration = generations[generations.length - 1];
      state.decisionIndex = 0;
      state.candidateIndex = null;
    }
  }
  if (state.decisionGeneration !== null) select.value = String(state.decisionGeneration);
}

async function showDecision() {
  const generation = state.decisionGeneration;
  if (generation === null) {
    $("decision-situation").textContent = "No new-best champion yet.";
    return;
  }
  let file;
  let decisions;
  try {
    file = await getGenome(generation);
    if (!state.decisions.has(generation)) state.decisions.set(generation, await fetchJson(`/api/decisions/${generation}`));
    decisions = state.decisions.get(generation);
  } catch (error) {
    $("decision-situation").textContent = `No recorded decisions for generation ${generation}.`;
    return;
  }
  if (generation !== state.decisionGeneration) return; // the user picked another champion meanwhile
  const list = decisions.decisions;
  const pick = $("decision-pick");
  const wanted = `${generation}:${list.length}`;
  if (pick.dataset.options !== wanted) {
    pick.dataset.options = wanted;
    pick.innerHTML = list.map((d, i) => `<option value="${i}">#${i + 1} · ${d.table ? `beat ${esc(d.table)}` : "lead"} → ${esc(d.candidates[d.chosen].description)}</option>`).join("");
  }
  state.decisionIndex = Math.min(state.decisionIndex, Math.max(0, list.length - 1));
  pick.value = String(state.decisionIndex);
  const decision = list[state.decisionIndex];
  if (!decision) return;
  if (state.candidateIndex === null || state.candidateIndex >= decision.candidates.length) state.candidateIndex = decision.chosen;

  $("decision-situation").innerHTML = `Hand: <b>${esc(decision.hand.join(" "))}</b> · ${decision.table ? `to beat: <b>${esc(decision.table)}</b>` : "<b>leading</b>"} · opponents hold ${esc(decision.opponent_hands.join(", "))} cards`;

  // Replay every candidate through the network; the recorded scores come from
  // the Rust implementation, so any difference is a bug.
  let worst = 0;
  const replays = decision.candidates.map((candidate) => {
    const replay = forwardPass(file.genome, candidate.features);
    const output = file.genome.nodes.find((n) => n.kind === "Output").id;
    worst = Math.max(worst, Math.abs(replay.raw.get(output) - candidate.raw_score));
    return replay;
  });
  state.checks.replays += decision.candidates.length;
  state.checks.maxError = Math.max(state.checks.maxError, worst);
  $("decision-check").innerHTML = worst < 1e-9
    ? `<span class="ok">✓</span> replayed all ${decision.candidates.length} candidates in the browser: scores match the recorded ones (max difference ${worst.toExponential(1)}).`
    : `<span class="bad">✗</span> the browser replay differs from the recorded scores by ${worst.toExponential(2)}.`;

  const rows = decision.candidates.map((candidate, i) => `<tr class="pick ${i === state.candidateIndex ? "selected" : ""}" data-index="${i}"><td>${i === decision.chosen ? "✓ chosen" : ""}</td><td>${esc(candidate.description)}</td><td>${fmt(candidate.raw_score, 3, true)}</td><td>${fmt(candidate.activation, 3, true)}</td></tr>`).join("");
  const table = $("decision-candidates");
  table.innerHTML = `<table class="candidates"><thead><tr><th></th><th>move</th><th>score</th><th>tanh</th></tr></thead><tbody>${rows}</tbody></table>`;
  for (const row of table.querySelectorAll("tr.pick")) {
    row.addEventListener("click", () => {
      state.candidateIndex = Number(row.dataset.index);
      showDecision();
    });
  }
  const candidate = decision.candidates[state.candidateIndex];
  renderNetwork($("decision-network"), file, {
    showDisabled: state.showDisabled,
    activations: { values: replays[state.candidateIndex].values, raw: replays[state.candidateIndex].raw, inputs: candidate.features },
  });
}

// ------------------------------------------------------------------ wiring
function render() {
  renderStatus();
  renderSummary();
  renderFitness();
  renderOpponents();
  renderSpecies();
  renderSpread();
  renderComplexity();
  showNetwork();
  renderDecisionControls();
  showDecision();
}

$("network-slider").addEventListener("input", (event) => {
  state.follow = false;
  state.networkGeneration = Number(event.target.value);
  showNetwork();
});
$("network-best").addEventListener("click", () => {
  const best = bestEvent();
  if (best) {
    state.follow = false;
    state.networkGeneration = best.generation;
    showNetwork();
  }
});
$("network-latest").addEventListener("click", () => {
  state.follow = true;
  showNetwork();
});
$("show-disabled").addEventListener("change", (event) => {
  state.showDisabled = event.target.checked;
  showNetwork();
  showDecision();
});
$("decision-generation").addEventListener("change", (event) => {
  state.decisionGeneration = Number(event.target.value);
  state.decisionIndex = 0;
  state.candidateIndex = null;
  showDecision();
});
$("decision-pick").addEventListener("change", (event) => {
  state.decisionIndex = Number(event.target.value);
  state.candidateIndex = null;
  showDecision();
});

poll();
setInterval(renderStatus, 5000);
