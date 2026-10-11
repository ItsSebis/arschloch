// The run-analysis page: fetches /api/analysis and draws it. The logic lives
// in analysis_lib.js; this file only touches the DOM.
import { fmt } from "./lib.js";
import {
  analysisUrl, championSpec, chartSvg, comparisonRows, esc, fmtSlope, keyRows, legend,
  opponentsSpec, trendColour, verdictTone,
} from "./analysis_lib.js";

const $ = (id) => document.getElementById(id);
let runs = [];

function setStatus(text, cls = "") {
  const el = $("status");
  el.textContent = text;
  el.className = `pill ${cls}`;
}

function swatches(items) {
  return items
    .map((i) => `<span class="item"><span class="swatch" style="background:${i.color}"></span>${esc(i.name)}</span>`)
    .join(" ");
}

function renderVerdict(run) {
  const items = run.analysis.verdict
    .map((v) => `<li class="verdict ${verdictTone(v.code)}"><code>${esc(v.code)}</code> ${esc(v.message)}</li>`)
    .join("");
  $("verdict").innerHTML = `<ul class="verdicts">${items}</ul>`;
}

function renderKeyNumbers(run) {
  const rows = keyRows(run.analysis)
    .map(([k, v, note]) => `<tr><th>${esc(k)}</th><td>${esc(v)}</td><td class="muted">${esc(note)}</td></tr>`)
    .join("");
  $("key-numbers").innerHTML = `<table class="data">${rows}</table>`;
}

function renderCharts(run) {
  const champion = championSpec(run);
  $("champion-chart").innerHTML = chartSvg(champion);
  $("champion-legend").innerHTML = swatches(legend(champion));
  const opponents = opponentsSpec(run);
  $("opponents-chart").innerHTML = chartSvg(opponents);
  $("opponents-legend").innerHTML = swatches(legend(opponents));
  const rows = run.analysis.opponents
    .map(
      (o) =>
        `<tr><th>${esc(o.name)}</th><td>${fmt(o.first_third_mean, 3, true)}</td><td>${fmt(o.last_third_mean, 3, true)}</td><td>${esc(fmtSlope(o.slope))}</td><td style="color:${trendColour(o.class)}">${esc(o.class ?? "n/a")}</td></tr>`,
    )
    .join("");
  $("opponents-table").innerHTML = rows
    ? `<table class="data"><tr><th>opponent</th><th>first third</th><th>last third</th><th>slope / 100 gens</th><th>trend</th></tr>${rows}</table>`
    : "";
}

function renderCompare() {
  $("compare-card").hidden = runs.length < 2;
  if (runs.length < 2) return;
  const rows = comparisonRows(runs)
    .map(
      (r, i) =>
        `<tr class="pick" data-index="${i}"><th>${esc(r.label)}</th><td>${r.generations}</td><td>${r.stall}</td><td>${fmt(r.finalReeval, 3, true)}</td><td>${fmt(r.lastThird, 3, true)}</td><td>${r.codes.map((c) => `<code>${esc(c)}</code>`).join(" ")}</td></tr>`,
    )
    .join("");
  $("compare").innerHTML = `<table class="data"><tr><th>run</th><th>generations</th><th>stall</th><th>final re-eval</th><th>last-third trend</th><th>verdict</th></tr>${rows}</table>`;
  for (const tr of $("compare").querySelectorAll("tr.pick")) {
    tr.addEventListener("click", () => select(Number(tr.dataset.index)));
  }
}

function select(index) {
  const run = runs[index];
  $("run-picker").value = String(index);
  renderVerdict(run);
  renderKeyNumbers(run);
  renderCharts(run);
}

async function load() {
  const path = new URLSearchParams(location.search).get("path");
  let response;
  try {
    response = await fetch(analysisUrl(path));
  } catch (error) {
    setStatus("server unreachable", "warn");
    return;
  }
  const body = await response.json().catch(() => ({ error: "unreadable reply" }));
  if (!response.ok) {
    setStatus("no analysis", "warn");
    $("verdict").innerHTML = `<p class="bad">${esc(body.error ?? "error")}</p><p class="muted">The analysis needs at least one finished generation in the served run directory.</p>`;
    return;
  }
  runs = body;
  $("run-picker").innerHTML = runs.map((r, i) => `<option value="${i}">${esc(r.label)}</option>`).join("");
  $("run-picker").addEventListener("change", (event) => select(Number(event.target.value)));
  setStatus(`${runs.length} run${runs.length === 1 ? "" : "s"}`, "done");
  renderCompare();
  select(0);
}

load();
