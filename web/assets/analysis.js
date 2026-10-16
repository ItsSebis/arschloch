// The run-analysis page: fetches /api/analysis and draws it. The logic lives
// in analysis_lib.js; this file only touches the DOM.
import {
  analysisUrl, championSpec, chartSvg, compareHtml, esc, keyNumbersHtml, legend, opponentsSpec,
  opponentsTableHtml, pickerHtml, skippedHtml, swatchesHtml, verdictHtml,
} from "./analysis_lib.js";

const $ = (id) => document.getElementById(id);
let runs = [];

function setStatus(text, cls = "") {
  const el = $("status");
  el.textContent = text;
  el.className = `pill ${cls}`;
}

function renderCharts(run) {
  const champion = championSpec(run);
  $("champion-chart").innerHTML = chartSvg(champion);
  $("champion-legend").innerHTML = swatchesHtml(legend(champion));
  const opponents = opponentsSpec(run);
  $("opponents-chart").innerHTML = chartSvg(opponents);
  $("opponents-legend").innerHTML = swatchesHtml(legend(opponents));
  $("opponents-table").innerHTML = opponentsTableHtml(run);
}

function renderCompare() {
  $("compare-card").hidden = runs.length < 2;
  if (runs.length < 2) return;
  $("compare").innerHTML = compareHtml(runs);
  for (const tr of $("compare").querySelectorAll("tr.pick")) {
    tr.addEventListener("click", () => select(Number(tr.dataset.index)));
  }
}

function select(index) {
  const run = runs[index];
  $("run-picker").value = String(index);
  $("verdict").innerHTML = verdictHtml(run);
  $("key-numbers").innerHTML = keyNumbersHtml(run);
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
  runs = body.runs;
  $("skipped").innerHTML = skippedHtml(body.skipped_runs);
  $("run-picker").innerHTML = pickerHtml(runs);
  $("run-picker").addEventListener("change", (event) => select(Number(event.target.value)));
  setStatus(`${runs.length} run${runs.length === 1 ? "" : "s"}`, "done");
  renderCompare();
  select(0);
}

load();
