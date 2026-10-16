// The simulation-results page: fetches /api/results and draws it. The logic lives
// in results_lib.js; this file only touches the DOM.
import {
  agreement, agreementHtml, compareHtml, compareTables, estimateHtml, estimateRows, explainHtml, explainIds,
  skillHtml, skillRows, strategyRows, tableHtml,
} from "./results_lib.js";

const $ = (id) => document.getElementById(id);

let current = null; // the first file, kept for comparing with a picked one

function setStatus(text, ok = true) {
  $("status").textContent = text;
  $("status").style.color = ok ? "" : "var(--neg)";
}

function drawCompare(other, label) {
  try {
    $("compare").innerHTML = compareHtml(compareTables(current, other));
    $("compare-status").textContent = `Second file: ${label}`;
  } catch (error) {
    $("compare").innerHTML = "";
    $("compare-status").textContent = String(error.message ?? error);
  }
}

async function loadServedCompare() {
  const response = await fetch("/api/compare", { cache: "no-store" });
  if (response.status === 404) return; // no second file on the command line
  const data = await response.json();
  if (!response.ok) throw new Error(data.error ?? `HTTP ${response.status}`);
  drawCompare(data, "given on the command line");
}

async function load() {
  setStatus("loading…");
  try {
    const response = await fetch("/api/results", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error ?? `HTTP ${response.status}`);
    const rows = strategyRows(data);
    const skill = skillRows(data);
    const estimate = estimateRows(data);
    current = data;
    $("summary").textContent = `${data.statistics?.matches_played ?? "?"} matches`;
    $("strategies").innerHTML = tableHtml(rows);
    $("skill-card").hidden = skill.length === 0;
    $("skill").innerHTML = skillHtml(skill);
    $("estimate-card").hidden = estimate.length === 0;
    $("estimate").innerHTML = estimateHtml(estimate);
    const both = agreement(data);
    $("agreement").innerHTML = both ? agreementHtml(both) : "";
    $("explain").innerHTML = explainHtml(data.statistics_catalog, explainIds(data));
    setStatus("loaded");
    await loadServedCompare();
  } catch (error) {
    setStatus(String(error.message ?? error), false);
  }
}

$("compare-file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file || !current) return;
  try {
    drawCompare(JSON.parse(await file.text()), file.name);
  } catch (error) {
    $("compare").innerHTML = "";
    $("compare-status").textContent = `${file.name} is not valid JSON: ${error.message}`;
  }
});

$("reload").addEventListener("click", load);
load();
