// The simulation-results page: fetches /api/results and draws it. The logic lives
// in results_lib.js; this file only touches the DOM.
import {
  afterServed, agreement, agreementHtml, compareHtml, compareTables, estimateHtml, estimateRows, explainHtml, explainIds,
  skillHtml, skillRows, strategyRows, tableHtml,
} from "./results_lib.js";

const $ = (id) => document.getElementById(id);

let current = null; // the first file, kept for comparing with a picked one
let other = null; // the second file: { data, label } from the browser or the command line
let pending = null; // a file picked before the first load finished
const SERVED_LABEL = "given on the command line";

function setStatus(text, ok = true) {
  $("status").textContent = text;
  $("status").style.color = ok ? "" : "var(--neg)";
}

function drawCompare() {
  if (!current || !other) return;
  try {
    $("compare").innerHTML = compareHtml(compareTables(current, other.data));
    $("compare-status").textContent = `Second file: ${other.label}`;
  } catch (error) {
    $("compare").innerHTML = "";
    $("compare-status").textContent = String(error.message ?? error);
  }
}

// Fetches the file given on the command line. It only returns it (or the error text): a file
// picked in the browser while this request was out must not be overwritten by the result.
async function fetchServedCompare() {
  try {
    const response = await fetch("/api/compare", { cache: "no-store" });
    if (response.status === 404) return {}; // no second file on the command line
    const data = await response.json();
    if (!response.ok) throw new Error(data.error ?? `HTTP ${response.status}`);
    return { other: { data, label: SERVED_LABEL } };
  } catch (error) {
    return { error: String(error.message ?? error) };
  }
}

async function readPicked(file) {
  try {
    other = { data: JSON.parse(await file.text()), label: file.name };
    drawCompare();
  } catch (error) {
    other = null;
    $("compare").innerHTML = "";
    $("compare-status").textContent = `${file.name} is not valid JSON: ${error.message}`;
  }
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
    // A file picked in the browser wins over the command line, and a reload redraws it
    // against the new first file.
    if (pending) {
      const file = pending;
      pending = null;
      await readPicked(file);
    } else if (other && other.label !== "given on the command line") drawCompare();
    else {
      other = null;
      await loadServedCompare();
      drawCompare();
    }
  } catch (error) {
    setStatus(String(error.message ?? error), false);
  }
}

$("compare-file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  if (!current) {
    pending = file; // the first file is still loading: draw as soon as it is there
    $("compare-status").textContent = `${file.name}: waiting for the first file…`;
    return;
  }
  await readPicked(file);
});

$("reload").addEventListener("click", load);
load();
