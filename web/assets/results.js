// The simulation-results page: fetches /api/results and draws it. The logic lives
// in results_lib.js; this file only touches the DOM.
import { explainHtml, skillHtml, skillNote, skillRows, strategyRows, tableHtml } from "./results_lib.js";

const $ = (id) => document.getElementById(id);

function setStatus(text, ok = true) {
  $("status").textContent = text;
  $("status").style.color = ok ? "" : "var(--neg)";
}

async function load() {
  setStatus("loading…");
  try {
    const response = await fetch("/api/results", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error ?? `HTTP ${response.status}`);
    const rows = strategyRows(data);
    const skill = skillRows(data);
    const note = skillNote(data);
    $("summary").textContent = `${data.statistics?.matches_played ?? "?"} matches${note ? ` · ${note}` : ""}`;
    $("strategies").innerHTML = tableHtml(rows);
    $("skill-card").hidden = skill.length === 0;
    $("skill").innerHTML = skillHtml(skill);
    $("explain").innerHTML = explainHtml(data.statistics_catalog);
    setStatus("loaded");
  } catch (error) {
    setStatus(String(error.message ?? error), false);
  }
}

$("reload").addEventListener("click", load);
load();
