// The play page: sets up a game, shows the table, and sends your moves.
// The server holds the game, so a reload picks it up again. Everything from
// the server is escaped before it is put into the page.
import {
  applyEvent, barFraction, cardColor, describeEvent, eventDelay, formatScore, groupHand,
  initialDisplay, selectionState, suggestSelection, toggleSelection,
} from "/play-lib.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const SUITS = ["d", "h", "s", "c"];

const state = {
  catalog: [],
  players: 4,
  gameId: null,
  view: null,
  names: [],
  eventsSeen: 0,
  display: null,
  selected: [],
  animating: false,
  busy: false, // a request is in flight
  generation: 0, // bumped when a game starts or is left, to cancel a running replay
  hideHand: false,
  removed: new Set(),
  message: "",
  messageClass: "",
  setup: null,
};
window.__play = state; // for the verification scripts

// ------------------------------------------------------------------ storage
function remember(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode */ }
}
function recall(key) {
  try { return JSON.parse(localStorage.getItem(key)); } catch { return null; }
}

// ---------------------------------------------------------------- networking
async function api(method, url, body) {
  const options = { method, cache: "no-store" };
  if (body !== undefined) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(body);
  }
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `${url}: ${response.status}`);
  return data;
}

function setStatus(ok) {
  const pill = $("status");
  pill.textContent = ok ? "connected" : "server not reachable";
  pill.className = ok ? "pill" : "pill warn";
}

// -------------------------------------------------------------------- setup
const DEFAULT_OPPONENTS = ["model:champion-v2", "endgame-denial", "adaptive:reading,tempo,bully", "lowest-legal", "card-counter"];

function optionsFor(selectedId) {
  const group = (kind, title) => {
    const entries = state.catalog.filter((e) => e.kind === kind);
    return `<optgroup label="${title}">${entries.map((e) => `<option value="${esc(e.id)}"${e.id === selectedId ? " selected" : ""}>${esc(e.label)}</option>`).join("")}</optgroup>`;
  };
  return group("model", "Trained models") + group("strategy", "Hand-written strategies");
}

function renderSetup() {
  const remembered = state.setup ?? {};
  $("players").innerHTML = [3, 4, 5, 6]
    .map((n) => `<button type="button" data-players="${n}" class="${n === state.players ? "on" : ""}">${n}</button>`)
    .join("");
  $("seat").innerHTML = `<option value="">random</option>${Array.from({ length: state.players }, (_, i) => `<option value="${i}">seat ${i + 1}</option>`).join("")}`;
  $("seat").value = remembered.human_seat !== undefined && remembered.human_seat < state.players ? String(remembered.human_seat) : "";
  $("opponents").innerHTML = Array.from({ length: state.players - 1 }, (_, i) => {
    const wanted = remembered.opponents?.[i] ?? DEFAULT_OPPONENTS[i % DEFAULT_OPPONENTS.length];
    const id = state.catalog.some((e) => e.id === wanted) ? wanted : state.catalog[0]?.id;
    return `<select data-opponent="${i}" aria-label="Opponent ${i + 1}">${optionsFor(id)}</select>`;
  }).join("");
  $("rule").hidden = $("deck").value !== "double";
}

function readSetup() {
  const seat = $("seat").value;
  return {
    players: state.players,
    deck: $("deck").value,
    duplicate_rule: $("rule").value,
    pass_rule: $("passrule").value,
    exchange_rule: $("exchangerule").value,
    rounds: Number($("rounds").value),
    ...(seat === "" ? {} : { human_seat: Number(seat) }),
    opponents: [...document.querySelectorAll("[data-opponent]")].map((s) => s.value),
  };
}

async function startGame(event) {
  event.preventDefault();
  $("setup-error").textContent = "";
  const setup = readSetup();
  state.setup = setup;
  remember("arschloch.setup", setup);
  try {
    const reply = await api("POST", "/api/games", setup);
    state.gameId = reply.id;
    remember("arschloch.game", reply.id);
    enterGame(reply.view, reply.events, { animate: true });
  } catch (error) {
    $("setup-error").textContent = error.message;
  }
}

// --------------------------------------------------------------------- game
function resetGameState() {
  state.selected = [];
  state.removed = new Set();
  state.hideHand = false;
  state.animating = false;
  state.message = "";
  $("log").innerHTML = "";
  $("advice").innerHTML = "The model sees only what you see. Click a suggestion to select those cards.";
  $("round-over").innerHTML = "";
}

function enterGame(view, events, { animate }) {
  state.generation += 1;
  state.busy = false;
  resetGameState();
  $("setup").hidden = true;
  $("game").hidden = false;
  state.view = view;
  state.names = view.seats.map((s) => s.name);
  state.eventsSeen = 0;
  if (animate) {
    // Show the table as it stood before the first event, then replay.
    state.display = initialDisplay({ ...view, seats: view.seats.map((s) => ({ ...s, hand_size: 0, place: null })), table: null, to_move: null });
    render();
    playEvents(events, view, state.generation);
  } else {
    for (const event of events) logEvent(event);
    state.eventsSeen = view.event_count;
    state.display = initialDisplay(view);
    render();
  }
}

async function continueGame() {
  const id = recall("arschloch.game");
  if (!id) return;
  try {
    const reply = await api("GET", `/api/games/${id}`);
    state.gameId = id;
    enterGame(reply.view, reply.events, { animate: false });
  } catch (error) {
    $("setup-error").textContent = `${error.message}`;
    $("continue").hidden = true;
  }
}

function leaveGame() {
  state.generation += 1; // stops a replay still running
  state.animating = false;
  state.busy = false;
  $("game").hidden = true;
  $("setup").hidden = false;
  state.gameId = null;
  state.view = null;
  renderSetup();
  $("continue").hidden = true;
}

function logEvent(event) {
  const text = describeEvent(event, state.names, state.view.human_seat);
  if (!text) return;
  const involvesYou = (event.seat ?? event.leader) === state.view.human_seat || event.type === "exchange_yours";
  const row = document.createElement("div");
  row.className = event.type === "round_start" || event.type === "round_end" || event.type === "match_end" ? "sys" : involvesYou ? "you" : "";
  row.textContent = text;
  const log = $("log");
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** Shows `events` one by one, then settles on the server's final `view`. */
async function playEvents(events, finalView, token) {
  state.animating = true;
  const speed = Number($("speed").value);
  for (const event of events) {
    if (token !== state.generation) return; // another game took over
    logEvent(event);
    state.display = applyEvent(state.display, event);
    if (event.type === "play" && event.seat === finalView.human_seat) {
      for (const card of event.cards) state.removed.add(card.id);
    }
    if (event.type === "round_start") state.hideHand = true;
    renderTable();
    renderHand();
    const delay = eventDelay(event, speed);
    if (delay) await sleep(delay);
  }
  if (token !== state.generation) return;
  state.eventsSeen += events.length;
  state.view = finalView;
  state.display = initialDisplay(finalView);
  state.removed = new Set();
  state.hideHand = false;
  state.animating = false;
  state.selected = [];
  state.message = "";
  state.eventsSeen = finalView.event_count;
  // Advice was about the old position.
  $("advice").textContent = "Ask again for this position.";
  render();
}

async function act(action, body = {}) {
  // One action at a time: a second click or key press while a request is in
  // flight would be applied to a position the player has not seen.
  if (state.busy || state.animating || !state.gameId) return;
  state.busy = true;
  renderActions();
  const token = state.generation;
  try {
    const reply = await api("POST", `/api/games/${state.gameId}/${action}`, body);
    if (token !== state.generation) return;
    state.message = "";
    state.selected = [];
    // The reply carries only this action's events; a gap means another tab
    // moved too, so refetch instead of guessing.
    if (reply.events_from !== state.eventsSeen) {
      const full = await api("GET", `/api/games/${state.gameId}?events_since=${state.eventsSeen}`);
      if (token !== state.generation) return;
      await playEvents(full.events, full.view, token);
    } else {
      await playEvents(reply.events, reply.view, token);
    }
  } catch (error) {
    if (token !== state.generation) return;
    state.message = error.message;
    state.messageClass = "bad";
    renderMessage();
  } finally {
    if (token === state.generation) {
      state.busy = false;
      if (state.view) renderActions();
    }
  }
}

// ---------------------------------------------------------------- rendering
function render() {
  renderTopbar();
  renderTable();
  renderHand();
  renderRoundOver();
}

function renderTopbar() {
  const view = state.view;
  $("round-info").textContent = `Round ${view.round} of ${view.rounds}`;
  const phase = { exchange: "card exchange", playing: view.to_move === view.human_seat ? "your turn" : "waiting", round_over: "round over", match_over: "match over" }[view.phase];
  $("phase-info").textContent = `${view.player_count} players · ${phase}`;
}

function cardHtml(card, extra = "") {
  const suit = card.label.slice(-1);
  const rank = card.label.slice(0, -1);
  return `<span class="pc ${cardColor(card)} ${extra}" title="${esc(card.label)}">${esc(rank)}<small>${esc(suit)}</small></span>`;
}

function renderTable() {
  const view = state.view;
  const display = state.display;
  $("seats").innerHTML = view.seats.map((seat, i) => {
    const size = display.handSizes[i];
    const place = display.places[i];
    const turn = !state.animating && view.to_move === i && view.phase === "playing";
    const out = display.passed[i] && display.passRule === "final";
    const backs = Array.from({ length: Math.min(size, 26) }, () => '<span class="back"></span>').join("");
    return `<div class="seat ${turn ? "turn" : ""} ${place || out ? "out" : ""}">
      ${place ? `<span class="place">${place}</span>` : ""}
      <div class="name">${esc(seat.name)}${seat.is_human ? " (you)" : ""}</div>
      <div class="role">${seat.role ? `${esc(roleName(seat.role))} · ` : ""}${size} card${size === 1 ? "" : "s"}</div>
      <div class="backs">${seat.is_human ? "" : backs}</div>
      <div class="act">${esc(display.actions[i] || (display.leader === i && !display.table ? "leads" : ""))}${display.passed[i] && display.passRule === "final" ? ' <span class="muted">· out of this trick</span>' : ""}</div>
    </div>`;
  }).join("");
  const table = display.table;
  if (table) {
    $("table").innerHTML = `<div class="cards">${table.cards.map((c) => cardHtml(c)).join("")}</div>
      <div class="caption">${esc(state.names[table.seat])} ${table.seat === view.human_seat ? "played" : "played"} ${table.cards.length} card${table.cards.length === 1 ? "" : "s"}</div>`;
  } else {
    const leader = display.leader ?? view.to_move;
    $("table").innerHTML = `<div class="caption">The table is empty${leader !== null ? ` — ${esc(state.names[leader])} ${leader === view.human_seat ? "lead" : "leads"}` : ""}</div>`;
  }
}

function roleName(role) {
  return { ViceArschloch: "Vice-Arschloch", Vize: "Vize" }[role] ?? role;
}

function currentSelectionState() {
  return selectionState(state.view, state.selected);
}

function renderHand() {
  const view = state.view;
  const hand = $("hand");
  if (state.hideHand) {
    hand.innerHTML = '<span class="muted">Dealing…</span>';
  } else {
    const cards = view.hand.filter((c) => !state.removed.has(c.id));
    const myTurn = !state.animating && view.phase === "playing" && view.to_move === view.human_seat;
    const playableRanks = new Set(view.playable.map((p) => p.rank));
    const exchange = view.phase === "exchange" && !state.animating;
    hand.innerHTML = groupHand(cards).map((group) => {
      const dim = myTurn && !playableRanks.has(group.rank);
      const head = myTurn && !dim ? `<button type="button" class="head" data-rank="${group.rank}" title="Select this rank as a whole">${esc(group.label)}</button>` : `<span class="head muted">${esc(group.label)}</span>`;
      const row = group.cards.map((c) => {
        const selected = state.selected.includes(c.id);
        return `<button type="button" class="pc ${cardColor(c)} ${selected ? "selected" : ""} ${dim ? "dim" : ""}" data-card="${c.id}" aria-pressed="${selected}" title="${esc(c.label)}"${myTurn || exchange ? "" : " disabled"}>${esc(c.label.slice(0, -1))}<small>${esc(c.label.slice(-1))}</small></button>`;
      }).join("");
      return `<div class="group">${head}<div class="row">${row}</div></div>`;
    }).join("");
  }
  renderActions();
  renderMessage();
}

function renderActions() {
  const view = state.view;
  const busy = state.animating || state.busy;
  const check = currentSelectionState();
  const exchange = view.phase === "exchange";
  document.querySelector(".actions").hidden = view.phase !== "playing" && !exchange;
  $("play").textContent = exchange ? `Give ${view.give_count} card${view.give_count === 1 ? "" : "s"}` : "Play selected";
  $("play").disabled = busy || !check.valid;
  $("play").title = check.valid ? "" : check.reason;
  const myTurn = view.phase === "playing" && view.to_move === view.human_seat;
  $("pass").hidden = exchange;
  $("pass").disabled = busy || !myTurn || view.must_lead;
  $("pass").title = view.must_lead && myTurn ? "You lead this trick, so you must play" : "";
  $("clear").disabled = state.selected.length === 0;
  $("ask").disabled = busy || !myTurn;
}

function renderMessage() {
  const view = state.view;
  const check = currentSelectionState();
  let text = state.message;
  let cls = state.messageClass;
  if (!text) {
    cls = "";
    if (state.animating) text = "…";
    else if (view.phase === "exchange") text = `You hold a low role: choose ${view.give_count} card${view.give_count === 1 ? "" : "s"} to give away.${state.selected.length ? ` ${check.valid ? "Ready." : check.reason}` : ""}`;
    else if (view.phase === "playing" && view.to_move === view.human_seat) {
      text = state.selected.length ? (check.valid ? "Ready to play." : check.reason) : (view.must_lead ? "You lead: play any cards of one rank." : "Your turn: beat the table or pass.");
      cls = state.selected.length && !check.valid ? "bad" : "";
    } else if (view.phase === "round_over") text = "Round over.";
    else if (view.phase === "match_over") text = "Match over.";
    else text = "";
  }
  $("message").textContent = text;
  $("message").className = `message ${cls}`;
}

function renderRoundOver() {
  const view = state.view;
  const box = $("round-over");
  if (view.phase !== "round_over" && view.phase !== "match_over") {
    box.innerHTML = "";
    return;
  }
  const roles = view.roles_history[view.roles_history.length - 1];
  const order = view.seats.map((s, i) => ({ seat: s, role: roles[i], place: s.place })).sort((a, b) => (a.place ?? 99) - (b.place ?? 99));
  const rows = order.map((o) => `<tr><td>${o.place ?? ""}</td><td>${esc(o.seat.name)}${o.seat.is_human ? " (you)" : ""}</td><td>${esc(roleName(o.role))}</td></tr>`).join("");
  let extra;
  if (view.phase === "round_over") {
    extra = '<button id="next" type="button" class="primary">Next round</button>';
  } else {
    const result = view.final;
    extra = `<p><strong>Your result: ${formatScore(result.score)}</strong> <span class="muted">(+1 = always President, −1 = always last; 0 = even)</span></p>
      <p class="muted">Your roles: ${result.roles.map((r) => esc(roleName(r))).join(", ")}</p>
      <button id="rematch" type="button" class="primary">Same setup again</button> <button id="again" type="button">New game</button>`;
  }
  box.innerHTML = `<div class="summary"><table class="t"><thead><tr><th>Place</th><th>Player</th><th>Role</th></tr></thead><tbody>${rows}</tbody></table>${extra}</div>`;
}

// ------------------------------------------------------------------- events
function selectedPlayable() {
  return !state.animating && (state.view.phase === "exchange" || (state.view.phase === "playing" && state.view.to_move === state.view.human_seat));
}

function submitSelection() {
  const check = currentSelectionState();
  if (!check.valid) return;
  act(state.view.phase === "exchange" ? "give" : "play", { cards: [...state.selected] });
}

function onHandClick(event) {
  if (!selectedPlayable()) return;
  const head = event.target.closest("[data-rank]");
  if (head) {
    const pick = suggestSelection(state.view, Number(head.dataset.rank));
    state.selected = state.selected.length === pick.length && pick.every((id) => state.selected.includes(id)) ? [] : pick;
    state.message = "";
    renderHand();
    return;
  }
  const button = event.target.closest("[data-card]");
  if (button) {
    state.selected = toggleSelection(state.selected, Number(button.dataset.card));
    state.message = "";
    renderHand();
  }
}

async function ask() {
  const model = $("advisor").value;
  if (!model || !state.gameId) return;
  $("advice").textContent = "thinking…";
  try {
    const reply = await api("GET", `/api/games/${state.gameId}/advice?model=${encodeURIComponent(model)}`);
    const scores = reply.advice.map((a) => a.raw_score);
    const lo = Math.min(...scores);
    const hi = Math.max(...scores);
    const rows = reply.advice.map((a, i) => {
      const label = a.is_pass ? "Pass" : a.cards.map((c) => esc(c.label)).join(" ");
      const width = hi > lo ? ((a.raw_score - lo) / (hi - lo)) * 100 : 100;
      const ids = a.cards.map((c) => c.id).join(",");
      return `<tr class="${a.is_pass ? "" : "pick"}" data-ids="${ids}"><td>${i === 0 ? "★" : ""}</td><td>${label}</td><td>${a.raw_score.toFixed(2)}</td><td><div class="bar"><i style="left:0;width:${width.toFixed(0)}%"></i></div></td></tr>`;
    }).join("");
    $("advice").innerHTML = `<div class="muted">${esc(reply.model)} ranks your legal moves (higher = better in its eyes):</div><table class="t"><tbody>${rows}</tbody></table>`;
  } catch (error) {
    $("advice").innerHTML = `<span class="bad">${esc(error.message)}</span>`;
  }
}

function onAdviceClick(event) {
  const row = event.target.closest("tr[data-ids]");
  if (!row || !row.dataset.ids || !selectedPlayable()) return;
  state.selected = row.dataset.ids.split(",").map(Number);
  renderHand();
}

// ------------------------------------------------------------------ records
// Which older rules a recorded game was played under, empty for the current ones.
function rulesNote(entry) {
  const old = [];
  if (entry.pass_rule === "free") old.push("free passing");
  if (entry.exchange_rule === "free") old.push("free exchange");
  return old.length ? ` <span class="muted">(${old.join(", ")})</span>` : "";
}

async function loadRecords() {
  const box = $("records");
  try {
    const data = await api("GET", "/api/records");
    if (!data.total_games) {
      box.innerHTML = "<h2>Your record</h2><p class=\"muted\">No finished games yet. Play a match to the end and it is recorded here.</p>";
      return;
    }
    const bar = (score) => `<div class="bar"><i style="left:${(Math.min(barFraction(score), 0.5) * 100).toFixed(0)}%;width:${(Math.abs(barFraction(score) - 0.5) * 100).toFixed(0)}%;${score < 0 ? "background:var(--neg)" : "background:var(--pos)"}"></i></div>`;
    const opp = data.by_opponent.map((o) => `<tr><td>${esc(o.opponent)}${rulesNote(o)}</td><td>${o.games}</td><td>${formatScore(o.mean_score)}</td><td>${bar(o.mean_score)}</td></tr>`).join("");
    const tables = data.by_table.map((t) => `<tr><td>${t.players}${rulesNote(t)}</td><td>${t.opponents.map(esc).join(", ")}</td><td>${t.games}</td><td>${formatScore(t.mean_score)}</td><td>${(t.president_rate * 100).toFixed(0)}%</td><td>${(t.last_rate * 100).toFixed(0)}%</td></tr>`).join("");
    const recent = data.recent.map((r) => `<tr><td>${new Date(r.finished_unix * 1000).toLocaleString()}</td><td>${r.player_count}p vs ${r.opponents.map(esc).join(", ")}</td><td>${formatScore(r.score)}</td></tr>`).join("");
    box.innerHTML = `<h2>Your record</h2>
      ${data.write_error ? `<p class="bad">${esc(data.write_error)} — new games are not being saved.</p>` : ""}
      <p class="muted">${data.total_games} finished game${data.total_games === 1 ? "" : "s"}. Score: +1 = always President, 0 = even, −1 = always last. For reference the built-in champion scores about +0.6 against the hand-written strategies (docs/baselines/neat-v2).</p>
      <h3>By opponent</h3><table class="t"><thead><tr><th>Opponent</th><th>Games</th><th>Mean score</th><th></th></tr></thead><tbody>${opp}</tbody></table>
      <h3>By table</h3><table class="t"><thead><tr><th>Players</th><th>Opponents</th><th>Games</th><th>Mean</th><th>President</th><th>Last</th></tr></thead><tbody>${tables}</tbody></table>
      <h3>Recent games</h3><table class="t"><tbody>${recent}</tbody></table>`;
  } catch (error) {
    box.innerHTML = `<p class="bad">${esc(error.message)}</p>`;
  }
}

function showTab(name) {
  for (const tab of document.querySelectorAll(".tab")) tab.classList.toggle("active", tab.dataset.tab === name);
  $("tab-play").hidden = name !== "play";
  $("tab-records").hidden = name !== "records";
  if (name === "records") loadRecords();
}

// --------------------------------------------------------------------- init
async function init() {
  state.setup = recall("arschloch.setup");
  if (state.setup?.rounds) $("rounds").value = state.setup.rounds;
  if (state.setup?.deck) $("deck").value = state.setup.deck;
  if (state.setup?.pass_rule) $("passrule").value = state.setup.pass_rule;
  if (state.setup?.exchange_rule) $("exchangerule").value = state.setup.exchange_rule;
  try {
    state.catalog = (await api("GET", "/api/catalog")).opponents;
    setStatus(true);
  } catch {
    setStatus(false);
    return;
  }
  const models = state.catalog.filter((e) => e.kind === "model");
  $("advisor").innerHTML = models.map((m) => `<option value="${esc(m.id)}">${esc(m.label)}</option>`).join("");
  renderSetup();
  $("continue").hidden = !recall("arschloch.game");

  $("players").addEventListener("click", (event) => {
    const button = event.target.closest("[data-players]");
    if (button) {
      state.players = Number(button.dataset.players);
      renderSetup();
    }
  });
  $("deck").addEventListener("change", () => { $("rule").hidden = $("deck").value !== "double"; });
  $("setup").addEventListener("submit", startGame);
  $("continue").addEventListener("click", continueGame);
  $("new-game").addEventListener("click", leaveGame);
  $("hand").addEventListener("click", onHandClick);
  $("advice").addEventListener("click", onAdviceClick);
  $("play").addEventListener("click", submitSelection);
  $("pass").addEventListener("click", () => act("pass"));
  $("clear").addEventListener("click", () => { state.selected = []; state.message = ""; renderHand(); });
  $("ask").addEventListener("click", ask);
  $("round-over").addEventListener("click", (event) => {
    if (event.target.id === "next") act("next");
    if (event.target.id === "again") leaveGame();
    if (event.target.id === "rematch") {
      leaveGame();
      $("setup").requestSubmit();
    }
  });
  for (const tab of document.querySelectorAll(".tab")) tab.addEventListener("click", () => showTab(tab.dataset.tab));
  document.addEventListener("keydown", (event) => {
    if (!state.view || $("game").hidden || event.target.matches("input, select, textarea")) return;
    if (event.key === "Escape") { state.selected = []; renderHand(); }
    if (event.key === "Enter" && !event.target.matches("button")) { event.preventDefault(); submitSelection(); }
  });
  setInterval(() => api("GET", "/api/catalog").then(() => setStatus(true)).catch(() => setStatus(false)), 5000);
}

init();
