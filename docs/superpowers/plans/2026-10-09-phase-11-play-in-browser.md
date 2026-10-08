# Phase 11: play against the models in the browser — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (the user chose native execution). Steps use checkbox (`- [ ]`) syntax.

**Goal:** A person sits at a table in the browser and plays full matches against trained NEAT models and hand-written strategies at 3-6 seats, can ask a model what it would play (and why), and sees their record against each opponent.

**Architecture:** `sim::session` is a pure, deterministic game session (no HTTP): it owns a `Round`, drives the AI seats through the same `Strategy` interface the simulator uses, pauses whenever the human must act, and logs everything as events. `web` gets a generic request/handler layer (POST with a capped JSON body, Host and Origin checks) and a `PlayApp` (catalog, sessions, records) beside the existing dashboard `App`. `cli play` builds the catalog from `--model` files and the hand-written strategies and serves it. The page is vanilla JS with pure helpers in `lib`-style modules tested by Node.

**Tech Stack:** Rust workspace (engine, sim, neat, cli, web), std-only HTTP, vanilla JS/SVG, node tests via `cargo test`.

**Spec:** `docs/ROADMAP.md`, Phase 11 ("Play against the models"); rules in `docs/RULES.md`.

## Interpretation notes (rulings)

- "Interactability on the web page" = the Phase 11 roadmap item "play against the models" (the part of Phase 11 where the page is interactive), plus making the opponent models inspectable while playing (ask-the-model advice). The simulation-statistics overview views of Phase 11 are not part of this phase.
- The human may choose any legal combination (any subset of one rank that beats the table), not only the canonical weakest/strongest subsets `Round::legal_moves` lists; the engine's `submit_move` is the authority.
- As the lower role in the exchange the human chooses the cards to give (RULES: free choice); as the higher role the lowest cards are handed back automatically, exactly as in the simulator.
- Only one human per game; they take a seat of their choosing (default random); AI seats are filled from a catalog.

## Global Constraints

- Server binds 127.0.0.1 only; every request's Host is checked; POST additionally requires `Content-Type: application/json`, a body of at most 16 KB, and, when an `Origin` header is present, an Origin whose host is localhost/127.0.0.1/::1 (cross-site pages must not be able to drive a game or write records).
- The human never receives other seats' hidden cards (hands) in any API response; only hand sizes, public plays and, for the exchange, the cards that involve the human.
- Same seed and same moves give the same game (determinism); AI seats use the match RNG as `run_match` does.
- Existing dashboard behavior is unchanged; all current tests keep passing.
- `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings` (stable and `+1.99.0`), `cargo test --workspace` by exit status; Windows and macOS CI must pass (no `/` in test assertions on paths).
- A bad request never panics the server and never corrupts a session (a rejected move leaves the game exactly as it was).

## Review Focus

1. A human submitting cards not in their hand, duplicated card ids, a combo of two ranks, a pass on the lead, a move out of turn, or a move after the round/match ended: clear 400-class error, game unchanged.
2. Hidden information: no response, event, or error text leaks another seat's hand (including via advice and via exchange events).
3. Session limits: creating thousands of games must not exhaust memory (cap and eviction), ids are unguessable enough for localhost, a finished game can be fetched but not moved.
4. POST hardening: oversized, chunked, missing Content-Length, wrong content type, cross-origin Origin, DNS-rebinding Host, slow clients.
5. Records file: concurrent finishes, unwritable path (game still works, error surfaced once), corrupt lines skipped on read.
6. Double deck and 3/5/6 seats, uneven deals (5 players), the human starting a round as leader, human as President/Arschloch with exchange counts 1-3.
7. UI: selecting cards for a rank, keyboard use, mobile width, animation not blocking input, reconnect after reload mid-game (state is server-side).

---

## Task 1: `sim::session` core

**Files:** Create `sim/src/session.rs`; modify `sim/src/lib.rs`, `sim/src/match_runner.rs` (make `turn_context_for` `pub(crate)`).

**Interfaces (produces):**
```rust
pub struct SessionConfig { pub player_count: u8, pub deck_variant: DeckVariant, pub duplicate_rule: DuplicateRule, pub rounds: usize, pub seed: u64, pub human_seat: u8 }
pub struct AiSeat { pub name: String, pub strategy: Arc<dyn Strategy> }
pub struct Session { /* private */ }
impl Session {
  pub fn new(config: SessionConfig, ai: Vec<AiSeat>) -> Result<Session, SessionError>; // ai.len() == player_count - 1, filling non-human seats in seat order
  pub fn view(&self) -> View;                       // everything the human may know (Serialize)
  pub fn events_since(&self, since: usize) -> &[GameEvent];
  pub fn play(&mut self, card_ids: &[u8]) -> Result<(), SessionError>;     // human plays these cards (deal_index ids)
  pub fn pass(&mut self) -> Result<(), SessionError>;
  pub fn give(&mut self, card_ids: &[u8]) -> Result<(), SessionError>;     // human, as lower role, gives these cards
  pub fn next_round(&mut self) -> Result<(), SessionError>;                // after RoundOver
  pub fn advice(&self, advisor: &NeatStrategy) -> Result<Vec<Advice>, SessionError>; // top moves with scores, human's turn only
}
pub enum SessionError { NotYourTurn, WrongPhase, UnknownCard(u8), DuplicateCard(u8), IllegalMove(String), BadConfig(String) }
```
Phases (in `View.phase`, snake_case strings): `exchange` (human must give `View.give_count`), `playing` (human to move), `round_over`, `match_over`. After every human action and in `new`/`next_round` the session auto-advances AI seats until the human must act. `GameEvent` is `Serialize` with `type` tag: `round_start`, `exchange`, `play`, `pass`, `trick_won`, `finished`, `round_end`, `match_end`; cards are `CardView { id, rank, suit }`.
`View` has: `phase, round (1-based), rounds, human_seat, hand (sorted), seats [{seat, name, is_human, hand_size, active, role (this round's, from the last round), place}], table {cards, seat} | null, to_move, playable [{rank, card_ids, sizes}] (sizes the *strongest* subset of that rank can legally play now), must_lead, give_count, roles_history: [[role per seat]] per finished round, event_count, final: {roles history, score} when match over`.

- [ ] **Step 1 (RED):** tests in `sim/src/session.rs`:
  - `a_session_plays_a_whole_match_with_a_scripted_human` — human plays `legal_moves()` first option via `play/pass/give/next_round` until `match_over`; roles history has `rounds` entries, every finishing order is a permutation; works for 3-6 seats and both decks.
  - `the_ai_seats_play_exactly_as_run_match_does` — with a strategy-driven stand-in for the human (the human's choices replayed from `LowestLegal`), the roles history equals `run_match` with the same config/seed when the human seat holds that same strategy (determinism cross-check of exchange, deal order and RNG use).
  - `the_view_never_contains_another_seats_cards` — serialize `view()` and all events; assert no card id from another seat's current hand appears except cards that were played publicly.
  - rejected moves leave the game unchanged: unknown card, duplicate id, cards of two ranks, card not beating, pass on lead, `play` during `exchange` phase, `give` with the wrong count, any call after `match_over`; compare `view()` JSON before/after.
  - `any_subset_of_a_rank_that_beats_is_accepted` — table has a pair; the human plays the weakest-beating and a non-canonical pair from a triple.
  - `the_human_as_arschloch_chooses_the_cards_to_give` and `as_president_receives_the_lowest_back_automatically` with exchange counts for 3, 4, 6 seats.
- [ ] **Step 2:** run, expect FAIL (module missing).
- [ ] **Step 3:** implement (reuse `PassTracker`, `turn_context_for`, `exchange_with_selection` with a closure that asks the AI strategy, and for the human seat a pre-collected selection).
- [ ] **Step 4:** all pass; `cargo test -p sim`.
- [ ] **Step 5:** commit `sim: an interactive game session with a human seat`.

## Task 2: advice from a model

**Files:** modify `sim/src/session.rs`.

- [ ] RED test `advice_lists_the_models_ranked_moves_for_the_humans_turn`: with the champion genome (`docs/baselines/neat-v1/champion.json`, loaded via `GenomeFile`), `advice` returns every candidate move sorted by raw score descending, the first equals what `NeatStrategy::choose_play` picks for that context, each entry has `cards` (or pass), `raw_score`; `advice` outside the human's `playing` phase is `WrongPhase`; advice never reveals hands (only the human's own options and scores).
- [ ] implement with `NeatStrategy::score_candidates`; commit `sim: ask a model what it would play`.

## Task 3: HTTP layer with POST

**Files:** modify `web/src/server.rs` (generic over a `Handler`), `web/src/routes.rs` (implement it for the dashboard `App`), `web/src/lib.rs`.

**Interfaces:** `pub struct HttpRequest { method, path_and_query, body: Vec<u8> }`; `pub trait Handler: Send + Sync + 'static { fn handle(&self, request: &HttpRequest) -> Response; }`; `Server::start(handler, port)` (the dashboard keeps `Dashboard::start(dir, port)` as a thin wrapper). Checks happen before the handler: Host (existing), POST needs `Content-Type: application/json` and `Content-Length` (<= 16 KB, no chunked), Origin host check, 405 for other methods.

- [ ] RED tests (raw sockets, like the existing ones): POST echo works; missing/oversized Content-Length, `Transfer-Encoding: chunked`, wrong content type, cross-origin `Origin: http://evil.example`, same-origin Origin accepted, slow body times out; existing GET tests unchanged.
- [ ] implement; commit `web: POST requests with a capped JSON body and origin checks`.

## Task 4: `PlayApp` (catalog, sessions, API, records)

**Files:** create `web/src/play.rs`, `web/src/records.rs`; modify `web/src/lib.rs`.

**API:** `GET /api/catalog` -> `[{id, label, kind: "model"|"strategy"}]`; `POST /api/games` `{players, deck, duplicate_rule, rounds, human_seat?: number, opponents: [catalog id; players-1], seed?}` -> `{id, view}`; `GET /api/games/ID?events_since=N` -> `{view, events}`; `POST /api/games/ID/play` `{cards:[id]}`; `.../pass`; `.../give` `{cards}`; `.../next`; `GET /api/games/ID/advice?model=CATALOG_ID` (models only); `GET /api/records` -> per-opponent-set and per-opponent stats (games, mean role score, President %, last games); errors are `{error: "..."}` with 400/404. Sessions: at most 16, least recently used evicted, ids 128-bit random hex. Records: one JSON line per finished game appended to the records file (`RecordStore`), unreadable lines skipped, a write failure is reported once through `/api/records` (`write_error`) and never breaks a game.

- [ ] RED tests (no sockets, call `PlayApp::handle`): full match through the API against `LowestLegal` seats; each error class in Review Focus 1 returns 400 and leaves the state; no hidden cards in any response (scan JSON for other seats' ids); 17th game evicts the oldest; records appended on `match_over` exactly once even if `next` is called repeatedly; corrupt records line skipped; advice only for models.
- [ ] implement; commit `web: play API with sessions and records`.

## Task 5: `cli play`

**Files:** create `cli/src/play.rs`; modify `cli/src/main.rs`; tests in `cli/tests/play_smoke.rs`.

`cli play [--model PATH|RUN_DIR]... [--port 8090] [--records FILE (default play-records.jsonl)]`. The catalog always contains the hand-written strategies (lowest-legal, random-legal, greedy-highest, hold-back-pairs, card-counter, endgame-denial, adaptive variants as in `--strategy`) and each `--model` (a genome file or a run directory's `best.json`, label `Neat(<stem>)`). Prints `play: http://127.0.0.1:PORT/`.

- [ ] RED smoke tests with the real binary and a socket: server starts, `/api/catalog` lists a given model, a whole one-round match is played through POSTs, bad `--model` fails before binding, busy port fails cleanly.
- [ ] implement; commit `cli: play against models in the browser`.

## Task 6: the page

**Files:** create `web/assets/play.html`, `play.js`, `play-lib.js`, `play.css`; route them from `PlayApp`; tests `web/tests/play-lib.test.mjs` (run by `web/tests/js.rs`).

Pure helpers in `play-lib.js` (unit-tested): `cardLabel`, `sortHand`, `groupByRank`, `toggleSelection(selected, card, rules)`, `selectionState(selection, view)` -> `{valid, reason}` (same rank, size equals table size or any size on a lead, within `playable` sizes), `describeEvent(event, names)`. UI: setup screen (players 3-6, deck, rules, rounds, pick each opponent from the catalog, seat), table with seats and hand sizes, the table's current combo, your hand as clickable cards grouped by rank (click a card to select, click a rank group header to select all, double-click to play, Esc clears), Play / Pass buttons enabled only when valid with the reason shown otherwise, exchange screen (choose N cards), animated replay of events with a speed control, "Ask <model>" panel listing ranked moves with scores and a button to apply one, round-over summary with roles, match-over summary, history log, records tab. State is server-side: reloading resumes the game.

- [ ] RED node tests for each helper; implement; then drive the real page in the browser (Chrome tools): full match against the champion on 4 seats, an exchange as Arschloch, an illegal selection message, ask-the-model, reload mid-game, a 3-seat and a 6-seat game, narrow window. Save screenshots; record in the ledger.
- [ ] commit `web: the play page`.

## Task 7: records view and comparison

- [ ] Records tab: per opponent (and per opponent set) games, mean role score, role distribution bars, next to the model's own measured score from `docs/baselines` when the opponent is the committed champion (read from `GET /api/records` fields only; no new data sources). RED test on the API shape, node test for the bar scaling helper, browser check. Commit `web: your record against each opponent`.

## Task 8: docs and review

- [ ] `docs/PLAYING.md` (how to play, ask-the-model, records), README/ROADMAP/ARCHITECTURE updates (Phase 11 progress: interactive part done, statistics overview still open), `docs/TRAINING.md` pointer.
- [ ] Whole-phase checks (fmt, clippy both toolchains, tests), fresh opus reviewer with the Review Focus list, one fix pass (RED then GREEN), minors to the ledger, push, PR, CI on three systems.
