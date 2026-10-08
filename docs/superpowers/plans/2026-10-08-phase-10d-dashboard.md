# Phase 10d — Browser Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Watch a training run in the browser, live while it trains and
afterwards: progress, fitness, the champion against each opponent,
species, complexity, a visualization of the champion's network, and an
inspector that shows why the champion chose one move over another.

**Architecture:** The `web` crate becomes the dashboard: an
`EventIndex` that reads `events.jsonl` incrementally, pure route handlers
over a run directory, a small std-only HTTP server (loopback only, one
thread per connection), and an embedded single-page app (vanilla JS and
inline SVG, no build step, no CDN). The trainer additionally records 12
real decisions of every new best champion (`decisions/gen-NNNN.json`);
the browser re-runs the network on them and checks itself against the
recorded Rust scores. `cli train --serve` and `cli watch` start the
dashboard. The dashboard only reads files, so it cannot affect training.

**Tech Stack:** Rust workspace (no new crates), `serde_json`, plain
browser JavaScript (ES modules), Node 22 for the JavaScript tests (the
Rust test harness skips them loudly if Node is missing).

**Spec:** `docs/superpowers/specs/2026-10-08-neat-engine-design.md`,
section 7a (live monitoring: the browser dashboard part). Task 7 records,
as section 7c, what was built and what was left out.

## How this plan was prepared

The code below was written into a scratch copy of the workspace first and
passed `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D
warnings` and `cargo test --workspace` (418 tests, including the Node
tests). The page was then looked at in a real browser against real
training runs, which found and fixed: the dashboard dying with the
training process (the browser then missed the last generations and the
"finished" state), run-together readouts, charts shrinking to illegibility
on narrow windows, and a page-level sideways scroll caused by grid items
defaulting to `min-width: auto`. The same real runs showed the browser's
JavaScript replay of the network matching the Rust-recorded scores
exactly (700+ candidates, maximum difference 0) and the page recovering
after the training process was killed with SIGKILL and resumed. Task 6
repeats those checks.

## Global Constraints

- No new Rust crates and no JavaScript dependencies or CDN: the page must work offline. `web` depends on `sim` (event types) and `serde_json`; `neat` only as a dev-dependency (tests).
- The dashboard server binds to `127.0.0.1` only, serves `GET` only, never reads outside the run directory (generation numbers in URLs are digits only), and bounds request size (16 KB) and time (5 s read/write timeouts). A bad request can only affect its own thread.
- The dashboard only **reads** the run directory (`events.jsonl`, `gen-NNNN.json`, `best.json`, `decisions/`); training results are identical with or without `--serve`.
- `events.jsonl` is consumed incrementally: only appended bytes are read, a partial last line is not consumed until its newline arrives, lines that do not parse are skipped, and a shrunken or replaced file (the trainer's resume rewrites it) resets the index and bumps `epoch`.
- All text that comes from the server is escaped before it enters the page's HTML.
- The browser's `forwardPass` must evaluate a genome exactly as the Rust `Network` does (inputs copied through, bias 1, `tanh` of the weighted sum over *enabled* connections); the decision inspector verifies this against the recorded scores on every replay.
- `cli train --serve` starts the dashboard before any run state is created (a busy port fails fast) and keeps serving after the run ends until Ctrl-C.
- The existing flat `cli` simulation arguments and `cli train`'s other options are unchanged.
- Between tasks the non-test build can report `dead_code`/unused warnings for items later tasks start using; any other warning is a defect.
- Per-task verification is `cargo test -p <crate>`; the full gate (`cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`) runs in Task 7.

## Review Focus

Failure modes the spec implies but a straightforward implementation tends to miss, each pinned by a named test or by Task 6:

1. **The server must survive hostile or broken clients and never leave the run directory:** `routes::tests::unknown_or_hostile_paths_are_404_and_never_leave_the_run_directory`, `server::tests::junk_and_oversized_requests_are_rejected_without_hurting_the_server`, `many_concurrent_clients_all_get_answers`, `binding_a_busy_port_is_an_error_not_a_panic`.
2. **A log being appended to, rewritten or half-written must not corrupt what the page shows:** `event_index::tests::a_partial_last_line_waits_for_its_newline`, `garbage_lines_are_skipped_not_fatal`, `a_rewritten_file_starts_over_and_bumps_the_epoch`, `a_replacement_that_grew_past_the_old_offset_is_still_detected`, `a_repeated_generation_replaces_the_earlier_one`.
3. **The browser's network replay must equal Rust's:** `lib.test.mjs` (hand-computed network, disabled edges ignored, cycle refused) and `replay.test.mjs` (every recorded decision of a real run, tolerance 1e-9), run by `cargo test -p web --test js`; and the live check in Task 6.
4. **The page must stay usable when the process serving it goes away or restarts:** Task 6 (SIGKILL, resume, reconnect with a contiguous history) and `dashboard_smoke::train_serve_shows_the_run_and_stays_up_after_it_finishes`.
5. **Operator mistakes fail loudly and early:** `dashboard_smoke::a_busy_port_fails_before_any_run_state_is_created` (also `cli watch` on a missing directory), and `watch_follows_a_run_that_starts_later` (an empty directory is a valid idle state).

---

## File Structure

```
sim/src/training/decisions.rs            # Task 1: record a champion's real decisions
sim/src/strategies/neat_player/mod.rs    # NeatStrategy::score_candidates
sim/src/training/{mod,run_dir,trainer}.rs, sim/src/{lib,strategies/mod}.rs   # Task 1 wiring
web/Cargo.toml, web/src/lib.rs
web/src/{event_index,routes,server,test_fixture}.rs    # Task 2
web/assets/lib.js                        # Task 3: pure helpers
web/assets/{index.html,style.css,app.js} # Task 4: the page
web/tests/{lib.test.mjs,replay.test.mjs,js.rs}          # Task 3
cli/Cargo.toml, cli/src/{main,train_args,train,watch}.rs, cli/tests/dashboard_smoke.rs   # Task 5
docs/ROADMAP.md, docs/ARCHITECTURE.md, docs/BUILDING.md, the spec   # Task 7
```

Not in this phase: per-champion Phase 4 statistics, mutation counts by kind, WebSocket/SSE, authentication or non-loopback binding, and the how-to-train guide.

---

### Task 1: Record a new best champion's real decisions

**Files:**
- Modify: `sim/src/strategies/neat_player/mod.rs`, `sim/src/strategies/mod.rs`, `sim/src/lib.rs`, `sim/src/training/mod.rs`, `sim/src/training/run_dir.rs`, `sim/src/training/trainer.rs`, `sim/tests/training_run.rs`
- Create: `sim/src/training/decisions.rs`

**Why:** The dashboard's decision inspector replays the champion's network in the browser on real situations. The recorded raw scores come from the Rust implementation, so the browser can verify its own replay against them.

**Interfaces:**
- Consumes: `NeatStrategy`, `TurnSummary`, `run_match`, `TableSpec`, `RunDir`, `Trainer`'s new-best branch (10c).
- Produces: `NeatStrategy::score_candidates(&[Move], DuplicateRule, &TurnContext) -> Vec<ScoredCandidate { candidate, features, raw_score, activation }>`; `record_decisions(&Genome, &TableSpec, &[Arc<dyn Strategy>], seed, limit, generation) -> DecisionFile`; `DecisionFile { generation, feature_set_version, feature_names, decisions: Vec<DecisionRecord { hand, table, opponent_hands, candidates: Vec<CandidateRecord { description, features, raw_score, activation }>, chosen }> }`; `RunDir::write_decisions`; the trainer writes `decisions/gen-NNNN.json` (12 decisions from one held-out match) for every new best champion.

- [ ] **Step 1: Edit**

In `sim/tests/training_run.rs` (Test first: a short run must leave a decisions file for its first (new best) champion), replace:

```rust
    assert!(NeatStrategy::from_file(&run.join("best.json")).is_ok());
    assert!(NeatStrategy::from_file(&run.join("gen-0002.json")).is_ok());
```

with:

```rust
    assert!(NeatStrategy::from_file(&run.join("best.json")).is_ok());
    assert!(NeatStrategy::from_file(&run.join("gen-0002.json")).is_ok());
    // A new best also records some of the champion's real decisions.
    let decisions: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(run.join("decisions/gen-0000.json")).unwrap()).unwrap();
    assert_eq!(decisions["generation"], 0);
    assert!(!decisions["decisions"].as_array().unwrap().is_empty());
```

- [ ] **Step 2: Run (expect failure)**

Run: `cargo test -p sim --test training_run a_short_run`

Expected: FAIL (the `decisions/gen-0000.json` file does not exist).

- [ ] **Step 3: Edit**

In `sim/src/training/mod.rs`, replace:

```rust
pub mod config;
```

with:

```rust
pub mod config;
pub mod decisions;
```

- [ ] **Step 4: Edit**

In `sim/src/training/mod.rs`, replace:

```rust
pub use evaluate::{
```

with:

```rust
pub use decisions::{record_decisions, DecisionFile, DecisionRecord};
pub use evaluate::{
```

- [ ] **Step 5: Create file**

Create `sim/src/training/decisions.rs` (Tests first):

```rust
#[cfg(test)]
mod tests {
    use engine::{Combo, DeckVariant};
    use neat::{InnovationTracker, NeatConfig};
    use rand::SeedableRng;

    use super::*;
    use crate::{CardCounter, LowestLegal, FEATURE_COUNT};

    fn genome(seed: u64) -> Genome {
        let mut tracker = InnovationTracker::new(u32::try_from(FEATURE_COUNT).unwrap() + 2);
        Genome::minimal(
            FEATURE_COUNT,
            &mut tracker,
            &NeatConfig::default(),
            &mut rand::rngs::StdRng::seed_from_u64(seed),
        )
    }

    fn table() -> TableSpec {
        TableSpec {
            player_count: 4,
            deck_variant: DeckVariant::Single,
            duplicate_rule: DuplicateRule::FirstDealtWins,
            rounds: 3,
        }
    }

    fn pool() -> Vec<Arc<dyn Strategy>> {
        vec![Arc::new(LowestLegal), Arc::new(CardCounter)]
    }

    #[test]
    fn cards_and_moves_have_readable_labels() {
        let ten = Card::new(Rank::Ten, Suit::Spades, 0);
        assert_eq!(card_label(&ten), "T♠");
        assert_eq!(move_label(&Move::Pass), "pass");
        let pair = Combo::new(vec![
            Card::new(Rank::Three, Suit::Diamonds, 0),
            Card::new(Rank::Three, Suit::Hearts, 0),
        ])
        .unwrap();
        assert_eq!(move_label(&Move::Play(pair)), "3♦ 3♥");
    }

    #[test]
    fn recorded_decisions_are_real_choices_that_match_the_scores() {
        let file = record_decisions(&genome(3), &table(), &pool(), 11, 12, 7);
        assert_eq!(file.generation, 7);
        assert_eq!(file.feature_names.len(), FEATURE_COUNT);
        assert!(!file.decisions.is_empty() && file.decisions.len() <= 12);
        for decision in &file.decisions {
            assert!(decision.candidates.len() >= 2);
            assert!(decision.chosen < decision.candidates.len());
            assert!(decision
                .candidates
                .iter()
                .all(|c| c.features.len() == FEATURE_COUNT));
            let best = decision
                .candidates
                .iter()
                .map(|c| c.raw_score)
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(
                (decision.candidates[decision.chosen].raw_score - best).abs() < 1e-12,
                "the champion plays its highest-scoring candidate"
            );
            for candidate in &decision.candidates {
                assert!((candidate.activation - candidate.raw_score.tanh()).abs() < 1e-12);
            }
            assert!(!decision.hand.is_empty());
        }
    }

    #[test]
    fn leading_decisions_have_no_table_and_following_ones_do() {
        let file = record_decisions(&genome(4), &table(), &pool(), 5, 200, 0);
        assert!(
            file.decisions.iter().any(|d| d.table.is_none()),
            "some leads"
        );
        assert!(
            file.decisions.iter().any(|d| d.table.is_some()),
            "some follows"
        );
        let following = file.decisions.iter().find(|d| d.table.is_some()).unwrap();
        assert!(following.candidates.iter().any(|c| c.description == "pass"));
        let leading = file.decisions.iter().find(|d| d.table.is_none()).unwrap();
        assert!(leading.candidates.iter().all(|c| c.description != "pass"));
    }

    #[test]
    fn recording_is_deterministic_and_respects_the_limit() {
        let a = record_decisions(&genome(3), &table(), &pool(), 11, 5, 1);
        let b = record_decisions(&genome(3), &table(), &pool(), 11, 5, 1);
        assert_eq!(a, b);
        assert!(a.decisions.len() <= 5);
        let all = record_decisions(&genome(3), &table(), &pool(), 11, usize::MAX, 1);
        assert!(all.decisions.len() >= a.decisions.len());
    }
}
```

- [ ] **Step 6: Run (expect failure)**

Run: `cargo test -p sim training::decisions`

Expected: FAIL (compile errors: `record_decisions`, `card_label` ... are not defined).

- [ ] **Step 7: Edit**

In `sim/src/strategies/neat_player/mod.rs` (`score_candidates`: every legal move with its features and score), replace:

```rust
impl Strategy for NeatStrategy {
```

with:

```rust
/// One legal move with the network's opinion of it, for explaining a
/// decision (see `NeatStrategy::score_candidates`).
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredCandidate {
    pub candidate: Move,
    pub features: [f64; FEATURE_COUNT],
    /// The output node's sum before `tanh`: what decisions are made on.
    pub raw_score: f64,
    /// `tanh(raw_score)`.
    pub activation: f64,
}

impl NeatStrategy {
    /// Every legal move with its feature vector and score, in the order
    /// given. `choose_play` picks the highest `raw_score` (ties: a play
    /// over a pass, then the smaller combo, then the first listed).
    #[must_use]
    pub fn score_candidates(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        context: &TurnContext<'_>,
    ) -> Vec<ScoredCandidate> {
        let summary = TurnSummary::new(context, duplicate_rule);
        let mut scratch = Vec::new();
        legal_moves
            .iter()
            .map(|candidate| {
                let features = summary.features(candidate);
                let raw_score = self.network.score(&features, &mut scratch);
                ScoredCandidate {
                    candidate: candidate.clone(),
                    features,
                    raw_score,
                    activation: raw_score.tanh(),
                }
            })
            .collect()
    }
}

impl Strategy for NeatStrategy {
```

- [ ] **Step 8: Edit**

In `sim/src/strategies/mod.rs`, replace:

```rust
NeatStrategy, TurnSummary,
```

with:

```rust
NeatStrategy, ScoredCandidate, TurnSummary,
```

- [ ] **Step 9: Edit**

In `sim/src/lib.rs`, replace:

```rust
NeatStrategy, RandomLegal, TurnSummary,
```

with:

```rust
NeatStrategy, RandomLegal, ScoredCandidate, TurnSummary,
```

- [ ] **Step 10: Implement**

Insert at the very top of `sim/src/training/decisions.rs`, above the `#[cfg(test)]` line:

```rust
//! A champion's recorded decisions, for the dashboard's decision
//! inspector: real situations from a real match, with every legal move,
//! its feature vector and the network's score, so a viewer (or the
//! browser, which re-runs the network) can see why one move beat another.

use std::sync::{Arc, Mutex};

use engine::{Card, DuplicateRule, Move, Rank, Suit};
use neat::Genome;
use serde::{Deserialize, Serialize};

use super::evaluate::TableSpec;
use crate::{
    run_match, MatchConfig, NeatStrategy, Strategy, TurnContext, FEATURE_NAMES, FEATURE_SET_VERSION,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateRecord {
    /// `pass` or the cards played, such as `9♠` or `3♦ 3♥`.
    pub description: String,
    pub features: Vec<f64>,
    /// The output node's sum before `tanh`: what the choice is made on.
    pub raw_score: f64,
    pub activation: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionRecord {
    /// The deciding seat's hand, weakest card first.
    pub hand: Vec<String>,
    /// The combo to beat, or `None` when leading.
    pub table: Option<String>,
    /// Hand sizes of the opponents still in the round.
    pub opponent_hands: Vec<usize>,
    pub candidates: Vec<CandidateRecord>,
    /// Index into `candidates` of the move the champion played.
    pub chosen: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionFile {
    pub generation: u32,
    pub feature_set_version: u32,
    pub feature_names: Vec<String>,
    pub decisions: Vec<DecisionRecord>,
}

#[must_use]
pub fn card_label(card: &Card) -> String {
    let rank = match card.rank {
        Rank::Two => "2",
        Rank::Three => "3",
        Rank::Four => "4",
        Rank::Five => "5",
        Rank::Six => "6",
        Rank::Seven => "7",
        Rank::Eight => "8",
        Rank::Nine => "9",
        Rank::Ten => "T",
        Rank::Jack => "J",
        Rank::Queen => "Q",
        Rank::King => "K",
        Rank::Ace => "A",
    };
    let suit = match card.suit {
        Suit::Diamonds => "♦",
        Suit::Hearts => "♥",
        Suit::Spades => "♠",
        Suit::Clubs => "♣",
    };
    format!("{rank}{suit}")
}

#[must_use]
pub fn move_label(candidate: &Move) -> String {
    match candidate {
        Move::Pass => "pass".to_owned(),
        Move::Play(combo) => combo
            .cards()
            .iter()
            .map(card_label)
            .collect::<Vec<_>>()
            .join(" "),
    }
}

/// Wraps a `NeatStrategy`, playing exactly as it does while logging each
/// decision with at least two options.
struct Recorder {
    inner: NeatStrategy,
    log: Mutex<Vec<DecisionRecord>>,
}

impl Strategy for Recorder {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn choose_play(
        &self,
        legal_moves: &[Move],
        duplicate_rule: DuplicateRule,
        context: &TurnContext<'_>,
        rng: &mut dyn rand::Rng,
    ) -> Move {
        let chosen_move = self
            .inner
            .choose_play(legal_moves, duplicate_rule, context, rng);
        if legal_moves.len() >= 2 {
            let mut hand = context.hand.to_vec();
            hand.sort_by(|a, b| a.compare(b, duplicate_rule));
            let candidates = self
                .inner
                .score_candidates(legal_moves, duplicate_rule, context)
                .into_iter()
                .map(|scored| CandidateRecord {
                    description: move_label(&scored.candidate),
                    features: scored.features.to_vec(),
                    raw_score: scored.raw_score,
                    activation: scored.activation,
                })
                .collect();
            let record = DecisionRecord {
                hand: hand.iter().map(card_label).collect(),
                table: context
                    .current_combo
                    .map(|combo| move_label(&Move::Play(combo.clone()))),
                opponent_hands: context
                    .opponents
                    .iter()
                    .filter(|o| o.active)
                    .map(|o| o.hand_size)
                    .collect(),
                candidates,
                chosen: legal_moves
                    .iter()
                    .position(|m| *m == chosen_move)
                    .expect("the chosen move is one of the legal moves"),
            };
            self.log.lock().expect("recorder lock").push(record);
        }
        chosen_move
    }

    fn choose_exchange_cards(
        &self,
        hand: &[Card],
        count: usize,
        duplicate_rule: DuplicateRule,
        rng: &mut dyn rand::Rng,
    ) -> Vec<Card> {
        self.inner
            .choose_exchange_cards(hand, count, duplicate_rule, rng)
    }
}

/// Plays one match with `genome` in seat 0 against pool members in the
/// other seats (in pool order, wrapping) and keeps up to `limit` of its
/// decisions that had a real choice, evenly spread over the match.
/// Deterministic for a given `seed`.
///
/// # Panics
///
/// Panics if `pool` is empty or the genome does not fit this build's
/// features (trained genomes always do).
#[must_use]
pub fn record_decisions(
    genome: &Genome,
    table: &TableSpec,
    pool: &[Arc<dyn Strategy>],
    seed: u64,
    limit: usize,
    generation: u32,
) -> DecisionFile {
    let recorder = Arc::new(Recorder {
        inner: NeatStrategy::new("champion", genome)
            .expect("trained genomes use this build's features"),
        log: Mutex::new(Vec::new()),
    });
    let strategies: Vec<Arc<dyn Strategy>> = (0..usize::from(table.player_count))
        .map(|seat| -> Arc<dyn Strategy> {
            if seat == 0 {
                recorder.clone()
            } else {
                pool[(seat - 1) % pool.len()].clone()
            }
        })
        .collect();
    let _ = run_match(
        &MatchConfig {
            player_count: table.player_count,
            deck_variant: table.deck_variant,
            duplicate_rule: table.duplicate_rule,
            rounds: table.rounds,
            seed,
        },
        &strategies,
    );
    let all = recorder.log.lock().expect("recorder lock").clone();
    let decisions = if all.len() <= limit {
        all
    } else {
        (0..limit)
            .map(|i| all[i * all.len() / limit].clone())
            .collect()
    };
    DecisionFile {
        generation,
        feature_set_version: FEATURE_SET_VERSION,
        feature_names: FEATURE_NAMES.iter().map(|&n| n.to_owned()).collect(),
        decisions,
    }
}
```

- [ ] **Step 11: Edit**

In `sim/src/training/run_dir.rs`, replace:

```rust
use super::events::{Event, ScoreStat, SCHEMA_VERSION};
```

with:

```rust
use super::decisions::DecisionFile;
use super::events::{Event, ScoreStat, SCHEMA_VERSION};
```

- [ ] **Step 12: Edit**

In `sim/src/training/run_dir.rs`, replace:

```rust
    /// # Errors
    ///
    /// As `write_champion`.
    pub fn write_best
```

with:

```rust
    /// Writes `decisions/gen-NNNN.json`.
    ///
    /// # Errors
    ///
    /// `TrainError::Io` on write failure.
    pub fn write_decisions(&self, decisions: &DecisionFile) -> Result<(), TrainError> {
        let directory = self.path("decisions");
        fs::create_dir_all(&directory).map_err(|e| io_error(&directory, &e))?;
        let text = serde_json::to_string(decisions).map_err(|e| TrainError::Config(e.to_string()))?;
        self.write_atomically(&format!("decisions/gen-{:04}.json", decisions.generation), &text)
    }

    /// # Errors
    ///
    /// As `write_champion`.
    pub fn write_best
```

- [ ] **Step 13: Edit**

In `sim/src/training/trainer.rs`, replace:

```rust
use super::config::TrainConfig;
```

with:

```rust
use super::config::TrainConfig;
use super::decisions::record_decisions;
```

- [ ] **Step 14: Edit**

In `sim/src/training/trainer.rs`, replace:

```rust
const PROGRESS_STEPS: usize = 10;
```

with:

```rust
const PROGRESS_STEPS: usize = 10;

/// Decisions recorded for each new best champion.
const DECISIONS_PER_BEST: usize = 12;
```

- [ ] **Step 15: Edit**

In `sim/src/training/trainer.rs`, replace:

```rust
    /// The champion's score on the held-out matches (see `heldout_seeds`).
```

with:

```rust
    /// A few real decisions of the champion (for the dashboard's decision
    /// inspector), from one held-out match.
    fn sample_decisions(
        &self,
        generation: u32,
        champion: &Genome,
    ) -> super::decisions::DecisionFile {
        let pool: Vec<Arc<dyn Strategy>> = self
            .opponents
            .iter()
            .map(|o| o.strategy.clone())
            .collect();
        record_decisions(
            champion,
            &self.config.table(),
            &pool,
            heldout_seeds(&self.config)[0],
            DECISIONS_PER_BEST,
            generation,
        )
    }

    /// The champion's score on the held-out matches (see `heldout_seeds`).
```

- [ ] **Step 16: Edit**

In `sim/src/training/trainer.rs`, replace:

```rust
            let heldout = self.confirm(&champion);
```

with:

```rust
            let heldout = self.confirm(&champion);
            self.dir
                .write_decisions(&self.sample_decisions(generation, &champion))?;
```

- [ ] **Step 17: Run (expect success)**

Run: `cargo fmt -p sim && cargo test -p sim`

Expected: PASS (4 new `decisions` tests: readable labels; recorded decisions are real choices whose chosen candidate has the highest score; leads have no table and follows offer a pass; deterministic and limited; plus the extended `a_short_run...` test and the rest of `sim`).

- [ ] **Step 18: Commit**

```bash
git add sim
git commit -F - <<'EOF'
sim: record a new best champion's real decisions for the dashboard (Phase 10d)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F3sC3JdcgTFAox87bBuikA
EOF
```


### Task 2: `web`: the dashboard server (event index, routes, HTTP)

**Files:**
- Modify: `web/Cargo.toml`, `web/src/lib.rs`
- Create: `web/src/{event_index,routes,server,test_fixture}.rs`, `web/assets/{index.html,app.js,lib.js,style.css}` (placeholders; Tasks 3 and 4 replace them)

**Why:** The dashboard only ever *reads* the run's files, so a slow or closed browser cannot affect training. `EventIndex` reads only the bytes appended since the last poll, waits for a partial last line to complete, skips lines that do not parse, and starts over (bumping `epoch`) when the log is rewritten, which is what the trainer's resume does.

**Interfaces:**
- Consumes: `sim::training::{Event, GenerationEvent, RunStart, SCHEMA_VERSION, Trainer, ...}`, the run directory layout (10c) and `decisions/` (Task 1).
- Produces: `web::Dashboard::{start(run_dir, port) -> io::Result<Dashboard>, address(), url(), wait()}`; `web::App::{new(run_dir), handle(method, target) -> Response}`; `web::EventIndex`. Routes: `GET /`, `/app.js`, `/lib.js`, `/style.css`, `/api/state`, `/api/events?since=N`, `/api/genome/N|best`, `/api/decisions/N`. Loopback only; std-only blocking HTTP, one thread per connection with 5 s read/write timeouts; GET only.

- [ ] **Step 1: Create file**

Create `web/Cargo.toml` (`sim` for the event types, `serde_json`; `neat` only for tests):

```toml
[package]
name = "web"
version.workspace = true
edition.workspace = true
license.workspace = true

[lints]
workspace = true

[dependencies]
serde_json = { version = "1.0.151", features = ["float_roundtrip"] }
sim = { version = "0.1.0", path = "../sim" }

[dev-dependencies]
neat = { version = "0.1.0", path = "../neat" }
```

- [ ] **Step 2: Create file**

Create `web/src/lib.rs`:

```rust
//! The training dashboard: a small HTTP server that shows a run directory
//! (`events.jsonl`, `gen-NNNN.json`, `best.json`, `decisions/`) in a
//! browser, live while training and afterwards. It only ever *reads* the
//! run's files, so a slow or closed browser cannot affect training.
//! See docs/superpowers/specs/2026-10-08-neat-engine-design.md, 7a.

mod event_index;
mod routes;
mod server;
#[cfg(test)]
mod test_fixture;

pub use event_index::EventIndex;
pub use routes::{App, Response};
pub use server::Dashboard;
```

- [ ] **Step 3: Create file**

Create `web/assets/index.html` (Placeholder (Task 4 replaces it)):

```
<!doctype html><html><head><title>placeholder</title></head><body></body></html>
```

- [ ] **Step 4: Create file**

Create `web/assets/app.js`:

```
// fetch (placeholder)
```

- [ ] **Step 5: Create file**

Create `web/assets/lib.js`:

```
export const placeholder = 1;
```

- [ ] **Step 6: Create file**

Create `web/assets/style.css`:

```
:root { --placeholder: 1; }
```

- [ ] **Step 7: Create file**

Create `web/src/test_fixture.rs` (Test support: a real tiny training run to serve and read):

```rust
//! Test support: a real (tiny) training run to serve and read.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use neat::NeatConfig;
use sim::training::{DeckChoice, DuplicateChoice, Opponent, TrainConfig, TrainObserver, Trainer};
use sim::{LowestLegal, RandomLegal};

pub fn temp_path(name: &str) -> PathBuf {
    let path =
        std::env::temp_dir().join(format!("arschloch-web-{name}-{}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&path);
    path
}

struct Silent;
impl TrainObserver for Silent {}

/// A finished 3-generation run, created once per test process.
pub fn fixture_run_dir() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir =
            std::env::temp_dir().join(format!("arschloch-web-fixture-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let config = TrainConfig {
            seed: 9,
            player_count: 4,
            deck: DeckChoice::Single,
            duplicate_rule: DuplicateChoice::FirstDealtWins,
            rounds_per_match: 3,
            matches_per_genome: 4,
            reeval_matches: 6,
            generations: 3,
            neat: NeatConfig {
                population_size: 12,
                ..NeatConfig::default()
            },
            opponent_specs: vec!["lowest-legal".into(), "random-legal".into()],
        };
        let opponents = vec![
            Opponent {
                name: "LowestLegal".into(),
                strategy: Arc::new(LowestLegal),
            },
            Opponent {
                name: "RandomLegal".into(),
                strategy: Arc::new(RandomLegal),
            },
        ];
        Trainer::new(config, opponents, &dir)
            .unwrap()
            .run(&mut Silent)
            .unwrap();
        dir
    })
    .clone()
}

/// The fixture run's `events.jsonl` lines: start, three generations, end.
pub fn sample_events() -> Vec<String> {
    std::fs::read_to_string(fixture_run_dir().join("events.jsonl"))
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect()
}
```

- [ ] **Step 8: Create file**

Create `web/src/event_index.rs` (Tests first):

```rust
#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Write;

    use super::*;
    use crate::test_fixture::{sample_events, temp_path};

    fn append(path: &std::path::Path, text: &str) {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        file.write_all(text.as_bytes()).unwrap();
    }

    #[test]
    fn a_missing_file_is_just_empty() {
        let mut index = EventIndex::new(temp_path("missing"));
        index.refresh();
        assert!(index.generations().is_empty() && index.run_start().is_none());
        assert!(!index.finished());
        assert_eq!(index.epoch(), 0);
    }

    #[test]
    fn it_reads_incrementally_and_tracks_start_and_end() {
        let lines = sample_events();
        let path = temp_path("incremental");
        let mut index = EventIndex::new(path.clone());
        append(&path, &format!("{}\n{}\n", lines[0], lines[1]));
        index.refresh();
        assert!(index.run_start().is_some());
        assert_eq!(index.generations().len(), 1);
        assert!(!index.finished());
        append(
            &path,
            &format!("{}\n{}\n", lines[2], lines[lines.len() - 1]),
        );
        index.refresh();
        assert_eq!(index.generations().len(), 2);
        assert!(index.finished(), "the last line is the run end");
        assert_eq!(index.epoch(), 0, "appends never bump the epoch");
        fs::remove_file(path).ok();
    }

    #[test]
    fn a_partial_last_line_waits_for_its_newline() {
        let lines = sample_events();
        let path = temp_path("partial");
        let mut index = EventIndex::new(path.clone());
        let half = lines[1].len() / 2;
        append(&path, &format!("{}\n{}", lines[0], &lines[1][..half]));
        index.refresh();
        assert!(
            index.generations().is_empty(),
            "half a line is not an event"
        );
        append(&path, &format!("{}\n", &lines[1][half..]));
        index.refresh();
        assert_eq!(index.generations().len(), 1);
        fs::remove_file(path).ok();
    }

    #[test]
    fn garbage_lines_are_skipped_not_fatal() {
        let lines = sample_events();
        let path = temp_path("garbage");
        append(
            &path,
            &format!(
                "{}\nnot json at all\n{{\"type\":\"nonsense\"}}\n{}\n",
                lines[0], lines[1]
            ),
        );
        let mut index = EventIndex::new(path.clone());
        index.refresh();
        assert_eq!(index.generations().len(), 1);
        fs::remove_file(path).ok();
    }

    #[test]
    fn a_rewritten_file_starts_over_and_bumps_the_epoch() {
        let lines = sample_events();
        let path = temp_path("rewrite");
        append(
            &path,
            &format!("{}\n{}\n{}\n", lines[0], lines[1], lines[2]),
        );
        let mut index = EventIndex::new(path.clone());
        index.refresh();
        assert_eq!(index.generations().len(), 2);
        // The trainer's resume rewrites the file (shorter, new inode).
        let replacement = path.with_extension("tmp");
        fs::write(&replacement, format!("{}\n{}\n", lines[0], lines[1])).unwrap();
        fs::rename(&replacement, &path).unwrap();
        index.refresh();
        assert_eq!(index.epoch(), 1);
        assert_eq!(index.generations().len(), 1);
        fs::remove_file(path).ok();
    }

    #[test]
    fn a_replacement_that_grew_past_the_old_offset_is_still_detected() {
        let lines = sample_events();
        let path = temp_path("regrew");
        append(&path, &format!("{}\n{}\n", lines[0], lines[1]));
        let mut index = EventIndex::new(path.clone());
        index.refresh();
        let replacement = path.with_extension("tmp");
        fs::write(
            &replacement,
            format!("{}\n{}\n{}\n{}\n", lines[0], lines[1], lines[2], lines[3]),
        )
        .unwrap();
        fs::rename(&replacement, &path).unwrap();
        index.refresh();
        assert_eq!(index.epoch(), 1, "same-or-longer replacement is a new file");
        assert_eq!(index.generations().len(), 3);
        fs::remove_file(path).ok();
    }

    #[test]
    fn a_repeated_generation_replaces_the_earlier_one() {
        let lines = sample_events();
        let path = temp_path("repeat");
        append(
            &path,
            &format!("{}\n{}\n{}\n", lines[0], lines[1], lines[1]),
        );
        let mut index = EventIndex::new(path.clone());
        index.refresh();
        assert_eq!(index.generations().len(), 1);
        fs::remove_file(path).ok();
    }
}
```

- [ ] **Step 9: Create file**

Create `web/src/routes.rs` (Tests first):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixture::fixture_run_dir;

    fn json_of(response: &Response) -> serde_json::Value {
        serde_json::from_slice(&response.body).expect("a JSON body")
    }

    fn app() -> App {
        App::new(fixture_run_dir())
    }

    #[test]
    fn the_page_and_its_assets_are_served_with_the_right_types() {
        let app = app();
        for (path, content_type, marker) in [
            ("/", "text/html", "<title>"),
            ("/app.js", "text/javascript", "fetch"),
            ("/lib.js", "text/javascript", "export"),
            ("/style.css", "text/css", "--"),
        ] {
            let response = app.handle("GET", path);
            assert_eq!(response.status, 200, "{path}");
            assert!(response.content_type.starts_with(content_type), "{path}");
            assert!(
                String::from_utf8_lossy(&response.body).contains(marker),
                "{path}"
            );
        }
    }

    #[test]
    fn state_describes_the_run() {
        let response = app().handle("GET", "/api/state");
        assert_eq!(response.status, 200);
        let state = json_of(&response);
        assert_eq!(state["schema_version"], SCHEMA_VERSION);
        assert_eq!(state["generations_logged"], 3);
        assert_eq!(state["last_generation"], 2);
        assert_eq!(state["finished"], true);
        assert_eq!(state["run_start"]["config"]["player_count"], 4);
        assert_eq!(state["run_start"]["opponents"][0], "LowestLegal");
    }

    #[test]
    fn events_can_be_fetched_whole_or_after_a_generation() {
        let app = app();
        let all = json_of(&app.handle("GET", "/api/events"));
        assert_eq!(all["events"].as_array().unwrap().len(), 3);
        let after = json_of(&app.handle("GET", "/api/events?since=0"));
        let generations: Vec<u64> = after["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["generation"].as_u64().unwrap())
            .collect();
        assert_eq!(generations, vec![1, 2]);
        let none = json_of(&app.handle("GET", "/api/events?since=2"));
        assert!(none["events"].as_array().unwrap().is_empty());
        assert_eq!(all["epoch"], 0);
        assert_eq!(all["finished"], true);
    }

    #[test]
    fn a_malformed_since_is_a_400_not_a_guess() {
        for query in ["since=abc", "since=-1", "since=", "since=1.5"] {
            let response = app().handle("GET", &format!("/api/events?{query}"));
            assert_eq!(response.status, 400, "{query}");
        }
    }

    #[test]
    fn genomes_and_decisions_come_from_the_run_directory() {
        let app = app();
        let genome = json_of(&app.handle("GET", "/api/genome/0"));
        assert!(genome["genome"]["nodes"].is_array());
        assert_eq!(
            genome["feature_names"].as_array().unwrap().len(),
            sim::FEATURE_COUNT
        );
        let best = app.handle("GET", "/api/genome/best");
        assert_eq!(best.status, 200);
        let decisions = json_of(&app.handle("GET", "/api/decisions/0"));
        assert_eq!(decisions["generation"], 0);
        assert!(!decisions["decisions"].as_array().unwrap().is_empty());
    }

    #[test]
    fn unknown_or_hostile_paths_are_404_and_never_leave_the_run_directory() {
        let app = app();
        for path in [
            "/nope",
            "/api/genome/999",
            "/api/genome/..%2f..%2fetc%2fpasswd",
            "/api/genome/../../etc/passwd",
            "/api/genome/-1",
            "/api/genome/0x10",
            "/api/decisions/",
            "/api/decisions/best",
            "/api/events/extra",
        ] {
            assert_eq!(app.handle("GET", path).status, 404, "{path}");
        }
    }

    #[test]
    fn only_get_is_supported() {
        for method in ["POST", "PUT", "DELETE", "HEAD"] {
            assert_eq!(app().handle(method, "/api/state").status, 405, "{method}");
        }
    }

    #[test]
    fn an_empty_directory_is_a_valid_idle_state() {
        let empty =
            std::env::temp_dir().join(format!("arschloch-web-empty-{}", std::process::id()));
        std::fs::create_dir_all(&empty).unwrap();
        let app = App::new(empty.clone());
        let state = json_of(&app.handle("GET", "/api/state"));
        assert_eq!(state["generations_logged"], 0);
        assert!(state["run_start"].is_null() && state["last_generation"].is_null());
        assert_eq!(
            json_of(&app.handle("GET", "/api/events"))["events"]
                .as_array()
                .unwrap()
                .len(),
            0
        );
        assert_eq!(app.handle("GET", "/api/genome/best").status, 404);
        std::fs::remove_dir_all(empty).ok();
    }
}
```

- [ ] **Step 10: Create file**

Create `web/src/server.rs` (Tests first):

```rust
#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpStream;

    use super::*;
    use crate::test_fixture::fixture_run_dir;

    fn raw_request(address: SocketAddr, request: &[u8]) -> String {
        let mut stream = TcpStream::connect(address).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        stream.write_all(request).unwrap();
        let mut reply = Vec::new();
        stream.read_to_end(&mut reply).unwrap();
        String::from_utf8_lossy(&reply).into_owned()
    }

    fn get(address: SocketAddr, target: &str) -> String {
        raw_request(
            address,
            format!("GET {target} HTTP/1.1\r\nHost: localhost\r\n\r\n").as_bytes(),
        )
    }

    fn body(reply: &str) -> &str {
        reply.split_once("\r\n\r\n").map_or("", |(_, b)| b)
    }

    #[test]
    fn it_serves_the_api_over_a_real_socket() {
        let dashboard = Dashboard::start(fixture_run_dir(), 0).unwrap();
        assert!(dashboard.address().ip().is_loopback());
        let reply = get(dashboard.address(), "/api/state");
        assert!(reply.starts_with("HTTP/1.1 200 OK\r\n"), "{reply}");
        assert!(reply.contains("Content-Type: application/json"));
        assert!(reply.contains("Cache-Control: no-store"));
        let state: serde_json::Value = serde_json::from_str(body(&reply)).unwrap();
        assert_eq!(state["generations_logged"], 3);
        let declared: usize = reply
            .lines()
            .find_map(|l| l.strip_prefix("Content-Length: "))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert_eq!(declared, body(&reply).len(), "Content-Length is exact");
        let page = get(dashboard.address(), "/");
        assert!(page.contains("text/html") && body(&page).contains("<title>"));
    }

    #[test]
    fn errors_come_back_as_proper_statuses() {
        let dashboard = Dashboard::start(fixture_run_dir(), 0).unwrap();
        assert!(get(dashboard.address(), "/nope").starts_with("HTTP/1.1 404 Not Found"));
        assert!(
            get(dashboard.address(), "/api/events?since=x").starts_with("HTTP/1.1 400 Bad Request")
        );
        let post = raw_request(
            dashboard.address(),
            b"POST /api/state HTTP/1.1\r\nContent-Length: 0\r\n\r\n",
        );
        assert!(
            post.starts_with("HTTP/1.1 405 Method Not Allowed"),
            "{post}"
        );
    }

    #[test]
    fn junk_and_oversized_requests_are_rejected_without_hurting_the_server() {
        let dashboard = Dashboard::start(fixture_run_dir(), 0).unwrap();
        let junk = raw_request(dashboard.address(), b"\x00\x01\x02 not http\r\n\r\n");
        assert!(
            junk.starts_with("HTTP/1.1 400")
                || junk.starts_with("HTTP/1.1 405")
                || junk.starts_with("HTTP/1.1 404"),
            "{junk}"
        );
        let huge = vec![b'a'; MAX_REQUEST_BYTES + 4096];
        let mut stream = TcpStream::connect(dashboard.address()).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        let _ = stream.write_all(&huge);
        let mut reply = Vec::new();
        let _ = stream.read_to_end(&mut reply);
        assert!(String::from_utf8_lossy(&reply).starts_with("HTTP/1.1 400") || reply.is_empty());
        assert!(
            get(dashboard.address(), "/api/state").starts_with("HTTP/1.1 200"),
            "still serving"
        );
    }

    #[test]
    fn many_concurrent_clients_all_get_answers() {
        let dashboard = Dashboard::start(fixture_run_dir(), 0).unwrap();
        let address = dashboard.address();
        let handles: Vec<_> = (0..16)
            .map(|_| thread::spawn(move || get(address, "/api/events?since=0")))
            .collect();
        for handle in handles {
            assert!(handle.join().unwrap().starts_with("HTTP/1.1 200"));
        }
    }

    #[test]
    fn binding_a_busy_port_is_an_error_not_a_panic() {
        let first = Dashboard::start(fixture_run_dir(), 0).unwrap();
        let second = Dashboard::start(fixture_run_dir(), first.address().port());
        assert!(second.is_err());
    }
}
```

- [ ] **Step 11: Run (expect failure)**

Run: `cargo test -p web`

Expected: FAIL (compile errors: `EventIndex`, `App`, `Dashboard` are not defined).

- [ ] **Step 12: Implement**

Insert at the very top of `web/src/event_index.rs`, above the `#[cfg(test)]` line:

```rust
//! An incremental reader of a run's `events.jsonl`.
//!
//! The trainer appends one JSON line per event, and on resume rewrites
//! the file (trimming events after the checkpoint). The index therefore:
//! reads only the bytes added since the last refresh; keeps a partial
//! last line unread until its newline arrives; skips lines that do not
//! parse; and starts over (bumping `epoch`) when the file shrinks or is
//! replaced, so a client can tell its cached events are stale.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use sim::training::{Event, GenerationEvent, RunStart};

pub struct EventIndex {
    path: PathBuf,
    offset: u64,
    identity: Option<u64>,
    epoch: u64,
    generations: Vec<GenerationEvent>,
    run_start: Option<RunStart>,
    finished: bool,
}

#[cfg(unix)]
#[allow(clippy::unnecessary_wraps)] // the non-unix variant has no file identity to offer
fn identity(metadata: &std::fs::Metadata) -> Option<u64> {
    use std::os::unix::fs::MetadataExt;
    Some(metadata.ino())
}

#[cfg(not(unix))]
fn identity(_: &std::fs::Metadata) -> Option<u64> {
    None
}

impl EventIndex {
    #[must_use]
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            offset: 0,
            identity: None,
            epoch: 0,
            generations: Vec::new(),
            run_start: None,
            finished: false,
        }
    }

    /// Bumped whenever previously returned events may no longer be valid.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    #[must_use]
    pub fn finished(&self) -> bool {
        self.finished
    }

    #[must_use]
    pub fn run_start(&self) -> Option<&RunStart> {
        self.run_start.as_ref()
    }

    #[must_use]
    pub fn generations(&self) -> &[GenerationEvent] {
        &self.generations
    }

    fn reset(&mut self) {
        let had_content = self.offset > 0 || !self.generations.is_empty();
        self.offset = 0;
        self.generations.clear();
        self.run_start = None;
        self.finished = false;
        if had_content {
            self.epoch += 1;
        }
    }

    /// Reads whatever was appended since the last call. A missing or
    /// unreadable file simply means "nothing (yet)".
    pub fn refresh(&mut self) {
        let Ok(mut file) = File::open(&self.path) else {
            self.reset();
            return;
        };
        let Ok(metadata) = file.metadata() else {
            return;
        };
        let current = identity(&metadata);
        if (self.identity.is_some() && current != self.identity) || metadata.len() < self.offset {
            self.reset();
        }
        self.identity = current;
        if metadata.len() == self.offset || file.seek(SeekFrom::Start(self.offset)).is_err() {
            return;
        }
        let mut added = Vec::new();
        if file.read_to_end(&mut added).is_err() {
            return;
        }
        // Only complete lines: the trainer may be mid-write.
        let Some(end) = added.iter().rposition(|&b| b == b'\n') else {
            return;
        };
        self.offset += (end + 1) as u64;
        for line in added[..end].split(|&b| b == b'\n') {
            if let Ok(event) = serde_json::from_slice::<Event>(line) {
                self.apply(event);
            }
        }
    }

    fn apply(&mut self, event: Event) {
        match event {
            Event::RunStart(start) => {
                self.run_start = Some(*start);
                self.finished = false;
            }
            Event::Generation(generation) => {
                match self
                    .generations
                    .iter_mut()
                    .find(|g| g.generation == generation.generation)
                {
                    Some(existing) => *existing = *generation,
                    None => self.generations.push(*generation),
                }
            }
            Event::RunEnd(_) => self.finished = true,
        }
    }
}
```

- [ ] **Step 13: Implement**

Insert at the very top of `web/src/routes.rs`, above the `#[cfg(test)]` line:

```rust
//! The dashboard's HTTP routes as a pure function of (method, target):
//! testable without sockets.
//!
//! - `GET /`, `/app.js`, `/lib.js`, `/style.css`: the embedded page;
//! - `GET /api/state`: run settings, progress and whether it finished;
//! - `GET /api/events?since=N`: generation events after generation `N`;
//! - `GET /api/genome/N|best`: a champion genome file;
//! - `GET /api/decisions/N`: the recorded decisions of a new-best champion.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::json;
use sim::training::SCHEMA_VERSION;

use crate::event_index::EventIndex;

pub struct Response {
    pub status: u16,
    pub content_type: &'static str,
    pub body: Vec<u8>,
}

impl Response {
    fn json(status: u16, value: &serde_json::Value) -> Self {
        Self {
            status,
            content_type: "application/json; charset=utf-8",
            body: serde_json::to_vec(value).expect("json serializes"),
        }
    }

    fn error(status: u16, message: &str) -> Self {
        Self::json(status, &json!({ "error": message }))
    }

    fn asset(content_type: &'static str, text: &'static str) -> Self {
        Self {
            status: 200,
            content_type,
            body: text.as_bytes().to_vec(),
        }
    }
}

pub struct App {
    run_dir: PathBuf,
    index: Mutex<EventIndex>,
}

/// A generation number from a URL segment: plain digits only.
fn parse_generation(segment: &str) -> Option<u32> {
    if segment.is_empty() || !segment.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    segment.parse().ok()
}

fn query_value<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .find(|(k, _)| *k == key)
        .map(|(_, v)| v)
}

impl App {
    #[must_use]
    pub fn new(run_dir: PathBuf) -> Self {
        let index = Mutex::new(EventIndex::new(run_dir.join("events.jsonl")));
        Self { run_dir, index }
    }

    #[must_use]
    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }

    /// Handles one request. `target` is the request target as sent
    /// (path plus optional query).
    #[must_use]
    pub fn handle(&self, method: &str, target: &str) -> Response {
        if method != "GET" {
            return Response::error(405, "only GET is supported");
        }
        let (path, query) = target.split_once('?').unwrap_or((target, ""));
        match path {
            "/" => Response::asset(
                "text/html; charset=utf-8",
                include_str!("../assets/index.html"),
            ),
            "/app.js" => Response::asset(
                "text/javascript; charset=utf-8",
                include_str!("../assets/app.js"),
            ),
            "/lib.js" => Response::asset(
                "text/javascript; charset=utf-8",
                include_str!("../assets/lib.js"),
            ),
            "/style.css" => Response::asset(
                "text/css; charset=utf-8",
                include_str!("../assets/style.css"),
            ),
            "/api/state" => self.state(),
            "/api/events" => self.events(query),
            _ => {
                if let Some(rest) = path.strip_prefix("/api/genome/") {
                    return self.genome(rest);
                }
                if let Some(rest) = path.strip_prefix("/api/decisions/") {
                    return self.decisions(rest);
                }
                Response::error(404, "not found")
            }
        }
    }

    fn refreshed<T>(&self, read: impl FnOnce(&EventIndex) -> T) -> T {
        let mut index = self.index.lock().expect("index lock");
        index.refresh();
        read(&index)
    }

    fn state(&self) -> Response {
        let value = self.refreshed(|index| {
            json!({
                "schema_version": SCHEMA_VERSION,
                "run_start": index.run_start(),
                "generations_logged": index.generations().len(),
                "last_generation": index.generations().iter().map(|g| g.generation).max(),
                "finished": index.finished(),
                "epoch": index.epoch(),
            })
        });
        Response::json(200, &value)
    }

    fn events(&self, query: &str) -> Response {
        let since = match query_value(query, "since") {
            None => None,
            Some(text) => match parse_generation(text) {
                Some(generation) => Some(generation),
                None => return Response::error(400, "since must be a generation number"),
            },
        };
        let value = self.refreshed(|index| {
            let events: Vec<&sim::training::GenerationEvent> = index
                .generations()
                .iter()
                .filter(|g| since.is_none_or(|s| g.generation > s))
                .collect();
            json!({ "epoch": index.epoch(), "finished": index.finished(), "events": events })
        });
        Response::json(200, &value)
    }

    fn file(&self, relative: &str) -> Response {
        match std::fs::read(self.run_dir.join(relative)) {
            Ok(body) => Response {
                status: 200,
                content_type: "application/json; charset=utf-8",
                body,
            },
            Err(_) => Response::error(404, "not found"),
        }
    }

    fn genome(&self, which: &str) -> Response {
        if which == "best" {
            return self.file("best.json");
        }
        match parse_generation(which) {
            Some(generation) => self.file(&format!("gen-{generation:04}.json")),
            None => Response::error(404, "not found"),
        }
    }

    fn decisions(&self, which: &str) -> Response {
        match parse_generation(which) {
            Some(generation) => self.file(&format!("decisions/gen-{generation:04}.json")),
            None => Response::error(404, "not found"),
        }
    }
}
```

- [ ] **Step 14: Implement**

Insert at the very top of `web/src/server.rs`, above the `#[cfg(test)]` line:

```rust
//! A deliberately small blocking HTTP/1.1 server on `std::net`: one
//! thread accepts, one short-lived thread serves each connection, every
//! response closes the connection. It binds to the loopback interface
//! only (use an SSH tunnel to view a remote run), and a slow or stuck
//! client can only ever hold its own thread (reads and writes time out).

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::routes::{App, Response};

const IO_TIMEOUT: Duration = Duration::from_secs(5);
/// Requests are tiny `GET`s; anything larger is not a browser.
const MAX_REQUEST_BYTES: usize = 16 * 1024;

pub struct Dashboard {
    address: SocketAddr,
    accept_thread: JoinHandle<()>,
}

impl Dashboard {
    /// Starts serving `run_dir` on `127.0.0.1:port` (`0` picks a free
    /// port) in background threads. The threads live until the process
    /// exits.
    ///
    /// # Errors
    ///
    /// Returns the bind error (for example, the port is in use).
    pub fn start(run_dir: PathBuf, port: u16) -> std::io::Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", port))?;
        let address = listener.local_addr()?;
        let app = Arc::new(App::new(run_dir));
        let accept_thread = thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let app = Arc::clone(&app);
                thread::spawn(move || serve_connection(stream, &app));
            }
        });
        Ok(Self {
            address,
            accept_thread,
        })
    }

    #[must_use]
    pub fn address(&self) -> SocketAddr {
        self.address
    }

    #[must_use]
    pub fn url(&self) -> String {
        format!("http://{}/", self.address)
    }

    /// Blocks for as long as the server runs (until the process exits).
    pub fn wait(self) {
        let _ = self.accept_thread.join();
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        _ => "Internal Server Error",
    }
}

/// Reads the request head and returns `(method, target)`.
fn read_request(stream: &mut TcpStream) -> Option<(String, String)> {
    let mut received = Vec::new();
    let mut chunk = [0u8; 1024];
    while !received.windows(4).any(|w| w == b"\r\n\r\n") {
        if received.len() > MAX_REQUEST_BYTES {
            return None;
        }
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            return None;
        }
        received.extend_from_slice(&chunk[..read]);
    }
    let head = String::from_utf8_lossy(&received);
    let mut parts = head.lines().next()?.split_whitespace();
    let (method, target) = (parts.next()?, parts.next()?);
    Some((method.to_owned(), target.to_owned()))
}

fn write_response(stream: &mut TcpStream, response: &Response) {
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n",
        response.status,
        reason(response.status),
        response.content_type,
        response.body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(&response.body);
    let _ = stream.flush();
}

fn serve_connection(mut stream: TcpStream, app: &App) {
    let _ = stream.set_read_timeout(Some(IO_TIMEOUT));
    let _ = stream.set_write_timeout(Some(IO_TIMEOUT));
    let response = match read_request(&mut stream) {
        Some((method, target)) => app.handle(&method, &target),
        None => Response {
            status: 400,
            content_type: "text/plain; charset=utf-8",
            body: b"bad request".to_vec(),
        },
    };
    write_response(&mut stream, &response);
}
```

- [ ] **Step 15: Run (expect success)**

Run: `cargo fmt -p web && cargo test -p web`

Expected: PASS (20 tests: the index reads incrementally, waits for partial lines, skips garbage, resets on a rewritten or regrown file; the routes serve assets with the right types, state, events after a generation, genomes and decisions, answer 400/404/405 correctly and never leave the run directory; over a real socket the server answers with exact `Content-Length`, survives junk and oversized requests, handles 16 concurrent clients, and reports a busy port as an error).

- [ ] **Step 16: Commit**

```bash
git add web Cargo.lock
git commit -F - <<'EOF'
web: add the dashboard server over a run directory (Phase 10d)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F3sC3JdcgTFAox87bBuikA
EOF
```


### Task 3: Dashboard JavaScript helpers (tested in Node, and against Rust)

**Files:**
- Modify: `web/assets/lib.js`
- Create: `web/tests/lib.test.mjs`, `web/tests/replay.test.mjs`, `web/tests/js.rs`

**Why:** The decision inspector shows node activations computed in the browser. The recorded scores come from Rust, so a replay test against a real run proves the two implementations agree.

**Interfaces:**
- Consumes: Node 22 (`node --test`), the genome and decision file formats.
- Produces: `lib.js` ES module: `fmt`, `formatDuration`, `niceScale`, `linearScale`, `linePath`, `bandPath`, `stackSpecies`, `speciesColor`, `layoutNetwork`, `forwardPass` (the Rust `Network` evaluated in JavaScript: inputs copied through, bias 1, `tanh` of the weighted sum over *enabled* connections, `raw` pre-activation sums), `divergingColor`, `edgeWidth`, `histogramOpacities`, `nearestIndex`. `cargo test -p web --test js` runs the Node tests (skipped with a loud message if Node is missing) including a replay of every recorded decision of a real run.

- [ ] **Step 1: Create file**

Create `web/tests/lib.test.mjs` (Tests first):

```
// Run with: node --test web/tests
import assert from "node:assert/strict";
import test from "node:test";
import {
  bandPath, divergingColor, edgeWidth, fmt, forwardPass, formatDuration, histogramOpacities,
  layoutNetwork, linePath, linearScale, nearestIndex, niceScale, speciesColor, stackSpecies,
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
```

- [ ] **Step 2: Create file**

Create `web/tests/replay.test.mjs` (Cross-language replay test (needs a real run, supplied by `js.rs`)):

```
// Cross-language check: the browser's network replay must reproduce the
// scores the Rust implementation recorded for real decisions.
// Needs RUN_DIR (a run directory written by `cli train`); skipped otherwise.
import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import test from "node:test";
import { forwardPass } from "../assets/lib.js";

const runDir = process.env.RUN_DIR;

test("browser replay matches the recorded Rust scores", { skip: !runDir && "RUN_DIR not set" }, () => {
  const decisionDir = `${runDir}/decisions`;
  assert.ok(existsSync(decisionDir), "the run recorded decisions");
  const files = readdirSync(decisionDir).filter((f) => f.endsWith(".json"));
  assert.ok(files.length > 0, "at least one new-best champion has decisions");
  let candidates = 0;
  for (const name of files) {
    const decisions = JSON.parse(readFileSync(`${decisionDir}/${name}`, "utf8"));
    const generation = String(decisions.generation).padStart(4, "0");
    const file = JSON.parse(readFileSync(`${runDir}/gen-${generation}.json`, "utf8"));
    assert.deepEqual(decisions.feature_names, file.feature_names);
    const output = file.genome.nodes.find((n) => n.kind === "Output").id;
    for (const decision of decisions.decisions) {
      for (const candidate of decision.candidates) {
        const { values, raw } = forwardPass(file.genome, candidate.features);
        assert.ok(Math.abs(raw.get(output) - candidate.raw_score) < 1e-9, `${name}: raw score`);
        assert.ok(Math.abs(values.get(output) - candidate.activation) < 1e-9, `${name}: activation`);
        candidates += 1;
      }
    }
  }
  assert.ok(candidates > 0);
});
```

- [ ] **Step 3: Create file**

Create `web/tests/js.rs` (Runs both under `cargo test`):

```rust
//! Runs the dashboard's JavaScript tests (`node --test`) as part of
//! `cargo test`, when Node is installed (they are skipped, loudly, when it
//! is not). The replay test runs against a real training run, so it proves
//! the browser-side network evaluation agrees with the Rust one.

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use neat::NeatConfig;
use sim::training::{DeckChoice, DuplicateChoice, Opponent, TrainConfig, TrainObserver, Trainer};
use sim::{LowestLegal, RandomLegal};

struct Silent;
impl TrainObserver for Silent {}

fn node_available() -> bool {
    Command::new("node")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn real_run() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("arschloch-web-js-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let config = TrainConfig {
        seed: 21,
        player_count: 4,
        deck: DeckChoice::Single,
        duplicate_rule: DuplicateChoice::FirstDealtWins,
        rounds_per_match: 3,
        matches_per_genome: 6,
        reeval_matches: 8,
        generations: 4,
        neat: NeatConfig {
            population_size: 16,
            add_node_rate: 0.4,
            add_connection_rate: 0.4,
            ..NeatConfig::default()
        },
        opponent_specs: vec!["lowest-legal".into(), "random-legal".into()],
    };
    let opponents = vec![
        Opponent {
            name: "LowestLegal".into(),
            strategy: Arc::new(LowestLegal),
        },
        Opponent {
            name: "RandomLegal".into(),
            strategy: Arc::new(RandomLegal),
        },
    ];
    Trainer::new(config, opponents, &dir)
        .unwrap()
        .run(&mut Silent)
        .unwrap();
    dir
}

#[test]
fn the_javascript_tests_pass() {
    if !node_available() {
        eprintln!(
            "SKIPPED: node is not installed, so the dashboard's JavaScript tests did not run"
        );
        return;
    }
    let run = real_run();
    let tests = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let output = Command::new("node")
        .arg("--test")
        .arg(tests.join("lib.test.mjs"))
        .arg(tests.join("replay.test.mjs"))
        .env("RUN_DIR", &run)
        .output()
        .expect("node runs");
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "node --test failed:\n{text}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(text.contains("# fail 0"), "{text}");
    assert!(
        !text.contains("# skipped 1"),
        "the replay test must run against the real run:\n{text}"
    );
    std::fs::remove_dir_all(run).ok();
}
```

- [ ] **Step 4: Run (expect failure)**

Run: `cargo test -p web --test js`

Expected: FAIL (Node reports that `fmt`, `forwardPass` ... are not exported by `lib.js`: it is still the placeholder).

- [ ] **Step 5: Create file**

Create `web/assets/lib.js` (Implement):

```
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
```

- [ ] **Step 6: Run (expect success)**

Run: `node --test web/tests/lib.test.mjs`

Expected: PASS (14 tests: formatting, scales and paths, species stacking, network layout, `forwardPass` against a hand-computed network that ignores disabled edges and refuses a cyclic genome).

- [ ] **Step 7: Run (expect success)**

Run: `cargo fmt -p web && cargo test -p web`

Expected: PASS (the Rust tests plus the Node tests, including the replay of a real run's recorded decisions with a tolerance of 1e-9).

- [ ] **Step 8: Commit**

```bash
git add web
git commit -F - <<'EOF'
web: dashboard JavaScript helpers, tested in Node and against Rust (Phase 10d)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F3sC3JdcgTFAox87bBuikA
EOF
```


### Task 4: The dashboard page

**Files:**
- Modify: `web/assets/index.html`, `web/assets/style.css`, `web/assets/app.js`

**Why:** This part is verified by looking at it: Task 6 runs a real training with `--serve` and inspects the page in a real browser. Narrow screens scroll the drawings sideways inside their card instead of shrinking them to illegibility.

**Interfaces:**
- Consumes: The API of Task 2 and the helpers of Task 3.
- Produces: One page, no build step, no CDN (works offline), light and dark themes: progress and headline numbers; fitness chart (selection best, population mean, median, the champion on fixed matches with a standard-error band, held-out dots); the champion against each opponent alone; species as stacked areas; population fitness spread as a heat strip; complexity; the champion's network with a generation slider, best/latest buttons and a disabled-connections toggle; and a decision inspector that replays a recorded decision's candidates through the network in the browser, colours every node by its activation and checks the replay against the recorded Rust scores. It polls once a second, shows `disconnected, retrying…` when the server is gone, and exposes its state as `window.__dashboard`.

- [ ] **Step 1: Create file**

Create `web/assets/index.html`:

```
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NEAT training</title>
  <link rel="stylesheet" href="/style.css">
</head>
<body>
  <header>
    <h1>NEAT training</h1>
    <span id="status" class="pill">connecting…</span>
  </header>
  <main>
    <section class="card" id="summary">
      <div id="kpis" class="kpis"></div>
      <div class="progress" aria-label="progress"><div id="progress-bar"></div></div>
      <div id="run-info" class="muted"></div>
    </section>

    <section class="card">
      <h2>Fitness</h2>
      <p class="muted">Selection fitness is each generation's best genome on that generation's training matches (inflated by selection). The champion line replays the champion on one fixed set of matches, so champions are compared like for like; dots are new bests confirmed on held-out matches.</p>
      <div id="fitness-chart" class="chart"></div>
      <div id="fitness-readout" class="readout"></div>
    </section>

    <section class="card">
      <h2>Champion against each opponent alone</h2>
      <p class="muted">Mean finishing-role score, +1 (always President) to −1 (always last), at a table of the champion and copies of one opponent. 0 is an even match.</p>
      <div id="opponents-chart" class="chart"></div>
      <div id="opponents-readout" class="readout"></div>
    </section>

    <section class="card">
      <h2>Species</h2>
      <p class="muted">Population split by structural similarity; one colour per species, stacked to the population size.</p>
      <div id="species-chart" class="chart"></div>
      <div id="species-readout" class="readout"></div>
    </section>

    <section class="card">
      <h2>Population fitness spread</h2>
      <p class="muted">Each column is a generation: ten buckets from its lowest to its best fitness, darker where more genomes fall.</p>
      <div id="spread-chart" class="chart"></div>
    </section>

    <section class="card">
      <h2>Complexity</h2>
      <div class="two">
        <div><h3>Hidden nodes</h3><div id="nodes-chart" class="chart"></div></div>
        <div><h3>Enabled connections</h3><div id="connections-chart" class="chart"></div></div>
      </div>
      <div id="complexity-readout" class="readout"></div>
    </section>

    <section class="card">
      <h2>Champion network</h2>
      <div class="controls">
        <label>Generation <input id="network-slider" type="range" min="0" max="0" value="0"></label>
        <output id="network-generation">–</output>
        <button id="network-best" type="button">best</button>
        <button id="network-latest" type="button">follow latest</button>
        <label><input id="show-disabled" type="checkbox"> show disabled connections</label>
      </div>
      <div id="network-note" class="muted"></div>
      <div id="network" class="network"></div>
      <p class="muted">Blue edges push the move's score up, orange edges push it down; thicker is stronger. Inputs are the features the network sees for each candidate move.</p>
    </section>

    <section class="card">
      <h2>Decision inspector</h2>
      <p class="muted">Real decisions of a new-best champion. Pick a candidate move to replay the network on its features and see every node's activation.</p>
      <div class="controls">
        <label>Champion <select id="decision-generation"></select></label>
        <label>Decision <select id="decision-pick"></select></label>
      </div>
      <div id="decision-situation" class="muted"></div>
      <div id="decision-candidates"></div>
      <div id="decision-check" class="muted"></div>
      <div id="decision-network" class="network"></div>
    </section>
  </main>
  <script type="module" src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create file**

Create `web/assets/style.css`:

```
:root {
  --bg: #f6f7f9; --card: #ffffff; --text: #1d2330; --muted: #5b6475; --line: #d9dde5;
  --accent: #2d6cdf; --pos: #2d6cdf; --neg: #e07a1f;
  --s1: #2d6cdf; --s2: #d4572a; --s3: #2a9d6f; --s4: #8a5cd0; --s5: #c9a227; --s6: #6b7280;
  --grid: #e7eaf0; --band: rgba(45, 108, 223, 0.18);
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #11151c; --card: #1a202b; --text: #e6e9ef; --muted: #98a2b3; --line: #2c3544;
    --accent: #6ea0ff; --pos: #6ea0ff; --neg: #f0963f;
    --s1: #6ea0ff; --s2: #f08a5d; --s3: #52c79b; --s4: #b895f0; --s5: #e3c14e; --s6: #9aa3b2;
    --grid: #262e3b; --band: rgba(110, 160, 255, 0.2);
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 15px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
header { display: flex; align-items: center; gap: 16px; padding: 14px 16px; border-bottom: 1px solid var(--line); background: var(--card); position: sticky; top: 0; z-index: 2; }
h1 { font-size: 18px; margin: 0; }
h2 { font-size: 16px; margin: 0 0 6px; }
h3 { font-size: 13px; margin: 0 0 4px; color: var(--muted); font-weight: 600; }
main { max-width: 1000px; margin: 0 auto; padding: 16px; display: grid; grid-template-columns: minmax(0, 1fr); gap: 16px; }
.card { min-width: 0; background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
.muted { color: var(--muted); font-size: 13px; }
.pill { font-size: 12px; padding: 3px 10px; border-radius: 999px; border: 1px solid var(--line); color: var(--muted); }
.pill.live { color: #fff; background: var(--s3); border-color: var(--s3); }
.pill.done { color: #fff; background: var(--accent); border-color: var(--accent); }
.pill.warn { color: #fff; background: var(--neg); border-color: var(--neg); }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 12px; }
.kpi .label { font-size: 12px; color: var(--muted); }
.kpi .value { font-size: 22px; font-weight: 650; font-variant-numeric: tabular-nums; }
.kpi .sub { font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
.progress { height: 8px; background: var(--grid); border-radius: 999px; overflow: hidden; margin-bottom: 8px; }
#progress-bar { height: 100%; width: 0; background: var(--accent); transition: width .4s; }
.chart, .network { overflow-x: auto; }
/* On a narrow screen keep the drawing legible and scroll sideways instead of shrinking it. */
.chart svg { width: 100%; min-width: 600px; height: auto; display: block; }
.chart text { fill: var(--muted); font-size: 11px; }
.chart .grid { stroke: var(--grid); stroke-width: 1; }
.chart .axis { stroke: var(--line); }
.chart .cursor { stroke: var(--muted); stroke-dasharray: 3 3; }
.chart .ref { stroke: var(--muted); stroke-dasharray: 4 4; opacity: .7; }
.readout { min-height: 20px; font-size: 13px; font-variant-numeric: tabular-nums; color: var(--text); }
.readout .item { display: inline-block; margin-right: 14px; }
.readout .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px 0 0; vertical-align: -1px; }
.two { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; }
.two > div { min-width: 0; }
@media (max-width: 700px) { .two { grid-template-columns: minmax(0, 1fr); } main { padding: 12px 16px; } }
.controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 8px 0; }
.controls input[type=range] { width: 260px; max-width: 60vw; }
button, select { font: inherit; color: var(--text); background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 3px 10px; }
button:hover { border-color: var(--accent); cursor: pointer; }
.network svg { width: 100%; min-width: 760px; height: auto; display: block; }
.network text { fill: var(--text); font-size: 11px; }
.network .dim { fill: var(--muted); }
table.candidates { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; margin: 8px 0; }
table.candidates th, table.candidates td { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--line); font-size: 13px; }
table.candidates tr.pick { cursor: pointer; }
table.candidates tr.pick:hover { background: var(--grid); }
table.candidates tr.selected { background: var(--band); }
.ok { color: var(--s3); } .bad { color: var(--neg); }
```

- [ ] **Step 3: Create file**

Create `web/assets/app.js`:

```
// The training dashboard. Polls the server once a second, keeps every
// generation event in memory and redraws. Text from the server is always
// escaped; charts are inline SVG built from the pure helpers in lib.js.
import {
  bandPath, divergingColor, edgeWidth, fmt, formatDuration, forwardPass, histogramOpacities,
  layoutNetwork, linePath, linearScale, nearestIndex, niceScale, speciesColor, stackSpecies,
} from "/lib.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const state = {
  events: [], runStart: null, epoch: 0, finished: false, connected: false, lastChange: Date.now(),
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
  try {
    const last = state.events.length ? state.events[state.events.length - 1].generation : null;
    const [info, data] = await Promise.all([
      fetchJson("/api/state"),
      fetchJson("/api/events" + (last === null ? "" : `?since=${last}`)),
    ]);
    let incoming = data.events;
    if (data.epoch !== state.epoch) {
      // The run's log was rewritten (a resume): everything cached is suspect.
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
    state.runStart = info.run_start;
    state.finished = info.finished;
    state.connected = true;
    render();
  } catch (error) {
    state.connected = false;
    renderStatus();
    console.warn("dashboard poll failed:", error);
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
  const y = niceScale(Math.min(...values), Math.max(...values));
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
  const top = Math.max(1, ...events.map((e) => e.species.reduce((s, x) => s + x.size, 0)));
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
```

- [ ] **Step 4: Run (expect success)**

Run: `node --check web/assets/app.js`

Expected: No output (valid syntax).

- [ ] **Step 5: Run (expect success)**

Run: `cargo fmt -p web && cargo test -p web`

Expected: PASS (the route tests now serve the real page: `<title>`, `fetch`, `export` and CSS variables are present).

- [ ] **Step 6: Commit**

```bash
git add web
git commit -F - <<'EOF'
web: the training dashboard page (Phase 10d)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F3sC3JdcgTFAox87bBuikA
EOF
```


### Task 5: CLI: `cli train --serve` and `cli watch`

**Files:**
- Modify: `cli/Cargo.toml`, `cli/src/main.rs`, `cli/src/train_args.rs`, `cli/src/train.rs`
- Create: `cli/src/watch.rs`, `cli/tests/dashboard_smoke.rs`

**Why:** A real run showed that if the dashboard died with the training process, the browser lost the final generations and the "finished" state, so `--serve` keeps serving afterwards.

**Interfaces:**
- Consumes: `web::Dashboard` (Task 2), `TrainArgs` and `train::run` (10c).
- Produces: `cli train --out DIR --serve [PORT]` (default 8080; starts the dashboard *before* any run state is created, so a busy port fails fast; keeps serving after the run ends until Ctrl-C because the server lives in the training process); `cli watch RUN_DIR [--port N]` (a dashboard for a run in another process, or a finished one; tolerates a directory with no run yet).

- [ ] **Step 1: Edit**

In `cli/src/train_args.rs` (Test first), replace:

```rust
    #[test]
    fn out_is_required() {
```

with:

```rust
    #[test]
    fn serve_takes_an_optional_port() {
        assert_eq!(parse(&["--out", "d"]).unwrap().serve, None);
        assert_eq!(parse(&["--out", "d", "--serve"]).unwrap().serve, Some(8080));
        assert_eq!(parse(&["--out", "d", "--serve", "9100"]).unwrap().serve, Some(9100));
        assert_eq!(parse(&["--out", "d", "--resume", "--serve"]).unwrap().serve, Some(8080));
        assert!(parse(&["--out", "d", "--serve", "notaport"]).is_err());
    }

    #[test]
    fn out_is_required() {
```

- [ ] **Step 2: Create file**

Create `cli/src/watch.rs` (Tests first):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_directory_is_required_and_the_port_defaults_to_8080() {
        let args = WatchArgs::try_parse_from(["cli watch", "runs/a"]).unwrap();
        assert_eq!((args.dir, args.port), (PathBuf::from("runs/a"), 8080));
        assert!(WatchArgs::try_parse_from(["cli watch"]).is_err());
        assert_eq!(
            WatchArgs::try_parse_from(["cli watch", "d", "--port", "0"])
                .unwrap()
                .port,
            0
        );
    }
}
```

- [ ] **Step 3: Edit**

In `cli/src/main.rs`, replace:

```rust
mod train_output;
```

with:

```rust
mod train_output;
mod watch;
```

- [ ] **Step 4: Create file**

Create `cli/tests/dashboard_smoke.rs` (Tests first: the real binary serving a real run over a real socket):

```rust
//! End-to-end tests of the dashboard commands: the real binary serving a
//! real run over a real socket.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

fn run_dir(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("arschloch-cli-dash-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

const SMALL: [&str; 12] = [
    "--population",
    "12",
    "--matches-per-genome",
    "4",
    "--reeval-matches",
    "6",
    "--rounds",
    "3",
    "--threads",
    "2",
    "--generations",
    "2",
];

/// A child process whose stdout lines arrive on a channel; killed on drop.
struct Server {
    child: Child,
    lines: mpsc::Receiver<String>,
}

impl Server {
    fn spawn(args: &[&str]) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_cli"))
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn cli");
        let stdout = child.stdout.take().unwrap();
        let (sender, lines) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if sender.send(line).is_err() {
                    break;
                }
            }
        });
        Self { child, lines }
    }

    fn next_line_containing(&self, needle: &str) -> String {
        let deadline = Instant::now() + Duration::from_secs(60);
        while Instant::now() < deadline {
            if let Ok(line) = self.lines.recv_timeout(Duration::from_millis(200)) {
                if line.contains(needle) {
                    return line;
                }
            }
        }
        panic!("no stdout line containing {needle:?} within 60 s");
    }

    fn address(&self) -> String {
        let line = self.next_line_containing("dashboard: http://");
        let url = line.split("http://").nth(1).unwrap();
        url.split(['/', ' ']).next().unwrap().to_owned()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn get(address: &str, target: &str) -> String {
    let mut stream = TcpStream::connect(address).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    write!(stream, "GET {target} HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
    let mut reply = String::new();
    stream.read_to_string(&mut reply).unwrap();
    reply
}

fn json(reply: &str) -> serde_json::Value {
    serde_json::from_str(reply.split_once("\r\n\r\n").unwrap().1).unwrap()
}

fn wait_until_finished(address: &str) -> serde_json::Value {
    let deadline = Instant::now() + Duration::from_secs(90);
    while Instant::now() < deadline {
        let state = json(&get(address, "/api/state"));
        if state["finished"] == true {
            return state;
        }
        thread::sleep(Duration::from_millis(200));
    }
    panic!("the run never reported finished");
}

#[test]
fn train_serve_shows_the_run_and_stays_up_after_it_finishes() {
    let out = run_dir("serve");
    let mut args = vec![
        "train",
        "--out",
        out.to_str().unwrap(),
        "--serve",
        "0",
        "--quiet",
    ];
    args.extend(SMALL);
    let server = Server::spawn(&args);
    let address = server.address();

    let state = wait_until_finished(&address);
    assert_eq!(state["generations_logged"], 2);
    assert_eq!(state["run_start"]["config"]["generations"], 2);
    let events = json(&get(&address, "/api/events"));
    assert_eq!(events["events"].as_array().unwrap().len(), 2);
    assert_eq!(events["finished"], true);

    // The process keeps serving after the run ends, and says so.
    server.next_line_containing("dashboard stays up");
    assert_eq!(json(&get(&address, "/api/state"))["finished"], true);
    let genome = json(&get(&address, "/api/genome/best"));
    assert_eq!(
        genome["feature_names"].as_array().unwrap().len(),
        sim::FEATURE_COUNT
    );
    let page = get(&address, "/");
    assert!(page.contains("<title>NEAT training</title>"));
    drop(server);
    std::fs::remove_dir_all(&out).unwrap();
}

#[test]
fn watch_serves_a_finished_run_directory() {
    let out = run_dir("watch");
    let mut args = vec!["train", "--out", out.to_str().unwrap(), "--quiet"];
    args.extend(SMALL);
    assert!(Command::new(env!("CARGO_BIN_EXE_cli"))
        .args(&args)
        .status()
        .unwrap()
        .success());

    let server = Server::spawn(&["watch", out.to_str().unwrap(), "--port", "0"]);
    let address = server.address();
    let state = json(&get(&address, "/api/state"));
    assert_eq!(
        (
            state["generations_logged"].clone(),
            state["finished"].clone()
        ),
        (2.into(), true.into())
    );
    let decisions = json(&get(&address, "/api/decisions/0"));
    assert!(!decisions["decisions"].as_array().unwrap().is_empty());
    assert!(get(&address, "/api/genome/999").starts_with("HTTP/1.1 404"));
    drop(server);
    std::fs::remove_dir_all(&out).unwrap();
}

#[test]
fn watch_follows_a_run_that_starts_later() {
    // The dashboard tolerates a directory that has no run yet.
    let out = run_dir("later");
    std::fs::create_dir_all(&out).unwrap();
    let server = Server::spawn(&["watch", out.to_str().unwrap(), "--port", "0"]);
    let address = server.address();
    assert_eq!(json(&get(&address, "/api/state"))["generations_logged"], 0);
    let mut args = vec!["train", "--out", out.to_str().unwrap(), "--quiet"];
    args.extend(SMALL);
    assert!(Command::new(env!("CARGO_BIN_EXE_cli"))
        .args(&args)
        .status()
        .unwrap()
        .success());
    assert_eq!(wait_until_finished(&address)["generations_logged"], 2);
    drop(server);
    std::fs::remove_dir_all(&out).unwrap();
}

#[test]
fn a_busy_port_fails_before_any_run_state_is_created() {
    let busy = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = busy.local_addr().unwrap().port().to_string();
    let out = run_dir("busy");
    let output = Command::new(env!("CARGO_BIN_EXE_cli"))
        .args([
            "train",
            "--out",
            out.to_str().unwrap(),
            "--serve",
            &port,
            "--generations",
            "1",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("cannot start the dashboard") && stderr.contains(&port),
        "{stderr}"
    );
    assert!(!stderr.contains("panicked"));
    assert!(!out.join("checkpoint.json").exists(), "nothing was created");

    let watch = Command::new(env!("CARGO_BIN_EXE_cli"))
        .args(["watch", out.to_str().unwrap(), "--port", &port])
        .output()
        .unwrap();
    assert!(!watch.status.success());
    let nothing = Command::new(env!("CARGO_BIN_EXE_cli"))
        .args(["watch", "/nonexistent/run"])
        .output()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&nothing.stderr).contains("is not a directory"),
        "{}",
        String::from_utf8_lossy(&nothing.stderr)
    );
}
```

- [ ] **Step 5: Run (expect failure)**

Run: `cargo test -p cli`

Expected: FAIL (compile errors: `TrainArgs::serve` and `watch::WatchArgs` do not exist).

- [ ] **Step 6: Edit**

In `cli/Cargo.toml`, replace:

```toml
sim = { version = "0.1.0", path = "../sim" }
```

with:

```toml
sim = { version = "0.1.0", path = "../sim" }
web = { version = "0.1.0", path = "../web" }
```

- [ ] **Step 7: Edit**

In `cli/src/train_args.rs`, replace:

```rust
    /// Print only the final summary
```

with:

```rust
    /// Serve the live dashboard at http://127.0.0.1:PORT/ (default port
    /// 8080) while training, and keep serving after the run ends until
    /// Ctrl-C (the dashboard lives in this process). It only reads the run
    /// directory, so a slow or closed browser cannot affect the run.
    #[arg(long, num_args = 0..=1, default_missing_value = "8080", value_name = "PORT")]
    pub serve: Option<u16>,

    /// Print only the final summary
```

- [ ] **Step 8: Edit**

In `cli/src/train.rs`, replace:

```rust
    let mut trainer = if args.resume {
```

with:

```rust
    // Start the dashboard first: a busy port fails before any run state
    // is created, and the dashboard tolerates a directory that is still
    // empty.
    let dashboard = match args.serve {
        Some(port) => {
            let dashboard = web::Dashboard::start(args.out.clone(), port).with_context(|| {
                format!("cannot start the dashboard on port {port} (is it in use? pick another with --serve PORT)")
            })?;
            println!("dashboard: {}", dashboard.url());
            Some(dashboard)
        }
        None => None,
    };

    let mut trainer = if args.resume {
```

- [ ] **Step 9: Edit**

In `cli/src/train.rs`, replace:

```rust
    trainer.run(&mut observer)?;
    Ok(())
```

with:

```rust
    trainer.run(&mut observer)?;
    if let Some(dashboard) = dashboard {
        // The dashboard lives in this process: exiting now would leave the
        // browser without the last generations and the "finished" state.
        println!(
            "run finished; the dashboard stays up at {} (Ctrl-C to exit)",
            dashboard.url()
        );
        dashboard.wait();
    }
    Ok(())
```

- [ ] **Step 10: Implement**

Insert at the very top of `cli/src/watch.rs`, above the `#[cfg(test)]` line:

```rust
//! `cli watch RUN_DIR`: the dashboard for a run directory, on its own.
//! It follows a run that is training in another process and shows a
//! finished one.

use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;

/// Show a training run in the browser.
#[derive(Parser, Debug)]
#[command(
    name = "cli watch",
    about = "Serve the training dashboard for a run directory (live while it trains, or afterwards)"
)]
pub struct WatchArgs {
    /// A run directory written by `cli train`.
    pub dir: PathBuf,

    /// Port to serve on, on 127.0.0.1 only (use an SSH tunnel to view a
    /// remote run).
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
}

pub fn run(raw_args: impl Iterator<Item = String>) -> anyhow::Result<()> {
    let args = WatchArgs::parse_from(std::iter::once("cli watch".to_owned()).chain(raw_args));
    anyhow::ensure!(
        args.dir.is_dir(),
        "{} is not a directory (expected a run directory written by `cli train`)",
        args.dir.display()
    );
    let dashboard = web::Dashboard::start(args.dir.clone(), args.port).with_context(|| {
        format!(
            "cannot start the dashboard on port {} (is it in use? pick another with --port)",
            args.port
        )
    })?;
    println!("dashboard: {}  (Ctrl-C to stop)", dashboard.url());
    dashboard.wait();
    Ok(())
}
```

- [ ] **Step 11: Edit**

In `cli/src/main.rs`, replace:

```rust
    if std::env::args().nth(1).as_deref() == Some("train") {
        return train::run(std::env::args().skip(2));
    }
```

with:

```rust
    match std::env::args().nth(1).as_deref() {
        Some("train") => return train::run(std::env::args().skip(2)),
        Some("watch") => return watch::run(std::env::args().skip(2)),
        _ => {}
    }
```

- [ ] **Step 12: Edit**

In `cli/src/main.rs`, replace:

```rust
    // `cli train ...` evolves a player; everything else is a simulation run.
```

with:

```rust
    // `cli train ...` evolves a player and `cli watch ...` shows a run in
    // the browser; everything else is a simulation run.
```

- [ ] **Step 13: Run (expect success)**

Run: `cargo fmt -p cli && cargo test -p cli`

Expected: PASS (2 new unit tests plus 4 smoke tests: `train --serve` shows the run and keeps serving after it finishes; `watch` serves a finished run (state, decisions, a 404 for a missing genome); `watch` follows a run that starts later; a busy port fails before anything is created, with a clear message and no panic).

- [ ] **Step 14: Commit**

```bash
git add cli Cargo.lock
git commit -F - <<'EOF'
cli: add `cli train --serve` and `cli watch` (Phase 10d)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F3sC3JdcgTFAox87bBuikA
EOF
```


### Task 6: Real-browser verification

**Files:**
- No files change unless a defect is found (then: fix with a test first, and record it in the ledger).

**Why:** Rust and Node tests cannot show that charts are legible, that controls work, or that the page recovers when the training process is killed. This task looks at the real page.

**Interfaces:**
- Consumes: The built `cli` binary and a real browser (the Chrome automation tools).
- Produces: Evidence that the dashboard works on real runs, live.

- [ ] **Step 1: Run (expect success)**

Run: `cargo build --release -p cli`

Expected: PASS.

- [ ] **Step 2: Look at it (real browser)**

**Live run.** Start a training with the dashboard (a script file that writes the child's pid to a file; do not use `pkill -f`, which can match its own shell):

```bash
target/release/cli train --out /tmp/dash-live --serve 8770 --population 100 --generations 40 --matches-per-genome 40 --reeval-matches 100 --rounds 6 --seed 7 --quiet
```

Open `http://127.0.0.1:8770/` in the browser while it trains.

- [ ] **Step 3: Look at it (real browser)**

**Check the page while it trains.** In the page's JavaScript console (`window.__dashboard` is the dashboard's state): `events` grows once a generation; the status pill reads `training`; every section renders (headline numbers, fitness, opponents, species, spread, complexity, network, decision inspector); `document.documentElement.scrollWidth === document.documentElement.clientWidth` (no page-level sideways scroll even in a narrow window); `__dashboard.checks.replays > 0` and `__dashboard.checks.maxError < 1e-9` (the browser's replay of recorded decisions reproduces the Rust scores).

- [ ] **Step 4: Look at it (real browser)**

**Exercise the controls.** Press `best` (the slider jumps to the latest new-best generation and `follow` turns off); move the slider; tick `show disabled connections` on a champion with hidden nodes (dashed lines appear); in the decision inspector pick another candidate row (its input values and node colours change) and another champion generation. Expected: no console errors from the page, and the replay check line stays green.

- [ ] **Step 5: Look at it (real browser)**

**When the run ends.** The pill reads `finished`, progress is 100%, all generations are listed, and the page stays connected (the server keeps running until Ctrl-C).

- [ ] **Step 6: Look at it (real browser)**

**Kill and resume while watching.** With the page open on a *running* training: `kill -9` the training process (the page keeps its data and shows `disconnected, retrying…`), then start `cli train --out /tmp/dash-live --resume --serve 8770 --generations 60` again. Expected: the page reconnects by itself, shows a contiguous list of generations (no gap, no duplicate) and `__dashboard.runStart.resumed_from_generation` is the checkpointed generation.

- [ ] **Step 7: Look at it (real browser)**

**`cli watch` on a finished run.** `target/release/cli watch /tmp/dash-live --port 8771` and open it: the pill reads `finished` and the same views render.


### Task 7: Docs and the full gate

**Files:**
- Modify: `docs/ROADMAP.md`, `docs/ARCHITECTURE.md`, `docs/BUILDING.md`, `web/src/lib.rs` (no), `docs/superpowers/specs/2026-10-08-neat-engine-design.md`

**Interfaces:**
- Consumes: Everything above.
- Produces: Docs that match the code; a clean phase-done gate. (The how-to-train guide is still deferred, as requested.)

- [ ] **Step 1: Edit**

In `docs/ROADMAP.md` (Roadmap), replace:

```markdown
10d: live browser
```

with:

```markdown
10d (done): live browser
```

- [ ] **Step 2: Edit**

In `docs/superpowers/specs/2026-10-08-neat-engine-design.md` (Spec: what was built), replace:

```markdown
## 8. Determinism and performance
```

with:

```markdown
## 7c. As built in Phase 10d

- **Where:** the server and page live in the `web` crate (std-only HTTP,
  no async runtime, no new dependencies); the page is embedded in the
  binary (`web/assets/`, vanilla JS and inline SVG, no CDN, light and
  dark themes). `cli train --serve [PORT]` runs it inside the training
  process (and keeps serving after the run ends, because the server lives
  there); `cli watch RUN_DIR [--port N]` is a standalone viewer for a run
  in another process or a finished one. It binds to `127.0.0.1` only
  (use an SSH tunnel to view a remote run) and only reads the run
  directory.
- **API:** `GET /api/state`, `/api/events?since=N` (with an `epoch` that
  changes when the log is rewritten by a resume), `/api/genome/N|best`,
  `/api/decisions/N`. The event reader consumes only appended bytes,
  waits for a partial last line, and skips lines that do not parse.
- **Views:** progress and headline numbers; fitness (selection best,
  population mean, median, the champion on fixed matches with its
  standard-error band, held-out dots); the champion against each opponent
  alone; species as stacked areas; the population fitness spread as a
  heat strip; complexity; the champion's network with a generation
  slider, best/latest buttons and a disabled-connections toggle; and the
  decision inspector.
- **Decision inspector:** for every new best champion the trainer records
  12 real decisions from one held-out match (`decisions/gen-NNNN.json`:
  hand, table, every legal move with its feature vector and score). The
  browser replays the network on a chosen candidate, colours every node
  by its activation and checks its own replay against the recorded Rust
  scores (a real run replayed 700+ candidates with a maximum difference
  of 0).
- **Tests:** Rust tests for the event reader, routes and a real socket;
  Node tests for the helpers; a `cargo test` harness that runs the Node
  tests (skipped loudly without Node) including a replay of a real run's
  decisions; and Task 6's real-browser checks, including killing and
  resuming a run while the page is open.
- **Not built:** the Phase 4 statistics per champion (voluntary-pass and
  retention rates), mutation counts by kind, and a WebSocket/SSE channel
  (polling once a second is enough and trivially robust).

## 8. Determinism and performance
```

- [ ] **Step 3: Edit**

In `docs/ARCHITECTURE.md`, replace:

```markdown
### `web` (starts in Phase 10d, extended in Phase 11)
```

with:

```markdown
### `web` (Phase 10d; extended in Phase 11)
```

- [ ] **Step 4: Edit**

In `docs/ARCHITECTURE.md`, replace:

```markdown
Not built yet. Phase 10d starts it with the NEAT training dashboard
(server, embedded page, JSON API over a training run directory); Phase 11
extends
```

with:

```markdown
The training dashboard (Phase 10d): a std-only HTTP server on
`127.0.0.1`, an embedded single-page app (`web/assets/`) and a JSON API
over a run directory (`EventIndex` reads `events.jsonl` incrementally;
routes serve state, events, genomes and recorded decisions). `cli train
--serve` and `cli watch` start it. Phase 11 extends
```

- [ ] **Step 5: Edit**

In `docs/BUILDING.md`, replace:

```markdown
`cli train --help` lists every option.
```

with:

```markdown
Watch a run in the browser: `cli train --out runs/first --serve` prints
`dashboard: http://127.0.0.1:8080/` (and keeps serving after the run
ends, until Ctrl-C), or, from another terminal or afterwards,
`cli watch runs/first`. It listens on `127.0.0.1` only; to view a remote
run use an SSH tunnel (`ssh -L 8080:127.0.0.1:8080 host`). The page
needs no internet access.

`cli train --help` lists every option.
```

- [ ] **Step 6: Run (expect success)**

Run: `cargo fmt --check`

Expected: PASS.

- [ ] **Step 7: Run (expect success)**

Run: `cargo clippy --workspace --all-targets -- -D warnings`

Expected: Clean.

- [ ] **Step 8: Run (expect success)**

Run: `cargo test --workspace`

Expected: PASS: every suite green (see the ledger for the count).

- [ ] **Step 9: Commit**

```bash
git add docs Cargo.lock
git commit -F - <<'EOF'
docs: Phase 10d done; dashboard as-built notes, usage

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F3sC3JdcgTFAox87bBuikA
EOF
```


---

## Self-Review (done while writing)

- **Spec coverage (7a, browser part):** progress and ETA, fitness curves with the fixed-match champion band and held-out dots, per-opponent scores with the 0 reference, species stacked areas, population spread, complexity, the champion's layered network with a generation slider (the "replay of how the topology grew"), the decision inspector with per-node activations, `--serve` and `cli watch`, an API over the run directory, polling instead of WebSockets (as the spec chose). Left out deliberately and written into the spec by Task 7 (section 7c): per-champion Phase 4 statistics and mutation counts by kind.
- **Placeholder scan:** none; every step carries its code (Task 6's steps are explicit manual checks with expected values).
- **Type consistency:** `Dashboard`, `App`, `EventIndex`, `DecisionFile`/`DecisionRecord`/`CandidateRecord`, `ScoredCandidate`, `WatchArgs` and the API paths are used with the same names in every task that mentions them.
- **Review Focus:** all five lines map to named tests or to Task 6.
