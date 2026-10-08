# Phase 14: a pass ends your part in the trick — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (native execution chosen by the user). Steps use checkbox syntax.

**Goal:** Make "once you pass in a trick you are out of that trick and cannot rejoin it" the rule of the game (docs/Notes.md, "Rules"), keep the old behaviour available for reproducing every earlier result, and re-baseline and retrain under the new rule.

**Architecture:** `engine::PassRule { Free, Final }` (default `Final`) lives in the trick bookkeeping; `Round::new` uses the default, `Round::with_pass_rule` takes it explicitly. The rule is a field of `sim::MatchConfig`, `sim::session::SessionConfig`, `training::TrainConfig` (old checkpoints and configs read as `Free`) and a `--pass-rule free|final` option of every CLI mode and the play page.

**Spec:** `docs/Notes.md`; `docs/ROADMAP.md` Phase 14; rules in `docs/RULES.md` ("Playing a Round").

## Rulings

- The new rule is the default everywhere: the notes say "make sure that once passing ... you are out", i.e. it is the correct rule, and a silent default of the old rule would train and play the wrong game. `free` stays selectable and is what the committed pre-neat and neat-v1 baselines were measured under, so their scripts now pass `--pass-rule free` explicitly. Cost if wrong: flip one `Default` impl.
- Old run directories (checkpoints, `config.json`) carry no rule; they are read as `free` so `--resume` continues them as they were.
- The neat feature set is unchanged (no "who is still in this trick" input yet): that is a Phase 13 idea. The existing genome files stay loadable; what changes is how well they play, which the new baselines measure.
- `PassCeilings` hand reading stays: a pass still proves the seat held no beater at that moment and refutation by a later play (in a later trick) still catches dishonest passes. The roadmap sentence claiming otherwise is corrected.

## Global Constraints

- Under `free` everything is bit-for-bit what it was (existing tests and the perf checksums in `docs/baselines/perf` with `--pass-rule free` must stay identical).
- Under `final`: a seat that passed is skipped until the trick ends; the trick ends when every active seat other than the last player is passed; the lead goes to the last player to play, or, if that player emptied their hand, to the next active seat; a pass is still always legal when following; passing on a lead is still illegal.
- fmt, clippy `-D warnings` on stable and `+1.99.0`, `cargo test --workspace` by exit status; CI on three systems.

## Review Focus

1. Trick resolution under `final` when seats finish mid-trick (the winner finishes; a passed seat is the only one left; two seats left; 3-6 seats; double deck): no deadlock, no seat asked twice, no seat skipped wrongly, rounds always end.
2. A play that leaves no eligible seat; the first pass after a lead; everyone passing around a lone leader.
3. Backward compatibility: old `config.json`/checkpoint resume as `free`; `--resume` rejects a conflicting `--pass-rule`; default CLI output keys unchanged except an additive rule line/field.
4. Strategies and the session under `final`: no strategy assumes it is asked again after passing; `Session::legal_moves`/`playable` after a pass in the same trick; the human cannot act after passing.
5. Reproducibility of the committed baselines with `--pass-rule free`.

## Task 1: engine — `PassRule` in `Trick` and `Round`

**Files:** `engine/src/{trick,round,lib}.rs`; tests inline and `engine/tests/full_round.rs`.

- `pub enum PassRule { Free, Final }` with `Default = Final`, `Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize` (snake_case names `free`, `final`), `Display`/`FromStr` for the CLI.
- `Trick::new(leader, rule)`; under `Final` keep `passed: Vec<bool>` (reset every trick). `record_play` returns `Option<SeatId>` (a new leader when no other seat can still act, else `None`; always `None` under `Free`); `record_pass` as before, skipping passed seats when advancing `turn` and resolving when every active non-winner seat has passed.
- `Round::with_pass_rule(hands, duplicate_rule, pass_rule, first_leader)`; `Round::new` = default rule; `Round::pass_rule()`.
- [ ] RED tests (trick): the existing free tests construct `Trick::new(l, PassRule::Free)` unchanged; new final tests: a passed seat is skipped after a beating play (4 seats: 0 leads, 1 passes, 2 beats, 3 passes, 0 beats, turn goes 2 not 1); resolution needs only the not-yet-passed seats; the winner finishing hands the lead to the next active seat; two active seats; a play with nobody eligible resolves at once.
- [ ] RED tests (round): same scenarios through `submit_move`: a passed seat is never `seat_to_move` again in the trick and `submit_move` for it is `NotYourTurn`; next trick it acts again; random full rounds for 3-6 seats, both decks, both rules, always complete with a valid finishing order (bounded step count) — 2000 rounds each.
- [ ] implement; fix existing tests that assumed the old rule by making them explicit `Free`; commit `engine: a pass ends your part in the trick (pass rule)`.

## Task 2: sim and CLI plumbing

**Files:** `sim/src/{match_config,match_runner,session}.rs`, `sim/src/training/{config,evaluate,decisions,trainer}.rs`, `cli/src/{args,main,evaluate,train,train_args,train_output}.rs`, tests.

- `MatchConfig.pass_rule`, `TableSpec.pass_rule`, `TrainConfig.pass_rule` (`#[serde(default = "legacy")]` returning `Free` so old files read as before; new configs write it), `SessionConfig.pass_rule`.
- CLI: `--pass-rule free|final` (default `final`) on the simulation mode, `train`, `evaluate`; `train --resume` refuses an explicit conflicting value (clap conflicts like the other settings); the simulation summary and JSON output name the rule (additive key `pass_rule`); training banner and `evaluate --json` too; `--from` takes the rule of the new command line (a warm start may change rule).
- [ ] RED tests: free output equals the old checksums (`docs/baselines/perf/bench.sh` with `--pass-rule free` equals the recorded checksums); final output differs and is deterministic and thread-independent; an old config JSON without `pass_rule` deserialises as `free` and resumes as `free`; `train --resume --pass-rule final` is rejected; the strategies all complete matches under final at 3-6 seats and both decks; training improves play under final (existing `training_improves_play` run with the rule).
- [ ] implement; commit `sim, cli: --pass-rule`.

## Task 3: playing — session, web, page

- `SessionConfig.pass_rule`; `POST /api/games` takes `"pass_rule"` (default `final`); the setup form gets a select; the human cannot move after passing in the same trick (view: `to_move` skips them; state shows who has passed this trick: `View.passed` per seat, and the seat boxes mark "passed").
- [ ] RED tests: session under `final` — after the human passes, AI seats continue without them until the trick ends, `play`/`pass` in that interval is `NotYourTurn`, the passed flag shows in the view and clears at the next trick, a whole match completes at 3-6 seats; API accepts/rejects `pass_rule`; node test for the replay display marking passed seats; browser check.
- [ ] commit `play: the pass rule in sessions and on the page`.

## Task 4: baselines and retraining

- Committed scripts (`docs/baselines/pre-neat/run.sh`, `neat-v1/run.sh`, experiments scripts, `perf/bench.sh`) pass `--pass-rule free` explicitly; verify they reproduce the committed summaries (rerun the 3-player part and compare, plus the perf checksums).
- New `docs/baselines/pass-final/`: the hand-written strategies' baseline under `final` (same seeds and counts as pre-neat), the neat-v1 champion evaluated under `final`, and the same for 3-6 players.
- Retrain under `final` with the neat-v1 recipe (`--seed 1 --population 150 --generations 120 --matches-per-genome 80 --reeval-matches 200 --rounds 8 --champion-candidates 5 --weight-power 0.2`) -> `docs/baselines/neat-v2/` (champion.json, run.sh, README with held-out score and the comparison with neat-v1 under both rules).
- `cli play` built-in champion becomes the v2 champion; v1 stays in the baselines.
- [ ] record every result in the ledger; commit `baselines: re-measured under the pass rule; neat-v2 champion`.

## Task 5: docs, review, PR

- [ ] RULES.md (the rule, with an example trick), README, BUILDING (flag), TRAINING (`--pass-rule`, resume note), PLAYING, ARCHITECTURE, ROADMAP (Phase 14 done; Phase 13 gains "who is still in the trick" input; the hand-reading sentence corrected).
- [ ] Whole checks; fresh opus review with the Review Focus; one fix pass; push, PR, CI.
