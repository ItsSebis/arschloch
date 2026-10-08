# Phase 15: the exchange gives your highest cards — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (native execution chosen by the user).

**Goal:** The lower role of an exchange pair must hand over its N highest cards (no choice), as in the real game. The old behaviour (the lower role, or its strategy, chooses any N cards) stays available as a modifier.

**Architecture:** `engine::ExchangeRule { Free, Forced }` (default `Forced`) with `exchange_with_rule`, which ignores the chooser under `Forced`. It is a field of `MatchConfig`, `TableSpec`, `TrainConfig` (old configs read as `Free`), `SessionConfig` and `Record` (old lines read as `Free`), and a `--exchange-rule free|forced` option on every CLI mode, the web API and the play page. Mirrors the pass-rule plumbing of Phase 14.

## Rulings

- Default is `Forced` (the user: "enforce the highest cards to be given away too ... default is forcing N highest cards"); `Free` is the modifier. The strategies' `choose_exchange_cards` stays (it is what `Free` uses).
- Under `Forced` nothing consumes the match RNG during the exchange; strategies that give their highest cards anyway (LowestLegal, GreedyHighest, EndgameDenial, CardCounter, Adaptive, NEAT) play byte-identical games under both rules. Only `RandomLegal` and `HoldBackPairs` differ.
- Old baselines (pre-neat, neat-v1, perf checksums) are reproduced with `--pass-rule free --exchange-rule free`; the Phase 14 `pass-final` baselines were measured with `--pass-rule final --exchange-rule free`; new `current-rules` baselines use both defaults.

## Review Focus

1. Forced exchange for every table size (3-6), both decks, duplicate rules: the lower role gives exactly its N highest by `Card::compare`, the higher role its N lowest, same as `engine::exchange`.
2. The human under `Forced` is never asked to choose (phase `exchange` never occurs) but still sees what was taken and received; under `Free` unchanged.
3. Backward compatibility: old config/checkpoint resume as `free` exchange (and `free` pass rule); `--resume` cannot change it; old records read as free and never mix with new in the stats.
4. Bit-for-bit: `--pass-rule free --exchange-rule free` reproduces every committed baseline and the perf checksums; forced equals free for the non-deviating strategies.

## Tasks

1. engine: `ExchangeRule`, `exchange_with_rule` (RED tests: forced ignores a chooser that tries to give low cards; free honours it; all sizes).
2. sim/cli plumbing (`--exchange-rule`, configs, banner, summary, evaluate JSON; RED tests: old config reads free, resume conflict, determinism, non-deviating strategies identical, deviating differ).
3. session/web/page (human never chooses under forced; API field; page select; records keep the rule).
4. baselines: scripts take `EXCHANGE_RULE`; reproduce old; `current-rules` baselines; retrain under both defaults (v3) and compare; built-in champion.
5. docs (RULES, TRAINING, BUILDING, PLAYING, ROADMAP, README of baselines), checks, fresh review, fix pass, push to the open PR, CI, merge, release v0.4.0.
