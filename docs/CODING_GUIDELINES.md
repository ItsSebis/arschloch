# Coding Guidelines

Goal: keep the codebase small and easy to hold in your head across
sessions. These aren't arbitrary — they're established Rust community
conventions, chosen so future sessions (human or AI) don't have to
rediscover them.

## Sources

- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/) —
  naming, interoperability, documentation, and predictability conventions
  for public types/functions. Follow these for anything `pub` in `engine`
  and `sim`, since those are the two crates other crates (and future
  sessions) depend on.
- `clippy` — run with `pedantic` enabled at the workspace level (see
  `Cargo.toml` / `clippy.toml` once scaffolded); treat pedantic lints as
  "consider it, then either fix or `#[allow]` with a one-line reason", not
  as noise to blanket-suppress.
- `rustfmt` with default settings — no custom `rustfmt.toml` unless a
  specific rule actively causes churn.

## Project-specific rules

- **One concept per module.** If a file is doing "cards and also the
  scoring math and also serialization," split it. A module that's grown
  past ~300-400 lines is a signal to split, not a target to hit before
  splitting.
- **`engine` stays dependency-free** (aside from maybe `serde` for
  round-tripping game state). No `rayon`, no `clap`, no I/O. This is what
  keeps its unit tests fast and its logic provably pure.
- **No premature abstraction.** Three near-identical match arms are fine.
  Don't introduce a trait or generic parameter until a second concrete use
  case actually needs it — the roadmap phases are sequenced so each one
  adds real requirements, not speculative ones.
- **Data over branching logic.** The role tables and exchange-count tables
  in `RULES.md` are naturally `const` arrays/lookup tables indexed by
  player count, not `match` trees duplicated across functions.
- **Comments explain *why*, not *what*.** A comment on the suit-ranking
  `Ord` impl explaining it's a house rule (with a pointer to `RULES.md`) is
  useful; a comment restating "returns true if legal" on a function named
  `is_legal` is not.
- **Errors:** `engine`'s public API should make illegal states
  unrepresentable via types where practical (e.g. `Combo::new` validates
  and returns `Option`/`Result`, so a `Combo` in hand is always valid).
  Reserve `panic!`/`unwrap()` for genuine invariant violations (a bug, not
  user input) and boundary code (CLI arg parsing) for `anyhow`/`thiserror`
  once that's introduced in Phase 3.
- **Tests live next to the code** (`#[cfg(test)] mod tests` in the same
  file) for unit tests; cross-module behavior (a full scripted round) gets
  a file under `tests/` in the relevant crate.

## Before finishing a phase

- `cargo fmt --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`

All three should be clean before a phase is considered done.
