# Building & Packaging

## Prerequisites

- Rust via [rustup](https://rustup.rs) (this repo pins no toolchain file;
  any current stable toolchain works).

## Native build & test

From the workspace root:

```bash
cargo build --release
cargo test --workspace
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
```

All four are the phase-done gate from `docs/CODING_GUIDELINES.md` and
should be clean before any phase is considered finished.

The `cli` crate is a binary: `cargo run -p cli -- <args>` runs it, and
`cargo build --release -p cli` produces `target/release/cli` (or
`cli.exe` on Windows, natively).

## Running the simulator

```bash
cargo run -p cli -- \
  --player-count 4 \
  --deck-variant single \
  --duplicate-rule first-dealt-wins \
  --matches 1000 \
  --rounds 10 \
  --strategy lowest-legal \
  --strategy greedy-highest \
  --strategy random-legal \
  --strategy lowest-legal \
  --threads 4 \
  --seed 1 \
  --output results.json
```

- `--strategy` is repeated once per seat, in seat order, and must be
  repeated exactly `--player-count` times. Valid values:
  `lowest-legal`, `greedy-highest`, `random-legal`.
- `--deck-variant` is `single` or `double`; `--duplicate-rule` is
  `first-dealt-wins` or `last-dealt-wins` (only matters for `double`).
- `--threads 0` (the default) lets `rayon` pick its own thread count;
  `--seed` is the base seed each match's own seed derives from, so a
  full run is reproducible byte-for-byte given the same flags.
- Two things happen: `results.json` (the `--output` path) gets every
  match's raw `MatchResult` plus the aggregated `Statistics`, and a
  human-readable summary table (role counts by strategy, and the pooled
  voluntary-pass rate) prints to stdout.

## Cross-compiling a Windows executable (from Linux or macOS)

This workspace has zero C dependencies — `rayon`, `rand`, and `serde` are
all pure Rust — so cross-compiling to Windows only needs Rust's own
toolchain plus a Windows CRT/SDK, which
[`cargo-xwin`](https://github.com/rust-cross/cargo-xwin) provides without
installing MinGW or needing `sudo`/root access:

```bash
rustup target add x86_64-pc-windows-msvc
cargo install cargo-xwin
cargo xwin build --release --target x86_64-pc-windows-msvc -p cli
```

The `.exe` lands at `target/x86_64-pc-windows-msvc/release/cli.exe`.

The first `cargo xwin build` downloads and caches the Windows SDK and CRT
headers (a few hundred MB, under `~/.cache/cargo-xwin`); later builds
reuse the cache and are as fast as a normal `cargo build`.

To cross-compile the whole workspace instead of just `cli`, drop `-p cli`.

### Why MSVC instead of the GNU target

The traditional route (`rustup target add x86_64-pc-windows-gnu` +
`apt install mingw-w64`) needs a system package manager and (on most
setups) `sudo`. `cargo-xwin` against `x86_64-pc-windows-msvc` needs
neither — everything it uses is fetched by `cargo`/`rustup` under your own
user account, in keeping with this project's preference for
self-installable tooling.
