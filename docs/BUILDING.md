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

Trained genomes play like any other strategy via `--strategy
neat:PATH` (the player appears in results as `Neat(<file name>)`); the
file records the feature names and the feature-set version it was
trained on, and a file from a different build is refused with an error
instead of silently misplaying. Two different genome files with the same
file name cannot share a table (their results would merge); rename one.

Quick reference for training (a full guide comes later):

```bash
cargo build --release -p cli
target/release/cli train --out runs/first --generations 50     # new run
target/release/cli train --out runs/first --resume --generations 100
target/release/cli --player-count 4 --matches 1000 \
  --strategy neat:runs/first/best.json --strategy lowest-legal \
  --strategy lowest-legal --strategy lowest-legal
```

Watch a run in the browser: `cli train --out runs/first --serve` prints
`dashboard: http://127.0.0.1:8080/` (and keeps serving after the run
ends, until Ctrl-C), or, from another terminal or afterwards,
`cli watch runs/first`. It listens on `127.0.0.1` only; to view a remote
run use an SSH tunnel (`ssh -L 8080:127.0.0.1:8080 host`). The page
needs no internet access.

Compare genomes (or a genome against hand-written strategies) with
`cli evaluate --genome runs/first/best.json` (a table of mean finishing-role
scores against a default battery of opponents; add `--opponent SPEC` to
choose them, repeat `--genome` to compare several). Learning options:
`--champion-candidates` (default 5), `--hall-of-fame` (default off) and
`--weight-power` (default 0.2); the evidence for the defaults is in
`docs/baselines/neat-v1/experiments.md`, and `docs/baselines/neat-v1`
holds a committed champion with its comparison against the pre-NEAT
strategies.

`cli train --help` lists every option. A run is reproducible from
`--seed`, resumable after any interruption, and independent of
`--threads`.

The `neat` crate has no game dependency: `cargo test -p neat` runs its
unit tests and the XOR end-to-end evolution test in about a second.

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

- `--strategy` is repeated once per seat, in seat order — this sets the
  seating for the batch's first match only; `sim` then rotates which
  seat each strategy occupies for every subsequent match, to cancel out
  a seat-position bias in how cards are dealt (see `docs/RULES.md`,
  "Players & Deck"). Must be repeated exactly `--player-count` times.
  Each value is a spec, `SPEC := FIXED | "adaptive" | "adaptive:"
  OPTIONS`:
  - `FIXED` is one of the six fixed strategy names: `lowest-legal`,
    `greedy-highest`, `random-legal`, `hold-back-pairs`, `card-counter`,
    `endgame-denial`.
  - `adaptive` on its own uses `sim::AdaptiveConfig::default()`
    (card-counting on, hand-reading endgame denial with `close=2`,
    deception off).
  - `adaptive:OPTIONS` configures it explicitly. `OPTIONS` is a
    comma-separated list of:
    - `counting` — enable the card-counting modifier (base play becomes
      `card-counter` instead of `lowest-legal`).
    - `denial` — enable endgame denial, judged purely by hand size
      (reproducing `endgame-denial` exactly).
    - `reading` — enable endgame denial sharpened by pass-based hand
      reading: spends the *lowest* card that provably locks a
      close-to-finishing opponent out, rather than a blanket highest-card
      push. If both `denial` and `reading` are set, `reading` wins.
    - `close=<n>` — the hand-size threshold for whichever of
      `denial`/`reading` is set (default `2`); an error if given without
      one of them.
    - `deception=<rate>` — probability in `[0, 1]` (default `0.0`, i.e.
      off) that this seat bluff-passes on a turn it could legally beat,
      to plant a false pass ceiling in an opponent's hand reading.
    - `tempo` — enable trick-lead tempo: once this seat's own hand is
      down to 2 cards, spend the cheapest play that's provably safe
      against the whole active field instead of the base strategy's
      cheapest-legal instinct, to win the race to lead the next trick.
      Not a configurable threshold (see `sim`'s `adaptive::tempo` module
      doc comment for why `close=2` is a fixed constant here, unlike
      `denial`/`reading`'s `close`).
    - `bully` — enable lead-order bullying: while leading, with this
      hand shaped mostly as same-rank groups (whole-group leads at
      least as numerous as leftover singles), lead the cheapest whole
      same-rank group before ever leading a single, banking singles
      for the end of the hand. Unlike `denial`/`tempo`, has no
      opponent-proximity threshold at all — see `sim`'s
      `adaptive::bully` module doc comment for the empirical finding
      behind that (a sweep showed such a gate only limited the
      benefit, never protected against a downside).
    - `none` — all modifiers off; on its own, byte-identical to plain
      `lowest-legal` for the same seed.
  - Options may be given in any order; each key may appear at most once;
    an empty option list, an unknown key, a flag given a value, or a
    parameter missing its value are all parse errors.

  For example, a three-seat table with one fixed strategy and two
  differently-configured `adaptive` seats:

  ```bash
  --strategy lowest-legal \
  --strategy adaptive:counting \
  --strategy "adaptive:reading,deception=0.2,tempo,bully"
  ```
- `--deck-variant` is `single` or `double`; `--duplicate-rule` is
  `first-dealt-wins` or `last-dealt-wins` (only matters for `double`).
- `--threads 0` (the default) lets `rayon` pick its own thread count;
  `--seed` is the base seed each match's own seed derives from, so a
  full run is reproducible byte-for-byte given the same flags.
- Two things happen: `results.json` (the `--output` path) gets every
  match's raw `MatchResult` plus the aggregated `Statistics`, and a
  human-readable summary table (role counts by strategy, and the pooled
  voluntary-pass rate) prints to stdout.
- The aggregated statistics include a first-round placement variance per
  strategy (`docs/ROADMAP.md`, Phase 4, "Luck-vs-skill signal"), which
  reports `null` until the batch repeats at least one exact seating (the
  same strategies in the same seats) twice. `sim::run_batch` cycles
  seatings every `--strategy`-flag-count matches, so pick `--matches` at
  least `2 * (number of --strategy flags)` if you want this number
  populated.
- `--exchange-rule` is `forced` (default: the lower role of an exchange pair must
  hand over its highest cards, the rules of the game) or `free` (the lower role
  or its strategy chooses which cards to give; the behaviour before Phase 15, for
  reproducing older results and as a variant). It only changes games with
  `random-legal` or `hold-back-pairs`.
- `--pass-rule` is `final` (default: a player who passes is out of the
  trick until it ends, the rules of the game) or `free` (the older behaviour,
  where a passed player may still play later in the same trick; use it to
  reproduce results measured before Phase 14). The summary line and the
  `evaluate --json` output name the rule.

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
