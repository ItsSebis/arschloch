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
`--weight-power` (default 0.2). `--fitness-skill-weight W` (0 to 1, default 0 = off, fixed for
the run) is an experimental luck-adjusted fitness term, see `docs/TRAINING.md`; the evidence for the defaults is in
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

### Luck, skill and the extra numbers (Phase 12)

Every run now also prints, after the original summary, a table
`Strategy | avg rank ±SE | mean score ±SE | strength rating ±SE | President %
[95% Wilson]` and the average rank and mean score per seat (position bias).
Place 1 is best; the mean score is +1 for the best role down to -1 for the
last; the strength rating is a Bradley-Terry fit on "finished above"
outcomes in Elo-like points with the field average at 0, so it measures
strength against this field (beating strong opponents counts for more);
`±` is a standard error over matches. `docs/STATISTICS.md` explains every
number; `cli stats-doc` regenerates it.

- `--explain` adds, after each new section, one line with the catalogue's
  meaning of its numbers, and at the end the lines for the original
  sections.
- `--bootstrap-resamples N` (default 200) is the number of resamples behind
  the rating's standard error; `0` skips it (the rating is then printed
  without `±` and its JSON `std_error` is `null`: no resampling, no error
  estimate, which is not the same as an error of 0).
- `--skill-score off|estimate|duplicate|both` (default `off`: no extra
  simulation, no extra table). It adds the table `Strategy | plain score
  ±SE | skill (duplicate) ±SE | skill (estimate) ±SE | variance reduction M
  | luck share`:
  - `estimate`: ordinary matches that also record the round-1 hand
    features; a regression removes the part of the first-round result that
    the dealt hand explains (cards of Queen or higher, pairs, triples,
    quads, lowest and mean card strength, and the hand size, since the
    uneven deal at 3, 5 and 6 players gives the first seats an extra card).
    Nearly free, round 1 only. With a single strategy name at the table
    there is nothing to adjust against and the estimate prints `n/a: a
    single strategy name`.
  - `duplicate`: groups of `--player-count` matches play identical deals
    with the strategies rotated through every seat, so every strategy plays
    every hand once. **`--matches` must be a multiple of `--player-count`**
    (otherwise the run stops with an error naming the number). These groups
    are the matches of the run, so all other numbers and the JSON `matches`
    come from them; deals come from a separate random stream, so results
    differ from an ordinary run with the same seed.
  - `both`: the duplicate groups (they are the matches of the run and the
    ground truth) **plus a second, independent ordinary batch** of the same
    number of matches (seeds offset by a constant, so different deals) with
    hand features, on which the estimator runs. The estimator is then judged
    against the duplicate result: difference in standard errors per
    strategy, rank agreement and the share of luck variance it removes. This
    plays about twice the matches, so it takes about twice as long. The
    estimator's standard errors are honest because its matches are
    independent.

  How to read it: *plain score* is the ordinary mean score. *Skill* is the
  same strategy's score after removing the luck of the deal (same value
  with a smaller `±` in a balanced batch, or shifted for the estimator).
  *M* (variance reduction) says how many ordinary games one duplicate game
  is worth; *luck share* = 1 - 1/M, the share of single-game variance the
  deal accounts for. Duplicate deals cancel the luck of the deal exactly in
  round 1; later rounds are luck-reduced, not luck-free, because roles and
  exchanged cards carried over depend partly on earlier play. The duplicate
  columns, M and luck share cover all rounds; the estimate covers round 1
  only (an `estimate`-only run reports its plain score, M and luck share for
  round 1 too). In duplicate mode the *plain score* column equals the
  duplicate skill value by construction (the mean of the group means is the
  mean of the matches); its `±` is the counterfactual standard error of an
  ordinary run of the same size, not an independent estimate. When one
  strategy name occupies several seats of a table, its standard errors use
  one series per match (the name's seats averaged), because those seats are
  negatively correlated.

- `--useful-passes K` (off unless given, `K >= 1`) measures whether voluntary
  passes (a pass although a play was legal) are useful. For a sample of them
  the round is replayed `K` times from that exact state after the pass, after
  the weakest legal play and after the strongest; a pass is useful if the
  average finishing place after it is better than after the better of the two
  plays (by more than `--useful-pass-margin`, default 0). `--useful-pass-sample F`
  (0..=1, default 0.05) is the fraction of the voluntary passes analysed; the
  cost is about `3 * K` partial rounds per sampled pass. The analysis is
  deterministic for the same flags, uses `--seed`, and replays the batch's
  matches as ordinary matches (in the duplicate modes it does not reuse their
  deals, so its pass counts belong to an ordinary batch of the same
  configs). It adds a summary table (`voluntary passes | sampled | useful
  share [95% Wilson] | mean gain ±SE`; strategies that never pass on purpose
  read "no voluntary passes") and, in `extended.by_strategy.<name>`, the keys
  `voluntary_passes_total`, `sampled_voluntary_passes`, `useful_passes`,
  `useful_pass_share` (`value`, `std_error`, `n`, `interval`) and
  `useful_pass_gain` (`value`, `std_error`, `n`; null without samples), only
  when the flag is given.

New keys of `results.json` (additive; `matches` and `statistics` are
unchanged, `matches[].first_hand_features` appears only with `estimate`; in
`both` the run's matches are the duplicate groups, which carry no features,
and the features live only in the separate estimator batch):

```json
{
  "matches": [ ... ],
  "statistics": { ... },
  "extended": {
    "by_strategy": { "LowestLegal": {
      "avg_rank": {"value": 2.86, "std_error": 0.02, "n": 2000},
      "mean_role_score": {"value": -0.243, "std_error": 0.012, "n": 2000},
      "rank_distribution": [14.2, 24.0, 30.1, 31.7],
      "rounds": 12000,
      "strength_rating": {"value": -73.0, "std_error": 3.0, "n": 2000} } },
    "by_seat": [ {"avg_rank": {...}, "mean_role_score": {...}} ],
    "role_retention_intervals": { "LowestLegal": { "President": {"low": 0.2, "high": 0.3} } },
    "voluntary_pass_rate_intervals": { "LowestLegal": {"low": 0.1, "high": 0.2} },
    "skill": {
      "mode": "both",
      "duplicate": { "group_count": 500, "k": 4, "rounds_per_match": 6,
        "strategies": [ {"name": "LowestLegal",
          "skill_score_duplicate": {"value": -0.243, "std_error": 0.008, "n": 500},
          "variance_reduction": 2.03, "luck_share": 0.51, "...": "..."} ] },
      "estimate": { "match_count": 2000, "feature_r2": 0.40,
        "strategies": [ {"name": "LowestLegal",
          "skill_score_estimate": {"value": -0.146, "std_error": 0.012, "n": 2000},
          "...": "..."} ] },
      "comparison": { "rank_agreement": 1.0, "verdict": "...", "...": "..." }
    }
  },
  "statistics_catalog": [ {"id": "avg_rank", "name": "Average finishing place",
    "scope": "per_strategy", "meaning": "...", "json_path": "...", "...": "..."} ]
}
```

`extended.skill` and its `duplicate`, `estimate`, `comparison` parts exist
only for the modes that compute them. Every catalogue entry's `json_path`
says where its value lives (`docs/STATISTICS.md` lists them), and a test
checks that against a real run.

`cli evaluate` additionally prints the average place (with its standard
error) per cell and, in its `--json`, adds `avg_rank` and
`avg_rank_std_error` to each cell and the `statistics_catalog` key; it has
`--explain` too. (The skill score is not offered there.)

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
