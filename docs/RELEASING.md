# Releasing

How the binaries on a GitHub release are produced.

- **Linux and Windows** are built by hand from the tagged commit and uploaded
  with the release:
  - Linux: `cargo build --release -p cli -p gui`, **inside an older distribution**: the binaries need the glibc of the
    machine that built them (built on a glibc 2.43 desktop, `arschloch-gui` needs glibc 2.43 for `atan2f` and would
    not start on Ubuntu 24.04 or Debian 12). Built in Debian 12 (bookworm) both need glibc 2.35 or newer (`hypot`)
    and produce the same results (checked on 2026-10-17):

    ```
    docker run --rm --user $(id -u):$(id -g) -v $PWD:/src:ro -v $HOME/.cache/docker-cargo-home:/cargo \
      -v $HOME/.cache/target-docker:/target -e CARGO_HOME=/cargo -e CARGO_TARGET_DIR=/target -w /src \
      rust:1.98-bookworm cargo build --release --locked -p cli -p gui
    ```

    Check the floor with `objdump -T BINARY | grep -o 'GLIBC_[0-9.]*' | sort -V | tail -1`. The window needs an X11
    or Wayland session and loads its libraries when it starts, so building needs no extra system packages.
  - Windows (cross-compiled from Linux with `cargo-xwin`, C runtime linked
    statically so no Visual C++ redistributable is needed):
    `RUSTFLAGS="-C target-feature=+crt-static" cargo xwin build --release
    --target x86_64-pc-windows-msvc -p cli -p gui`. The window is a Windows GUI program (no console behind it).
  - Each archive holds the `cli` binary, `arschloch-gui` (`arschloch-gui.exe`; it must stay next to `cli`, which it
    runs), `README.md`, `docs/TRAINING.md`, `docs/RULES.md`, `docs/PLAYING.md`, `docs/GUI.md` and the baseline
    champion (`champion.json`, from `docs/baselines/neat-v3`; a `hidden` genome that the casino bot cannot load), plus a `.sha256` file. Layout (the same on every system,
    `.exe` on Windows):

    ```
    arschloch-vX.Y.Z-linux-x86_64/
      cli  arschloch-gui  README.md  champion.json  docs/{TRAINING,RULES,PLAYING,GUI}.md
    ```

  - What has been checked for the window: it builds for Linux and (cross) for Windows and runs on Linux. Nobody has
    started the Windows or macOS window yet: do that once on each system before announcing the release (start it
    from the extracted folder, press Run on the Simulation tab with a few matches, press Solve on the Solver tab).
    On macOS the downloaded binaries are unsigned: `xattr -dr com.apple.quarantine arschloch-*/` once after
    extracting, or start them from a terminal.
- **macOS** (Apple silicon and Intel) is built by
  `.github/workflows/release.yml` when a release is published (or by hand with
  its `workflow_dispatch` trigger), because macOS binaries can only be built on
  macOS. It builds `cli` and `arschloch-gui` (`-p cli -p gui`) and packs both.
- **CI** (`.github/workflows/ci.yml`) runs formatting, clippy and the whole test
  suite on Linux, Windows and macOS for every push.

## Release notes draft (next release)

- **`cli train` now trains the `hidden` feature set by default** (28 inputs: the 20 original ones, the trick group and
  the hidden-hand group). `--features base` gives the old behaviour and reproduces old runs byte for byte; saved genomes,
  configs and checkpoints load and play as before (a config without a `features` key is a `base` run).
- **New shipped champion `neat-v3`** (`champion.json` in the archives, `Neat(champion-v3)` in `cli play`): trained 300
  generations x 200 matches, selected among 11 candidates on one seed stream and confirmed on another: battery +0.745
  against +0.677 for neat-v2, head to head +0.164 against three copies of neat-v2. neat-v2 stays available
  (`Neat(champion-v2)`, `docs/baselines/neat-v2`).
- **The casino bot cannot play `neat-v3`, nor any genome trained with the new default.** Its TypeScript port (outside
  this repository) knows feature set v1 (`base`) only. Genomes for the casino must be trained with `--features base`;
  neat-v1 and neat-v2 still work there. Porting the `hidden` inputs to the casino is open work (ROADMAP open work item 4).
