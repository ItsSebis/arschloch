# Releasing

How the binaries on a GitHub release are produced.

- **Linux and Windows** are built by hand from the tagged commit and uploaded
  with the release:
  - Linux: `cargo build --release -p cli -p gui` (needs glibc 2.34 or newer to run; the window needs an X11 or Wayland
    session and loads its libraries when it starts, so building needs no extra system packages).
  - Windows (cross-compiled from Linux with `cargo-xwin`, C runtime linked
    statically so no Visual C++ redistributable is needed):
    `RUSTFLAGS="-C target-feature=+crt-static" cargo xwin build --release
    --target x86_64-pc-windows-msvc -p cli -p gui`. The window is a Windows GUI program (no console behind it).
  - Each archive holds the `cli` binary, `arschloch-gui` (`arschloch-gui.exe`; it must stay next to `cli`, which it
    runs), `README.md`, `docs/TRAINING.md`, `docs/RULES.md`, `docs/PLAYING.md`, `docs/GUI.md` and the baseline
    champion (`champion.json`, from `docs/baselines/neat-v2`), plus a `.sha256` file. Layout (the same on every system,
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
