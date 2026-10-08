# Releasing

How the binaries on a GitHub release are produced.

- **Linux and Windows** are built by hand from the tagged commit and uploaded
  with the release:
  - Linux: `cargo build --release -p cli` (needs glibc 2.34 or newer to run).
  - Windows (cross-compiled from Linux with `cargo-xwin`, C runtime linked
    statically so no Visual C++ redistributable is needed):
    `RUSTFLAGS="-C target-feature=+crt-static" cargo xwin build --release
    --target x86_64-pc-windows-msvc -p cli`.
  - Each archive holds the `cli` binary, `README.md`, `docs/TRAINING.md`,
    `docs/RULES.md` and the baseline champion (`champion.json`), plus a
    `.sha256` file.
- **macOS** (Apple silicon and Intel) is built by
  `.github/workflows/release.yml` when a release is published (or by hand with
  its `workflow_dispatch` trigger), because macOS binaries can only be built on
  macOS.
- **CI** (`.github/workflows/ci.yml`) runs formatting, clippy and the whole test
  suite on Linux, Windows and macOS for every push.
