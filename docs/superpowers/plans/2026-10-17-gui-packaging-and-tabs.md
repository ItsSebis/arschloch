# Desktop window: packaging, pickers, progress, solver editor, training form

Open-work items 1 and 2 (Phase 22 leftovers). Worktree branch `open-work-v0.7`.

## Goal

1. `arschloch-gui` ships next to `cli` in the Linux, Windows and macOS archives.
2. The window gets file pickers, a progress bar, a solver position editor and a training form.

## Decisions

- **Pickers: built in, no `rfd`.** `rfd` on Linux needs either GTK3 (system packages on CI) or the xdg-portal
  backend (a D-Bus stack of ~60 crates, and no dialog at all on a machine without a portal). I cannot verify a macOS or
  Windows CI build here. A small egui window that lists a folder (`browse.rs`, pure listing logic, unit-tested) works
  identically on all three systems and can be tested headlessly. Modes: open file (extension filter), save file (name
  field), folder.
- **Progress: a documented line, not a scrape.** Existing commands print no machine-readable progress when piped
  (`cli train` draws `\r` updates only on a terminal; a plain run prints nothing until the end). New, additive:
  with `ARSCHLOCH_PROGRESS=1` in the environment `cli train` prints `progress: FRACTION TEXT` lines (throttled to
  0.5 % steps) on stderr. The window sets that variable for every job, shows the last such line as a bar and keeps
  these lines out of the log. Without the variable the CLI's output is unchanged.
  Cut: a plain simulation has no progress hook (`sim::run_batch_with` is one `par_iter`; chunking it would change the
  seat rotation index). The bar for it is an indeterminate "running, mm:ss".
- **Solver editor: writes a position file and runs `cli solve`.** `cli` is a binary crate, so the window cannot call
  its parser; running the real command keeps one implementation (like the rest of the window). The editor paints
  cards (brush: hand / table / unseen / played) on a 52-card grid, takes the other seats' card counts, the table
  holder and who passed, writes the text of `docs/SOLVER.md`, and shows the ranked moves parsed from the output as a
  table. Single deck only (a double deck needs card counts per face, left as open work).
- **Training form: runs `cli train-advisor`, parses its last line.** The advisor's last line is the `cli train ...`
  command (pinned by an advisor test). `Recommendation::parse` splits it into editable fields and keeps flags it does
  not know as pass-through pairs, so item 4 can add advisor output without breaking the form.
- **Windows window:** `windows_subsystem = "windows"` in release builds (no console behind the window) and
  `CREATE_NO_WINDOW` for the `cli.exe` children, otherwise every job flashes a console.

## Tasks

1. `gui/src/browse.rs` (listing, parent, filter, hidden files, name check) + tests; picker widget in `main.rs`; buttons
   next to the strategy specs (`neat:PATH`), output file, preset path, run folder, solver file.
2. `gui/src/progress.rs` (parse `progress:` lines) + tests; `Job` sets the env var; bar in the activity panel;
   `cli/src/train_output.rs` emits the lines (test: fraction, throttle, off by default).
3. `gui/src/solver.rs` (card grid model, validation, position text, output parser) + tests, incl. the text equals the
   `docs/examples/solve-endgame.txt` position and a test that runs the real `cli solve` on it; tab in `main.rs`.
4. `gui/src/advisor.rs` (form args, recommendation parser, train args) + tests, incl. real advisor probe through
   the real `cli` (small probe) and the parsed command accepted by `cli train --help`-level parsing; tab in `main.rs`.
5. Packaging: `release.yml` (macOS builds `-p cli -p gui`, copies `arschloch-gui`), `docs/RELEASING.md`, README,
   `docs/GUI.md`, `docs/BUILDING.md`.
6. Verify: Linux release build of both, the archive layout built by hand and started from the extracted folder,
   `cargo xwin` Windows build of both (cross only), a headless window run if an X server exists (none installed here:
   check `Xvfb`/`Xephyr`; otherwise state it).
7. Roadmap text: items 1 and 2 tagged with what is verified and what is not.

## Risks

- Windows/macOS windows cannot be started here: only the Windows build is checked, nothing about its runtime.
- The `progress:` change touches `cli train`; the golden train output must be unchanged without the variable.
- `cli solve` output format is parsed by the window: a test runs the real command, so a format change fails CI.
- Folder listing on unreadable folders: show the error, do not panic.

## Revision (after re-reading)

- Dropped a "copy to clipboard" button (egui's `copy_text` is enough but not testable); the command is shown selectable.
- Dropped a double-deck editor and a plain-run progress hook (see above); both go to open work.
- Test that spawn the real `cli` locate it like the existing form test (`target/<profile>/cli`, skipped with a message
  when `cargo test -p gui` alone has not built it).
