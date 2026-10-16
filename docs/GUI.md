# The desktop window

A small native window (`arschloch-gui`, the `gui` crate, built with egui) to configure and run simulations without
typing the command line, and to read what the program is doing. It runs the real `cli` program with the arguments its
form builds, so it behaves exactly like the command line. Chart-heavy parts stay in the browser, where they are already
good: the window only starts them.

## Starting it

In a release archive: run `arschloch-gui` (`arschloch-gui.exe`) from the extracted folder. From a checkout:
`cargo run --release -p gui` (it is left out of plain `cargo build/test`, which skip it; `--workspace` and `-p gui`
include it). It looks for `cli` next to itself, then on the `PATH`, so keep both binaries in one folder. On Linux it
needs a running X11 or Wayland session; there is nothing else to install. `arschloch-gui --tab solver` opens on that
tab (`simulation`, `solver`, `training`, `command`, `browser`). On macOS an unsigned download may need
`xattr -dr com.apple.quarantine` on the extracted folder.

## The tabs

- **Simulation:** every option of a plain `cli` run: players, deck, duplicate rule, pass rule, exchange rule, matches,
  rounds, seed, threads, skill score, bootstrap resamples, output file, one strategy per seat (a name, an
  `adaptive:...` spec, `neat:PATH`, `endgame-solver:...`, or pick one from the list), the statistics explanations and the
  useful-passes analysis. The exact command line is shown under the Run button, and the form tells you in words what
  the CLI would reject (for example a duplicate skill score needs matches in whole groups). Presets save and load the
  form as JSON.
- **Solver:** analyse one real situation with the exact solver (Phase 21, `docs/SOLVER.md`) without writing the position
  file. Pick a brush (my hand, on the table, unseen, played), click cards (a new position starts with every card
  played; `Load the example` fills in `docs/examples/solve-endgame.txt`), set the other seats' card counts, who holds
  the combo on the table and who passed (the seats between the holder and you count as passed on their own), then
  press **Solve**. The window writes the position file and runs the real `cli solve`; the ranked moves (expected
  finishing place, score, best and worst place, how often a move is best) appear as a table. The form says in words
  what is missing (for example "6 cards are painted unseen but the other seats hold 5"). Single deck only; a double
  deck position needs the text file.
- **Training:** the advisor of Phase 18 as a form. Set players, deck, a budget (preset or `2h`), the candidate
  opponents (empty: the advisor's battery) and the run folder, press **Ask the advisor** (it plays a probe, a minute or
  so), read what it measured, edit the recommended population, generations, matches, rounds, hall of fame, run
  folder, and press **Start this training**. Settings of later advisor versions that the form does not know are kept
  and passed to `cli train` unchanged.
- **Any command:** type any `cli` command (`train ...`, `evaluate ...`, `solve file.txt`, `solver-gap ...`); output is
  streamed to the log.
- **In the browser:** starts `cli play` (play against the models), `cli view FILE` (the charts and tables of the
  simulation's results file, `cli view` also works on its own) or `cli watch RUNFOLDER` (training dashboard and the
  run analysis page) and opens the address they print.

## File pickers

Every file or folder field has a button (Browse, Genome file): a small built-in window that lists the folder, goes up,
enters folders and chooses a file with a click (a save picker also takes a typed name; hidden entries are not
shown). It is the same on every system and needs no desktop dialog service. You can still type or paste a path.

## The activity log

The bottom panel shows everything that runs: the command line, every line of its output (stdout and stderr) and how
it ended (`[finished: exit code 0]` or `[cancelled]`). **Cancel** stops the running job (also a running server).
While a job runs a bar shows how far it is. A command reports progress by printing `progress: FRACTION TEXT` on
stderr when the environment variable `ARSCHLOCH_PROGRESS` is set (the window sets it for every job; the lines are
not written to the log). Today only `cli train` does (generation and genomes, steps of 0.5 %); a plain simulation
prints nothing until it ends, so its bar only moves to say "running" with the elapsed time.
Every line is also appended to `arschloch-gui.log` in the folder the window was started from.

## Limits

One job at a time (closing the window stops it); the log file and the default
`results.json` are written in the folder the window was started from (started from a file manager that may be a folder
you cannot write to: start it from a terminal or give full paths); the results of a simulation are read in the log and in the JSON
file, and the browser page `cli view` shows them as tables and bars. The window is in the release archives next to `cli` (it was only run on Linux; the Windows and macOS builds
have not been started by anyone yet).

## For developers

`form.rs` (the form and the arguments it builds, unit-tested; a test runs the real `cli` with them and checks that a
wrong flag fails), `job.rs` (a child process with streamed output and cancel), `browse.rs` (the picker's folder
listing), `progress.rs` (reads the `progress:` lines), `solver.rs` (the card grid, the position text, reading the
ranked table of `cli solve`), `advisor.rs` (the advisor form and the recommended command split into fields), `main.rs`
(drawing only). Tests of the last three run the real `cli` (`solve`, `train-advisor`, `train`), so the formats the
window reads cannot drift unnoticed; `testing.rs` finds the binary the same way the form test does.
