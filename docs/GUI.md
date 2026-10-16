# The desktop window

A small native window (`arschloch-gui`, the `gui` crate, built with egui) to configure and run simulations without
typing the command line, and to read what the program is doing. It runs the real `cli` program with the arguments its
form builds, so it behaves exactly like the command line. Chart-heavy parts stay in the browser, where they are already
good: the window only starts them.

## Starting it

`cargo run --release -p gui` (it is left out of plain `cargo build/test`, which skip it; `--workspace` and `-p gui`
include it). It looks for `cli` next to itself, then on the `PATH`, so keep both binaries in one folder. On Linux it
needs a running X11 or Wayland session; there is nothing else to install.

## The tabs

- **Simulation:** every option of a plain `cli` run: players, deck, duplicate rule, pass rule, exchange rule, matches,
  rounds, seed, threads, skill score, bootstrap resamples, output file, one strategy per seat (a name, an
  `adaptive:...` spec, `neat:PATH`, `endgame-solver:...`, or pick one from the list), the statistics explanations and the
  useful-passes analysis. The exact command line is shown under the Run button, and the form tells you in words what
  the CLI would reject (for example a duplicate skill score needs matches in whole groups). Presets save and load the
  form as JSON.
- **Any command:** type any `cli` command (`train ...`, `evaluate ...`, `solve file.txt`, `solver-gap ...`); output is
  streamed to the log.
- **In the browser:** starts `cli play` (play against the models), `cli view FILE` (the charts and tables of the
  simulation's results file, `cli view` also works on its own) or `cli watch RUNFOLDER` (training dashboard and the
  run analysis page) and opens the address they print.

## The activity log

The bottom panel shows everything that runs: the command line, every line of its output (stdout and stderr) and how
it ended (`[finished: exit code 0]` or `[cancelled]`). **Cancel** stops the running job (also a running server).
Every line is also appended to `arschloch-gui.log` in the folder the window was started from.

## Limits

One job at a time; no file pickers (paths are typed); the results of a simulation are read in the log and in the JSON
file, and the browser page `cli view` shows them as tables and bars. The window is not part of the release archives yet.

## For developers

`form.rs` (the form and the arguments it builds, unit-tested; a test runs the real `cli` with them and checks that a
wrong flag fails), `job.rs` (a child process with streamed output and cancel), `main.rs` (drawing only).
