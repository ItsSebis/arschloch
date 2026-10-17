# Training advisor, the rest (Phase 18, open item 4)

Goal: finish what `cli train-advisor` left open. Existing flags, JSON keys and the printed `cli train ...` last line
(the desktop window parses it) stay as they are; everything is additive.

## Tasks

1. Speed over the chosen pool: each probe row keeps its own rounds and seconds; the recommended plan uses
   rounds-per-second of the pool rows only (the old code divided all rounds by the whole probe time, slow or fast
   candidates that were dropped included).
2. Better reference players: `--reference SPEC` becomes repeatable (default `adaptive` when no reference option is
   given); `--reference neat:PATH` is the starting genome; `--reference-models DIR` (every genome `*.json` of a folder,
   at most 5) and `--reference-hall RUN_DIR` (the hall of fame of an earlier run, read from its checkpoint, at most 5).
   A candidate's score is the mean over the references; the probe matches per candidate stay `--probe-matches`, split
   over the references (so cost does not grow), at least 2 each.
3. Luck adjustment where it can cut noise: the probe uses `evaluate_with_round1`; per candidate the round-1 scores are
   regressed on the six hand features (`skill_term::fit_deal_model`) and the advisor prints how much of the round-1
   variance the hand explains. It does NOT recommend `--fitness-skill-weight` for training: all genomes of a generation
   play the same deals, so the adjustment cannot cancel deal luck inside a generation (module doc of `skill_term`).
   The number is reported so the user can see the size of the effect; the duplicate-deal skill score itself belongs to
   comparing finished players (`cli --skill-score duplicate`, `cli evaluate`), where it is already used.
4. `cli train --auto`: runs the advisor (candidates = `--opponent` or the battery; `--time-budget`, `--preset`,
   `--reference*`, `--probe-matches`, all `requires = auto`), applies pool, population, generations, matches and hall of
   fame to the arguments, prints the advice and runs the training. The settings land in the run's `config.json` as for
   any run; the advice text is also saved as `advisor.txt` in the run folder. Conflicts: `--resume`, `--from`, `--runs`,
   and every flag the advisor decides.
5. Flat-slope stop/advice: `--when-flat advise|stop` (any `cli train`, also with `--resume`). Every 5 generations from
   generation 30 the observer analyses the run's `events.jsonl` with the Phase 17 analysis. Flat = verdict has
   `stop_early`, `noise_limited`, `pool_limited` or `model_limited` (flat last-third trend, not still improving). `advise`
   prints the verdict messages once per change; `stop` stops when `stop_early` fires (the last 40% added nothing
   visible) through a new default-false `TrainObserver::should_stop`, so `--resume` continues the run.
6. Cut: the web "New training" form (needs a server-side hook from the web crate into the advisor and a long-running
   probe job with progress; the desktop window already has the form with reasons). Stays open work.

## Tests

Unit: pool-only speed, reference parsing/caps, mean over references, luck share arithmetic, auto argument conflicts,
`--when-flat` decision from verdict codes, `should_stop` ends a run early and the run resumes. Real scenario: `train --auto`
with a tiny budget; `--when-flat stop` on a real run; advisor against a models folder / hall of an earlier run.

## Risks

Timing numbers vary run to run (the probe measures speed); tests do not pin them. `should_stop` is checked once per
generation, so a stop takes effect after the current generation; no change to a run that never asks for it (reference
batch outputs unchanged).
