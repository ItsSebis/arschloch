# Results page: the estimate, comparing two files, page-only explanations (Phase 11b, open item 3)

Goal: finish the Phase 11b page. Logic stays in `web/assets/results_lib.js` (pure, tested in Node), DOM in `results.js`.

## Tasks

1. Estimate view: `estimateRows` (best first; plain column is the round-1 mean because the estimator is fitted on round 1),
   `agreement` (estimate next to the duplicate round-1 score, gap in SE, luck removed, the file's verdict) when `both` ran.
   Remove the old "estimate exists but is not drawn" note.
2. Comparison: `compareTables(a, b)` matches strategies by name; per table (mean score, average place, strength rating,
   skill duplicate, skill estimate) the difference with the combined standard error; "clear" above 2 SE, direction-aware
   (lower average place is better). Notes for strategies in one file only (scores depend on opponents), different match
   counts, a skill table only one file has. Two ways to give the second file: `cli view A.json B.json`
   (`ResultsApp::with_compare`, `/api/compare`, 404 without one) and a file input in the page (read locally, not uploaded).
3. Explanations: `explainIds(data)` lists only what the page shows (base statistics, plus skill ones present in the file);
   `explainHtml(catalog, ids)` also drops `run_analysis`-scope entries.

## Tests

Node: estimate rows/agreement, escaping, ids vs catalogue (no `analysis_*`), comparison (order, SE, direction, notes,
one-sided skill, refusal of files without `by_strategy`). Rust: `/api/compare`, the optional CLI argument.

## Real-scenario check

Two real files from `cli` (4 seats, 120 matches, 5 rounds: `--skill-score both` and `--skill-score estimate` with a
different opponent set and seed), `cli view A B`, fetch the served JS and JSON, run the served lib in Node, and load the
page in a browser (console clean, tables present).

## Risks / cut

Comparing files of different kinds (evaluate vs cli) and more than two files are not done (open work, ROADMAP item 3).
No existing JSON key changes; the page reads only.
