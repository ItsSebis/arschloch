//! The text the advisor prints.

use std::fmt::Write as _;

use super::plan::{matches_for_error, RunPlan, POOL_SIZE, TARGET_ERROR};
use super::probe::{matches_per_reference, Probe, MIN_LUCK_OBSERVATIONS};
use super::references::Reference;
use super::AdvisorArgs;

/// The text the advisor prints: what was measured, what was chosen and why. With `auto` the
/// last line says the run starts now, instead of a command to copy.
pub(super) fn render(
    args: &AdvisorArgs,
    references: &[Reference],
    probe: &Probe,
    pool: &[String],
    plan: &RunPlan,
    command: &str,
    auto: bool,
) -> String {
    let mut out = String::new();
    let names: Vec<String> = references.iter().map(|r| format!("`{}`", r.name)).collect();
    let per_reference = matches_per_reference(args.probe_matches, references.len());
    let _ = writeln!(
        out,
        "Probe: {} matches per candidate, reference player{} {}.",
        per_reference * references.len(),
        if references.len() > 1 { "s" } else { "" },
        names.join(", ")
    );
    if references.len() > 1 {
        let _ = writeln!(
            out,
            "  The matches are split over the references ({per_reference} against each); a candidate's score is their mean."
        );
    }
    let _ = writeln!(
        out,
        "\nCandidate opponents (the reference player's score against each, +1 best .. -1 worst):"
    );
    for row in &probe.rows {
        let _ = writeln!(
            out,
            "  {:<44} {:+.3} +- {:.3}  {}",
            row.spec,
            row.mean,
            row.error,
            row.verdict.reason()
        );
    }
    let _ = writeln!(
        out,
        "\nRecommended pool: {}.\n  Why: the strongest {} opponents that left headroom.",
        pool.join(", "),
        POOL_SIZE
    );
    let _ = writeln!(
        out,
        "Matches per genome: {}.\n  Why: one match's score varies by {:.3} (measured), so this many matches bring the \
         standard error of a fitness measurement to about {TARGET_ERROR}{}.",
        plan.matches_per_genome,
        probe.sd_match,
        if plan.matches_per_genome < matches_for_error(probe.sd_match, TARGET_ERROR) {
            " only if the time allowed it: it did not, so the error is larger"
        } else {
            ""
        }
    );
    let _ = writeln!(
        out,
        "Population {} and {} generations.\n  Why: the pool opponents were played at about {:.0} rounds per second; the plan needs about {:.0} s of the {:.0} s budget \
         (the rest is kept for what the estimate leaves out).",
        plan.population, plan.generations, probe.rounds_per_sec, plan.estimated_seconds, plan.budget_seconds
    );
    if plan.hall_of_fame > 0 {
        let _ = writeln!(out, "Hall of fame {} every {} generations.\n  Why: a long run needs past champions as opponents.", plan.hall_of_fame, plan.hall_interval);
    }
    let luck = probe.luck_share.map_or_else(
        || {
            format!(
                "n/a (only {} round-1 observations against the pool, at least {MIN_LUCK_OBSERVATIONS} needed: raise --probe-matches)",
                probe.luck_observations
            )
        },
        |share| {
            format!(
                "{:.0}% (from {} round-1 observations)",
                100.0 * share,
                probe.luck_observations
            )
        },
    );
    let _ = writeln!(
        out,
        "Luck adjustment: the six hand features explain {luck} of the variance of a round-1 score against the pool.\n  \
         Why this is only reported: every genome of a generation plays the same deals, so `--fitness-skill-weight` cannot cancel \
         deal luck inside a generation (it only weights round 1 more). The duplicate-deal skill score cuts this noise when \
         comparing finished players (`cli --skill-score duplicate`, `cli evaluate`)."
    );
    if auto {
        let _ = writeln!(
            out,
            "\nStarting the run with these settings (the same as:\n  {command}\n)"
        );
    } else {
        let _ = writeln!(out, "\nRun it with:\n  {command}");
    }
    out
}
