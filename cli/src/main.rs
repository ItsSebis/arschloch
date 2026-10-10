//! Entry point: parse CLI args, run a batch of simulated matches, write
//! the JSON results file, and print a human-readable summary. See
//! docs/ARCHITECTURE.md, "cli".

mod args;
mod batch;
mod evaluate;
mod extended_summary;
mod output;
mod play;
mod stats_doc;
mod summary;
mod train;
mod train_args;
mod train_output;
mod watch;

use std::sync::Arc;

use anyhow::Context;
use clap::Parser;

fn main() -> anyhow::Result<()> {
    // `cli train ...` evolves a player, `cli watch ...` shows a run in the
    // browser and `cli play ...` lets you play in it; everything else is a
    // simulation run.
    match std::env::args().nth(1).as_deref() {
        Some("train") => return train::run(std::env::args().skip(2)),
        Some("watch") => return watch::run(std::env::args().skip(2)),
        Some("evaluate") => return evaluate::run(std::env::args().skip(2)),
        Some("stats-doc") => return stats_doc::run(std::env::args().skip(2)),
        Some("play") => return play::run(std::env::args().skip(2)),
        _ => {}
    }
    let args = args::Args::parse();
    args::validate(&args)?;

    let output_file = output::create_output_file(&args.output)?;

    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .context("failed to configure thread pool")?;
    }

    let strategies: Vec<Arc<dyn sim::Strategy>> = args
        .strategies
        .iter()
        .cloned()
        .map(args::StrategyArg::build)
        .collect();

    let configs: Vec<sim::MatchConfig> = (0..args.matches)
        .map(|i| sim::MatchConfig {
            player_count: args.player_count,
            deck_variant: args.deck_variant.into(),
            duplicate_rule: args.duplicate_rule.into(),
            rounds: args.rounds,
            seed: args
                .seed
                .wrapping_add(u64::try_from(i).expect("match index fits in u64")),
            pass_rule: args.pass_rule,
            exchange_rule: args.exchange_rule,
        })
        .collect();

    let batch::BatchRun { results, skill } = batch::run(args.skill_score, &configs, &strategies)?;
    let statistics = sim::aggregate(&results);
    let extended = sim::extended_stats::aggregate_extended_with(
        &results,
        &sim::extended_stats::ExtendedOptions {
            bootstrap_resamples: args.bootstrap_resamples,
            ..sim::extended_stats::ExtendedOptions::default()
        },
    );

    let useful = batch::useful_passes(&args, &configs, &strategies);

    output::write_json_output(
        output_file,
        &results,
        &statistics,
        &extended,
        skill.as_ref(),
        useful.as_ref(),
    )?;

    // The original summary comes first and is unchanged; the Phase 12
    // sections are appended after it.
    println!("{}", summary::render_summary(&args, &statistics));
    print!(
        "{}",
        extended_summary::render_extended_summary(
            &args,
            &statistics,
            &extended,
            skill.as_ref(),
            useful.as_ref()
        )
    );
    Ok(())
}
