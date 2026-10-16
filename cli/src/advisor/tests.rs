use super::plan::{judge, MAX_MATCHES, MIN_GENERATIONS, MIN_MATCHES, MIN_POPULATION};
use super::probe::{matches_per_reference, round1_variances};
use super::references::thin;
use super::*;
use sim::training::evaluate::Round1Obs;

#[test]
fn durations_are_read_in_seconds_minutes_and_hours() {
    assert_eq!(parse_duration("90s"), Some(90.0));
    assert_eq!(parse_duration("30m"), Some(1800.0));
    assert_eq!(parse_duration("2h"), Some(7200.0));
    assert_eq!(parse_duration("1h30m"), Some(5400.0));
    for bad in ["", "30", "m", "1x", "-5m", "0m"] {
        assert_eq!(parse_duration(bad), None, "{bad}");
    }
}

#[test]
fn noisier_matches_need_more_of_them_to_reach_the_target_error() {
    // standard error = sd / sqrt(n): sd 0.4 and a target of 0.02 need 400 matches.
    assert_eq!(matches_for_error(0.4, 0.02), 400);
    assert_eq!(matches_for_error(0.2, 0.02), 100);
    assert_eq!(
        matches_for_error(0.01, 0.02),
        MIN_MATCHES,
        "never fewer than the floor"
    );
    assert_eq!(
        matches_for_error(5.0, 0.02),
        MAX_MATCHES,
        "never more than the cap"
    );
    assert_eq!(matches_for_error(0.213, 0.02) % 10, 0, "rounded up to tens");
}

#[test]
fn the_budget_decides_generations_and_a_tight_one_shrinks_the_population() {
    // 1000 rounds/s, 100 matches x 8 rounds per genome.
    let roomy = plan_run(36_000.0, 1000.0, 8, 100, 150, 4);
    assert_eq!(roomy.population, 150);
    assert!(roomy.generations >= 100, "{roomy:?}");
    let tight = plan_run(900.0, 1000.0, 8, 100, 150, 4);
    assert!(tight.population < 150, "{tight:?}");
    assert!(tight.generations >= MIN_GENERATIONS, "{tight:?}");
    // The plan fits the time it was given.
    for plan in [roomy, tight] {
        assert!(plan.estimated_seconds <= plan.budget_seconds, "{plan:?}");
    }
}

#[test]
fn the_cost_model_counts_what_the_trainer_plays() {
    // The trainer plays population x matches plus reeval (200) matches for each of the
    // champion candidates (5), the champion itself and every pool opponent.
    let plan = plan_run(3600.0, 1000.0, 8, 100, 150, 4);
    let per_generation =
        (plan.population * plan.matches_per_genome + 200 * (5 + 1 + 4)) as f64 * 8.0 / 1000.0;
    let expected = f64::from(plan.generations) * per_generation;
    assert!((plan.estimated_seconds - expected).abs() < 1e-6, "{plan:?}");
    // A bigger pool costs more per generation, so fewer generations fit.
    let bigger_pool = plan_run(3600.0, 1000.0, 8, 100, 150, 8);
    assert!(bigger_pool.generations <= plan.generations);
}

#[test]
fn a_plan_that_cannot_fit_says_so() {
    let hopeless = plan_run(10.0, 1000.0, 8, 100, 150, 4);
    let problem = hopeless
        .problem()
        .expect("10 seconds cannot train anything");
    assert!(
        problem.contains("generations") && problem.contains("budget"),
        "{problem}"
    );
    assert!(plan_run(36_000.0, 1000.0, 8, 100, 150, 4)
        .problem()
        .is_none());
}

#[test]
fn the_spread_is_pooled_as_a_variance_not_an_average_of_deviations() {
    // Two opponents with spreads 0.1 and 0.5: the pooled spread is the root of the mean variance.
    let pooled = pooled_sd(&[0.1, 0.5]);
    assert!((pooled - (0.13_f64).sqrt()).abs() < 1e-12, "{pooled}");
    assert!(pooled > 0.3, "larger than the plain average 0.3");
    assert!(pooled_sd(&[]).abs() < 1e-12);
}

#[test]
fn a_probe_needs_at_least_two_matches() {
    for bad in ["0", "1"] {
        assert!(
            AdvisorArgs::try_parse_from(["cli train-advisor", "--probe-matches", bad]).is_err(),
            "{bad}"
        );
    }
    assert!(AdvisorArgs::try_parse_from(["cli train-advisor", "--probe-matches", "2"]).is_ok());
}

#[test]
fn opponents_with_no_headroom_are_dropped_with_a_reason() {
    assert_eq!(judge(0.95), Verdict::TooWeak);
    assert_eq!(judge(0.30), Verdict::Keep);
    assert_eq!(judge(-0.20), Verdict::Keep);
    assert_eq!(judge(-0.80), Verdict::TooStrong);
    assert!(Verdict::TooWeak.reason().contains("wins almost every"));
    assert!(Verdict::TooStrong.reason().contains("almost never"));
}

#[test]
fn the_recommended_command_carries_every_chosen_setting() {
    let plan = RunPlan {
        population: 80,
        generations: 120,
        matches_per_genome: 150,
        rounds: 8,
        hall_of_fame: 5,
        hall_interval: 6,
        estimated_seconds: 100.0,
        budget_seconds: 200.0,
    };
    let command = command_line(
        &plan,
        4,
        "single",
        &["lowest-legal".into(), "adaptive:reading".into()],
        "runs/auto",
    );
    assert_eq!(
        command,
        "cli train --out runs/auto --player-count 4 --deck-variant single --population 80 --generations 120 \
         --matches-per-genome 150 --rounds 8 --hall-of-fame 5 --hall-interval 6 \
         --opponent lowest-legal --opponent adaptive:reading"
    );
}

#[test]
fn a_run_folder_with_spaces_is_quoted() {
    let plan = RunPlan {
        population: 10,
        generations: 2,
        matches_per_genome: 4,
        rounds: 1,
        hall_of_fame: 0,
        hall_interval: 0,
        estimated_seconds: 1.0,
        budget_seconds: 2.0,
    };
    let command = command_line(&plan, 4, "single", &[], "my runs/a b");
    assert!(
        command.starts_with("cli train --out \"my runs/a b\" --player-count 4 "),
        "{command}"
    );
}

#[test]
fn the_advice_runs_a_real_probe_and_ends_in_a_runnable_command() {
    let args = AdvisorArgs::try_parse_from([
        "cli train-advisor",
        "--probe-matches",
        "8",
        "--time-budget",
        "10m",
        "--opponent",
        "lowest-legal",
        "--opponent",
        "random-legal",
        "--out",
        "runs/x",
    ])
    .unwrap();
    let text = advise(&args, false).unwrap().text;
    assert!(
        text.contains("lowest-legal") && text.contains("random-legal"),
        "{text}"
    );
    assert!(text.contains("rounds per second"), "{text}");
    let command = text.lines().last().unwrap().trim();
    assert!(
        command.starts_with("cli train --out runs/x --player-count 4 "),
        "{command}"
    );
    assert!(command.contains("--opponent "), "{command}");
    // `cli train` itself accepts the recommended settings.
    let words = command.split_whitespace().skip(2); // `cli train` is dispatched before parsing
    let train =
        crate::train_args::TrainArgs::try_parse_from(std::iter::once("cli train").chain(words));
    assert!(train.is_ok(), "{:?}: {command}", train.err());
}

#[test]
fn a_bad_budget_is_explained() {
    let args = AdvisorArgs::try_parse_from(["cli train-advisor", "--time-budget", "soon"]).unwrap();
    assert!(advise(&args, false)
        .err()
        .unwrap()
        .to_string()
        .contains("--time-budget"));
}

fn row(spec: &str, rounds: usize, seconds: f64) -> Row {
    Row {
        spec: spec.into(),
        mean: 0.0,
        error: 0.0,
        verdict: Verdict::Keep,
        sd: 0.3,
        rounds,
        seconds,
        var_round1: 1.0,
        var_adjusted: 0.8,
        observations: 40,
    }
}

#[test]
fn the_speed_is_that_of_the_chosen_pool_not_of_every_candidate() {
    let rows = [row("fast", 1000, 1.0), row("slow-and-dropped", 1000, 99.0)];
    let chosen: Vec<&Row> = rows.iter().filter(|r| r.spec == "fast").collect();
    assert!((speed_of(&chosen) - 1000.0).abs() < 1e-9);
    let all: Vec<&Row> = rows.iter().collect();
    assert!(
        (speed_of(&all) - 20.0).abs() < 1e-9,
        "what the old code measured"
    );
}

#[test]
fn references_are_thinned_evenly_and_default_to_adaptive() {
    assert_eq!(thin(&[1, 2, 3], 6), vec![1, 2, 3]);
    assert_eq!(thin(&(0..12).collect::<Vec<_>>(), 4), vec![0, 3, 6, 9]);
    let args = AdvisorArgs::try_parse_from(["cli train-advisor"]).unwrap();
    let refs = build_references(&args).unwrap();
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].name, "adaptive");
}

#[test]
fn bad_references_are_explained() {
    let bad = |extra: &[&str]| {
        let args = AdvisorArgs::try_parse_from(
            std::iter::once("cli train-advisor").chain(extra.iter().copied()),
        )
        .unwrap();
        format!("{:#}", build_references(&args).err().unwrap())
    };
    assert!(bad(&["--reference", "nonsense"]).contains("--reference `nonsense`"));
    assert!(bad(&["--reference-models", "/nonexistent-folder"]).contains("--reference-models"));
    assert!(bad(&["--reference-hall", "/nonexistent-folder"]).contains("--reference-hall"));
    let empty =
        std::env::temp_dir().join(format!("arschloch-advisor-empty-{}", std::process::id()));
    std::fs::create_dir_all(&empty).unwrap();
    assert!(bad(&["--reference-models", empty.to_str().unwrap()]).contains("no genome file"));
    std::fs::remove_dir_all(&empty).unwrap();
}

#[test]
fn the_luck_adjustment_cannot_leave_more_variance_than_the_scores_had() {
    // The score is the hand's strength plus noise: the regression removes the first part.
    let observations: Vec<Round1Obs> = (0..60)
        .map(|i| {
            let x = f64::from(i % 10) / 10.0;
            let noise = f64::from((i * 7) % 5) / 50.0;
            Round1Obs {
                score: 2.0 * x + noise,
                features: [x, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        })
        .collect();
    let (plain, adjusted) = round1_variances(&observations);
    assert!(plain > 0.0 && adjusted < 0.2 * plain, "{plain} {adjusted}");
    assert_eq!(round1_variances(&[]), (0.0, 0.0));
}

#[test]
fn several_references_are_split_over_the_probe_matches_and_the_run_is_auto_ready() {
    let args = AdvisorArgs::try_parse_from([
        "cli train-advisor",
        "--probe-matches",
        "8",
        "--time-budget",
        "10m",
        "--opponent",
        "lowest-legal",
        "--opponent",
        "random-legal",
        "--reference",
        "lowest-legal",
        "--reference",
        "random-legal",
        "--out",
        "runs/x",
    ])
    .unwrap();
    let advice = advise(&args, true).unwrap();
    assert!(
        advice
            .text
            .contains("references `lowest-legal`, `random-legal`")
            || advice.text.contains("`lowest-legal`, `random-legal`"),
        "{}",
        advice.text
    );
    assert!(advice.text.contains("4 against each"), "{}", advice.text);
    assert!(advice.text.contains("Luck adjustment"), "{}", advice.text);
    assert!(advice.text.contains("Starting the run"), "{}", advice.text);
    assert!(!advice.pool.is_empty() && advice.plan.population >= MIN_POPULATION);
}

#[test]
fn the_header_states_the_true_match_count_and_a_small_sample_has_no_luck_share() {
    // 3 matches over 2 references: 2 against each (at least 2), so 4 per candidate.
    assert_eq!(matches_per_reference(3, 2), 2);
    assert_eq!(matches_per_reference(40, 3), 14);
    let probe = |matches: &str| {
        let args = AdvisorArgs::try_parse_from([
            "cli train-advisor",
            "--probe-matches",
            matches,
            "--time-budget",
            "10m",
            "--opponent",
            "lowest-legal",
            "--reference",
            "lowest-legal",
            "--reference",
            "random-legal",
        ])
        .unwrap();
        advise(&args, false).unwrap().text
    };
    let few = probe("3");
    assert!(few.contains("Probe: 4 matches per candidate"), "{few}");
    assert!(
        few.contains("n/a (only 4 round-1 observations"),
        "a fit on 4 observations is not a luck share: {few}"
    );
    let many = probe("40");
    assert!(many.contains("Probe: 40 matches per candidate"), "{many}");
    assert!(
        many.contains("from 40 round-1 observations") && !many.contains("n/a (only"),
        "{many}"
    );
}
