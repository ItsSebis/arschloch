use super::*;

fn parse(args: &[&str]) -> Result<TrainArgs, clap::Error> {
    TrainArgs::try_parse_from(std::iter::once("cli train").chain(args.iter().copied()))
}

#[test]
fn only_out_is_required_and_the_defaults_are_sensible() {
    let args = parse(&["--out", "runs/a"]).unwrap();
    assert_eq!(
        (args.player_count, args.population, args.rounds),
        (4, 150, 8)
    );
    assert_eq!(args.generations, None);
    assert_eq!(args.opponent, [] as [String; 0]);
    assert!(!args.resume && !args.quiet);
}

#[test]
fn from_conflicts_with_resume_and_population() {
    let ok = parse(&["--out", "d", "--from", "old"]).unwrap();
    assert_eq!(ok.from.as_deref(), Some(std::path::Path::new("old")));
    assert!(parse(&["--out", "d", "--from", "old", "--resume"]).is_err());
    assert!(parse(&["--out", "d", "--from", "old", "--population", "20"]).is_err());
    assert_eq!(parse(&["--out", "d"]).unwrap().from, None);
}

#[test]
fn runs_defaults_to_one_and_conflicts_with_resume() {
    assert_eq!(parse(&["--out", "d"]).unwrap().runs, 1);
    assert_eq!(parse(&["--out", "d", "--runs", "5"]).unwrap().runs, 5);
    assert!(parse(&["--out", "d", "--runs", "0"]).is_err());
    assert!(parse(&["--out", "d", "--resume", "--runs", "2"]).is_err());
}

#[test]
fn the_pass_rule_defaults_to_final_and_cannot_change_on_resume() {
    assert_eq!(
        parse(&["--out", "d"]).unwrap().pass_rule,
        engine::PassRule::Final
    );
    assert_eq!(
        parse(&["--out", "d", "--pass-rule", "free"])
            .unwrap()
            .pass_rule,
        engine::PassRule::Free
    );
    assert!(parse(&["--out", "d", "--pass-rule", "sometimes"]).is_err());
    assert!(parse(&["--out", "d", "--resume", "--pass-rule", "free"]).is_err());
}

#[test]
fn the_exchange_rule_defaults_to_forced_and_cannot_change_on_resume() {
    assert_eq!(
        parse(&["--out", "d"]).unwrap().exchange_rule,
        engine::ExchangeRule::Forced
    );
    assert_eq!(
        parse(&["--out", "d", "--exchange-rule", "free"])
            .unwrap()
            .exchange_rule,
        engine::ExchangeRule::Free
    );
    assert!(parse(&["--out", "d", "--exchange-rule", "sometimes"]).is_err());
    assert!(parse(&["--out", "d", "--resume", "--exchange-rule", "free"]).is_err());
}

#[test]
fn serve_takes_an_optional_port() {
    assert_eq!(parse(&["--out", "d"]).unwrap().serve, None);
    assert_eq!(parse(&["--out", "d", "--serve"]).unwrap().serve, Some(8080));
    assert_eq!(
        parse(&["--out", "d", "--serve", "9100"]).unwrap().serve,
        Some(9100)
    );
    assert_eq!(
        parse(&["--out", "d", "--resume", "--serve"]).unwrap().serve,
        Some(8080)
    );
    assert!(parse(&["--out", "d", "--serve", "notaport"]).is_err());
}

#[test]
fn learning_options_default_to_the_measured_settings_and_conflict_with_resume() {
    let args = parse(&["--out", "d"]).unwrap();
    assert_eq!(
        (
            args.champion_candidates,
            args.hall_of_fame,
            args.hall_interval
        ),
        (5, 0, 5)
    );
    assert!((args.weight_power - 0.2).abs() < f64::EPSILON);
    let old_style = parse(&[
        "--out",
        "d",
        "--champion-candidates",
        "1",
        "--hall-of-fame",
        "3",
        "--weight-power",
        "0.5",
    ])
    .unwrap();
    assert_eq!(
        (old_style.champion_candidates, old_style.hall_of_fame),
        (1, 3)
    );
    assert!((old_style.weight_power - 0.5).abs() < f64::EPSILON);
    assert!(parse(&["--out", "d", "--champion-candidates", "0"]).is_err());
    assert!(parse(&["--out", "d", "--hall-interval", "0"]).is_err());
    for conflicting in [
        ["--hall-of-fame", "2"],
        ["--champion-candidates", "3"],
        ["--weight-power", "0.1"],
        ["--hall-interval", "7"],
    ] {
        let mut args = vec!["--out", "d", "--resume"];
        args.extend(conflicting);
        assert!(parse(&args).is_err(), "{conflicting:?}");
    }
}

#[test]
fn the_skill_weight_defaults_to_zero_is_a_unit_interval_and_cannot_change_on_resume() {
    assert!(parse(&["--out", "d"]).unwrap().fitness_skill_weight.abs() < f64::EPSILON);
    let args = parse(&["--out", "d", "--fitness-skill-weight", "0.5"]).unwrap();
    assert!((args.fitness_skill_weight - 0.5).abs() < f64::EPSILON);
    for bad in ["-0.1", "1.5", "NaN", "inf", "x"] {
        assert!(
            parse(&["--out", "d", "--fitness-skill-weight", bad]).is_err(),
            "{bad}"
        );
    }
    assert!(parse(&["--out", "d", "--resume", "--fitness-skill-weight", "0.5"]).is_err());
}

#[test]
fn the_feature_set_defaults_to_hidden_is_validated_and_cannot_change_on_resume() {
    assert_eq!(parse(&["--out", "d"]).unwrap().features, FeatureSet::Hidden);
    for (text, set) in [
        ("base", FeatureSet::Base),
        ("trick", FeatureSet::Trick),
        ("hidden", FeatureSet::Hidden),
        ("all", FeatureSet::All),
    ] {
        assert_eq!(
            parse(&["--out", "d", "--features", text]).unwrap().features,
            set
        );
    }
    assert!(parse(&["--out", "d", "--features", "everything"]).is_err());
    assert!(parse(&["--out", "d", "--resume", "--features", "all"]).is_err());
}

#[test]
fn the_weight_power_must_be_a_positive_finite_number() {
    for bad in ["NaN", "inf", "-inf", "0", "-0.5", "abc"] {
        assert!(
            parse(&["--out", "d", "--weight-power", bad]).is_err(),
            "{bad}"
        );
    }
    assert!(parse(&["--out", "d", "--weight-power", "0.05"]).is_ok());
}

#[test]
fn the_phase_13_options_default_to_off_parse_and_cannot_change_on_resume() {
    let args = parse(&["--out", "d"]).unwrap();
    assert_eq!(args.fitness, FitnessKind::Mean);
    assert!(args.racing_top.is_none() && args.racing_matches.is_none());
    assert!(args.player_counts.is_empty() && args.decks.is_empty());
    for (text, kind) in [
        ("mean", FitnessKind::Mean),
        ("risk:0.5", FitnessKind::Risk { lambda: 0.5 }),
        ("tanh:4", FitnessKind::Tanh { kappa: 4.0 }),
    ] {
        assert_eq!(
            parse(&["--out", "d", "--fitness", text]).unwrap().fitness,
            kind
        );
    }
    for bad in ["median", "risk", "risk:-1", "tanh:0", "tanh:x", "risk:nan"] {
        assert!(parse(&["--out", "d", "--fitness", bad]).is_err(), "{bad}");
    }
    let race = parse(&[
        "--out",
        "d",
        "--racing-top",
        "0.2",
        "--racing-matches",
        "30",
    ])
    .unwrap();
    assert_eq!(
        (race.racing_top, race.racing_matches),
        (Some(0.2), Some(30))
    );
    assert!(parse(&["--out", "d", "--racing-top", "0.2"]).is_err());
    assert!(parse(&["--out", "d", "--racing-matches", "30"]).is_err());
    for bad in ["0", "1.5", "-0.1", "nan"] {
        assert!(
            parse(&["--out", "d", "--racing-top", bad, "--racing-matches", "3"]).is_err(),
            "{bad}"
        );
    }
    assert!(parse(&["--out", "d", "--racing-top", "0.5", "--racing-matches", "0"]).is_err());
    let multi = parse(&[
        "--out",
        "d",
        "--player-counts",
        "3,4,5,6",
        "--decks",
        "single,double",
    ])
    .unwrap();
    assert_eq!(multi.player_counts, vec![3, 4, 5, 6]);
    assert_eq!(
        multi.decks,
        vec![DeckVariantArg::Single, DeckVariantArg::Double]
    );
    assert!(parse(&["--out", "d", "--player-counts", "2,4"]).is_err());
    assert!(parse(&["--out", "d", "--player-counts", "7"]).is_err());
    assert!(parse(&["--out", "d", "--decks", "triple"]).is_err());
    // The lists replace the single flags: giving both is an error.
    assert!(parse(&[
        "--out",
        "d",
        "--player-counts",
        "3,4",
        "--player-count",
        "5"
    ])
    .is_err());
    assert!(parse(&[
        "--out",
        "d",
        "--decks",
        "single",
        "--deck-variant",
        "double"
    ])
    .is_err());
    for conflicting in [
        vec!["--fitness", "risk:1"],
        vec!["--racing-top", "0.5", "--racing-matches", "4"],
        vec!["--player-counts", "3,4"],
        vec!["--decks", "double"],
    ] {
        let mut args = vec!["--out", "d", "--resume"];
        args.extend(conflicting.iter());
        assert!(parse(&args).is_err(), "{conflicting:?}");
    }
}

#[test]
fn out_is_required() {
    assert!(parse(&[]).is_err());
}

#[test]
fn resume_accepts_only_generations_threads_and_quiet() {
    assert!(parse(&[
        "--out",
        "d",
        "--resume",
        "--generations",
        "50",
        "--threads",
        "2",
        "--quiet"
    ])
    .is_ok());
    for conflicting in [
        ["--population", "10"],
        ["--seed", "3"],
        ["--opponent", "lowest-legal"],
        ["--player-count", "5"],
    ] {
        let mut args = vec!["--out", "d", "--resume"];
        args.extend(conflicting);
        assert!(parse(&args).is_err(), "{conflicting:?}");
    }
}

#[test]
fn out_of_range_values_are_rejected() {
    assert!(parse(&["--out", "d", "--player-count", "2"]).is_err());
    assert!(parse(&["--out", "d", "--population", "1"]).is_err());
    assert!(parse(&["--out", "d", "--generations", "0"]).is_err());
}

#[test]
fn opponents_can_be_repeated() {
    let args = parse(&[
        "--out",
        "d",
        "--opponent",
        "lowest-legal",
        "--opponent",
        "adaptive:counting",
    ])
    .unwrap();
    assert_eq!(args.opponent, vec!["lowest-legal", "adaptive:counting"]);
}

#[test]
fn auto_decides_the_settings_so_those_flags_conflict_with_it() {
    let ok = parse(&[
        "--out",
        "d",
        "--auto",
        "--time-budget",
        "5m",
        "--preset",
        "quick",
        "--reference",
        "neat:g.json",
        "--probe-matches",
        "6",
        "--when-flat",
        "stop",
    ])
    .unwrap();
    assert!(ok.auto && ok.when_flat == Some(crate::flat::WhenFlat::Stop));
    for flag in [
        &["--population", "20"][..],
        &["--generations", "9"],
        &["--matches-per-genome", "9"],
        &["--hall-of-fame", "2"],
        &["--resume"],
        &["--runs", "2"],
        &["--from", "x"],
        &["--reeval-matches", "50"],
        &["--champion-candidates", "3"],
        &["--racing-top", "0.5"],
        &["--racing-matches", "20"],
        &["--fitness", "mean"],
        &["--features", "all"],
        &["--duplicate-rule", "last-dealt-wins"],
        &["--pass-rule", "final"],
        &["--exchange-rule", "forced"],
    ] {
        let mut args = vec!["--out", "d", "--auto"];
        args.extend_from_slice(flag);
        assert!(parse(&args).is_err(), "{flag:?}");
    }
    for flag in [
        "--time-budget",
        "--preset",
        "--reference",
        "--reference-models",
        "--reference-hall",
        "--probe-matches",
    ] {
        let value = if flag == "--preset" {
            "quick"
        } else if flag == "--probe-matches" {
            "5"
        } else {
            "5m"
        };
        assert!(
            parse(&["--out", "d", flag, value]).is_err(),
            "{flag} needs --auto"
        );
    }
    assert!(
        parse(&["--out", "d", "--resume", "--when-flat", "stop"]).is_ok(),
        "a resumed run may be watched"
    );
}
