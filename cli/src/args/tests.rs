use super::*;

fn args_with_strategies(player_count: u8, strategy_count: usize) -> Args {
    Args {
        player_count,
        deck_variant: DeckVariantArg::Single,
        duplicate_rule: DuplicateRuleArg::FirstDealtWins,
        pass_rule: engine::PassRule::default(),
        exchange_rule: engine::ExchangeRule::default(),
        matches: 1,
        rounds: 1,
        strategies: vec![StrategyArg::Fixed(FixedStrategy::LowestLegal); strategy_count],
        threads: 0,
        seed: 0,
        output: PathBuf::from("results.json"),
        skill_score: SkillScoreArg::Off,
        explain: false,
        bootstrap_resamples: 200,
        useful_passes: None,
        useful_pass_sample: 0.05,
        useful_pass_margin: 0.0,
    }
}

#[test]
fn validate_accepts_matching_strategy_count() {
    assert!(validate(&args_with_strategies(4, 4)).is_ok());
}

#[test]
fn duplicate_modes_need_matches_in_whole_groups() {
    let mut args = args_with_strategies(4, 4);
    args.matches = 6;
    args.skill_score = SkillScoreArg::Duplicate;
    let error = validate(&args).unwrap_err().to_string();
    assert!(error.contains("multiple of --player-count (4)"), "{error}");
    args.skill_score = SkillScoreArg::Both;
    assert!(validate(&args).is_err());
    args.skill_score = SkillScoreArg::Estimate;
    assert!(validate(&args).is_ok());
    args.skill_score = SkillScoreArg::Duplicate;
    args.matches = 8;
    assert!(validate(&args).is_ok());
}

#[test]
fn skill_score_flags_parse_with_defaults() {
    let args = Args::try_parse_from([
        "cli",
        "--player-count",
        "3",
        "--matches",
        "3",
        "--strategy",
        "lowest-legal",
        "--strategy",
        "lowest-legal",
        "--strategy",
        "lowest-legal",
    ])
    .unwrap();
    assert_eq!(args.skill_score, SkillScoreArg::Off);
    assert!(!args.explain);
    assert_eq!(args.bootstrap_resamples, 200);
    let args = Args::try_parse_from([
        "cli",
        "--player-count",
        "3",
        "--matches",
        "3",
        "--strategy",
        "lowest-legal",
        "--strategy",
        "lowest-legal",
        "--strategy",
        "lowest-legal",
        "--skill-score",
        "both",
        "--explain",
        "--bootstrap-resamples",
        "0",
    ])
    .unwrap();
    assert!(args.skill_score.uses_duplicate() && args.skill_score == SkillScoreArg::Both);
    assert!(args.explain && args.bootstrap_resamples == 0);
}

#[test]
fn useful_pass_flags_default_off_and_are_range_checked() {
    let base = [
        "cli",
        "--player-count",
        "3",
        "--matches",
        "3",
        "--strategy",
        "lowest-legal",
        "--strategy",
        "lowest-legal",
        "--strategy",
        "lowest-legal",
    ];
    let args = Args::try_parse_from(base).unwrap();
    assert_eq!(args.useful_passes, None);
    assert!((args.useful_pass_sample - 0.05).abs() < 1e-12);
    assert!(args.useful_pass_margin.abs() < 1e-12);
    let with = |extra: &[&str]| Args::try_parse_from(base.iter().chain(extra));
    let args = with(&[
        "--useful-passes",
        "8",
        "--useful-pass-sample",
        "1",
        "--useful-pass-margin",
        "0.1",
    ])
    .unwrap();
    assert_eq!(args.useful_passes, Some(8));
    assert!(with(&["--useful-passes", "0"]).is_err());
    assert!(with(&["--useful-pass-sample", "1.5"]).is_err());
    assert!(with(&["--useful-pass-margin", "-1"]).is_err());
}

#[test]
fn validate_rejects_strategy_count_mismatch() {
    assert!(validate(&args_with_strategies(4, 3)).is_err());
}

#[test]
fn strategy_arg_builds_matching_strategy_names() {
    assert_eq!(
        StrategyArg::Fixed(FixedStrategy::LowestLegal)
            .build()
            .name(),
        "LowestLegal"
    );
    assert_eq!(
        StrategyArg::Fixed(FixedStrategy::GreedyHighest)
            .build()
            .name(),
        "GreedyHighest"
    );
    assert_eq!(
        StrategyArg::Fixed(FixedStrategy::RandomLegal)
            .build()
            .name(),
        "RandomLegal"
    );
    assert_eq!(
        StrategyArg::Fixed(FixedStrategy::HoldBackPairs)
            .build()
            .name(),
        "HoldBackPairs"
    );
    assert_eq!(
        StrategyArg::Fixed(FixedStrategy::CardCounter)
            .build()
            .name(),
        "CardCounter"
    );
    assert_eq!(
        StrategyArg::Fixed(FixedStrategy::EndgameDenial)
            .build()
            .name(),
        "EndgameDenial"
    );
}

#[test]
fn parses_fixed_strategy_names_unchanged() {
    assert_eq!(
        "lowest-legal".parse::<StrategyArg>().unwrap(),
        StrategyArg::Fixed(FixedStrategy::LowestLegal)
    );
    assert_eq!(
        "card-counter".parse::<StrategyArg>().unwrap(),
        StrategyArg::Fixed(FixedStrategy::CardCounter)
    );
}

#[test]
fn parses_bare_adaptive_to_defaults() {
    assert_eq!(
        "adaptive".parse::<StrategyArg>().unwrap(),
        StrategyArg::Adaptive(sim::AdaptiveConfig::default())
    );
}

#[test]
fn parses_configured_adaptive() {
    let parsed = "adaptive:counting".parse::<StrategyArg>().unwrap();
    assert_eq!(
        parsed,
        StrategyArg::Adaptive(sim::AdaptiveConfig {
            counting: true,
            denial: sim::DenialMode::Off,
            deception_rate: 0.0,
            tempo: false,
            bully: false,
        })
    );
}

#[test]
fn rejects_options_on_a_fixed_strategy() {
    assert!("lowest-legal:counting".parse::<StrategyArg>().is_err());
}

#[test]
fn rejects_unknown_strategy_name() {
    assert!("nonexistent".parse::<StrategyArg>().is_err());
}

#[test]
fn build_produces_an_adaptive_strategy_instance() {
    let arg: StrategyArg = "adaptive:reading,deception=0.2".parse().unwrap();
    let strategy = arg.build();
    assert!(strategy.name().starts_with("Adaptive("));
}

fn genome_file(name: &str) -> PathBuf {
    let mut population = neat::Population::new(
        sim::FEATURE_COUNT,
        neat::NeatConfig {
            population_size: 4,
            ..neat::NeatConfig::default()
        },
        1,
    )
    .unwrap();
    let genome = population.genomes()[0].clone();
    population.set_fitness(vec![0.0; 4]);
    let path = std::env::temp_dir().join(format!("{name}-{}.json", std::process::id()));
    sim::GenomeFile::new(genome).unwrap().save(&path).unwrap();
    path
}

#[test]
fn parses_a_neat_spec_and_names_the_player_after_the_file() {
    let path = genome_file("champ");
    let arg: StrategyArg = format!("neat:{}", path.display()).parse().unwrap();
    std::fs::remove_file(&path).unwrap();
    assert!(matches!(arg, StrategyArg::Neat(_)));
    assert!(arg.build().name().starts_with("Neat(champ-"));
}

#[test]
fn a_neat_spec_without_a_path_is_rejected() {
    for spec in ["neat", "neat:", "neat:   "] {
        let error = spec.parse::<StrategyArg>().unwrap_err();
        assert!(error.contains("neat:PATH"), "{spec}: {error}");
    }
}

#[test]
fn a_missing_genome_file_is_an_argument_error_naming_the_path() {
    let error = "neat:/nonexistent/champ.json"
        .parse::<StrategyArg>()
        .unwrap_err();
    assert!(error.contains("/nonexistent/champ.json"), "{error}");
}

#[test]
fn a_stale_genome_file_is_refused_at_parse_time() {
    let path = genome_file("stale");
    let mut value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    value["feature_names"][0] = serde_json::json!("renamed_feature");
    std::fs::write(&path, value.to_string()).unwrap();
    let error = format!("neat:{}", path.display())
        .parse::<StrategyArg>()
        .unwrap_err();
    std::fs::remove_file(&path).unwrap();
    assert!(error.contains("does not fit this build"), "{error}");
}

#[test]
fn two_different_genome_files_with_the_same_name_are_rejected() {
    // Results are grouped by player name, so two different genomes
    // called `Neat(champion)` would be silently merged into one row.
    let source = genome_file("dupe-source");
    let base = std::env::temp_dir().join(format!("dupe-dirs-{}", std::process::id()));
    let (a, b) = (base.join("a/champion.json"), base.join("b/champion.json"));
    for target in [&a, &b] {
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        std::fs::copy(&source, target).unwrap();
    }
    let spec = |p: &PathBuf| {
        format!("neat:{}", p.display())
            .parse::<StrategyArg>()
            .unwrap()
    };
    let mut args = args_with_strategies(3, 0);
    args.strategies = vec![
        spec(&a),
        spec(&b),
        StrategyArg::Fixed(FixedStrategy::LowestLegal),
    ];
    let error = validate(&args).unwrap_err().to_string();
    assert!(error.contains("Neat(champion)"), "{error}");
    // The same file in two seats is fine: it is one player.
    args.strategies = vec![
        spec(&a),
        spec(&a),
        StrategyArg::Fixed(FixedStrategy::LowestLegal),
    ];
    assert!(validate(&args).is_ok());
    std::fs::remove_dir_all(&base).unwrap();
    std::fs::remove_file(&source).unwrap();
}

#[test]
fn deck_variant_arg_converts_to_engine_type() {
    assert_eq!(
        engine::DeckVariant::from(DeckVariantArg::Single),
        engine::DeckVariant::Single
    );
    assert_eq!(
        engine::DeckVariant::from(DeckVariantArg::Double),
        engine::DeckVariant::Double
    );
}

#[test]
fn duplicate_rule_arg_converts_to_engine_type() {
    assert_eq!(
        engine::DuplicateRule::from(DuplicateRuleArg::FirstDealtWins),
        engine::DuplicateRule::FirstDealtWins
    );
    assert_eq!(
        engine::DuplicateRule::from(DuplicateRuleArg::LastDealtWins),
        engine::DuplicateRule::LastDealtWins
    );
}

// The tests above all construct `Args` via a struct literal, so none of
// them exercise clap's actual parsing pipeline (attribute macros,
// `value_parser`s, required-arg checks). The tests below drive
// `Args::try_parse_from` with real argv to close that gap — in
// particular, to catch a `value_parser` that is mistyped relative to its
// field (e.g. an `i64`-typed range parser attached to a `u8`/`usize`
// field), which previously caused a runtime panic on every parse rather
// than a compile error.

#[test]
fn try_parse_from_accepts_player_count_at_range_boundaries() {
    for player_count in ["3", "6"] {
        let argv = [
            "arschloch",
            "--player-count",
            player_count,
            "--matches",
            "1",
            "--strategy",
            "lowest-legal",
        ];
        let parsed = Args::try_parse_from(argv);
        assert!(
            parsed.is_ok(),
            "player_count={player_count} should be accepted, got {parsed:?}"
        );
    }
}

#[test]
fn try_parse_from_rejects_player_count_outside_range() {
    for player_count in ["2", "7"] {
        let argv = [
            "arschloch",
            "--player-count",
            player_count,
            "--matches",
            "1",
            "--strategy",
            "lowest-legal",
        ];
        assert!(
            Args::try_parse_from(argv).is_err(),
            "player_count={player_count} should be rejected"
        );
    }
}

#[test]
fn try_parse_from_accepts_minimum_matches_and_rounds() {
    let argv = [
        "arschloch",
        "--player-count",
        "4",
        "--matches",
        "1",
        "--rounds",
        "1",
        "--strategy",
        "lowest-legal",
    ];
    let parsed = Args::try_parse_from(argv).expect("minimum matches/rounds should parse");
    assert_eq!(parsed.matches, 1);
    assert_eq!(parsed.rounds, 1);
}

#[test]
fn try_parse_from_rejects_zero_matches_and_zero_rounds() {
    let zero_matches = [
        "arschloch",
        "--player-count",
        "4",
        "--matches",
        "0",
        "--strategy",
        "lowest-legal",
    ];
    assert!(Args::try_parse_from(zero_matches).is_err());

    let zero_rounds = [
        "arschloch",
        "--player-count",
        "4",
        "--matches",
        "1",
        "--rounds",
        "0",
        "--strategy",
        "lowest-legal",
    ];
    assert!(Args::try_parse_from(zero_rounds).is_err());
}
