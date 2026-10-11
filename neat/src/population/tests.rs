use super::*;

fn tiny_config() -> NeatConfig {
    NeatConfig {
        population_size: 30,
        ..NeatConfig::default()
    }
}

#[test]
fn allocate_sums_to_the_total_and_respects_proportions() {
    assert_eq!(allocate(&[1.0, 1.0, 2.0], 8), vec![2, 2, 4]);
    let uneven = allocate(&[1.0, 1.0, 1.0], 10);
    assert_eq!(uneven.iter().sum::<usize>(), 10);
    assert!(uneven.iter().all(|&n| (3..=4).contains(&n)));
}

#[test]
fn allocate_never_seats_a_zero_weight_entry() {
    let seats = allocate(&[0.0, 3.0, 0.0, 1.0], 7);
    assert_eq!(seats[0], 0);
    assert_eq!(seats[2], 0);
    assert_eq!(seats.iter().sum::<usize>(), 7);
}

#[test]
fn survivor_count_keeps_at_least_one_and_at_most_all() {
    assert_eq!(survivor_count(1, 0.2), 1);
    assert_eq!(survivor_count(10, 0.2), 2);
    assert_eq!(survivor_count(10, 1.0), 10);
    assert_eq!(survivor_count(3, 0.0), 1);
}

#[test]
fn new_population_has_the_configured_size_and_valid_genomes() {
    let population = Population::new(3, tiny_config(), 1).unwrap();
    assert_eq!(population.genomes().len(), 30);
    assert_eq!(population.generation(), 0);
    assert!(population.best().is_none());
    assert!(population.genomes().iter().all(|g| g.num_inputs() == 3));
}

#[test]
fn invalid_setups_are_rejected() {
    assert!(Population::new(0, tiny_config(), 1).is_err());
    let bad = NeatConfig {
        population_size: 1,
        ..tiny_config()
    };
    assert!(Population::new(2, bad, 1).is_err());
}

#[test]
fn advance_keeps_the_population_size_constant() {
    let mut population = Population::new(2, tiny_config(), 7).unwrap();
    for generation in 0..10 {
        let fitness = (0..30).map(|i| f64::from(i % 7) - 3.0).collect();
        population.set_fitness(fitness);
        let report = population.advance();
        assert_eq!(report.generation, generation);
        assert_eq!(population.genomes().len(), 30);
        assert_eq!(report.species.iter().map(|s| s.size).sum::<usize>(), 30);
    }
    assert_eq!(population.generation(), 10);
}

#[test]
fn report_statistics_describe_the_evaluated_generation() {
    let mut population = Population::new(2, tiny_config(), 3).unwrap();
    let fitness: Vec<f64> = (0..30).map(f64::from).collect();
    population.set_fitness(fitness);
    let report = population.advance();
    assert!((report.best_fitness - 29.0).abs() < f64::EPSILON);
    assert!((report.min_fitness - 0.0).abs() < f64::EPSILON);
    assert!((report.mean_fitness - 14.5).abs() < 1e-12);
    assert!((report.median_fitness - 14.5).abs() < 1e-12);
    assert_eq!(report.champion_index, 29);
    let (best, fitness) = population.best().unwrap();
    assert!((fitness - 29.0).abs() < f64::EPSILON);
    assert_eq!(best.num_inputs(), 2);
}

#[test]
fn best_never_regresses() {
    let mut population = Population::new(2, tiny_config(), 3).unwrap();
    population.set_fitness((0..30).map(f64::from).collect());
    population.advance();
    population.set_fitness(vec![-5.0; 30]);
    population.advance();
    assert!((population.best().unwrap().1 - 29.0).abs() < f64::EPSILON);
}

#[test]
fn the_champion_survives_unchanged_in_a_large_species() {
    let config = NeatConfig {
        population_size: 20,
        compatibility_threshold: 1000.0,
        ..NeatConfig::default()
    };
    let mut population = Population::new(2, config, 5).unwrap();
    let champion = population.genomes()[4].clone();
    let mut fitness = vec![0.0; 20];
    fitness[4] = 10.0;
    population.set_fitness(fitness);
    population.advance();
    assert!(population.genomes().contains(&champion));
}

#[test]
fn stagnant_species_stop_reproducing_but_the_best_one_never_does() {
    let config = NeatConfig {
        population_size: 40,
        compatibility_threshold: 0.0001,
        min_compatibility_threshold: 0.0001,
        stagnation_limit: 2,
        ..NeatConfig::default()
    };
    let mut population = Population::new(2, config, 9).unwrap();
    for _ in 0..6 {
        // Constant fitness: nothing ever improves, every species
        // stagnates, yet the population must keep its size.
        population.set_fitness(vec![1.0; 40]);
        population.advance();
        assert_eq!(population.genomes().len(), 40);
    }
}

#[test]
#[should_panic(expected = "set_fitness must be called")]
fn advancing_without_fitness_panics() {
    Population::new(2, tiny_config(), 1).unwrap().advance();
}

#[test]
#[should_panic(expected = "finite")]
fn nan_fitness_panics() {
    let mut population = Population::new(2, tiny_config(), 1).unwrap();
    let mut fitness = vec![0.0; 30];
    fitness[3] = f64::NAN;
    population.set_fitness(fitness);
}

#[test]
#[should_panic(expected = "one fitness per genome")]
fn wrong_fitness_length_panics() {
    Population::new(2, tiny_config(), 1)
        .unwrap()
        .set_fitness(vec![0.0; 3]);
}

#[test]
fn the_smallest_legal_population_keeps_evolving() {
    let config = NeatConfig {
        population_size: 2,
        compatibility_threshold: 0.0001,
        min_compatibility_threshold: 0.0001,
        ..NeatConfig::default()
    };
    let mut population = Population::new(2, config, 1).unwrap();
    for generation in 0..20 {
        population.set_fitness(vec![f64::from(generation), -1.0]);
        population.advance();
        assert_eq!(population.genomes().len(), 2);
    }
}

#[test]
fn a_wide_input_layer_works() {
    let mut population = Population::new(40, tiny_config(), 2).unwrap();
    for _ in 0..3 {
        population.set_fitness((0..30).map(f64::from).collect());
        population.advance();
    }
    assert!(population.genomes().iter().all(|g| g.num_inputs() == 40));
}

#[test]
fn every_genome_stays_valid_across_many_mixed_lineage_generations() {
    // High structural mutation rates make lineages that discovered
    // different hidden nodes interbreed constantly, which is where
    // out-of-order node ids and cycles would show up.
    let config = NeatConfig {
        population_size: 60,
        add_node_rate: 0.4,
        add_connection_rate: 0.4,
        toggle_enable_rate: 0.2,
        ..NeatConfig::default()
    };
    let mut population = Population::new(3, config, 11).unwrap();
    for generation in 0..40 {
        for genome in population.genomes() {
            Genome::from_parts(
                genome.num_inputs(),
                genome.nodes().to_vec(),
                genome.connections().to_vec(),
            )
            .unwrap_or_else(|error| panic!("generation {generation}: {error}"));
        }
        let fitness = (0..60)
            .map(|i| f64::from((i * 13 + generation) % 17))
            .collect();
        population.set_fitness(fitness);
        population.advance();
    }
}

#[test]
fn a_culled_species_does_not_survive_to_capture_next_generations_offspring() {
    let config = NeatConfig {
        population_size: 30,
        compatibility_threshold: 0.0001,
        min_compatibility_threshold: 0.0001,
        stagnation_limit: 3,
        ..NeatConfig::default()
    };
    let mut population = Population::new(2, config, 4).unwrap();
    population.set_fitness(vec![1.0; 30]);
    population.advance();
    assert!(population.species.len() > 2, "need several species");
    // Species 1 holds the best fitness (protected). Species 0 is
    // hopelessly stagnant, and is first in line to capture any
    // genome near its representative (first-fit assignment), which is
    // exactly what lets a zombie species swallow healthy offspring.
    population.species[1].best_fitness = 100.0;
    population.species[0].best_fitness = 50.0;
    population.species[0].stagnation = 99;
    population.species[0].representative = population.genomes()[0].clone();
    population.species[1].representative = population.genomes()[1].clone();
    let doomed = population.species[0].id;
    // The champion (genome 1) is in the protected species, not the doomed one.
    let mut fitness = vec![1.0; 30];
    fitness[1] = 2.0;
    population.set_fitness(fitness);
    population.advance();
    assert!(
        population.species.iter().all(|s| s.id != doomed),
        "a species with no offspring quota must be dropped, not kept as a trap"
    );
}

#[test]
fn the_species_holding_the_generations_champion_is_never_culled() {
    // Fitness is noisy, so a species' record is the luckiest sample it
    // ever had and a stagnant species can easily hold *this*
    // generation's best genome. Culling it would delete the champion's
    // whole lineage.
    let config = NeatConfig {
        population_size: 30,
        compatibility_threshold: 0.0001,
        min_compatibility_threshold: 0.0001,
        stagnation_limit: 3,
        ..NeatConfig::default()
    };
    let mut population = Population::new(2, config, 4).unwrap();
    population.set_fitness(vec![1.0; 30]);
    population.advance();
    assert!(population.species.len() > 2, "need several species");
    population.species[1].best_fitness = 100.0; // holds the all-time record
    population.species[0].best_fitness = 50.0;
    population.species[0].stagnation = 99;
    population.species[0].representative = population.genomes()[0].clone();
    population.species[1].representative = population.genomes()[1].clone();
    let champions_species = population.species[0].id;
    let mut fitness = vec![1.0; 30];
    fitness[0] = 5.0; // genome 0 (species 0) is this generation's champion
    population.set_fitness(fitness);
    population.advance();
    assert!(
        population.species.iter().any(|s| s.id == champions_species),
        "the champion's species must keep reproducing"
    );
}

#[test]
fn default_config_forms_several_species_within_the_first_generations() {
    // Random initial weights put genomes ~0.3 apart; a threshold far
    // above that would hold the whole population in one species for
    // dozens of generations, with no protection for new structure.
    let probe = [0.3, -0.7, 0.1, 0.9, -0.2, 0.5, -0.4, 0.8, 0.0, -0.6];
    let mut population = Population::new(10, NeatConfig::default(), 21).unwrap();
    let mut scratch = Vec::new();
    let mut species_at_generation_14 = 0;
    for generation in 0..15 {
        let fitness: Vec<f64> = population
            .genomes()
            .iter()
            .map(|g| crate::Network::compile(g).activate(&probe, &mut scratch))
            .collect();
        population.set_fitness(fitness);
        let report = population.advance();
        if generation == 14 {
            species_at_generation_14 = report.species.len();
        }
    }
    assert!(
        species_at_generation_14 >= 3,
        "only {species_at_generation_14} species after 15 generations"
    );
}

#[test]
fn default_config_keeps_the_species_count_stable() {
    // Fitness is a network's output on a fixed probe input: smooth,
    // deterministic, and it rewards drifting weights, so the
    // population keeps changing structure like a real run.
    let probe = [0.3, -0.7, 0.1, 0.9, -0.2, 0.5, -0.4, 0.8, 0.0, -0.6];
    let config = NeatConfig::default();
    let target = config.target_species;
    let mut population = Population::new(10, config, 21).unwrap();
    let mut scratch = Vec::new();
    for generation in 0..200 {
        let fitness: Vec<f64> = population
            .genomes()
            .iter()
            .map(|g| crate::Network::compile(g).activate(&probe, &mut scratch))
            .collect();
        population.set_fitness(fitness);
        let report = population.advance();
        if generation >= 100 {
            let count = report.species.len();
            assert!(
                (2..=3 * target).contains(&count),
                "generation {generation}: {count} species (target {target})"
            );
        }
    }
}

fn run_generations(population: &mut Population, from: u32, to: u32) {
    for generation in from..to {
        let fitness = (0..30)
            .map(|i| f64::from((i * 7 + generation) % 11))
            .collect();
        population.set_fitness(fitness);
        population.advance();
    }
}

#[test]
fn a_warm_start_keeps_the_genomes_but_restarts_the_counters() {
    let mut pop = Population::new(2, tiny_config(), 5).unwrap();
    run_generations(&mut pop, 0, 3);
    let genomes_before = pop.genomes().to_vec();
    let warm = Population::warm_start(pop.snapshot(), tiny_config(), 99).unwrap();
    assert_eq!(warm.generation(), 0);
    assert!(warm.best().is_none());
    assert_eq!(warm.genomes(), &genomes_before[..]);
}

#[test]
fn a_warm_start_forgets_the_old_fitness_scale_of_every_species() {
    let mut pop = Population::new(2, tiny_config(), 5).unwrap();
    for _ in 0..6 {
        // A flat fitness never improves a species' best: stagnation builds.
        pop.set_fitness(vec![1.0; 30]);
        pop.advance();
    }
    let state = pop.snapshot();
    assert!(
        pop.species.iter().any(|s| s.stagnation > 0 && s.age > 0),
        "the source has history to forget"
    );
    let warm = Population::warm_start(state, tiny_config(), 1).unwrap();
    assert!(!warm.species.is_empty());
    assert!(warm
        .species
        .iter()
        .all(|s| s.stagnation == 0 && s.age == 0 && s.best_fitness == f64::NEG_INFINITY));
}

#[test]
fn a_warm_start_with_another_population_size_is_refused() {
    let pop = Population::new(2, tiny_config(), 1).unwrap();
    let bigger = NeatConfig {
        population_size: 31,
        ..NeatConfig::default()
    };
    assert!(matches!(
        Population::warm_start(pop.snapshot(), bigger, 1),
        Err(NeatError::InvalidConfig(_))
    ));
}

#[test]
fn a_warm_start_is_deterministic_in_its_seed_and_differs_between_seeds() {
    let mut pop = Population::new(2, tiny_config(), 5).unwrap();
    run_generations(&mut pop, 0, 3);
    let state = pop.snapshot();
    let advance = |seed| {
        let mut warm = Population::warm_start(state.clone(), tiny_config(), seed).unwrap();
        run_generations(&mut warm, 0, 2);
        serde_json::to_string(&warm.snapshot()).unwrap()
    };
    assert_eq!(advance(7), advance(7));
    assert_ne!(advance(7), advance(8));
}

#[test]
fn a_restored_snapshot_continues_exactly_like_the_original() {
    let mut straight = Population::new(2, tiny_config(), 5).unwrap();
    run_generations(&mut straight, 0, 8);

    let mut first_half = Population::new(2, tiny_config(), 5).unwrap();
    run_generations(&mut first_half, 0, 4);
    // Through JSON, as a checkpoint file would be.
    let json = serde_json::to_string(&first_half.snapshot()).unwrap();
    let mut resumed = Population::restore(serde_json::from_str(&json).unwrap()).unwrap();
    assert_eq!(resumed.generation(), 4);
    run_generations(&mut resumed, 4, 8);

    assert_eq!(
        serde_json::to_string(&resumed.snapshot()).unwrap(),
        serde_json::to_string(&straight.snapshot()).unwrap()
    );
    assert_eq!(
        serde_json::to_string(resumed.genomes()).unwrap(),
        serde_json::to_string(straight.genomes()).unwrap()
    );
}

#[test]
fn restore_rejects_a_snapshot_that_does_not_fit_its_config() {
    let population = Population::new(2, tiny_config(), 1).unwrap();
    let mut value = serde_json::to_value(population.snapshot()).unwrap();
    value["genomes"].as_array_mut().unwrap().pop();
    let state = serde_json::from_value(value).unwrap();
    assert!(Population::restore(state).is_err());
}

#[test]
#[should_panic(expected = "fitness is pending")]
fn snapshotting_mid_generation_panics() {
    let mut population = Population::new(2, tiny_config(), 1).unwrap();
    population.set_fitness(vec![0.0; 30]);
    let _ = population.snapshot();
}

#[test]
fn same_seed_and_fitness_give_identical_populations() {
    let run = || {
        let mut population = Population::new(2, tiny_config(), 42).unwrap();
        for _ in 0..8 {
            let fitness = (0..30).map(|i| f64::from((i * 7) % 11)).collect();
            population.set_fitness(fitness);
            population.advance();
        }
        serde_json::to_string(population.genomes()).unwrap()
    };
    assert_eq!(run(), run());
}
