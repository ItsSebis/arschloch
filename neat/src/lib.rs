//! A generic NEAT (neuroevolution of augmenting topologies) library:
//! genomes, historical markings, mutation, crossover, speciation, a
//! feedforward evaluator and the generational loop. It knows nothing
//! about cards or games; callers evaluate genomes and report fitness.
//! See docs/superpowers/specs/2026-10-08-neat-engine-design.md.

pub mod config;
pub mod crossover;
pub mod error;
pub mod genome;
pub mod innovation;
pub mod mutation;
pub mod network;
pub mod population;
pub mod species;

pub use config::NeatConfig;
pub use error::NeatError;
pub use genome::{ConnectionGene, Genome, NodeGene, NodeKind};
pub use innovation::InnovationTracker;
pub use network::Network;
pub use population::{GenerationReport, Population, PopulationState};
pub use species::SpeciesStats;
