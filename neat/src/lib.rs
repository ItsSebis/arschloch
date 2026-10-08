//! A generic NEAT (neuroevolution of augmenting topologies) library:
//! genomes, historical markings, mutation, crossover, speciation, a
//! feedforward evaluator and the generational loop. It knows nothing
//! about cards or games; callers evaluate genomes and report fitness.
//! See docs/superpowers/specs/2026-10-08-neat-engine-design.md.

pub mod config;
pub mod error;

pub use config::NeatConfig;
pub use error::NeatError;

pub mod innovation;

pub use innovation::InnovationTracker;
