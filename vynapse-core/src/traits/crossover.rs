use std::fmt::Debug;
use vynapse_common::Result;

use crate::traits::genome::Genome;

pub trait Crossover: Clone + Debug {
    fn crossover<G: Genome>(&self, parent1: &G, parent2: &G) -> Result<G>;
}
