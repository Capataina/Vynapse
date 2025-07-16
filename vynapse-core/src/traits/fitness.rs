use std::fmt::Debug;

use crate::traits::genome::Genome;
use vynapse_common::Result;

pub trait Fitness: Clone + Debug {
    fn evaluate<G: Genome>(&self, genome: &G) -> Result<f32>;
}
