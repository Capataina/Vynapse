use crate::traits::genome::Genome;
use std::fmt::Debug;
use vynapse_common::Result;

pub trait Mutation: Clone + Debug {
    fn mutate<G: Genome>(&self, genome: &mut G, rate: f32) -> Result<()>;
}
