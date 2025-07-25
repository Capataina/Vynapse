use std::fmt::Debug;

use vynapse_common::Result;

pub trait Genome: Clone + Debug {
    fn get_weights(&self) -> Vec<f32>;
    fn set_weights(&mut self, weights: Vec<f32>) -> Result<()>;
}
