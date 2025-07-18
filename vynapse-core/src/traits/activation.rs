use std::fmt::Debug;

use vynapse_common::Result;

pub trait Activation: Clone + Debug {
    fn activate(&self, input: f32) -> Result<f32>;
    fn activate_tensor(&self, inputs: &[f32]) -> Result<Vec<f32>>;
    fn get_name(&self) -> Result<&str>;
}
