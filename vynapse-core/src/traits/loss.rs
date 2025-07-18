use std::fmt::Debug;

use vynapse_common::Result;

pub trait Loss: Clone + Debug {
    fn calculate(&self, predicted: &[f32], actual: &[f32]) -> Result<f32>;
    fn get_name(&self) -> Result<&str>;
}
