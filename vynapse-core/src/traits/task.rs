use std::fmt::Debug;

use vynapse_common::Result;

pub trait Task: Clone + Debug {
    fn get_dataset(&self) -> Result<Vec<(Vec<f32>, Vec<f32>)>>;
    fn get_input_size(&self) -> Result<usize>;
    fn get_output_size(&self) -> Result<usize>;
    fn get_name(&self) -> Result<&str>;
}
