use vynapse_common::Result;

use crate::traits::task::Task;

#[derive(Clone, Debug)]
pub struct Xor;

impl Xor {
    pub fn new() -> Result<Self> {
        Ok(Xor)
    }
}

impl Task for Xor {
    fn get_dataset(&self) -> vynapse_common::Result<Vec<(Vec<f32>, Vec<f32>)>> {
        let data: Vec<(Vec<f32>, Vec<f32>)> = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];
        Ok(data)
    }

    fn get_input_size(&self) -> vynapse_common::Result<usize> {
        Ok(2)
    }

    fn get_output_size(&self) -> vynapse_common::Result<usize> {
        Ok(1)
    }

    fn get_name(&self) -> vynapse_common::Result<&str> {
        Ok("XOR Task")
    }
}
