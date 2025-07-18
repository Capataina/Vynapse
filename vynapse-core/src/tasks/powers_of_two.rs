use vynapse_common::{Result, VynapseError};

use crate::traits::task::Task;

#[derive(Debug, Clone)]
pub struct PowersOfTwo {
    max_input: usize,
}

impl PowersOfTwo {
    pub fn new(max_input: usize) -> Result<Self> {
        if max_input > 12 {
            return Err(VynapseError::ConfigError("Max input cannot be larger than 12 due to unnecessarily complex and large numbers.".to_string()));
        }

        if max_input < 1 {
            return Err(VynapseError::ConfigError(
                "Max input cannot be smaller than 1 as no learning would happen.".to_string(),
            ));
        }

        Ok(PowersOfTwo { max_input })
    }
}

impl Task for PowersOfTwo {
    fn get_dataset(&self) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
        let mut dataset = Vec::with_capacity(self.max_input);

        for i in 0..self.max_input {
            let input_vec = vec![i as f32];
            let output_vec = vec![(2_i32.pow(i as u32)) as f32];
            dataset.push((input_vec, output_vec));
        }

        Ok(dataset)
    }

    fn get_input_size(&self) -> Result<usize> {
        Ok(1)
    }

    fn get_output_size(&self) -> Result<usize> {
        Ok(1)
    }

    fn get_name(&self) -> Result<&str> {
        Ok("Powers of Two Task")
    }
}

#[test]
fn test_powers_of_two_normal_case() {
    let task = PowersOfTwo::new(3).unwrap();
    let dataset = task.get_dataset().unwrap();

    assert_eq!(dataset.len(), 3);
    assert_eq!(dataset[0], (vec![0.0], vec![1.0])); // 2^0 = 1
    assert_eq!(dataset[1], (vec![1.0], vec![2.0])); // 2^1 = 2
    assert_eq!(dataset[2], (vec![2.0], vec![4.0])); // 2^2 = 4
}

#[test]
fn test_powers_of_two_max_input_too_large() {
    let result = PowersOfTwo::new(13);
    assert!(result.is_err());
}

#[test]
fn test_powers_of_two_zero_input() {
    let result = PowersOfTwo::new(0);
    assert!(result.is_err());
}
