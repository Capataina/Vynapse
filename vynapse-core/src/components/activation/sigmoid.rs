use vynapse_common::{Result, VynapseError};

use crate::traits::activation::Activation;

#[derive(Debug, Clone)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Result<Self> {
        Ok(Sigmoid)
    }
}

impl Activation for Sigmoid {
    fn get_name(&self) -> Result<&str> {
        Ok("Sigmoid Activation Function")
    }

    fn activate(&self, input: f32) -> Result<f32> {
        let result = 1.0 / (1.0 + (-input).exp());

        if result.is_nan() || result.is_infinite() {
            return Err(VynapseError::TensorError(
                "The result of activation is an invalid value.".to_string(),
            ));
        }

        Ok(result)
    }

    fn activate_tensor(&self, inputs: &[f32]) -> Result<Vec<f32>> {
        let mut activated_tensor = Vec::with_capacity(inputs.len());

        for input in inputs {
            activated_tensor.push(self.activate(*input)?);
        }

        Ok(activated_tensor)
    }
}

#[test]
fn test_sigmoid_zero_input() {
    let sigmoid = Sigmoid::new().unwrap();
    let result = sigmoid.activate(0.0).unwrap();
    assert!((result - 0.5).abs() < 1e-6); // sigmoid(0) = 0.5
}

#[test]
fn test_sigmoid_tensor_multiple_values() {
    let sigmoid = Sigmoid::new().unwrap();
    let inputs = vec![0.0, 1.0, -1.0];
    let results = sigmoid.activate_tensor(&inputs).unwrap();

    assert_eq!(results.len(), 3);
    assert!((results[0] - 0.5).abs() < 1e-6); // sigmoid(0) ≈ 0.5
    assert!((results[1] - 0.7311).abs() < 1e-3); // sigmoid(1) ≈ 0.7311
    assert!((results[2] - 0.2689).abs() < 1e-3); // sigmoid(-1) ≈ 0.2689
}

#[test]
fn test_sigmoid_extreme_values() {
    let sigmoid = Sigmoid::new().unwrap();

    // Very large positive should approach 1.0
    let large_pos = sigmoid.activate(10.0).unwrap();
    assert!(large_pos > 0.99);

    // Very large negative should approach 0.0
    let large_neg = sigmoid.activate(-10.0).unwrap();
    assert!(large_neg < 0.01);
}

#[test]
fn test_sigmoid_empty_tensor() {
    let sigmoid = Sigmoid::new().unwrap();
    let inputs = vec![];
    let results = sigmoid.activate_tensor(&inputs).unwrap();
    assert_eq!(results.len(), 0);
}
