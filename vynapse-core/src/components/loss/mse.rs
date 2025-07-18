use vynapse_common::{Result, VynapseError};

use crate::traits::loss::Loss;

#[derive(Clone, Debug)]
pub struct MeanSquaredError;

impl MeanSquaredError {
    pub fn new() -> Result<Self> {
        Ok(MeanSquaredError)
    }
}

impl Loss for MeanSquaredError {
    fn get_name(&self) -> Result<&str> {
        Ok("Mean Squared Error")
    }

    fn calculate(&self, predicted: &[f32], actual: &[f32]) -> Result<f32> {
        if predicted.len() != actual.len() {
            return Err(VynapseError::EvolutionError(
                "The number of predictions do not match the number of actuals.".to_string(),
            ));
        }

        if predicted.is_empty() || actual.is_empty() {
            return Err(VynapseError::EvolutionError(
                "Cannot calculate predictions on empty arrays.".to_string(),
            ));
        }

        let mut sum_squared_errors: f32 = 0 as f32;

        for i in 0..predicted.len() {
            let diff = predicted[i] - actual[i];
            sum_squared_errors += diff * diff;
        }

        let mse = sum_squared_errors / predicted.len() as f32;

        Ok(mse)
    }
}

#[test]
fn test_mse_perfect_prediction() {
    let mse = MeanSquaredError::new().unwrap();
    let predicted = vec![1.0, 2.0, 3.0];
    let actual = vec![1.0, 2.0, 3.0];

    let result = mse.calculate(&predicted, &actual).unwrap();
    assert_eq!(result, 0.0);
}

#[test]
fn test_mse_known_calculation() {
    let mse = MeanSquaredError::new().unwrap();
    let predicted = vec![2.0, 4.0];
    let actual = vec![1.0, 2.0];

    let result = mse.calculate(&predicted, &actual).unwrap();
    assert_eq!(result, 2.5); // ((2-1)² + (4-2)²)/2 = (1+4)/2 = 2.5
}

#[test]
fn test_mse_mismatched_lengths() {
    let mse = MeanSquaredError::new().unwrap();
    let predicted = vec![1.0, 2.0];
    let actual = vec![1.0];

    let result = mse.calculate(&predicted, &actual);
    assert!(result.is_err());
}

#[test]
fn test_mse_empty_arrays() {
    let mse = MeanSquaredError::new().unwrap();
    let predicted = vec![];
    let actual = vec![];

    let result = mse.calculate(&predicted, &actual);
    assert!(result.is_err());
}
