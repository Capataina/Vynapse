use rand::{Rng, rng};

use crate::traits::genome::Genome;
use vynapse_common::{Result, VynapseError};

#[derive(Clone, Debug)]
pub struct FixedTopologyGenome {
    pub weights: Vec<f32>,
    pub shape: Vec<usize>,
}

impl FixedTopologyGenome {
    pub fn new_random(shape: Vec<usize>) -> Self {
        let mut weight_number = 0;

        if shape.is_empty() {
            return FixedTopologyGenome {
                weights: Vec::new(),
                shape,
            };
        }

        if shape.len() == 1 {
            weight_number = shape[0];
        } else {
            for i in 0..shape.len() - 1 {
                weight_number += shape[i] * shape[i + 1];
            }
        }

        let mut rng = rng();
        let mut weights: Vec<f32> = Vec::new();

        for _ in 0..weight_number {
            weights.push(rng.random_range(-1.0..=1.0));
        }

        FixedTopologyGenome { weights, shape }
    }
}

impl Genome for FixedTopologyGenome {
    fn get_weights(&self) -> Vec<f32> {
        self.weights.clone()
    }

    fn set_weights(&mut self, weights: Vec<f32>) -> Result<()> {
        if self.weights.len() != weights.len() {
            return Err(VynapseError::EvolutionError(
                "Given weight's length doesn't match the genome's weight's length.".to_string(),
            ));
        } else {
            self.weights = weights;
        }
        Ok(())
    }
}

#[cfg(test)]
#[test]
fn test_simple_network_shape() {
    let shape = vec![2, 3, 1];
    let genome = FixedTopologyGenome::new_random(shape.clone());

    assert_eq!(genome.shape, vec![2, 3, 1]);
    // For [2,3,1]: (2*3) + (3*1) = 6 + 3 = 9 weights
    assert_eq!(genome.weights.len(), 9);
}

#[test]
fn test_single_layer_network() {
    let shape = vec![5];
    let genome = FixedTopologyGenome::new_random(shape.clone());

    assert_eq!(genome.shape, vec![5]);
    assert_eq!(genome.weights.len(), 5);
}

#[test]
fn test_xor_network_shape() {
    let shape = vec![2, 4, 1]; // XOR network from your roadmap
    let genome = FixedTopologyGenome::new_random(shape.clone());

    assert_eq!(genome.shape, vec![2, 4, 1]);
    // (2*4) + (4*1) = 8 + 4 = 12 weights
    assert_eq!(genome.weights.len(), 12);
}

#[test]
fn test_weights_are_in_range() {
    let shape = vec![2, 3, 1];
    let genome = FixedTopologyGenome::new_random(shape);

    for weight in &genome.weights {
        assert!(
            *weight >= -1.0 && *weight <= 1.0,
            "Weight {weight} is outside range [-1.0, 1.0]",
        );
    }
}

#[test]
fn test_different_random_genomes() {
    let shape = vec![2, 3, 1];
    let genome1 = FixedTopologyGenome::new_random(shape.clone());
    let genome2 = FixedTopologyGenome::new_random(shape);

    // Very unlikely that all weights are identical
    assert_ne!(genome1.weights, genome2.weights);
}

#[test]
fn test_larger_network() {
    let shape = vec![10, 20, 15, 5];
    let genome = FixedTopologyGenome::new_random(shape.clone());

    // (10*20) + (20*15) + (15*5) = 200 + 300 + 75 = 575
    assert_eq!(genome.weights.len(), 575);
    assert_eq!(genome.shape, shape);
}
