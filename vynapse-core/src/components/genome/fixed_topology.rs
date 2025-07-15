use rand::{Rng, random_bool, random_ratio, rng};

use crate::{VynapseError, traits::genome::Genome};

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

    fn set_weights(&mut self, weights: Vec<f32>) -> crate::Result<()> {
        if self.weights.len() != weights.len() {
            return Err(VynapseError::EvolutionError(
                "Given weight's length doesn't match the genome's weight's length.".to_string(),
            ));
        } else {
            self.weights = weights;
        }
        Ok(())
    }

    fn mutate(&mut self, mutation_rate: f32) -> crate::Result<()> {
        if !(0.0..=1.0).contains(&mutation_rate) {
            return Err(VynapseError::EvolutionError(
                "Mutation rate has to be between 0% and 100% (0.0 - 1.0).".to_string(),
            ));
        }

        let mut rng = rng();
        for weight in &mut self.weights {
            if random_bool(mutation_rate as f64 / 1.0) {
                let noise = rng.random_range(-1.0..=1.0);
                *weight = (*weight + noise).clamp(-5.0, 5.0);
            }
        }

        Ok(())
    }

    fn crossover(&self, other: &Self) -> crate::Result<Self> {
        if self.shape != other.shape {
            return Err(VynapseError::EvolutionError(
                "The shapes of the genomes don't match for the crossover.".to_string(),
            ));
        }

        let mut child_genome: Vec<f32> = Vec::new();

        for i in 0..self.weights.len() {
            if random_ratio(1, 2) {
                child_genome.push(self.weights[i]);
            } else {
                child_genome.push(other.weights[i]);
            }
        }

        Ok(FixedTopologyGenome {
            weights: child_genome,
            shape: self.shape.clone(),
        })
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

#[test]
fn test_mutation_changes_weights() {
    let shape = vec![2, 3, 1];
    let mut genome = FixedTopologyGenome::new_random(shape);
    let original_weights = genome.weights.clone();

    genome.mutate(1.0).unwrap(); // 100% mutation rate

    assert_ne!(genome.weights, original_weights);
}

#[test]
fn test_mutation_no_change_with_zero_rate() {
    let shape = vec![2, 3, 1];
    let mut genome = FixedTopologyGenome::new_random(shape);
    let original_weights = genome.weights.clone();

    genome.mutate(0.0).unwrap(); // 0% mutation rate

    assert_eq!(genome.weights, original_weights);
}

#[test]
fn test_mutation_preserves_shape() {
    let shape = vec![5, 10, 3];
    let mut genome = FixedTopologyGenome::new_random(shape.clone());

    genome.mutate(0.5).unwrap();

    assert_eq!(genome.shape, shape);
    assert_eq!(genome.weights.len(), 80); // (5*10) + (10*3) = 80
}

#[test]
fn test_mutation_weights_stay_in_bounds() {
    let shape = vec![2, 2];
    let mut genome = FixedTopologyGenome::new_random(shape);

    for _ in 0..100 {
        genome.mutate(1.0).unwrap();
    }

    for weight in &genome.weights {
        assert!(*weight >= -5.0 && *weight <= 5.0);
    }
}

#[test]
fn test_mutation_single_weight() {
    let shape = vec![1];
    let mut genome = FixedTopologyGenome::new_random(shape);

    genome.mutate(1.0).unwrap();

    assert_eq!(genome.weights.len(), 1);
    assert!(genome.weights[0] >= -5.0 && genome.weights[0] <= 5.0);
}

#[test]
fn test_crossover_basic() {
    let shape = vec![2, 2, 1];
    let parent1 = FixedTopologyGenome::new_random(shape.clone());
    let parent2 = FixedTopologyGenome::new_random(shape.clone());

    let child = parent1.crossover(&parent2).unwrap();

    assert_eq!(child.shape, shape);
    assert_eq!(child.weights.len(), parent1.weights.len());
}

#[test]
fn test_crossover_inherits_from_both_parents() {
    let shape = vec![2, 2];
    let mut parent1 = FixedTopologyGenome::new_random(shape.clone());
    let mut parent2 = FixedTopologyGenome::new_random(shape.clone());

    parent1.weights = vec![1.0, 1.0, 1.0, 1.0];
    parent2.weights = vec![2.0, 2.0, 2.0, 2.0];

    let child = parent1.crossover(&parent2).unwrap();

    for weight in &child.weights {
        assert!(*weight == 1.0 || *weight == 2.0);
    }
}

#[test]
fn test_crossover_single_weight() {
    let shape = vec![1];
    let mut parent1 = FixedTopologyGenome::new_random(shape.clone());
    let mut parent2 = FixedTopologyGenome::new_random(shape.clone());

    parent1.weights = vec![5.0];
    parent2.weights = vec![10.0];

    let child = parent1.crossover(&parent2).unwrap();

    assert!(child.weights[0] == 5.0 || child.weights[0] == 10.0);
}

#[test]
fn test_crossover_preserves_shape() {
    let shape = vec![3, 4, 2, 1];
    let parent1 = FixedTopologyGenome::new_random(shape.clone());
    let parent2 = FixedTopologyGenome::new_random(shape.clone());

    let child = parent1.crossover(&parent2).unwrap();

    assert_eq!(child.shape, shape);
}

#[test]
fn test_crossover_different_shapes_error() {
    let shape1 = vec![2, 3, 1];
    let shape2 = vec![2, 4, 1];
    let parent1 = FixedTopologyGenome::new_random(shape1);
    let parent2 = FixedTopologyGenome::new_random(shape2);

    let result = parent1.crossover(&parent2);

    assert!(result.is_err());
}

#[test]
fn test_crossover_empty_shape_error() {
    let shape2 = vec![];

    let result = FixedTopologyGenome::new_random(shape2);
    assert!(result.shape.is_empty());
}

#[test]
fn test_crossover_large_network() {
    let shape = vec![100, 50, 25, 10];
    let parent1 = FixedTopologyGenome::new_random(shape.clone());
    let parent2 = FixedTopologyGenome::new_random(shape.clone());

    let child = parent1.crossover(&parent2).unwrap();

    assert_eq!(child.weights.len(), 6500); // (100*50) + (50*25) + (25*10)
}

#[test]
fn test_mutation_extreme_rates() {
    let shape = vec![2, 2];
    let mut genome = FixedTopologyGenome::new_random(shape);

    let result1 = genome.mutate(999.0); // Way above 1.0
    let result2 = genome.mutate(-5.0); // Negative rate

    assert!(result1.is_err());
    assert!(result2.is_err());
}

#[test]
fn test_crossover_identical_parents() {
    let shape = vec![2, 2];
    let parent1 = FixedTopologyGenome::new_random(shape.clone());
    let parent2 = parent1.clone();

    let child = parent1.crossover(&parent2).unwrap();

    assert_eq!(child.weights, parent1.weights);
    assert_eq!(child.shape, parent1.shape);
}

#[test]
fn test_multiple_crossovers() {
    let shape = vec![3, 2];
    let parent1 = FixedTopologyGenome::new_random(shape.clone());
    let parent2 = FixedTopologyGenome::new_random(shape.clone());

    for _ in 0..10 {
        let child = parent1.crossover(&parent2).unwrap();
        assert_eq!(child.shape, shape);
        assert_eq!(child.weights.len(), 6);
    }
}

#[test]
fn test_mutation_then_crossover() {
    let shape = vec![2, 3];
    let mut parent1 = FixedTopologyGenome::new_random(shape.clone());
    let mut parent2 = FixedTopologyGenome::new_random(shape.clone());

    parent1.mutate(0.5).unwrap();
    parent2.mutate(0.3).unwrap();

    let child = parent1.crossover(&parent2).unwrap();

    assert_eq!(child.shape, shape);
}

#[test]
fn test_crossover_then_mutation() {
    let shape = vec![4, 2];
    let parent1 = FixedTopologyGenome::new_random(shape.clone());
    let parent2 = FixedTopologyGenome::new_random(shape.clone());

    let mut child = parent1.crossover(&parent2).unwrap();
    child.mutate(0.2).unwrap();

    assert_eq!(child.shape, shape);
}

#[test]
fn test_deep_network_crossover() {
    let shape = vec![2, 8, 16, 8, 4, 1];
    let parent1 = FixedTopologyGenome::new_random(shape.clone());
    let parent2 = FixedTopologyGenome::new_random(shape.clone());

    let child = parent1.crossover(&parent2).unwrap();

    assert_eq!(child.shape, shape);
    let expected_weights = 308;
    assert_eq!(child.weights.len(), expected_weights);
}

#[test]
fn test_wide_network_mutation() {
    let shape = vec![50, 100];
    let mut genome = FixedTopologyGenome::new_random(shape.clone());

    genome.mutate(0.01).unwrap(); // Low mutation rate on large network

    assert_eq!(genome.shape, shape);
    assert_eq!(genome.weights.len(), 5000);
}
