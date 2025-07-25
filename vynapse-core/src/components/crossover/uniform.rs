use rand::{Rng, rng};
use vynapse_common::{Result, VynapseError};

use crate::traits::crossover::Crossover;

#[derive(Debug, Clone)]
pub struct UniformCrossover {
    inheritance_probability: f32,
}

impl UniformCrossover {
    pub fn new(inheritance_probability: f32) -> Result<Self> {
        if (0.0..1.0).contains(&inheritance_probability) {
            Ok(UniformCrossover {
                inheritance_probability,
            })
        } else {
            Err(VynapseError::EvolutionError(
                "Inheritance probability has to be within the range (0, 1).".to_string(),
            ))
        }
    }
}

impl Crossover for UniformCrossover {
    fn crossover<G: crate::traits::genome::Genome>(&self, parent1: &G, parent2: &G) -> Result<G> {
        let mut child = parent1.clone();
        let parent1_weights = parent1.get_weights();
        let parent2_weights = parent2.get_weights();

        if parent1_weights.len() != parent2_weights.len() {
            return Err(VynapseError::EvolutionError(
                "Parent's weights have to be the same length for crossovers.".to_string(),
            ));
        }

        let mut crossover_weights: Vec<f32> = Vec::with_capacity(parent1_weights.len());

        let mut rng = rng();

        for (weight1, weight2) in parent1_weights.iter().zip(parent2_weights.iter()) {
            if rng.random_bool(self.inheritance_probability as f64) {
                crossover_weights.push(*weight1);
            } else {
                crossover_weights.push(*weight2);
            }
        }

        child.set_weights(crossover_weights)?;

        Ok(child)
    }
}
