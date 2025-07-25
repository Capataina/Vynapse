use rand::{Rng, rng};
use rand_distr::{Distribution, Normal};
use vynapse_common::{Result, VynapseError};

use crate::traits::{genome::Genome, mutation::Mutation};

#[derive(Clone, Debug)]
pub struct GaussianMutation {
    sigma: f32,
}

impl GaussianMutation {
    pub fn new(sigma: f32) -> Result<Self> {
        if sigma > 0.0 {
            Ok(GaussianMutation { sigma })
        } else {
            Err(VynapseError::EvolutionError(
                "Sigma cannot be smaller than 0.".to_string(),
            ))
        }
    }
}

impl Mutation for GaussianMutation {
    fn mutate<G: Genome>(&self, genome: &mut G, rate: f32) -> Result<()> {
        if !(0.0..=1.0).contains(&rate) {
            return Err(VynapseError::EvolutionError(
                "Mutation rate cannot be outside of the range (0, 1).".to_string(),
            ));
        }

        let normal = Normal::new(0.0, self.sigma).map_err(|_| {
            VynapseError::EvolutionError(
                "Invalid sigma value for Gaussian distribution".to_string(),
            )
        })?;

        let genome_weights = genome.get_weights();
        let mut mutated_weights: Vec<f32> = Vec::with_capacity(genome_weights.len());

        let mut rng = rng();

        for weight in genome_weights {
            if rng.random_bool(rate as f64) {
                let noise: f32 = normal.sample(&mut rng);

                let net_weight = (weight + noise).clamp(-5.0, 5.0);

                mutated_weights.push(net_weight);
            } else {
                mutated_weights.push(weight);
            }
        }

        genome.set_weights(mutated_weights)?;

        Ok(())
    }
}
