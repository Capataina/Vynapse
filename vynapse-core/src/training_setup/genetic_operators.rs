use rand::Rng;
use vynapse_common::*;

use crate::traits::crossover::Crossover;
use crate::traits::genome::Genome;
use crate::traits::mutation::Mutation;

pub struct GeneticOperators<M, C>
where
    M: Mutation,
    C: Crossover,
{
    mutation: M,
    crossover: C,
    mutation_rate: f32,
    crossover_rate: f32,
}

impl<M, C> GeneticOperators<M, C>
where
    M: Mutation,
    C: Crossover,
{
    pub fn new(mutation: M, crossover: C, mutation_rate: f32, crossover_rate: f32) -> Result<Self> {
        if mutation_rate < 0.0 || mutation_rate > 1.0 {
            return Err(VynapseError::ConfigError(
                "Mutation rate must be between 0.0 and 1.0.".to_string(),
            ));
        }

        if crossover_rate < 0.0 || crossover_rate > 1.0 {
            return Err(VynapseError::ConfigError(
                "Crossover rate must be between 0.0 and 1.0.".to_string(),
            ));
        }

        Ok(GeneticOperators {
            mutation,
            crossover,
            mutation_rate,
            crossover_rate,
        })
    }

    pub fn apply_mutation<G: Genome>(&self, genome: &mut G) -> Result<()> {
        self.mutation.mutate(genome, self.mutation_rate)
    }

    pub fn apply_crossover<G: Genome>(&self, parent1: &G, parent2: &G) -> Result<G> {
        // pick a random number from 0 to 1
        let mut rng = rand::rng();
        let random_value: f32 = rng.random();

        if random_value > self.crossover_rate {
            Ok(parent1.clone())
        } else {
            self.crossover.crossover(parent1, parent2)
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.mutation_rate) {
            return Err(VynapseError::ConfigError(
                "Mutation rate must be between 0.0 and 1.0.".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&self.crossover_rate) {
            return Err(VynapseError::ConfigError(
                "Crossover rate must be between 0.0 and 1.0.".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::genome::Genome;

    #[derive(Debug, Clone)]
    struct TestGenome {
        weights: Vec<f32>,
    }

    impl Genome for TestGenome {
        fn get_weights(&self) -> Vec<f32> {
            self.weights.clone()
        }

        fn set_weights(&mut self, weights: Vec<f32>) -> Result<()> {
            self.weights = weights;
            Ok(())
        }
    }

    #[derive(Debug, Clone)]
    struct TestMutation;

    impl Mutation for TestMutation {
        fn mutate<G: Genome>(&self, genome: &mut G, rate: f32) -> Result<()> {
            let mut weights = genome.get_weights();
            for w in weights.iter_mut() {
                *w += rate; // simple deterministic mutation
            }
            genome.set_weights(weights)?;
            Ok(())
        }
    }

    #[derive(Debug, Clone)]
    struct TestCrossover;

    impl Crossover for TestCrossover {
        fn crossover<G: Genome>(&self, parent1: &G, parent2: &G) -> Result<G> {
            let mut child = parent1.clone();
            let parent1_weights = parent1.get_weights();
            let parent2_weights = parent2.get_weights();

            if parent1_weights.len() != parent2_weights.len() {
                return Err(VynapseError::EvolutionError(
                    "Parent's weights have to be the same length for crossovers.".to_string(),
                ));
            }

            let crossover_weights: Vec<f32> = parent1_weights
                .iter()
                .zip(parent2_weights.iter())
                .map(|(w1, w2)| (w1 + w2) / 2.0) // average for determinism
                .collect();

            child.set_weights(crossover_weights)?;
            Ok(child)
        }
    }

    #[test]
    fn test_genetic_operators_apply_both() {
        let mutation = TestMutation;
        let crossover = TestCrossover;
        // Force crossover to always occur
        let ops = GeneticOperators::new(mutation, crossover, 0.1, 1.0).unwrap();

        let mut genome = TestGenome {
            weights: vec![1.0, 2.0, 3.0],
        };
        ops.apply_mutation(&mut genome).unwrap();
        assert_eq!(genome.get_weights(), vec![1.1, 2.1, 3.1]);

        let parent1 = TestGenome {
            weights: vec![1.0, 2.0, 3.0],
        };
        let parent2 = TestGenome {
            weights: vec![4.0, 5.0, 6.0],
        };
        let child = ops.apply_crossover(&parent1, &parent2).unwrap();
        assert_eq!(child.get_weights(), vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_invalid_rates_rejected() {
        let mutation = TestMutation;
        let crossover = TestCrossover;

        assert!(GeneticOperators::new(mutation.clone(), crossover.clone(), -0.1, 0.5).is_err());
        assert!(GeneticOperators::new(mutation.clone(), crossover.clone(), 0.5, 1.5).is_err());
    }

    #[test]
    fn test_crossover_skipped_when_rate_zero() {
        let mutation = TestMutation;
        let crossover = TestCrossover;
        let ops = GeneticOperators::new(mutation, crossover, 0.1, 0.0).unwrap(); // never crossover

        let parent1 = TestGenome {
            weights: vec![1.0, 2.0, 3.0],
        };
        let parent2 = TestGenome {
            weights: vec![4.0, 5.0, 6.0],
        };
        let child = ops.apply_crossover(&parent1, &parent2).unwrap();
        assert_eq!(child.get_weights(), parent1.get_weights()); // cloned parent1
    }

    #[test]
    fn test_mutation_uses_rate() {
        let mutation = TestMutation;
        let crossover = TestCrossover;
        let ops = GeneticOperators::new(mutation, crossover, 0.5, 0.0).unwrap();

        let mut genome = TestGenome {
            weights: vec![0.0, 0.0, 0.0],
        };
        ops.apply_mutation(&mut genome).unwrap();
        assert_eq!(genome.get_weights(), vec![0.5, 0.5, 0.5]);
    }
}
