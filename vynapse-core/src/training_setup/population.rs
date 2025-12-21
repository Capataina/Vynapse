use vynapse_common::*;

use crate::traits::genome::Genome;
use crate::traits::mutation::Mutation;

pub struct Population<G>
where
    G: Genome,
{
    pub genomes: Vec<G>,
    pub population_size: usize,
    pub expected_size: usize,
}

impl<G> Population<G>
where
    G: Genome,
{
    pub fn new(expected_size: usize) -> Result<Self> {
        if expected_size == 0 {
            return Err(VynapseError::ConfigError(
                "Expected size must be greater than zero.".to_string(),
            ));
        }

        let genomes = Vec::with_capacity(expected_size);
        Ok(Population {
            genomes,
            population_size: 0,
            expected_size,
        })
    }

    pub fn init_from_template<M: Mutation>(
        &mut self,
        templates: &G,
        mutation: &M,
        mutation_rate: f32,
    ) -> Result<()> {
        // clear self.genomes
        self.clear();

        //make space for population_size genomes
        self.genomes.reserve(self.expected_size);

        // push one clone of the template genome
        self.genomes.push(templates.clone());

        while self.genomes.len() < self.expected_size {
            let mut new_genome = templates.clone();
            mutation.mutate(&mut new_genome, mutation_rate)?;
            self.genomes.push(new_genome);
        }

        self.population_size = self.genomes.len();

        if self.population_size != self.expected_size {
            return Err(VynapseError::ConfigError(
                "Population size does not match expected size after initialization.".to_string(),
            ));
        }

        Ok(())
    }

    pub fn set_all_genomes(&mut self, genomes: Vec<G>) -> Result<()> {
        if genomes.len() != self.expected_size {
            return Err(VynapseError::ConfigError(
                "The number of genomes provided does not match the expected size.".to_string(),
            ));
        }

        self.genomes = genomes;
        self.population_size = self.genomes.len();

        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        if self.population_size > self.expected_size {
            return Err(VynapseError::ConfigError(
                "Population size cannot be greater than expected size.".to_string(),
            ));
        }

        if self.population_size == 0 || self.expected_size == 0 {
            return Err(VynapseError::ConfigError(
                "Population size and expected size cannot be 0. If they are 0, the population is invalid and no learning can happen.".to_string(),
            ));
        }

        Ok(())
    }

    pub fn get_genomes(&self) -> &[G] {
        &self.genomes
    }

    pub fn get_genomes_mut(&mut self) -> &mut [G] {
        &mut self.genomes
    }

    pub fn get_population_size(&self) -> usize {
        self.population_size
    }

    pub fn get_expected_size(&self) -> usize {
        self.expected_size
    }

    pub fn is_full(&self) -> bool {
        self.population_size == self.expected_size
    }

    pub fn clear(&mut self) {
        self.genomes.clear();
        self.population_size = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::genome::Genome;

    #[derive(Clone, Debug)]
    struct TestGenome {
        data: Vec<f32>,
    }

    impl Genome for TestGenome {
        fn get_weights(&self) -> Vec<f32> {
            self.data.clone()
        }

        fn set_weights(&mut self, weights: Vec<f32>) -> Result<()> {
            self.data = weights;
            Ok(())
        }
    }

    #[test]
    fn test_population_initialization() {
        let expected_size = 5;
        let mut population = Population::<TestGenome>::new(expected_size).unwrap();

        let template_genome = TestGenome {
            data: vec![0.0; 10],
        };

        #[derive(Clone, Debug)]
        struct DummyMutation;

        impl Mutation for DummyMutation {
            fn mutate<G: Genome>(&self, genome: &mut G, rate: f32) -> Result<()> {
                let mut weights = genome.get_weights();
                for w in weights.iter_mut() {
                    *w += rate; // simple mutation: increment each weight by the mutation rate
                }
                genome.set_weights(weights)?;
                Ok(())
            }
        }

        let mutation = DummyMutation;

        population
            .init_from_template(&template_genome, &mutation, 0.1)
            .unwrap();

        assert_eq!(population.get_population_size(), expected_size);
        assert_eq!(population.get_genomes().len(), expected_size);

        // check if the first genome is identical to the template
        assert_eq!(
            population.get_genomes()[0].get_weights(),
            template_genome.get_weights()
        );

        // check if all genomes are initialized
        for genome in population.get_genomes() {
            assert_eq!(
                genome.get_weights().len(),
                template_genome.get_weights().len()
            );
        }

        //check if there is variation among genomes
        let mut all_same = true;
        for genome in population.get_genomes().iter().skip(1) {
            if genome.get_weights() != template_genome.get_weights() {
                all_same = false;
                break;
            }
        }

        assert!(
            !all_same,
            "All genomes are identical to the template, mutation may not have been applied."
        );
    }
}
