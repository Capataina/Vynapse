use crate::{
    components::mutation::gaussian::GaussianMutation,
    traits::{
        crossover::Crossover, fitness::Fitness, genome::Genome, mutation::Mutation,
        selection::Selection, trainer::TrainingStats,
    },
};

use vynapse_common::*;

pub struct EvolutionaryTrainer<G, F, S, M, C>
where
    G: Genome,
    F: Fitness,
    S: Selection,
    M: Mutation,
    C: Crossover,
{
    population: Vec<G>,
    fitness_function: F,
    selection_strategy: S,
    mutation_operator: M,
    crossover_operator: C,
    population_size: usize,
    max_generations: usize,
    current_generation: usize,
    mutation_rate: f32,
    crossover_rate: f32,
    training_stats: TrainingStats,
}

impl<G, F, S, M, C> EvolutionaryTrainer<G, F, S, M, C>
where
    G: Genome,
    F: Fitness,
    S: Selection,
    M: Mutation,
    C: Crossover,
{
    pub fn new(
        fitness_function: F,
        selection_strategy: S,
        mutation_operator: M,
        crossover_operator: C,
        population_size: usize,
        max_generations: usize,
        mutation_rate: f32,
        crossover_rate: f32,
    ) -> Result<Self> {
        if population_size == 0 {
            return Err(VynapseError::ConfigError(
                "Population size cannot be less than 1.".to_string(),
            ));
        }

        if max_generations == 0 {
            return Err(VynapseError::ConfigError(
                "Max generations cannot be less than 1.".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&mutation_rate) {
            return Err(VynapseError::ConfigError(
                "Mutation rate cannot be outside of the range (0, 1).".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&crossover_rate) {
            return Err(VynapseError::ConfigError(
                "Crossover rate cannot be outside of the range (0, 1).".to_string(),
            ));
        }

        let population = Vec::with_capacity(population_size);
        let current_generation = 0;
        let training_stats = TrainingStats::default();

        Ok(EvolutionaryTrainer {
            population,
            fitness_function,
            selection_strategy,
            mutation_operator,
            crossover_operator,
            population_size,
            max_generations,
            current_generation,
            mutation_rate,
            crossover_rate,
            training_stats,
        })
    }

    pub fn initialise_population(&mut self, template_genome: G) -> Result<()> {
        self.population.clear();
        self.population.reserve_exact(self.population_size);
        self.population.push(template_genome.clone());

        for _ in 0..self.population_size - 1 {
            let mut new_genome = template_genome.clone();
            self.mutation_operator.mutate(&mut new_genome, 1.0)?;
            self.population.push(new_genome);
        }

        if self.population.len() != self.population_size {
            return Err(VynapseError::EvolutionError("An error occured during the population initialisation, the population size doesn't match the number of genomes in the population.".to_string()));
        }

        Ok(())
    }
}
