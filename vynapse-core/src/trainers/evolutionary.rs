use crate::traits::{
    crossover::Crossover, fitness::Fitness, genome::Genome, mutation::Mutation,
    selection::Selection,
};

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
    mutation_rate: usize,
    crossover_rate: usize,
}
