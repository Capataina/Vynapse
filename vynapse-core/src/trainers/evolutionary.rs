use crate::traits::{fitness::Fitness, genome::Genome, selection::Selection};

pub struct EvolutionaryTrainer<G, F, S>
where
    G: Genome,
    F: Fitness,
    S: Selection,
{
    population: Vec<G>,
    fitness_function: F,
    selection_strategy: S,
    population_size: usize,
    max_generations: usize,
    mutation_rate: usize,
    crossover_rate: usize,
}
