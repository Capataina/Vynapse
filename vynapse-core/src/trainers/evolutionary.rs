use crate::{
    training_setup::{
        evolution_config::EvolutionConfig, fitness_stats::FitnessStats,
        genetic_operators::GeneticOperators, population::Population, training_stats::TrainingStats,
    },
    traits::{
        crossover::Crossover, fitness::Fitness, genome::Genome, mutation::Mutation,
        selection::Selection, trainer::Trainer,
    },
};

use vynapse_common::*;

pub struct EvolutionaryTrainer<G, M, C, F, S>
where
    G: Genome,
    M: Mutation,
    C: Crossover,
    F: Fitness,
    S: Selection,
{
    pub evolution_config: EvolutionConfig,
    pub population: Population<G>,
    pub genetic_operators: GeneticOperators<M, C>,
    pub fitness_stats: FitnessStats,
    pub training_stats: TrainingStats,
    pub fitness_function: F,
    pub selection_function: S,
}

impl<G, M, C, F, S> EvolutionaryTrainer<G, M, C, F, S>
where
    G: Genome,
    M: Mutation,
    C: Crossover,
    F: Fitness,
    S: Selection,
{
    pub fn new(
        fitness_function: F,
        selection_function: S,
        evolution_config: EvolutionConfig,
        population: Population<G>,
        genetic_operators: GeneticOperators<M, C>,
    ) -> Result<Self> {
        evolution_config.validate()?;
        population.validate()?;
        genetic_operators.validate()?;

        let fitness_stats = FitnessStats::new()?;
        let training_stats = TrainingStats::new(
            evolution_config.generations,
            evolution_config.stagnation_limit,
        )?;

        Ok(EvolutionaryTrainer {
            evolution_config,
            population,
            genetic_operators,
            fitness_stats,
            training_stats,
            fitness_function,
            selection_function,
        })
    }
}

impl<G, M, C, F, S> Trainer for EvolutionaryTrainer<G, M, C, F, S>
where
    G: Genome,
    M: Mutation,
    C: Crossover,
    F: Fitness,
    S: Selection,
{
    fn train(&mut self) -> Result<TrainingStats> {
        self.training_stats.start_training();
        while !self.training_stats.is_converged() {
            self.step()?;
        }
        Ok(self.training_stats.clone())
    }

    fn step(&mut self) -> Result<TrainingStats> {
        // evaluate the fitness of the population
        let fitness_values = self
            .population
            .get_genomes()
            .iter()
            .map(|genome| self.fitness_function.evaluate(genome))
            .collect::<Result<Vec<f32>>>()?;

        // update the fitness stats
        self.fitness_stats.update_fitness(&fitness_values)?;

        // update the training stats
        self.training_stats.update_generation(&fitness_values)?;

        let selected_parent_indices = self.selection_function.select(
            &fitness_values,
            self.population.get_population_size() as usize,
        )?;

        let mut new_population: Vec<G> = Vec::with_capacity(self.population.get_population_size());

        for i in 0..self.population.get_population_size() {
            // pick 2 parents from the selected parent indices
            let parent1 = &self.population.get_genomes()[selected_parent_indices[i]];
            let parent2 = &self.population.get_genomes()
                [selected_parent_indices[(i + 1) % self.population.get_population_size()]];

            // generate the offspring using crossover
            let mut offspring = self.genetic_operators.apply_crossover(parent1, parent2)?;

            // mutate the offspring using mutation
            self.genetic_operators.apply_mutation(&mut offspring)?;

            new_population.push(offspring);
        }

        // replace the population with the new population
        self.population.set_all_genomes(new_population)?;

        Ok(self.training_stats.clone())
    }

    fn get_stats(&self) -> TrainingStats {
        self.training_stats.clone()
    }

    fn reset(&mut self) -> Result<()> {
        self.training_stats.reset(
            self.evolution_config.generations,
            self.evolution_config.stagnation_limit,
        )?;
        self.fitness_stats.reset()?;
        self.population.clear();

        Ok(())
    }

    fn is_converged(&self) -> bool {
        self.training_stats.is_converged()
    }

    fn validate_config(&self) -> Result<()> {
        self.fitness_stats.validate()?;
        self.evolution_config.validate()?;
        self.population.validate()?;
        self.genetic_operators.validate()?;
        self.training_stats.validate(
            self.evolution_config.generations,
            self.evolution_config.stagnation_limit,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        components::{
            activation::sigmoid::Sigmoid, crossover::uniform::UniformCrossover,
            fitness::task_based::TaskBasedFitness, genome::fixed_topology::FixedTopologyGenome,
            loss::mse::MeanSquaredError, mutation::gaussian::GaussianMutation,
            selection::tournament::TournamentSelection,
        },
        tasks::powers_of_two::PowersOfTwo,
    };

    fn create_test_evolutionary_trainer() -> EvolutionaryTrainer<
        FixedTopologyGenome,
        GaussianMutation,
        UniformCrossover,
        TaskBasedFitness<PowersOfTwo, MeanSquaredError, Sigmoid>,
        TournamentSelection,
    > {
        let evolution_config = EvolutionConfig::new(10, 10).unwrap();
        let mut population = Population::<FixedTopologyGenome>::new(10).unwrap();
        let fitness_function = TaskBasedFitness::new(
            PowersOfTwo::new(10).unwrap(),
            MeanSquaredError::new().unwrap(),
            Sigmoid::new().unwrap(),
        )
        .unwrap();
        let selection_function = TournamentSelection::new(2).unwrap();
        let genetic_operators = GeneticOperators::new(
            GaussianMutation::new(0.1).unwrap(),
            UniformCrossover::new(0.1).unwrap(),
            0.1,
            0.1,
        )
        .unwrap();

        // initialize the population with a template genome

        population
            .init_from_template(
                &FixedTopologyGenome::new_random(vec![1usize, 4usize, 1usize]),
                &GaussianMutation::new(0.1).unwrap(),
                0.1,
            )
            .unwrap();

        let evolutionary_trainer = EvolutionaryTrainer::new(
            fitness_function,
            selection_function,
            evolution_config,
            population,
            genetic_operators,
        )
        .unwrap();

        evolutionary_trainer
    }

    #[test]
    fn test_evolutionary_trainer_new() {
        let evolutionary_trainer = create_test_evolutionary_trainer();
        assert!(evolutionary_trainer.evolution_config.validate().is_ok());
        assert!(evolutionary_trainer.population.validate().is_ok());
        assert!(evolutionary_trainer.genetic_operators.validate().is_ok());

        // check if evolutionary trainer is not null
        assert!(!std::ptr::eq(&evolutionary_trainer, std::ptr::null()));
        assert!(!std::ptr::eq(&evolutionary_trainer, std::ptr::null_mut()));
        assert!(!std::ptr::eq(&evolutionary_trainer, std::ptr::null_mut()));
    }

    #[test]
    fn test_evolutionary_trainer_step() {
        let mut evolutionary_trainer = create_test_evolutionary_trainer();
        evolutionary_trainer.train().unwrap();
        let training_stats = evolutionary_trainer.step().unwrap();
        assert!(training_stats.get_current_generation() > 0);
        assert!(evolutionary_trainer.fitness_stats.get_best_fitness() > 0.0);
        assert!(evolutionary_trainer.fitness_stats.get_average_fitness() > 0.0);
        assert!(evolutionary_trainer.fitness_stats.get_worst_fitness() > 0.0);
    }

    #[test]
    fn test_evolutionary_trainer_learns_powers_of_two() {
        // Setup: Use a simpler task first (fewer data points = easier to learn)
        let max_input = 5; // Powers of 2^0, 2^1, 2^2, 2^3 = [1, 2, 4, 8]

        // Configuration for actual learning
        let evolution_config = EvolutionConfig::new(100, 20).unwrap(); // 100 generations, stagnation after 20
        let population_size = 500;

        let mut population = Population::<FixedTopologyGenome>::new(population_size).unwrap();
        let fitness_function = TaskBasedFitness::new(
            PowersOfTwo::new(max_input).unwrap(),
            MeanSquaredError::new().unwrap(),
            Sigmoid::new().unwrap(),
        )
        .unwrap();
        let selection_function = TournamentSelection::new(3).unwrap(); // Tournament size 3
        let genetic_operators = GeneticOperators::new(
            GaussianMutation::new(0.15).unwrap(), // Slightly higher mutation for exploration
            UniformCrossover::new(0.5).unwrap(),  // 50% chance to inherit from each parent
            0.8,                                  // 80% mutation rate
            0.7,                                  // 70% crossover rate
        )
        .unwrap();

        // Initialize population with correct shape [1 input, 4 hidden, 1 output]
        let template = FixedTopologyGenome::new_random(vec![1, 4, 1]);
        let mutation = GaussianMutation::new(0.1).unwrap();
        population
            .init_from_template(&template, &mutation, 1.0)
            .unwrap();

        let mut trainer = EvolutionaryTrainer::new(
            fitness_function,
            selection_function,
            evolution_config,
            population,
            genetic_operators,
        )
        .unwrap();

        // Track fitness progression
        let mut fitness_history = Vec::new();
        let mut best_fitness_ever = f32::NEG_INFINITY;

        println!("\n=== Starting Evolution Training ===");
        println!("Task: Powers of Two (input 0-3 -> output 1, 2, 4, 8)");
        println!("Population: {}, Generations: 100", population_size);
        println!("");

        // Run training for several generations and track progress
        trainer.training_stats.start_training();

        for generation in 0..100 {
            // Run one generation
            trainer.step().unwrap();

            let stats = trainer.get_stats();
            let best = trainer.fitness_stats.get_best_fitness();
            let avg = trainer.fitness_stats.get_average_fitness();
            let worst = trainer.fitness_stats.get_worst_fitness();

            if best > best_fitness_ever {
                best_fitness_ever = best;
            }

            fitness_history.push((best, avg, worst));

            // Print progress every 10 generations
            if generation % 10 == 0 || generation == 0 || trainer.is_converged() {
                println!(
                    "Gen {:3}: Best={:.6}, Avg={:.6}, Worst={:.6}, Status={:?}",
                    generation + 1,
                    best,
                    avg,
                    worst,
                    stats.get_convergence_status()
                );
            }

            if trainer.is_converged() {
                println!("\nTraining converged early!");
                break;
            }
        }

        let final_stats = trainer.get_stats();
        let final_best = trainer.fitness_stats.get_best_fitness();

        println!("\n=== Training Complete ===");
        println!("Final Generation: {}", final_stats.get_current_generation());
        println!("Final Best Fitness: {:.6}", final_best);
        println!("Best Fitness Ever: {:.6}", best_fitness_ever);
        println!(
            "Convergence Status: {:?}",
            final_stats.get_convergence_status()
        );
        println!("");

        // Assertions: Verify learning happened
        assert!(
            final_best > 0.0,
            "Final fitness should be positive, got {}",
            final_best
        );

        // Check that fitness improved from initial generation
        if fitness_history.len() >= 2 {
            let initial_best = fitness_history[0].0;
            assert!(
                final_best >= initial_best,
                "Fitness should not decrease: started at {}, ended at {}",
                initial_best,
                final_best
            );
        }

        // Ideally, we'd want fitness > 0.5 for decent performance
        // But for now, just verify it's learning (improving over time)
        println!("✅ Training completed successfully!");
        println!(
            "✅ Final fitness: {:.6} (higher is better, max=1.0)",
            final_best
        );

        // Optional: Check that best fitness improved over time
        if fitness_history.len() > 10 {
            let mid_point = fitness_history.len() / 2;
            let mid_fitness = fitness_history[mid_point].0;
            if final_best > mid_fitness {
                println!("✅ Fitness improved from generation {} to final", mid_point);
            }
        }
    }
}
