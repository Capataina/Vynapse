use vynapse_common::{Result, VynapseError};

#[derive(Debug, Clone)]
pub struct FitnessStats {
    best_fitness: f32,
    average_fitness: f32,
    worst_fitness: f32,
    fitness_history: Vec<f32>,
    current_generation_fitness: Vec<f32>,
}

impl FitnessStats {
    pub fn new() -> Result<Self> {
        let best_fitness = f32::NEG_INFINITY;
        let average_fitness = f32::NEG_INFINITY;
        let worst_fitness = f32::NEG_INFINITY;
        let fitness_history: Vec<f32> = Vec::new();
        let current_generation_fitness: Vec<f32> = Vec::new();
        Ok(FitnessStats {
            best_fitness,
            average_fitness,
            worst_fitness,
            fitness_history,
            current_generation_fitness,
        })
    }

    pub fn update_fitness(&mut self, fitness_values: &[f32]) -> Result<()> {
        if fitness_values.is_empty() {
            return Err(VynapseError::EvolutionError(
                "Couldn't update fitness stats, the given fitness array is empty.".to_string(),
            ));
        }

        self.current_generation_fitness.clear();

        let best = self.calculate_best(fitness_values)?;
        let worst = self.calculate_worst(fitness_values)?;
        let average = self.calculate_average(fitness_values)?;

        self.best_fitness = best;
        self.average_fitness = average;
        self.worst_fitness = worst;
        self.fitness_history.push(best);

        self.current_generation_fitness
            .extend_from_slice(fitness_values);

        Ok(())
    }

    pub fn calculate_best(&self, values: &[f32]) -> Result<f32> {
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        Ok(max)
    }

    pub fn calculate_worst(&self, values: &[f32]) -> Result<f32> {
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        Ok(min)
    }

    pub fn calculate_average(&self, values: &[f32]) -> Result<f32> {
        let sum: f32 = values.iter().sum();
        let avg = sum / values.len() as f32;
        Ok(avg)
    }

    pub fn get_best_fitness(&self) -> f32 {
        self.best_fitness
    }

    pub fn get_average_fitness(&self) -> f32 {
        self.average_fitness
    }

    pub fn get_worst_fitness(&self) -> f32 {
        self.worst_fitness
    }

    pub fn get_fitness_history(&self) -> &[f32] {
        &self.fitness_history
    }

    pub fn get_current_generation_fitness(&self) -> &[f32] {
        &self.current_generation_fitness
    }
}
