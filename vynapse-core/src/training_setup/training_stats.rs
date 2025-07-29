use std::time::{Duration, Instant};
use vynapse_common::*;

use crate::training_setup::fitness_stats::FitnessStats;

#[derive(Debug, Clone)]
pub struct TrainingStats {
    fitness_stats: FitnessStats,
    current_generation: u32,
    max_generations: u32,
    convergence_status: ConvergenceStatus,
    starting_time: Option<Instant>,
    elapsed_time: Duration,
    stagnation_counter: u32,
    stagnation_limit: u32,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum ConvergenceStatus {
    #[default]
    WaitingToStart,
    Running,
    TargetReached,
    Stagnated,
    MaxIterationsReached,
}

impl TrainingStats {
    pub fn new(max_generations: u32, stagnation_limit: u32) -> Result<Self> {
        let fitness_stats = FitnessStats::new()?;
        let convergence_status = ConvergenceStatus::default();
        let current_generation = 0;
        let starting_time = None;
        let elapsed_time = Duration::from_secs(0);
        let stagnation_counter = 0;

        Ok(TrainingStats {
            fitness_stats,
            current_generation,
            max_generations,
            convergence_status,
            starting_time,
            elapsed_time,
            stagnation_counter,
            stagnation_limit,
        })
    }

    pub fn update_generation(&mut self, fitness_values: &[f32]) -> Result<()> {
        if self.convergence_status == ConvergenceStatus::WaitingToStart {
            return Err(VynapseError::EvolutionError(
                "Training must be started before updating generations".to_string(),
            ));
        }

        self.fitness_stats.update_fitness(fitness_values)?;
        self.current_generation += 1;
        self.update_elapsed_time()?;

        if !self.is_last_fitness_best(self.fitness_stats.get_fitness_history()) {
            self.stagnation_counter += 1;
        } else {
            self.stagnation_counter = 0;
        }

        if self.stagnation_counter >= self.stagnation_limit {
            self.convergence_status = ConvergenceStatus::Stagnated;
        }

        if self.current_generation >= self.max_generations {
            self.convergence_status = ConvergenceStatus::MaxIterationsReached;
        }

        Ok(())
    }

    pub fn start_training(&mut self) {
        self.starting_time = Some(Instant::now());
        self.elapsed_time = Duration::from_secs(0);
        self.current_generation = 0;
        self.convergence_status = ConvergenceStatus::Running;
    }

    pub fn update_elapsed_time(&mut self) -> Result<()> {
        if self.starting_time.is_none() {
            return Err(VynapseError::EvolutionError(
                "Training hasn't started yet, so cannot update elapsed time.".to_string(),
            ));
        }

        self.elapsed_time = Instant::now().duration_since(self.starting_time.unwrap());
        Ok(())
    }

    pub fn is_last_fitness_best(&self, fitness_values: &[f32]) -> bool {
        if fitness_values.is_empty() {
            return false;
        }

        if fitness_values.len() < 2 {
            return true;
        }

        let last = *fitness_values.last().unwrap();
        let max_of_rest = fitness_values[..fitness_values.len() - 1]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        last > max_of_rest
    }

    pub fn get_best_fitness(&self) -> f32 {
        self.fitness_stats.get_best_fitness()
    }

    pub fn get_average_fitness(&self) -> f32 {
        self.fitness_stats.get_average_fitness()
    }

    pub fn get_worst_fitness(&self) -> f32 {
        self.fitness_stats.get_worst_fitness()
    }

    pub fn get_current_generation(&self) -> u32 {
        self.current_generation
    }

    pub fn get_convergence_status(&self) -> &ConvergenceStatus {
        &self.convergence_status
    }

    pub fn get_elapsed_time(&self) -> Duration {
        self.elapsed_time
    }

    pub fn is_converged(&self) -> bool {
        self.convergence_status != ConvergenceStatus::Running
    }
}
