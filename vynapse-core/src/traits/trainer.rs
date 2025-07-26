use std::{fmt::Debug, time::Duration};
use vynapse_common::{Result, VynapseError};

#[derive(Debug, Clone, Default)]
pub struct TrainingStats {
    current_iteration: u32,
    best_fitness: f32,
    average_fitness: f32,
    worst_fitness: f32,
    convergence_status: ConvergenceStatus,
    elapsed_time: Duration,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum ConvergenceStatus {
    #[default]
    Running,
    TargetReached,
    Stagnated,
    MaxIterationsReached,
}

impl TrainingStats {
    pub fn new(
        current_iteration: u32,
        best_fitness: f32,
        average_fitness: f32,
        worst_fitness: f32,
        convergence_status: ConvergenceStatus,
        elapsed_time: Duration,
    ) -> Result<Self> {
        if !(best_fitness >= average_fitness && average_fitness >= worst_fitness) {
            return Err(VynapseError::ConfigError("Best fitness must be bigger or equal to average and average should be bigger or equal to worst fitness.".to_string()));
        }

        Ok(TrainingStats {
            current_iteration,
            best_fitness,
            average_fitness,
            worst_fitness,
            convergence_status,
            elapsed_time,
        })
    }

    pub fn increment_iteration(&mut self) -> Result<()> {
        self.current_iteration += 1;
        Ok(())
    }

    pub fn update_fitness(
        &mut self,
        best_fitness: f32,
        average_fitness: f32,
        worst_fitness: f32,
    ) -> Result<()> {
        if !(best_fitness >= average_fitness && average_fitness >= worst_fitness) {
            return Err(VynapseError::ConfigError("Best fitness must be bigger or equal to average and average should be bigger or equal to worst fitness.".to_string()));
        }

        self.best_fitness = best_fitness;
        self.average_fitness = average_fitness;
        self.worst_fitness = worst_fitness;
        Ok(())
    }

    pub fn set_convergence_status(&mut self, status: ConvergenceStatus) -> Result<()> {
        self.convergence_status = status;
        Ok(())
    }

    pub fn update_elapsed_time(&mut self, duration: Duration) -> Result<()> {
        self.elapsed_time = duration;
        Ok(())
    }

    pub fn get_current_iteration(&self) -> u32 {
        self.current_iteration
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

    pub fn get_convergence_status(&self) -> ConvergenceStatus {
        self.convergence_status.clone()
    }

    pub fn get_elapsed_time(&self) -> Duration {
        self.elapsed_time
    }
}

pub trait Trainer: Debug {
    fn train(&mut self) -> Result<TrainingStats>;
    fn step(&mut self) -> Result<TrainingStats>;
    fn get_stats(&self) -> TrainingStats;
    fn reset(&mut self) -> Result<()>;
    fn is_converged(&self) -> bool;
    fn validate_config(&self) -> Result<()>;
}
