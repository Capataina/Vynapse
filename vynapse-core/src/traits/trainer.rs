use vynapse_common::Result;

use crate::training_setup::training_stats::TrainingStats;

pub trait Trainer {
    fn train(&mut self) -> Result<TrainingStats>;
    fn step(&mut self) -> Result<TrainingStats>;
    fn get_stats(&self) -> TrainingStats;
    fn reset(&mut self) -> Result<()>;
    fn is_converged(&self) -> bool;
    fn validate_config(&self) -> Result<()>;
}
