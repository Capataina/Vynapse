use vynapse_common::*;

pub struct EvolutionConfig {
    pub generations: u32,
    pub stagnation_limit: u32,
}

impl EvolutionConfig {
    pub fn new(generations: u32, stagnation_limit: u32) -> Result<Self> {
        if generations <= 0 {
            return Err(VynapseError::ConfigError(
                "Generations cannot be less than 1.".to_string(),
            ));
        }

        if stagnation_limit <= 0 {
            return Err(VynapseError::ConfigError(
                "Stagnation limit cannot be less than 1.".to_string(),
            ));
        }

        Ok(EvolutionConfig {
            generations,
            stagnation_limit,
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.generations <= 0 {
            return Err(VynapseError::ConfigError(
                "Generations cannot be less than 1.".to_string(),
            ));
        }

        if self.stagnation_limit <= 0 {
            return Err(VynapseError::ConfigError(
                "Stagnation limit cannot be less than 1.".to_string(),
            ));
        }

        Ok(())
    }
}
