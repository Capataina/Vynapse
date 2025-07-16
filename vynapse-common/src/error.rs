#[derive(thiserror::Error, Debug)]
pub enum VynapseError {
    #[error("Tensor operation failed: {0}")]
    TensorError(String),

    #[error("Evolution algorithm error: {0}")]
    EvolutionError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

pub type Result<T> = std::result::Result<T, VynapseError>;
