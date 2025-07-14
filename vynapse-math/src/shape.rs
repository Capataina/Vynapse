use vynapse_core::{Result, VynapseError};

use std::fmt;

#[derive(Debug, Clone)]
pub struct Shape {
    pub dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Result<Self> {
        for &num in &dims {
            if num == 0 {
                return Err(VynapseError::TensorError(
                    "A dimension cannot be 0.".to_string(),
                ));
            }
        }

        Ok(Shape { dims })
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn total_elements(&self) -> usize {
        self.dims.iter().product()
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}
