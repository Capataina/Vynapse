use std::fmt::Debug;

use crate::Result;

pub trait Selection: Clone + Debug {
    fn select(&self, fitness: &[f32], count: usize) -> Result<Vec<usize>>;
}
