use std::fmt::Debug;

use vynapse_common::Result;

pub trait Selection: Clone + Debug {
    fn select(&self, fitness: &[f32], count: usize) -> Result<Vec<usize>>;
}
