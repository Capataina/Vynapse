use std::fmt::Debug;

pub trait Selection: Clone + Debug {
    fn select(&self, fitness: &[f32], count: usize) -> Vec<usize>;
}
