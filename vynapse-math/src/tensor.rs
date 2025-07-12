use vynapse_core::Result;

use crate::Shape;

pub struct Tensor<T> {
    pub data: Vec<T>,
    pub shape: Shape,
    pub strides: Vec<usize>,
}

impl<T> Tensor<T> {
    pub fn new(shape: Shape) -> Result<Self>
    where
        T: Default + Clone,
    {
        let total_elements = shape.total_elements();
        let strides = Self::calculate_strides(&shape);
        let data = vec![T::default(); total_elements];

        Ok(Tensor {
            data,
            shape,
            strides,
        })
    }

    pub fn calculate_strides(shape: &Shape) -> Vec<usize> {
        let mut strides: Vec<usize> = Vec::new();

        for i in 0..shape.rank() {
            let stride = shape.dims[i + 1..].iter().product();
            strides.push(stride);
        }

        strides
    }
}
