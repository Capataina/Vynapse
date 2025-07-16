use vynapse_common::{Result, VynapseError};

use crate::Shape;

#[derive(Debug, Clone)]
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

    pub fn get(&self, indices: &[usize]) -> Result<&T> {
        if self.shape.rank() != indices.len() {
            return Err(VynapseError::TensorError(
                "The indices do not match the shape of the tensor.".to_string(),
            ));
        }

        let mut flat_index = 0;

        for (i, &index) in indices.iter().enumerate() {
            if index >= self.shape.dims[i] {
                return Err(VynapseError::TensorError(format!(
                    "Index {} is out of bounds for dimension {} (size: {})",
                    index, i, self.shape.dims[i]
                )));
            }

            flat_index += index * self.strides[i];
        }

        Ok(&self.data[flat_index])
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> Result<&mut T> {
        if self.shape.rank() != indices.len() {
            return Err(VynapseError::TensorError(
                "The indices do not match the shape of the tensor.".to_string(),
            ));
        }

        let mut flat_index = 0;

        for (i, &index) in indices.iter().enumerate() {
            if index >= self.shape.dims[i] {
                return Err(VynapseError::TensorError(format!(
                    "Index {} is out of bounds for dimension {} (size: {})",
                    index, i, self.shape.dims[i]
                )));
            }

            flat_index += index * self.strides[i];
        }

        Ok(&mut self.data[flat_index])
    }

    pub fn set(&mut self, indices: &[usize], value: T) -> Result<()> {
        if self.shape.rank() != indices.len() {
            return Err(VynapseError::TensorError(
                "The indices do not match the shape of the tensor.".to_string(),
            ));
        }

        let mut flat_index = 0;

        for (i, &index) in indices.iter().enumerate() {
            if index >= self.shape.dims[i] {
                return Err(VynapseError::TensorError(format!(
                    "Index {} is out of bounds for dimension {} (size: {})",
                    index, i, self.shape.dims[i]
                )));
            }

            flat_index += index * self.strides[i];
        }

        self.data[flat_index] = value;
        Ok(())
    }

    pub fn zeros(shape: Shape) -> Result<Self>
    where
        T: Default + Clone + num_traits::Zero,
    {
        let total_elements = shape.total_elements();
        let strides = Self::calculate_strides(&shape);
        let data = vec![T::zero(); total_elements];

        Ok(Tensor {
            data,
            shape,
            strides,
        })
    }

    pub fn ones(shape: Shape) -> Result<Self>
    where
        T: Default + Clone + num_traits::One,
    {
        let total_elements = shape.total_elements();
        let strides = Self::calculate_strides(&shape);
        let data = vec![T::one(); total_elements];

        Ok(Tensor {
            data,
            shape,
            strides,
        })
    }

    pub fn from_vec(data: Vec<T>, shape: Shape) -> Result<Self>
    where
        T: Default + Clone,
    {
        if data.len() != shape.total_elements() {
            return Err(VynapseError::TensorError(format!(
                "Entered data length: {} doesn't fit the given shape: {}.",
                data.len(),
                shape.total_elements(),
            )));
        }

        let strides = Self::calculate_strides(&shape);

        Ok(Tensor {
            data,
            shape,
            strides,
        })
    }

    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.data.fill(value);
    }
}

impl Tensor<f32> {
    pub fn apply_sigmoid(&self) -> Result<Tensor<f32>> {
        let mut new_tensor = self.clone();
        for x in &mut new_tensor.data {
            *x = 1.0 / (1.0 + (-*x).exp());
        }
        Ok(new_tensor)
    }
}
