use std::ops::{Add, Mul};

use vynapse_core::Result;

use crate::{Shape, Tensor};

pub fn matrix_vector_mult<T>(matrix: &Tensor<T>, vector: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Default + Clone + Mul<T, Output = T> + Add<T, Output = T>,
{
    if matrix.shape.rank() != 2 {
        return Err(vynapse_core::VynapseError::TensorError(
            "The first parameter is not a matrix as it's tensor shape isn't 2.".to_string(),
        ));
    }

    if vector.shape.rank() != 1 {
        return Err(vynapse_core::VynapseError::TensorError(
            "The second parameter is not a vector as it's tensor shape isn't 1.".to_string(),
        ));
    }

    if matrix.shape.dims[1] != vector.data.len() {
        return Err(vynapse_core::VynapseError::TensorError(
            "The matrix columns do not match the length of the vector.".to_string(),
        ));
    }

    let mut new_vector: Vec<T> = Vec::new();

    for row in 0..matrix.shape.dims[0] {
        let mut num_in_vector = T::default();

        for col in 0..matrix.shape.dims[1] {
            let matrix_element = matrix.get(&[row, col])?.clone();
            let vector_element = vector.data[col].clone();

            num_in_vector = num_in_vector + matrix_element * vector_element;
        }

        new_vector.push(num_in_vector);
    }

    let new_vector_shape = Shape::new(vec![matrix.shape.dims[0]])?;

    Tensor::from_vec(new_vector, new_vector_shape)
}

#[cfg(test)]
#[test]
fn simple_2x2_matrix() {
    let matrix = vec![1.0, 2.0, 3.0, 4.0];
    let matrix_shape = Shape::new(vec![2, 2]).unwrap();
    let matrix_tensor = Tensor::from_vec(matrix, matrix_shape).unwrap();

    let vector = vec![5.0, 6.0];
    let vector_shape = Shape::new(vec![2]).unwrap();
    let vector_tensor = Tensor::from_vec(vector, vector_shape).unwrap();

    let product = matrix_vector_mult(&matrix_tensor, &vector_tensor).unwrap();
    assert_eq!(product.data, [17.0, 39.0]);
    assert_eq!(product.shape.rank(), 1);
    assert_eq!(product.data.len(), 2);
}

#[test]
fn simple_2x3_matrix() {
    let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let matrix_shape = Shape::new(vec![2, 3]).unwrap();
    let matrix_tensor = Tensor::from_vec(matrix, matrix_shape).unwrap();

    let vector = vec![7.0, 8.0, 9.0];
    let vector_shape = Shape::new(vec![3]).unwrap();
    let vector_tensor = Tensor::from_vec(vector, vector_shape).unwrap();

    let product = matrix_vector_mult(&matrix_tensor, &vector_tensor).unwrap();
    assert_eq!(product.data, [50.0, 122.0]);
    assert_eq!(product.shape.rank(), 1);
    assert_eq!(product.data.len(), 2);
}

#[test]
fn shape_mismatch_error() {
    let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let matrix_shape = Shape::new(vec![2, 3]).unwrap();
    let matrix_tensor = Tensor::from_vec(matrix, matrix_shape).unwrap();

    let vector = vec![7.0, 8.0];
    let vector_shape = Shape::new(vec![2]).unwrap();
    let vector_tensor = Tensor::from_vec(vector, vector_shape).unwrap();

    let product = matrix_vector_mult(&matrix_tensor, &vector_tensor);
    assert!(product.is_err());
}
