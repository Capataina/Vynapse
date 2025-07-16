use num_traits::Zero;
use vynapse_common::{Result, VynapseError};

use crate::{Shape, Tensor};

pub fn transpose_2d<T>(matrix: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Default + Zero + Clone,
{
    if matrix.shape.rank() != 2 {
        return Err(VynapseError::TensorError(
            "transpose_2d requires a tensor with rank 2 (matrix)".to_string(),
        ));
    }

    let matrix_rows = matrix.shape.dims[0];
    let matrix_cols = matrix.shape.dims[1];
    let transposed_tensor_shape = Shape::new(vec![matrix_cols, matrix_rows])?;
    let mut transposed_tensor: Tensor<T> = Tensor::new(transposed_tensor_shape)?;

    for row in 0..matrix.shape.dims[0] {
        for col in 0..matrix.shape.dims[1] {
            let current_num = matrix.get(&[row, col])?.clone();
            transposed_tensor.set(&[col, row], current_num)?;
        }
    }

    Ok(transposed_tensor)
}

pub fn reshape_tensor<T>(tensor: &Tensor<T>, new_shape: Shape) -> Result<Tensor<T>>
where
    T: Default + Clone,
{
    if tensor.shape.total_elements() != new_shape.total_elements() {
        return Err(VynapseError::TensorError(
            "Tensor's total elements do not match the new shape.".to_string(),
        ));
    }

    Tensor::from_vec(tensor.data.clone(), new_shape)
}

#[cfg(test)]
#[test]
fn transpose_2x2_square_matrix() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = Shape::new(vec![2, 2]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let transposed = transpose_2d(&tensor).unwrap();

    assert_eq!(transposed.shape.dims, vec![2, 2]);
    assert_eq!(transposed.data, vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn transpose_2x3_rectangle_matrix() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = Shape::new(vec![2, 3]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let transposed = transpose_2d(&tensor).unwrap();

    assert_eq!(transposed.shape.dims, vec![3, 2]);
    assert_eq!(transposed.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn transpose_3x2_rectangle_matrix() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = Shape::new(vec![3, 2]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let transposed = transpose_2d(&tensor).unwrap();

    assert_eq!(transposed.shape.dims, vec![2, 3]);
    assert_eq!(transposed.data, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
}

#[test]
fn transpose_1x4_row_vector() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = Shape::new(vec![1, 4]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let transposed = transpose_2d(&tensor).unwrap();

    assert_eq!(transposed.shape.dims, vec![4, 1]);
    assert_eq!(transposed.data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn transpose_4x1_column_vector() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = Shape::new(vec![4, 1]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let transposed = transpose_2d(&tensor).unwrap();

    assert_eq!(transposed.shape.dims, vec![1, 4]);
    assert_eq!(transposed.data, vec![1.0, 2.0, 3.0, 4.0]);
}

// ============ EDGE CASES ============
#[test]
fn transpose_1x1_single_element() {
    let data = vec![42.0];
    let shape = Shape::new(vec![1, 1]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let transposed = transpose_2d(&tensor).unwrap();

    assert_eq!(transposed.shape.dims, vec![1, 1]);
    assert_eq!(transposed.data, vec![42.0]);
}

#[test]
fn transpose_with_negative_numbers() {
    let data = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0];
    let shape = Shape::new(vec![2, 3]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let transposed = transpose_2d(&tensor).unwrap();

    assert_eq!(transposed.shape.dims, vec![3, 2]);
    assert_eq!(transposed.data, vec![-1.0, 4.0, 2.0, -5.0, -3.0, 6.0]);
}

#[test]
fn transpose_with_zeros() {
    let data = vec![0.0, 1.0, 2.0, 0.0, 3.0, 0.0];
    let shape = Shape::new(vec![2, 3]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let transposed = transpose_2d(&tensor).unwrap();

    assert_eq!(transposed.shape.dims, vec![3, 2]);
    assert_eq!(transposed.data, vec![0.0, 0.0, 1.0, 3.0, 2.0, 0.0]);
}

#[test]
fn transpose_with_fractional_numbers() {
    let data = vec![1.5, 2.25, 3.75, 4.125];
    let shape = Shape::new(vec![2, 2]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let transposed = transpose_2d(&tensor).unwrap();

    assert_eq!(transposed.shape.dims, vec![2, 2]);
    assert_eq!(transposed.data, vec![1.5, 3.75, 2.25, 4.125]);
}

// ============ DOUBLE TRANSPOSE (IDENTITY) ============
#[test]
fn double_transpose_identity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = Shape::new(vec![2, 3]).unwrap();
    let original = Tensor::from_vec(data, shape).unwrap();

    let transposed = transpose_2d(&original).unwrap();
    let double_transposed = transpose_2d(&transposed).unwrap();

    assert_eq!(double_transposed.shape.dims, original.shape.dims);
    assert_eq!(double_transposed.data, original.data);
}

// ============ ERROR CASES ============
#[test]
fn transpose_1d_tensor_error() {
    let data = vec![1.0, 2.0, 3.0];
    let shape = Shape::new(vec![3]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let result = transpose_2d(&tensor);
    assert!(result.is_err());

    if let Err(VynapseError::TensorError(msg)) = result {
        assert!(msg.contains("rank") || msg.contains("2") || msg.contains("dimension"));
    } else {
        panic!("Expected TensorError for 1D tensor");
    }
}

#[test]
fn transpose_3d_tensor_error() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let shape = Shape::new(vec![2, 2, 2]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let result = transpose_2d(&tensor);
    assert!(result.is_err());

    if let Err(VynapseError::TensorError(msg)) = result {
        assert!(msg.contains("rank") || msg.contains("2") || msg.contains("dimension"));
    } else {
        panic!("Expected TensorError for 3D tensor");
    }
}

// ============ LARGE MATRIX TEST ============
#[test]
fn transpose_larger_matrix() {
    let mut data = Vec::new();
    for i in 0..20 {
        data.push(i as f32);
    }
    let shape = Shape::new(vec![4, 5]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let transposed = transpose_2d(&tensor).unwrap();

    assert_eq!(transposed.shape.dims, vec![5, 4]);

    // Verify a few key elements
    assert_eq!(transposed.get(&[0, 0]).unwrap(), &0.0); // (0,0) -> (0,0)
    assert_eq!(transposed.get(&[0, 1]).unwrap(), &5.0); // (1,0) -> (0,1)
    assert_eq!(transposed.get(&[1, 0]).unwrap(), &1.0); // (0,1) -> (1,0)
    assert_eq!(transposed.get(&[4, 3]).unwrap(), &19.0); // (3,4) -> (4,3)
}

// ============ INTEGER TYPE TEST ============
#[test]
fn transpose_integer_matrix() {
    let data = vec![1, 2, 3, 4, 5, 6];
    let shape = Shape::new(vec![2, 3]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let transposed = transpose_2d(&tensor).unwrap();

    assert_eq!(transposed.shape.dims, vec![3, 2]);
    assert_eq!(transposed.data, vec![1, 4, 2, 5, 3, 6]);
}

#[test]
fn reshape_1d_to_2d() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = Shape::new(vec![6]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let new_shape = Shape::new(vec![2, 3]).unwrap();
    let reshaped = reshape_tensor(&tensor, new_shape).unwrap();

    assert_eq!(reshaped.shape.dims, vec![2, 3]);
    assert_eq!(reshaped.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Verify we can access elements correctly
    assert_eq!(reshaped.get(&[0, 0]).unwrap(), &1.0);
    assert_eq!(reshaped.get(&[0, 2]).unwrap(), &3.0);
    assert_eq!(reshaped.get(&[1, 0]).unwrap(), &4.0);
    assert_eq!(reshaped.get(&[1, 2]).unwrap(), &6.0);
}

#[test]
fn reshape_2d_to_1d() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = Shape::new(vec![2, 3]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let new_shape = Shape::new(vec![6]).unwrap();
    let reshaped = reshape_tensor(&tensor, new_shape).unwrap();

    assert_eq!(reshaped.shape.dims, vec![6]);
    assert_eq!(reshaped.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn reshape_2d_to_2d_different_dimensions() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let shape = Shape::new(vec![2, 4]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let new_shape = Shape::new(vec![4, 2]).unwrap();
    let reshaped = reshape_tensor(&tensor, new_shape).unwrap();

    assert_eq!(reshaped.shape.dims, vec![4, 2]);
    assert_eq!(reshaped.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    // Verify accessing works with new shape
    assert_eq!(reshaped.get(&[0, 0]).unwrap(), &1.0);
    assert_eq!(reshaped.get(&[0, 1]).unwrap(), &2.0);
    assert_eq!(reshaped.get(&[1, 0]).unwrap(), &3.0);
    assert_eq!(reshaped.get(&[3, 1]).unwrap(), &8.0);
}

#[test]
fn reshape_1d_to_3d() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let shape = Shape::new(vec![8]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let new_shape = Shape::new(vec![2, 2, 2]).unwrap();
    let reshaped = reshape_tensor(&tensor, new_shape).unwrap();

    assert_eq!(reshaped.shape.dims, vec![2, 2, 2]);
    assert_eq!(reshaped.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    // Verify 3D indexing works
    assert_eq!(reshaped.get(&[0, 0, 0]).unwrap(), &1.0);
    assert_eq!(reshaped.get(&[0, 0, 1]).unwrap(), &2.0);
    assert_eq!(reshaped.get(&[1, 1, 1]).unwrap(), &8.0);
}

#[test]
fn reshape_3d_to_1d() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let shape = Shape::new(vec![2, 2, 2]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let new_shape = Shape::new(vec![8]).unwrap();
    let reshaped = reshape_tensor(&tensor, new_shape).unwrap();

    assert_eq!(reshaped.shape.dims, vec![8]);
    assert_eq!(reshaped.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

// ============ EDGE CASES ============
#[test]
fn reshape_single_element() {
    let data = vec![42.0];
    let shape = Shape::new(vec![1]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let new_shape = Shape::new(vec![1, 1]).unwrap();
    let reshaped = reshape_tensor(&tensor, new_shape).unwrap();

    assert_eq!(reshaped.shape.dims, vec![1, 1]);
    assert_eq!(reshaped.data, vec![42.0]);
    assert_eq!(reshaped.get(&[0, 0]).unwrap(), &42.0);
}

#[test]
fn reshape_single_element_to_3d() {
    let data = vec![7.5];
    let shape = Shape::new(vec![1]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let new_shape = Shape::new(vec![1, 1, 1]).unwrap();
    let reshaped = reshape_tensor(&tensor, new_shape).unwrap();

    assert_eq!(reshaped.shape.dims, vec![1, 1, 1]);
    assert_eq!(reshaped.data, vec![7.5]);
    assert_eq!(reshaped.get(&[0, 0, 0]).unwrap(), &7.5);
}

#[test]
fn reshape_row_vector_to_column_vector() {
    let data = vec![1.0, 2.0, 3.0];
    let shape = Shape::new(vec![1, 3]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let new_shape = Shape::new(vec![3, 1]).unwrap();
    let reshaped = reshape_tensor(&tensor, new_shape).unwrap();

    assert_eq!(reshaped.shape.dims, vec![3, 1]);
    assert_eq!(reshaped.data, vec![1.0, 2.0, 3.0]);
    assert_eq!(reshaped.get(&[0, 0]).unwrap(), &1.0);
    assert_eq!(reshaped.get(&[2, 0]).unwrap(), &3.0);
}

#[test]
fn reshape_with_negative_numbers() {
    let data = vec![-1.0, 2.0, -3.0, 4.0];
    let shape = Shape::new(vec![4]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let new_shape = Shape::new(vec![2, 2]).unwrap();
    let reshaped = reshape_tensor(&tensor, new_shape).unwrap();

    assert_eq!(reshaped.shape.dims, vec![2, 2]);
    assert_eq!(reshaped.data, vec![-1.0, 2.0, -3.0, 4.0]);
    assert_eq!(reshaped.get(&[0, 0]).unwrap(), &-1.0);
    assert_eq!(reshaped.get(&[1, 1]).unwrap(), &4.0);
}

#[test]
fn reshape_with_zeros() {
    let data = vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
    let shape = Shape::new(vec![6]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let new_shape = Shape::new(vec![3, 2]).unwrap();
    let reshaped = reshape_tensor(&tensor, new_shape).unwrap();

    assert_eq!(reshaped.shape.dims, vec![3, 2]);
    assert_eq!(reshaped.data, vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0]);
}

// ============ IDENTITY reshape_tensor ============
#[test]
fn reshape_identity_same_shape() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = Shape::new(vec![2, 2]).unwrap();
    let tensor = Tensor::from_vec(data.clone(), shape.clone()).unwrap();

    let reshaped = reshape_tensor(&tensor, shape).unwrap();

    assert_eq!(reshaped.shape.dims, tensor.shape.dims);
    assert_eq!(reshaped.data, tensor.data);
}

// ============ ERROR CASES ============
#[test]
fn reshape_mismatched_total_elements() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = Shape::new(vec![2, 3]).unwrap(); // 6 elements
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let new_shape = Shape::new(vec![2, 2]).unwrap(); // 4 elements - mismatch!
    let result = reshape_tensor(&tensor, new_shape);

    assert!(result.is_err());
    if let Err(VynapseError::TensorError(msg)) = result {
        assert!(msg.contains("elements") || msg.contains("match") || msg.contains("total"));
    } else {
        panic!("Expected TensorError for mismatched element count");
    }
}

#[test]
fn reshape_larger_mismatch() {
    let data = vec![1.0, 2.0, 3.0];
    let shape = Shape::new(vec![3]).unwrap(); // 3 elements
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let new_shape = Shape::new(vec![2, 3]).unwrap(); // 6 elements - mismatch!
    let result = reshape_tensor(&tensor, new_shape);

    assert!(result.is_err());
    if let Err(VynapseError::TensorError(msg)) = result {
        assert!(msg.contains("elements") || msg.contains("match") || msg.contains("total"));
    } else {
        panic!("Expected TensorError for mismatched element count");
    }
}

// ============ LARGE TENSOR TEST ============
#[test]
fn reshape_large_tensor() {
    let mut data = Vec::new();
    for i in 0..24 {
        data.push(i as f32);
    }
    let shape = Shape::new(vec![24]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let new_shape = Shape::new(vec![4, 6]).unwrap();
    let reshaped = reshape_tensor(&tensor, new_shape).unwrap();

    assert_eq!(reshaped.shape.dims, vec![4, 6]);
    assert_eq!(reshaped.data.len(), 24);

    // Verify first and last elements
    assert_eq!(reshaped.get(&[0, 0]).unwrap(), &0.0);
    assert_eq!(reshaped.get(&[3, 5]).unwrap(), &23.0);

    // Verify a middle element
    assert_eq!(reshaped.get(&[2, 3]).unwrap(), &15.0); // 2*6 + 3 = 15
}

// ============ INTEGER TYPE TEST ============
#[test]
fn reshape_integer_tensor() {
    let data = vec![1, 2, 3, 4, 5, 6];
    let shape = Shape::new(vec![6]).unwrap();
    let tensor = Tensor::from_vec(data, shape).unwrap();

    let new_shape = Shape::new(vec![2, 3]).unwrap();
    let reshaped = reshape_tensor(&tensor, new_shape).unwrap();

    assert_eq!(reshaped.shape.dims, vec![2, 3]);
    assert_eq!(reshaped.data, vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(reshaped.get(&[1, 2]).unwrap(), &6);
}

// ============ COMPLEX reshape_tensor CHAIN ============
#[test]
fn reshape_chain_transformations() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let shape = Shape::new(vec![8]).unwrap();
    let tensor = Tensor::from_vec(data.clone(), shape).unwrap();

    // 1D -> 2D -> 3D -> back to 1D
    let shape_2d = Shape::new(vec![4, 2]).unwrap();
    let reshaped_2d = reshape_tensor(&tensor, shape_2d).unwrap();

    let shape_3d = Shape::new(vec![2, 2, 2]).unwrap();
    let reshaped_3d = reshape_tensor(&reshaped_2d, shape_3d).unwrap();

    let shape_1d = Shape::new(vec![8]).unwrap();
    let final_reshaped = reshape_tensor(&reshaped_3d, shape_1d).unwrap();

    // Should be back to original
    assert_eq!(final_reshaped.shape.dims, vec![8]);
    assert_eq!(final_reshaped.data, data);
}
