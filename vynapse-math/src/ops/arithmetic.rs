use std::ops::{Add, Div, Mul, Sub};

use num_traits::Zero;
use vynapse_common::{Result, VynapseError};

use crate::Tensor;

pub fn tensor_add<T>(tensor_one: &Tensor<T>, tensor_two: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Add<T, Output = T> + Zero,
{
    if tensor_one.shape.dims != tensor_two.shape.dims {
        return Err(VynapseError::TensorError(
            "Given tensors have different shapes.".to_string(),
        ));
    }

    let mut new_tensor: Tensor<T> = Tensor::zeros(tensor_one.shape.clone())?;

    for i in 0..tensor_one.data.len() {
        new_tensor.data[i] = tensor_one.data[i].clone() + tensor_two.data[i].clone();
    }

    Ok(new_tensor)
}

pub fn tensor_sub<T>(tensor_one: &Tensor<T>, tensor_two: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Sub<T, Output = T> + Zero,
{
    if tensor_one.shape.dims != tensor_two.shape.dims {
        return Err(VynapseError::TensorError(
            "Given tensors have different shapes.".to_string(),
        ));
    }

    let mut new_tensor: Tensor<T> = Tensor::zeros(tensor_one.shape.clone())?;

    for i in 0..tensor_one.data.len() {
        new_tensor.data[i] = tensor_one.data[i].clone() - tensor_two.data[i].clone();
    }

    Ok(new_tensor)
}

pub fn tensor_mul<T>(tensor_one: &Tensor<T>, tensor_two: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Mul<T, Output = T> + Zero,
{
    if tensor_one.shape.dims != tensor_two.shape.dims {
        return Err(VynapseError::TensorError(
            "Given tensors have different shapes.".to_string(),
        ));
    }

    let mut new_tensor: Tensor<T> = Tensor::zeros(tensor_one.shape.clone())?;

    for i in 0..tensor_one.data.len() {
        new_tensor.data[i] = tensor_one.data[i].clone() * tensor_two.data[i].clone();
    }

    Ok(new_tensor)
}

pub fn tensor_div<T>(tensor_one: &Tensor<T>, tensor_two: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Div<T, Output = T> + Zero + PartialEq,
{
    if tensor_one.shape.dims != tensor_two.shape.dims {
        return Err(VynapseError::TensorError(
            "Given tensors have different shapes.".to_string(),
        ));
    }

    for number in tensor_two.data.clone() {
        if number.clone() == T::zero() {
            return Err(VynapseError::TensorError(
                "Cannot do the division as the second tensor has a 0 in it's data.".to_string(),
            ));
        }
    }

    let mut new_tensor: Tensor<T> = Tensor::zeros(tensor_one.shape.clone())?;

    for i in 0..tensor_one.data.len() {
        new_tensor.data[i] = tensor_one.data[i].clone() / tensor_two.data[i].clone();
    }

    Ok(new_tensor)
}

#[cfg(test)]
use crate::Shape;
#[test]
fn two_1d_tensors() {
    let tensor1_array = vec![1.0, 2.0];
    let tensor1_shape = Shape::new(vec![2]).unwrap();
    let tensor1 = Tensor::from_vec(tensor1_array, tensor1_shape).unwrap();

    let tensor2_array = vec![2.0, 5.0];
    let tensor2_shape = Shape::new(vec![2]).unwrap();
    let tensor2 = Tensor::from_vec(tensor2_array, tensor2_shape).unwrap();

    let addition_result = tensor_add(&tensor1, &tensor2).unwrap();
    assert_eq!(addition_result.data, vec![3.0, 7.0]);
}

#[test]
fn two_2d_tensors() {
    let tensor1_array = vec![1.0, 2.0, 7.0, 4.0];
    let tensor1_shape = Shape::new(vec![2, 2]).unwrap();
    let tensor1 = Tensor::from_vec(tensor1_array, tensor1_shape).unwrap();

    let tensor2_array = vec![2.0, 5.0, 9.0, 11.0];
    let tensor2_shape = Shape::new(vec![2, 2]).unwrap();
    let tensor2 = Tensor::from_vec(tensor2_array, tensor2_shape).unwrap();

    let addition_result = tensor_add(&tensor1, &tensor2).unwrap();
    assert_eq!(addition_result.data, vec![3.0, 7.0, 16.0, 15.0]);
}

#[test]
fn shape_mismatch() {
    let tensor1_array = vec![1.0, 2.0, 7.0, 4.0];
    let tensor1_shape = Shape::new(vec![1, 4]).unwrap();
    let tensor1 = Tensor::from_vec(tensor1_array, tensor1_shape).unwrap();

    let tensor2_array = vec![2.0, 5.0, 9.0, 11.0];
    let tensor2_shape = Shape::new(vec![2, 2]).unwrap();
    let tensor2 = Tensor::from_vec(tensor2_array, tensor2_shape).unwrap();

    let addition_result = tensor_add(&tensor1, &tensor2);
    assert!(addition_result.is_err());
}

#[test]
fn tensor_sub_1d_tensors() {
    let tensor1_array = vec![5.0, 8.0, 3.0];
    let tensor1_shape = Shape::new(vec![3]).unwrap();
    let tensor1 = Tensor::from_vec(tensor1_array, tensor1_shape).unwrap();

    let tensor2_array = vec![2.0, 3.0, 1.0];
    let tensor2_shape = Shape::new(vec![3]).unwrap();
    let tensor2 = Tensor::from_vec(tensor2_array, tensor2_shape).unwrap();

    let result = tensor_sub(&tensor1, &tensor2).unwrap();
    assert_eq!(result.data, vec![3.0, 5.0, 2.0]);
    assert_eq!(result.shape.dims, vec![3]);
}

#[test]
fn tensor_sub_2d_tensors() {
    let tensor1_array = vec![10.0, 8.0, 6.0, 4.0];
    let tensor1_shape = Shape::new(vec![2, 2]).unwrap();
    let tensor1 = Tensor::from_vec(tensor1_array, tensor1_shape).unwrap();

    let tensor2_array = vec![3.0, 2.0, 1.0, 4.0];
    let tensor2_shape = Shape::new(vec![2, 2]).unwrap();
    let tensor2 = Tensor::from_vec(tensor2_array, tensor2_shape).unwrap();

    let result = tensor_sub(&tensor1, &tensor2).unwrap();
    assert_eq!(result.data, vec![7.0, 6.0, 5.0, 0.0]);
    assert_eq!(result.shape.dims, vec![2, 2]);
}

#[test]
fn tensor_sub_negative_results() {
    let tensor1_array = vec![1.0, 2.0];
    let tensor1_shape = Shape::new(vec![2]).unwrap();
    let tensor1 = Tensor::from_vec(tensor1_array, tensor1_shape).unwrap();

    let tensor2_array = vec![5.0, 3.0];
    let tensor2_shape = Shape::new(vec![2]).unwrap();
    let tensor2 = Tensor::from_vec(tensor2_array, tensor2_shape).unwrap();

    let result = tensor_sub(&tensor1, &tensor2).unwrap();
    assert_eq!(result.data, vec![-4.0, -1.0]);
}

#[test]
fn tensor_sub_shape_mismatch() {
    let tensor1_array = vec![1.0, 2.0, 3.0, 4.0];
    let tensor1_shape = Shape::new(vec![1, 4]).unwrap();
    let tensor1 = Tensor::from_vec(tensor1_array, tensor1_shape).unwrap();

    let tensor2_array = vec![2.0, 5.0, 9.0, 11.0];
    let tensor2_shape = Shape::new(vec![2, 2]).unwrap();
    let tensor2 = Tensor::from_vec(tensor2_array, tensor2_shape).unwrap();

    let result = tensor_sub(&tensor1, &tensor2);
    assert!(result.is_err());
}

#[test]
fn tensor_mul_1d_tensors() {
    let tensor1_array = vec![2.0, 3.0, 4.0];
    let tensor1_shape = Shape::new(vec![3]).unwrap();
    let tensor1 = Tensor::from_vec(tensor1_array, tensor1_shape).unwrap();

    let tensor2_array = vec![5.0, 2.0, 3.0];
    let tensor2_shape = Shape::new(vec![3]).unwrap();
    let tensor2 = Tensor::from_vec(tensor2_array, tensor2_shape).unwrap();

    let result = tensor_mul(&tensor1, &tensor2).unwrap();
    assert_eq!(result.data, vec![10.0, 6.0, 12.0]);
    assert_eq!(result.shape.dims, vec![3]);
}

#[test]
fn tensor_mul_2d_tensors() {
    let tensor1_array = vec![2.0, 3.0, 4.0, 5.0];
    let tensor1_shape = Shape::new(vec![2, 2]).unwrap();
    let tensor1 = Tensor::from_vec(tensor1_array, tensor1_shape).unwrap();

    let tensor2_array = vec![1.0, 2.0, 3.0, 4.0];
    let tensor2_shape = Shape::new(vec![2, 2]).unwrap();
    let tensor2 = Tensor::from_vec(tensor2_array, tensor2_shape).unwrap();

    let result = tensor_mul(&tensor1, &tensor2).unwrap();
    assert_eq!(result.data, vec![2.0, 6.0, 12.0, 20.0]);
    assert_eq!(result.shape.dims, vec![2, 2]);
}

#[test]
fn tensor_mul_with_zeros() {
    let tensor1_array = vec![2.0, 0.0, 4.0];
    let tensor1_shape = Shape::new(vec![3]).unwrap();
    let tensor1 = Tensor::from_vec(tensor1_array, tensor1_shape).unwrap();

    let tensor2_array = vec![5.0, 3.0, 0.0];
    let tensor2_shape = Shape::new(vec![3]).unwrap();
    let tensor2 = Tensor::from_vec(tensor2_array, tensor2_shape).unwrap();

    let result = tensor_mul(&tensor1, &tensor2).unwrap();
    assert_eq!(result.data, vec![10.0, 0.0, 0.0]);
}

#[test]
fn tensor_mul_with_negatives() {
    let tensor1_array = vec![-2.0, 3.0, -4.0];
    let tensor1_shape = Shape::new(vec![3]).unwrap();
    let tensor1 = Tensor::from_vec(tensor1_array, tensor1_shape).unwrap();

    let tensor2_array = vec![5.0, -2.0, -3.0];
    let tensor2_shape = Shape::new(vec![3]).unwrap();
    let tensor2 = Tensor::from_vec(tensor2_array, tensor2_shape).unwrap();

    let result = tensor_mul(&tensor1, &tensor2).unwrap();
    assert_eq!(result.data, vec![-10.0, -6.0, 12.0]);
}

#[test]
fn tensor_mul_shape_mismatch() {
    let tensor1_array = vec![1.0, 2.0];
    let tensor1_shape = Shape::new(vec![2]).unwrap();
    let tensor1 = Tensor::from_vec(tensor1_array, tensor1_shape).unwrap();

    let tensor2_array = vec![2.0, 5.0, 9.0];
    let tensor2_shape = Shape::new(vec![3]).unwrap();
    let tensor2 = Tensor::from_vec(tensor2_array, tensor2_shape).unwrap();

    let result = tensor_mul(&tensor1, &tensor2);
    assert!(result.is_err());
}

#[test]
fn tensor_div_1d_tensors() {
    let tensor1_array = vec![10.0, 15.0, 8.0];
    let tensor1_shape = Shape::new(vec![3]).unwrap();
    let tensor1 = Tensor::from_vec(tensor1_array, tensor1_shape).unwrap();

    let tensor2_array = vec![2.0, 3.0, 4.0];
    let tensor2_shape = Shape::new(vec![3]).unwrap();
    let tensor2 = Tensor::from_vec(tensor2_array, tensor2_shape).unwrap();

    let result = tensor_div(&tensor1, &tensor2).unwrap();
    assert_eq!(result.data, vec![5.0, 5.0, 2.0]);
    assert_eq!(result.shape.dims, vec![3]);
}

#[test]
fn tensor_div_2d_tensors() {
    let tensor1_array = vec![8.0, 12.0, 15.0, 20.0];
    let tensor1_shape = Shape::new(vec![2, 2]).unwrap();
    let tensor1 = Tensor::from_vec(tensor1_array, tensor1_shape).unwrap();

    let tensor2_array = vec![2.0, 4.0, 5.0, 4.0];
    let tensor2_shape = Shape::new(vec![2, 2]).unwrap();
    let tensor2 = Tensor::from_vec(tensor2_array, tensor2_shape).unwrap();

    let result = tensor_div(&tensor1, &tensor2).unwrap();
    assert_eq!(result.data, vec![4.0, 3.0, 3.0, 5.0]);
    assert_eq!(result.shape.dims, vec![2, 2]);
}

#[test]
fn tensor_div_with_negatives() {
    let tensor1_array = vec![-10.0, 12.0, -8.0];
    let tensor1_shape = Shape::new(vec![3]).unwrap();
    let tensor1 = Tensor::from_vec(tensor1_array, tensor1_shape).unwrap();

    let tensor2_array = vec![2.0, -3.0, 4.0];
    let tensor2_shape = Shape::new(vec![3]).unwrap();
    let tensor2 = Tensor::from_vec(tensor2_array, tensor2_shape).unwrap();

    let result = tensor_div(&tensor1, &tensor2).unwrap();
    assert_eq!(result.data, vec![-5.0, -4.0, -2.0]);
}

#[test]
fn tensor_div_fractional_results() {
    let tensor1_array = vec![1.0, 3.0, 7.0];
    let tensor1_shape = Shape::new(vec![3]).unwrap();
    let tensor1 = Tensor::from_vec(tensor1_array, tensor1_shape).unwrap();

    let tensor2_array = vec![2.0, 2.0, 2.0];
    let tensor2_shape = Shape::new(vec![3]).unwrap();
    let tensor2 = Tensor::from_vec(tensor2_array, tensor2_shape).unwrap();

    let result = tensor_div(&tensor1, &tensor2).unwrap();
    assert_eq!(result.data, vec![0.5, 1.5, 3.5]);
}

#[test]
fn tensor_div_by_zero_error() {
    let tensor1_array = vec![1.0, 2.0, 3.0];
    let tensor1_shape = Shape::new(vec![3]).unwrap();
    let tensor1 = Tensor::from_vec(tensor1_array, tensor1_shape).unwrap();

    let tensor2_array = vec![2.0, 0.0, 4.0]; // Contains zero
    let tensor2_shape = Shape::new(vec![3]).unwrap();
    let tensor2 = Tensor::from_vec(tensor2_array, tensor2_shape).unwrap();

    let result = tensor_div(&tensor1, &tensor2);
    assert!(result.is_err());

    // Verify the error message contains "division by zero" or similar
    if let Err(VynapseError::TensorError(msg)) = result {
        assert!(msg.to_lowercase().contains("zero") || msg.to_lowercase().contains("division"));
    } else {
        panic!("Expected TensorError with division by zero message");
    }
}

#[test]
fn tensor_div_shape_mismatch() {
    let tensor1_array = vec![8.0, 12.0, 15.0, 20.0];
    let tensor1_shape = Shape::new(vec![4]).unwrap();
    let tensor1 = Tensor::from_vec(tensor1_array, tensor1_shape).unwrap();

    let tensor2_array = vec![2.0, 4.0, 5.0, 4.0];
    let tensor2_shape = Shape::new(vec![2, 2]).unwrap();
    let tensor2 = Tensor::from_vec(tensor2_array, tensor2_shape).unwrap();

    let result = tensor_div(&tensor1, &tensor2);
    assert!(result.is_err());
}
