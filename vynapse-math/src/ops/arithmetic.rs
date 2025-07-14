use std::ops::Add;

use num_traits::Zero;
use vynapse_core::Result;

use crate::Tensor;

pub fn tensor_add<T>(tensor_one: &Tensor<T>, tensor_two: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Add<T, Output = T> + Zero,
{
    if tensor_one.shape.dims != tensor_two.shape.dims {
        return Err(vynapse_core::VynapseError::TensorError(
            "Given tensors have different shapes.".to_string(),
        ));
    }

    let mut new_tensor: Tensor<T> = Tensor::zeros(tensor_one.shape.clone())?;

    for i in 0..tensor_one.data.len() {
        new_tensor.data[i] = tensor_one.data[i].clone() + tensor_two.data[i].clone();
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
