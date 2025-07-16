use crate::traits::fitness::Fitness;
use vynapse_common::Result;

#[derive(Debug, Clone)]
pub struct XorFitness;

impl XorFitness {
    pub fn new() -> Result<Self> {
        Ok(XorFitness)
    }
}

impl Fitness for XorFitness {
    fn evaluate<G: crate::traits::genome::Genome>(&self, genome: &G) -> Result<f32> {
        let xor_cases = [
            ([0.0, 0.0], 0.0),
            ([0.0, 1.0], 1.0),
            ([1.0, 0.0], 1.0),
            ([1.0, 1.0], 0.0),
        ];

        let assumed_shape: Vec<usize> = vec![2, 4, 1];
        let weights = genome.get_weights();

        let first_layer_shape_weights = assumed_shape[0] * assumed_shape[1];
        let second_layer_shape_weights = assumed_shape[1] * assumed_shape[2];

        let first_layer_shape = &assumed_shape[0..=1].to_vec();
        let second_layer_shape = &assumed_shape[1..=2].to_vec();

        let first_layer_weights = &weights[0..first_layer_shape_weights];
        let second_layer_weights = &weights
            [first_layer_shape_weights..first_layer_shape_weights + second_layer_shape_weights];

        let first_layer: Vec<f32> = first_layer_weights.to_vec();
        let second_layer: Vec<f32> = second_layer_weights.to_vec();

        todo!()
    }
}

#[cfg(test)]
#[test]

fn test_if_layers_work() {
    let assumed_shape: Vec<usize> = vec![2, 4, 1];
    let first_layer_shape = &assumed_shape[0..=1].to_vec();
    let second_layer_shape = &assumed_shape[1..=2].to_vec();

    assert_eq!(first_layer_shape, &vec![2_usize, 4_usize]);
    assert_eq!(second_layer_shape, &vec![4_usize, 1_usize]);
}
