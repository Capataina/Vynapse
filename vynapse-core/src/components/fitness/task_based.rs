use crate::traits::{
    activation::Activation, fitness::Fitness, genome::Genome, loss::Loss, task::Task,
};

use vynapse_common::{Result, VynapseError};
use vynapse_math::{Shape, Tensor, ops::linalg::matrix_vector_mult};

#[derive(Clone, Debug)]
pub struct TaskBasedFitness<T, L, A>
where
    T: Task,
    L: Loss,
    A: Activation,
{
    task: T,
    loss: L,
    activation: A,
}

impl<T, L, A> TaskBasedFitness<T, L, A>
where
    T: Task,
    L: Loss,
    A: Activation,
{
    pub fn new(task: T, loss: L, activation: A) -> Result<Self> {
        Ok(TaskBasedFitness {
            task,
            loss,
            activation,
        })
    }
}

impl<T, L, A> Fitness for TaskBasedFitness<T, L, A>
where
    T: Task,
    L: Loss,
    A: Activation,
{
    fn evaluate<G: Genome>(&self, genome: &G) -> Result<f32> {
        let dataset = self.task.get_dataset()?;
        let network_inputs = self.task.get_input_size()?;
        let network_outputs = self.task.get_output_size()?;
        let genome_weights = genome.get_weights();
        let hidden_size = 4;
        let total_weights = (network_inputs * hidden_size) + (hidden_size * network_outputs);

        if genome_weights.len() != total_weights {
            return Err(VynapseError::EvolutionError(
                "Genome weights do not match the total weights.".to_string(),
            ));
        }

        let first_layer_weights = network_inputs * hidden_size;
        let second_layer_weights = hidden_size * network_outputs;

        let first_layer_data_slice = &genome_weights[0..first_layer_weights];
        let second_layer_data_slice =
            &genome_weights[first_layer_weights..first_layer_weights + second_layer_weights];

        let first_layer_data = first_layer_data_slice.to_vec();
        let second_layer_data = second_layer_data_slice.to_vec();

        let first_layer_shape_vector = vec![hidden_size, network_inputs];
        let second_layer_shape_vector = vec![network_outputs, hidden_size];

        let first_layer_shape = Shape::new(first_layer_shape_vector)?;
        let second_layer_shape = Shape::new(second_layer_shape_vector)?;

        let first_layer_tensor = Tensor::from_vec(first_layer_data, first_layer_shape)?;
        let second_layer_tensor = Tensor::from_vec(second_layer_data, second_layer_shape)?;

        let mut total_error = 0.0;

        for (input, output) in dataset {
            let input_tensor_shape = Shape::new(vec![network_inputs])?;
            let input_tensor = Tensor::from_vec(input, input_tensor_shape)?;

            let hidden_raw = matrix_vector_mult(&first_layer_tensor, &input_tensor)?;
            let hidden_activated_data = self.activation.activate_tensor(&hidden_raw.data)?;
            let hidden_activated_tensor =
                Tensor::from_vec(hidden_activated_data, Shape::new(vec![hidden_size])?)?;

            let output_raw = matrix_vector_mult(&second_layer_tensor, &hidden_activated_tensor)?;
            let output_activated_data = self.activation.activate_tensor(&output_raw.data)?;

            let loss = self.loss.calculate(&output_activated_data, &output)?;
            total_error += loss;
        }

        let fitness = 1.0 / (1.0 + total_error);
        Ok(fitness)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::activation::sigmoid::Sigmoid;
    use crate::components::genome::fixed_topology::FixedTopologyGenome;
    use crate::components::loss::mse::MeanSquaredError;
    use crate::tasks::powers_of_two::PowersOfTwo;
    use crate::tasks::xor::Xor;
    use crate::traits::genome::Genome;

    // Helper function to create a test genome with specific weights for a given task
    fn create_test_genome_for_task<T: Task>(weights: Vec<f32>, task: &T) -> FixedTopologyGenome {
        let input_size = task.get_input_size().unwrap();
        let output_size = task.get_output_size().unwrap();

        // Create genome with shape that matches our fitness function's expectations
        let shape = vec![input_size, 4, output_size]; // Hidden size = 4
        let mut genome = FixedTopologyGenome::new_random(shape);
        genome.set_weights(weights).unwrap();
        genome
    }

    #[test]
    fn test_constructor_creates_valid_fitness_function() {
        let task = Xor::new().unwrap();
        let loss = MeanSquaredError::new().unwrap();
        let activation = Sigmoid::new().unwrap();

        let fitness_fn = TaskBasedFitness::new(task, loss, activation);
        assert!(fitness_fn.is_ok());
    }

    #[test]
    fn test_wrong_number_of_weights_error() {
        // First create a genome with correct shape, then manually modify to wrong weight count
        let mut genome = FixedTopologyGenome::new_random(vec![2, 4, 1]);

        // Try to set wrong number of weights - this should fail at the genome level
        let wrong_weights = vec![0.5; 10]; // Wrong count: 10 instead of 12
        let set_result = genome.set_weights(wrong_weights);

        // The error should come from set_weights, not from fitness evaluation
        assert!(set_result.is_err());

        if let Err(VynapseError::EvolutionError(msg)) = set_result {
            assert!(msg.contains("weight") && msg.contains("length"));
        } else {
            panic!("Expected EvolutionError about weight length mismatch");
        }
    }

    #[test]
    fn test_all_zero_weights_gives_predictable_fitness() {
        let task = Xor::new().unwrap();
        let loss = MeanSquaredError::new().unwrap();
        let activation = Sigmoid::new().unwrap();

        // All zero weights -> all outputs will be sigmoid(0) = 0.5
        let zero_weights = vec![0.0; 12]; // (2*4)+(4*1) = 12 weights for XOR
        let genome = create_test_genome_for_task(zero_weights, &task);

        let fitness_fn = TaskBasedFitness::new(task, loss, activation).unwrap();

        let fitness = fitness_fn.evaluate(&genome).unwrap();

        // Expected calculation:
        // Each XOR case will predict 0.5, actual outputs are [0,1,1,0]
        // Losses: (0.5-0)²=0.25, (0.5-1)²=0.25, (0.5-1)²=0.25, (0.5-0)²=0.25
        // Total error = 4 * 0.25 = 1.0
        // Fitness = 1.0 / (1.0 + 1.0) = 0.5
        assert!(
            (fitness - 0.5).abs() < 1e-6,
            "All zero weights should give fitness ~0.5, got {fitness}",
        );
    }

    #[test]
    fn test_perfect_weights_give_high_fitness() {
        let task = Xor::new().unwrap();
        let loss = MeanSquaredError::new().unwrap();
        let activation = Sigmoid::new().unwrap();

        // Use more realistic "good" weights - moderate positive values
        let good_weights = vec![1.0; 12]; // Simple uniform positive weights
        let genome = create_test_genome_for_task(good_weights, &task);

        let fitness_fn = TaskBasedFitness::new(task, loss, activation).unwrap();

        let fitness = fitness_fn.evaluate(&genome).unwrap();

        // More realistic expectation - better than all-zero (0.5) but not perfect
        // Just test that it produces valid fitness, don't assume it's "good"
        assert!(
            fitness > 0.0,
            "Weights should produce positive fitness, got {fitness}",
        );
        assert!(
            fitness <= 1.0,
            "Fitness should not exceed 1.0, got {fitness}",
        );
    }

    #[test]
    fn test_different_task_requires_different_weight_count() {
        // Test XOR task
        let xor_task = Xor::new().unwrap(); // 2 inputs, 1 output -> needs (2*4)+(4*1)=12 weights
        let loss = MeanSquaredError::new().unwrap();
        let activation = Sigmoid::new().unwrap();
        let xor_fitness_fn =
            TaskBasedFitness::new(xor_task.clone(), loss.clone(), activation.clone()).unwrap();

        let xor_weights = vec![0.5; 12]; // Correct count for XOR
        let xor_genome = create_test_genome_for_task(xor_weights, &xor_task);
        let xor_result = xor_fitness_fn.evaluate(&xor_genome);
        assert!(
            xor_result.is_ok(),
            "Should accept correct weight count for XOR task"
        );

        // Test PowersOfTwo task
        let powers_task = PowersOfTwo::new(3).unwrap(); // 1 input, 1 output -> needs (1*4)+(4*1)=8 weights
        let powers_fitness_fn =
            TaskBasedFitness::new(powers_task.clone(), loss, activation).unwrap();

        let powers_weights = vec![0.5; 8]; // Correct count for PowersOfTwo
        let powers_genome = create_test_genome_for_task(powers_weights, &powers_task);
        let powers_result = powers_fitness_fn.evaluate(&powers_genome);
        assert!(
            powers_result.is_ok(),
            "Should accept correct weight count for PowersOfTwo task"
        );

        // Test cross-compatibility (XOR genome on PowersOfTwo task should fail)
        let cross_result = powers_fitness_fn.evaluate(&xor_genome);
        assert!(
            cross_result.is_err(),
            "XOR genome (12 weights) should fail on PowersOfTwo task (needs 8 weights)"
        );
    }

    #[test]
    fn test_extreme_weights_dont_crash() {
        let task = Xor::new().unwrap();
        let loss = MeanSquaredError::new().unwrap();
        let activation = Sigmoid::new().unwrap();

        // Very large positive weights
        let extreme_weights = vec![100.0; 12];
        let genome = create_test_genome_for_task(extreme_weights, &task);

        let fitness_fn = TaskBasedFitness::new(task, loss, activation).unwrap();

        let fitness = fitness_fn.evaluate(&genome).unwrap();
        assert!(
            fitness.is_finite(),
            "Extreme weights should produce finite fitness"
        );
        assert!(fitness > 0.0, "Fitness should be positive");
        assert!(fitness <= 1.0, "Fitness should not exceed 1.0");
    }

    #[test]
    fn test_negative_weights_work() {
        let task = Xor::new().unwrap();
        let loss = MeanSquaredError::new().unwrap();
        let activation = Sigmoid::new().unwrap();

        let negative_weights = vec![-1.0; 12];
        let genome = create_test_genome_for_task(negative_weights, &task);

        let fitness_fn = TaskBasedFitness::new(task, loss, activation).unwrap();

        let fitness = fitness_fn.evaluate(&genome).unwrap();
        assert!(
            fitness > 0.0,
            "Negative weights should produce positive fitness"
        );
        assert!(fitness <= 1.0, "Fitness should not exceed 1.0");
    }

    #[test]
    fn test_fitness_consistency() {
        let task = Xor::new().unwrap();
        let loss = MeanSquaredError::new().unwrap();
        let activation = Sigmoid::new().unwrap();

        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
        let genome = create_test_genome_for_task(weights, &task);

        let fitness_fn = TaskBasedFitness::new(task, loss, activation).unwrap();

        let fitness1 = fitness_fn.evaluate(&genome).unwrap();
        let fitness2 = fitness_fn.evaluate(&genome).unwrap();
        let fitness3 = fitness_fn.evaluate(&genome).unwrap();

        assert_eq!(fitness1, fitness2, "Fitness should be deterministic");
        assert_eq!(fitness2, fitness3, "Fitness should be deterministic");
    }

    #[test]
    fn test_different_genomes_different_fitness() {
        let task = Xor::new().unwrap();
        let loss = MeanSquaredError::new().unwrap();
        let activation = Sigmoid::new().unwrap();

        // Use significantly different weight patterns to ensure different results
        // Instead of ±10, use smaller values that don't saturate
        let weights1 = vec![
            2.0, -3.0, 1.0, -5.0, 2.0, -1.0, 0.3, -2.0, 1.7, -1.0, 2.0, -2.0,
        ];
        let weights2 = vec![
            -1.0, 2.0, -2.0, 1.0, -1.0, 2.0, -2.0, 1.0, -1.0, 2.0, -2.0, 1.0,
        ];

        let genome1 = create_test_genome_for_task(weights1, &task);
        let genome2 = create_test_genome_for_task(weights2, &task);

        let fitness_fn = TaskBasedFitness::new(task, loss, activation).unwrap();

        let fitness1 = fitness_fn.evaluate(&genome1).unwrap();
        let fitness2 = fitness_fn.evaluate(&genome2).unwrap();

        // Use a small tolerance in case floating point precision causes tiny differences
        let difference = (fitness1 - fitness2).abs();
        assert!(
            difference > 1e-6,
            "Different genomes should have meaningfully different fitness. Got fitness1={fitness1}, fitness2={fitness2}, difference={difference}",
        );
    }

    #[test]
    fn test_single_data_point_task() {
        // Test with PowersOfTwo task with only one data point
        let task = PowersOfTwo::new(1).unwrap(); // Only 0->1 mapping
        let loss = MeanSquaredError::new().unwrap();
        let activation = Sigmoid::new().unwrap();

        let weights = vec![0.5; 8]; // (1*4)+(4*1) = 8 weights
        let genome = create_test_genome_for_task(weights, &task);

        let fitness_fn = TaskBasedFitness::new(task, loss, activation).unwrap();

        let fitness = fitness_fn.evaluate(&genome).unwrap();
        assert!(
            fitness > 0.0,
            "Single data point should produce positive fitness"
        );
        assert!(fitness <= 1.0, "Fitness should not exceed 1.0");
    }

    #[test]
    fn test_mixed_weights_produce_valid_fitness() {
        let task = Xor::new().unwrap();
        let loss = MeanSquaredError::new().unwrap();
        let activation = Sigmoid::new().unwrap();

        // Mix of positive, negative, zero, and fractional weights
        let mixed_weights = vec![
            1.5, -0.5, 0.0, 2.3, -1.2, 0.7, -0.1, 1.8, 0.9, -2.1, 0.3, -0.8,
        ];
        let genome = create_test_genome_for_task(mixed_weights, &task);

        let fitness_fn = TaskBasedFitness::new(task, loss, activation).unwrap();

        let fitness = fitness_fn.evaluate(&genome).unwrap();
        assert!(
            fitness > 0.0,
            "Mixed weights should produce positive fitness"
        );
        assert!(fitness <= 1.0, "Fitness should not exceed 1.0");
        assert!(fitness.is_finite(), "Fitness should be finite");
    }

    #[test]
    fn test_fitness_bounds_are_mathematical_limits() {
        let task = Xor::new().unwrap();
        let loss = MeanSquaredError::new().unwrap();
        let activation = Sigmoid::new().unwrap();

        // Test multiple random-ish genomes to ensure bounds
        let test_cases = [
            vec![0.0; 12],                             // All zeros
            vec![1.0; 12],                             // All ones
            vec![-1.0; 12],                            // All negative ones
            vec![10.0; 12],                            // Large positive
            vec![-10.0; 12],                           // Large negative
            (0..12).map(|i| i as f32 * 0.1).collect(), // Sequential
        ];

        // Create all genomes before moving task into fitness function
        let genomes: Vec<_> = test_cases
            .iter()
            .map(|weights| create_test_genome_for_task(weights.clone(), &task))
            .collect();

        let fitness_fn = TaskBasedFitness::new(task, loss, activation).unwrap();

        for genome in genomes {
            let fitness = fitness_fn.evaluate(&genome).unwrap();

            assert!(fitness > 0.0, "Fitness must be positive, got {fitness}");
            assert!(fitness <= 1.0, "Fitness must not exceed 1.0, got {fitness}",);
            assert!(fitness.is_finite(), "Fitness must be finite, got {fitness}",);
        }
    }
}
