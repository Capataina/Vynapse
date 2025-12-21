---
name: evolutionary-trainer-modular-refactor
description: Plan to modularize evolutionary trainer with config, population, operators, and stats
---

# Plan

Refactor the evolutionary trainer to use modular training_setup components (config, population, operators, stats) while keeping generic traits and adding minimal validation/testing.

## Requirements

- Centralize TrainingStats usage to training_setup version and remove inline duplicate.
- Add EvolutionConfig with trainer-level coordination parameters (generations, stagnation limit only).
- Implement Population<G> lifecycle (init with mutation-based diversity, size validation, replacement/elitism hooks).
- Implement GeneticOperators<M, C> wrapping mutation/crossover and probability logic.
- Refactor EvolutionaryTrainer to compose new pieces and drive the generation loop.
- Add integration test demonstrating learning on PowersOfTwo task.

## Scope

- In: training_setup modules, traits/trainer, trainers/evolutionary, genome/mutation/crossover/selection traits.
- Out: CLI/config parsing, advanced selection strategies beyond current traits, logging/CSV output.

## Files and entry points

- vynapse-core/src/training_setup/evolution_config.rs
- vynapse-core/src/training_setup/population.rs
- vynapse-core/src/training_setup/fitness_stats.rs
- vynapse-core/src/training_setup/genetic_operators.rs
- vynapse-core/src/training_setup/training_stats.rs
- vynapse-core/src/traits/trainer.rs
- vynapse-core/src/trainers/evolutionary.rs
- vynapse-core/src/traits/genome.rs, vynapse-core/src/traits/{mutation.rs,crossover.rs,selection.rs,fitness.rs}

## Data model / API changes

- EvolutionConfig struct with validated hyperparameters (generations, stagnation_limit only). Note: population_size belongs to Population, mutation/crossover rates belong to GeneticOperators.
- Population<G> owns Vec<G> plus initialization/replacement methods. Population is initialized externally before being passed to trainer.
- GeneticOperators<M, C> bundles operators and rates, exposes offspring generation.
- Trainer trait returns training_setup::TrainingStats; inline ConvergenceStatus/TrainingStats removed.
- EvolutionaryTrainer<G, M, C, F, S> fields updated to use the new structs and stats.

## Action items

[x] Align TrainingStats usage: adopt training_setup::TrainingStats in traits/trainer.rs and remove inline version; ensure ConvergenceStatus coherence.
[x] Implement EvolutionConfig with max generations and stagnation limit only; include validation helpers. Note: population_size, mutation_rate, and crossover_rate belong to Population and GeneticOperators respectively, not EvolutionConfig.
[x] Implement Population<G>: store genomes, initialize from template + mutation for diversity, validate size, expose replacement hooks (set_all_genomes). Population initialization happens externally before trainer construction.
[x] Implement GeneticOperators<M, C>: hold mutation/crossover + rates, generate offspring given parents and probabilities; validate rate ranges.
[x] Refactor EvolutionaryTrainer<G, M, C, F, S> to compose Population, GeneticOperators, EvolutionConfig, TrainingStats; wire generation loop: evaluate via F, record FitnessStats/TrainingStats, select via S, generate offspring via GeneticOperators, replace population, check convergence/limits.
[x] Add integration test (test_evolutionary_trainer_learns_powers_of_two) that runs full training loop demonstrating learning on PowersOfTwo task with population size, stats updates, and convergence detection.

## Testing and validation

- Run `cargo test` (all tests passing including integration test).
- Integration test demonstrates actual learning with fitness improvement over generations.

## Risks and edge cases

- Generic bounds mismatched across Population/GeneticOperators/Trainer causing trait impl issues. ✅ Resolved
- Duplicate or conflicting TrainingStats definitions if inline version not fully removed. ✅ Resolved - only training_setup::TrainingStats remains
- Mutation/crossover rates or population size misvalidated, leading to runtime panics. ✅ Handled with validation in respective components

## Design decisions

- **EvolutionConfig scope**: Only contains trainer-level coordination parameters (generations, stagnation_limit). Component-specific parameters (population_size, mutation_rate, crossover_rate) belong to their respective components. This maintains clean separation of concerns.
- **Population initialization**: Population is initialized externally before being passed to trainer. This provides maximum flexibility (template-based, file-based, custom initialization strategies) and keeps the trainer focused on training logic.
- **Trainer trait implementation**: Separate `impl EvolutionaryTrainer` block for type-specific methods (new), and `impl Trainer for EvolutionaryTrainer` for trait methods (train, step, get_stats, etc.). This follows standard Rust patterns.

## Open questions

- ~~Should Trainer::get_stats return owned or reference to avoid copies?~~ ✅ Decided: Returns owned TrainingStats (acceptable for MVP).
- ~~Do we enforce deterministic seeding for initialization/mutation in the MVP test?~~ ✅ Not enforced in MVP - uses random seeding.
