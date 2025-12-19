---
name: evolutionary-trainer-modular-refactor
description: Plan to modularize evolutionary trainer with config, population, operators, and stats
---

# Plan

Refactor the evolutionary trainer to use modular training_setup components (config, population, operators, stats) while keeping generic traits and adding minimal validation/testing.

## Requirements

- Centralize TrainingStats usage to training_setup version and remove inline duplicate.
- Add EvolutionConfig with core hyperparameters and validation (pop size, generations, rates, stagnation, target fitness).
- Implement Population<G> lifecycle (init with mutation-based diversity, size validation, replacement/elitism hooks).
- Implement GeneticOperators<M, C> wrapping mutation/crossover and probability logic.
- Refactor EvolutionaryTrainer to compose new pieces and drive the generation loop.
- Add a minimal test/harness covering population size, stats updates, and convergence.

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

- New EvolutionConfig struct with validated hyperparameters and accessors.
- Population<G> owns Vec<G> plus initialization/replacement methods.
- GeneticOperators<M, C> bundles operators and rates, exposes offspring generation.
- Trainer trait returns training_setup::TrainingStats; inline ConvergenceStatus/TrainingStats removed or adapted.
- EvolutionaryTrainer fields updated to use the new structs and stats.

## Action items

[ ] Align TrainingStats usage: adopt training_setup::TrainingStats in traits/trainer.rs and remove/adapter inline version; ensure ConvergenceStatus coherence.
[ ] Implement EvolutionConfig with population size, max generations, mutation/crossover rates, stagnation limit, optional target fitness; include validation helpers.
[ ] Implement Population<G>: store genomes, initialize from template + mutation for diversity, validate size, expose replacement/elitism hooks (e.g., set_population, inject_elite).
[ ] Implement GeneticOperators<M, C>: hold mutation/crossover + rates, generate offspring given parents and probabilities; validate rate ranges.
[ ] Refactor EvolutionaryTrainer<G,F,S,M,C> to compose Population, GeneticOperators, EvolutionConfig, TrainingStats; wire generation loop: init population, evaluate via F, record FitnessStats/TrainingStats, select via S, generate offspring, replace population, check convergence/limits.
[ ] Add minimal test/harness (e.g., under vynapse-core/tests) that runs a tiny population for one generation to assert population size, stats update, and convergence triggers on limits.

## Testing and validation

- Run `cargo test` (focus on new test module/harness).
- Optional manual harness run to ensure no panics and stats progress.

## Risks and edge cases

- Generic bounds mismatched across Population/GeneticOperators/Trainer causing trait impl issues.
- Duplicate or conflicting TrainingStats definitions if inline version not fully removed.
- Mutation/crossover rates or population size misvalidated, leading to runtime panics.

## Open questions

- Should Trainer::get_stats return owned or reference to avoid copies?
- Do we enforce deterministic seeding for initialization/mutation in the MVP test?
