use rand::{Rng, rng};

use crate::{Result, VynapseError};

pub struct TournamentSelection {
    tournament_size: usize,
}

impl TournamentSelection {
    pub fn new(tournament_size: usize) -> Result<Self> {
        if tournament_size < 1 {
            return Err(VynapseError::EvolutionError(
                "Tournament size cannot be less than 1.".to_string(),
            ));
        }

        Ok(TournamentSelection { tournament_size })
    }

    fn run_single_tournament(&self, fitness: &[f32]) -> Result<usize> {
        if fitness.is_empty() {
            return Err(VynapseError::EvolutionError(
                "Tournament selection cannot be run on an empty fitness array.".to_string(),
            ));
        }

        if self.tournament_size > fitness.len() {
            return Err(VynapseError::EvolutionError(
                "The tournament size cannot be larger than the fitness array.".to_string(),
            ));
        }

        let mut rng = rng();
        let mut tournament_participants: Vec<usize> = Vec::new();

        for _ in 0..self.tournament_size {
            let tournament_participant: usize = rng.random_range(0..fitness.len());
            tournament_participants.push(tournament_participant);
        }

        let mut current_best_participant: usize = tournament_participants[0];

        for i in 1..tournament_participants.len() {
            if fitness[tournament_participants[i]] > fitness[current_best_participant] {
                current_best_participant = tournament_participants[i];
            }
        }

        Ok(current_best_participant)
    }
}

#[cfg(test)]
#[test]
fn test_tournament_selection_creation() {
    let selection = TournamentSelection::new(3).unwrap();
    assert_eq!(selection.tournament_size, 3);
}

#[test]
fn test_tournament_selection_invalid_size() {
    let result = TournamentSelection::new(0);
    assert!(result.is_err());
}

#[test]
fn test_single_tournament_clear_winner() {
    let selection = TournamentSelection::new(2).unwrap();
    let fitness = vec![0.1, 0.9, 0.3, 0.2]; // index 1 has highest fitness

    // Run tournament many times - index 1 should win most often
    let mut wins_for_best = 0;
    let trials = 1000;

    for _ in 0..trials {
        let winner = selection.run_single_tournament(&fitness).unwrap();
        if winner == 1 {
            wins_for_best += 1;
        }
    }

    // Index 1 should win significantly more than random chance (25%)
    assert!(wins_for_best > trials / 3); // Should win > 50% of time
}

#[test]
fn test_single_tournament_large_tournament_bias() {
    let selection = TournamentSelection::new(4).unwrap(); // Full population tournament
    let fitness = vec![0.1, 0.9, 0.3, 0.2];

    // With tournament size = population size, best individual should win MOST of the time
    let mut wins_for_best = 0;
    let trials = 1000;

    for _ in 0..trials {
        let winner = selection.run_single_tournament(&fitness).unwrap();
        if winner == 1 {
            // Index 1 has fitness 0.9 (highest)
            wins_for_best += 1;
        }
    }
    println!(
        "Best individual won {}/{} times ({}%)",
        wins_for_best,
        trials,
        wins_for_best * 100 / trials
    );

    assert!(wins_for_best > trials * 65 / 100); // Should win >80% of time
}

#[test]
fn test_single_tournament_size_one() {
    let selection = TournamentSelection::new(1).unwrap();
    let fitness = vec![0.1, 0.9, 0.3, 0.2];

    // With tournament size 1, any individual can win (pure random)
    let mut seen_indices = std::collections::HashSet::new();

    for _ in 0..100 {
        let winner = selection.run_single_tournament(&fitness).unwrap();
        assert!(winner < fitness.len()); // Valid index
        seen_indices.insert(winner);
    }

    // Should see multiple different winners (not deterministic)
    assert!(seen_indices.len() > 1);
}

#[test]
fn test_single_tournament_empty_fitness() {
    let selection = TournamentSelection::new(2).unwrap();
    let fitness = vec![];

    let result = selection.run_single_tournament(&fitness);
    assert!(result.is_err());
}

#[test]
fn test_single_tournament_size_larger_than_population() {
    let selection = TournamentSelection::new(10).unwrap();
    let fitness = vec![0.1, 0.9, 0.3]; // Only 3 individuals

    let result = selection.run_single_tournament(&fitness);
    assert!(result.is_err()); // Should fail with your current validation
}

#[test]
fn test_single_tournament_negative_fitness() {
    let selection = TournamentSelection::new(3).unwrap();
    let fitness = vec![-0.5, -0.1, -0.8, -0.2]; // All negative, index 1 is best (-0.1)

    let mut wins_for_best = 0;
    let trials = 1000;

    for _ in 0..trials {
        let winner = selection.run_single_tournament(&fitness).unwrap();
        if winner == 1 {
            wins_for_best += 1;
        }
    }

    // Index 1 (-0.1) should win most often
    assert!(wins_for_best > trials / 3);
}

#[test]
fn test_single_tournament_identical_fitness() {
    let selection = TournamentSelection::new(2).unwrap();
    let fitness = vec![0.5, 0.5, 0.5, 0.5]; // All identical

    let mut seen_indices = std::collections::HashSet::new();

    for _ in 0..100 {
        let winner = selection.run_single_tournament(&fitness).unwrap();
        assert!(winner < fitness.len());
        seen_indices.insert(winner);
    }

    // With identical fitness, should see random distribution
    assert!(seen_indices.len() > 1);
}

#[test]
fn test_single_tournament_single_individual() {
    let selection = TournamentSelection::new(1).unwrap();
    let fitness = vec![0.7]; // Only one individual

    for _ in 0..10 {
        let winner = selection.run_single_tournament(&fitness).unwrap();
        assert_eq!(winner, 0); // Only possible winner
    }
}
