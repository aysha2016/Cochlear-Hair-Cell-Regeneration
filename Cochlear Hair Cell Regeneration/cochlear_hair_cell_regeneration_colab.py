#@title Cochlear Hair Cell Regeneration System
#@markdown This notebook implements a prototype system for cochlear hair cell regeneration using AI.

# Install required packages
!pip install torch numpy matplotlib scipy

# Import required libraries
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import signal
from typing import List, Tuple
import random

#@title Scaffold Design Module
#@markdown This module implements a genetic algorithm to evolve 3D biomaterial scaffolds.

class ScaffoldGA:
    def __init__(self, grid_size: int = 10, population_size: int = 20, mutation_rate: float = 0.1):
        self.grid_size = grid_size
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self._initialize_population()
    
    def _initialize_population(self) -> List[np.ndarray]:
        # Initialize a population of random 3D scaffolds
        return [np.random.choice([0, 1], size=(self.grid_size, self.grid_size, self.grid_size), p=[0.7, 0.3]) 
                for _ in range(self.population_size)]
    
    def _calculate_fitness(self, scaffold: np.ndarray) -> float:
        # Calculate fitness based on porosity and connectivity
        porosity = 1 - np.mean(scaffold)  # Fraction of void space
        connectivity = np.mean(scaffold)   # Fraction of material
        # Penalize scaffolds that are too dense or too sparse
        if porosity < 0.3 or porosity > 0.7:
            return 0.0
        return porosity * connectivity
    
    def _select_parents(self) -> Tuple[np.ndarray, np.ndarray]:
        # Select two parents based on fitness
        fitnesses = [self._calculate_fitness(scaffold) for scaffold in self.population]
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            return random.sample(self.population, 2)
        probs = [f / total_fitness for f in fitnesses]
        parents = np.random.choice(self.population, size=2, p=probs, replace=False)
        return parents[0], parents[1]
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        # Perform crossover by randomly selecting slices from each parent
        child = np.zeros_like(parent1)
        for i in range(self.grid_size):
            if random.random() < 0.5:
                child[i, :, :] = parent1[i, :, :]
            else:
                child[i, :, :] = parent2[i, :, :]
        return child
    
    def _mutate(self, scaffold: np.ndarray) -> np.ndarray:
        # Randomly flip bits with probability mutation_rate
        mask = np.random.random(scaffold.shape) < self.mutation_rate
        scaffold[mask] = 1 - scaffold[mask]
        return scaffold
    
    def evolve(self, num_generations: int = 100) -> np.ndarray:
        for generation in range(num_generations):
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self._select_parents()
                child1 = self._mutate(self._crossover(parent1, parent2))
                child2 = self._mutate(self._crossover(parent2, parent1))
                new_population.extend([child1, child2])
            self.population = new_population
            
            # Print progress
            if (generation + 1) % 10 == 0:
                best_fitness = max(self._calculate_fitness(scaffold) for scaffold in self.population)
                print(f"Generation {generation + 1}, Best Fitness: {best_fitness:.4f}")
        
        # Return the best scaffold
        return max(self.population, key=self._calculate_fitness)

#@title Mechanical Resonance Simulation
#@markdown This module simulates the mechanical resonance of the scaffold.

def simulate_resonance(scaffold: np.ndarray, freq_range: Tuple[float, float] = (20, 20000)) -> Tuple[np.ndarray, np.ndarray]:
    # Simplified resonance simulation based on scaffold properties
    # In a real system, this would use a finite-element solver
    
    # Calculate scaffold properties
    density = np.mean(scaffold)
    connectivity = np.sum(scaffold) / scaffold.size
    
    # Generate frequency range (log scale for audio frequencies)
    freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 1000)
    
    # Simplified resonance model
    # Resonance frequency is inversely proportional to density and directly proportional to connectivity
    resonance_freq = 1000 * (1 - density) * connectivity
    
    # Generate resonance response
    response = 1 / (1 + ((freqs - resonance_freq) / (resonance_freq * 0.1)) ** 2)
    
    return freqs, response

#@title Cell Response Prediction Model
#@markdown This module uses a neural network to predict cell behavior.

class CellResponseNet(nn.Module):
    def __init__(self, input_size: int = 3):
        super(CellResponseNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Output: [differentiation_score, survival_score]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def extract_scaffold_features(scaffold: np.ndarray, resonance_freq: float) -> torch.Tensor:
    # Extract relevant features from the scaffold and resonance
    porosity = 1 - np.mean(scaffold)
    connectivity = np.mean(scaffold)
    features = torch.tensor([porosity, connectivity, resonance_freq], dtype=torch.float32)
    return features.unsqueeze(0)  # Add batch dimension

#@title Visualization
#@markdown This module provides functions to visualize the results.

def plot_scaffold(scaffold: np.ndarray):
    plt.figure(figsize=(10, 10))
    # Plot a 2D slice of the 3D scaffold
    plt.imshow(scaffold[:, :, scaffold.shape[2]//2], cmap='binary')
    plt.title('Scaffold Structure (Middle Slice)')
    plt.colorbar(label='Material Density')
    plt.show()

def plot_resonance(freqs: np.ndarray, response: np.ndarray):
    plt.figure(figsize=(10, 6))
    plt.semilogx(freqs, 20 * np.log10(response))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Response (dB)')
    plt.title('Scaffold Resonance Response')
    plt.grid(True)
    plt.show()

def plot_cell_response(differentiation_score: float, survival_score: float):
    plt.figure(figsize=(8, 6))
    scores = [differentiation_score, survival_score]
    labels = ['Differentiation', 'Survival']
    plt.bar(labels, scores)
    plt.ylim(0, 1)
    plt.title('Predicted Cell Response')
    plt.ylabel('Score')
    plt.show()

#@title Main Simulation
#@markdown Run the complete simulation to see how it works.

def main():
    # Initialize the genetic algorithm
    ga = ScaffoldGA(grid_size=10, population_size=20, mutation_rate=0.1)
    
    # Evolve the scaffold
    print("Evolving scaffold...")
    best_scaffold = ga.evolve(num_generations=50)
    
    # Simulate resonance
    print("\nSimulating resonance...")
    freqs, response = simulate_resonance(best_scaffold)
    
    # Initialize cell response model
    model = CellResponseNet()
    
    # Extract features and predict cell response
    print("\nPredicting cell response...")
    features = extract_scaffold_features(best_scaffold, freqs[np.argmax(response)])
    with torch.no_grad():
        cell_response = model(features)
    differentiation_score, survival_score = cell_response[0].numpy()
    
    # Visualize results
    print("\nVisualizing results...")
    plot_scaffold(best_scaffold)
    plot_resonance(freqs, response)
    plot_cell_response(differentiation_score, survival_score)

# Run the simulation
if __name__ == "__main__":
    main() 