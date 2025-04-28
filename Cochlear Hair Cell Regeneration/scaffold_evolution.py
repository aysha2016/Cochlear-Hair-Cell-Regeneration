import numpy as np
import matplotlib.pyplot as plt
import random

def generate_scaffold(shape=(10, 10)):
    return np.random.choice([0, 1], size=shape)

def calculate_fitness(scaffold):
    mass = np.sum(scaffold)
    stiffness = np.sum(np.abs(np.gradient(scaffold)[0])) + 1
    frequency = (stiffness / mass) * 50
    target_frequency = 100
    return -abs(frequency - target_frequency)

def mutate(scaffold, rate=0.1):
    new = scaffold.copy()
    for i in range(scaffold.shape[0]):
        for j in range(scaffold.shape[1]):
            if random.random() < rate:
                new[i, j] = 1 - scaffold[i, j]
    return new

def evolve_scaffolds(generations=50, pop_size=20):
    population = [generate_scaffold() for _ in range(pop_size)]
    for generation in range(generations):
        fitness = [calculate_fitness(ind) for ind in population]
        best_idx = np.argmax(fitness)
        best = population[best_idx]
        new_population = [best]
        for _ in range(pop_size - 1):
            child = mutate(best, rate=0.05)
            new_population.append(child)
        population = new_population
        print(f"Gen {generation} Best Fitness: {fitness[best_idx]:.2f}")
    return best

if __name__ == "__main__":
    best_scaffold = evolve_scaffolds()
    plt.imshow(best_scaffold, cmap='gray')
    plt.title('Best Evolved Scaffold')
    plt.show()
