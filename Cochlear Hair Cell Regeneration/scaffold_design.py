import numpy as np

class ScaffoldGA:
    def __init__(self, pop_size=20, grid_size=8, mutation_rate=0.1):
        self.pop_size = pop_size
        self.grid_size = grid_size
        self.mutation_rate = mutation_rate
        self.population = self._init_population()

    def _init_population(self):
        # Each scaffold is a 3‑D grid (0: void, 1: material) of shape (pop_size, grid_size, grid_size, grid_size)
        return np.random.randint(0, 2, (self.pop_size, self.grid_size, self.grid_size, self.grid_size))

    def fitness(self, scaffold):
        # Example fitness: (porosity) * (connectivity)  
        porosity = 1.0 – np.mean(scaffold)  
        # (Simplified “connectivity” – sum of material voxels divided by total voxels)  
        connectivity = np.sum(scaffold) / (self.grid_size ** 3)  
        return porosity * connectivity

    def select(self):
        scores = np.array([self.fitness(s) for s in self.population])
        idx = np.argsort(scores)[-self.pop_size//2:]
        return self.population[idx]

    def crossover(self, parent1, parent2):
        mask = np.random.rand(*parent1.shape) > 0.5
        child = np.where(mask, parent1, parent2)
        return child

    def mutate(self, scaffold):
        mask = np.random.rand(*scaffold.shape) < self.mutation_rate
        scaffold[mask] = 1 – scaffold[mask]
        return scaffold

    def evolve(self):
        selected = self.select()
        children = []
        for _ in range(self.pop_size):
            p1, p2 = selected[np.random.randint(len(selected), size=2)]
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            children.append(child)
        self.population = np.array(children)
