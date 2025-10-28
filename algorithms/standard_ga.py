"""
Standard GA variant for Emergency Vehicle Dispatch and Routing Problem (EVDRP).

This implements the classic genetic algorithm with:
- Random population initialization
- Tournament selection
- Order crossover (OX) for routing
- Swap mutation
- Elitist replacement (keeping best solution)
"""
import time
import random
import numpy as np
from typing import List, Dict, Tuple, Any


class StandardGA:
    """Standard Genetic Algorithm for EVDRP."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize GA with configuration.
        
        Args:
            config: Configuration dict with:
                - population_size: Number of individuals
                - generations: Number of generations
                - crossover_rate: Probability of crossover (0-1)
                - mutation_rate: Probability of mutation (0-1)
                - tournament_size: Size of tournament selection
                - seed: Random seed for reproducibility
        """
        self.pop_size = config.get('population_size', 100)
        self.generations = config.get('generations', 200)
        self.crossover_rate = config.get('crossover_rate', 0.8)
        self.mutation_rate = config.get('mutation_rate', 0.1)
        self.tournament_size = config.get('tournament_size', 5)
        self.seed = config.get('seed', 42)
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []
    
    def create_individual(self, num_locations: int) -> List[int]:
        """Create a random route (permutation of locations).
        
        Args:
            num_locations: Number of emergency locations to visit
            
        Returns:
            List of location indices in visit order
        """
        individual = list(range(num_locations))
        random.shuffle(individual)
        return individual
    
    def initialize_population(self, num_locations: int) -> List[List[int]]:
        """Create initial population of random routes.
        
        Args:
            num_locations: Number of emergency locations
            
        Returns:
            List of individuals (routes)
        """
        return [self.create_individual(num_locations) for _ in range(self.pop_size)]
    
    def evaluate_fitness(self, individual: List[int], problem: Dict) -> float:
        """Evaluate fitness of a route.
        
        Fitness = total travel time from depot through all emergencies and back.
        Lower is better.
        
        Args:
            individual: Route (list of location indices)
            problem: Problem instance with depot, locations, distance matrix
            
        Returns:
            Fitness value (total time)
        """
        if not problem or 'distance_matrix' not in problem:
            # Placeholder: random fitness for testing
            return random.uniform(100, 1000)
        
        distance_matrix = problem['distance_matrix']
        depot = problem.get('depot_idx', 0)
        
        total_time = 0.0
        
        # Start from depot to first location
        current = depot
        for next_loc in individual:
            total_time += distance_matrix[current][next_loc]
            current = next_loc
        
        # Return to depot
        total_time += distance_matrix[current][depot]
        
        return total_time
    
    def tournament_selection(self, population: List[List[int]], 
                            fitness_values: List[float]) -> List[int]:
        """Select an individual using tournament selection.
        
        Args:
            population: List of individuals
            fitness_values: Fitness for each individual
            
        Returns:
            Selected individual
        """
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [(idx, fitness_values[idx]) for idx in tournament_indices]
        winner_idx = min(tournament_fitness, key=lambda x: x[1])[0]
        return population[winner_idx].copy()
    
    def order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Order crossover (OX) operator for permutation encoding.
        
        Args:
            parent1: First parent route
            parent2: Second parent route
            
        Returns:
            Two offspring routes
        """
        size = len(parent1)
        
        # Select two random crossover points
        cx_point1, cx_point2 = sorted(random.sample(range(size), 2))
        
        # Create offspring 1
        offspring1 = [-1] * size
        offspring1[cx_point1:cx_point2] = parent1[cx_point1:cx_point2]
        
        # Fill remaining positions from parent2
        remaining = [x for x in parent2 if x not in offspring1]
        idx = 0
        for i in range(size):
            if offspring1[i] == -1:
                offspring1[i] = remaining[idx]
                idx += 1
        
        # Create offspring 2
        offspring2 = [-1] * size
        offspring2[cx_point1:cx_point2] = parent2[cx_point1:cx_point2]
        
        remaining = [x for x in parent1 if x not in offspring2]
        idx = 0
        for i in range(size):
            if offspring2[i] == -1:
                offspring2[i] = remaining[idx]
                idx += 1
        
        return offspring1, offspring2
    
    def swap_mutation(self, individual: List[int]) -> List[int]:
        """Swap mutation: randomly swap two positions.
        
        Args:
            individual: Route to mutate
            
        Returns:
            Mutated route
        """
        mutated = individual.copy()
        if len(mutated) > 1:
            idx1, idx2 = random.sample(range(len(mutated)), 2)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        return mutated
    
    def evolve(self, problem: Dict) -> Dict:
        """Run the genetic algorithm.
        
        Args:
            problem: Problem instance with emergency locations and distance info
            
        Returns:
            Results dict with best solution, fitness, and history
        """
        # Determine problem size
        if problem and 'num_emergencies' in problem:
            num_locations = problem['num_emergencies']
        else:
            num_locations = 10  # Default for placeholder
        
        # Initialize population
        population = self.initialize_population(num_locations)
        
        # Evolution loop
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_values = [self.evaluate_fitness(ind, problem) for ind in population]
            
            # Track best solution
            gen_best_idx = np.argmin(fitness_values)
            gen_best_fitness = fitness_values[gen_best_idx]
            
            if gen_best_fitness < self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_solution = population[gen_best_idx].copy()
            
            self.fitness_history.append(self.best_fitness)
            
            # Create new population
            new_population = [self.best_solution.copy()]  # Elitism: keep best
            
            while len(new_population) < self.pop_size:
                # Selection
                parent1 = self.tournament_selection(population, fitness_values)
                parent2 = self.tournament_selection(population, fitness_values)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    offspring1, offspring2 = self.order_crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if random.random() < self.mutation_rate:
                    offspring1 = self.swap_mutation(offspring1)
                if random.random() < self.mutation_rate:
                    offspring2 = self.swap_mutation(offspring2)
                
                new_population.append(offspring1)
                if len(new_population) < self.pop_size:
                    new_population.append(offspring2)
            
            population = new_population[:self.pop_size]
        
        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'final_population': population
        }


def run(problem=None, config=None):
    """Run Standard GA on the Emergency Vehicle Dispatch and Routing Problem.

    Args:
        problem: Problem instance dict with:
            - num_emergencies: Number of emergency locations
            - distance_matrix: NxN matrix of travel times (optional)
            - depot_idx: Index of depot location (optional, default 0)
        config: Configuration dict with GA parameters:
            - population_size: Population size (default 100)
            - generations: Number of generations (default 200)
            - crossover_rate: Crossover probability (default 0.8)
            - mutation_rate: Mutation probability (default 0.1)
            - tournament_size: Tournament size (default 5)

    Returns:
        dict: Results with algorithm name, objective value, time, and status
    """
    start = time.time()
    
    # Default config
    if config is None:
        config = {}
    
    config.setdefault('population_size', 100)
    config.setdefault('generations', 200)
    config.setdefault('crossover_rate', 0.8)
    config.setdefault('mutation_rate', 0.1)
    config.setdefault('tournament_size', 5)
    config.setdefault('seed', 42)
    
    # Run GA
    ga = StandardGA(config)
    result = ga.evolve(problem)
    
    end = time.time()
    
    return {
        "algorithm": "standard_ga",
        "objective": result['best_fitness'],
        "best_solution": result['best_solution'],
        "convergence": result['fitness_history'][-10:],  # Last 10 generations
        "time_s": end - start,
        "generations": config['generations'],
        "population_size": config['population_size'],
        "status": "success"
    }
