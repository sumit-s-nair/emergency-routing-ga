"""
Hybrid GA variant for EVDRP.

This algorithm combines Genetic Algorithm with Local Search optimization.
The hybrid approach uses 2-opt local search to improve solutions found by GA,
providing better exploitation of the search space while maintaining GA's exploration.

Key Features:
- Combines GA with 2-opt local search for route improvement
- Local search applied to elite individuals and best offspring
- Adaptive local search intensity based on generation
- Maintains population diversity while improving solution quality
- Support for both real OSM datasets and synthetic problems

Reference:
Kumar, R., & Singh, L. (2017). Hybrid genetic algorithm for vehicle routing problem.
International Journal of Computer Applications, 175(22), 7-12.
"""
import time
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional


class HybridGA:
    """
    Hybrid Genetic Algorithm combining GA with 2-opt local search for EVDRP.
    
    The algorithm uses standard GA operators (selection, crossover, mutation) 
    combined with 2-opt local search to refine solutions.
    """
    
    def __init__(self, problem: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize the Hybrid GA.
        
        Args:
            problem: Dict containing problem data (same format as ElitistGA)
            config: Dict containing:
                - All ElitistGA parameters
                - 'local_search_rate': Probability of applying local search (default: 0.3)
                - 'local_search_intensity': Max iterations for 2-opt (default: 50)
                - 'adaptive_ls': Whether to adapt LS intensity (default: True)
        """
        self.problem = problem
        self.config = config
        self.dataset_name = problem.get('dataset_name', 'unknown')
        
        # Load problem data
        if 'edges_df' in problem and problem['edges_df'] is not None:
            self._load_from_dataframe(problem)
        else:
            self.nodes = problem.get('nodes', list(range(10)))
            self.distances = problem.get('distances', self._generate_random_distances())
            self.traffic_weights = problem.get('traffic_weights', self._generate_random_traffic())
            self.start_node = problem.get('start_node', self.nodes[0])
            self.target_nodes = problem.get('target_nodes', self.nodes[1:])
        
        # GA parameters
        self.population_size = config.get('population_size', 100)
        self.generations = config.get('generations', 200)
        self.elite_size = config.get('elite_size', 0.1)
        self.tournament_size = config.get('tournament_size', 5)
        self.crossover_rate = config.get('crossover_rate', 0.8)
        self.mutation_rate = config.get('mutation_rate', 0.2)
        self.distance_weight = config.get('distance_weight', 0.5)
        self.time_weight = config.get('time_weight', 0.5)
        
        # Hybrid-specific parameters
        self.local_search_rate = config.get('local_search_rate', 0.3)
        self.local_search_intensity = config.get('local_search_intensity', 50)
        self.adaptive_ls = config.get('adaptive_ls', True)
        
        # Calculate elite count
        if isinstance(self.elite_size, float):
            self.elite_count = max(1, int(self.population_size * self.elite_size))
        else:
            self.elite_count = max(1, self.elite_size)
        
        # State
        self.population = []
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_history = []
        self.ls_applications = 0  # Track local search applications
        self.ls_improvements = 0  # Track successful improvements
    
    def _load_from_dataframe(self, problem: Dict[str, Any]) -> None:
        """Load problem data from DataFrame format."""
        edges_df = problem['edges_df']
        nodes_df = problem.get('nodes_df', None)
        
        all_nodes = set(edges_df['from_node'].unique()) | set(edges_df['to_node'].unique())
        self.nodes = sorted(list(all_nodes))
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        n = len(self.nodes)
        self.distances = np.full((n, n), float('inf'))
        self.traffic_weights = np.ones((n, n))
        self.adjacency = {i: set() for i in range(n)}
        
        for _, edge in edges_df.iterrows():
            from_idx = self.node_to_idx[edge['from_node']]
            to_idx = self.node_to_idx[edge['to_node']]
            
            self.distances[from_idx][to_idx] = edge['distance_km']
            self.traffic_weights[from_idx][to_idx] = edge['traffic_weight']
            self.distances[to_idx][from_idx] = edge['distance_km']
            self.traffic_weights[to_idx][from_idx] = edge['traffic_weight']
            
            self.adjacency[from_idx].add(to_idx)
            self.adjacency[to_idx].add(from_idx)
        
        np.fill_diagonal(self.distances, 0)
        
        print(f"  Computing shortest paths between all node pairs...")
        self._compute_shortest_paths()
        print(f"  Shortest paths computed")
        
        # Find largest connected component
        visited = set()
        components = []
        
        def dfs(node):
            component = []
            stack = [node]
            while stack:
                curr = stack.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                component.append(curr)
                stack.extend(self.adjacency[curr] - visited)
            return component
        
        for node in range(n):
            if node not in visited:
                comp = dfs(node)
                components.append(comp)
        
        largest_component = max(components, key=len)
        print(f"  Using largest connected component: {len(largest_component)} nodes "
              f"(out of {n} total)")
        
        # Set start and targets
        if 'start_node' in problem and problem['start_node'] in self.node_to_idx:
            start = problem['start_node']
            self.start_node = self.node_to_idx[start]
            if self.start_node not in largest_component:
                self.start_node = random.choice(largest_component)
        else:
            self.start_node = random.choice(largest_component)
        
        if 'target_nodes' in problem:
            targets = problem['target_nodes']
            self.target_nodes = [self.node_to_idx[t] for t in targets 
                               if t in self.node_to_idx and self.node_to_idx[t] in largest_component]
            if not self.target_nodes:
                num_targets = min(10, len(largest_component) - 1)
                available = [i for i in largest_component if i != self.start_node]
                self.target_nodes = random.sample(available, min(num_targets, len(available)))
        else:
            num_targets = min(10, len(largest_component) - 1)
            available = [i for i in largest_component if i != self.start_node]
            self.target_nodes = random.sample(available, min(num_targets, len(available)))
    
    def _generate_random_distances(self) -> np.ndarray:
        """Generate random distance matrix."""
        n = len(self.nodes)
        dist = np.random.uniform(0.5, 10.0, (n, n))
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        return dist
    
    def _generate_random_traffic(self) -> np.ndarray:
        """Generate random traffic weight matrix."""
        n = len(self.nodes)
        traffic = np.random.uniform(1.0, 5.0, (n, n))
        traffic = (traffic + traffic.T) / 2
        np.fill_diagonal(traffic, 1)
        return traffic
    
    def _compute_shortest_paths(self) -> None:
        """Compute all-pairs shortest paths using Floyd-Warshall."""
        n = len(self.nodes)
        dist = self.distances.copy()
        traffic = self.traffic_weights.copy()
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        traffic[i][j] = (traffic[i][k] + traffic[k][j]) / 2
        
        self.distances = dist
        self.traffic_weights = traffic
    
    def calculate_travel_time(self, distance: float, traffic_weight: float) -> float:
        """Calculate travel time based on distance and traffic."""
        return distance * (1 + 0.2 * traffic_weight)
    
    def calculate_fitness(self, route: List[int]) -> float:
        """Calculate fitness of a route (lower is better)."""
        if len(route) < 2:
            return float('inf')
        
        total_distance = 0.0
        total_time = 0.0
        
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            distance = self.distances[from_node][to_node]
            
            if distance == float('inf'):
                return float('inf')
            
            traffic = self.traffic_weights[from_node][to_node]
            travel_time = self.calculate_travel_time(distance, traffic)
            
            total_distance += distance
            total_time += travel_time
        
        fitness = self.distance_weight * total_distance + self.time_weight * total_time
        return fitness
    
    def two_opt_local_search(self, route: List[int], max_iterations: int = 50) -> List[int]:
        """
        Apply 2-opt local search to improve a route.
        
        2-opt swaps two edges in the route to find improvements.
        For route [a, b, c, d, e], a 2-opt move reverses a subsequence,
        e.g., [a, b, c, d, e] -> [a, c, b, d, e]
        
        Args:
            route: Route to improve
            max_iterations: Maximum number of iterations
            
        Returns:
            Improved route
        """
        improved_route = route.copy()
        best_fitness = self.calculate_fitness(improved_route)
        improved = False
        
        iterations = 0
        while iterations < max_iterations:
            local_improved = False
            
            # Try all possible 2-opt swaps (excluding start node)
            for i in range(1, len(improved_route) - 2):
                for j in range(i + 1, len(improved_route)):
                    # Create new route by reversing segment [i:j]
                    new_route = improved_route.copy()
                    new_route[i:j] = reversed(new_route[i:j])
                    
                    # Evaluate new route
                    new_fitness = self.calculate_fitness(new_route)
                    
                    if new_fitness < best_fitness:
                        improved_route = new_route
                        best_fitness = new_fitness
                        local_improved = True
                        improved = True
                        break
                
                if local_improved:
                    break
            
            # If no improvement found, stop
            if not local_improved:
                break
            
            iterations += 1
        
        if improved:
            self.ls_improvements += 1
        
        return improved_route
    
    def create_initial_population(self) -> List[List[int]]:
        """Create initial population using multiple strategies."""
        population = []
        
        # Nearest Neighbor (30%)
        nn_count = int(0.3 * self.population_size)
        for _ in range(nn_count):
            route = self._nearest_neighbor_route()
            population.append(route)
        
        # Random Insertion (40%)
        ri_count = int(0.4 * self.population_size)
        for _ in range(ri_count):
            route = self._random_insertion_route()
            population.append(route)
        
        # Random Permutation (remaining)
        remaining = self.population_size - len(population)
        for _ in range(remaining):
            route = [self.start_node] + random.sample(self.target_nodes, len(self.target_nodes))
            population.append(route)
        
        # Apply local search to some initial solutions
        ls_count = int(0.2 * self.population_size)
        for i in range(ls_count):
            population[i] = self.two_opt_local_search(population[i], max_iterations=20)
            self.ls_applications += 1
        
        return population
    
    def _nearest_neighbor_route(self) -> List[int]:
        """Create route using nearest neighbor heuristic."""
        route = [self.start_node]
        remaining = set(self.target_nodes)
        current = self.start_node
        
        while remaining:
            nearest = min(remaining, key=lambda node: self.distances[current][node])
            route.append(nearest)
            remaining.remove(nearest)
            current = nearest
        
        return route
    
    def _random_insertion_route(self) -> List[int]:
        """Create route using random insertion heuristic."""
        route = [self.start_node]
        remaining = list(self.target_nodes)
        random.shuffle(remaining)
        
        for node in remaining:
            best_pos = 1
            best_cost = float('inf')
            
            for pos in range(1, len(route) + 1):
                if pos == len(route):
                    cost_increase = self.distances[route[pos-1]][node]
                else:
                    old_cost = self.distances[route[pos-1]][route[pos]]
                    new_cost = (self.distances[route[pos-1]][node] + 
                               self.distances[node][route[pos]])
                    cost_increase = new_cost - old_cost
                
                if cost_increase < best_cost:
                    best_cost = cost_increase
                    best_pos = pos
            
            route.insert(best_pos, node)
        
        return route
    
    def tournament_selection(self, population: List[List[int]], 
                            fitness_values: List[float]) -> List[int]:
        """Select parent using tournament selection."""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()
    
    def order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Perform Order Crossover (OX)."""
        size = len(parent1)
        
        if size <= 2:
            return parent1.copy(), parent2.copy()
        
        start_idx = random.randint(1, size - 2)
        end_idx = random.randint(start_idx + 1, size - 1)
        
        offspring1 = [None] * size
        offspring2 = [None] * size
        
        offspring1[0] = parent1[0]
        offspring2[0] = parent2[0]
        
        offspring1[start_idx:end_idx] = parent1[start_idx:end_idx]
        offspring2[start_idx:end_idx] = parent2[start_idx:end_idx]
        
        self._fill_offspring(offspring1, parent2, start_idx, end_idx)
        self._fill_offspring(offspring2, parent1, start_idx, end_idx)
        
        return offspring1, offspring2
    
    def _fill_offspring(self, offspring: List[int], parent: List[int], 
                       start: int, end: int) -> None:
        """Helper to fill offspring after crossover."""
        offspring_set = set(n for n in offspring if n is not None)
        parent_idx = 1
        offspring_idx = 1
        
        while None in offspring:
            if offspring_idx == start:
                offspring_idx = end
            
            if offspring_idx >= len(offspring):
                break
            
            while parent_idx < len(parent) and parent[parent_idx] in offspring_set:
                parent_idx += 1
            
            if parent_idx < len(parent):
                offspring[offspring_idx] = parent[parent_idx]
                offspring_set.add(parent[parent_idx])
                parent_idx += 1
            
            offspring_idx += 1
    
    def swap_mutation(self, route: List[int]) -> List[int]:
        """Perform swap mutation."""
        mutated = route.copy()
        
        if len(mutated) <= 2:
            return mutated
        
        idx1, idx2 = random.sample(range(1, len(mutated)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        return mutated
    
    def evolve_population(self, population: List[List[int]], 
                         fitness_values: List[float],
                         generation: int) -> List[List[int]]:
        """
        Evolve population with hybrid approach.
        
        Combines GA operators with local search applied selectively.
        """
        # Sort population
        sorted_indices = np.argsort(fitness_values)
        sorted_population = [population[i] for i in sorted_indices]
        
        # Elitism with local search on elite
        elite = []
        for i in range(self.elite_count):
            route = sorted_population[i].copy()
            # Apply local search to elite with high probability
            if random.random() < 0.5:
                route = self.two_opt_local_search(route, self.local_search_intensity)
                self.ls_applications += 1
            elite.append(route)
        
        # Generate offspring
        offspring = []
        offspring_needed = self.population_size - self.elite_count
        
        # Adaptive local search intensity
        if self.adaptive_ls:
            ls_intensity = max(20, self.local_search_intensity - (generation // 10))
        else:
            ls_intensity = self.local_search_intensity
        
        while len(offspring) < offspring_needed:
            # Selection
            parent1 = self.tournament_selection(population, fitness_values)
            parent2 = self.tournament_selection(population, fitness_values)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self.order_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < self.mutation_rate:
                child1 = self.swap_mutation(child1)
            if random.random() < self.mutation_rate:
                child2 = self.swap_mutation(child2)
            
            # Apply local search with probability
            if random.random() < self.local_search_rate:
                child1 = self.two_opt_local_search(child1, ls_intensity)
                self.ls_applications += 1
            if random.random() < self.local_search_rate:
                child2 = self.two_opt_local_search(child2, ls_intensity)
                self.ls_applications += 1
            
            offspring.extend([child1, child2])
        
        new_population = elite + offspring[:offspring_needed]
        return new_population
    
    def run_algorithm(self) -> Dict[str, Any]:
        """Run the Hybrid GA algorithm."""
        start_time = time.time()
        
        # Initialize
        self.population = self.create_initial_population()
        population_diversity_history = []
        
        # Evolution loop
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_values = [self.calculate_fitness(route) for route in self.population]
            
            # Filter invalid solutions
            valid_indices = [i for i, f in enumerate(fitness_values) if f != float('inf')]
            
            if not valid_indices:
                self.population = self.create_initial_population()
                continue
            
            # Track best
            min_fitness_idx = valid_indices[np.argmin([fitness_values[i] for i in valid_indices])]
            min_fitness = fitness_values[min_fitness_idx]
            
            if min_fitness < self.best_fitness:
                self.best_fitness = min_fitness
                self.best_solution = self.population[min_fitness_idx].copy()
            
            self.convergence_history.append(self.best_fitness)
            
            # Calculate diversity
            diversity = self._calculate_diversity(self.population)
            population_diversity_history.append(diversity)
            
            # Print progress
            if generation % 20 == 0 or generation == self.generations - 1:
                ls_success_rate = (self.ls_improvements / self.ls_applications * 100 
                                  if self.ls_applications > 0 else 0)
                print(f"  Gen {generation:3d}: Best Fitness = {self.best_fitness:.4f}, "
                      f"Diversity = {diversity:.4f}, LS Success = {ls_success_rate:.1f}%")
            
            # Early stopping
            if generation > 50 and len(set(self.convergence_history[-30:])) == 1:
                print(f"  Converged at generation {generation}")
                break
            
            # Evolve
            self.population = self.evolve_population(self.population, fitness_values, generation)
        
        end_time = time.time()
        
        # Calculate metrics
        best_distance = 0.0
        best_time = 0.0
        
        if self.best_solution:
            for i in range(len(self.best_solution) - 1):
                from_node = self.best_solution[i]
                to_node = self.best_solution[i + 1]
                distance = self.distances[from_node][to_node]
                if distance != float('inf'):
                    traffic = self.traffic_weights[from_node][to_node]
                    best_distance += distance
                    best_time += self.calculate_travel_time(distance, traffic)
        
        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'best_distance': best_distance,
            'best_time': best_time,
            'convergence_history': self.convergence_history,
            'diversity_history': population_diversity_history,
            'generations_run': len(self.convergence_history),
            'time_elapsed': end_time - start_time,
            'ls_applications': self.ls_applications,
            'ls_improvements': self.ls_improvements
        }
    
    def _calculate_diversity(self, population: List[List[int]]) -> float:
        """Calculate population diversity."""
        if len(population) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        sample_size = min(50, len(population))
        sampled = random.sample(population, sample_size)
        
        for i in range(len(sampled)):
            for j in range(i + 1, len(sampled)):
                distance = sum(1 for a, b in zip(sampled[i], sampled[j]) if a != b)
                total_distance += distance / len(sampled[i])
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0


def load_dataset_problem(dataset_name: str, data_dir: str = "datasets", 
                        num_targets: Optional[int] = 10) -> Optional[Dict[str, Any]]:
    """Load a problem instance from dataset files."""
    import os
    
    nodes_path = os.path.join(data_dir, f"{dataset_name}_nodes.csv")
    edges_path = os.path.join(data_dir, f"{dataset_name}_edges.csv")
    
    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        print(f"Dataset not found: {dataset_name}")
        return None
    
    try:
        nodes_df = pd.read_csv(nodes_path)
        edges_df = pd.read_csv(edges_path)
        
        print(f"Loaded dataset '{dataset_name}': {len(nodes_df)} nodes, {len(edges_df)} edges")
        
        return {
            'dataset_name': dataset_name,
            'nodes_df': nodes_df,
            'edges_df': edges_df,
        }
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None


def run(problem=None, config=None):
    """
    Run Hybrid GA on the given problem.
    
    Args:
        problem: Problem instance or dataset name string
        config: Dict with GA parameters
        
    Returns:
        Dict with metrics and results
    """
    start = time.time()
    
    # Default configuration
    if config is None:
        config = {}
    
    # Set defaults
    config.setdefault('population_size', 100)
    config.setdefault('generations', 200)
    config.setdefault('elite_size', 0.1)
    config.setdefault('tournament_size', 5)
    config.setdefault('crossover_rate', 0.8)
    config.setdefault('mutation_rate', 0.2)
    config.setdefault('distance_weight', 0.5)
    config.setdefault('time_weight', 0.5)
    config.setdefault('local_search_rate', 0.3)
    config.setdefault('local_search_intensity', 50)
    config.setdefault('adaptive_ls', True)
    
    # Handle string dataset name
    if isinstance(problem, str):
        problem = load_dataset_problem(problem)
        if problem is None:
            return {
                "algorithm": "hybrid_ga",
                "objective": float('inf'),
                "time_s": time.time() - start,
                "status": "dataset-not-found"
            }
    
    # Create placeholder if needed
    if problem is None:
        n_nodes = 10
        nodes = list(range(n_nodes))
        problem = {
            'nodes': nodes,
            'start_node': 0,
            'target_nodes': nodes[1:],
            'dataset_name': 'placeholder',
        }
    
    print(f"\n{'='*60}")
    print(f"Running Hybrid GA on dataset: {problem.get('dataset_name', 'unknown')}")
    print(f"{'='*60}")
    print(f"Config: pop_size={config['population_size']}, "
          f"generations={config['generations']}, "
          f"elite_size={config['elite_size']}, "
          f"ls_rate={config['local_search_rate']}")
    
    # Run Hybrid GA
    try:
        ga = HybridGA(problem, config)
        results = ga.run_algorithm()
        
        end = time.time()
        
        ls_success_rate = (results['ls_improvements'] / results['ls_applications'] * 100 
                          if results['ls_applications'] > 0 else 0)
        
        print(f"\n{'='*60}")
        print(f"Hybrid GA Results:")
        print(f"  Best Fitness: {results['best_fitness']:.4f}")
        print(f"  Best Distance: {results['best_distance']:.4f} km")
        print(f"  Best Time: {results['best_time']:.4f} min")
        print(f"  Generations: {results['generations_run']}")
        print(f"  Local Search: {results['ls_applications']} applications, "
              f"{results['ls_improvements']} improvements ({ls_success_rate:.1f}%)")
        print(f"  Total Time: {end - start:.2f}s")
        print(f"{'='*60}\n")
        
        return {
            "algorithm": "hybrid_ga",
            "dataset": problem.get('dataset_name', 'unknown'),
            "objective": float(results['best_fitness']),
            "best_distance": float(results['best_distance']),
            "best_time": float(results['best_time']),
            "time_s": end - start,
            "generations": results['generations_run'],
            "elite_count": ga.elite_count,
            "convergence_history": results['convergence_history'],
            "diversity_history": results.get('diversity_history', []),
            "best_route": results['best_solution'],
            "num_nodes": len(ga.nodes),
            "num_targets": len(ga.target_nodes),
            "ls_applications": results['ls_applications'],
            "ls_improvements": results['ls_improvements'],
            "ls_success_rate": ls_success_rate,
            "status": "ok"
        }
    except Exception as e:
        print(f"Error running Hybrid GA: {e}")
        import traceback
        traceback.print_exc()
        return {
            "algorithm": "hybrid_ga",
            "objective": float('inf'),
            "time_s": time.time() - start,
            "status": f"error: {str(e)}"
        }