"""
Elitist GA variant for EVDRP.

This algorithm preserves the best individuals in each generation to ensure solution quality.
The elitism mechanism guarantees that the best solutions are carried over to the next generation,
preventing the loss of good solutions during the evolutionary process.

Key Features:
- Elitism: Top N% individuals automatically survive to next generation
- Tournament Selection for parent selection
- Order Crossover (OX) for route-based solutions
- Swap Mutation to maintain route validity
- Fitness based on weighted sum of distance and response time
- Supports real OSM datasets and synthetic problems

Reference:
Saxena, K., & Chauhan, D. S. (2010). Elitism-based genetic algorithm for 
solving vehicle routing problem. IJCA, 8(4), 6–10.
"""
import time
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional


class ElitistGA:
    """
    Elitist Genetic Algorithm for Emergency Vehicle Dispatch and Routing Problem.
    
    Attributes:
        problem: Problem instance containing graph/network data
        config: Configuration dictionary with GA parameters
        population: Current population of solutions
        best_solution: Best solution found so far
        best_fitness: Fitness of the best solution
        convergence_history: List of best fitness values per generation
    """
    
    def __init__(self, problem: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize the Elitist GA.
        
        Args:
            problem: Dict containing:
                - 'graph': Road network as adjacency matrix or edge list (optional)
                - 'nodes': List of node IDs (intersections)
                - 'edges_df': DataFrame with edges (from_node, to_node, distance_km, traffic_weight, etc.)
                - 'nodes_df': DataFrame with nodes (node_id, lat, lon) (optional)
                - 'distances': Distance matrix (if edges_df not provided)
                - 'traffic_weights': Traffic weight matrix (1-5) (if edges_df not provided)
                - 'start_node': Emergency vehicle starting position
                - 'target_nodes': List of nodes to visit (emergency locations)
                - 'dataset_name': Name of the dataset (for logging)
            config: Dict containing:
                - 'population_size': Size of population (default: 100)
                - 'generations': Number of generations (default: 200)
                - 'elite_size': Number/percentage of elite individuals (default: 0.1)
                - 'tournament_size': Tournament selection size (default: 5)
                - 'crossover_rate': Probability of crossover (default: 0.8)
                - 'mutation_rate': Probability of mutation (default: 0.2)
                - 'distance_weight': Weight for distance in fitness (default: 0.5)
                - 'time_weight': Weight for time in fitness (default: 0.5)
        """
        self.problem = problem
        self.config = config
        self.dataset_name = problem.get('dataset_name', 'unknown')
        
        # Extract problem parameters - support both DataFrame and matrix formats
        if 'edges_df' in problem and problem['edges_df'] is not None:
            # Load from DataFrame format (real OSM data)
            self._load_from_dataframe(problem)
        else:
            # Use matrix format or generate random
            self.nodes = problem.get('nodes', list(range(10)))  # Default 10 nodes
            self.distances = problem.get('distances', self._generate_random_distances())
            self.traffic_weights = problem.get('traffic_weights', self._generate_random_traffic())
            self.start_node = problem.get('start_node', self.nodes[0])
            self.target_nodes = problem.get('target_nodes', self.nodes[1:])
        
        # Extract config parameters
        self.population_size = config.get('population_size', 100)
        self.generations = config.get('generations', 200)
        self.elite_size = config.get('elite_size', 0.1)  # Can be int or float (percentage)
        self.tournament_size = config.get('tournament_size', 5)
        self.crossover_rate = config.get('crossover_rate', 0.8)
        self.mutation_rate = config.get('mutation_rate', 0.2)
        self.distance_weight = config.get('distance_weight', 0.5)
        self.time_weight = config.get('time_weight', 0.5)
        
        # Calculate elite count
        if isinstance(self.elite_size, float):
            self.elite_count = max(1, int(self.population_size * self.elite_size))
        else:
            self.elite_count = max(1, self.elite_size)
        
        # Initialize state
        self.population = []
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_history = []
        
    def _load_from_dataframe(self, problem: Dict[str, Any]) -> None:
        """
        Load problem data from DataFrame format (real OSM datasets).
        
        Args:
            problem: Problem dict containing edges_df and optionally nodes_df
        """
        edges_df = problem['edges_df']
        nodes_df = problem.get('nodes_df', None)
        
        # Get unique nodes
        all_nodes = set(edges_df['from_node'].unique()) | set(edges_df['to_node'].unique())
        self.nodes = sorted(list(all_nodes))
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        # Build distance and traffic matrices
        n = len(self.nodes)
        self.distances = np.full((n, n), float('inf'))
        self.traffic_weights = np.ones((n, n))
        
        # Build adjacency list for efficient neighbor lookup
        self.adjacency = {i: set() for i in range(n)}
        
        # Fill matrices from edges
        for _, edge in edges_df.iterrows():
            from_idx = self.node_to_idx[edge['from_node']]
            to_idx = self.node_to_idx[edge['to_node']]
            
            self.distances[from_idx][to_idx] = edge['distance_km']
            self.traffic_weights[from_idx][to_idx] = edge['traffic_weight']
            
            # Make symmetric (assume bidirectional roads)
            self.distances[to_idx][from_idx] = edge['distance_km']
            self.traffic_weights[to_idx][from_idx] = edge['traffic_weight']
            
            # Update adjacency list
            self.adjacency[from_idx].add(to_idx)
            self.adjacency[to_idx].add(from_idx)
        
        # Set diagonal to 0
        np.fill_diagonal(self.distances, 0)
        
        # Compute all-pairs shortest paths using Floyd-Warshall
        print(f"  Computing shortest paths between all node pairs...")
        self._compute_shortest_paths()
        print(f"  Shortest paths computed")
        
        # Select connected nodes for targets
        # Find a large connected component
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
        
        # Use largest connected component
        largest_component = max(components, key=len)
        print(f"  Using largest connected component: {len(largest_component)} nodes "
              f"(out of {n} total)")
        
        # Set start and target nodes from the connected component
        if 'start_node' in problem and problem['start_node'] in self.node_to_idx:
            start = problem['start_node']
            self.start_node = self.node_to_idx[start]
            # Ensure start is in connected component
            if self.start_node not in largest_component:
                self.start_node = random.choice(largest_component)
        else:
            self.start_node = random.choice(largest_component)
        
        # Select random target nodes from connected component if not provided
        if 'target_nodes' in problem:
            targets = problem['target_nodes']
            self.target_nodes = [self.node_to_idx[t] for t in targets 
                               if t in self.node_to_idx and self.node_to_idx[t] in largest_component]
            if not self.target_nodes:
                # Fallback to random selection
                num_targets = min(10, len(largest_component) - 1)
                available = [i for i in largest_component if i != self.start_node]
                self.target_nodes = random.sample(available, min(num_targets, len(available)))
        else:
            # Select random subset of nodes as targets from connected component
            num_targets = min(10, len(largest_component) - 1)  # Visit up to 10 nodes
            available = [i for i in largest_component if i != self.start_node]
            self.target_nodes = random.sample(available, min(num_targets, len(available)))
        
    def _generate_random_distances(self) -> np.ndarray:
        """Generate random distance matrix for placeholder problem."""
        n = len(self.nodes)
        dist = np.random.uniform(0.5, 10.0, (n, n))
        # Make symmetric
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        return dist
    
    def _generate_random_traffic(self) -> np.ndarray:
        """Generate random traffic weight matrix (1-5 scale)."""
        n = len(self.nodes)
        traffic = np.random.uniform(1.0, 5.0, (n, n))
        traffic = (traffic + traffic.T) / 2
        np.fill_diagonal(traffic, 1)
        return traffic
    
    def _compute_shortest_paths(self) -> None:
        """
        Compute all-pairs shortest paths using Floyd-Warshall algorithm.
        Updates self.distances and self.traffic_weights matrices.
        """
        n = len(self.nodes)
        dist = self.distances.copy()
        traffic = self.traffic_weights.copy()
        
        # Floyd-Warshall algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        # Average traffic weight along the path
                        traffic[i][j] = (traffic[i][k] + traffic[k][j]) / 2
        
        self.distances = dist
        self.traffic_weights = traffic
    
    def calculate_travel_time(self, distance: float, traffic_weight: float) -> float:
        """
        Calculate travel time based on distance and traffic.
        
        Formula: TravelTime = Distance × (1 + 0.2 × Traffic)
        
        Args:
            distance: Distance in km
            traffic_weight: Traffic congestion factor (1-5)
            
        Returns:
            Travel time in appropriate units
        """
        return distance * (1 + 0.2 * traffic_weight)
    
    def calculate_fitness(self, route: List[int]) -> float:
        """
        Calculate fitness of a route (lower is better).
        
        Fitness = distance_weight × total_distance + time_weight × total_time
        
        Handles cases where nodes are not directly connected by using shortest path.
        
        Args:
            route: List of node indices representing the route
            
        Returns:
            Fitness value (lower is better)
        """
        if len(route) < 2:
            return float('inf')
        
        total_distance = 0.0
        total_time = 0.0
        
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            
            # Check if direct connection exists
            distance = self.distances[from_node][to_node]
            
            # If no direct connection (infinity), penalize heavily
            if distance == float('inf'):
                return float('inf')
            
            traffic = self.traffic_weights[from_node][to_node]
            travel_time = self.calculate_travel_time(distance, traffic)
            
            total_distance += distance
            total_time += travel_time
        
        # Weighted fitness
        fitness = self.distance_weight * total_distance + self.time_weight * total_time
        return fitness
    
    def create_initial_population(self) -> List[List[int]]:
        """
        Create initial population of routes using different strategies.
        
        Strategies:
        1. Nearest neighbor heuristic (greedy)
        2. Random insertion
        3. Pure random permutation
        
        Returns:
            List of routes (each route is a list of node indices)
        """
        population = []
        
        # Strategy 1: Nearest Neighbor (30% of population)
        nn_count = int(0.3 * self.population_size)
        for _ in range(nn_count):
            route = self._nearest_neighbor_route()
            population.append(route)
        
        # Strategy 2: Random Insertion (40% of population)
        ri_count = int(0.4 * self.population_size)
        for _ in range(ri_count):
            route = self._random_insertion_route()
            population.append(route)
        
        # Strategy 3: Random Permutation (remaining)
        remaining = self.population_size - len(population)
        for _ in range(remaining):
            route = [self.start_node] + random.sample(self.target_nodes, len(self.target_nodes))
            population.append(route)
        
        return population
    
    def _nearest_neighbor_route(self) -> List[int]:
        """
        Create a route using nearest neighbor heuristic.
        Starts from start_node and greedily picks nearest unvisited target.
        
        Returns:
            Route as list of node indices
        """
        route = [self.start_node]
        remaining = set(self.target_nodes)
        current = self.start_node
        
        while remaining:
            # Find nearest unvisited node
            nearest = min(remaining, key=lambda node: self.distances[current][node])
            route.append(nearest)
            remaining.remove(nearest)
            current = nearest
        
        return route
    
    def _random_insertion_route(self) -> List[int]:
        """
        Create a route using random insertion heuristic.
        Starts with start node and randomly inserts remaining nodes
        at positions that minimize increase in route cost.
        
        Returns:
            Route as list of node indices
        """
        route = [self.start_node]
        remaining = list(self.target_nodes)
        random.shuffle(remaining)
        
        for node in remaining:
            # Find best insertion position
            best_pos = 1
            best_cost = float('inf')
            
            for pos in range(1, len(route) + 1):
                # Calculate cost increase of inserting node at pos
                if pos == len(route):
                    # Insert at end
                    cost_increase = self.distances[route[pos-1]][node]
                else:
                    # Insert between route[pos-1] and route[pos]
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
        """
        Select a parent using tournament selection.
        
        Args:
            population: Current population
            fitness_values: Fitness values for each individual
            
        Returns:
            Selected individual (route)
        """
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()
    
    def order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Perform Order Crossover (OX) on two parent routes.
        
        OX preserves the relative order of nodes from parents while creating valid routes.
        
        Args:
            parent1: First parent route
            parent2: Second parent route
            
        Returns:
            Tuple of two offspring routes
        """
        size = len(parent1)
        
        # Special case: if route is just start node + one target
        if size <= 2:
            return parent1.copy(), parent2.copy()
        
        # Select two crossover points (excluding start node)
        start_idx = random.randint(1, size - 2)
        end_idx = random.randint(start_idx + 1, size - 1)
        
        # Create offspring
        offspring1 = [None] * size
        offspring2 = [None] * size
        
        # Keep start node
        offspring1[0] = parent1[0]
        offspring2[0] = parent2[0]
        
        # Copy segment from parents
        offspring1[start_idx:end_idx] = parent1[start_idx:end_idx]
        offspring2[start_idx:end_idx] = parent2[start_idx:end_idx]
        
        # Fill remaining positions with nodes from other parent
        self._fill_offspring(offspring1, parent2, start_idx, end_idx)
        self._fill_offspring(offspring2, parent1, start_idx, end_idx)
        
        return offspring1, offspring2
    
    def _fill_offspring(self, offspring: List[int], parent: List[int], 
                       start: int, end: int) -> None:
        """Helper to fill offspring after crossover."""
        offspring_set = set(n for n in offspring if n is not None)
        parent_idx = 1  # Skip start node
        offspring_idx = 1
        
        while None in offspring:
            if offspring_idx == start:
                offspring_idx = end
            
            if offspring_idx >= len(offspring):
                break
                
            # Find next node from parent not already in offspring
            while parent_idx < len(parent) and parent[parent_idx] in offspring_set:
                parent_idx += 1
            
            if parent_idx < len(parent):
                offspring[offspring_idx] = parent[parent_idx]
                offspring_set.add(parent[parent_idx])
                parent_idx += 1
            
            offspring_idx += 1
    
    def swap_mutation(self, route: List[int]) -> List[int]:
        """
        Perform swap mutation on a route.
        
        Randomly swaps two nodes in the route (excluding start node).
        
        Args:
            route: Route to mutate
            
        Returns:
            Mutated route
        """
        mutated = route.copy()
        
        if len(mutated) <= 2:
            return mutated
        
        # Select two random positions (excluding start node at index 0)
        idx1, idx2 = random.sample(range(1, len(mutated)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        return mutated
    
    def evolve_population(self, population: List[List[int]], 
                         fitness_values: List[float]) -> List[List[int]]:
        """
        Evolve population for one generation using elitism.
        
        Steps:
        1. Sort population by fitness
        2. Select elite individuals
        3. Generate offspring through selection, crossover, and mutation
        4. Combine elite and offspring
        
        Args:
            population: Current population
            fitness_values: Fitness values for each individual
            
        Returns:
            New population
        """
        # Sort population by fitness (lower is better)
        sorted_indices = np.argsort(fitness_values)
        sorted_population = [population[i] for i in sorted_indices]
        
        # Elitism: Keep best individuals
        elite = sorted_population[:self.elite_count]
        
        # Generate offspring
        offspring = []
        offspring_needed = self.population_size - self.elite_count
        
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
            
            offspring.extend([child1, child2])
        
        # Combine elite and offspring
        new_population = elite + offspring[:offspring_needed]
        
        return new_population
    
    def run_algorithm(self) -> Dict[str, Any]:
        """
        Run the Elitist GA algorithm.
        
        Returns:
            Dict containing:
                - 'best_solution': Best route found
                - 'best_fitness': Fitness of best route
                - 'best_distance': Total distance of best route
                - 'best_time': Total time of best route
                - 'convergence_history': Fitness values per generation
                - 'generations_run': Number of generations executed
                - 'population_diversity': Average diversity per generation
        """
        start_time = time.time()
        
        # Initialize population
        self.population = self.create_initial_population()
        population_diversity_history = []
        
        # Evolution loop
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_values = [self.calculate_fitness(route) for route in self.population]
            
            # Filter out invalid solutions (inf fitness)
            valid_indices = [i for i, f in enumerate(fitness_values) if f != float('inf')]
            
            if not valid_indices:
                # No valid solutions, regenerate population
                self.population = self.create_initial_population()
                continue
            
            # Track best solution
            min_fitness_idx = valid_indices[np.argmin([fitness_values[i] for i in valid_indices])]
            min_fitness = fitness_values[min_fitness_idx]
            
            if min_fitness < self.best_fitness:
                self.best_fitness = min_fitness
                self.best_solution = self.population[min_fitness_idx].copy()
            
            self.convergence_history.append(self.best_fitness)
            
            # Calculate population diversity (average pairwise distance)
            diversity = self._calculate_diversity(self.population)
            population_diversity_history.append(diversity)
            
            # Print progress every 20 generations
            if generation % 20 == 0 or generation == self.generations - 1:
                print(f"  Gen {generation:3d}: Best Fitness = {self.best_fitness:.4f}, "
                      f"Diversity = {diversity:.4f}")
            
            # Optional: Early stopping if converged
            if generation > 50 and len(set(self.convergence_history[-30:])) == 1:
                # No improvement in last 30 generations
                print(f"  Converged at generation {generation}")
                break
            
            # Evolve population
            self.population = self.evolve_population(self.population, fitness_values)
        
        end_time = time.time()
        
        # Calculate detailed metrics for best solution
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
            'time_elapsed': end_time - start_time
        }
    
    def _calculate_diversity(self, population: List[List[int]]) -> float:
        """
        Calculate population diversity as average pairwise Hamming distance.
        
        Args:
            population: Current population
            
        Returns:
            Average diversity score
        """
        if len(population) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        # Sample a subset for efficiency in large populations
        sample_size = min(50, len(population))
        sampled = random.sample(population, sample_size)
        
        for i in range(len(sampled)):
            for j in range(i + 1, len(sampled)):
                # Hamming distance between routes
                distance = sum(1 for a, b in zip(sampled[i], sampled[j]) if a != b)
                total_distance += distance / len(sampled[i])
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0


def load_dataset_problem(dataset_name: str, data_dir: str = "datasets", 
                        num_targets: Optional[int] = 10) -> Optional[Dict[str, Any]]:
    """
    Load a problem instance from dataset files.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'koramangala', 'bangalore_central')
        data_dir: Directory containing the datasets
        num_targets: Number of target nodes to visit (emergency locations)
        
    Returns:
        Problem dictionary or None if dataset not found
    """
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
            # Start and target nodes will be selected randomly in __init__
        }
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None


def run(problem=None, config=None):
    """
    Run Elitist GA on the given problem.
    
    Args:
        problem: Problem instance containing graph/network data, or None for placeholder
                Can also be a string with dataset name (e.g., 'koramangala')
        config: Dict with GA parameters, or None for defaults
        
    Returns:
        Dict with metrics and results
    """
    start = time.time()
    
    # Default configuration
    if config is None:
        config = {}
    
    # Set defaults for missing config values
    config.setdefault('population_size', 100)
    config.setdefault('generations', 200)
    config.setdefault('elite_size', 0.1)
    config.setdefault('tournament_size', 5)
    config.setdefault('crossover_rate', 0.8)
    config.setdefault('mutation_rate', 0.2)
    config.setdefault('distance_weight', 0.5)
    config.setdefault('time_weight', 0.5)
    
    # Handle string dataset name
    if isinstance(problem, str):
        problem = load_dataset_problem(problem)
        if problem is None:
            return {
                "algorithm": "elitist_ga",
                "objective": float('inf'),
                "time_s": time.time() - start,
                "status": "dataset-not-found"
            }
    
    # Create placeholder problem if none provided
    if problem is None:
        n_nodes = 10
        nodes = list(range(n_nodes))
        problem = {
            'nodes': nodes,
            'start_node': 0,
            'target_nodes': nodes[1:],
            'dataset_name': 'placeholder',
            # Distances and traffic will be generated randomly in __init__
        }
    
    print(f"\n{'='*60}")
    print(f"Running Elitist GA on dataset: {problem.get('dataset_name', 'unknown')}")
    print(f"{'='*60}")
    print(f"Config: pop_size={config['population_size']}, "
          f"generations={config['generations']}, "
          f"elite_size={config['elite_size']}")
    
    # Run Elitist GA
    try:
        ga = ElitistGA(problem, config)
        results = ga.run_algorithm()
        
        end = time.time()
        
        print(f"\n{'='*60}")
        print(f"Elitist GA Results:")
        print(f"  Best Fitness: {results['best_fitness']:.4f}")
        print(f"  Best Distance: {results['best_distance']:.4f} km")
        print(f"  Best Time: {results['best_time']:.4f} min")
        print(f"  Generations: {results['generations_run']}")
        print(f"  Total Time: {end - start:.2f}s")
        print(f"{'='*60}\n")
        
        # Format output
        return {
            "algorithm": "elitist_ga",
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
            "status": "ok"
        }
    except Exception as e:
        print(f"Error running Elitist GA: {e}")
        import traceback
        traceback.print_exc()
        return {
            "algorithm": "elitist_ga",
            "objective": float('inf'),
            "time_s": time.time() - start,
            "status": f"error: {str(e)}"
        }
