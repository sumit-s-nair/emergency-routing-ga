"""Test script for Standard GA implementation using real Koramangala dataset."""
from algorithms.standard_ga import run
import numpy as np
import pandas as pd
import random

# Load real dataset
print("Loading Koramangala dataset...")
nodes_df = pd.read_csv('datasets/koramangala_nodes.csv')
edges_df = pd.read_csv('datasets/koramangala_edges.csv')

print(f"✓ Loaded {len(nodes_df)} nodes, {len(edges_df)} edges")

# Build adjacency/distance lookup
node_to_idx = {node_id: idx for idx, node_id in enumerate(nodes_df['node_id'])}
distance_lookup = {}

for _, edge in edges_df.iterrows():
    from_node = edge['from_node']
    to_node = edge['to_node']
    time = edge['travel_time_min']
    
    distance_lookup[(from_node, to_node)] = time

print(f"✓ Built distance lookup with {len(distance_lookup)} edges")

# Select random emergency locations from the network
random.seed(42)
np.random.seed(42)

num_emergencies = 10
all_nodes = nodes_df['node_id'].tolist()
selected_nodes = random.sample(all_nodes, num_emergencies + 1)  # +1 for depot

depot_node = selected_nodes[0]
emergency_nodes = selected_nodes[1:]

print(f"\n✓ Selected {num_emergencies} emergency locations")
print(f"  Depot: {depot_node}")
print(f"  Emergencies: {emergency_nodes[:5]}... (showing first 5)")

# Build distance matrix using shortest path approximation
# For simplicity, use direct edges or large value if no direct edge
def get_distance(from_node, to_node):
    """Get travel time between two nodes."""
    if from_node == to_node:
        return 0.0
    
    # Try direct edge
    if (from_node, to_node) in distance_lookup:
        return distance_lookup[(from_node, to_node)]
    
    # No direct edge - use average edge time * 3 as approximation
    avg_time = edges_df['travel_time_min'].mean()
    return avg_time * 3  # Rough estimate for indirect path

# Create distance matrix
all_locations = [depot_node] + emergency_nodes
n = len(all_locations)
distance_matrix = np.zeros((n, n))

for i, node_i in enumerate(all_locations):
    for j, node_j in enumerate(all_locations):
        distance_matrix[i, j] = get_distance(node_i, node_j)

print(f"✓ Built {n}x{n} distance matrix")
print(f"  Avg distance: {distance_matrix[distance_matrix > 0].mean():.2f} min")

# Create problem instance
problem = {
    'num_emergencies': num_emergencies,
    'distance_matrix': distance_matrix,
    'depot_idx': 0,
    'node_ids': all_locations  # Keep track of actual node IDs
}

config = {
    'population_size': 50,
    'generations': 100,
    'crossover_rate': 0.8,
    'mutation_rate': 0.1,
    'seed': 42
}

print("\n" + "="*60)
print("Running Standard GA on real Koramangala dataset...")
print("="*60)

result = run(problem, config)

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Algorithm: {result['algorithm']}")
print(f"Best objective: {result['objective']:.2f} minutes")
print(f"Best route (indices): {result['best_solution']}")
print(f"Best route (node IDs): {[all_locations[i+1] for i in result['best_solution']]}")  # +1 to skip depot in indices
print(f"Execution time: {result['time_s']:.3f}s")
print(f"Status: {result['status']}")
print(f"\nConvergence (last 10 generations):")
for i, fitness in enumerate(result['convergence'], start=len(result['convergence'])-9):
    print(f"  Gen {config['generations']-10+i}: {fitness:.2f} min")

print("\n" + "="*60)
print("✓ Standard GA successfully tested on real dataset!")
print("="*60)
