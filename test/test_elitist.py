"""
Test script for Elitist GA
"""
from algorithms import run_elitist

# Test with koramangala dataset
print("Testing Elitist GA with koramangala dataset...")
result = run_elitist('koramangala', {
    'population_size': 30,
    'generations': 30,
    'elite_size': 0.2  # 20% elitism
})

print("\n" + "="*60)
print("TEST RESULTS:")
print("="*60)
print(f"Status: {result['status']}")
print(f"Dataset: {result.get('dataset', 'N/A')}")
print(f"Objective: {result['objective']}")
if result['objective'] != float('inf'):
    print(f"Best Distance: {result.get('best_distance', 0):.4f} km")
    print(f"Best Time: {result.get('best_time', 0):.4f} min")
print(f"Generations: {result.get('generations', 0)}")
print(f"Time: {result['time_s']:.2f}s")
print(f"Nodes: {result.get('num_nodes', 0)}")
print(f"Targets: {result.get('num_targets', 0)}")

# Show convergence
if result.get('convergence_history'):
    hist = result['convergence_history']
    if len(hist) > 0:
        print(f"\nConvergence: {hist[0]:.4f} -> {hist[-1]:.4f}")
print("="*60)
