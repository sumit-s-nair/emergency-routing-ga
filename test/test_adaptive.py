"""
Test script for Adaptive GA
"""
from algorithms import run_adaptive

# Test with koramangala dataset
print("Testing Adaptive GA with koramangala dataset...")
result = run_adaptive('koramangala', {
    'population_size': 30,
    'generations': 30,
    'elite_size': 0.2,  # Adaptive GA can still use elitism
    
    # --- Adaptive Parameters ---
    'initial_mutation_rate': 0.1,  # Starting rate
    'min_mutation_rate': 0.01,     # Floor for mutation rate
    'max_mutation_rate': 0.5,      # Ceiling for mutation rate
    'stagnation_limit': 10         # Gens to wait before increasing rate
})

print("\n" + "="*60)
print("TEST RESULTS (ADAPTIVE GA):")
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

# Show mutation rate history
if result.get('mutation_rate_history'):
    hist = result['mutation_rate_history']
    if len(hist) > 0:
        print(f"Mutation Rate: {hist[0]:.4f} -> {hist[-1]:.4f}")
print("="*60)