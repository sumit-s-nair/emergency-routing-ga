"""
Test script for Hybrid GA
"""
from algorithms import run_hybrid

# Test with koramangala dataset
print("Testing Hybrid GA with koramangala dataset...")
result = run_hybrid('koramangala', {
    'population_size': 30,
    'generations': 30,
    'elite_size': 0.2,  # 20% elitism
    'local_search_rate': 0.3,  # Apply LS to 30% of offspring
    'local_search_intensity': 50,  # Max 50 2-opt iterations
    'adaptive_ls': True  # Reduce LS intensity over generations
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

# Show hybrid-specific metrics
if result.get('ls_applications'):
    print(f"\nLocal Search Stats:")
    print(f"  Applications: {result['ls_applications']}")
    print(f"  Improvements: {result['ls_improvements']}")
    print(f"  Success Rate: {result.get('ls_success_rate', 0):.1f}%")

# Show convergence
if result.get('convergence_history'):
    hist = result['convergence_history']
    if len(hist) > 0:
        print(f"\nConvergence: {hist[0]:.4f} -> {hist[-1]:.4f}")
        improvement = ((hist[0] - hist[-1]) / hist[0] * 100) if hist[0] > 0 else 0
        print(f"Improvement: {improvement:.2f}%")

# Show diversity trend
if result.get('diversity_history'):
    div_hist = result['diversity_history']
    if len(div_hist) > 0:
        print(f"\nDiversity: {div_hist[0]:.4f} -> {div_hist[-1]:.4f}")

print("="*60)