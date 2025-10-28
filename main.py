"""
Main runner for Emergency Routing GA experiments.

Simple runner that calls all four GA variants and saves results.
"""
import json
from algorithms import run_standard, run_elitist, run_adaptive, run_hybrid


def run_all(problem=None, config=None):
    """Run all GA variants on a problem.
    
    Args:
        problem: Problem instance (e.g., dataset with emergency locations)
        config: Algorithm configuration dict
        
    Returns:
        list of result dicts from each algorithm
    """
    results = []
    for fn in (run_standard, run_elitist, run_adaptive, run_hybrid):
        res = fn(problem=problem, config=config)
        results.append(res)
    return results


if __name__ == "__main__":
    # Placeholder problem/config
    config = {"population": 100, "generations": 200}
    
    print("Running Emergency Routing GA variants...")
    results = run_all(problem=None, config=config)
    print(json.dumps(results, indent=2))
    
    # Save results
    try:
        import pandas as pd
        from pathlib import Path
        Path("results").mkdir(exist_ok=True)
        df = pd.DataFrame(results)
        df.to_csv("results/ga_results.csv", index=False)
        print("\n✓ Saved results/ga_results.csv")
    except ImportError:
        print("\n(pandas not installed, skipping CSV save)")

