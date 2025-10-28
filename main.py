"""
Main runner for Emergency Routing GA experiments.

This script integrates the four GA variants and provides a simple entry point
for running experiments. Right now it calls the placeholder `run` functions
from each algorithm module and prints results.
"""
import json
from algorithms import run_standard, run_elitist, run_adaptive, run_hybrid


def run_all(problem=None, config=None):
    results = []
    for fn in (run_standard, run_elitist, run_adaptive, run_hybrid):
        res = fn(problem=problem, config=config)
        results.append(res)
    return results


if __name__ == "__main__":
    # Placeholder problem/config
    config = {"population": 100, "generations": 200}
    print("Running Emergency Routing GA variants (placeholders)...")
    results = run_all(problem=None, config=config)
    print(json.dumps(results, indent=2))
    # Save a results CSV placeholder
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv("results/placeholder_results.csv", index=False)
        print("Saved results/placeholder_results.csv")
    except Exception:
        print("pandas not installed — skipping CSV save. Add pandas to requirements.txt to enable saving.")
