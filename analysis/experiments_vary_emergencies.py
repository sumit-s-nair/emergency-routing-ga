"""
Run experiments sweeping number of emergencies and collect results.
Saves: results/emergencies_sweep_results.csv, results/emergencies_sweep_summary.csv,
results/emergencies_obj_vs_count.png, results/emergencies_time_vs_count.png, results/emergencies_report.md
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Reuse utilities from compare_algorithms
import compare_algorithms as ca
from algorithms import run_standard, run_elitist, run_adaptive, run_hybrid

OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(exist_ok=True)

CITIES = ['koramangala', 'bangalore_central', 'cambridge_ma']
EMERGENCY_COUNTS = [5, 10, 15, 20, 25]
# Increased trial count for statistical confidence. To keep runtime reasonable we reduce generations to 50.
TRIALS = 10

ALGORITHMS = [
    (run_standard, 'Standard GA'),
    (run_elitist, 'Elitist GA'),
    (run_adaptive, 'Adaptive GA'),
    (run_hybrid, 'Hybrid GA'),
]

# Use a lighter config to keep runtime reasonable
BASE_CONFIG = {
    'population_size': 100,
    'generations': 50,  # reduced from 100 to speed up multi-trial runs
    'crossover_rate': 0.8,
    'mutation_rate': 0.1,
    'tournament_size': 5,
    'seed': 42
}

all_runs = []

start_all = time.time()
for city in CITIES:
    for count in EMERGENCY_COUNTS:
        for trial in range(1, TRIALS+1):
            # Create problem instance per trial with fixed seed variation
            seed = BASE_CONFIG.get('seed', 42) + trial
            problem = ca.create_problem_instance(city, num_emergencies=count, seed=seed)

            for algo_fn, algo_name in ALGORITHMS:
                cfg = BASE_CONFIG.copy()
                cfg['seed'] = seed
                run_start = time.time()
                res = algo_fn(problem=problem, config=cfg)
                run_time = time.time() - run_start

                # Normalize keys (handle different naming conventions)
                objective = res.get('objective', res.get('best_fitness', float('inf')))
                time_s = res.get('time_s', run_time)
                convergence = res.get('convergence', res.get('convergence_history', []))

                all_runs.append({
                    'city': city,
                    'num_emergencies': count,
                    'trial': trial,
                    'algorithm': algo_name,
                    'objective': float(objective),
                    'time_s': float(time_s),
                    'status': res.get('status', 'unknown'),
                    'convergence_len': len(convergence),
                })

# Save raw runs
raw_df = pd.DataFrame(all_runs)
raw_path = OUTPUT_DIR / f'emergencies_sweep_results_trials{TRIALS}.csv'
raw_df.to_csv(raw_path, index=False)

# Aggregate summary (mean and std per algorithm/city/count)
summary = raw_df.groupby(['city', 'num_emergencies', 'algorithm']).agg(
    objective_mean=('objective', 'mean'),
    objective_std=('objective', 'std'),
    time_mean=('time_s', 'mean'),
    time_std=('time_s', 'std'),
    runs=('objective', 'count')
).reset_index()

summary_path = OUTPUT_DIR / f'emergencies_sweep_summary_trials{TRIALS}.csv'
summary.to_csv(summary_path, index=False)

# Plot: objective vs num_emergencies (one subplot per city)
sns.set(style='whitegrid')
for city in CITIES:
    city_df = summary[summary['city'] == city]
    plt.figure(figsize=(8,5))
    sns.lineplot(data=city_df, x='num_emergencies', y='objective_mean', hue='algorithm', marker='o')
    plt.fill_between([], [], [])
    plt.title(f'Objective vs Number of Emergencies - {city}')
    plt.xlabel('Number of Emergencies')
    plt.ylabel('Objective (minutes)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'emergencies_obj_vs_count_{city}.png')
    plt.close()

# Plot: time vs num_emergencies
for city in CITIES:
    city_df = summary[summary['city'] == city]
    plt.figure(figsize=(8,5))
    sns.lineplot(data=city_df, x='num_emergencies', y='time_mean', hue='algorithm', marker='o')
    plt.title(f'Execution Time vs Number of Emergencies - {city}')
    plt.xlabel('Number of Emergencies')
    plt.ylabel('Time (s)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'emergencies_time_vs_count_{city}.png')
    plt.close()

# Generate mini-report
report_lines = []
report_lines.append('# Emergencies Sweep Experiments')
report_lines.append(f'Generated: {time.ctime()}')
report_lines.append(f'Config: generations={BASE_CONFIG["generations"]}, population={BASE_CONFIG["population_size"]}, trials={TRIALS}')
report_lines.append('\n## Summary Tables')
report_lines.append(f'Raw results: `{raw_path}`')
report_lines.append(f'Summary: `{summary_path}`')
report_lines.append('\n## Plots')
for city in CITIES:
    report_lines.append(f'- Objective plot: `results/emergencies_obj_vs_count_{city}.png`')
    report_lines.append(f'- Time plot: `results/emergencies_time_vs_count_{city}.png`')

report_lines.append('\n## Quick Observations (fill after inspection)')
report_lines.append('- TODO: Inspect plots and add interpretations')

report_path = OUTPUT_DIR / f'emergencies_report_trials{TRIALS}.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print('Done. Outputs:')
print(' -', raw_path)
print(' -', summary_path)
print(' -', report_path)
print(' - plots per city in results/ (objective/time)')

end_all = time.time()
print(f'Total time: {end_all - start_all:.2f}s')
