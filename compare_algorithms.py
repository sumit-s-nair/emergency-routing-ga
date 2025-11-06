"""Comprehensive comparison analysis of all 4 GA variants on all city datasets.

This script runs Standard GA, Elitist GA, Adaptive GA, and Hybrid GA on
Koramangala, Bangalore Central, and Cambridge datasets, then generates:
- Convergence plots
- Performance comparison charts
- Statistical analysis tables
- LaTeX tables for research paper
- Detailed analysis report
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random

from algorithms import run_standard, run_elitist, run_adaptive, run_hybrid


def load_dataset(city_name, data_dir="datasets"):
    """Load city dataset from CSV files."""
    nodes_path = f"{data_dir}/{city_name}_nodes.csv"
    edges_path = f"{data_dir}/{city_name}_edges.csv"
    
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    
    return nodes_df, edges_df


def build_distance_matrix(nodes_df, edges_df, selected_nodes):
    """Build distance matrix for selected nodes."""
    distance_lookup = {}
    for _, edge in edges_df.iterrows():
        from_node = edge['from_node']
        to_node = edge['to_node']
        time = edge['travel_time_min']
        distance_lookup[(from_node, to_node)] = time
    
    def get_distance(from_node, to_node):
        if from_node == to_node:
            return 0.0
        if (from_node, to_node) in distance_lookup:
            return distance_lookup[(from_node, to_node)]
        avg_time = edges_df['travel_time_min'].mean()
        return avg_time * 3  # Approximation
    
    n = len(selected_nodes)
    distance_matrix = np.zeros((n, n))
    for i, node_i in enumerate(selected_nodes):
        for j, node_j in enumerate(selected_nodes):
            distance_matrix[i, j] = get_distance(node_i, node_j)
    
    return distance_matrix


def create_problem_instance(city_name, num_emergencies=15, seed=42):
    """Create a problem instance from a city dataset."""
    print(f"\nLoading {city_name}...")
    nodes_df, edges_df = load_dataset(city_name)
    
    random.seed(seed)
    all_nodes = nodes_df['node_id'].tolist()
    selected_nodes = random.sample(all_nodes, num_emergencies + 1)
    
    depot_node = selected_nodes[0]
    emergency_nodes = selected_nodes[1:]
    
    distance_matrix = build_distance_matrix(nodes_df, edges_df, selected_nodes)
    
    print(f"  ✓ {len(nodes_df)} nodes, {len(edges_df)} edges")
    print(f"  ✓ {num_emergencies} emergency locations selected")
    
    # Create nodes list (indices from 0 to num_emergencies)
    nodes = list(range(num_emergencies + 1))
    
    # Create traffic weights (assuming uniform traffic for now, could be extracted from edges)
    traffic_weights = np.ones_like(distance_matrix) * 2.5  # Medium traffic
    
    return {
        'city': city_name,
        'num_emergencies': num_emergencies,
        'distance_matrix': distance_matrix,  # For Standard GA
        'distances': distance_matrix,  # For Elitist/Adaptive/Hybrid GAs
        'traffic_weights': traffic_weights,  # Traffic weights for other GAs
        'nodes': nodes,  # Node indices for the problem
        'depot_idx': 0,
        'start_node': 0,  # For consistency
        'target_nodes': list(range(1, num_emergencies + 1)),  # Indices 1 to num_emergencies
        'node_ids': selected_nodes,  # Original node IDs from dataset
        'num_nodes': len(nodes_df),
        'num_edges': len(edges_df)
    }


def run_experiment(algorithm_fn, algorithm_name, problem, config):
    """Run a single experiment and return results."""
    print(f"    Running {algorithm_name}...", end=" ")
    start = time.time()
    
    try:
        result = algorithm_fn(problem=problem, config=config)
        elapsed = time.time() - start
        print(f"✓ ({elapsed:.2f}s)")
        
        # Handle different naming conventions between algorithms
        convergence = result.get('convergence', result.get('convergence_history', []))
        best_solution = result.get('best_solution', result.get('best_route', []))
        
        return {
            'algorithm': algorithm_name,
            'city': problem['city'],
            'objective': result.get('objective', float('inf')),
            'time_s': result.get('time_s', elapsed),
            'status': result.get('status', 'unknown'),
            'convergence': convergence,
            'best_solution': best_solution,
            'num_nodes': problem['num_nodes'],
            'num_edges': problem['num_edges'],
            'num_emergencies': problem['num_emergencies']
        }
    except Exception as e:
        print(f"✗ Error: {e}")
        return {
            'algorithm': algorithm_name,
            'city': problem['city'],
            'objective': float('inf'),
            'time_s': 0.0,
            'status': 'error',
            'error': str(e)
        }


def run_all_experiments(cities, num_emergencies=15, config=None):
    """Run all GA variants on all datasets."""
    if config is None:
        config = {
            'population_size': 100,
            'generations': 200,
            'crossover_rate': 0.8,
            'mutation_rate': 0.1,
            'tournament_size': 5,
            'seed': 42
        }
    
    algorithms = [
        (run_standard, "Standard GA"),
        (run_elitist, "Elitist GA"),
        (run_adaptive, "Adaptive GA"),
        (run_hybrid, "Hybrid GA")
    ]
    
    all_results = []
    
    print("="*70)
    print("COMPREHENSIVE GA COMPARISON ANALYSIS")
    print("="*70)
    
    for city in cities:
        print(f"\n{'='*70}")
        print(f"Dataset: {city}")
        print(f"{'='*70}")
        
        problem = create_problem_instance(city, num_emergencies, config['seed'])
        
        for algo_fn, algo_name in algorithms:
            result = run_experiment(algo_fn, algo_name, problem, config)
            all_results.append(result)
    
    return pd.DataFrame(all_results), config


def plot_convergence_comparison(results_df, output_dir="results"):
    """Generate convergence plots for all algorithms across datasets."""
    Path(output_dir).mkdir(exist_ok=True)
    
    cities = results_df['city'].unique()
    fig, axes = plt.subplots(1, len(cities), figsize=(6*len(cities), 5))
    
    if len(cities) == 1:
        axes = [axes]
    
    for idx, city in enumerate(cities):
        ax = axes[idx]
        city_data = results_df[results_df['city'] == city]
        
        for _, row in city_data.iterrows():
            if row['convergence'] and len(row['convergence']) > 0:
                ax.plot(row['convergence'], label=row['algorithm'], linewidth=2)
        
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Best Fitness (minutes)', fontsize=12)
        ax.set_title(f'{city.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{output_dir}/convergence_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved convergence plot: {plot_path}")
    plt.close()


def plot_performance_comparison(results_df, output_dir="results"):
    """Generate performance comparison charts."""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GA Variants Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Objective values by algorithm and city
    ax1 = axes[0, 0]
    results_pivot = results_df.pivot(index='city', columns='algorithm', values='objective')
    results_pivot.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_ylabel('Best Objective (minutes)', fontsize=11)
    ax1.set_title('Best Solution Quality', fontsize=12, fontweight='bold')
    ax1.legend(title='Algorithm', fontsize=9)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Execution time comparison
    ax2 = axes[0, 1]
    time_pivot = results_df.pivot(index='city', columns='algorithm', values='time_s')
    time_pivot.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_ylabel('Execution Time (seconds)', fontsize=11)
    ax2.set_title('Computational Efficiency', fontsize=12, fontweight='bold')
    ax2.legend(title='Algorithm', fontsize=9)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Relative performance (normalized)
    ax3 = axes[1, 0]
    for city in results_df['city'].unique():
        city_data = results_df[results_df['city'] == city]
        min_obj = city_data['objective'].min()
        city_data['relative_quality'] = (city_data['objective'] / min_obj - 1) * 100
        ax3.bar(city_data['algorithm'], city_data['relative_quality'], 
                alpha=0.7, label=city.replace('_', ' ').title())
    ax3.set_ylabel('% Worse than Best', fontsize=11)
    ax3.set_title('Relative Solution Quality', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    summary_data = results_df.groupby('algorithm').agg({
        'objective': ['mean', 'std'],
        'time_s': 'mean'
    }).round(3)
    
    ax4.axis('off')
    table_data = []
    for algo in summary_data.index:
        table_data.append([
            algo,
            f"{summary_data.loc[algo, ('objective', 'mean')]:.2f}",
            f"{summary_data.loc[algo, ('objective', 'std')]:.2f}",
            f"{summary_data.loc[algo, ('time_s', 'mean')]:.3f}"
        ])
    
    table = ax4.table(cellText=table_data,
                      colLabels=['Algorithm', 'Avg Obj', 'Std Dev', 'Avg Time (s)'],
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(len(table_data) + 1):
        cell = table[(i, 0)]
        cell.set_facecolor('#E8E8E8' if i == 0 else '#F5F5F5')
    
    ax4.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plot_path = f"{output_dir}/performance_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved performance comparison: {plot_path}")
    plt.close()


def generate_latex_table(results_df, output_dir="results"):
    """Generate LaTeX table for research paper."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create summary table
    summary = results_df.groupby(['city', 'algorithm']).agg({
        'objective': 'mean',
        'time_s': 'mean'
    }).reset_index()
    
    pivot_obj = summary.pivot(index='city', columns='algorithm', values='objective')
    pivot_time = summary.pivot(index='city', columns='algorithm', values='time_s')
    
    latex_str = "\\begin{table}[h]\n\\centering\n\\caption{GA Variants Performance Comparison}\n"
    latex_str += "\\label{tab:ga_comparison}\n\\begin{tabular}{l" + "c"*len(pivot_obj.columns) + "}\n\\hline\n"
    latex_str += "\\textbf{Dataset} & " + " & ".join([f"\\textbf{{{col}}}" for col in pivot_obj.columns]) + " \\\\\n\\hline\n"
    
    for city in pivot_obj.index:
        city_name = city.replace('_', ' ').title()
        row = [city_name] + [f"{pivot_obj.loc[city, col]:.2f}" for col in pivot_obj.columns]
        latex_str += " & ".join(row) + " \\\\\n"
    
    latex_str += "\\hline\n\\end{tabular}\n\\end{table}\n"
    
    latex_path = f"{output_dir}/ga_comparison_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_str)
    
    print(f"✓ Saved LaTeX table: {latex_path}")


def generate_analysis_report(results_df, config, output_dir="results"):
    """Generate comprehensive markdown analysis report."""
    Path(output_dir).mkdir(exist_ok=True)
    
    report = []
    report.append("# GA Variants Comparison Analysis Report\n")
    report.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Configuration:** Population={config['population_size']}, Generations={config['generations']}\n")
    
    report.append("\n## Executive Summary\n")
    
    # Find best algorithm overall
    best_algo = results_df.groupby('algorithm')['objective'].mean().idxmin()
    best_obj = results_df.groupby('algorithm')['objective'].mean().min()
    report.append(f"**Best Overall Algorithm:** {best_algo} (Avg: {best_obj:.2f} minutes)\n")
    
    # Dataset statistics
    report.append("\n## Datasets\n")
    for city in results_df['city'].unique():
        city_row = results_df[results_df['city'] == city].iloc[0]
        report.append(f"- **{city.replace('_', ' ').title()}:** {city_row['num_nodes']} nodes, "
                     f"{city_row['num_edges']} edges, {city_row['num_emergencies']} emergencies\n")
    
    report.append("\n## Results by Dataset\n")
    for city in results_df['city'].unique():
        report.append(f"\n### {city.replace('_', ' ').title()}\n")
        city_data = results_df[results_df['city'] == city].sort_values('objective')
        
        report.append("| Rank | Algorithm | Objective (min) | Time (s) | Status |\n")
        report.append("|------|-----------|-----------------|----------|--------|\n")
        for rank, (_, row) in enumerate(city_data.iterrows(), 1):
            report.append(f"| {rank} | {row['algorithm']} | {row['objective']:.2f} | "
                        f"{row['time_s']:.3f} | {row['status']} |\n")
    
    report.append("\n## Statistical Analysis\n")
    summary_stats = results_df.groupby('algorithm').agg({
        'objective': ['mean', 'std', 'min', 'max'],
        'time_s': ['mean', 'std']
    }).round(3)
    
    report.append("\n| Algorithm | Avg Obj | Std Dev | Min | Max | Avg Time (s) |\n")
    report.append("|-----------|---------|---------|-----|-----|-------------|\n")
    for algo in summary_stats.index:
        report.append(f"| {algo} | {summary_stats.loc[algo, ('objective', 'mean')]:.2f} | "
                     f"{summary_stats.loc[algo, ('objective', 'std')]:.2f} | "
                     f"{summary_stats.loc[algo, ('objective', 'min')]:.2f} | "
                     f"{summary_stats.loc[algo, ('objective', 'max')]:.2f} | "
                     f"{summary_stats.loc[algo, ('time_s', 'mean')]:.3f} |\n")
    
    report.append("\n## Key Findings\n")
    report.append(f"1. **Best Performance:** {best_algo} achieved the best average objective value\n")
    
    fastest_algo = results_df.groupby('algorithm')['time_s'].mean().idxmin()
    report.append(f"2. **Fastest Execution:** {fastest_algo} had the shortest average execution time\n")
    
    most_consistent = results_df.groupby('algorithm')['objective'].std().idxmin()
    report.append(f"3. **Most Consistent:** {most_consistent} showed the lowest variance across datasets\n")
    
    report.append("\n## Recommendations for Research Paper\n")
    report.append("- Use convergence plots to show algorithm behavior over generations\n")
    report.append("- Highlight trade-offs between solution quality and computational time\n")
    report.append("- Discuss scalability across different network sizes\n")
    report.append("- Include statistical significance tests if needed\n")
    
    report_path = f"{output_dir}/analysis_report.md"
    with open(report_path, 'w') as f:
        f.writelines(report)
    
    print(f"✓ Saved analysis report: {report_path}")


def main():
    """Run complete comparison analysis."""
    # Available datasets
    cities = ['koramangala', 'bangalore_central', 'cambridge_ma']
    
    # Configuration
    config = {
        'population_size': 100,
        'generations': 200,
        'crossover_rate': 0.8,
        'mutation_rate': 0.1,
        'tournament_size': 5,
        'seed': 42
    }
    
    # Run all experiments
    results_df, config = run_all_experiments(cities, num_emergencies=15, config=config)
    
    # Save raw results
    results_path = "results/ga_comparison_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Saved raw results: {results_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_convergence_comparison(results_df)
    plot_performance_comparison(results_df)
    
    # Generate tables
    print("\nGenerating tables...")
    generate_latex_table(results_df)
    
    # Generate report
    print("\nGenerating analysis report...")
    generate_analysis_report(results_df, config)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - results/ga_comparison_results.csv")
    print("  - results/convergence_comparison.png")
    print("  - results/performance_comparison.png")
    print("  - results/ga_comparison_table.tex")
    print("  - results/analysis_report.md")
    print("\nReview analysis_report.md for detailed findings!")


if __name__ == "__main__":
    main()
