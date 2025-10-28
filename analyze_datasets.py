"""
Analyze generated datasets and produce research-ready statistics.

This script loads the generated city datasets and computes key metrics
for inclusion in research papers: network characteristics, connectivity,
and complexity measures.
"""
import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path


def load_dataset(city_name, data_dir="datasets"):
    """Load a city dataset from CSV files.
    
    Args:
        city_name: name of the city dataset
        data_dir: directory containing datasets
        
    Returns:
        tuple: (nodes_df, edges_df)
    """
    nodes_path = os.path.join(data_dir, f"{city_name}_nodes.csv")
    edges_path = os.path.join(data_dir, f"{city_name}_edges.csv")
    
    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        print(f"Dataset not found for {city_name}")
        return None, None
    
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    
    return nodes_df, edges_df


def compute_dataset_statistics(city_name, nodes_df, edges_df):
    """Compute comprehensive statistics for a dataset.
    
    Args:
        city_name: name of the city
        nodes_df: DataFrame with nodes
        edges_df: DataFrame with edges
        
    Returns:
        dict with statistics
    """
    stats = {
        'city': city_name,
        'num_nodes': len(nodes_df),
        'num_edges': len(edges_df),
        'avg_degree': (2 * len(edges_df)) / len(nodes_df) if len(nodes_df) > 0 else 0,
    }
    
    # Edge statistics
    stats['total_distance_km'] = edges_df['distance_km'].sum()
    stats['avg_edge_length_km'] = edges_df['distance_km'].mean()
    stats['median_edge_length_km'] = edges_df['distance_km'].median()
    stats['max_edge_length_km'] = edges_df['distance_km'].max()
    
    # Traffic statistics
    stats['avg_traffic_weight'] = edges_df['traffic_weight'].mean()
    stats['min_traffic_weight'] = edges_df['traffic_weight'].min()
    stats['max_traffic_weight'] = edges_df['traffic_weight'].max()
    
    # Travel time statistics
    stats['avg_travel_time_min'] = edges_df['travel_time_min'].mean()
    stats['total_travel_time_h'] = edges_df['travel_time_min'].sum() / 60
    
    # Network density (edges / possible_edges)
    n = len(nodes_df)
    if n > 1:
        stats['network_density'] = len(edges_df) / (n * (n - 1))
    else:
        stats['network_density'] = 0
    
    return stats


def analyze_all_datasets(data_dir="datasets", output_dir="results"):
    """Analyze all datasets in the data directory.
    
    Args:
        data_dir: directory containing datasets
        output_dir: directory to save analysis results
        
    Returns:
        DataFrame with statistics for all datasets
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Find all edge files
    edge_files = list(Path(data_dir).glob("*_edges.csv"))
    
    if not edge_files:
        print(f"No datasets found in {data_dir}")
        return None
    
    print("=" * 60)
    print("DATASET ANALYSIS FOR RESEARCH")
    print("=" * 60)
    
    all_stats = []
    
    for edge_file in edge_files:
        city_name = edge_file.stem.replace("_edges", "")
        print(f"\nAnalyzing: {city_name}")
        print("-" * 60)
        
        nodes_df, edges_df = load_dataset(city_name, data_dir)
        
        if nodes_df is None or edges_df is None:
            continue
        
        stats = compute_dataset_statistics(city_name, nodes_df, edges_df)
        all_stats.append(stats)
        
        # Print key metrics
        print(f"  Nodes: {stats['num_nodes']:,}")
        print(f"  Edges: {stats['num_edges']:,}")
        print(f"  Avg Degree: {stats['avg_degree']:.2f}")
        print(f"  Total Distance: {stats['total_distance_km']:.2f} km")
        print(f"  Network Density: {stats['network_density']:.6f}")
        print(f"  Avg Traffic: {stats['avg_traffic_weight']:.2f}")
    
    # Create summary DataFrame
    stats_df = pd.DataFrame(all_stats)
    
    # Sort by network size
    stats_df = stats_df.sort_values('num_nodes')
    
    # Save detailed statistics
    stats_path = os.path.join(output_dir, "dataset_statistics.csv")
    stats_df.to_csv(stats_path, index=False)
    print("\n" + "=" * 60)
    print(f"✓ Statistics saved: {stats_path}")
    print("=" * 60)
    print("\nSUMMARY TABLE (for research paper):")
    print(stats_df[['city', 'num_nodes', 'num_edges', 'avg_degree', 
                     'total_distance_km', 'network_density']].to_string(index=False))
    
    return stats_df


def plot_dataset_comparison(stats_df, output_dir="results"):
    """Create comparison plots for datasets.
    
    Args:
        stats_df: DataFrame with dataset statistics
        output_dir: directory to save plots
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Dataset Characteristics Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Network size
    ax1 = axes[0, 0]
    ax1.bar(stats_df['city'], stats_df['num_nodes'], alpha=0.7, label='Nodes')
    ax1.bar(stats_df['city'], stats_df['num_edges'], alpha=0.7, label='Edges')
    ax1.set_ylabel('Count')
    ax1.set_title('Network Size (Nodes & Edges)')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Average degree
    ax2 = axes[0, 1]
    ax2.bar(stats_df['city'], stats_df['avg_degree'], color='green', alpha=0.7)
    ax2.set_ylabel('Average Degree')
    ax2.set_title('Network Connectivity')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Network density
    ax3 = axes[1, 0]
    ax3.bar(stats_df['city'], stats_df['network_density'], color='purple', alpha=0.7)
    ax3.set_ylabel('Density')
    ax3.set_title('Network Density')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Total distance
    ax4 = axes[1, 1]
    ax4.bar(stats_df['city'], stats_df['total_distance_km'], color='orange', alpha=0.7)
    ax4.set_ylabel('Distance (km)')
    ax4.set_title('Total Road Network Distance')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "dataset_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved: {plot_path}")
    plt.close()


def generate_latex_table(stats_df, output_dir="results"):
    """Generate LaTeX table for research paper.
    
    Args:
        stats_df: DataFrame with dataset statistics
        output_dir: directory to save table
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Select columns for paper
    paper_cols = ['city', 'num_nodes', 'num_edges', 'avg_degree', 
                  'total_distance_km', 'network_density', 'avg_traffic_weight']
    
    table_df = stats_df[paper_cols].copy()
    
    # Rename for paper
    table_df.columns = ['City', 'Nodes', 'Edges', 'Avg. Degree', 
                        'Distance (km)', 'Density', 'Avg. Traffic']
    
    # Format numbers
    table_df['Nodes'] = table_df['Nodes'].apply(lambda x: f"{x:,}")
    table_df['Edges'] = table_df['Edges'].apply(lambda x: f"{x:,}")
    table_df['Avg. Degree'] = table_df['Avg. Degree'].apply(lambda x: f"{x:.2f}")
    table_df['Distance (km)'] = table_df['Distance (km)'].apply(lambda x: f"{x:.2f}")
    table_df['Density'] = table_df['Density'].apply(lambda x: f"{x:.6f}")
    table_df['Avg. Traffic'] = table_df['Avg. Traffic'].apply(lambda x: f"{x:.2f}")
    
    # Generate LaTeX
    latex_str = table_df.to_latex(index=False, escape=False, 
                                   caption="Dataset Characteristics for Emergency Routing Experiments",
                                   label="tab:datasets")
    
    latex_path = os.path.join(output_dir, "datasets_table.tex")
    with open(latex_path, 'w') as f:
        f.write(latex_str)
    
    print(f"✓ LaTeX table saved: {latex_path}")


if __name__ == "__main__":
    # Analyze all datasets
    stats_df = analyze_all_datasets()
    
    if stats_df is not None and len(stats_df) > 0:
        # Generate visualizations
        plot_dataset_comparison(stats_df)
        
        # Generate LaTeX table
        generate_latex_table(stats_df)
        
        print("\n✓ Analysis complete! Ready for research paper.")
