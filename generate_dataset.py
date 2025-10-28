"""
Dataset generation using OpenStreetMap data via osmnx.

This module fetches real road network data for multiple cities to provide
diverse datasets for validating GA performance across different network scales.
"""
import os
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
from pathlib import Path


# City configurations: (name, place_query, description)
CITY_CONFIGS = [
    # Small city - for quick testing and baseline
    ("koramangala", "Koramangala, Bangalore, India", "small_urban"),
    
    # Medium cities - diverse characteristics
    ("indiranagar", "Indiranagar, Bangalore, India", "medium_urban"),
    ("jayanagar", "Jayanagar, Bangalore, India", "medium_planned"),
    
    # Large city areas - for scalability testing
    ("bangalore_central", "Bangalore, Karnataka, India", "large_metropolitan"),
    
    # International comparison cities
    ("manhattan_midtown", "Midtown Manhattan, New York, USA", "large_grid"),
    ("cambridge_ma", "Cambridge, Massachusetts, USA", "medium_mixed"),
]


def fetch_city_network(place_name, network_type="drive", simplify=True):
    """Fetch road network from OSM using place name.
    
    Args:
        place_name: Geocodable place name (city, neighborhood, etc.)
        network_type: Type of network ('drive', 'walk', 'bike', 'all')
        simplify: Whether to simplify the graph topology
        
    Returns:
        networkx.MultiDiGraph: Road network graph
    """
    print(f"Fetching network for: {place_name}")
    try:
        # Configure osmnx
        ox.settings.use_cache = True
        ox.settings.log_console = True
        
        # Fetch graph from place name
        G = ox.graph_from_place(place_name, network_type=network_type, simplify=simplify)
        print(f"  ✓ Fetched {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    except Exception as e:
        print(f"  ✗ Error fetching {place_name}: {e}")
        return None


def fetch_bbox_network(north, south, east, west, network_type="drive", simplify=True):
    """Fetch road network from OSM using bounding box.
    
    Args:
        north, south, east, west: Bounding box coordinates (lat/lon)
        network_type: Type of network ('drive', 'walk', 'bike', 'all')
        simplify: Whether to simplify the graph topology
        
    Returns:
        networkx.MultiDiGraph: Road network graph
    """
    print(f"Fetching network for bbox: ({north}, {south}, {east}, {west})")
    try:
        ox.settings.use_cache = True
        ox.settings.log_console = True
        
        G = ox.graph_from_bbox(north, south, east, west, network_type=network_type, simplify=simplify)
        print(f"  ✓ Fetched {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    except Exception as e:
        print(f"  ✗ Error fetching bbox: {e}")
        return None


def add_traffic_weights(G, seed=42):
    """Add synthetic traffic weights to edges.
    
    Traffic weights simulate congestion levels (1-5 scale):
    1 = free flow, 5 = heavy congestion
    
    Args:
        G: networkx graph
        seed: random seed for reproducibility
        
    Returns:
        networkx graph with 'traffic_weight' edge attribute
    """
    np.random.seed(seed)
    
    for u, v, key, data in G.edges(keys=True, data=True):
        # Simulate traffic based on road type
        highway_type = data.get('highway', 'residential')
        
        if isinstance(highway_type, list):
            highway_type = highway_type[0]
        
        # Major roads tend to have more traffic
        if highway_type in ['motorway', 'trunk', 'primary']:
            base_traffic = np.random.uniform(2.5, 4.5)
        elif highway_type in ['secondary', 'tertiary']:
            base_traffic = np.random.uniform(2.0, 3.5)
        else:
            base_traffic = np.random.uniform(1.0, 2.5)
        
        # Add some randomness
        traffic = np.clip(base_traffic + np.random.normal(0, 0.3), 1.0, 5.0)
        G[u][v][key]['traffic_weight'] = round(traffic, 2)
    
    return G


def graph_to_edge_dataframe(G, add_traffic=True):
    """Convert networkx graph to edge list DataFrame.
    
    Args:
        G: networkx graph with geometry and length attributes
        add_traffic: Whether to add synthetic traffic weights
        
    Returns:
        pandas.DataFrame with columns: from_node, to_node, distance_km, 
                                       traffic_weight, travel_time_min, highway_type
    """
    if add_traffic:
        G = add_traffic_weights(G)
    
    edges = []
    for u, v, key, data in G.edges(keys=True, data=True):
        # Get distance in km
        distance_m = data.get('length', 0)
        distance_km = distance_m / 1000.0
        
        # Get traffic weight (1-5)
        traffic = data.get('traffic_weight', 1.5)
        
        # Calculate travel time (assuming base speed 40 km/h, reduced by traffic)
        # TravelTime = Distance / (BaseSpeed / (1 + 0.2 * Traffic))
        base_speed_kmh = 40
        effective_speed = base_speed_kmh / (1 + 0.2 * traffic)
        travel_time_h = distance_km / effective_speed
        travel_time_min = travel_time_h * 60
        
        highway_type = data.get('highway', 'unknown')
        if isinstance(highway_type, list):
            highway_type = highway_type[0]
        
        edges.append({
            'from_node': u,
            'to_node': v,
            'edge_key': key,
            'distance_km': round(distance_km, 3),
            'traffic_weight': traffic,
            'travel_time_min': round(travel_time_min, 2),
            'highway_type': highway_type,
            'maxspeed': data.get('maxspeed', 'unknown')
        })
    
    return pd.DataFrame(edges)


def save_dataset(G, edge_df, city_name, output_dir="datasets"):
    """Save graph and edge list to files.
    
    Args:
        G: networkx graph
        edge_df: pandas DataFrame with edge list
        city_name: name for output files
        output_dir: directory to save datasets
        
    Returns:
        dict with paths to saved files
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save edge list as CSV
    csv_path = os.path.join(output_dir, f"{city_name}_edges.csv")
    edge_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved edge list: {csv_path}")
    
    # Save graph as GraphML (for visualization/further processing)
    graphml_path = os.path.join(output_dir, f"{city_name}_graph.graphml")
    ox.save_graphml(G, graphml_path)
    print(f"  ✓ Saved graph: {graphml_path}")
    
    # Save node coordinates
    nodes = []
    for node, data in G.nodes(data=True):
        nodes.append({
            'node_id': node,
            'lat': data.get('y', 0),
            'lon': data.get('x', 0)
        })
    nodes_df = pd.DataFrame(nodes)
    nodes_path = os.path.join(output_dir, f"{city_name}_nodes.csv")
    nodes_df.to_csv(nodes_path, index=False)
    print(f"  ✓ Saved nodes: {nodes_path}")
    
    # Save metadata
    metadata = {
        'city': city_name,
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'is_strongly_connected': nx.is_strongly_connected(G),
        'num_strongly_connected_components': nx.number_strongly_connected_components(G)
    }
    
    return {
        'csv': csv_path,
        'graphml': graphml_path,
        'nodes': nodes_path,
        'metadata': metadata
    }


def generate_all_datasets(output_dir="datasets"):
    """Generate datasets for all configured cities.
    
    Args:
        output_dir: directory to save datasets
        
    Returns:
        list of metadata dicts for all generated datasets
    """
    print("=" * 60)
    print("GENERATING DATASETS FOR RESEARCH")
    print("=" * 60)
    
    all_metadata = []
    
    for city_name, place_query, description in CITY_CONFIGS:
        print(f"\n[{description}] {city_name}")
        print("-" * 60)
        
        # Fetch network
        G = fetch_city_network(place_query, network_type="drive", simplify=True)
        
        if G is None:
            print(f"  ✗ Skipping {city_name} due to fetch error")
            continue
        
        # Convert to edge dataframe
        edge_df = graph_to_edge_dataframe(G, add_traffic=True)
        
        # Save dataset
        result = save_dataset(G, edge_df, city_name, output_dir)
        result['metadata']['description'] = description
        result['metadata']['place_query'] = place_query
        all_metadata.append(result['metadata'])
    
    # Save summary
    summary_df = pd.DataFrame(all_metadata)
    summary_path = os.path.join(output_dir, "datasets_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\n" + "=" * 60)
    print(f"✓ All datasets generated. Summary: {summary_path}")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    
    return all_metadata


if __name__ == "__main__":
    # Generate all datasets
    metadata = generate_all_datasets()
