#!/usr/bin/env python3
"""
Visualize transitability edge removal at each step.

This script creates detailed visualizations showing:
1. Base adjacency graph
2. After road connectivity analysis
3. After water barrier removal
4. Final graph with edge removal analysis
"""

import sys
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import networkx as nx
from pathlib import Path

# Add project root to path
sys.path.append('.')

from utgc.transitability import (
    load_road_network, load_water_bodies, identify_road_connected_precincts,
    apply_hierarchical_fallback, identify_water_crossings
)
from gerrychain import Graph

def visualize_edge_removal_steps(precincts, sample_size=200, save_plots=True, water_threshold=0.5):
    """
    Visualize edge removal at each step of transitability analysis.
    
    Parameters
    ----------
    precincts : gpd.GeoDataFrame
        Precinct geometries
    sample_size : int
        Number of precincts to visualize
    save_plots : bool
        Whether to save plots
    """
    print("=== VISUALIZING TRANSITABILITY STEPS ===")
    
    # Use sample for visualization
    if sample_size < len(precincts):
        sample_precincts = precincts.head(sample_size)
    else:
        sample_precincts = precincts
    
    print(f"Visualizing with {len(sample_precincts)} precincts")
    
    # Step 1: Base adjacency graph
    print("\n=== STEP 1: BASE ADJACENCY GRAPH ===")
    base_graph = Graph.from_geodataframe(sample_precincts)
    print(f"Base graph: {len(base_graph.nodes)} nodes, {len(base_graph.edges)} edges")
    
    # Step 2: Load datasets
    print("\n=== STEP 2: LOADING DATASETS ===")
    roads = load_road_network()
    water_bodies = load_water_bodies()
    
    # Step 3: Road connectivity analysis
    print("\n=== STEP 3: ROAD CONNECTIVITY ANALYSIS ===")
    road_connections = identify_road_connected_precincts(sample_precincts, roads, buffer_meters=500.0)
    
    # Create graph after road analysis
    road_graph = Graph()
    for node in base_graph.nodes():
        road_graph.add_node(node, **base_graph.nodes[node])
    
    for _, row in road_connections.iterrows():
        if row['connected']:
            road_graph.add_edge(row['precinct_1'], row['precinct_2'])
    
    print(f"Road graph: {len(road_graph.nodes)} nodes, {len(road_graph.edges)} edges")
    road_removed = len(base_graph.edges) - len(road_graph.edges)
    print(f"Edges removed by road analysis: {road_removed}")
    
    # Step 4: Hierarchical fallback
    print("\n=== STEP 4: HIERARCHICAL FALLBACK ===")
    connections = apply_hierarchical_fallback(sample_precincts, road_connections, base_graph)
    
    # Step 5: Water barrier analysis
    print("\n=== STEP 5: WATER BARRIER ANALYSIS ===")
    connections_with_water = identify_water_crossings(
        sample_precincts, water_bodies, connections, base_graph, water_threshold=water_threshold
    )
    
    # Create final graph
    final_graph = Graph()
    for node in base_graph.nodes():
        final_graph.add_node(node, **base_graph.nodes[node])
    
    valid_connections = connections_with_water[connections_with_water['connected'] == True]
    for _, row in valid_connections.iterrows():
        final_graph.add_edge(row['precinct_1'], row['precinct_2'])
    
    print(f"Final graph: {len(final_graph.nodes)} nodes, {len(final_graph.edges)} edges")
    water_removed = len(road_graph.edges) - len(final_graph.edges)
    total_removed = len(base_graph.edges) - len(final_graph.edges)
    print(f"Edges removed by water analysis: {water_removed}")
    print(f"Total edges removed: {total_removed}")
    
    # Create comprehensive visualization
    create_step_visualization(
        sample_precincts, base_graph, road_graph, final_graph,
        connections_with_water, save_plots, water_threshold=water_threshold
    )
    
    # Analyze edge removal by type
    analyze_edge_removal_by_type(base_graph, road_graph, final_graph, connections_with_water)
    
    return base_graph, road_graph, final_graph, connections_with_water

def create_step_visualization(precincts, base_graph, road_graph, final_graph, 
                            connections_df, save_plots=True, water_threshold=0.5):
    """Create comprehensive visualization of all steps."""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    # Plot 1: Base adjacency
    ax = axes[0]
    precincts.plot(ax=ax, color='lightblue', edgecolor='black', linewidth=0.5, alpha=0.7)
    plot_graph_edges(precincts, base_graph, ax, color='blue', alpha=0.6, linewidth=1)
    ax.set_title(f'Base Adjacency\n{len(base_graph.edges)} edges', fontsize=12)
    ax.set_aspect('equal')
    
    # Plot 2: After road analysis
    ax = axes[1]
    precincts.plot(ax=ax, color='lightblue', edgecolor='black', linewidth=0.5, alpha=0.7)
    plot_graph_edges(precincts, road_graph, ax, color='green', alpha=0.6, linewidth=1)
    ax.set_title(f'After Road Analysis\n{len(road_graph.edges)} edges', fontsize=12)
    ax.set_aspect('equal')
    
    # Plot 3: Final graph
    ax = axes[2]
    precincts.plot(ax=ax, color='lightblue', edgecolor='black', linewidth=0.5, alpha=0.7)
    plot_graph_edges(precincts, final_graph, ax, color='red', alpha=0.6, linewidth=1)
    ax.set_title(f'Final Graph\n{len(final_graph.edges)} edges', fontsize=12)
    ax.set_aspect('equal')
    
    # Plot 4: Edges removed by road analysis
    ax = axes[3]
    precincts.plot(ax=ax, color='lightblue', edgecolor='black', linewidth=0.5, alpha=0.7)
    plot_removed_edges(precincts, base_graph, road_graph, ax, color='orange', alpha=0.8, linewidth=2)
    ax.set_title('Edges Removed by Road Analysis', fontsize=12)
    ax.set_aspect('equal')
    
    # Plot 5: Edges removed by water analysis
    ax = axes[4]
    precincts.plot(ax=ax, color='lightblue', edgecolor='black', linewidth=0.5, alpha=0.7)
    plot_water_removed_edges(precincts, connections_df, ax, color='purple', alpha=0.8, linewidth=2)
    ax.set_title('Edges Removed by Water Analysis', fontsize=12)
    ax.set_aspect('equal')
    
    # Plot 6: Summary
    ax = axes[5]
    ax.axis('off')
    
    # Create summary text
    road_removed = len(base_graph.edges) - len(road_graph.edges)
    water_removed = len(road_graph.edges) - len(final_graph.edges)
    total_removed = len(base_graph.edges) - len(final_graph.edges)
    
    summary_text = f"""
TRANSITABILITY ANALYSIS SUMMARY

Base Graph:
  • Nodes: {len(base_graph.nodes):,}
  • Edges: {len(base_graph.edges):,}

Road Analysis:
  • Edges removed: {road_removed:,}
  • Remaining edges: {len(road_graph.edges):,}

Water Analysis:
  • Edges removed: {water_removed:,}
  • Remaining edges: {len(final_graph.edges):,}
  • Water threshold: {water_threshold}

Total Impact:
  • Total removed: {total_removed:,} ({total_removed/len(base_graph.edges)*100:.1f}%)
  • Final edges: {len(final_graph.edges):,}
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('plots/transitability_steps_analysis.png', dpi=300, bbox_inches='tight')
        print("Saved comprehensive analysis: plots/transitability_steps_analysis.png")
    
    plt.show()

def plot_graph_edges(precincts, graph, ax, color='blue', alpha=0.6, linewidth=1):
    """Plot graph edges on the map."""
    for edge in graph.edges():
        try:
            p1_geom = precincts.loc[edge[0], 'geometry']
            p2_geom = precincts.loc[edge[1], 'geometry']
            
            x_coords = [p1_geom.centroid.x, p2_geom.centroid.x]
            y_coords = [p1_geom.centroid.y, p2_geom.centroid.y]
            ax.plot(x_coords, y_coords, color=color, alpha=alpha, linewidth=linewidth)
        except (KeyError, IndexError):
            continue

def plot_removed_edges(precincts, base_graph, road_graph, ax, color='orange', alpha=0.8, linewidth=2):
    """Plot edges that were removed by road analysis."""
    base_edges = set(base_graph.edges())
    road_edges = set(road_graph.edges())
    removed_edges = base_edges - road_edges
    
    for edge in removed_edges:
        try:
            p1_geom = precincts.loc[edge[0], 'geometry']
            p2_geom = precincts.loc[edge[1], 'geometry']
            
            x_coords = [p1_geom.centroid.x, p2_geom.centroid.x]
            y_coords = [p1_geom.centroid.y, p2_geom.centroid.y]
            ax.plot(x_coords, y_coords, color=color, alpha=alpha, linewidth=linewidth)
        except (KeyError, IndexError):
            continue

def plot_water_removed_edges(precincts, connections_df, ax, color='purple', alpha=0.8, linewidth=2):
    """Plot edges that were removed by water analysis."""
    water_removed = connections_df[connections_df['water_barrier'] == True]
    
    for _, row in water_removed.iterrows():
        try:
            p1_geom = precincts.loc[row['precinct_1'], 'geometry']
            p2_geom = precincts.loc[row['precinct_2'], 'geometry']
            
            x_coords = [p1_geom.centroid.x, p2_geom.centroid.x]
            y_coords = [p1_geom.centroid.y, p2_geom.centroid.y]
            ax.plot(x_coords, y_coords, color=color, alpha=alpha, linewidth=linewidth)
        except (KeyError, IndexError):
            continue

def analyze_edge_removal_by_type(base_graph, road_graph, final_graph, connections_df):
    """Analyze which types of edges are removed."""
    print("\n=== EDGE REMOVAL ANALYSIS BY TYPE ===")
    
    # Road analysis removal
    base_edges = set(base_graph.edges())
    road_edges = set(road_graph.edges())
    road_removed = base_edges - road_edges
    
    # Water analysis removal
    water_removed = connections_df[connections_df['water_barrier'] == True]
    water_removed_edges = set(zip(water_removed['precinct_1'], water_removed['precinct_2']))
    
    print(f"Edges removed by road analysis: {len(road_removed)}")
    print(f"Edges removed by water analysis: {len(water_removed_edges)}")
    
    # Analyze distances of removed edges
    if road_removed:
        road_distances = []
        for edge in list(road_removed)[:50]:  # Sample first 50
            try:
                # This would need precincts data - placeholder for now
                road_distances.append(0)  # Placeholder
            except:
                continue
        
        print(f"Road-removed edges: {len(road_distances)} analyzed")
    
    if len(water_removed_edges) > 0:
        print(f"Water-removed edges: {len(water_removed_edges)}")
        print("Sample water-removed edges:", list(water_removed_edges)[:10])

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize transitability edge removal steps')
    parser.add_argument('--sample-size', type=int, default=200,
                       help='Number of precincts to visualize (default: 200)')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save visualization plots')
    parser.add_argument('--water-threshold', type=float, default=0.5,
                       help='Fraction of line length crossing water required to remove edge (default: 0.5)')
    
    args = parser.parse_args()
    
    # Create plots directory
    if args.save_plots:
        Path("plots").mkdir(exist_ok=True)
    
    # Load precincts
    print("Loading precincts...")
    precincts = gpd.read_file('data/UT_precincts.geojson')
    
    try:
        base_graph, road_graph, final_graph, connections_df = visualize_edge_removal_steps(
            precincts, sample_size=args.sample_size, save_plots=args.save_plots,
            water_threshold=args.water_threshold
        )
        print("\n✅ Visualization completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
