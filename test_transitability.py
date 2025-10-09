#!/usr/bin/env python3
"""
Test and visualize transitability graph construction.

This script helps debug transitability issues by:
1. Testing graph connectivity at each step
2. Visualizing the graph construction process
3. Identifying contiguity issues with initial plans
4. Providing detailed analysis of edge removal

Usage:
    python test_transitability.py [--sample-size N] [--save-plots]
"""

import argparse
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import networkx as nx
from pathlib import Path
import sys
import warnings

# Add project root to path
sys.path.append('.')

from utgc.transitability import (
    load_road_network, load_water_bodies, identify_road_connected_precincts,
    apply_hierarchical_fallback, identify_water_crossings, build_transitable_graph,
    analyze_transitability_impact
)
from gerrychain import Graph
from gerrychain.graph.graph import FrozenGraph

def test_graph_connectivity(graph, step_name="Graph", raise_error=True):
    """
    Test if graph is fully connected and optionally raise error if not.
    
    Parameters
    ----------
    graph : Graph or networkx.Graph
        Graph to test
    step_name : str
        Name of the step for error reporting
    raise_error : bool
        Whether to raise error if not connected
        
    Returns
    -------
    bool
        True if connected, False otherwise
    """
    if isinstance(graph, Graph):
        nx_graph = graph
    else:
        nx_graph = graph
    
    # Check connectivity
    if not nx.is_connected(nx_graph):
        components = list(nx.connected_components(nx_graph))
        component_sizes = [len(comp) for comp in components]
        
        error_msg = f"{step_name} is not fully connected! Found {len(components)} components: {component_sizes}"
        print(f"❌ {error_msg}")
        
        # Additional debugging info
        largest_component = max(components, key=len)
        isolated_components = [comp for comp in components if len(comp) == 1]
        
        print(f"   Largest component: {len(largest_component)} nodes")
        print(f"   Isolated nodes: {len(isolated_components)}")
        
        if isolated_components:
            isolated_nodes = [list(comp)[0] for comp in isolated_components]
            print(f"   Isolated node indices: {isolated_nodes[:10]}{'...' if len(isolated_nodes) > 10 else ''}")
        
        if raise_error:
            raise ValueError(error_msg)
        return False
    else:
        print(f"✅ {step_name} is fully connected ({len(nx_graph.nodes)} nodes, {len(nx_graph.edges)} edges)")
        return True

def visualize_graph_step(precincts, graph, step_name, save_path=None, figsize=(15, 10)):
    """
    Visualize a graph at a specific step of the transitability process.
    
    Parameters
    ----------
    precincts : gpd.GeoDataFrame
        Precinct geometries
    graph : Graph or networkx.Graph
        Graph to visualize
    step_name : str
        Name of the step
    save_path : str, optional
        Path to save the plot
    figsize : tuple
        Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Map view with edges
    precincts.plot(ax=ax1, color='lightblue', edgecolor='black', linewidth=0.5, alpha=0.7)
    
    # Add edges
    for edge in graph.edges():
        try:
            # Get precinct geometries
            p1_geom = precincts.loc[edge[0], 'geometry']
            p2_geom = precincts.loc[edge[1], 'geometry']
            
            # Draw line between centroids
            x_coords = [p1_geom.centroid.x, p2_geom.centroid.x]
            y_coords = [p1_geom.centroid.y, p2_geom.centroid.y]
            ax1.plot(x_coords, y_coords, 'r-', alpha=0.6, linewidth=1)
        except (KeyError, IndexError):
            # Handle node index issues
            continue
    
    ax1.set_title(f'{step_name} - Map View')
    ax1.set_aspect('equal')
    
    # Plot 2: Network visualization
    pos = nx.spring_layout(graph, k=1, iterations=50)
    nx.draw(graph, pos, ax=ax2, node_size=50, node_color='lightblue', 
            edge_color='red', alpha=0.7, with_labels=False)
    ax2.set_title(f'{step_name} - Network View')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")
    
    plt.show()

def analyze_edge_removal(base_graph, final_graph, precincts):
    """
    Analyze which edges were removed and why.
    
    Parameters
    ----------
    base_graph : Graph
        Original adjacency graph
    final_graph : Graph
        Transitability-modified graph
    precincts : gpd.GeoDataFrame
        Precinct geometries
    """
    print("\n=== EDGE REMOVAL ANALYSIS ===")
    
    base_edges = set(base_graph.edges())
    final_edges = set(final_graph.edges())
    removed_edges = base_edges - final_edges
    
    print(f"Original edges: {len(base_edges)}")
    print(f"Final edges: {len(final_edges)}")
    print(f"Removed edges: {len(removed_edges)} ({len(removed_edges)/len(base_edges)*100:.1f}%)")
    
    if removed_edges:
        print(f"\nSample removed edges: {list(removed_edges)[:10]}")
        
        # Analyze removed edges by distance
        distances = []
        for edge in list(removed_edges)[:50]:  # Sample first 50
            try:
                p1_geom = precincts.loc[edge[0], 'geometry']
                p2_geom = precincts.loc[edge[1], 'geometry']
                distance = p1_geom.centroid.distance(p2_geom.centroid)
                distances.append(distance)
            except (KeyError, IndexError):
                continue
        
        if distances:
            print(f"Removed edge distances: min={min(distances):.0f}m, max={max(distances):.0f}m, mean={np.mean(distances):.0f}m")

def test_initial_plan_contiguity(precincts, initial_plan_path, graph):
    """
    Test if the initial plan is contiguous under the modified graph.
    
    Parameters
    ----------
    precincts : gpd.GeoDataFrame
        Precinct geometries
    initial_plan_path : str
        Path to initial plan shapefile
    graph : Graph
        Transitability-modified graph
        
    Returns
    -------
    bool
        True if plan is contiguous, False otherwise
    """
    print("\n=== TESTING INITIAL PLAN CONTIGUITY ===")
    
    try:
        # Load initial plan
        initial_plan = gpd.read_file(initial_plan_path)
        print(f"Loaded initial plan: {len(initial_plan)} districts")
        
        # Create assignment mapping
        # This is a simplified approach - in practice you'd need proper spatial join
        assignment = {}
        for idx, precinct in precincts.iterrows():
            # Find which district contains this precinct
            # This is a placeholder - you'd need proper spatial intersection
            assignment[idx] = 0  # Placeholder: assign all to district 0
        
        # Test contiguity for each district
        districts = set(assignment.values())
        contiguous = True
        
        for district in districts:
            district_nodes = [node for node, dist in assignment.items() if dist == district]
            if len(district_nodes) > 1:
                # Create subgraph for this district
                subgraph = graph.subgraph(district_nodes)
                if not nx.is_connected(subgraph):
                    print(f"❌ District {district} is not contiguous ({len(district_nodes)} nodes)")
                    contiguous = False
                else:
                    print(f"✅ District {district} is contiguous ({len(district_nodes)} nodes)")
        
        return contiguous
        
    except Exception as e:
        print(f"❌ Error testing initial plan contiguity: {e}")
        return False

def run_transitability_test(sample_size=100, save_plots=False, initial_plan_path=None):
    """
    Run comprehensive transitability test with visualizations.
    
    Parameters
    ----------
    sample_size : int
        Number of precincts to test with
    save_plots : bool
        Whether to save visualization plots
    initial_plan_path : str, optional
        Path to initial plan for contiguity testing
    """
    print("=== TRANSITABILITY TESTING AND VISUALIZATION ===")
    
    # Load data
    print("Loading precincts...")
    precincts = gpd.read_file('data/UT_precincts.geojson')
    
    if sample_size < len(precincts):
        print(f"Using sample of {sample_size} precincts...")
        sample_precincts = precincts.head(sample_size)
    else:
        sample_precincts = precincts
    
    print(f"Testing with {len(sample_precincts)} precincts")
    
    # Step 1: Create base adjacency graph
    print("\n=== STEP 1: BASE ADJACENCY GRAPH ===")
    base_graph = Graph.from_geodataframe(sample_precincts)
    base_connected = test_graph_connectivity(base_graph, "Base adjacency graph", raise_error=False)
    
    if save_plots:
        visualize_graph_step(sample_precincts, base_graph, "Base Adjacency", 
                            save_path="plots/base_adjacency.png")
    
    # Step 2: Load datasets
    print("\n=== STEP 2: LOADING DATASETS ===")
    roads = load_road_network()
    water_bodies = load_water_bodies()
    
    # Step 3: Road connectivity analysis
    print("\n=== STEP 3: ROAD CONNECTIVITY ANALYSIS ===")
    road_connections = identify_road_connected_precincts(sample_precincts, roads, buffer_meters=500.0)
    
    # Step 4: Hierarchical fallback
    print("\n=== STEP 4: HIERARCHICAL FALLBACK ===")
    connections = apply_hierarchical_fallback(sample_precincts, road_connections, base_graph)
    
    # Step 5: Water barrier analysis
    print("\n=== STEP 5: WATER BARRIER ANALYSIS ===")
    connections_with_water = identify_water_crossings(
        sample_precincts, water_bodies, connections, base_graph, water_threshold=0.5
    )
    
    # Step 6: Build final graph
    print("\n=== STEP 6: BUILDING FINAL GRAPH ===")
    final_graph = Graph()
    
    # Add all nodes
    for node in base_graph.nodes():
        final_graph.add_node(node, **base_graph.nodes[node])
    
    # Add only valid connections
    valid_connections = connections_with_water[connections_with_water['connected'] == True]
    for _, row in valid_connections.iterrows():
        final_graph.add_edge(row['precinct_1'], row['precinct_2'])
    
    # Test final connectivity
    final_connected = test_graph_connectivity(final_graph, "Final transitability graph", raise_error=False)
    
    if save_plots:
        visualize_graph_step(sample_precincts, final_graph, "Final Transitability", 
                            save_path="plots/final_transitability.png")
    
    # Analyze edge removal
    analyze_edge_removal(base_graph, final_graph, sample_precincts)
    
    # Test initial plan contiguity if provided
    if initial_plan_path:
        test_initial_plan_contiguity(sample_precincts, initial_plan_path, final_graph)
    
    # Summary
    print("\n=== SUMMARY ===")
    impact = analyze_transitability_impact(base_graph, final_graph)
    print(f"Original edges: {impact['original_edges']:,}")
    print(f"Final edges: {impact['transitable_edges']:,}")
    print(f"Edges removed: {impact['edges_removed']:,} ({impact['removal_percentage']:.1f}%)")
    print(f"Connectivity preserved: {impact['connectivity_preserved']}")
    
    return final_graph, impact

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Test and visualize transitability graph construction')
    parser.add_argument('--sample-size', type=int, default=100, 
                       help='Number of precincts to test with (default: 100)')
    parser.add_argument('--save-plots', action='store_true', 
                       help='Save visualization plots to plots/ directory')
    parser.add_argument('--initial-plan', type=str, 
                       help='Path to initial plan shapefile for contiguity testing')
    
    args = parser.parse_args()
    
    # Create plots directory if saving
    if args.save_plots:
        Path("plots").mkdir(exist_ok=True)
    
    # Run test
    try:
        final_graph, impact = run_transitability_test(
            sample_size=args.sample_size,
            save_plots=args.save_plots,
            initial_plan_path=args.initial_plan
        )
        print("\n✅ Transitability test completed successfully!")
        
    except ValueError as e:
        print(f"\n❌ Transitability test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
