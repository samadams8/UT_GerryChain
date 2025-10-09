#!/usr/bin/env python3
"""
Test contiguity of initial plans under transitability constraints.

This script helps debug why initial plans may not be contiguous
when using transitability-aware graphs.
"""

import sys
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append('.')

from utgc.transitability import build_transitable_graph
from gerrychain import Graph
import networkx as nx

def test_plan_contiguity(precincts_path, plan_path, sample_size=100):
    """
    Test if a redistricting plan is contiguous under transitability constraints.
    
    Parameters
    ----------
    precincts_path : str
        Path to precincts GeoJSON
    plan_path : str
        Path to redistricting plan shapefile
    sample_size : int
        Number of precincts to test with
    """
    print("=== TESTING PLAN CONTIGUITY UNDER TRANSITABILITY ===")
    
    # Load data
    print("Loading precincts...")
    precincts = gpd.read_file(precincts_path)
    
    if sample_size < len(precincts):
        print(f"Using sample of {sample_size} precincts...")
        sample_precincts = precincts.head(sample_size)
    else:
        sample_precincts = precincts
    
    print("Loading redistricting plan...")
    plan = gpd.read_file(plan_path)
    print(f"Plan has {len(plan)} districts")
    
    # Create spatial join to assign precincts to districts
    print("Assigning precincts to districts...")
    precincts_with_districts = gpd.sjoin(sample_precincts, plan, how='left', predicate='intersects')
    
    # Check for unassigned precincts
    unassigned = precincts_with_districts['index_right'].isna().sum()
    print(f"Unassigned precincts: {unassigned}")
    
    if unassigned > 0:
        print("⚠️  Some precincts are not assigned to any district!")
        return False
    
    # Test contiguity with standard adjacency
    print("\n=== TESTING WITH STANDARD ADJACENCY ===")
    standard_graph = Graph.from_geodataframe(sample_precincts)
    standard_contiguous = test_district_contiguity(standard_graph, precincts_with_districts, "Standard adjacency")
    
    # Test contiguity with transitability
    print("\n=== TESTING WITH TRANSITABILITY ===")
    transitability_params = {
        'enable': True,
        'remove_water_barriers': True,
        'verify_road_connectivity': True,
        'min_lake_size_sqkm': 1.0,
        'min_river_size_sqkm': 0.5,
        'road_buffer_meters': 500.0,
        'water_threshold': 0.5
    }
    
    transitable_graph = build_transitable_graph(sample_precincts, transitability_params)
    transitable_contiguous = test_district_contiguity(transitable_graph, precincts_with_districts, "Transitability")
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"Standard adjacency contiguity: {'✅ PASS' if standard_contiguous else '❌ FAIL'}")
    print(f"Transitability contiguity: {'✅ PASS' if transitable_contiguous else '❌ FAIL'}")
    
    if standard_contiguous and not transitable_contiguous:
        print("\n🔍 Transitability constraints are breaking plan contiguity!")
        print("   This suggests the plan relies on connections that are not transitable.")
    elif not standard_contiguous:
        print("\n⚠️  Plan is not contiguous even with standard adjacency.")
        print("   This suggests the plan itself has contiguity issues.")
    
    return transitable_contiguous

def test_district_contiguity(graph, precincts_with_districts, graph_type):
    """
    Test if all districts are contiguous under the given graph.
    
    Parameters
    ----------
    graph : Graph
        Graph to test contiguity on
    precincts_with_districts : gpd.GeoDataFrame
        Precincts with district assignments
    graph_type : str
        Type of graph for reporting
        
    Returns
    -------
    bool
        True if all districts are contiguous
    """
    print(f"Testing contiguity with {graph_type}...")
    
    # Get district assignments
    district_assignments = {}
    for idx, row in precincts_with_districts.iterrows():
        if pd.notna(row['index_right']):
            district_assignments[idx] = row['index_right']
    
    print(f"  Assigned precincts: {len(district_assignments)}")
    
    # Test each district
    districts = set(district_assignments.values())
    all_contiguous = True
    
    for district in districts:
        district_nodes = [node for node, dist in district_assignments.items() if dist == district]
        
        if len(district_nodes) <= 1:
            print(f"  District {district}: {len(district_nodes)} nodes (trivially contiguous)")
            continue
        
        # Create subgraph for this district
        subgraph = graph.subgraph(district_nodes)
        
        if nx.is_connected(subgraph):
            print(f"  District {district}: {len(district_nodes)} nodes ✅ CONTIGUOUS")
        else:
            components = list(nx.connected_components(subgraph))
            print(f"  District {district}: {len(district_nodes)} nodes ❌ DISCONTIGUOUS ({len(components)} components)")
            all_contiguous = False
    
    return all_contiguous

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test plan contiguity under transitability')
    parser.add_argument('--precincts', default='data/UT_precincts.geojson',
                       help='Path to precincts GeoJSON')
    parser.add_argument('--plan', required=True,
                       help='Path to redistricting plan shapefile')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Number of precincts to test with')
    
    args = parser.parse_args()
    
    try:
        contiguous = test_plan_contiguity(args.precincts, args.plan, args.sample_size)
        return 0 if contiguous else 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
