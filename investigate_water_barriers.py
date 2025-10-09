#!/usr/bin/env python3
"""
Investigate water barrier detection issues.

This script analyzes:
1. Which nodes become isolated
2. Why Colorado River and Lake Powell connections aren't being removed
3. Water barrier detection effectiveness
"""

import sys
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append('.')

from utgc.transitability import (
    load_road_network, load_water_bodies, identify_road_connected_precincts,
    apply_hierarchical_fallback, identify_water_crossings, build_transitable_graph
)
from gerrychain import Graph
import networkx as nx

def investigate_isolated_nodes(precincts, final_graph):
    """Investigate which nodes are isolated and why."""
    print("\n=== INVESTIGATING ISOLATED NODES ===")
    
    # Find components
    components = list(nx.connected_components(final_graph))
    component_sizes = [len(comp) for comp in components]
    print(f"Components: {component_sizes}")
    
    # Get isolated nodes
    isolated_nodes = []
    for comp in components:
        if len(comp) == 1:
            isolated_nodes.extend(list(comp))
    
    print(f"Isolated nodes: {isolated_nodes}")
    
    # Analyze each isolated node
    for node in isolated_nodes:
        print(f"\n--- Isolated Node {node} ---")
        precinct = precincts.loc[node]
        
        # Basic info
        print(f"Geometry type: {precinct.geometry.geom_type}")
        print(f"Bounds: {precinct.geometry.bounds}")
        print(f"Area: {precinct.geometry.area:.0f} sq meters")
        
        # Check original neighbors
        base_graph = Graph.from_geodataframe(precincts)
        original_neighbors = list(base_graph.neighbors(node))
        print(f"Original neighbors: {len(original_neighbors)} - {original_neighbors[:5]}{'...' if len(original_neighbors) > 5 else ''}")
        
        # Check if any neighbors are in the final graph
        final_neighbors = []
        for neighbor in original_neighbors:
            if neighbor in final_graph.nodes():
                final_neighbors.append(neighbor)
        
        print(f"Neighbors in final graph: {len(final_neighbors)}")
        
        # Check if this node has road access
        roads = load_road_network()
        precincts_proj = precincts.to_crs(roads.crs)
        precinct_buffered = precincts_proj.loc[[node]].copy()
        precinct_buffered['geometry'] = precincts_proj.loc[[node]].geometry.buffer(500.0)
        
        import maup
        intersections = maup.intersections(precinct_buffered, roads)
        has_road_access = len(intersections) > 0
        print(f"Has road access: {has_road_access}")
        
        # Check municipality/county info
        if 'MUNIID' in precincts.columns:
            print(f"Municipality: {precincts.loc[node, 'MUNIID']}")
        if 'COUNTYID' in precincts.columns:
            print(f"County: {precincts.loc[node, 'COUNTYID']}")

def investigate_water_barriers(precincts, sample_size=500):
    """Investigate why certain water barriers aren't being detected."""
    print("\n=== INVESTIGATING WATER BARRIER DETECTION ===")
    
    # Use sample for analysis
    if sample_size < len(precincts):
        sample_precincts = precincts.head(sample_size)
    else:
        sample_precincts = precincts
    
    print(f"Analyzing with {len(sample_precincts)} precincts")
    
    # Load water bodies
    water_bodies = load_water_bodies()
    print(f"Water bodies: {len(water_bodies)}")
    
    # Check water body types and sizes
    print("\nWater body analysis:")
    if 'NAME' in water_bodies.columns:
        major_water_bodies = water_bodies[water_bodies['NAME'].str.contains('Great Salt Lake|Lake Powell|Colorado River', case=False, na=False)]
        print(f"Major water bodies found: {len(major_water_bodies)}")
        for _, water in major_water_bodies.iterrows():
            print(f"  {water['NAME']}: {water.geometry.area:.0f} sq meters")
    
    # Create base graph
    base_graph = Graph.from_geodataframe(sample_precincts)
    
    # Find long-distance edges that might cross water
    long_edges = []
    for edge in base_graph.edges():
        try:
            p1_geom = sample_precincts.loc[edge[0], 'geometry']
            p2_geom = sample_precincts.loc[edge[1], 'geometry']
            distance = p1_geom.centroid.distance(p2_geom.centroid)
            if distance > 50000:  # 50km threshold
                long_edges.append((edge, distance))
        except (KeyError, IndexError):
            continue
    
    print(f"\nLong-distance edges (>50km): {len(long_edges)}")
    
    # Analyze a few long edges
    for (edge, distance) in long_edges[:10]:
        print(f"\nLong edge {edge}: {distance:.0f}m")
        
        # Check if this edge crosses major water bodies
        p1_geom = sample_precincts.loc[edge[0], 'geometry']
        p2_geom = sample_precincts.loc[edge[1], 'geometry']
        
        from shapely.geometry import LineString
        line = LineString([p1_geom.centroid, p2_geom.centroid])
        
        # Check intersections with major water bodies
        for _, water in water_bodies.iterrows():
            if line.intersects(water.geometry):
                intersection = line.intersection(water.geometry)
                if intersection.length > 0:
                    intersection_ratio = intersection.length / line.length
                    print(f"  Crosses {water.get('NAME', 'Unknown')}: {intersection_ratio:.2%} of edge length")
    
    return long_edges

def analyze_water_detection_effectiveness(precincts, sample_size=200):
    """Analyze how effective water barrier detection is."""
    print("\n=== ANALYZING WATER DETECTION EFFECTIVENESS ===")
    
    # Use sample for analysis
    if sample_size < len(precincts):
        sample_precincts = precincts.head(sample_size)
    else:
        sample_precincts = precincts
    
    # Build transitability graph
    transitability_params = {
        'enable': True,
        'remove_water_barriers': True,
        'verify_road_connectivity': True,
        'min_lake_size_sqkm': 1.0,
        'min_river_size_sqkm': 0.5,
        'road_buffer_meters': 500.0,
        'water_threshold': 0.5
    }
    
    print("Building transitability graph...")
    final_graph = build_transitable_graph(sample_precincts, transitability_params)
    
    # Analyze water barrier detection
    print("\nWater barrier analysis:")
    
    # Load water bodies
    water_bodies = load_water_bodies()
    
    # Check which water bodies are being used
    print(f"Total water bodies: {len(water_bodies)}")
    
    # Check water body sizes
    if 'SQ_KM' in water_bodies.columns:
        large_water = water_bodies[water_bodies['SQ_KM'] > 1.0]
        print(f"Large water bodies (>1 sq km): {len(large_water)}")
        
        if len(large_water) > 0:
            print("Largest water bodies:")
            largest = large_water.nlargest(10, 'SQ_KM')
            for _, water in largest.iterrows():
                print(f"  {water.get('NAME', 'Unknown')}: {water['SQ_KM']:.1f} sq km")
    
    # Check if Colorado River and Lake Powell are in the dataset
    if 'NAME' in water_bodies.columns:
        colorado_river = water_bodies[water_bodies['NAME'].str.contains('Colorado River', case=False, na=False)]
        lake_powell = water_bodies[water_bodies['NAME'].str.contains('Lake Powell', case=False, na=False)]
        
        print(f"\nColorado River features: {len(colorado_river)}")
        print(f"Lake Powell features: {len(lake_powell)}")
        
        if len(colorado_river) > 0:
            print("Colorado River details:")
            for _, water in colorado_river.iterrows():
                print(f"  {water['NAME']}: {water.geometry.area:.0f} sq meters")
        
        if len(lake_powell) > 0:
            print("Lake Powell details:")
            for _, water in lake_powell.iterrows():
                print(f"  {water['NAME']}: {water.geometry.area:.0f} sq meters")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Investigate water barrier detection issues')
    parser.add_argument('--sample-size', type=int, default=500,
                       help='Number of precincts to analyze (default: 500)')
    
    args = parser.parse_args()
    
    # Load precincts
    print("Loading precincts...")
    precincts = gpd.read_file('data/UT_precincts.geojson')
    
    try:
        # Build transitability graph
        transitability_params = {
            'enable': True,
            'remove_water_barriers': True,
            'verify_road_connectivity': True,
            'min_lake_size_sqkm': 1.0,
            'min_river_size_sqkm': 0.5,
            'road_buffer_meters': 500.0,
            'water_threshold': 0.5
        }
        
        print("Building transitability graph...")
        final_graph = build_transitable_graph(precincts, transitability_params)
        
        # Investigate isolated nodes
        investigate_isolated_nodes(precincts, final_graph)
        
        # Investigate water barriers
        long_edges = investigate_water_barriers(precincts, args.sample_size)
        
        # Analyze water detection effectiveness
        analyze_water_detection_effectiveness(precincts, args.sample_size)
        
        print("\n✅ Investigation completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
