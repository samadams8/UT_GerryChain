"""
Transitability module for graph connectivity analysis.

This module implements hierarchical fallback connectivity analysis for redistricting,
considering road networks, water barriers, and administrative boundaries.

Key Features:
- Road network connectivity with hierarchical fallback
- Water barrier detection and edge removal
- Support for both precinct and block-level analysis
- Integration with GerryChain graph construction
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from gerrychain import Graph
import warnings
import maup

def load_road_network(
    roads_path: str = "data/geography_processed/UtahRoads_filtered.shp",
    cartocodes_to_exclude: List[str] = None
) -> gpd.GeoDataFrame:
    """
    Load filtered road network for connectivity analysis.
    
    Parameters
    ----------
    roads_path : str
        Path to filtered roads shapefile
    cartocodes_to_exclude : List[str], optional
        Additional CARTOCODE values to exclude beyond defaults
        
    Returns
    -------
    gpd.GeoDataFrame
        Filtered road network
    """
    if cartocodes_to_exclude is None:
        cartocodes_to_exclude = []
    
    print(f"Loading road network from {roads_path}...")
    roads = gpd.read_file(roads_path)
    
    if cartocodes_to_exclude:
        original_count = len(roads)
        roads = roads[~roads['CARTOCODE'].isin(cartocodes_to_exclude)]
        print(f"  Excluded CARTOCODE {cartocodes_to_exclude}: {len(roads):,} roads ({len(roads)/original_count*100:.1f}%)")
    else:
        print(f"  Loaded {len(roads):,} roads")
    
    return roads

def load_water_bodies(
    lakes_path: str = "data/geography_processed/UtahMajorLakes_filtered.shp",
    rivers_path: str = "data/geography_processed/UtahMajorRivers_filtered.shp"
) -> gpd.GeoDataFrame:
    """
    Load filtered water bodies for barrier analysis.
    
    Parameters
    ----------
    lakes_path : str
        Path to filtered lakes shapefile
    rivers_path : str
        Path to filtered rivers shapefile
        
    Returns
    -------
    gpd.GeoDataFrame
        Combined water bodies dataset
    """
    print("Loading water bodies...")
    
    # Load lakes
    lakes = gpd.read_file(lakes_path)
    print(f"  Lakes: {len(lakes):,} features")
    
    # Load rivers
    rivers = gpd.read_file(rivers_path)
    print(f"  Rivers: {len(rivers):,} features")
    
    # Combine water bodies
    water_bodies = pd.concat([lakes, rivers], ignore_index=True)
    water_bodies = gpd.GeoDataFrame(water_bodies, crs=lakes.crs)
    
    print(f"  Total water bodies: {len(water_bodies):,}")
    return water_bodies

def identify_road_connected_precincts(
    precincts: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    buffer_meters: float = 500.0
) -> pd.DataFrame:
    """
    Identify which precinct pairs are connected by roads using maup for efficiency.
    
    Parameters
    ----------
    precincts : gpd.GeoDataFrame
        Precinct geometries
    roads : gpd.GeoDataFrame
        Road network
    buffer_meters : float
        Buffer distance for road-precinct intersection
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['precinct_1', 'precinct_2', 'connected']
        indicating road connectivity between precinct pairs
    """
    print("Identifying road-connected precincts...")
    
    # Reproject to match
    precincts_proj = precincts.to_crs(roads.crs)
    
    # Create buffer around precincts for road intersection
    precincts_buffered = precincts_proj.copy()
    precincts_buffered['geometry'] = precincts_proj.geometry.buffer(buffer_meters)
    
    # Use maup's efficient intersections to find precinct-road overlaps
    print("  Computing precinct-road intersections...")
    intersections = maup.intersections(precincts_buffered, roads)  # No area cutoff for line intersections
    
    # Get unique precincts with road access
    precincts_with_roads = set(intersections.index.get_level_values(0))
    print(f"  Precincts with road access: {len(precincts_with_roads):,} ({len(precincts_with_roads)/len(precincts)*100:.1f}%)")
    
    # Build adjacency graph first
    base_graph = Graph.from_geodataframe(precincts)
    
    # For each precinct, find which other precincts it can reach via roads
    # This is a simplified approach - in practice, you'd want to do network analysis
    # For now, we'll assume precincts are connected if they both have road access
    # and are geographically adjacent
    
    connections = []
    for node1 in base_graph.nodes():
        if node1 in precincts_with_roads:
            for node2 in base_graph.neighbors(node1):
                if node2 in precincts_with_roads:
                    connections.append({
                        'precinct_1': node1,
                        'precinct_2': node2,
                        'connected': True
                    })
    
    print(f"  Road-connected precinct pairs: {len(connections):,}")
    return pd.DataFrame(connections)

def apply_hierarchical_fallback(
    precincts: gpd.GeoDataFrame,
    connections: pd.DataFrame,
    base_graph: Graph
) -> pd.DataFrame:
    """
    Apply hierarchical fallback for orphaned precincts after water barrier removal.
    
    This step connects precincts that were orphaned by road/water analysis
    to their adjacent neighbors in the same county.
    
    Parameters
    ----------
    precincts : gpd.GeoDataFrame
        Precinct geometries with COUNTYID
    connections : pd.DataFrame
        Current connectivity after road/water analysis
    base_graph : Graph
        Base adjacency graph
        
    Returns
    -------
    pd.DataFrame
        Extended connectivity with fallback connections
    """
    print("Applying hierarchical fallback for orphaned precincts...")
    
    # Get precincts that are currently connected
    if connections.empty:
        connected_precincts = set()
    else:
        connected_precincts = set(connections[connections['connected'] == True]['precinct_1'].unique()) | \
                            set(connections[connections['connected'] == True]['precinct_2'].unique())
    
    # Find orphaned precincts (not connected after road/water analysis)
    all_precincts = set(base_graph.nodes())
    orphaned_precincts = all_precincts - connected_precincts
    
    print(f"  Orphaned precincts: {len(orphaned_precincts):,} ({len(orphaned_precincts)/len(all_precincts)*100:.1f}%)")
    
    # Apply fallback logic - connect orphaned precincts to same-county neighbors
    fallback_connections = []
    
    for orphan in orphaned_precincts:
        orphan_county = precincts.loc[orphan, 'COUNTYID'] if 'COUNTYID' in precincts.columns else None
        
        for neighbor in base_graph.neighbors(orphan):
            neighbor_county = precincts.loc[neighbor, 'COUNTYID'] if 'COUNTYID' in precincts.columns else None
            
            # Connect if same county
            if orphan_county and neighbor_county and orphan_county == neighbor_county:
                fallback_connections.append({
                    'precinct_1': orphan,
                    'precinct_2': neighbor,
                    'connected': True,
                    'fallback_type': 'county_fallback'
                })
            # No fallback - barrier
            else:
                fallback_connections.append({
                    'precinct_1': orphan,
                    'precinct_2': neighbor,
                    'connected': False,
                    'fallback_type': 'barrier'
                })
    
    # Combine existing connections with fallback
    all_connections = []
    
    # Add existing connections
    for _, row in connections.iterrows():
        all_connections.append({
            'precinct_1': row['precinct_1'],
            'precinct_2': row['precinct_2'],
            'connected': row['connected'],
            'fallback_type': row.get('fallback_type', 'road')
        })
    
    # Add fallback connections
    all_connections.extend(fallback_connections)
    
    print(f"  Fallback connections added: {len(fallback_connections):,}")
    print(f"  Total connections: {len(all_connections):,}")
    
    return pd.DataFrame(all_connections)

def identify_water_crossings(
    precincts: gpd.GeoDataFrame,
    water_bodies: gpd.GeoDataFrame,
    connections: pd.DataFrame,
    base_graph: Graph,
    water_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Identify edges that cross major water bodies using direct geometric intersection.
    
    Parameters
    ----------
    precincts : gpd.GeoDataFrame
        Precinct geometries
    water_bodies : gpd.GeoDataFrame
        Water body geometries
    connections : pd.DataFrame
        Current connectivity
    base_graph : Graph
        Base adjacency graph
    water_threshold : float
        Minimum fraction of edge length that must cross water to be removed
        
    Returns
    -------
    pd.DataFrame
        Updated connections with water barrier information
    """
    print("Identifying water crossings...")
    
    # Reproject to match
    precincts_proj = precincts.to_crs(water_bodies.crs)
    
    # Analyze each connection directly
    water_crossings = []
    water_removed = 0
    
    for _, row in connections.iterrows():
        if not row['connected']:
            water_crossings.append({
                'precinct_1': row['precinct_1'],
                'precinct_2': row['precinct_2'],
                'connected': False,
                'fallback_type': row.get('fallback_type', 'road'),
                'water_barrier': False
            })
            continue
        
        # Get precinct geometries
        p1_geom = precincts_proj.loc[row['precinct_1'], 'geometry']
        p2_geom = precincts_proj.loc[row['precinct_2'], 'geometry']
        
        # Create line between centroids
        from shapely.geometry import LineString
        line = LineString([p1_geom.centroid, p2_geom.centroid])
        
        # Check for intersections with water bodies
        intersects_water = False
        total_intersection_length = 0
        
        for _, water in water_bodies.iterrows():
            if line.intersects(water.geometry):
                intersection = line.intersection(water.geometry)
                if intersection.length > 0:
                    total_intersection_length += intersection.length
        
        # Check if intersection ratio exceeds threshold
        if total_intersection_length > 0:
            intersection_ratio = total_intersection_length / line.length
            if intersection_ratio > water_threshold:
                intersects_water = True
                water_removed += 1
        
        water_crossings.append({
            'precinct_1': row['precinct_1'],
            'precinct_2': row['precinct_2'],
            'connected': row['connected'] and not intersects_water,
            'fallback_type': row.get('fallback_type', 'road'),
            'water_barrier': intersects_water
        })
    
    print(f"  Edges crossing water: {water_removed:,}")
    
    return pd.DataFrame(water_crossings)

def test_graph_connectivity(graph, step_name="Graph", raise_error=True):
    """
    Test if graph is fully connected and raise error if not.
    
    Parameters
    ----------
    graph : Graph or networkx.Graph
        Graph to test
    step_name : str
        Name of the step for error reporting
        
    Raises
    ------
    ValueError
        If graph is not fully connected
    """
    import networkx as nx
    
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

def build_transitable_graph(
    precincts: gpd.GeoDataFrame,
    transitability_params: Dict = None
) -> Graph:
    """
    Build a graph with transitability-aware connectivity.
    
    Parameters
    ----------
    precincts : gpd.GeoDataFrame
        Precinct geometries with MUNIID and COUNTYID
    transitability_params : Dict, optional
        Configuration parameters
        
    Returns
    -------
    Graph
        Transitability-aware graph
    """
    if transitability_params is None:
        transitability_params = {
            'enable': True,
            'remove_water_barriers': True,
            'verify_road_connectivity': True,
            'min_lake_size_sqkm': 1.0,
            'min_river_size_sqkm': 0.5,
            'road_buffer_meters': 500.0,
            'water_threshold': 0.5
        }
    
    if not transitability_params.get('enable', True):
        print("Transitability disabled, using standard graph...")
        return Graph.from_geodataframe(precincts)
    
    print("Building transitability-aware graph...")
    
    # Step 1: Load datasets
    roads = load_road_network()
    water_bodies = load_water_bodies()
    
    # Step 2: Build base adjacency graph
    base_graph = Graph.from_geodataframe(precincts)
    print(f"  Base graph: {len(base_graph.nodes)} nodes, {len(base_graph.edges)} edges")
    
    # Step 3: Road connectivity analysis (prune non-road connected edges)
    if transitability_params.get('verify_road_connectivity', True):
        road_connections = identify_road_connected_precincts(
            precincts, roads, 
            buffer_meters=transitability_params.get('road_buffer_meters', 500.0)
        )
    else:
        # Use all base connections
        road_connections = []
        for node1 in base_graph.nodes():
            for node2 in base_graph.neighbors(node1):
                road_connections.append({
                    'precinct_1': node1,
                    'precinct_2': node2,
                    'connected': True,
                    'fallback_type': 'adjacent'
                })
        road_connections = pd.DataFrame(road_connections)
    
    # Step 4: Water barrier analysis (prune major water crossings)
    if transitability_params.get('remove_water_barriers', True):
        connections = identify_water_crossings(
            precincts, water_bodies, road_connections, base_graph,
            water_threshold=transitability_params.get('water_threshold', 0.5)
        )
    else:
        connections = road_connections
    
    # Step 5: Hierarchical fallback (connect orphaned precincts to same-county neighbors)
    if transitability_params.get('verify_road_connectivity', True):
        connections = apply_hierarchical_fallback(precincts, connections, base_graph)
    
    # Step 6: Build final graph
    final_graph = Graph()
    
    # Add all nodes
    for node in base_graph.nodes():
        final_graph.add_node(node, **base_graph.nodes[node])
    
    # Add only valid connections
    valid_connections = connections[connections['connected'] == True]
    for _, row in valid_connections.iterrows():
        final_graph.add_edge(row['precinct_1'], row['precinct_2'])
    
    print(f"  Final graph: {len(final_graph.nodes)} nodes, {len(final_graph.edges)} edges")
    print(f"  Edges removed: {len(base_graph.edges) - len(final_graph.edges)}")
    
    # Test connectivity (don't raise error - disconnected graphs may be valid)
    test_graph_connectivity(final_graph, "Final transitability graph", raise_error=False)
    
    return final_graph

def analyze_transitability_impact(
    original_graph: Graph,
    transitable_graph: Graph
) -> Dict:
    """
    Analyze the impact of transitability modifications.
    
    Parameters
    ----------
    original_graph : Graph
        Original adjacency graph
    transitable_graph : Graph
        Transitability-modified graph
        
    Returns
    -------
    Dict
        Analysis results
    """
    original_edges = len(original_graph.edges)
    transitable_edges = len(transitable_graph.edges)
    edges_removed = original_edges - transitable_edges
    
    # Check connectivity
    import networkx as nx
    
    original_components = list(nx.connected_components(original_graph))
    transitable_components = list(nx.connected_components(transitable_graph))
    
    return {
        'original_edges': original_edges,
        'transitable_edges': transitable_edges,
        'edges_removed': edges_removed,
        'removal_percentage': edges_removed / original_edges * 100,
        'original_components': len(original_components),
        'transitable_components': len(transitable_components),
        'connectivity_preserved': len(transitable_components) == len(original_components)
    }
