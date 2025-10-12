"""
Transitability module for graph connectivity analysis.

This module provides two main methods for building a transitability-aware graph:
1. roads_only: Prunes by direct road connectivity, then uses a geographic
   fallback and county-level repair to ensure connectivity.
2. hierarchical (old method): Uses a fallback system of roads, water,
   and administrative boundaries.
"""

from __future__ import annotations
import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple
from gerrychain import Graph
import warnings
from shapely.ops import unary_union
import networkx as nx

# Import the new road pruning algorithm
from data.scripts.transitability.road_pruning import build_graph_from_road_connectivity

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# --- Helper Functions (used by both methods) ---

def find_and_set_index(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Finds a suitable unique ID column and sets it as the GeoDataFrame index.
    Searches for common ID names and falls back to the existing index if none are found.
    """
    candidate_cols = ["GEOID20", "ID", "GEOID", "geoid", "id"]
    for col in candidate_cols:
        if col in gdf.columns:
            print(f"Found unique ID column: '{col}'. Setting as index.")
            return gdf.set_index(col)
    
    print("Warning: No standard unique ID column found. Using existing index.")
    return gdf


def load_road_network(
    roads_path: str = "data/geography_processed/UtahRoads_filtered.shp"
) -> gpd.GeoDataFrame:
    """
    Load filtered road network for connectivity analysis.
    """
    print(f"Loading road network from {roads_path}...")
    roads = gpd.read_file(roads_path)
    return roads


def test_graph_connectivity(graph: Graph, name: str, raise_error: bool = False):
    """Tests and reports on the connectivity of a graph."""
    is_connected = nx.is_connected(graph)
    print(f"Connectivity test for '{name}': {'PASS' if is_connected else 'FAIL'}")
    if not is_connected:
        components = list(nx.connected_components(graph))
        msg = f"{name} is not connected. Found {len(components)} components."
        if raise_error:
            raise RuntimeError(msg)
        else:
            print(f"  - {msg}")


def add_geographic_fallback_edges(graph: Graph, precincts: gpd.GeoDataFrame, hierarchy: List[str]):
    """
    Adds edges to a graph based on administrative hierarchies to improve connectivity.
    It iterates through hierarchies (e.g., county, then municipality) and adds
    an edge between any two nodes in the same administrative unit if they are adjacent.
    """
    print(f"Adding geographic fallback edges using hierarchy: {hierarchy}")
    base_adj_graph = Graph.from_geodataframe(precincts)
    
    for level in hierarchy:
        print(f"  - Processing fallback for '{level}'...")
        edges_added = 0
        for u, v, data in base_adj_graph.edges(data=True):
            # If this edge doesn't already exist in our graph
            if not graph.has_edge(u, v):
                node_u = precincts.loc[u]
                node_v = precincts.loc[v]
                # If they share the same ID for the current hierarchy level
                if level in node_u and level in node_v and node_u[level] == node_v[level]:
                    graph.add_edge(u, v)
                    # Copy over the shared_perim attribute
                    graph.edges[(u,v)]["shared_perim"] = data.get("shared_perim")
                    edges_added += 1
        print(f"    - Added {edges_added} edges for {level} contiguity.")
    return graph


def repair_county_connectivity(graph: Graph, precincts: gpd.GeoDataFrame, county_col: str) -> Graph:
    """
    Ensures that all precincts within each county form a single connected component.
    """
    print("\nRepairing county-level connectivity...")
    base_adj_graph = Graph.from_geodataframe(precincts)
    counties = precincts[county_col].unique()
    edges_added_total = 0

    for county in counties:
        nodes_in_county = precincts[precincts[county_col] == county].index
        county_subgraph = graph.subgraph(nodes_in_county)

        if not nx.is_connected(county_subgraph):
            print(f"  - County '{county}' is disconnected. Repairing...")
            components = list(nx.connected_components(county_subgraph))
            
            # Find the largest component to connect others to
            largest_component = max(components, key=len)
            
            for component in components:
                if component == largest_component:
                    continue
                
                # Find the shortest path in the original adjacency graph
                # between this small component and the largest one.
                path_found = False
                for source_node in component:
                    for target_node in largest_component:
                        if nx.has_path(base_adj_graph, source_node, target_node):
                            path = nx.shortest_path(base_adj_graph, source_node, target_node)
                            print(f"    - Connecting component via path: {path}")
                            for i in range(len(path) - 1):
                                u, v = path[i], path[i+1]
                                if not graph.has_edge(u, v):
                                    graph.add_edge(u, v)
                                    # Copy over the shared_perim attribute
                                    graph.edges[(u,v)]["shared_perim"] = base_adj_graph.edges[(u,v)].get("shared_perim")
                                    edges_added_total += 1
                            path_found = True
                            break
                    if path_found:
                        break

    print(f"  - Added {edges_added_total} edges to ensure county connectivity.")
    return graph


# --- Main Graph Building Methods ---

def build_roads_only_graph(
    precincts_path: str,
    roads_path: str,
    reproject: bool = False
) -> Tuple[Graph, Graph]:
    """
    Builds a transitability graph using the roads-first approach.
    1. Builds a sparse graph of only road-connected edges.
    2. Adds fallback edges for county and municipality contiguity.
    3. Repairs connectivity within each county.
    """
    print("--- Building Roads-Only Transitivity Graph ---")
    precincts_gdf = gpd.read_file(precincts_path)
    precincts = find_and_set_index(precincts_gdf)
    roads = load_road_network(roads_path)

    base_graph = Graph.from_geodataframe(precincts)
    
    # 1. Build initial graph from road connectivity
    road_graph = build_graph_from_road_connectivity(precincts, roads, graph_crs=precincts.crs)
    test_graph_connectivity(road_graph, "Initial road-based graph")

    # 2. Add geographic fallback edges
    # fallback_hierarchy = ["COUNTYID", "MUNIID"]
    # fallback_graph = add_geographic_fallback_edges(road_graph, precincts, fallback_hierarchy)

    # 3. Repair county connectivity
    final_graph = repair_county_connectivity(road_graph, precincts, "COUNTYID")

    print("\n--- Final Graph Summary ---")
    print(f"Base graph has {len(base_graph.edges)} edges")
    print(f"Final graph has {len(final_graph.edges)} edges")
    print(f"Edges removed: {len(base_graph.edges) - len(final_graph.edges)}")
    test_graph_connectivity(final_graph, "Final repaired graph")

    return base_graph, final_graph


def build_transitable_graph(
    precincts_path: str,
    prune_water: bool,
    prune_roads: bool,
    reproject: bool,
    lakes_path: str,
    rivers_path: str
) -> Graph:
    """
    Original hierarchical pruning method. Kept for comparison.
    """
    print("\n--- Building Transitivity Graph (Hierarchical Method) ---")
    precincts_gdf = gpd.read_file(precincts_path)
    precincts = find_and_set_index(precincts_gdf)

    base_graph = Graph.from_geodataframe(precincts)
    final_graph = base_graph.copy()

    if prune_water:
        print("\nPruning edges by water barriers...")
        lakes = gpd.read_file(lakes_path)
        rivers = gpd.read_file(rivers_path)
        if reproject:
            if lakes.crs != precincts.crs: lakes = lakes.to_crs(precincts.crs)
            if rivers.crs != precincts.crs: rivers = rivers.to_crs(precincts.crs)
        
        water_union = unary_union(list(lakes.geometry) + list(rivers.geometry))
        
        edges_to_remove = []
        for u, v, data in final_graph.edges(data=True):
            if data["shared_perim"] and data["shared_perim"].intersects(water_union):
                edges_to_remove.append((u, v))
        final_graph.remove_edges_from(edges_to_remove)
        print(f"  - Removed {len(edges_to_remove)} edges crossing water barriers.")

    if prune_roads:
        print("\nNote: Hierarchical road pruning logic is a placeholder in this version.")
        pass

    print(f"\nBase graph has {len(base_graph.edges)} edges")
    print(f"Final graph has {len(final_graph.edges)} edges")
    
    test_graph_connectivity(final_graph, "Final transitability graph", raise_error=False)
    
    return base_graph, final_graph
