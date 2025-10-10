#!/usr/bin/env python3
"""Wrapper CLI: build transitability-aware graph, repair connectivity, and export artifacts.

Outputs to data/transitability (by default):
- transitability.graphml
- transitability.json
- metadata.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Tuple
import sys

import geopandas as gpd
import pandas as pd
import yaml
import networkx as nx

# Ensure project root is on sys.path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utgc.transitability import build_transitable_graph  # type: ignore

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the script."""
    p = argparse.ArgumentParser(
        description="Build and repair transitability artifacts (wrapper)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--precincts", default="data/UT_precincts.geojson", help="Path to precincts GeoJSON file.")
    p.add_argument("--out-dir", default="data/transitability", help="Directory to save output artifacts.")

    # Create a dedicated group for transitability options for clarity in the --help message.
    trans_group = p.add_argument_group("Transitability Pruning Options")
    
    # Road-based pruning arguments
    trans_group.add_argument("--prune-roads", action="store_true", help="Enable pruning of edges not connected by the road network.")
    trans_group.add_argument("--road-buffer-m", type=int, default=500, help="Buffer around roads in meters for road connectivity checks.")
    trans_group.add_argument("--roads", default="data/geography_processed/UtahRoads_filtered.shp", help="Path to roads shapefile for road-based pruning.")

    # Water-based pruning arguments
    trans_group.add_argument("--prune-water-centroid", action="store_true", help="Enable pruning of edges where the centroid-connecting line crosses a major water body.")
    trans_group.add_argument("--centroid-threshold", type=float, default=0.1, help="Proportion of centroid-connecting line allowed to cross water (0 to 1).")
    
    trans_group.add_argument("--prune-water-border", action="store_true", help="Enable pruning of edges where the shared border is a major water body.")
    trans_group.add_argument("--boundary-threshold", type=float, default=0.75, help="Proportion of shared border allowed to be water for border method (0 to 1).")
    trans_group.add_argument("--boundary-water-buffer-m", type=int, default=150, help="Buffer around water bodies in meters for border method.")
    trans_group.add_argument("--boundary-road-buffer-m", type=int, default=150, help="Buffer around roads in meters for border method.")
    
    trans_group.add_argument("--lakes", default="data/geography_processed/UtahMajorLakes_filtered.shp", help="Path to lakes shapefile for water-based pruning.")
    trans_group.add_argument("--rivers", default="data/geography_processed/UtahMajorRivers_filtered.shp", help="Path to rivers shapefile for water-based pruning.")
    
    return p.parse_args()

def export_artifacts(graph: nx.Graph, out_dir: Path, precincts: str, params: Dict, args: argparse.Namespace) -> Tuple[Path, Path]:
    out_dir.mkdir(exist_ok=True)
    gml_path = out_dir / "transitability.graphml"
    json_path = out_dir / "transitability.json"

    # Create a copy of the graph for exporting to avoid modifying the original.
    graph_to_export = graph.copy()

    # The GraphML writer cannot serialize complex objects like Shapely geometries.
    # We must remove them from node attributes before exporting.
    for node, data in graph_to_export.nodes(data=True):
        if 'geometry' in data:
            del data['geometry']

    # Export the cleaned graph
    nx.write_graphml(graph_to_export, gml_path)
    
    edges = [{"source": u, "target": v} for u, v in graph.edges()]
    with open(json_path, "w") as f:
        import json
        json.dump(edges, f)

    # Export metadata
    meta = {
        "timestamp": datetime.now().isoformat(),
        "precincts": precincts,
        "parameters": params,
        "cli_args": vars(args),
    }
    with open(out_dir / "metadata.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False)
    
    return gml_path, json_path

def repair_connectivity(graph_to_repair: nx.Graph, base_adj_graph: nx.Graph, node_subset: List[int]) -> Set[Tuple[int, int]]:
    """
    Checks connectivity of a node subset and adds edges from the base graph to connect it.
    
    Args:
        graph_to_repair: The graph that may have disconnected components.
        base_adj_graph: The fully connected adjacency graph to source repair edges from.
        node_subset: The list of nodes to check for connectivity.
    
    Returns:
        A set of edges that were added to repair the graph.
    """
    if not node_subset or len(node_subset) <= 1:
        return set()

    subgraph = graph_to_repair.subgraph(node_subset)
    if nx.is_connected(subgraph):
        return set()

    added_edges = set()
    components = list(nx.connected_components(subgraph))
    
    print(f"  - Found {len(components)} disconnected components. Attempting to repair.")
    for i, component in enumerate(components):
        print(f"    - Component {i+1} ({len(component)} nodes): {sorted(list(component))}")

    main_component = components[0]
    
    for i in range(1, len(components)):
        component_to_connect = components[i]
        connection_found = False
        
        for u in main_component:
            for v in base_adj_graph.neighbors(u):
                if v in component_to_connect:
                    graph_to_repair.add_edge(u, v)
                    added_edges.add(tuple(sorted((u, v))))
                    print(f"    - Connecting components by adding edge: ({u}, {v})")
                    connection_found = True
                    break
            if connection_found:
                break
        
        if not connection_found:
             print(f"    - WARNING: Could not find a direct adjacency to connect component {i}. The graph may remain disconnected.")
        else:
            newly_connected_graph = graph_to_repair.subgraph(node_subset)
            for comp in nx.connected_components(newly_connected_graph):
                if main_component.issubset(comp):
                    main_component = comp
                    break
                    
    return added_edges


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    # Merge CLI arguments to feed to API
    if args.prune_water_centroid and args.prune_water_border:
        water_method = 'both'
    elif args.prune_water_border:
        water_method = 'boundary_with_road_exception'
    elif args.prune_water_centroid:
        water_method = 'centroid'

    # Map the user-friendly CLI arguments to the specific parameter names
    # expected by the `build_transitable_graph` function in the API.
    params = {
        "verify_road_connectivity": args.prune_roads,
        "road_buffer_meters": args.road_buffer_m,
        "roads_path": args.roads,
        "remove_water_barriers": args.prune_water_centroid or args.prune_water_border,
        "water_method": water_method,
        "water_threshold_centroid": args.centroid_threshold,
        "water_threshold": args.boundary_threshold,
        "water_buffer_m": args.boundary_water_buffer_m,
        "road_boundary_buffer_m": args.boundary_road_buffer_m,
        "lakes_path": args.lakes,
        "rivers_path": args.rivers,
    }

    # Build the base graph from simple adjacency first. This will be our
    # reference for all possible connections.
    print("Building base adjacency graph...")
    precincts = gpd.read_file(args.precincts)
    # Reset index to ensure it is a clean 0-n integer range.
    precincts = precincts.reset_index(drop=True)

    # Perform the spatial join. The resulting index is from the left gdf,
    # and a new 'index_right' column is added from the right gdf.
    adjacencies = gpd.sjoin(precincts, precincts, predicate='intersects', how='left')

    # Create a clean DataFrame for the edgelist.
    edge_df = pd.DataFrame({
        "source": adjacencies.index,
        "target": adjacencies["index_right"]
    })

    # Create the graph from this reliable edgelist.
    base_graph = nx.from_pandas_edgelist(edge_df)
    base_graph.remove_edges_from(nx.selfloop_edges(base_graph))

    # Determine if any transitability modifications should be made.
    is_transitability_enabled = args.prune_roads or args.prune_water_centroid or args.prune_water_border
    
    print("\nBuilding transitability-aware graph...")
    if is_transitability_enabled:
        # Build the graph that may have edges removed due to transitability rules.
        precincts_gdf = gpd.read_file(args.precincts)
        graph = build_transitable_graph(precincts_gdf, params)
    else:
        print("No transitability pruning options selected. Using base adjacency graph.")
        # If not enabled, the graph to check is just the base graph.
        graph = base_graph.copy()


    # --- GRAPH REPAIR AND VERIFICATION ---
    print("\nVerifying and repairing graph connectivity...")
    
    # 1. Map nodes to counties for county-level checks
    county_col = 'COUNTY' # Adjust if your GeoJSON uses a different column name
    if county_col not in precincts.columns:
        raise ValueError(f"Could not find county column '{county_col}' in precincts file.")

    nodes_by_county = precincts.groupby(county_col).groups
    total_repaired_edges = set()

    # 2. Repair connectivity for each county individually
    for county, nodes in nodes_by_county.items():
        node_list = list(nodes)
        if len(node_list) <= 1:
            continue
        
        print(f"Checking county: {county} ({len(node_list)} nodes)")
        repaired_edges = repair_connectivity(graph, base_graph, node_list)
        total_repaired_edges.update(repaired_edges)

    # 3. Repair connectivity for the entire state graph
    print("\nChecking statewide connectivity...")
    all_nodes = list(graph.nodes)
    repaired_edges = repair_connectivity(graph, base_graph, all_nodes)
    total_repaired_edges.update(repaired_edges)

    if total_repaired_edges:
        print(f"\nRepair complete. Added a total of {len(total_repaired_edges)} edge(s) to ensure connectivity.")
    else:
        print("\nGraph connectivity checks passed. No repairs needed.")

    # --- FINAL POST-REPAIR TESTS ---
    print("\nRunning final post-repair connectivity tests...")
    fully_connected = nx.is_connected(graph)
    print(f"Connected (entire graph): {fully_connected}")
    if not fully_connected:
        print("WARNING: Entire graph is still not connected after repair.")
        components = list(nx.connected_components(graph))
        print(f"  - Found {len(components)} statewide components.")
        for i, component in enumerate(components):
            node_list = sorted(list(component))
            sample = f"{node_list[:10]}..." if len(node_list) > 10 else node_list
            print(f"    - Component {i+1} ({len(node_list)} nodes): {sample}")

    all_counties_connected = True
    for county, nodes in nodes_by_county.items():
        node_list = list(nodes)
        if len(node_list) <= 1:
            continue
        sub = graph.subgraph(node_list)
        if not nx.is_connected(sub):
            all_counties_connected = False
            print(f"WARNING: County '{county}' not connected after repair.")
            components = list(nx.connected_components(sub))
            for i, component in enumerate(components):
                 print(f"  - Component {i+1} ({len(component)} nodes): {sorted(list(component))}")
    print(f"Connected (each county): {all_counties_connected}")


    # --- EXPORT ARTIFACTS ---
    print("\nExporting artifacts...")
    gml_path, json_path = export_artifacts(graph, out_dir, args.precincts, params, args)

    print("Artifacts:")
    print(f"  - graphml: {gml_path}")
    print(f"  - json: {json_path}")
    print(f"  - metadata: {out_dir / 'metadata.yaml'}")

if __name__ == "__main__":
    main()

