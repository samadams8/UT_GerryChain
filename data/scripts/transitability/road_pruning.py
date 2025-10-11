"""
road_pruning.py

Provides a function to build a gerrychain graph from direct road connectivity.
This uses a robust "additive" approach: starting with no edges, it identifies
all shared boundaries and adds edges only where a road provides transitability
across that boundary.
"""

from typing import Any
import geopandas as gpd
from gerrychain import Graph
from shapely.strtree import STRtree
from shapely.geometry.base import BaseGeometry
import warnings
import maup  # Use the top-level maup package
from tqdm import tqdm # For progress bars

# Suppress pandas FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

def build_graph_from_road_connectivity(
    precincts: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    graph_crs: Any,
    buffer_dist: float = 2.0
) -> Graph:
    """
    Builds a precinct graph with edges only where roads cross or are coincident
    with shared boundaries.

    This robust method uses maup.adjacencies to reliably identify touching
    precincts, then verifies that a road provides a path of transit for each
    shared boundary. It uses a small buffer to account for minor geometric
    inaccuracies, making it resilient to floating-point issues and slight
    misalignments.

    Args:
        precincts (gpd.GeoDataFrame): GeoDataFrame of precincts, indexed by precinct ID.
        roads (gpd.GeoDataFrame): GeoDataFrame of road linestrings.
        graph_crs (Any): The target CRS for the analysis (typically the precincts' CRS).
        buffer_dist (float): The buffer distance in the units of the graph_crs
                             to apply to boundaries when checking for nearby roads.
                             Defaults to 2.0 (e.g., 2 meters for a projected CRS).

    Returns:
        Graph: A gerrychain Graph with edges only where roads cross or run along boundaries.
    """
    print("Building initial graph from road connectivity (robust maup adjacencies method)...")

    if roads.crs != graph_crs:
        print(f"  - Reprojecting roads CRS from {roads.crs} to {graph_crs}...")
        roads = roads.to_crs(graph_crs)

    print("  - Healing precinct geometries with maup.smart_repair()...")
    precincts = maup.smart_repair(precincts)

    # 1. Use maup.adjacencies to robustly find all touching precincts.
    # This is more reliable than gerrychain's default for messy shapefiles.
    print("  - Identifying adjacencies with maup.adjacencies()...")
    adjacencies = maup.adjacencies(precincts, warn_for_islands=False)
    
    # 2. Compute the shared boundaries (intersections) for each adjacent pair.
    shared_boundaries_data = []
    edge_to_boundary_map = {} # Store for later lookup
    print(f"  - Computing shared boundaries for {len(adjacencies)} adjacencies...")
    
    for u, v in tqdm(adjacencies.index, desc="  - Computing boundaries"):
        geom_u = precincts.geometry.loc[u]
        geom_v = precincts.geometry.loc[v]
        shared_perim = geom_u.intersection(geom_v)

        if isinstance(shared_perim, BaseGeometry) and not shared_perim.is_empty:
            edge = tuple(sorted((u, v)))
            shared_boundaries_data.append({'geometry': shared_perim, 'edge': edge})
            edge_to_boundary_map[edge] = shared_perim

    if not shared_boundaries_data:
        print("  - Warning: No shared boundaries found after maup processing. Returning empty graph.")
        graph = Graph()
        graph.add_nodes_from(precincts.index)
        graph.add_data(precincts)
        return graph

    boundaries = gpd.GeoDataFrame(
        shared_boundaries_data,
        geometry='geometry',
        crs=precincts.crs
    ).drop_duplicates(subset='edge').reset_index(drop=True)

    # 3. Build a spatial index from the road geometries for efficient querying
    print(f"  - Building spatial index for {len(roads)} roads...")
    road_geometries = roads.geometry.tolist()
    road_index = STRtree(road_geometries)

    # 4. For each boundary, check if a road intersects it or its buffer
    print(f"  - Verifying road transitability for {len(boundaries)} boundaries...")
    edges_to_add = set()

    for boundary in tqdm(boundaries.itertuples(), total=len(boundaries), desc="  - Checking boundaries"):
        query_geom = boundary.geometry.buffer(buffer_dist)
        possible_road_indices = road_index.query(query_geom, predicate='intersects')

        if len(possible_road_indices) > 0:
            edges_to_add.add(boundary.edge)

    print(f"  - Found {len(edges_to_add)} unique road-transitable edges.")

    # 5. Construct the final graph from the identified edges
    print(f"  - Building final graph with {len(edges_to_add)} road-connected edges...")
    graph = Graph()
    graph.add_nodes_from(precincts.index)
    graph.add_data(precincts)
    graph.add_edges_from(list(edges_to_add))

    # 6. Re-add the shared_perim attribute for potential later use
    for u, v in graph.edges:
        edge = tuple(sorted((u, v)))
        if edge in edge_to_boundary_map:
            graph.edges[edge]["shared_perim"] = edge_to_boundary_map[edge]

    print("Road connectivity graph built successfully.")
    return graph
