"""
Supporting utilities for ensemble runner operations.

This module contains helper functions and classes used by the EnsembleRunner
for graph operations, constraints, and metric computations.
"""
import random
import math
from typing import Optional, Dict, Tuple

import networkx as nx
from networkx.algorithms import tree as nx_tree
from gerrychain.constraints import Validator
from shapely.geometry import Point
from shapely.ops import unary_union

def random_spanning_tree_with_edge_penalties(
    graph: nx.Graph,
    edge_penalties: Optional[Dict[Tuple[int, int], float]] = None,
    region_surcharge: Optional[Dict[str, float]] = None,
) -> nx.Graph:
    """
    Builds a spanning tree using Kruskal's method with random weights,
    allowing for region-based surcharges (standard GerryChain behavior) and specific edge penalties (e.g., to impose transitability constraints).

    This function is a flexible replacement for GerryChain's default
    `random_spanning_tree`, enabling more complex weighting schemes.

    :param graph: The input graph to build the spanning tree from.
    :param region_surcharge: A dictionary where keys are column names in the graph nodes
        (e.g., 'county_id') and values are surcharges for edges crossing
        those regional boundaries.
    :param edge_penalties: A dictionary where keys are edge tuples (u, v)
        and values are penalty weights to be added to those specific edges.
        The function checks for edges in a canonical (sorted) form.
    :returns: The minimum spanning tree based on the calculated random weights.
    """
    edge_penalties = edge_penalties or {}
    region_surcharge = region_surcharge or {}

    # print("  DEBUG: random spanning tree with edge penalties")
    # print(f"  DEBUG: edge penalties: {edge_penalties}")
    # print(f"  DEBUG: region surcharge: {region_surcharge}")

    for u, v in graph.edges():
        weight = random.random()

        # Apply region surcharges (GerryChain standard behavior)
        for key, value in region_surcharge.items():
            if (
                graph.nodes[u].get(key) != graph.nodes[v].get(key)
                or graph.nodes[u].get(key) is None
                or graph.nodes[v].get(key) is None
            ):
                weight += value

        # Apply specific edge penalties from our configuration
        edge_canonical = tuple(sorted((u, v)))
        if edge_canonical in edge_penalties:
            weight += edge_penalties[edge_canonical]

        graph.edges[(u, v)]["random_weight"] = weight

    return nx_tree.minimum_spanning_tree(
        graph, algorithm="kruskal", weight="random_weight"
    )

def _assignment_hash(partition):
    hashes = []
    for v in partition.parts.values():
        hashes.append(hash(v))
    hashes = sorted(hashes)
    return tuple(hashes)
    # return hash(frozenset(partition.assignment.items()))

def _reock_score(partition):
    """
    Compute Reock score for each district in the partition.
    
    Reock score is the ratio of district area to the area of the minimum
    bounding circle that contains the district. Higher values indicate
    more compact districts.
    
    Parameters
    ----------
    partition : Partition
        The partition to compute Reock scores for
        
    Returns
    -------
    dict
        Dictionary mapping district ID to Reock score
    """
    scores = {}
    for district in partition:
        # Get geometries for all nodes in this district
        geometries = []
        for node in partition.parts[district]:
            node_data = partition.graph.nodes[node]
            geom = node_data.get('geometry')
            if geom is not None:
                geometries.append(geom)
        
        if not geometries:
            scores[district] = 0.0
            continue
        
        # Union all geometries in the district
        district_geom = unary_union(geometries)
        district_area = district_geom.area
        
        # Compute minimum bounding circle
        # Get the bounding box and compute the circle that encompasses it
        minx, miny, maxx, maxy = district_geom.bounds
        
        # Center of bounding box
        center_x = (minx + maxx) / 2.0
        center_y = (miny + maxy) / 2.0
        center = Point(center_x, center_y)
        
        # Find the maximum distance from center to any point in the geometry
        # This gives us the radius of the minimum bounding circle
        max_dist = 0.0
        for geom in geometries:
            if hasattr(geom, 'exterior'):
                coords = geom.exterior.coords
            elif hasattr(geom, 'coords'):
                coords = geom.coords
            else:
                continue
            for coord in coords:
                pt = Point(coord)
                dist = center.distance(pt)
                if dist > max_dist:
                    max_dist = dist
        
        # Area of minimum bounding circle
        bounding_circle_area = math.pi * (max_dist ** 2)
        
        # Reock score
        if bounding_circle_area > 0:
            scores[district] = district_area / bounding_circle_area
        else:
            scores[district] = 0.0
    
    return scores

class NotEqual(Validator):
    """
    A constraint that is satisfied if the proposed partition is not the same as
    the partition it is being compared to (its parent). It uses a hash of the
    assignment to perform this check efficiently.

    Requires the `assignment_hash` updater to be active.
    """
    def __init__(self):
        """
        Initializes the NotEqual constraint.
        """
        pass

    def __call__(self, partition):
        """
        Checks if the current partition's assignment hash is different from
        its parent's.

        :param partition: The proposed partition to check.
        :return: True if the partition is different from its parent, False otherwise.
        """
        if partition.parent is None:
            return True  # The initial partition is always valid

        # The 'assignment_hash' must be in the updaters for this to work.
        # This check provides a helpful error message if the updater is missing.
        if "assignment_hash" not in partition.updaters:
            raise KeyError(
                "The 'NotEqual' constraint requires the 'assignment_hash' updater. "
                "Please add it to your Partition's updaters."
            )

        return partition["assignment_hash"] != partition.parent["assignment_hash"]
