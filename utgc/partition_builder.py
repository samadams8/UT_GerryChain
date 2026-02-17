"""
Build GeographicPartition from graph, assignment, and updaters.

Provides reusable partition creation and contiguity repair so config and
notebooks can build partitions for the same graph with different assignments
or updaters.
"""
import random
from typing import Dict, Optional, Union

import pandas as pd

from gerrychain import GeographicPartition, Partition
from gerrychain.constraints import contiguous

def repair_contiguity(
    partition: Partition,
    num_districts: int,
) -> Dict:
    """
    Repair non-contiguous districts by reassigning disconnected components
    to adjacent districts. Iterates until contiguity is achieved or
    max_iterations is reached.

    Parameters
    ----------
    partition : Partition
        Partition that may have non-contiguous districts.
    num_districts : int
        Number of districts (used to bound repair iterations).

    Returns
    -------
    dict
        Repaired assignment mapping node -> district.
    """
    import networkx as nx

    graph = partition.graph
    repaired_assignment = dict(partition.assignment)

    def _is_contiguous(assignment_dict):
        temp_partition = GeographicPartition(
            graph,
            assignment=assignment_dict,
            updaters={},
        )
        return contiguous(temp_partition)

    for iteration in range(2 * num_districts):
        if _is_contiguous(repaired_assignment):
            break

        # Handle unassigned nodes first
        unassigned_nodes = [
            node
            for node in graph.nodes
            if repaired_assignment[node] is None or pd.isna(repaired_assignment[node])
        ]

        for node in unassigned_nodes:
            neighbors = list(graph.neighbors(node))
            adjacent_districts = []
            for neighbor in neighbors:
                neighbor_dist = repaired_assignment.get(neighbor)
                if neighbor_dist is not None and not pd.isna(neighbor_dist):
                    adjacent_districts.append(neighbor_dist)

            if adjacent_districts:
                repaired_assignment[node] = max(
                    set(adjacent_districts), key=adjacent_districts.count
                )
            else:
                available_districts = [
                    d
                    for d in set(repaired_assignment.values())
                    if d is not None and not pd.isna(d)
                ]
                if available_districts:
                    repaired_assignment[node] = random.choice(available_districts)

        districts_to_check = set(repaired_assignment.values())
        districts_to_check.discard(None)

        repairs_made = False
        for district in districts_to_check:
            district_nodes = [
                node
                for node in graph.nodes
                if repaired_assignment[node] == district
            ]

            if not district_nodes:
                continue

            district_subgraph = graph.subgraph(district_nodes)
            components = list(nx.connected_components(district_subgraph))

            if len(components) <= 1:
                continue

            repairs_made = True
            components = sorted(components, key=len, reverse=True)

            for component in components[1:]:
                neighbor_districts = []
                for node in component:
                    for neighbor in graph.neighbors(node):
                        neighbor_dist = repaired_assignment.get(neighbor)
                        if (
                            neighbor_dist is not None
                            and neighbor_dist != district
                            and not pd.isna(neighbor_dist)
                        ):
                            neighbor_districts.append(neighbor_dist)

                if neighbor_districts:
                    target_district = max(
                        set(neighbor_districts), key=neighbor_districts.count
                    )
                else:
                    available_districts = [
                        d for d in districts_to_check if d != district
                    ]
                    if available_districts:
                        target_district = random.choice(available_districts)
                    else:
                        target_district = district

                for node in component:
                    repaired_assignment[node] = target_district

        if not repairs_made:
            break

    return repaired_assignment


def build_initial_partition(
    graph,
    assignment: Union[str, Dict],
    updaters: Dict,
    num_districts: int,
    repair: bool = True,
) -> Partition:
    """
    Build a GeographicPartition from a graph, assignment, and updaters,
    optionally repairing contiguity.

    Parameters
    ----------
    graph : gerrychain.Graph
        Graph built from geodata.
    assignment : str or dict
        Column name in graph nodes (e.g. "initial_plan") or assignment dict.
    updaters : dict
        Partition updaters (e.g. population Tally and others).
    num_districts : int
        Number of districts (used for contiguity repair iteration bound).
    repair : bool, optional
        If True (default), attempt to repair non-contiguous districts.

    Returns
    -------
    Partition
        GeographicPartition, repaired for contiguity when repair=True and needed.
    """
    from warnings import warn

    part = GeographicPartition(
        graph,
        assignment=assignment,
        updaters=updaters,
    )

    if repair and not contiguous(part):
        repaired_assignment = repair_contiguity(part, num_districts)
        part = GeographicPartition(
            graph,
            assignment=repaired_assignment,
            updaters=updaters,
        )
        if not contiguous(part):
            warn(
                "Contiguity repair may not have fully resolved all issues. "
                "You should check your initial plan or population geodata for compatibility issues."
            )

    return part
