import os
from functools import partial
from typing import Optional, Dict, Tuple, Any
import random
import numpy as np
import pandas as pd
from gerrychain import Graph, GeographicPartition, updaters, constraints
import networkx as nx
from networkx.algorithms import tree as nx_tree
from gerrychain.proposals import recom
from gerrychain.tree import bipartition_tree
from gerrychain.constraints import contiguous, UpperBound, Validator
from gerrychain.updaters.locality_split_scores import LocalitySplits
from gerrychain.metrics import polsby_popper

def create_graph(precincts):
    """
    Create GerryChain graph from precincts, pruning edges based on a CSV
    file of removed edges if provided.
    """
    print("Creating graph...")
    # Always start with the graph from the geodataframe to ensure all geometric
    # attributes like 'shared_perim' are correctly calculated.
    graph = Graph.from_geodataframe(precincts, reproject=False)
    print(f"Base graph created: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    return graph

def create_updaters(elections=None, election_columns=None, num_muniids=None, num_countyids=None):
    """Create updaters for the ensemble analysis."""
    elections = elections or []
    election_columns = election_columns or []
    print("Creating updaters...")
    
    updaters_dict = {
        "population": updaters.Tally("TOTPOP", alias="population"),
        "cut_edges": updaters.cut_edges,
        "perimeter": updaters.perimeter,
        "area": updaters.Tally("area", alias="area"),
        "muni_locality_splits": LocalitySplits(
            name="muni_locality_splits", col_id="MUNIID", pop_col="TOTPOP",
            scores_to_compute=["num_split_localities", "num_parts"],
        ),
        "split_munis": lambda p: p["muni_locality_splits"].get("num_split_localities", 0),
        "muni_multi_splits": lambda p: p["muni_locality_splits"].get("num_parts", 0) - p["split_munis"] - (num_muniids or 0),
        "county_locality_splits": LocalitySplits(
            name="county_locality_splits", col_id="COUNTYID", pop_col="TOTPOP",
            scores_to_compute=["num_split_localities", "num_parts"],
        ),
        "split_counties": lambda p: p["county_locality_splits"].get("num_split_localities", 0),
        "county_multi_splits": lambda p: p["county_locality_splits"].get("num_parts", 0) - p["split_counties"] - (num_countyids or 0),
        "assignment_hash": assignment_hash,
        "polsby_popper": polsby_popper,
    }

    if elections:
        for election in elections:
            year, office = election.split('_')
            dem_col = f"{int(year)%100:02d}{office}D"
            rep_col = f"{int(year)%100:02d}{office}R"
            if dem_col in election_columns and rep_col in election_columns:
                updaters_dict[election] = updaters.Election(
                    name=election,
                    parties_to_columns={"Democratic": dem_col, "Republican": rep_col},
                )
    return updaters_dict

def create_initial_partition(graph, precincts, updaters_dict):
    """Create initial partition from the supplied plan assignment."""
    print("Creating initial partition...")
    initial_partition = GeographicPartition(
        graph,
        assignment="CONGDIST",
        updaters=updaters_dict,
    )
    print(f"Initial partition created with {len(initial_partition)} districts")
    return initial_partition


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seed set to {seed}")


def create_constraints(
    initial_partition, 
    pop_deviation=0.001,
    split_munis_constraint=None, 
    split_counties_constraint=None,
    muni_multi_splits_constraint=None,
    county_multi_splits_constraint=None,
    include_not_equal_constraint=True,
):
    """Create constraints according to Utah redistricting requirements."""
    print("Creating constraints...")
    # population_constraint = constraints.within_percent_of_ideal_population(
    #     initial_partition, pop_deviation
    # )
    constraints_list = [contiguous,]

    if split_munis_constraint is not None:
        constraints_list.append(UpperBound(get_muni_splits, split_munis_constraint))
        print(f"  Added municipality splits constraint: max {split_munis_constraint}")
    
    if muni_multi_splits_constraint is not None:
        constraints_list.append(UpperBound(get_muni_multi_splits, muni_multi_splits_constraint))
        print(f"  Added municipality multi-splits constraint: max {muni_multi_splits_constraint}")
    
    if split_counties_constraint is not None:
        constraints_list.append(UpperBound(get_county_splits, split_counties_constraint))
        print(f"  Added county splits constraint: max {split_counties_constraint}")
    
    if county_multi_splits_constraint is not None:
        constraints_list.append(UpperBound(get_county_multi_splits, county_multi_splits_constraint))
        print(f"  Added county multi-splits constraint: max {county_multi_splits_constraint}")
    
    if include_not_equal_constraint:
        constraints_list.append(NotEqual())
        print("  Added NotEqual constraint.")

    return constraints_list

def get_muni_splits(partition):
    """Extract municipality split count (num_split_localities) from partition."""
    try:
        return int(partition["split_munis"])  # central updater
    except Exception:
        return 0

def get_muni_multi_splits(partition):
    """Extract municipality multi-splits (num_parts - num_split_localities - total_munis) from partition."""
    try:
        return int(partition["muni_multi_splits"])  # central updater
    except Exception:
        return 0

def get_county_splits(partition):
    """Extract county split count (num_split_localities) from partition."""
    try:
        return int(partition["split_counties"])  # central updater
    except Exception:
        return 0

def get_county_multi_splits(partition):
    """Extract county multi-splits (num_parts - num_split_localities - total_counties) from partition."""
    try:
        return int(partition["county_multi_splits"])  # central updater
    except Exception:
        return 0

def create_proposal(ideal_population, precincts, region_surcharge_params, pop_deviation, node_repeats, edge_penalty_params={}):
    """Create ReCom proposal with region surcharges and optional transitability constraints."""
    print("Creating ReCom proposal...")
    column_mapping = {
        'muni': 'MUNIID', 'county': 'COUNTYID', 'highered': 'HIGHERED_ID',
        'metro': 'METRO_ID', 'school_district': 'SCHDIST_ID',
        'water_region': 'WATER_ID', 'basin': 'BASIN_ID'
    }

    # transit_params is a dictionary where the keys are edge tuples and the values are penalties
    if edge_penalty_params:
        print(f"  Edge penalties: {edge_penalty_params}")
    else:
        print("  No edge penalties provided.")
    edge_penalties = {}
    for path, penalty in edge_penalty_params.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Edge file not found: {path}")
        
        edges_df = pd.read_csv(path)
        for _, row in edges_df.iterrows():
            u, v = int(row['u']), int(row['v'])
            edge_penalties[tuple(sorted((u, v)))] = penalty
    
    # Create the spanning tree with the edge penalties
    # ReCom will handle passing region surcharges
    spanning_tree_fn = partial(
        random_spanning_tree_with_edge_penalties,
        edge_penalties=edge_penalties
    )

    if region_surcharge_params:
        print(f"  Region surcharges: {region_surcharge_params}")
    else:
        print("  No region surcharges provided.")

    region_surcharge = {}
    for param_name, surcharge_value in region_surcharge_params.items():
        column_name = column_mapping.get(param_name)
        if column_name and column_name in precincts.columns:
            region_surcharge[column_name] = surcharge_value

    proposal = partial(
        recom,
        pop_col="TOTPOP",
        pop_target=ideal_population,
        epsilon=pop_deviation, 
        node_repeats=node_repeats,
        region_surcharge=region_surcharge,
        method=partial(
            bipartition_tree,
            max_attempts=1000,
            allow_pair_reselection=True,
            spanning_tree_fn=spanning_tree_fn,
        ),
    )
    return proposal

def random_spanning_tree_with_edge_penalties(
    graph: nx.Graph,
    region_surcharge: Optional[Dict] = None,
    edge_penalties: Optional[Dict] = None,
) -> nx.Graph:
    """
    Builds a spanning tree using Kruskal's method with random weights,
    allowing for region-based surcharges (standard GerryChain behavior) and specific edge penalties (e.g., to impose transitability constraints).

    This function is a flexible replacement for GerryChain's default
    `random_spanning_tree`, enabling more complex weighting schemes.

    :param graph: The input graph to build the spanning tree from.
    :param region_surcharge: A dictionary where keys are region identifiers
        (e.g., 'county_id') and values are surcharges for edges crossing
        those regional boundaries.
    :param edge_penalties: A dictionary where keys are edge tuples (u, v)
        and values are penalty weights to be added to those specific edges.
        The function checks for edges in a canonical (sorted) form.
    :returns: The minimum spanning tree based on the calculated random weights.
    """
    if region_surcharge is None:
        region_surcharge = {}
    if edge_penalties is None:
        edge_penalties = {}

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

    spanning_tree = nx_tree.minimum_spanning_tree(
        graph, algorithm="kruskal", weight="random_weight"
    )
    return spanning_tree

def assignment_hash(partition):
    """
    Updater that computes a hash of the partition's assignment.
    The assignment is converted to a frozenset of (node, district) pairs to
    ensure it is hashable and canonical (order-independent).
    """
    return hash(frozenset(partition.assignment.items()))

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
