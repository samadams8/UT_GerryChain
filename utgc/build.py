from functools import partial
import random
import numpy as np
from gerrychain import Graph, GeographicPartition, updaters, constraints
import os
import networkx as nx
from gerrychain.proposals import recom
from gerrychain.tree import bipartition_tree
from gerrychain.constraints import contiguous, UpperBound, Validator
from gerrychain.updaters.locality_split_scores import LocalitySplits
from gerrychain.metrics import polsby_popper

def create_graph(precincts, transitability_params=None):
    """Create GerryChain graph from precincts with optional transitability analysis or precomputed graph."""
    print("Creating graph...")
    params = transitability_params or {}
    precomputed_path = params.get("precomputed_path")

    # Always start with the graph from geodataframe, which has all attributes.
    graph = Graph.from_geodataframe(precincts, reproject=False)
    print(f"Base graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # If a precomputed transitability graph is provided, prune the base graph.
    if precomputed_path:
        if not os.path.exists(precomputed_path):
            raise FileNotFoundError(f"Precomputed graph not found: {precomputed_path}")
        
        print(f"Loading transitability edges from {precomputed_path}...")
        ext = os.path.splitext(precomputed_path)[1].lower()
        transitability_edges = []

        if ext == ".json":
            import json
            with open(precomputed_path, "r") as f:
                edge_list = json.load(f)
            for e in edge_list:
                u, v = e.get("source"), e.get("target")
                if u is not None and v is not None:
                    transitability_edges.append((int(u), int(v)))
        else:
            raise ValueError(f"Unsupported precomputed graph format: {ext}")
        
        # Create a set of allowed edges for fast lookups
        allowed_edges = {tuple(sorted(e)) for e in transitability_edges}
        
        # Prune edges not in the transitability set
        edges_to_remove = []
        for u, v in graph.edges:
            if tuple(sorted((u, v))) not in allowed_edges:
                edges_to_remove.append((u, v))
        
        graph.remove_edges_from(edges_to_remove)
        print(f"Pruned to transitability graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    else:
        print(f"Using default adjacency graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges")

    return graph

def create_updaters(elections=[], election_columns=[], num_muniids=None, num_countyids=None):
    """Create updaters for the ensemble analysis."""
    print("Creating updaters...")

    updaters_dict = {
        "population": updaters.Tally("TOTPOP", alias="population"),
        "cut_edges": updaters.cut_edges,
        "num_cut_edges": lambda p: len(p["cut_edges"]),
        "perimeter": updaters.perimeter,
        "area": updaters.Tally("area", alias="area"),
        "muni_locality_splits": LocalitySplits(
            name="muni_locality_splits",
            col_id="MUNIID",
            pop_col="TOTPOP",
            scores_to_compute=["num_split_localities", "num_parts"],
        ),
        "split_munis": lambda p: p["muni_locality_splits"].get(
            "num_split_localities", 0),
        "muni_multi_splits": lambda p: p["muni_locality_splits"].get(
            "num_parts", 0) - p["split_munis"] - num_muniids,
        "county_locality_splits": LocalitySplits(
            name="county_locality_splits",
            col_id="COUNTYID",
            pop_col="TOTPOP",
            scores_to_compute=["num_split_localities", "num_parts"],
        ),
        "split_counties": lambda p: p["county_locality_splits"].get(
            "num_split_localities", 0),
        "county_multi_splits": lambda p: p["county_locality_splits"].get(
            "num_parts", 0) - p["split_counties"] - num_countyids,
        "assignment_hash": assignment_hash,
        "polsby_popper": lambda p: polsby_popper(p),
    }

    if len(elections) > 0:
        for election in elections:
            year, office = election.split('_')
            year_int = int(year)
            dem_col = f"{year_int%100:02d}{office}D"
            rep_col = f"{year_int%100:02d}{office}R"
            if dem_col in election_columns and rep_col in election_columns:
                election_updater = updaters.Election(
                    name=election,
                    parties_to_columns={"Democratic": dem_col, "Republican": rep_col},
                )
                updaters_dict[election] = election_updater

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

    population_constraint = constraints.within_percent_of_ideal_population(
        initial_partition, pop_deviation
    )
    contiguity_constraint = contiguous

    constraints_list = [population_constraint, contiguity_constraint]
    
    if include_not_equal_constraint:
        constraints_list.append(NotEqual())
        print("  Added NotEqual constraint to prevent redundant steps.")

    # Add municipality split constraint if specified
    if split_munis_constraint is not None:
        muni_constraint = UpperBound(get_muni_splits, split_munis_constraint)
        constraints_list.append(muni_constraint)
        print(f"  Added municipality splits constraint: max {split_munis_constraint}")
    
    # Add municipality multi-splits constraint if specified
    if muni_multi_splits_constraint is not None:
        muni_multi_bound = UpperBound(get_muni_multi_splits, muni_multi_splits_constraint)
        constraints_list.append(muni_multi_bound)
        print(f"  Added municipality multi-splits constraint: max {muni_multi_splits_constraint}")
    
    # Add county split constraint if specified
    if split_counties_constraint is not None:
        county_constraint = UpperBound(get_county_splits, split_counties_constraint)
        constraints_list.append(county_constraint)
        print(f"  Added county splits constraint: max {split_counties_constraint}")
    
    # Add county multi-splits constraint if specified
    if county_multi_splits_constraint is not None:
        county_multi_bound = UpperBound(get_county_multi_splits, county_multi_splits_constraint)
        constraints_list.append(county_multi_bound)
        print(f"  Added county multi-splits constraint: max {county_multi_splits_constraint}")

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

def create_proposal(ideal_population, precincts, region_surcharge_params):
    """Create ReCom proposal with region surcharges.
    
    Args:
        ideal_population: Target population per district
        precincts: GeoDataFrame with precinct data
        region_surcharge_params: Dict with keys matching notebook interface:
            {'muni': 3, 'county': 2, 'highered': 1, 'metro': 1, 
             'school_district': 0.1, 'water_region': 0.1, 'basin': 0.1}
    """
    print("Creating ReCom proposal...")

    # Map notebook parameter names to column names
    column_mapping = {
        'muni': 'MUNIID',
        'county': 'COUNTYID', 
        'highered': 'HIGHERED_ID',
        'metro': 'METRO_ID',
        'school_district': 'SCHDIST_ID',
        'water_region': 'WATER_ID',
        'basin': 'BASIN_ID'
    }

    region_surcharge = {}
    for param_name, surcharge_value in region_surcharge_params.items():
        if param_name in column_mapping:
            column_name = column_mapping[param_name]
            if column_name in precincts.columns and surcharge_value > 0:
                region_surcharge[column_name] = surcharge_value

    proposal = partial(
        recom,
        pop_col="TOTPOP",
        pop_target=ideal_population,
        epsilon=0.001,
        node_repeats=2,
        region_surcharge=region_surcharge,
        method=partial(
            bipartition_tree,
            max_attempts=1000,
            allow_pair_reselection=True,
        ),
    )

    print(f"Region surcharges: {region_surcharge}")
    return proposal

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
