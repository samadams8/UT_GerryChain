from functools import partial
import random
import numpy as np
from gerrychain import Graph, GeographicPartition, updaters, constraints
from gerrychain.proposals import recom
from gerrychain.tree import bipartition_tree
from gerrychain.constraints import contiguous, UpperBound
from gerrychain.updaters.locality_split_scores import LocalitySplits


def create_graph(precincts):
    """Create GerryChain graph from precincts."""
    print("Creating graph...")
    graph = Graph.from_geodataframe(precincts)
    print(f"Graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    return graph


def create_updaters(elections=[], election_columns=[]):
    """Create updaters for the ensemble analysis."""
    print("Creating updaters...")

    updaters_dict = {
        "population": updaters.Tally("TOTPOP", alias="population"),
        "cut_edges": updaters.cut_edges,
        "num_cut_edges": lambda partition: len(partition["cut_edges"]),
        "perimeter": updaters.perimeter,
        "area": updaters.Tally("area", alias="area"),
        "muni_locality_splits": LocalitySplits(
            name="muni_locality_splits",
            col_id="MUNIID",
            pop_col="TOTPOP",
            scores_to_compute=["num_split_localities", "num_parts"],
        ),
        "county_locality_splits": LocalitySplits(
            name="county_locality_splits",
            col_id="COUNTYID",
            pop_col="TOTPOP",
            scores_to_compute=["num_split_localities", "num_parts"],
        ),
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
    county_multi_splits_constraint=None
):
    """Create constraints according to Utah redistricting requirements."""
    print("Creating constraints...")

    population_constraint = constraints.within_percent_of_ideal_population(
        initial_partition, pop_deviation
    )
    contiguity_constraint = contiguous

    constraints_list = [population_constraint, contiguity_constraint]
    
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
        muni_ls = partition["muni_locality_splits"]
        return muni_ls.get("num_split_localities", 0)
    except Exception:
        return 0


def get_muni_multi_splits(partition):
    """Extract municipality multi-splits (num_parts - num_split_localities - total_munis) from partition."""
    try:
        muni_ls = partition["muni_locality_splits"]
        num_parts = muni_ls.get("num_parts", 0)
        num_split_localities = muni_ls.get("num_split_localities", 0)
        total_munis = len(set(partition.graph.nodes[node].get("MUNIID") for node in partition.graph.nodes if partition.graph.nodes[node].get("MUNIID")))
        return num_parts - num_split_localities - total_munis
    except Exception:
        return 0


def get_county_splits(partition):
    """Extract county split count (num_split_localities) from partition."""
    try:
        county_ls = partition["county_locality_splits"]
        return county_ls.get("num_split_localities", 0)
    except Exception:
        return 0


def get_county_multi_splits(partition):
    """Extract county multi-splits (num_parts - num_split_localities - total_counties) from partition."""
    try:
        county_ls = partition["county_locality_splits"]
        num_parts = county_ls.get("num_parts", 0)
        num_split_localities = county_ls.get("num_split_localities", 0)
        total_counties = len(set(partition.graph.nodes[node].get("COUNTYID") for node in partition.graph.nodes if partition.graph.nodes[node].get("COUNTYID")))
        return num_parts - num_split_localities - total_counties
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


