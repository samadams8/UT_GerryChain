from functools import partial
from gerrychain import Graph, GeographicPartition, updaters, constraints
from gerrychain.proposals import recom
from gerrychain.tree import bipartition_tree
from gerrychain.constraints import contiguous
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
        "perimeter": updaters.perimeter,
        "area": updaters.Tally("area", alias="area"),
        "county_locality_splits": LocalitySplits(
            name="county_locality_splits",
            col_id="COUNTYID",
            pop_col="TOTPOP",
            scores_to_compute=["num_split_localities", "num_parts"],
        ),
        "muni_locality_splits": LocalitySplits(
            name="muni_locality_splits",
            col_id="MUNIID",
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


def create_constraints(initial_partition, use_cut_edges=False, max_muni_splits=None, max_county_splits=None):
    """Create constraints according to Utah redistricting requirements."""
    print("Creating constraints...")

    population_constraint = constraints.within_percent_of_ideal_population(
        initial_partition, 0.001
    )
    contiguity_constraint = contiguous

    constraints_list = [population_constraint, contiguity_constraint]

    if use_cut_edges:
        print("Adding cut edges constraint for compactness...")
        initial_cut_edges = len(initial_partition["cut_edges"])
        max_cut_edges = int(initial_cut_edges * 1.1)

        def cut_edges_constraint(partition):
            return len(partition["cut_edges"]) <= max_cut_edges

        constraints_list.append(cut_edges_constraint)

    return constraints_list


def create_proposal(ideal_population, precincts, muni_surcharge=9, county_surcharge=3, highered_surcharge=1, metro_surcharge=0.5, schdist_surcharge=0.5, basin_surcharge=2.0, water_surcharge=2.0):
    """Create ReCom proposal with region surcharges."""
    print("Creating ReCom proposal...")

    region_surcharge = {}
    if "MUNIID" in precincts.columns and muni_surcharge > 0:
        region_surcharge["MUNIID"] = muni_surcharge
    if "COUNTYID" in precincts.columns and county_surcharge > 0:
        region_surcharge["COUNTYID"] = county_surcharge
    if "HIGHERED_ID" in precincts.columns and highered_surcharge > 0:
        region_surcharge["HIGHERED_ID"] = highered_surcharge
    if "METRO_ID" in precincts.columns and metro_surcharge > 0:
        region_surcharge["METRO_ID"] = metro_surcharge
    if "SCHDIST_ID" in precincts.columns and schdist_surcharge > 0:
        region_surcharge["SCHDIST_ID"] = schdist_surcharge
    if "BASIN_ID" in precincts.columns and basin_surcharge > 0:
        region_surcharge["BASIN_ID"] = basin_surcharge
    if "WATER_ID" in precincts.columns and water_surcharge > 0:
        region_surcharge["WATER_ID"] = water_surcharge

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


