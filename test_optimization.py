import os
from warnings import warn
import maup
import pandas as pd
import geopandas as gpd
from functools import partial
import random
import math 
from typing import Optional, Dict, Any, List, Callable, Tuple, Literal, Union
import yaml
from math import ceil
from statistics import mean
import json
from datetime import datetime
import networkx as nx

from gerrychain import Graph, GeographicPartition, MarkovChain, Partition, updaters, accept
from gerrychain.optimization import SingleMetricOptimizer
from gerrychain.proposals import recom, propose_random_flip
from gerrychain.tree import bipartition_tree, random_spanning_tree
from gerrychain.constraints import contiguous, UpperBound
from gerrychain.metrics import polsby_popper
from gerrychain.updaters.locality_split_scores import LocalitySplits
import numpy as np

# Configuration files and example maps will be saved to a directory with the current date and time plus an optional user-defined tag
config_tag = "polish_test"  

# Data file path - provided files are in data/ 
# UT_blocks.geojson, UT_vtds.geojson, etc.
pop_geodata_path = "data/UT_blocks.geojson"
# pop_geodata_path = "data/UT_vtds.geojson"   

initial_plan_path = "maps/US-House/2025_USH_Leg-C/2025_USH_Leg-C.shp"

random_seed = 0

def _load_geodata(
    pop_geodata_path: str,
    initial_plan_path: str,
    crs: Optional[str] = "EPSG:26912",
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    geodata = gpd.read_file(pop_geodata_path)
    print(f"Loaded {len(geodata)} segments from {pop_geodata_path}")
    initial_plan = gpd.read_file(initial_plan_path)
    print(f"Loaded {len(initial_plan)} districts from {initial_plan_path}")

    print(f"Projecting to {crs}")
    geodata = geodata.to_crs(crs)
    initial_plan = initial_plan.to_crs(crs)

    # Create unique IDs for unincorporated municipalities
    if any(geodata["MUNIID"] == ""):
        print("Found %d nodes assigned to %d incorporated municipalities" % (
            (geodata["MUNIID"] != "").sum(),
            len(set(geodata[geodata["MUNIID"] != ""]["MUNIID"]))
        ))
        print("Assigning unique IDs to unincorporated nodes...")
        
        # Get existing numeric MUNIIDs
        existing_muniids = geodata[geodata["MUNIID"] != ""]["MUNIID"]
        if len(existing_muniids) > 0:
            max_id = int(existing_muniids.astype(int).max())
        else:
            max_id = 0

        # Generate unique sequential IDs for unincorporated areas
        unincorporated_mask = geodata["MUNIID"] == ""
        unincorporated_count = unincorporated_mask.sum()
        if unincorporated_count > 0:
            geodata.loc[unincorporated_mask, "MUNIID"] = np.arange(max_id + 1, max_id + 1 + unincorporated_count)
            print(f"Assigned unique IDs to {unincorporated_count} unincorporated nodes")
        
        # Print final municipality count
        num_unique_munis = len(set(geodata["MUNIID"]))
        print(f"Total unique MUNIIDs: {num_unique_munis}")

    # Assign initial plan to geodata
    geodata["initial_plan"] = maup.assign(geodata, initial_plan)
    geodata["area"] = geodata.geometry.area

    return geodata, initial_plan

def _repair_contiguity(partition: Partition) -> Dict:
    """
    Repair non-contiguous districts by reassigning disconnected components
    to adjacent districts. This function iterates until contiguity is achieved
    or max_iterations is reached.
    
    Parameters
    ----------
    partition : Partition
        The partition to repair
        
    Returns
    -------
    Dict
        A repaired assignment dictionary
    """
    print("Repairing contiguity...")
    graph = partition.graph
    repaired_assignment = dict(partition.assignment)
    
    # Create a temporary partition to check contiguity
    def _is_contiguous(assignment_dict):
        temp_partition = GeographicPartition(
            graph,
            assignment=assignment_dict,
            updaters={}
        )
        return contiguous(temp_partition)
    
    for iteration in range(2*len(partition)):
        if _is_contiguous(repaired_assignment):
            break
        
        # Handle unassigned nodes first
        unassigned_nodes = [
            node for node in graph.nodes 
            if repaired_assignment[node] is None or pd.isna(repaired_assignment[node])
        ]
        
        for node in unassigned_nodes:
            # Find adjacent districts
            neighbors = list(graph.neighbors(node))
            adjacent_districts = []
            for neighbor in neighbors:
                neighbor_dist = repaired_assignment.get(neighbor)
                if neighbor_dist is not None and not pd.isna(neighbor_dist):
                    adjacent_districts.append(neighbor_dist)
            
            if adjacent_districts:
                # Assign to the most common adjacent district
                repaired_assignment[node] = max(set(adjacent_districts), 
                                                key=adjacent_districts.count)
            else:
                # No adjacent districts found, assign to first available district
                available_districts = [d for d in set(repaired_assignment.values()) 
                                        if d is not None and not pd.isna(d)]
                if available_districts:
                    repaired_assignment[node] = random.choice(available_districts)
        
        # Now repair non-contiguous districts
        districts_to_check = set(repaired_assignment.values())
        districts_to_check.discard(None)
        
        repairs_made = False
        for district in districts_to_check:
            # Get all nodes in this district
            district_nodes = [node for node in graph.nodes 
                            if repaired_assignment[node] == district]
            
            if not district_nodes:
                continue
                
            # Find connected components within this district
            district_subgraph = graph.subgraph(district_nodes)
            components = list(nx.connected_components(district_subgraph))
            
            if len(components) <= 1:
                # District is already contiguous
                continue
            
            repairs_made = True
            # Sort components by size (largest first)
            components = sorted(components, key=len, reverse=True)
            
            # Keep the largest component in the original district
            # Reassign smaller components to adjacent districts
            for component in components[1:]:
                # Find adjacent districts for this component
                neighbor_districts = []
                for node in component:
                    neighbors = list(graph.neighbors(node))
                    for neighbor in neighbors:
                        neighbor_dist = repaired_assignment.get(neighbor)
                        if (neighbor_dist is not None and 
                            neighbor_dist != district and 
                            not pd.isna(neighbor_dist)):
                            neighbor_districts.append(neighbor_dist)
                
                if neighbor_districts:
                    # Use most common adjacent district
                    target_district = max(set(neighbor_districts), 
                                        key=neighbor_districts.count)
                else:
                    # No adjacent districts found, try to find any available district
                    available_districts = [d for d in districts_to_check if d != district]
                    if available_districts:
                        target_district = random.choice(available_districts)
                    else:
                        # Fallback: keep in original district (shouldn't happen)
                        target_district = district
                
                # Reassign all nodes in this component
                for node in component:
                    repaired_assignment[node] = target_district
        
        # If no repairs were made, break to avoid infinite loop
        if not repairs_made:
            break
    
    return repaired_assignment

geodata, initial_plan = _load_geodata(pop_geodata_path, initial_plan_path)

graph = Graph.from_geodataframe(geodata)

pop_column = "TOTPOP"
total_population = sum(geodata[pop_column])
num_districts = len(initial_plan)
ideal_pop = total_population / num_districts

constraints = [contiguous]
updaters = {
    "population": updaters.Tally(pop_column, alias="population"),
    "pop_dev": lambda p, target=ideal_pop: {
        k: v - target for k, v in p["population"].items()
    }
}

initial_partition = GeographicPartition(
    graph,
    assignment="initial_plan",
    updaters=updaters
)

if not contiguous(initial_partition):
    repaired_assignment = _repair_contiguity(initial_partition)
    initial_partition = GeographicPartition(
        graph,
        assignment=repaired_assignment
    )

print("Running ReCom randomizer ...")
chain = MarkovChain(
    proposal=partial(
        recom,
        pop_col=pop_column,
        pop_target=ideal_pop,
        epsilon=0.001,
        method=partial(
            bipartition_tree,
            max_attempts=1000,
            node_repeats=4,
            allow_pair_reselection=True,
            spanning_tree_fn=random_spanning_tree,
        )
    ),
    constraints=constraints,
    accept=accept.always_accept,
    initial_state=initial_partition,
    total_steps=10
)

from utgc.proposals import propose_population_flip

def popdev_metric(p: Partition) -> float:
    sse = sum([v**2 for v in p["pop_dev"].values()])

    if sse <= 0:
        return -1
    else:
        return math.log(sse)

for i, random_partition in enumerate(chain):
    print("New random partition", i)
    print(i, random_partition["pop_dev"])

    smo = SingleMetricOptimizer(
        # proposal=propose_random_flip,
        proposal=partial(
            propose_population_flip,
            ideal_pop=ideal_pop,
            pop_key="population"
        ),
        constraints=constraints,
        initial_state=random_partition,
        optimization_metric=popdev_metric,
        maximize=False
    )

    steps = 10000
    beta_schedule = SingleMetricOptimizer.linear_jumpcycle_beta_function(
        duration_hot=0,
        duration_cooldown=10000,
        duration_cold=0
    )

    for p in smo.simulated_annealing(
        num_steps=steps,
        beta_function=beta_schedule,
        beta_magnitude=100,
        with_progress_bar=True
    ):            
        if all([abs(v) < 1 for v in p["pop_dev"].values()]):
            smo._best_part = p
            smo._best_score = smo.score(p)
            break

    print(i, smo.best_part["pop_dev"])