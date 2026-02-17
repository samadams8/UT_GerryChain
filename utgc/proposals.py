import random
from functools import partial
from typing import Callable, Dict, Optional

from gerrychain import Partition
from gerrychain.proposals import recom
from gerrychain.tree import bipartition_tree

from . import run_utils as rutil


def create_recom_proposal(
    population_params: Dict,
    region_surcharges: Dict,
    edge_penalties: Dict,
    num_districts: Optional[int] = None,
    pop_tolerance: Optional[float] = None,
) -> Callable[[Partition], Partition]:
    """
    Create a ReCom proposal function with optional edge penalties and region surcharges.

    Parameters
    ----------
    population_params : dict
        Must contain "column_id", "ideal_pop". "num_districts" is used for
        node_repeats if num_districts is not provided.
    region_surcharges : dict
        Map from column name to surcharge for edges crossing that boundary.
    edge_penalties : dict
        Map from edge tuple (u, v) (canonical sorted) to penalty weight.
    num_districts : int, optional
        Number of districts (node_repeats). Defaults to population_params["num_districts"].
    pop_tolerance : float, optional
        Population deviation tolerance (epsilon). Defaults to population_params["pop_tolerance"].

    Returns
    -------
    callable
        A proposal function with signature (partition) -> partition.
    """
    num_districts = num_districts or population_params.get("num_districts")
    if num_districts is None:
        raise ValueError("num_districts must be provided or present in population_params")
    ideal_population = population_params["ideal_pop"]
    tol = pop_tolerance if pop_tolerance is not None else population_params.get("pop_tolerance", 0.01)

    spanning_tree_fn = partial(
        rutil.random_spanning_tree_with_edge_penalties,
        edge_penalties=edge_penalties,
    )

    return partial(
        recom,
        pop_col=population_params["column_id"],
        pop_target=ideal_population,
        epsilon=tol,
        node_repeats=num_districts,
        region_surcharge=region_surcharges,
        method=partial(
            bipartition_tree,
            max_attempts=1000,
            allow_pair_reselection=True,
            spanning_tree_fn=spanning_tree_fn,
        ),
    )


def propose_population_flip(
    partition: Partition,
    ideal_pop: float = 0.0,
    pop_key: str = "population"
) -> Partition:
    """
    A proposal function that selects a random flip from among cut edges touching at least one unbalanced district.
    
    When determining whether a cut edge should be included in the set of candidates, we consider the population of the district at each of the constituent nodes:
    - If both nodes belong to a population balanced district, i.e., 
      |actual population - ideal population| < 1, then the edge is not included
      as a candidate
    - Otherwise, at least one node belongs to a district with unbalanced
      population, and the edge is included for random selection.
    
    Args:
        partition: The current partition.
        ideal_pop: The target ideal population for districts.
        pop_key: The key for the population updater in the partition. Defaults to "population".
        
    Returns:
        A new partition after the flip, or the original partition if no valid flip is found.
    """
    
    dist_pops = partition[pop_key]
    candidate_edges = []
    
    # Iterate over cut edges to filter them
    for u, v in partition.cut_edges:
        dist_u = partition.assignment[u]
        dist_v = partition.assignment[v]
        
        pop_u = dist_pops[dist_u]
        pop_v = dist_pops[dist_v]
        
        # Check strict balance condition (|diff| < 1)
        balanced_u = abs(pop_u - ideal_pop) < 1
        balanced_v = abs(pop_v - ideal_pop) < 1
        
        # If both districts are balanced, exclude this edge
        if balanced_u and balanced_v:
            continue
            
        candidate_edges.append((u, v))
    
    # If no edges qualify, return the current partition (no move)
    if not candidate_edges:
        return partition
        
    # Choose a random edge from the filtered candidates
    edge = random.choice(candidate_edges)
    index = random.choice((0, 1))
    flipped_node, other_node = edge[index], edge[1 - index]
    flip = {flipped_node: partition.assignment.mapping[other_node]}
    return partition.flip(flip)
