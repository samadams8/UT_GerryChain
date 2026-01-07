import random
from gerrychain import Partition

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
