"""
Congressional Equal-Population Map Generator
=============================================

Two-phase pipeline for generating diverse, equal-population congressional maps:

Phase 1 (Coarse):
    Run ReCom on a capped dataset to generate diverse maps respecting statutory
    constraints (UTC 20A-19-103). Each output step corresponds to multiple internal
    ReCom steps (coupon-collector spacing) so that, on average, every district
    boundary is perturbed at least once between outputs.

Phase 2 (Fine):
    Transfer each coarse map to census-block-level data and run lexicographic
    simulated-annealing optimization in statutory priority order:
        1. Equal population  (minimize max |pop - ideal|)
        2. Municipality splits (minimize)
        3. County splits (minimize)
        4. Compactness (maximize mean Polsby-Popper)

Outputs per map:
    - Rendered image (full state + Wasatch Front zoom)
    - Metrics in output.jsonl (all updaters evaluated on the fine partition)
    - Optional: GeoJSON / Shapefile export

Designed for reuse with any map type (congressional, senate, house, school board)
by changing only the configuration section below.
"""

import os
import json
import math
import random
import asyncio
from math import ceil
from functools import partial
from typing import Optional, Dict, Any, List, Tuple, Iterator
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from networkx.readwrite import json_graph

from gerrychain import (
    Graph, GeographicPartition, Partition, MarkovChain,
    updaters, accept,
)
from gerrychain.constraints import contiguous
from gerrychain.proposals import propose_random_flip
from gerrychain.optimization import SingleMetricOptimizer
from gerrychain.metrics import polsby_popper

from concurrent.futures import ProcessPoolExecutor

from utgc.geography import GeographyManager
from utgc.configuration import ConfigurationManager
from utgc.optimization import LexicographicOptimizer, OptimizationMetric
from utgc.preconditioning import precondition
import utgc.plotting as gcplt
from utgc.notebookhelper import load_boundaries_from_shapefiles

# =============================================================================
# Configuration — change this section for different map types
# =============================================================================

# Tag for output directory naming
config_tag = "cong_equalpop"

# Initial plan (district boundaries shapefile)
initial_plan = "maps/US-House/2025_USH_Leg-C/2025_USH_Leg-C.shp"

# Population geodata at two resolutions
geo = GeographyManager(
    pop_data={
        "blocks": "data/UT_blocks.geojson",
        "d4-cap": "data/UT_capped_d4_eps1e-3.geojson",
    },
    crs="EPSG:26912",
)

# Number of output maps to generate
num_output_steps = 25

# Random seed
random_seed = 1847

# Parallel optimization workers
max_workers = 4

# Annealing steps per optimization metric phase
annealing_steps = 10000

# Output directory
output_dir = os.path.join("output", config_tag or datetime.now().strftime("%Y%m%d%H%M%S"))

# Export options
export_geojson = False
export_shapefile = False





# =============================================================================
# Build full configuration (mirrors notebooks 01 + 02)
# =============================================================================

def build_configuration(
    geo: GeographyManager,
    pop_key: str,
    num_districts: int,
) -> ConfigurationManager:
    """Build a fully configured ConfigurationManager with all updaters."""
    cfg = ConfigurationManager()

    # Population
    cfg = cfg.set_pop_column("TOTPOP")
    cfg = cfg.add_pop_dev_updater()

    # Region split constraints + surcharges
    cfg = (cfg
        .constrain_region_splits(
            name="muni",
            column_id="MUNIID",
            num_split=math.floor(num_districts / 2),
            num_multi_splits=math.floor(num_districts / 6),
        )
        .constrain_region_splits(
            name="county",
            column_id="COUNTYID",
            num_split=round(num_districts / 2) + 1,
            num_multi_splits=math.floor(num_districts / 5),
        )
        .constrain_not_equal(not_equal_constraint=True)
        .surcharge_region(column_id="MUNIID", surcharge=1)
        .surcharge_region(column_id="COUNTYID", surcharge=0.5)
        .surcharge_region(column_id="HIGHEREDID", surcharge=0.1)
        .surcharge_region(column_id="AIANNHID", surcharge=0.1)
        .surcharge_region(column_id="MILITID", surcharge=0.1)
        .surcharge_region(column_id="CBSAID", surcharge=0.1)
    )

    # Shape metrics
    cfg = cfg.add_shape_metrics(["polsby_popper"])

    # Election updaters
    election_dicts = geo.build_election_dicts(
        pop_key,
        years=[2016, 2020, 2024],
        offices=["PRE", "GOV", "ATG", "AUD", "TRE"],
        parties=["D", "R", "-"],
        overrides={
            "2024GOV": {"R1": "G24GOVRHEN", "R2": "G24GOVNCLA"},
        },
    )
    cfg = cfg.add_election_updaters(
        elections=election_dicts,
        skip_if_missing_parties=True,
    )

    # Partisan data aggregator
    sb1011_elections = [
        name for name in sorted(election_dicts.keys())
        if any(name.startswith(str(y)) for y in [2016, 2020, 2024])
        and any(name.endswith(o) for o in ["PRE", "GOV", "ATG", "AUD", "TRE"])
    ]
    cfg = cfg.add_election_aggregator(
        "sb1011_data",
        sb1011_elections,
        parties=["D", "R", "-"],
    )
    cfg = cfg.add_election_metric_updaters(
        "sb1011_data",
        [
            "partisan_bias_utah",
            "partisan_bias",
            "mean_median",
            "efficiency_gap",
            "stdev_partisan_share",
            "majority_partisan_shares",
            "majority_seats",
        ],
    )

    return cfg


# =============================================================================
# Coupon-collector spacing
# =============================================================================

def compute_recom_spacing(k: int) -> int:
    """
    Compute the number of ReCom steps needed between output maps so that,
    on average, every district boundary has been perturbed at least once.

    ReCom touches 2 districts per step. By the coupon-collector problem,
    expected steps = k * H(k) / 2, where H(k) is the k-th harmonic number.
    """
    harmonic = sum(1.0 / i for i in range(1, k + 1))
    return ceil(k * harmonic / 2)


# =============================================================================
# Phase 1: Coarse ReCom chain
# =============================================================================

def run_coarse_chain(
    geo: GeographyManager,
    cfg: ConfigurationManager,
    plan_path: str,
    coarse_key: str,
    seed: int,
    max_output_steps: int = 1000,
) -> Iterator[Tuple[int, Partition]]:
    """
    Run ReCom on the coarse dataset and yield every `spacing`-th partition.

    Returns an iterator yielding (output_step, partition).
    """
    random.seed(seed)
    np.random.seed(seed)

    # Fill empty IDs so locality splits work
    geo.fill_empty_ids(coarse_key, ["MUNIID"])

    # Build initial partition on coarse data
    partition = geo.build_partition(
        coarse_key,
        plan_path,
        updaters=cfg.updaters,
        repair_contiguity=True,
    )

    num_districts = len(partition)
    total_pop = sum(partition["population"].values())
    spacing = compute_recom_spacing(num_districts)
    total_internal_steps = max_output_steps * spacing

    print(f"Coarse chain: {num_districts} districts, "
          f"spacing={spacing} steps/output, "
          f"max_internal={total_internal_steps}")

    # Build proposal
    proposal = cfg.proposal(
        partition,
        total_population=total_pop,
        num_districts=num_districts,
        pop_tolerance=0.01,
    )

    # Precondition: bring the initial partition into constraint compliance
    # Extract constraint params from construction history
    constraint_params = {}
    for step in cfg.construction_history:
        if step.get("method") == "constrain_region_splits":
            kwargs = step.get("kwargs") or {}
            name = kwargs.get("name") or kwargs.get("column_id") or "unknown"
            if kwargs.get("num_split") is not None:
                constraint_params[f"split_{name}"] = kwargs["num_split"]
            if kwargs.get("num_multi_splits") is not None:
                constraint_params[f"{name}_multi_splits"] = kwargs["num_multi_splits"]
        elif step.get("method") == "constrain_not_equal":
            constraint_params["not_equal_constraint"] = (
                step.get("kwargs") or {}
            ).get("not_equal_constraint", True)
    population_params = {
        "ideal_pop": total_pop / num_districts,
        "pop_tolerance": 0.01,
        "column_id": cfg.population_params["column_id"],
        "num_districts": num_districts,
        "total_pop": total_pop,
    }
    coarse_graph = geo.get_graph(coarse_key)
    partition = precondition(
        initial_partition=partition,
        proposal=proposal,
        constraints=cfg.constraints,
        constraint_params=constraint_params,
        population_params=population_params,
        graph=coarse_graph,
        updaters=cfg.updaters,
        steps=50,
        max_attempts=3,
    )

    chain = MarkovChain(
        proposal=proposal,
        constraints=cfg.constraints,
        accept=accept.always_accept,
        initial_state=partition,
        total_steps=total_internal_steps,
    )

    for i, part in enumerate(chain, 1):
        if i % spacing == 0:
            output_step = i // spacing
            print(f"  Coarse step {output_step} "
                  f"(internal step {i})")
            yield output_step, part
        # Free memory
        part.parent = None


# =============================================================================
# Phase 1→2: Transfer coarse assignment to blocks
# =============================================================================

def transfer_to_blocks(
    geo: GeographyManager,
    coarse_partition: Partition,
    coarse_key: str,
    blocks_key: str,
    updaters_dict: Dict,
) -> GeographicPartition:
    """
    Transfer a coarse partition to block-level data.

    Dissolves the coarse geometry by district assignment to create a plan
    GeoDataFrame, then uses geo.build_partition on the blocks dataset.
    """
    coarse_gdf = geo.get_pop_geodata(coarse_key).copy()

    # Assign districts from the coarse partition
    assignment_map = dict(coarse_partition.assignment)
    coarse_gdf["_district"] = coarse_gdf.index.map(
        lambda idx: assignment_map.get(idx)
    )

    # Drop any nodes without assignment
    assigned = coarse_gdf.dropna(subset=["_district"])

    # Dissolve by district to create plan polygons
    plan_gdf = assigned.dissolve(by="_district", aggfunc="first").reset_index()
    plan_gdf = plan_gdf[["_district", "geometry"]]

    # Build partition on blocks
    blocks_partition = geo.build_partition(
        blocks_key,
        plan_gdf,
        updaters=updaters_dict,
        repair_contiguity=True,
    )

    return blocks_partition


# =============================================================================
# Phase 2: Worker initialization and optimization (parallelized)
# =============================================================================

# Global variables for worker processes
worker_graph = None
worker_ideal_pop = None
worker_updaters = None
worker_num_districts = None
worker_annealing_steps = None


def _pop_dev_updater_fn(p: Partition) -> Dict:
    pop = p["population"]
    if not pop:
        return {}
    ideal = sum(pop.values()) / len(pop)
    return {k: v - ideal for k, v in pop.items()}


def init_worker(
    graph_data: Dict,
    ideal_pop: float,
    num_districts: int,
    optimization_steps: int,
):
    """Initialize worker process with graph and config."""
    global worker_graph, worker_ideal_pop, worker_updaters
    global worker_num_districts, worker_annealing_steps

    g = json_graph.adjacency_graph(graph_data)
    worker_graph = Graph.from_networkx(g)
    worker_ideal_pop = ideal_pop
    worker_num_districts = num_districts
    worker_annealing_steps = optimization_steps

    worker_updaters = {
        "population": updaters.Tally("TOTPOP", alias="population"),
        "pop_dev": _pop_dev_updater_fn,
    }


def _popdev_absmax(p: Partition) -> float:
    """Metric: maximum absolute population deviation across all districts."""
    return max(abs(v) for v in p["pop_dev"].values())


def run_optimization_task(
    assignment: Dict,
    seed: int,
    step_num: int,
) -> Tuple[int, Dict, float]:
    """
    Run lexicographic optimization on a blocks-level partition.

    Uses propose_random_flip for all phases. Priority order:
        1. Population deviation (minimize absmax)
        2. (future: muni splits, county splits, compactness — requires
           locality split updaters which are difficult to serialize;
           for now, focus on pop dev)

    Returns (step_num, pop_dev_dict, best_score).
    """
    random.seed(seed)
    np.random.seed(seed)

    partition = GeographicPartition(
        worker_graph,
        assignment=assignment,
        updaters=worker_updaters,
    )

    initial_absmax = _popdev_absmax(partition)
    print(f"Step {step_num}: Initial max pop dev = {initial_absmax:.1f}")

    best_part = partition
    best_score = initial_absmax

    from gerrychain.optimization import SingleMetricOptimizer
    beta_schedule = SingleMetricOptimizer.linear_jumpcycle_beta_function(
        duration_hot=0,
        duration_cooldown=worker_annealing_steps,
        duration_cold=0,
    )

    def sa_accept(part):
        if part.parent is None:
            return True
        parent_score = _popdev_absmax(part.parent)
        part_score = _popdev_absmax(part)
        score_delta = part_score - parent_score # maximize=False
        
        step = part["step"] if "step" in part.updaters else (
            0 if part.parent is None else part.parent.get("step", 0) + 1
        )
        beta = beta_schedule(step)
        
        exponent = -beta * 100 * score_delta
        if exponent > 700: # math.exp overflows ~709
            return True
        return random.random() < math.exp(exponent)

    # Add step updater
    if "step" not in partition.updaters:
        partition.updaters["step"] = lambda p: 0 if p.parent is None else p.parent["step"] + 1

    chain = MarkovChain(
        proposal=propose_random_flip,
        constraints=[contiguous],
        accept=sa_accept,
        initial_state=partition,
        total_steps=worker_annealing_steps,
    )

    for i, p in enumerate(chain):
        score = _popdev_absmax(p)
        if score < best_score:
            best_part = p
            best_score = score
        
        if score < 1.0:
            best_part = p
            print(f"  Step {step_num}: Equal pop achieved at annealing step {i}")
            break

    final_popdev = best_part["pop_dev"]
    final_absmax = max(abs(v) for v in final_popdev.values())
    print(f"  Step {step_num}: Final max pop dev = {final_absmax:.1f}")

    # Return optimized assignment so main process can rebuild full partition
    optimized_assignment = dict(best_part.assignment)
    return step_num, optimized_assignment, dict(final_popdev), best_score


# =============================================================================
# Output: render map + save metrics
# =============================================================================

def render_map(
    partition: GeographicPartition,
    step: int,
    output_dir: str,
    subdir: str = "maps",
    municipalities: Optional[gpd.GeoDataFrame] = None,
    counties: Optional[gpd.GeoDataFrame] = None,
):
    """Render partition with full state + Wasatch Front zoom."""
    maps_dir = os.path.join(output_dir, subdir)
    gcplt.visualize_partition(
        partition,
        step,
        maps_dir,
        split_munis_count=(
            partition["split_muni"]
            if "split_muni" in partition.updaters else None
        ),
        split_counties_count=(
            partition["split_county"]
            if "split_county" in partition.updaters else None
        ),
        municipalities=municipalities,
        counties=counties,
        auto_load_boundaries=False if (municipalities is not None or counties is not None) else True,
    )


def collect_metrics(
    partition: GeographicPartition,
    step: int,
    updater_names: List[str],
) -> Dict[str, Any]:
    """Collect all updater values into a serializable dict."""
    data: Dict[str, Any] = {"step": step}
    for name in updater_names:
        try:
            value = partition[name]
            # Handle ElectionResults object specifically
            if hasattr(value, "percents"):
                data[name] = {
                    "percents": value.percents("D"),
                    "counts": value.counts("D"),
                    "wins": value.wins("D"),
                    "seats": value.seats("D"),
                }
            elif isinstance(value, dict):
                data[name] = {str(k): v for k, v in sorted(value.items())}
            elif isinstance(value, pd.DataFrame):
                data[name] = value.to_dict(orient="records")
            else:
                data[name] = value
        except Exception:
            pass  # Skip updaters that fail (e.g., missing data)
    return data


def save_geo_output(
    partition: GeographicPartition,
    step: int,
    pop_gdf: gpd.GeoDataFrame,
    output_dir: str,
    geojson: bool = False,
    shapefile: bool = False,
):
    """Export the partition assignment as GeoJSON or Shapefile."""
    if not geojson and not shapefile:
        return

    geo_dir = os.path.join(output_dir, "geo")
    os.makedirs(geo_dir, exist_ok=True)

    # Build output GeoDataFrame
    assignment_map = dict(partition.assignment)
    out_gdf = pop_gdf.copy()
    out_gdf["district"] = out_gdf.index.map(lambda idx: assignment_map.get(idx))
    dissolved = out_gdf.dissolve(by="district", aggfunc="sum").reset_index()

    if geojson:
        path = os.path.join(geo_dir, f"step_{step:05d}.geojson")
        dissolved.to_file(path, driver="GeoJSON")

    if shapefile:
        path = os.path.join(geo_dir, f"step_{step:05d}.shp")
        dissolved.to_file(path, driver="ESRI Shapefile")


# =============================================================================
# Main pipeline
# =============================================================================

async def main():
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Determine number of districts from initial plan
    plan_gdf = gpd.read_file(initial_plan)
    num_districts = len(plan_gdf)
    print(f"Number of districts: {num_districts}")

    # Build configuration
    print("Building configuration...")
    cfg = build_configuration(geo, "blocks", num_districts)

    # Fill empty IDs on blocks for locality splits
    geo.fill_empty_ids("blocks", ["MUNIID"])

    # Load boundaries once to avoid reloading during map generation
    print("Loading municipality and county boundaries...")
    municipalities, counties = load_boundaries_from_shapefiles("data/bounds")

    # ------------------------------------------------------------------
    # Phase 1: Coarse ReCom chain (Initialization)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 1: Coarse ReCom Chain")
    print("=" * 60)

    coarse_generator = run_coarse_chain(
        geo=geo,
        cfg=cfg,
        plan_path=initial_plan,
        coarse_key="d4-cap",
        seed=random_seed,
        max_output_steps=1000, # Large batch max
    )

    # ------------------------------------------------------------------
    # Phase 2: Transfer to blocks + parallel optimization + output
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 2 & 3: Parallel Optimization and Progressive Output")
    print("=" * 60)

    # Build minimal updaters for the optimization workers (pop only)
    blocks_opt_updaters = {
        "population": updaters.Tally("TOTPOP", alias="population"),
        "pop_dev": _pop_dev_updater_fn,
    }

    # Transfer first partition to get graph data for workers
    print("Generating first coarse partition to prepare worker data...")
    first_coarse_step, first_coarse_part = next(coarse_generator)
    
    # Save the first coarse map
    render_map(first_coarse_part, first_coarse_step, output_dir, subdir="maps/coarse", municipalities=municipalities, counties=counties)

    first_blocks_partition = transfer_to_blocks(
        geo, first_coarse_part, "d4-cap", "blocks", blocks_opt_updaters
    )
    blocks_graph = first_blocks_partition.graph

    total_pop = sum(first_blocks_partition["population"].values())
    ideal_pop = total_pop / num_districts
    print(f"Total population: {total_pop:,.0f}")
    print(f"Ideal population: {ideal_pop:,.1f}")

    # Serialize graph for workers — convert to plain networkx Graph first
    print("Serializing blocks graph for workers...")
    import networkx as nx
    nx_graph = nx.Graph()
    for node in blocks_graph.nodes:
        node_data = dict(blocks_graph.nodes[node])
        node_data.pop("geometry", None)
        nx_graph.add_node(node, **node_data)
    for u, v in blocks_graph.edges:
        nx_graph.add_edge(u, v)
    graph_data = json_graph.adjacency_data(nx_graph)

    # Create a fresh configuration for the fine outputs so updaters don't cache the coarse graph
    fine_cfg = build_configuration(geo, "blocks", num_districts)
    full_updaters = dict(fine_cfg.updaters)
    full_updaters["pop_dev"] = _pop_dev_updater_fn
    blocks_gdf = geo.get_pop_geodata("blocks")
    blocks_graph_full = geo.get_graph("blocks")
    output_jsonl_path = os.path.join(output_dir, "output.jsonl")
    updater_names = list(cfg.updaters.keys())

    print(f"\nStarting generator pool ({max_workers} workers, "
          f"{annealing_steps} steps per partition)...")

    loop = asyncio.get_running_loop()
    
    successful_maps = 0
    generated_coarse = 1
    
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(graph_data, ideal_pop, num_districts, annealing_steps),
    ) as executor:
        
        tasks = set()
        
        def submit_optimization_task(step_seq, assignment, seed_val):
            task = loop.run_in_executor(
                executor,
                run_optimization_task,
                assignment,
                seed_val,
                step_seq,
            )
            tasks.add(task)

        # Submit the very first one
        submit_optimization_task(1, dict(first_blocks_partition.assignment), random_seed + 1)
        
        def dispatch_next_coarse():
            nonlocal generated_coarse
            generated_coarse += 1
            step_seq = generated_coarse
            
            c_step, c_part = next(coarse_generator)
            render_map(c_part, c_step, output_dir, subdir="maps/coarse", municipalities=municipalities, counties=counties)
            
            b_part = transfer_to_blocks(
                geo, c_part, "d4-cap", "blocks", blocks_opt_updaters
            )
            assign = dict(b_part.assignment)
            submit_optimization_task(step_seq, assign, random_seed + step_seq)

        # Build up initial workers
        initial_deploy = min(max_workers, num_output_steps)
        while len(tasks) < initial_deploy:
            dispatch_next_coarse()

        with open(output_jsonl_path, "w") as f:
            while successful_maps < num_output_steps and tasks:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                
                for completed_task in done:
                    tasks.remove(completed_task)
                    
                    try:
                        step_num, opt_assignment, pop_dev, score = completed_task.result()
                        print(f"  Completed fine step {step_num}: score={score:.4f}")
                        
                        full_partition = GeographicPartition(
                            blocks_graph_full,
                            assignment=opt_assignment,
                            updaters=full_updaters,
                        )
                        render_map(full_partition, step_num, output_dir, municipalities=municipalities, counties=counties)
                        
                        metrics = collect_metrics(full_partition, step_num, updater_names)
                        metrics["optimization_score"] = score
                        f.write(json.dumps(metrics) + "\n")
                        f.flush()
                        
                        if export_geojson or export_shapefile:
                            save_geo_output(
                                full_partition, step_num, blocks_gdf, output_dir,
                                geojson=export_geojson,
                                shapefile=export_shapefile,
                            )
                            
                        if score < 1.0:
                            successful_maps += 1
                            print(f"  --> Map {step_num} was successful. (Total success: {successful_maps}/{num_output_steps})")
                        else:
                            print(f"  --> Map {step_num} failed equal pop threshold. Queueing replacement.")
                            
                    except Exception as e:
                        print(f"  WARNING: Failed to process task: {e}")
                        import traceback
                        traceback.print_exc()
                        
                # Keep generating more tasks until we hit enough successes
                while len(tasks) < max_workers and (successful_maps + len(tasks)) < num_output_steps:
                    dispatch_next_coarse()
                    
        print(f"\nDone! Results saved to {output_dir}")
        print(f"  Maps: {os.path.join(output_dir, 'maps')}")
        print(f"  Metrics: {output_jsonl_path}")


if __name__ == "__main__":
    asyncio.run(main())
