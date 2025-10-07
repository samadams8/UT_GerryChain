"""
Run an ensemble of plans in accordance with Utah's redistricting requirements.

This script now includes an optimization phase that runs for 20 steps (configurable) to minimize 
population deviation and city/county splits before running the main ensemble analysis.

Allow different input data to be used, but start by using the UT_precincts file
and using the 2021 Utah Congressional District plan as the initial partition

Political data can not be used to help draw lines, only to evaluate whether a plan is fair after it is drawn.

Neutral redistricting standards, in priority order:
1. Adhering to the Constitution of the United States and federal laws, such as the Voting Rights Act, 52 U.S.C. Secs. 10101 through 10702, including, to the extent required, achieving equal population among districts using the most recent national decennial enumeration made by the authority of the United States; [No more than 0.1% population deviation from the ideal is permitted]
2. Minimizing the division of municipalities and counties across multiple districts, giving first priority to minimizing the division of municipalities and second priority to minimizing the division of counties; [Use the municipal and county region assignments as a surcharge on region splitting; after each iteration, count how many cities and counties are split across districts]
3. creating districts that are geographically compact; [Optional cut edges constraint via --use-cut-edges flag]
4. creating districts that are contiguous and that allow for the ease of transportation throughout the district; [contiguity enforced; ease of transportation seems to emerge naturally from the sum of 5 and 6, as well]
5. preserving traditional neighborhoods and local communities of interest; [Use the COI data for higher ed, metro/micro statistical areas, and school districts and surcharges]
6. following natural and geographic features, boundaries, and barriers; and [Aligns well with county lines in some cases; also includes hydrologic basins and water planning areas]
7. maximizing boundary agreement among different types of districts. [No additional work]

The optimization phase uses GerryChain's SingleMetricOptimizer with tilted run optimization 
to minimize a combined objective function that includes population deviation and city/county 
split penalties with surcharges.

Command line options for region surcharges:
--muni-surcharge: Municipality region surcharge (default: 9.0, use 0 to disable)
--county-surcharge: County region surcharge (default: 3.0, use 0 to disable)
--highered-surcharge: Higher education COI surcharge (default: 1.0, use 0 to disable)
--metro-surcharge: Metro/micro statistical area COI surcharge (default: 0.5, use 0 to disable)
--schdist-surcharge: School district COI surcharge (default: 0.5, use 0 to disable)
--basin-surcharge: Hydrologic basin COI surcharge (default: 2.0, use 0 to disable)
--water-surcharge: Water planning area COI surcharge (default: 2.0, use 0 to disable)
--use-cut-edges: Enable compactness constraint via cut edges minimization

Metrics saved in results/ensemble_results.json and results/ensemble_summary.csv:

Basic metrics:
- step: ensemble step number
- population_deviation: per-district population deviation from ideal (as fraction)
- vote_share_agg: aggregation method used ("median", "mean", or "none")

Split metrics:
- split_counties_count: number of counties split across districts
- split_counties_extra_parts: total extra parts (n-1) for split counties
- split_munis_count: number of municipalities split across districts  
- split_munis_extra_parts: total extra parts (n-1) for split municipalities
- split_counties_names: list of split county names
- split_munis_names: list of split municipality names


Partisan metrics (when aggregation enabled):
- Republican_agg_seats: number of Republican seats under aggregated vote shares
- mean_median: mean minus median of Republican vote shares across districts
- partisan_bias: fraction of districts above mean Republican share minus 0.5
- efficiency_gap: efficiency gap computed from aggregated partisan shares
- partisan_gini: Gini coefficient of Republican vote shares across districts
- Republican_agg_share_d1, d2, etc.: sorted Republican vote shares by district

Per-election metrics (when aggregation disabled):
- {election}_Republican_total: total Republican votes for election
- {election}_Republican_wins: Republican seats won for election
- {election}_Republican_share_by_district: Republican vote share per district
- {election}_margin_pct_by_district: Republican margin per district

All data saved to results/ directory. Visualizations saved as .png files with split counts labeled.
"""

from math import ceil
import os
import sys
import argparse
import random
import json
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# GerryChain imports
from gerrychain import (
    Partition, Graph, MarkovChain, GeographicPartition,
    updaters, constraints, accept
)
from gerrychain.proposals import recom
from gerrychain.tree import bipartition_tree
from gerrychain.constraints import contiguous
from gerrychain.metrics import (
    partisan_bias, mean_median, efficiency_gap,
    polsby_popper
)
from gerrychain.optimization import SingleMetricOptimizer
from functools import partial
from gerrychain.updaters.locality_split_scores import LocalitySplits

# Spatial operations
import maup

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def load_data():
    """Load precinct data and initial congressional plan."""
    print("Loading data...")
    
    # Load precinct data
    precincts_path = "data/UT_precincts.geojson"
    if not os.path.exists(precincts_path):
        print(f"Error: {precincts_path} not found. Run 02_compile_precincts.py first.")
        sys.exit(1)
    
    precincts = gpd.read_file(precincts_path)
    print(f"Loaded {len(precincts)} precincts")
    
    # Load initial congressional plan
    initial_plan_path = "plans/CONG/2025_UT-C/2025_UT-C.shp"
    if not os.path.exists(initial_plan_path):
        print(f"Error: {initial_plan_path} not found.")
        sys.exit(1)
    
    initial_plan = gpd.read_file(initial_plan_path)
    print(f"Loaded initial plan with {len(initial_plan)} districts")
    
    # Ensure same CRS
    if precincts.crs != initial_plan.crs:
        initial_plan = initial_plan.to_crs(precincts.crs)
    
    # Assign precincts to districts
    precincts["CONGDIST"] = maup.assign(precincts, initial_plan)
    
    # Calculate area for each precinct
    precincts["area"] = precincts.geometry.area
    
    return precincts, initial_plan

def create_graph(precincts):
    """Create GerryChain graph from precincts."""
    print("Creating graph...")
    graph = Graph.from_geodataframe(precincts)
    print(f"Graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    return graph

def load_county_boundaries(precincts):
    """Load county boundaries for visualization overlay."""
    county_path = "data/cois/UtahCountyBoundaries/ut_cnty_2020_bound.shp"
    counties = None
    if os.path.exists(county_path):
        print(f"Loading county boundaries from {county_path}...")
        counties = gpd.read_file(county_path)
        # Transform to same CRS as precincts
        counties = counties.to_crs(precincts.crs)
        print(f"Loaded {len(counties)} counties")
    else:
        print(f"Warning: {county_path} not found.")
    return counties

def load_municipality_boundaries(precincts):
    """Load municipality boundaries for visualization overlay."""
    muni_path = "data/cois/UtahMunicipalBoundaries/Municipalities.shp"
    municipalities = None
    if os.path.exists(muni_path):
        print(f"Loading municipality boundaries from {muni_path}...")
        municipalities = gpd.read_file(muni_path)
        # Transform to same CRS as precincts
        municipalities = municipalities.to_crs(precincts.crs)
        print(f"Loaded {len(municipalities)} municipalities")
    else:
        print(f"Warning: {muni_path} not found.")
    return municipalities

def get_num_split_munis(partition):
    """Number of municipalities touching 2+ districts (from LocalitySplits)."""
    try:
        muni_ls = partition["muni_locality_splits"]
        return int(muni_ls.get("num_split_localities", 0))
    except Exception:
        return 0

def get_num_split_counties(partition):
    """Number of counties touching 2+ districts (from LocalitySplits)."""
    try:
        county_ls = partition["county_locality_splits"]
        return int(county_ls.get("num_split_localities", 0))
    except Exception:
        return 0

def combined_optimization_objective(
    partition,
    muni_surcharge=9.0,
    muni_splits_tolerance=None,
    county_surcharge=3.0, 
    county_splits_tolerance=None,
    pop_tolerance=0.001,
    num_districts=4
    ):
    """Combined objective function minimizing both population deviation and splits."""
    # Compute the total population
    total_population = sum(partition["population"].values())
    ideal_population = total_population / len(partition)

    # Get raw metrics
    pop_dev = population_deviation_objective(partition)
    muni_splits = get_num_split_munis(partition)
    county_splits = get_num_split_counties(partition)
    
    # Set default tolerances if not provided
    if muni_splits_tolerance is None:
        muni_splits_tolerance = 2 * num_districts
    if county_splits_tolerance is None:
        county_splits_tolerance = 2 * num_districts
    
    # Normalize components so that 1.0 is exactly passing threshold
    pop_component = pop_dev / pop_tolerance
    muni_component = muni_splits / muni_splits_tolerance
    county_component = county_splits / county_splits_tolerance
    
    # Combine normalized components (all should be minimized, 1.0 = passing threshold)
    return pop_component + muni_component + county_component

def population_deviation_objective(partition):
    """Objective function to minimize population deviation from ideal."""
    total_population = sum(partition["population"].values())
    ideal_population = total_population / len(partition)
    
    # Calculate maximum deviation from ideal
    max_deviation = 0
    for pop in partition["population"].values():
        max_deviation = max(max_deviation, abs(float(pop) - ideal_population) / ideal_population)
    
    return max_deviation

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
            scores_to_compute=["num_split_localities", "num_parts"]
        ),
        "muni_locality_splits": LocalitySplits(
            name="muni_locality_splits",
            col_id="MUNIID",
            pop_col="TOTPOP",
            scores_to_compute=["num_split_localities", "num_parts"]
        ),
    }

    # Add election updaters for each available election
    if len(elections) > 0:
        for election in elections:
            year, office = election.split('_')
            year_int = int(year)
            # Create election updater with the specific columns for this race
            dem_col = f"{year_int%100:02d}{office}D"
            rep_col = f"{year_int%100:02d}{office}R"
            
            # Check if both columns exist
            if dem_col in election_columns and rep_col in election_columns:
                # Create election updater with proper column mapping
                election_updater = updaters.Election(
                    name=election,
                    parties_to_columns={"Democratic": dem_col, "Republican": rep_col}
                )
                updaters_dict[election] = election_updater
    
    return updaters_dict

def create_constraints(initial_partition, use_cut_edges=False, max_muni_splits=None, max_county_splits=None):
    """Create constraints according to Utah redistricting requirements."""
    print("Creating constraints...")
    
    # Population constraint: no more than 0.1% deviation
    population_constraint = constraints.within_percent_of_ideal_population(
        initial_partition, 0.001
    )
    
    # Contiguity constraint
    contiguity_constraint = contiguous
    
    constraints_list = [population_constraint, contiguity_constraint]
    
    # Optional cut edges constraint for compactness
    if use_cut_edges:
        print("Adding cut edges constraint for compactness...")
        # Create a constraint that limits the number of cut edges
        # We'll use the initial partition's cut edges as a baseline
        initial_cut_edges = len(initial_partition["cut_edges"])
        max_cut_edges = int(initial_cut_edges * 1.1)  # Allow 10% increase
        
        def cut_edges_constraint(partition):
            return len(partition["cut_edges"]) <= max_cut_edges
        
        constraints_list.append(cut_edges_constraint)
    
    return constraints_list

def create_proposal(ideal_population, precincts, muni_surcharge=9, county_surcharge=3, 
                   highered_surcharge=1, metro_surcharge=0.5, schdist_surcharge=0.5,
                   basin_surcharge=2.0, water_surcharge=2.0):
    """Create ReCom proposal with region surcharges."""
    print("Creating ReCom proposal...")
    
    # Region surcharges for municipalities and counties
    region_surcharge = {}
    
    # Add municipality surcharge if MUNIID column exists
    if "MUNIID" in precincts.columns and muni_surcharge > 0:
        region_surcharge["MUNIID"] = muni_surcharge
    
    # Add county surcharge if COUNTYID column exists  
    if "COUNTYID" in precincts.columns and county_surcharge > 0:
        region_surcharge["COUNTYID"] = county_surcharge
    
    # Add COI surcharges
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
        method = partial(
            bipartition_tree,
            max_attempts=1000,
            allow_pair_reselection=True
        )
    )
    
    print(f"Region surcharges: {region_surcharge}")
    return proposal

def detect_election_data(precincts):
    """Detect available election data in precincts."""
    election_years = [2016, 2018, 2020, 2024]
    offices = ["PRE", "GOV", "ATG", "AUD", "TRE", "USS"]
    
    available_elections = []
    
    for year in election_years:
        for office in offices:
            # Check if we have both D and R data for this election
            dem_col = f"{year%100:02d}{office}D"
            rep_col = f"{year%100:02d}{office}R"
            
            if dem_col in precincts.columns and rep_col in precincts.columns:
                # Exclude elections where either party has zero statewide votes
                try:
                    dem_total = float(precincts[dem_col].fillna(0).sum())
                    rep_total = float(precincts[rep_col].fillna(0).sum())
                    if dem_total > 0 and rep_total > 0:
                        available_elections.append(f"{year}_{office}")
                except Exception:
                    # If summation fails, skip this election
                    pass
    
    return available_elections

def filter_elections(available_elections, years=None, offices=None):
    """Filter the detected elections by selected years and offices."""
    if years:
        years_set = set(int(y) for y in years)
    else:
        years_set = None
    if offices:
        offices_set = set(offices)
    else:
        offices_set = None
    filtered = []
    for e in available_elections:
        try:
            y_str, office = e.split('_')
            y = int(y_str)
            if (years_set is None or y in years_set) and (offices_set is None or office in offices_set):
                filtered.append(e)
        except Exception:
            # keep if parsing fails and no filters provided
            if years_set is None and offices_set is None:
                filtered.append(e)
    return filtered

def get_election_columns(precincts):
    """Get all election-related columns from precincts data."""
    election_columns = []
    
    # Pattern: YYOFFICEPARTY (e.g., 16PRER, 20GOVD, 24ATGO)
    for col in precincts.columns:
        if len(col) == 6 and col[:2].isdigit() and col[2:5] in ["PRE", "GOV", "ATG", "AUD", "TRE", "USS"] and col[5] in ["R", "D", "O"]:
            election_columns.append(col)
    
    return sorted(election_columns)

def calculate_partisan_metrics(partition, available_elections):
    """Calculate partisan metrics per election using GerryChain metrics."""
    metrics = {}
    if not available_elections:
        return metrics

    for election in available_elections:
        if election in partition.updaters:
            try:
                election_results = partition[election]
                # Compute standard metrics
                metrics[f"{election}_efficiency_gap"] = efficiency_gap(election_results)
                metrics[f"{election}_mean_median"] = mean_median(election_results)
                metrics[f"{election}_partisan_bias"] = partisan_bias(election_results)
            except Exception as e:
                print(f"Warning: partisan metrics failed for {election}: {e}")
                metrics[f"{election}_efficiency_gap"] = None
                metrics[f"{election}_mean_median"] = None
                metrics[f"{election}_partisan_bias"] = None

    return metrics

def calculate_compactness_metrics(partition):
    """Calculate compactness metrics for a partition."""
    metrics = {}
    
    try:
        metrics["polsby_popper"] = polsby_popper(partition)
    except:
        metrics["polsby_popper"] = np.nan
    
    return metrics

def create_initial_partition(graph, precincts, updaters_dict):
    """Create initial partition from the supplied plan assignment."""
    print("Creating initial partition...")
    initial_partition = GeographicPartition(
        graph,
        assignment="CONGDIST",
        updaters=updaters_dict
    )
    print(f"Initial partition created with {len(initial_partition)} districts")
    return initial_partition

def run_optimization(initial_partition, proposal, muni_surcharge=9.0, county_surcharge=3.0, 
                    popdev_tolerance=0.001, optimization_steps=20, optimization_probability=0.1,
                    split_munis_tolerance=None, split_counties_tolerance=None, max_attempts=5):
    """Run optimization to minimize population deviation and city/county splits using tilted run method."""
    num_districts = len(initial_partition)
    
    print(f"Running optimization for {optimization_steps} steps...")
    if split_munis_tolerance is not None and split_counties_tolerance is not None:
        print(f"Tolerance thresholds: pop_dev={popdev_tolerance:.4f}, muni_splits={split_munis_tolerance}, county_splits={split_counties_tolerance}")
    else:
        print(f"Tolerance thresholds: pop_dev={popdev_tolerance:.4f}, muni_splits={'unlimited' if split_munis_tolerance is None else split_munis_tolerance}, county_splits={'unlimited' if split_counties_tolerance is None else split_counties_tolerance}")
        
    # Create combined objective function with normalized components
    def objective_function(partition):
        return combined_optimization_objective(
            partition, 
            muni_surcharge=muni_surcharge, 
            county_surcharge=county_surcharge,
            pop_tolerance=popdev_tolerance,
            muni_splits_tolerance=split_munis_tolerance,
            county_splits_tolerance=split_counties_tolerance,
            num_districts=num_districts
        )
    
    # Retry logic - attempt optimization up to max_attempts times if tolerance is not met
    optimized_partition = initial_partition
    for attempt in range(max_attempts):
        if attempt > 0:
            print(f"Retrying optimization (attempt {attempt + 1}/{max_attempts})...")
        
        # Create optimizer
        optimizer = SingleMetricOptimizer(
            proposal=proposal,
            constraints=[contiguous],
            initial_state=optimized_partition,
            optimization_metric=objective_function,
            maximize=False  # We want to minimize the objective
        )
        
        if attempt == 0:
            print("Starting optimization...")
        
        # Run tilted run optimization (accepts worse plans with fixed probability)
        for i, partition in enumerate(optimizer.short_bursts(
            5, ceil(optimization_steps/5), with_progress_bar=True)):
            pass  # The optimizer automatically tracks best score and partition
        
        print(f"Optimized score: {optimizer.best_score}")

        # Use the optimizer's built-in best partition and score
        optimized_partition = optimizer.best_part
        
        # Check if the result passes tolerance tests
        pop_dev = population_deviation_objective(optimized_partition)
        muni_splits = get_num_split_munis(optimized_partition)
        county_splits = get_num_split_counties(optimized_partition)
        
        # Check if all tolerances are met
        pop_passes = pop_dev <= popdev_tolerance
        muni_passes = (split_munis_tolerance is None) or (muni_splits <= split_munis_tolerance)
        county_passes = (split_counties_tolerance is None) or (county_splits <= split_counties_tolerance)
        
        if pop_passes and muni_passes and county_passes:
            if attempt > 0:
                print(f"✓ Optimization successful on attempt {attempt + 1}! All tolerances met.")
            else:
                print(f"✓ Optimization successful! All tolerances met.")
            print(f"Final population deviation: {pop_dev:.6f}")
            print(f"Final municipality splits: {muni_splits}")
            print(f"Final county splits: {county_splits}")
            return optimized_partition
        else:
            if attempt < max_attempts - 1:  # Don't print on the last attempt
                print(f"✗ Attempt {attempt + 1} failed tolerance tests, retrying...")
    
    # If we get here, all attempts failed
    print(f"⚠️  WARNING: Optimization failed to meet tolerance requirements after {max_attempts} attempts")
    return optimized_partition

def run_ensemble(initial_partition, proposal, constraints_list, available_elections, counties=None, municipalities=None, num_steps=5000, visualize_every=10, vote_share_agg="median"):
    """Run the ensemble analysis."""
    print(f"Running ensemble analysis with {num_steps} steps...")
    
    chain = MarkovChain(
        proposal=proposal,
        constraints=constraints_list,
        accept=accept.always_accept,
        initial_state=initial_partition,
        total_steps=num_steps
    )
    
    results = []
    
    for i, partition in enumerate(chain.with_progress_bar()):
        # Calculate metrics for this partition
        # Compute population deviation from ideal per district
        pop_dict = dict(partition["population"]) if "population" in partition.updaters else {}
        if pop_dict:
            ideal_pop = sum(pop_dict.values()) / len(pop_dict)
            pop_dev = {k: (abs(float(v) - ideal_pop) / ideal_pop if ideal_pop > 0 else None) for k, v in pop_dict.items()}
        else:
            pop_dev = {}
        step_results = {
            "step": i,
            "population_deviation": pop_dev,
            "vote_share_agg": vote_share_agg,
        }
        
        # Use LocalitySplits updaters to compute split counts and extra pieces
        try:
            county_ls = partition["county_locality_splits"]
            # num_split_localities = number of split counties, num_parts = total locality-district pairs
            step_results["split_counties_count"] = county_ls.get("num_split_localities", 0)
            # Get total number of counties to compute extra pieces
            total_counties = len(set(partition.graph.nodes[node].get("COUNTYID") for node in partition.graph.nodes if partition.graph.nodes[node].get("COUNTYID")))
            step_results["split_counties_extra_parts"] = county_ls.get("num_parts", 0) - total_counties
        except Exception as e:
            print(f"Warning: county locality splits failed: {e}")
            step_results["split_counties_count"] = 0
            step_results["split_counties_extra_parts"] = 0

        try:
            muni_ls = partition["muni_locality_splits"]
            # num_split_localities = number of split municipalities, num_parts = total locality-district pairs
            step_results["split_munis_count"] = muni_ls.get("num_split_localities", 0)
            # Get total number of municipalities to compute extra pieces
            total_munis = len(set(partition.graph.nodes[node].get("MUNIID") for node in partition.graph.nodes if partition.graph.nodes[node].get("MUNIID")))
            step_results["split_munis_extra_parts"] = muni_ls.get("num_parts", 0) - total_munis
        except Exception as e:
            print(f"Warning: municipality locality splits failed: {e}")
            step_results["split_munis_count"] = 0
            step_results["split_munis_extra_parts"] = 0

        # Build ID->name maps for split names (still needed for reporting)
        county_id_to_name = {}
        muni_id_to_name = {}
        for node in partition.graph.nodes:
            nd = partition.graph.nodes[node]
            cid = nd.get("COUNTYID")
            cname = nd.get("COUNTYNAME") or nd.get("COUNTY")
            if cid is not None and cid != "" and cid not in county_id_to_name and cname:
                county_id_to_name[cid] = cname
            mid = nd.get("MUNIID")
            mname = nd.get("MUNINAME")
            if mid is not None and mid != "" and mid not in muni_id_to_name and mname:
                muni_id_to_name[mid] = mname

        # Get split names for reporting (using custom logic for now)
        county_to_districts = {}
        for node in partition.graph.nodes:
            node_data = partition.graph.nodes[node]
            county_id = node_data.get("COUNTYID")
            if county_id:
                dist = partition.assignment[node]
                if county_id not in county_to_districts:
                    county_to_districts[county_id] = set()
                county_to_districts[county_id].add(dist)
        split_counties = sorted([cid for cid, dists in county_to_districts.items() if len(dists) > 1])
        step_results["split_counties_names"] = sorted([county_id_to_name.get(cid, str(cid)) for cid in split_counties])

        muni_to_districts = {}
        for node in partition.graph.nodes:
            node_data = partition.graph.nodes[node]
            muni_id = node_data.get("MUNIID")
            if muni_id:
                dist = partition.assignment[node]
                if muni_id not in muni_to_districts:
                    muni_to_districts[muni_id] = set()
                muni_to_districts[muni_id].add(dist)
        split_munis = sorted([m for m, dists in muni_to_districts.items() if len(dists) > 1])
        step_results["split_munis_names"] = sorted([muni_id_to_name.get(mid, str(mid)) for mid in split_munis])
        
        # Add election results for each available election
        # If aggregating, do not record per-election outputs; compute only aggregated Republican share
        rep_shares_matrix = []
        for election in available_elections:
            if election in partition.updaters:
                election_results = partition[election]
                try:
                    # Per-district Republican and Democratic votes
                    rep_votes = list(election_results.votes("Republican"))
                    dem_votes = list(election_results.votes("Democratic"))
                except Exception as e:
                    print(f"Error getting per-district votes for {election}: {e}")
                    rep_votes = []
                    dem_votes = []

                # Republican share using partisan votes only: R / (R + D)
                shares = []
                n = min(len(rep_votes), len(dem_votes))
                for j in range(n):
                    r = rep_votes[j] or 0
                    d = dem_votes[j] or 0
                    total = r + d
                    shares.append((r / total) if total > 0 else None)

                if vote_share_agg == "none":
                    # Record per-election Republican totals, wins, and shares
                    try:
                        step_results[f"{election}_Republican_total"] = sum(rep_votes)
                    except Exception:
                        step_results[f"{election}_Republican_total"] = 0
                    step_results[f"{election}_Republican_votes_by_district"] = rep_votes
                    try:
                        step_results[f"{election}_Republican_wins"] = election_results.wins("Republican")
                    except Exception:
                        step_results[f"{election}_Republican_wins"] = 0
                    step_results[f"{election}_Republican_share_by_district"] = shares
                    # Also compute Republican margin (R-D)/(R+D)
                    margins_pct = []
                    for j in range(n):
                        r = rep_votes[j] or 0
                        d = dem_votes[j] or 0
                        total = r + d
                        margins_pct.append(((r - d) / total) if total > 0 else None)
                    step_results[f"{election}_margin_pct_by_district"] = margins_pct
                else:
                    # Aggregation mode: collect shares only (do not record per-election outputs)
                    if shares:
                        rep_shares_matrix.append(shares)

        # Optional aggregation of party vote share across selected elections (default median)
        if vote_share_agg in ("median", "mean") and len(available_elections) > 0:
            import statistics
            try:
                # Determine number of districts
                district_ids = list(partition.parts)
                num_districts = len(district_ids)
                # aggregate across elections
                rep_agg = []
                if rep_shares_matrix:
                    n = min(len(row) for row in rep_shares_matrix)
                    for j in range(n):
                        vals = [row[j] for row in rep_shares_matrix if row[j] is not None]
                        if len(vals) == 0:
                            rep_agg.append(None)
                        else:
                            rep_agg.append(statistics.median(vals) if vote_share_agg == "median" else sum(vals) / len(vals))
                    # Store sorted ascending for consistent positional reporting
                    rep_agg_sorted = sorted(rep_agg)
                    step_results["Republican_agg_share_by_district"] = rep_agg_sorted
                    # Aggregated Republican seats: count districts with share > 0.5
                    rep_seats = sum(1 for v in rep_agg_sorted if v is not None and v > 0.5)
                    step_results["Republican_agg_seats"] = int(rep_seats)
                    # Aggregated partisan metrics computed from aggregated shares
                    valid_shares = [v for v in rep_agg_sorted if v is not None]
                    if len(valid_shares) > 0:
                        try:
                            mean_share = sum(valid_shares) / len(valid_shares)
                            median_share = statistics.median(valid_shares)
                            step_results["mean_median"] = float(mean_share - median_share)
                            # Partisan bias: fraction of districts above mean minus 0.5
                            above_mean = sum(1 for v in valid_shares if v > mean_share)
                            step_results["partisan_bias"] = float(above_mean / len(valid_shares) - 0.5)
                            # Efficiency gap under equal-turnout assumption using partisan shares
                            # EG = (sum wasted_D - sum wasted_R) / num_districts
                            wasted_R = 0.0
                            wasted_D = 0.0
                            for s in valid_shares:
                                if s > 0.5:
                                    wasted_R += s - 0.5
                                    wasted_D += 1 - s
                                else:
                                    wasted_R += s
                                    wasted_D += 0.5 - s
                            step_results["efficiency_gap"] = float((wasted_D - wasted_R) / len(valid_shares))
                            # Partisan Gini: area between seats-votes curve and its reflection about (.5, .5)
                            # Sort shares ascending and compute Gini
                            sorted_shares = sorted(valid_shares)
                            n = len(sorted_shares)
                            if n > 1:
                                # Gini = 1 - 2 * sum(i * x_i) / (n * sum(x_i))
                                # For partisan Gini, we use the seats-votes curve
                                gini = 1.0 - 2.0 * sum((j + 1) * x for j, x in enumerate(sorted_shares)) / (n * sum(sorted_shares))
                                step_results["partisan_gini"] = float(gini)
                            else:
                                step_results["partisan_gini"] = None
                        except Exception as e:
                            print(f"Aggregation metrics error: {e}")
            except Exception as e:
                print(f"Aggregation error: {e}")
        
        # Add compactness metrics
        # compactness = calculate_compactness_metrics(partition)
        # step_results.update(compactness)
        
        # Add partisan metrics: if aggregation is disabled, compute per-election metrics; otherwise use aggregated metrics already computed
        if vote_share_agg == "none":
            partisan_metrics = calculate_partisan_metrics(partition, available_elections)
            step_results.update(partisan_metrics)
        
        results.append(step_results)
        
        # Save visualization every N steps
        if i % visualize_every == 0:
            save_visualization(partition, i, step_results, counties, municipalities)
    
    return results

def save_visualization(partition, step, results, counties=None, municipalities=None):
    """Save visualization of the partition."""
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Create figure with proper aspect ratio for Utah
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot partition
    partition.plot(ax=ax, cmap='tab20c')
    
    # Add municipality boundaries if available (plot first so they appear under county boundaries)
    if municipalities is not None:
        municipalities.boundary.plot(ax=ax, color='black', linewidth=0.25, alpha=0.5)
    
    # Add county boundaries if available
    if counties is not None:
        counties.boundary.plot(ax=ax, color='black', linewidth=1, alpha=0.5)
    
    # Add title with metrics
    title = f"Step {step}: Muni Splits: {results.get('split_munis_count', 0)}, County Splits: {results.get('split_counties_count', 0)}"
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    # Save figure with higher quality
    plt.savefig(f"results/step_{step:05d}.png", dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()

def save_results(results, available_elections):
    """Save results to JSON and CSV files."""
    print("Saving results...")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Save detailed results as JSON
    with open("results/ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Create summary DataFrame
    summary_data = []
    for result in results:
        summary_row = {
            "step": result["step"],
            "vote_share_agg": result.get("vote_share_agg", "none"),
            "split_counties_count": result.get("split_counties_count", 0),
            "split_counties_extra_parts": result.get("split_counties_extra_parts", 0),
            "split_munis_count": result.get("split_munis_count", 0),
            "split_munis_extra_parts": result.get("split_munis_extra_parts", 0),
        }
        
        # Add aggregated partisan metrics when present
        for metric_key in ["mean_median", "partisan_bias", "efficiency_gap", "partisan_gini"]:
            if metric_key in result:
                summary_row[metric_key] = result[metric_key]

        # Add election metrics (Republican-focused). If aggregation is enabled, do not include per-election outputs
        for election in available_elections:
            if "Republican_agg_share_by_district" not in result:
                rep_total_key = f"{election}_Republican_total"
                if rep_total_key in result:
                    summary_row[rep_total_key] = result[rep_total_key]
                # Include per-election partisan metrics if present
                for metric_name in ["efficiency_gap", "mean_median", "partisan_bias"]:
                    key = f"{election}_{metric_name}"
                    if key in result:
                        summary_row[key] = result[key]
                # Include Republican seat counts
                rep_wins_key = f"{election}_Republican_wins"
                if rep_wins_key in result:
                    summary_row[rep_wins_key] = result[rep_wins_key]
                # Aggregate per-district margins (mean margin pct) for compact summary
                margin_pct_key = f"{election}_margin_pct_by_district"
                if margin_pct_key in result and isinstance(result[margin_pct_key], list) and len(result[margin_pct_key]) > 0:
                    valid = [x for x in result[margin_pct_key] if x is not None]
                    if len(valid) > 0:
                        summary_row[f"{election}_avg_margin_pct"] = float(sum(valid) / len(valid))
        
        # Include aggregated Republican vote share and seats if present
        key = "Republican_agg_share_by_district"
        if "Republican_agg_seats" in result:
            summary_row["Republican_agg_seats"] = int(result["Republican_agg_seats"]) if result["Republican_agg_seats"] is not None else None

        # Include aggregated share by district as separate columns at the end
        if key in result and isinstance(result[key], list) and len(result[key]) > 0:
            for idx, share in enumerate(result[key], start=1):
                col_name = f"Republican_agg_share_d{idx}"
                summary_row[col_name] = None if share is None else float(share)
        
        summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)

    # Reorder columns to ensure district aggregated share columns come last
    district_cols = [c for c in summary_df.columns if c.startswith("Republican_agg_share_d")]
    non_district_cols = [c for c in summary_df.columns if c not in district_cols]
    summary_df = summary_df[non_district_cols + district_cols]
    summary_df.to_csv("results/ensemble_summary.csv", index=False)
    
    print(f"Results saved to results/ directory")
    print(f"Summary statistics:")
    print(f"  Munis split (avg count): {summary_df['split_munis_count'].mean():.2f}")
    print(f"  Munis extra parts (avg total): {summary_df['split_munis_extra_parts'].mean():.2f}")
    print(f"  Counties split (avg count): {summary_df['split_counties_count'].mean():.2f}")
    print(f"  Counties extra parts (avg total): {summary_df['split_counties_extra_parts'].mean():.2f}")

    # Print election summary (Republican-focused). Skip per-election printing if aggregation used
    if "Republican_agg_share_by_district" not in summary_df.columns:
        for election in available_elections:
            rep_col = f"{election}_Republican_total"
            if rep_col in summary_df.columns:
                print(f"  {election} - Average Republican votes: {summary_df[rep_col].mean():.0f}")
    else:
        # Print aggregated partisan metrics
        if "Republican_agg_seats" in summary_df.columns:
            print(f"  Aggregated Republican seats (avg): {summary_df['Republican_agg_seats'].mean():.2f}")
        if "mean_median" in summary_df.columns:
            print(f"  Mean-median: {summary_df['mean_median'].mean():.3f}")
        if "partisan_bias" in summary_df.columns:
            print(f"  Partisan bias: {summary_df['partisan_bias'].mean():.3f}")
        if "efficiency_gap" in summary_df.columns:
            print(f"  Efficiency gap: {summary_df['efficiency_gap'].mean():.3f}")
        if "partisan_gini" in summary_df.columns:
            print(f"  Partisan Gini: {summary_df['partisan_gini'].mean():.3f}")

def create_partisan_histogram_plots(summary_df):
    """Create histograms for partisan metrics (mean-median, partisan bias, efficiency gap, partisan gini)."""
    
    # Define the four partisan metrics to plot
    metrics = {
        'mean_median': 'Mean-Median Difference',
        'partisan_bias': 'Partisan Bias', 
        'efficiency_gap': 'Efficiency Gap',
        'partisan_gini': 'Partisan Gini'
    }
    
    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (col, title) in enumerate(metrics.items()):
        if col in summary_df.columns:
            ax = axes[i]
            data = summary_df[col].dropna()
            
            if len(data) > 0:
                # Special handling for partisan_bias (discrete values)
                if col == 'partisan_bias':
                    # Count number of districts
                    share_cols = [col for col in summary_df.columns if col.startswith('Republican_agg_share_d')]
                    num_districts = len(share_cols)

                    # Get unique values and sort them
                    unique_vals = sorted(data.unique())
                    # Create bins shifted by half their width
                    bin_edges = []
                    for val in unique_vals:
                        bin_edges.extend([val - 1/(num_districts*2), val + 1/(num_districts*2)])
                    # Remove duplicates and sort
                    bin_edges = sorted(list(set(bin_edges)))
                    ax.hist(data, bins=bin_edges, alpha=0.7, color='#6B7280', edgecolor='white', linewidth=0.8)
                    # Set x-axis ticks to show the discrete values
                    ax.set_xticks(unique_vals)
                    # Shift the x-axis by half the bin width (0.25)
                    ax.set_xlim([min(unique_vals) - 0.5, max(unique_vals) + 0.5])
                else:
                    # Regular histogram for continuous data
                    ax.hist(data, bins=20, alpha=0.7, color='#6B7280', edgecolor='white', linewidth=0.8)
                
                ax.set_title(f'Distribution of {title}', fontsize=12, fontweight='bold')
                ax.set_xlabel(title)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                mean_val = data.mean()
                median_val = data.median()
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
                ax.axvline(median_val, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_val:.3f}')
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, f'No data for {title}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Distribution of {title}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/ensemble_partisan_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_split_histogram_plots(summary_df):
    """Create histograms for split counts (muni splits, muni extra parts, county splits, county extra parts)."""
    
    # Define the four split metrics to plot
    metrics = {
        'split_munis_count': 'Municipality Splits',
        'split_munis_extra_parts': 'Municipality Extra Parts',
        'split_counties_count': 'County Splits',
        'split_counties_extra_parts': 'County Extra Parts'
    }
    
    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (col, title) in enumerate(metrics.items()):
        if col in summary_df.columns:
            ax = axes[i]
            data = summary_df[col].dropna()
            
            if len(data) > 0:
                # Use integer bins centered on integers with width 1
                min_val = int(data.min())
                max_val = int(data.max())
                # Add one empty bin on each side (unless min would be negative)
                bin_min = max(0, min_val - 1)
                bin_max = max_val + 1
                bins = [i - 0.5 for i in range(bin_min, bin_max + 1)]
                ax.hist(data, bins=bins, alpha=0.7, color='#6B7280', edgecolor='white', linewidth=0.8)
                ax.set_title(f'Distribution of {title}', fontsize=12, fontweight='bold')
                ax.set_xlabel(title)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                # Set x-axis ticks to integers only
                ax.set_xticks(range(bin_min, bin_max + 1))
                
                # Add statistics
                mean_val = data.mean()
                median_val = data.median()
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.1f}')
                ax.axvline(median_val, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_val:.1f}')
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, f'No data for {title}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Distribution of {title}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/ensemble_split_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_shares_and_seats_plots(summary_df):
    """Create violin plots for Republican vote shares across districts and histogram for Republican seats."""
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Violin plots for vote shares by district
    share_cols = [col for col in summary_df.columns if col.startswith('Republican_agg_share_d')]
    
    if share_cols:
        # Prepare data for violin plots
        share_data = []
        
        for col in share_cols:
            district_num = col.split('_')[-1]  # Extract district number (d1, d2, etc.)
            shares = summary_df[col].dropna()
            
            for share in shares:
                share_data.append({
                    'District': f'District {district_num[1:]}',  # Remove 'd' prefix
                    'Republican Share': share
                })
        
        if share_data:
            share_df = pd.DataFrame(share_data)
            
            # Create violin plot with diverging colormap
            sns.violinplot(data=share_df, x='District', y='Republican Share', ax=ax1, hue='District', palette='vlag', legend=False)
            
            # Create the specified diverging palette
            cmap = sns.color_palette("vlag", as_cmap=True)
            # cmap = sns.diverging_palette(250, 20, l=65, center="dark", as_cmap=True)
            
            # Get the actual range of Republican shares in the data
            min_share = share_df['Republican Share'].min()
            max_share = share_df['Republican Share'].max()
            
            # Create colors using the diverging palette with range [0, 1]
            colors = []
            for district in share_df['District'].unique():
                district_data = share_df[share_df['District'] == district]['Republican Share']
                # Use the mean share for this district to determine color
                median_share = district_data.median()
                # Get color from diverging colormap
                color = cmap(median_share)
                colors.append(color)
            
            # Apply colors to violin plot patches
            for i, patch in enumerate(ax1.collections):
                if hasattr(patch, 'set_facecolor'):
                    patch.set_facecolor(colors[i % len(colors)])
            ax1.set_title('Distribution of Republican Vote Shares by District', fontsize=12, fontweight='bold')
            ax1.set_xlabel('District')
            ax1.set_ylabel('Republican Vote Share')
            ax1.axhline(0.5, color='black', linestyle='--', alpha=0.7, label='50% Threshold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
        else:
            ax1.text(0.5, 0.5, 'No share data available', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Republican Vote Shares by District', fontsize=12, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No Republican vote share data found', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Republican Vote Shares by District', fontsize=12, fontweight='bold')
    
    # Right plot: Histogram for Republican seats
    if 'Republican_agg_seats' in summary_df.columns:
        seats_data = summary_df['Republican_agg_seats'].dropna()
        
        if len(seats_data) > 0:
            # Determine the range of possible seats (0 to total districts)
            # Get total number of districts from the data
            total_districts = len([col for col in summary_df.columns if col.startswith('Republican_agg_share_d')])
            if total_districts == 0:
                # Fallback: use max observed seats + 1
                max_seats = int(seats_data.max()) + 1
                bins = range(0, max_seats + 1)
            else:
                bins = range(0, total_districts + 1)
            
            # Create histogram with explicit bin edges to ensure all values 0-4 are shown
            bin_edges = [i - 0.5 for i in range(total_districts + 2)]
            ax2.hist(seats_data, bins=bin_edges, alpha=0.7, color='#6B7280', edgecolor='white', linewidth=0.8)
            ax2.set_title('Distribution of Republican Seats', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Number of Republican Seats')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = seats_data.mean()
            median_val = seats_data.median()
            ax2.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.1f}')
            ax2.axvline(median_val, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_val:.1f}')
            ax2.legend()
            
            # Set x-axis to show full range from 0 to total_districts
            ax2.set_xlim(-0.5, total_districts + 0.5)
            # Set x-axis ticks to show all integer values
            ax2.set_xticks(range(total_districts + 1))
        else:
            ax2.text(0.5, 0.5, 'No seat data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Distribution of Republican Seats', fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No Republican seat data found', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Distribution of Republican Seats', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/ensemble_shares_and_seats.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_plots(summary_df):
    """Create all summary plots."""
    print("Creating ensemble summary plots...")
    
    # Set style for better-looking plots
    plt.style.use('default')
    sns.set_palette("Blues")
    
    # Create plots
    create_partisan_histogram_plots(summary_df)
    create_split_histogram_plots(summary_df)
    create_shares_and_seats_plots(summary_df)

def main():
    """Main function to run the ensemble analysis."""
    print("Starting Utah redistricting ensemble analysis...")
    parser = argparse.ArgumentParser(description="Run Utah redistricting ensemble analysis")
    parser.add_argument("--years", type=str, default="2016,2020,2024", help="Comma-separated list of years to include, e.g., 2016,2020")
    parser.add_argument("--offices", type=str, default="PRE,GOV,ATG,AUD,TRE", help="Comma-separated list of offices to include, e.g., PRE,GOV,ATG,AUD,TRE,USS")
    parser.add_argument("--vote-share-agg", type=str, choices=["median", "mean", "none"], default="median", help="Aggregate party vote share across selected elections")
    parser.add_argument("--steps", type=int, default=21, help="Number of ensemble steps to run")
    parser.add_argument("--optimization-steps", type=int, default=20, help="Number of optimization steps to run before ensemble")
    parser.add_argument("--optimization-probability", type=float, default=0.1, help="Probability of accepting worse plans during optimization (tilted run)")
    parser.add_argument("--max-muni-splits", type=int, default=None, help="Maximum number of municipality splits allowed (None for no constraint)")
    parser.add_argument("--max-county-splits", type=int, default=None, help="Maximum number of county splits allowed (None for no constraint)")
    parser.add_argument("--viz-every", type=int, default=5, help="Save visualization every N steps")
    
    # Region surcharge arguments
    parser.add_argument("--muni-surcharge", type=float, default=9, help="Municipality region surcharge (0 to disable)")
    parser.add_argument("--county-surcharge", type=float, default=3, help="County region surcharge (0 to disable)")
    parser.add_argument("--highered-surcharge", type=float, default=1, help="Higher education COI surcharge (0 to disable)")
    parser.add_argument("--metro-surcharge", type=float, default=0.1, help="Metro/micro statistical area COI surcharge (0 to disable)")
    parser.add_argument("--schdist-surcharge", type=float, default=0.1, help="School district COI surcharge (0 to disable)")
    parser.add_argument("--water-surcharge", type=float, default=0.1, help="Water planning area surcharge (0 to disable)")
    parser.add_argument("--basin-surcharge", type=float, default=0.1, help="Hydrologic basin surcharge (0 to disable)")
    
    # Compactness argument
    parser.add_argument("--use-cut-edges", action="store_true", help="Enable compactness constraint via cut edges minimization")
    
    args = parser.parse_args()

    # Load data
    precincts, initial_plan = load_data()
    
    # Load county boundaries for visualization
    print("Loading county boundaries...")
    counties = load_county_boundaries(precincts)
    
    # Load municipality boundaries for visualization
    print("Loading municipality boundaries...")
    municipalities = load_municipality_boundaries(precincts)
    
    # Detect available election data
    available_elections = detect_election_data(precincts)
    election_columns = get_election_columns(precincts)
    # Apply user filters
    years = [int(x) for x in args.years.split(',') if x.strip().isdigit()] if args.years else None
    offices = [x.strip() for x in args.offices.split(',') if x.strip()] if args.offices else None
    filtered_elections = filter_elections(available_elections, years=years, offices=offices)
    print(f"Available elections: {filtered_elections}")

    print("Initializing MCMC...")
    # Create graph
    graph = create_graph(precincts)
    
    # Create updaters
    updaters_dict = create_updaters(elections=filtered_elections, election_columns=election_columns)
    
    # Create initial partition
    initial_partition = create_initial_partition(graph, precincts, updaters_dict)
    
    # Calculate ideal population
    ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)
    print(f"Ideal population per district: {ideal_population:,.0f}")
    
    # Create constraints
    constraints_list = create_constraints(
        initial_partition,
        use_cut_edges=args.use_cut_edges,
        max_muni_splits=args.max_muni_splits,
        max_county_splits=args.max_county_splits)
    
    # Create proposal
    print(f"Using region surcharges:")
    print(f"  Municipality: {args.muni_surcharge}")
    print(f"  County: {args.county_surcharge}")
    print(f"  Higher Ed COI: {args.highered_surcharge}")
    print(f"  Metro/Micro COI: {args.metro_surcharge}")
    print(f"  School District COI: {args.schdist_surcharge}")
    print(f"  Hydrologic Basin COI: {args.basin_surcharge}")
    print(f"  Water Planning Area COI: {args.water_surcharge}")
    print(f"  Cut edges constraint: {'enabled' if args.use_cut_edges else 'disabled'}")
    
    proposal = create_proposal(
        ideal_population, precincts, 
        muni_surcharge=args.muni_surcharge,
        county_surcharge=args.county_surcharge,
        highered_surcharge=args.highered_surcharge,
        metro_surcharge=args.metro_surcharge,
        schdist_surcharge=args.schdist_surcharge,
        basin_surcharge=args.basin_surcharge,
        water_surcharge=args.water_surcharge
    )
    
    # Run optimization first to get a better starting partition
    print("\n" + "="*60)
    print("RUNNING OPTIMIZATION PHASE")
    print("="*60)
    optimized_partition = run_optimization(
        initial_partition, 
        proposal, 
        muni_surcharge=args.muni_surcharge,
        county_surcharge=args.county_surcharge,
        optimization_steps=args.optimization_steps,
        optimization_probability=args.optimization_probability,
        split_munis_tolerance=args.max_muni_splits,
        split_counties_tolerance=args.max_county_splits
    )
    
    print("\n" + "="*60)
    print("RUNNING ENSEMBLE ANALYSIS")
    print("="*60)
    
    # Check if optimized partition meets ensemble constraints; fall back if not
    def partition_satisfies_all(partition, constraints_to_check):
        for c in constraints_to_check:
            try:
                if not c(partition):
                    return False
            except Exception:
                # If a constraint errors, treat as failing
                return False
        return True

    if partition_satisfies_all(optimized_partition, constraints_list):
        print("Using optimized partition as starting point for ensemble")
        ensemble_start_partition = optimized_partition
        active_constraints = constraints_list
    elif partition_satisfies_all(initial_partition, constraints_list):
        print("Warning: Optimized partition does not meet ensemble constraints, using original partition")
        ensemble_start_partition = initial_partition
        active_constraints = constraints_list
    else:
        print("Warning: Neither optimized nor initial partition meets all constraints. Filtering to satisfied subset.")
        filtered_constraints = []
        for c in constraints_list:
            try:
                if c(initial_partition):
                    filtered_constraints.append(c)
            except Exception:
                # Drop constraints that error
                pass
        ensemble_start_partition = initial_partition
        active_constraints = filtered_constraints

    # Run ensemble using the appropriate starting partition
    results = run_ensemble(
        ensemble_start_partition,
        proposal,
        active_constraints,
        filtered_elections,
        counties=counties,
        municipalities=municipalities,
        num_steps=args.steps,
        visualize_every=args.viz_every,
        vote_share_agg=args.vote_share_agg,
    )
    
    # Save results
    save_results(results, filtered_elections)
    print("Ensemble analysis complete!")
    
    # Create summary plots
    summary_df = pd.read_csv("results/ensemble_summary.csv")
    create_summary_plots(summary_df)

if __name__ == "__main__":
    main()