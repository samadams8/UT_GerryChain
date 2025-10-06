"""
Run an ensemble of plans in accordance with Utah's redistricting requirements.

Allow different input data to be used, but start by using the UT_precincts file
and using the 2021 Utah Congressional District plan as the initial partition

Political data can not be used to help draw lines, only to evaluate whether a plan is fair after it is drawn.

Neutral redistricting standards, in priority order:
1. Adhering to the Constitution of the United States and federal laws, such as the Voting Rights Act, 52 U.S.C. Secs. 10101 through 10702, including, to the extent required, achieving equal population among districts using the most recent national decennial enumeration made by the authority of the United States; [No more than 0.1% population deviation from the ideal is permitted]
2. Minimizing the division of municipalities and counties across multiple districts, giving first priority to minimizing the division of municipalities and second priority to minimizing the division of counties; [Use the municipal and county region assignments as a surcharge on region splitting; after each iteration, count how many cities and counties are split across districts]
3. creating districts that are geographically compact; [Do not apply]
4. creating districts that are contiguous and that allow for the ease of transportation throughout the district; [No data; unguided]
5. preserving traditional neighborhoods and local communities of interest; [Use the COI data for higher ed, metro/micro statistical areas, and school districts and surcharges]
6. following natural and geographic features, boundaries, and barriers; and [Aligns well with county lines; no additional work]
7. maximizing boundary agreement among different types of districts. [No additional work]

Things to measure after each iteration:
- Number of city splits
- Number of county splits
- Compactness of districts
- Partisan metrics, based on the average partisan preference for each election year with data
    - Partisan bias
    - Mean median difference
    - Efficiency gap
    - Number of seats each party wins
    - R/D margins in each district
All data should be saved to results/ directory. Also save out a .png of each districting plan with the number of city and county splits labeled on the map.
"""

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
    initial_plan_path = "plans/CONG/ut_cong_2021/ut_cong_2021.shp"
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

def count_municipality_splits(partition):
    """Count number of municipalities split across districts."""
    # Get municipality assignments from the graph nodes
    muni_districts = {}
    
    for node in partition.graph.nodes:
        node_data = partition.graph.nodes[node]
        if 'MUNIID' in node_data and node_data['MUNIID']:
            muni_id = node_data['MUNIID']
            district = partition.assignment[node]
            
            if muni_id not in muni_districts:
                muni_districts[muni_id] = set()
            muni_districts[muni_id].add(district)
    
    # Count splits: number of districts each municipality appears in minus 1
    splits = 0
    for muni_id, districts in muni_districts.items():
        if len(districts) > 1:
            splits += len(districts) - 1
    
    return splits

def count_county_splits(partition):
    """Count number of counties split across districts."""
    # Get county assignments from the graph nodes
    county_districts = {}
    
    for node in partition.graph.nodes:
        node_data = partition.graph.nodes[node]
        if 'COUNTYID' in node_data and node_data['COUNTYID']:
            county_id = node_data['COUNTYID']
            district = partition.assignment[node]
            
            if county_id not in county_districts:
                county_districts[county_id] = set()
            county_districts[county_id].add(district)
    
    # Count splits: number of districts each county appears in minus 1
    splits = 0
    for county_id, districts in county_districts.items():
        if len(districts) > 1:
            splits += len(districts) - 1
    
    return splits

def count_coi_splits(partition):
    """Count number of Communities of Interest split across districts."""
    # Get COI assignments from the graph nodes
    coi_districts = {}
    
    # Check for different COI types
    coi_columns = ['HIGHERED_ID', 'METRO_ID', 'SCHDIST_ID']
    
    for node in partition.graph.nodes:
        node_data = partition.graph.nodes[node]
        district = partition.assignment[node]
        
        for coi_col in coi_columns:
            if coi_col in node_data and node_data[coi_col]:
                coi_id = f"{coi_col}_{node_data[coi_col]}"
                
                if coi_id not in coi_districts:
                    coi_districts[coi_id] = set()
                coi_districts[coi_id].add(district)
    
    # Count splits: number of districts each COI appears in minus 1
    total_splits = 0
    for coi_id, districts in coi_districts.items():
        if len(districts) > 1:
            total_splits += len(districts) - 1
    
    return total_splits

def create_updaters(elections=[], election_columns=[]):
    """Create updaters for the ensemble analysis."""
    print("Creating updaters...")
    
    updaters_dict = {
        "population": updaters.Tally("TOTPOP", alias="population"),
        "cut_edges": updaters.cut_edges,
        "perimeter": updaters.perimeter,
        "area": updaters.Tally("area", alias="area"),
        # Custom split counting methods (for comparison)
        "muni_splits_custom": count_municipality_splits,
        "county_splits_custom": count_county_splits,
        "coi_splits": count_coi_splits,
        # Locality split scores for counties and municipalities
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
                print(f"Added election updater for {election}: {dem_col} vs {rep_col}")
    
    return updaters_dict

def create_constraints(initial_partition):
    """Create constraints according to Utah redistricting requirements."""
    print("Creating constraints...")
    
    # Population constraint: no more than 0.1% deviation
    population_constraint = constraints.within_percent_of_ideal_population(
        initial_partition, 0.001
    )
    
    # Contiguity constraint
    contiguity_constraint = contiguous
    
    return [population_constraint, contiguity_constraint]

def create_proposal(ideal_population, precincts):
    """Create ReCom proposal with region surcharges."""
    print("Creating ReCom proposal...")
    
    # Region surcharges for municipalities and counties
    region_surcharge = {}
    
    # Add municipality surcharge if MUNIID column exists
    if "MUNIID" in precincts.columns:
        region_surcharge["MUNIID"] = 5  # Higher priority for municipalities
    
    # Add county surcharge if COUNTYID column exists  
    if "COUNTYID" in precincts.columns:
        region_surcharge["COUNTYID"] = 4  # Second priority for counties
    
    # Add COI surcharges
    if "HIGHERED_ID" in precincts.columns:
        region_surcharge["HIGHERED_ID"] = 1
    if "METRO_ID" in precincts.columns:
        region_surcharge["METRO_ID"] = 0.5
    if "SCHDIST_ID" in precincts.columns:
        region_surcharge["SCHDIST_ID"] = 0.5
    
    proposal = partial(
        recom,
        pop_col="TOTPOP",
        pop_target=ideal_population,
        epsilon=0.001,
        node_repeats=2,
        region_surcharge=region_surcharge
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

def run_ensemble(initial_partition, proposal, constraints_list, available_elections, num_steps=5000, visualize_every=10, vote_share_agg="median"):
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
            # Custom split counting methods
            "muni_splits_custom": partition["muni_splits_custom"],
            "county_splits_custom": partition["county_splits_custom"],
            "coi_splits": partition["coi_splits"],
            # (removed) GerryChain direct split counters; using custom + locality splits
        }
        
        # Build county split info (IDs, names, n-1 parts) using assignment
        county_to_districts = {}
        for node in partition.graph.nodes:
            node_data = partition.graph.nodes[node]
            county_id = node_data.get("COUNTYID")
            if county_id:
                dist = partition.assignment[node]
                if county_id not in county_to_districts:
                    county_to_districts[county_id] = set()
                county_to_districts[county_id].add(dist)
        # Build compact county split info for counts only
        county_splits_info = {cid: (len(dists) > 1, list(dists)) for cid, dists in county_to_districts.items()}

        # Add list of split county IDs from GerryChain tracker
        step_results["split_counties"] = sorted([cid for cid, (is_split, seen) in county_splits_info.items() if is_split])

        # Build ID->name maps (one pass over nodes)
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

        # County splits using gerrychain tracker (counts and (n-1) parts total)
        split_counties = step_results["split_counties"]
        counties_split_count = len(split_counties)
        counties_parts_minus_one_total = sum(len(seen) - 1 for (is_split, seen) in county_splits_info.values() if is_split)
        step_results["counties_split_count"] = counties_split_count
        step_results["counties_parts_minus_one_total"] = int(counties_parts_minus_one_total)
        step_results["split_counties_names"] = sorted([county_id_to_name.get(cid, str(cid)) for cid in split_counties])

        # Municipality splits using node attributes (counts and (n-1) parts total)
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
        munis_split_count = len(split_munis)
        munis_parts_minus_one_total = sum(len(dists) - 1 for m, dists in muni_to_districts.items() if len(dists) > 1)
        step_results["split_munis"] = split_munis
        step_results["munis_split_count"] = munis_split_count
        step_results["munis_parts_minus_one_total"] = int(munis_parts_minus_one_total)
        step_results["split_munis_names"] = sorted([muni_id_to_name.get(mid, str(mid)) for mid in split_munis])

        # Track LocalitySplits scores
        # Each LocalitySplits updater stores computed scores on access
        try:
            county_ls = partition["county_locality_splits"]
            step_results["county_ls_num_split_localities"] = county_ls.get("num_split_localities")
            step_results["county_ls_num_parts"] = county_ls.get("num_parts")
        except Exception:
            step_results["county_ls_num_split_localities"] = None
            step_results["county_ls_num_parts"] = None
        try:
            muni_ls = partition["muni_locality_splits"]
            step_results["muni_ls_num_split_localities"] = muni_ls.get("num_split_localities")
            step_results["muni_ls_num_parts"] = muni_ls.get("num_parts")
        except Exception:
            step_results["muni_ls_num_split_localities"] = None
            step_results["muni_ls_num_parts"] = None
        
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
                for i in range(n):
                    r = rep_votes[i] or 0
                    d = dem_votes[i] or 0
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
                    for i in range(n):
                        r = rep_votes[i] or 0
                        d = dem_votes[i] or 0
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
                    for i in range(n):
                        vals = [row[i] for row in rep_shares_matrix if row[i] is not None]
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
                            step_results["agg_mean_median"] = float(mean_share - median_share)
                            # Partisan bias: fraction of districts above mean minus 0.5
                            above_mean = sum(1 for v in valid_shares if v > mean_share)
                            step_results["agg_partisan_bias"] = float(above_mean / len(valid_shares) - 0.5)
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
                            step_results["agg_efficiency_gap"] = float((wasted_D - wasted_R) / len(valid_shares))
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
        
        # Save visualization every 10 steps (since we're running fewer steps)
        if i % visualize_every == 0:
            save_visualization(partition, i, step_results)
    
    return results

def save_visualization(partition, step, results):
    """Save visualization of the partition."""
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Load county boundaries for overlay
    county_path = "data/cois/UtahCountyBoundaries/ut_cnty_2020_bound.shp"
    counties = None
    if os.path.exists(county_path):
        print(f"Loading county boundaries from {county_path}...")
        counties = gpd.read_file(county_path)
        # Get CRS from the first node's geometry
        first_node = list(partition.graph.nodes)[0]
        node_geometry = partition.graph.nodes[first_node]["geometry"]
        if hasattr(node_geometry, 'crs'):
            counties = counties.to_crs(node_geometry.crs)
    else:
        print(f"Warning: {county_path} not found.")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Plot partition
    partition.plot(ax=ax, cmap='tab20c')
    
    # Add county boundaries if available
    if counties is not None:
        counties.plot(ax=ax, color='black', linewidth=1, alpha=0.5)
    
    # Add title with metrics
    title = f"Step {step}: Muni Splits: {results['muni_splits_custom']}, County Splits: {results['county_splits_custom']}"
    ax.set_title(title, fontsize=14)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save figure
    plt.savefig(f"results/step_{step:04d}.png", dpi=150, bbox_inches='tight')
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
            # Custom split counting methods
            "muni_splits_custom": result["muni_splits_custom"],
            "county_splits_custom": result["county_splits_custom"],
            "coi_splits": result["coi_splits"],
            # LocalitySplits-driven counts and (n-1) totals
            "counties_split_count": result.get("counties_split_count", 0),
            "counties_parts_minus_one_total": result.get("counties_parts_minus_one_total", 0),
            "munis_split_count": result.get("munis_split_count", 0),
            "munis_parts_minus_one_total": result.get("munis_parts_minus_one_total", 0),
            # LocalitySplits scores
            "county_ls_num_split_localities": result.get("county_ls_num_split_localities"),
            "county_ls_num_parts": result.get("county_ls_num_parts"),
            "muni_ls_num_split_localities": result.get("muni_ls_num_split_localities"),
            "muni_ls_num_parts": result.get("muni_ls_num_parts"),
        }
        
        # Add aggregated partisan metrics when present
        for metric_key in ["agg_mean_median", "agg_partisan_bias", "agg_efficiency_gap"]:
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

        # Add compactness metrics
        # for metric in ["polsby_popper"]:
        #     if metric in result:
        #         summary_row[metric] = result[metric]
        
        summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)

    # Reorder columns to ensure district aggregated share columns come last
    district_cols = [c for c in summary_df.columns if c.startswith("Republican_agg_share_d")]
    non_district_cols = [c for c in summary_df.columns if c not in district_cols]
    summary_df = summary_df[non_district_cols + district_cols]
    summary_df.to_csv("results/ensemble_summary.csv", index=False)
    
    print(f"Results saved to results/ directory")
    print(f"Summary statistics:")
    print(f"  Average municipality splits (custom): {summary_df['muni_splits_custom'].mean():.2f}")
    print(f"  Average county splits (custom): {summary_df['county_splits_custom'].mean():.2f}")
    print(f"  Average COI splits: {summary_df['coi_splits'].mean():.2f}")
    print(f"  Counties split (avg count): {summary_df['counties_split_count'].mean():.2f}")
    print(f"  Counties parts minus one (avg total): {summary_df['counties_parts_minus_one_total'].mean():.2f}")
    print(f"  Munis split (avg count): {summary_df['munis_split_count'].mean():.2f}")
    print(f"  Munis parts minus one (avg total): {summary_df['munis_parts_minus_one_total'].mean():.2f}")
    
    # Comparison block removed (no GerryChain split counters retained)
    
    # Print election summary (Republican-focused). Skip per-election printing if aggregation used
    if "Republican_agg_share_mean" not in summary_df.columns:
        for election in available_elections:
            rep_col = f"{election}_Republican_total"
            if rep_col in summary_df.columns:
                print(f"  {election} - Average Republican votes: {summary_df[rep_col].mean():.0f}")
    else:
        # Print aggregated partisan metrics
        if "Republican_agg_seats" in summary_df.columns:
            print(f"  Aggregated Republican seats (avg): {summary_df['Republican_agg_seats'].mean():.2f}")
        if "agg_mean_median" in summary_df.columns:
            print(f"  Aggregated mean-median: {summary_df['agg_mean_median'].mean():.3f}")
        if "agg_partisan_bias" in summary_df.columns:
            print(f"  Aggregated partisan bias: {summary_df['agg_partisan_bias'].mean():.3f}")
        if "agg_efficiency_gap" in summary_df.columns:
            print(f"  Aggregated efficiency gap: {summary_df['agg_efficiency_gap'].mean():.3f}")

def main():
    """Main function to run the ensemble analysis."""
    print("Starting Utah redistricting ensemble analysis...")
    parser = argparse.ArgumentParser(description="Run Utah redistricting ensemble analysis")
    parser.add_argument("--years", type=str, default="2016,2020,2024", help="Comma-separated list of years to include, e.g., 2016,2020")
    parser.add_argument("--offices", type=str, default="PRE,GOV,ATG,AUD,TRE", help="Comma-separated list of offices to include, e.g., PRE,GOV,ATG,AUD,TRE,USS")
    parser.add_argument("--vote-share-agg", type=str, choices=["median", "mean", "none"], default="median", help="Aggregate party vote share across selected elections")
    parser.add_argument("--steps", type=int, default=20, help="Number of ensemble steps to run")
    parser.add_argument("--viz-every", type=int, default=5, help="Save visualization every N steps")
    args = parser.parse_args()
    
    # Load data
    precincts, initial_plan = load_data()
    
    # Detect available election data
    available_elections = detect_election_data(precincts)
    election_columns = get_election_columns(precincts)
    # Apply user filters
    years = [int(x) for x in args.years.split(',') if x.strip().isdigit()] if args.years else None
    offices = [x.strip() for x in args.offices.split(',') if x.strip()] if args.offices else None
    filtered_elections = filter_elections(available_elections, years=years, offices=offices)
    print(f"Available elections: {filtered_elections}")
    print(f"Found {len(election_columns)} election columns")
    
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
    constraints_list = create_constraints(initial_partition)
    
    # Create proposal
    proposal = create_proposal(ideal_population, precincts)
    
    # Run ensemble
    results = run_ensemble(initial_partition, proposal, constraints_list, filtered_elections, num_steps=args.steps, visualize_every=args.viz_every, vote_share_agg=args.vote_share_agg)
    
    # Save results
    save_results(results, filtered_elections)
    
    print("Ensemble analysis complete!")

if __name__ == "__main__":
    main()