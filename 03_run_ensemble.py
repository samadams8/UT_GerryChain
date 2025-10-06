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

# GerryTools imports for enhanced metrics
from gerrytools.scoring import (
    seats, efficiency_gap as gt_efficiency_gap, 
    mean_median as gt_mean_median, partisan_bias as gt_partisan_bias,
    partisan_gini, summarize
)

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
                available_elections.append(f"{year}_{office}")
    
    return available_elections

def get_election_columns(precincts):
    """Get all election-related columns from precincts data."""
    election_columns = []
    
    # Pattern: YYOFFICEPARTY (e.g., 16PRER, 20GOVD, 24ATGO)
    for col in precincts.columns:
        if len(col) == 6 and col[:2].isdigit() and col[2:5] in ["PRE", "GOV", "ATG", "AUD", "TRE", "USS"] and col[5] in ["R", "D", "O"]:
            election_columns.append(col)
    
    return sorted(election_columns)

def calculate_partisan_metrics(partition, available_elections):
    """Calculate partisan metrics for a partition using gerrytools."""
    metrics = {}
    
    if not available_elections:
        return metrics
    
    try:
        # Create election data for gerrytools
        elections = []
        for election in available_elections:
            year, office = election.split('_')
            dem_col = f"{year%100:02d}{office}D"
            rep_col = f"{year%100:02d}{office}R"
            
            # Check if columns exist in the graph nodes
            first_node = list(partition.graph.nodes)[0]
            if dem_col in partition.graph.nodes[first_node] and rep_col in partition.graph.nodes[first_node]:
                elections.append({
                    "name": election,
                    "dem_col": dem_col,
                    "rep_col": rep_col
                })
        
        if elections:
            # Use gerrytools scoring functions
            partisan_scores = [
                seats(elections, "Dem"),
                seats(elections, "Rep"),
                gt_efficiency_gap(elections, mean=True),
                gt_mean_median(elections),
                gt_partisan_bias(elections),
                partisan_gini(elections),
            ]
            
            # Apply scores to partition
            partisan_results = summarize(partition, partisan_scores)
            metrics.update(partisan_results)
            
    except Exception as e:
        print(f"Warning: Could not calculate partisan metrics: {e}")
    
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
    """Create initial partition from the 2021 congressional plan."""
    print("Creating initial partition...")
    
    initial_partition = GeographicPartition(
        graph,
        assignment="CONGDIST",
        updaters=updaters_dict
    )
    
    print(f"Initial partition created with {len(initial_partition)} districts")
    return initial_partition

def run_ensemble(initial_partition, proposal, constraints_list, available_elections, num_steps=5000, visualize_every=10):
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
        step_results = {
            "step": i,
            "population": dict(partition["population"]),
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
        for election in available_elections:
            if election in partition.updaters:
                election_results = partition[election]
                
                # Get vote totals for each party
                for party in ["Democratic", "Republican"]:
                    try:
                        # Get vote totals for this party across all districts
                        # Use the votes() method which returns a tuple of district totals
                        party_votes = election_results.votes(party)
                        step_results[f"{election}_{party}_total"] = sum(party_votes)
                    except Exception as e:
                        print(f"Error getting {election} {party} totals: {e}")
                        step_results[f"{election}_{party}_total"] = 0
                
                # Store wins for each party
                for party in ["Democratic", "Republican"]:
                    try:
                        step_results[f"{election}_{party}_wins"] = election_results.wins(party)
                    except:
                        step_results[f"{election}_{party}_wins"] = 0
        
        # Add compactness metrics
        # compactness = calculate_compactness_metrics(partition)
        # step_results.update(compactness)
        
        # Add partisan metrics using gerrytools
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
        
        # Add election metrics
        for election in available_elections:
            for party in ["Democratic", "Republican"]:
                col_name = f"{election}_{party}_total"
                if col_name in result:
                    summary_row[col_name] = result[col_name]
        
        # Add compactness metrics
        # for metric in ["polsby_popper"]:
        #     if metric in result:
        #         summary_row[metric] = result[metric]
        
        summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)
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
    
    # Print election summary
    for election in available_elections:
        dem_col = f"{election}_Democratic_total"
        rep_col = f"{election}_Republican_total"
        if dem_col in summary_df.columns and rep_col in summary_df.columns:
            print(f"  {election} - Average Democratic votes: {summary_df[dem_col].mean():.0f}")
            print(f"  {election} - Average Republican votes: {summary_df[rep_col].mean():.0f}")

def main():
    """Main function to run the ensemble analysis."""
    print("Starting Utah redistricting ensemble analysis...")
    
    # Load data
    precincts, initial_plan = load_data()
    
    # Detect available election data
    available_elections = detect_election_data(precincts)
    election_columns = get_election_columns(precincts)
    print(f"Available elections: {available_elections}")
    print(f"Found {len(election_columns)} election columns")
    
    # Create graph
    graph = create_graph(precincts)
    
    # Create updaters
    updaters_dict = create_updaters(elections=available_elections, election_columns=election_columns)
    
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
    results = run_ensemble(initial_partition, proposal, constraints_list, available_elections, num_steps=20, visualize_every=5)
    
    # Save results
    save_results(results, available_elections)
    
    print("Ensemble analysis complete!")

if __name__ == "__main__":
    main()