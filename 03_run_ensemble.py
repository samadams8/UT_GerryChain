"""Thin CLI for running Utah redistricting ensemble analysis via utgc modules."""

import argparse
import os

import warnings
import pandas as pd
warnings.filterwarnings('ignore')

from utgc.data_io import (
    load_data,
    load_county_boundaries,
    load_municipality_boundaries,
    detect_election_data,
    get_election_columns,
    filter_elections,
)
from utgc.build import (
    create_graph,
    create_updaters,
    create_initial_partition,
    create_constraints,
    create_proposal,
)
from utgc.optimization import run_optimization
from utgc.ensemble import run_ensemble
from utgc.reporting import (
    save_visualization,
    save_results,
    create_summary_plots,
)


def main():
    print("Starting Utah redistricting ensemble analysis...")
    parser = argparse.ArgumentParser(description="Run Utah redistricting ensemble analysis")
    parser.add_argument("--years", type=str, default="2016,2020,2024", help="Comma-separated list of years to include, e.g., 2016,2020")
    parser.add_argument("--offices", type=str, default="PRE,GOV,ATG,AUD,TRE", help="Comma-separated list of offices to include, e.g., PRE,GOV,ATG,AUD,TRE,USS")
    parser.add_argument("--vote-share-agg", type=str, choices=["median", "mean", "none"], default="median", help="Aggregate party vote share across selected elections")
    parser.add_argument("--steps", type=int, default=21, help="Number of ensemble steps to run")
    parser.add_argument("--optim-steps", type=int, default=20, help="Number of optimization steps to run before ensemble")
    parser.add_argument("--max-muni-splits", type=int, default=None, help="Maximum number of municipality splits allowed (None for no constraint)")
    parser.add_argument("--max-county-splits", type=int, default=None, help="Maximum number of county splits allowed (None for no constraint)")
    parser.add_argument("--viz-every", type=int, default=5, help="Save visualization every N steps")
    # Region surcharge arguments
    parser.add_argument("--muni-surcharge", type=float, default=9, help="Municipality region surcharge (0 to disable)")
    parser.add_argument("--county-surcharge", type=float, default=3, help="County region surcharge (0 to disable)")
    parser.add_argument("--highered-surcharge", type=float, default=1, help="Higher education COI surcharge (0 to disable)")
    parser.add_argument("--metro-surcharge", type=float, default=1, help="Metro/micro statistical area COI surcharge (0 to disable)")
    parser.add_argument("--schdist-surcharge", type=float, default=0.1, help="School district COI surcharge (0 to disable)")
    parser.add_argument("--water-surcharge", type=float, default=0.1, help="Water planning area surcharge (0 to disable)")
    parser.add_argument("--basin-surcharge", type=float, default=0.1, help="Hydrologic basin surcharge (0 to disable)")
    # Compactness argument
    parser.add_argument("--use-cut-edges", action="store_true", help="Enable compactness constraint via cut edges minimization")
    args = parser.parse_args()

    precincts, initial_plan = load_data()
    print("Loading county boundaries...")
    counties = load_county_boundaries(precincts)
    print("Loading municipality boundaries...")
    municipalities = load_municipality_boundaries(precincts)

    available_elections = detect_election_data(precincts)
    election_columns = get_election_columns(precincts)
    years = [int(x) for x in args.years.split(',') if x.strip().isdigit()] if args.years else None
    offices = [x.strip() for x in args.offices.split(',') if x.strip()] if args.offices else None
    filtered_elections = filter_elections(available_elections, years=years, offices=offices)
    print(f"Available elections: {filtered_elections}")

    print("Initializing MCMC...")
    graph = create_graph(precincts)
    updaters_dict = create_updaters(elections=filtered_elections, election_columns=election_columns)
    initial_partition = create_initial_partition(graph, precincts, updaters_dict)
    ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)
    print(f"Ideal population per district: {ideal_population:,.0f}")

    constraints_list = create_constraints(
        initial_partition,
        use_cut_edges=args.use_cut_edges,
        max_muni_splits=args.max_muni_splits,
        max_county_splits=args.max_county_splits,
    )

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
        ideal_population,
        precincts,
        muni_surcharge=args.muni_surcharge,
        county_surcharge=args.county_surcharge,
        highered_surcharge=args.highered_surcharge,
        metro_surcharge=args.metro_surcharge,
        schdist_surcharge=args.schdist_surcharge,
        basin_surcharge=args.basin_surcharge,
        water_surcharge=args.water_surcharge,
    )

    print("\n" + "=" * 60)
    print("RUNNING OPTIMIZATION PHASE")
    print("=" * 60)
    optimized_partition = run_optimization(
        initial_partition,
        proposal,
        muni_surcharge=args.muni_surcharge,
        county_surcharge=args.county_surcharge,
        optimization_steps=args.optim_steps,
        split_munis_tolerance=args.max_muni_splits,
        split_counties_tolerance=args.max_county_splits,
    )

    print("\n" + "=" * 60)
    print("RUNNING ENSEMBLE ANALYSIS")
    print("=" * 60)

    def partition_satisfies_all(partition, constraints_to_check):
        for c in constraints_to_check:
            try:
                if not c(partition):
                    return False
            except Exception:
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
                pass
        ensemble_start_partition = initial_partition
        active_constraints = filtered_constraints

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
        save_visualization_fn=save_visualization,
    )

    save_results(results, filtered_elections)
    print("Ensemble analysis complete!")
    summary_df = pd.read_csv("results/ensemble_summary.csv")
    create_summary_plots(summary_df)

if __name__ == "__main__":
    main()