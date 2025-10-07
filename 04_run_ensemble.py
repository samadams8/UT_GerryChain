"""Thin CLI for running Utah redistricting ensemble analysis via utgc modules.

Transitional config behavior:
- Prefer YAML via --config; CLI flags still accepted but deprecated when used without YAML.
- Precedence: CLI flags > YAML values > hardcoded defaults.
"""

import argparse
import os
import sys
import yaml

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
from utgc.ensemble import run_ensemble, run_ensemble_tilted
from utgc.reporting import (
    save_visualization,
    save_results,
    create_summary_plots,
)


def main():
    print("Starting Utah redistricting ensemble analysis...")
    parser = argparse.ArgumentParser(description="Run Utah redistricting ensemble analysis")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file (preferred)")
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
    # Compactness argument and tilted-run control
    parser.add_argument("--use-cut-edges", action="store_true", help="Enable compactness constraint via cut edges minimization")
    parser.add_argument(
        "--tilted-run",
        type=float,
        default=0.5,
        help="Tilt intensity in [0,1]: 0=neutral sampler, 1=maximally tilted (optimizer p=1-value)",
    )
    args = parser.parse_args()

    # Build defaults dict to apply precedence: CLI > YAML > defaults
    defaults = {
        "years": "2016,2020,2024",
        "offices": "PRE,GOV,ATG,AUD,TRE",
        "vote_share_agg": "median",
        "steps": 21,
        "optim_steps": 20,
        "max_muni_splits": None,
        "max_county_splits": None,
        "viz_every": 5,
        "muni_surcharge": 9.0,
        "county_surcharge": 3.0,
        "highered_surcharge": 1.0,
        "metro_surcharge": 1.0,
        "schdist_surcharge": 0.1,
        "water_surcharge": 0.1,
        "basin_surcharge": 0.1,
        "use_cut_edges": False,
        "tilted_run": 0.5,
    }

    yaml_config = {}
    if args.config is not None:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        with open(args.config, "r") as f:
            try:
                loaded = yaml.safe_load(f) or {}
                if not isinstance(loaded, dict):
                    raise ValueError("YAML config must be a mapping of keys to values")
                yaml_config = loaded
            except Exception as e:
                raise RuntimeError(f"Failed to load YAML config: {e}")
    else:
        # Deprecation warning when running without YAML
        print("[DEPRECATION] Running without --config YAML is deprecated and will be removed in a future release. Please provide a YAML config.")

    # Determine which CLI flags were explicitly provided (present in sys.argv)
    argv = sys.argv[1:]
    cli_present = {
        "years": any(arg.startswith("--years") for arg in argv),
        "offices": any(arg.startswith("--offices") for arg in argv),
        "vote_share_agg": any(arg.startswith("--vote-share-agg") for arg in argv),
        "steps": any(arg.startswith("--steps") for arg in argv),
        "optim_steps": any(arg.startswith("--optim-steps") for arg in argv),
        "max_muni_splits": any(arg.startswith("--max-muni-splits") for arg in argv),
        "max_county_splits": any(arg.startswith("--max-county-splits") for arg in argv),
        "viz_every": any(arg.startswith("--viz-every") for arg in argv),
        "muni_surcharge": any(arg.startswith("--muni-surcharge") for arg in argv),
        "county_surcharge": any(arg.startswith("--county-surcharge") for arg in argv),
        "highered_surcharge": any(arg.startswith("--highered-surcharge") for arg in argv),
        "metro_surcharge": any(arg.startswith("--metro-surcharge") for arg in argv),
        "schdist_surcharge": any(arg.startswith("--schdist-surcharge") for arg in argv),
        "water_surcharge": any(arg.startswith("--water-surcharge") for arg in argv),
        "basin_surcharge": any(arg.startswith("--basin-surcharge") for arg in argv),
        "use_cut_edges": "--use-cut-edges" in argv,
        "tilted_run": any(arg.startswith("--tilted-run") for arg in argv),
    }

    # Merge values
    merged = dict(defaults)
    merged.update({k: v for k, v in yaml_config.items() if k in merged})
    # Overlay CLI values only if explicitly provided
    for key in merged.keys():
        if key == "use_cut_edges":
            if cli_present[key]:
                merged[key] = bool(getattr(args, key))
        else:
            if cli_present.get(key, False):
                merged[key] = getattr(args, key)

    # If both YAML and flags provided, log overrides
    if args.config is not None and any(cli_present.values()):
        print("Using --config and CLI flags; applying precedence (flags override YAML). Overrides:")
        for k, present in cli_present.items():
            if present and k in yaml_config and yaml_config.get(k) != merged.get(k):
                print(f"  {k}: YAML={yaml_config.get(k)} -> CLI={merged.get(k)}")

    precincts, initial_plan = load_data()
    print("Loading county boundaries...")
    counties = load_county_boundaries(precincts)
    print("Loading municipality boundaries...")
    municipalities = load_municipality_boundaries(precincts)

    available_elections = detect_election_data(precincts)
    election_columns = get_election_columns(precincts)
    years = [int(x) for x in str(merged["years"]).split(',') if x.strip().isdigit()] if merged.get("years") else None
    offices = [x.strip() for x in str(merged["offices"]).split(',') if x.strip()] if merged.get("offices") else None
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
        use_cut_edges=bool(merged["use_cut_edges"]),
        max_muni_splits=merged["max_muni_splits"],
        max_county_splits=merged["max_county_splits"],
    )

    print(f"Using region surcharges:")
    print(f"  Municipality: {merged['muni_surcharge']}")
    print(f"  County: {merged['county_surcharge']}")
    print(f"  Higher Ed COI: {merged['highered_surcharge']}")
    print(f"  Metro/Micro COI: {merged['metro_surcharge']}")
    print(f"  School District COI: {merged['schdist_surcharge']}")
    print(f"  Hydrologic Basin COI: {merged['basin_surcharge']}")
    print(f"  Water Planning Area COI: {merged['water_surcharge']}")
    print(f"  Cut edges constraint: {'enabled' if merged['use_cut_edges'] else 'disabled'}")

    proposal = create_proposal(
        ideal_population,
        precincts,
        muni_surcharge=merged["muni_surcharge"],
        county_surcharge=merged["county_surcharge"],
        highered_surcharge=merged["highered_surcharge"],
        metro_surcharge=merged["metro_surcharge"],
        schdist_surcharge=merged["schdist_surcharge"],
        basin_surcharge=merged["basin_surcharge"],
        water_surcharge=merged["water_surcharge"],
    )

    print("\n" + "=" * 60)
    print("RUNNING OPTIMIZATION PHASE")
    print("=" * 60)
    optimized_partition = run_optimization(
        initial_partition,
        proposal,
        muni_surcharge=merged["muni_surcharge"],
        county_surcharge=merged["county_surcharge"],
        optimization_steps=merged["optim_steps"],
        split_munis_tolerance=merged["max_muni_splits"],
        split_counties_tolerance=merged["max_county_splits"],
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

    # Choose sampler based on tilted-run intensity (map to optimizer p=1-value)
    tilt_intensity = max(0.0, min(1.0, float(merged["tilted_run"])) )
    if tilt_intensity <= 0.0:
        print("Using neutral sampler (tilted-run=0.0)")
        results = run_ensemble(
            ensemble_start_partition,
            proposal,
            active_constraints,
            filtered_elections,
            counties=counties,
            municipalities=municipalities,
            num_steps=int(merged["steps"]),
            visualize_every=int(merged["viz_every"]),
            vote_share_agg=str(merged["vote_share_agg"]),
            save_visualization_fn=save_visualization,
        )
    else:
        p = 1.0 - tilt_intensity
        print(f"Using tilted-run sampler with intensity={tilt_intensity} (optimizer p={p})")
        results = run_ensemble_tilted(
            ensemble_start_partition,
            proposal,
            active_constraints,
            filtered_elections,
            counties=counties,
            municipalities=municipalities,
            num_steps=int(merged["steps"]),
            visualize_every=int(merged["viz_every"]),
            vote_share_agg=str(merged["vote_share_agg"]),
            save_visualization_fn=save_visualization,
            p=p,
        )

    # Neutral mode output when there are no elections
    neutral_mode = len(filtered_elections) == 0
    save_results(results, filtered_elections, mode=("neutral" if neutral_mode else None))
    print("Ensemble analysis complete!")
    summary_df = pd.read_csv("results/ensemble_summary.csv")
    create_summary_plots(summary_df)

if __name__ == "__main__":
    main()