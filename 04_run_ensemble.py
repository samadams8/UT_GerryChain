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
from utgc.preconditioning import run_preconditioning
from utgc.ensemble import run_ensemble, run_ensemble_tilted
from utgc.reporting import (
    save_visualization,
    save_results,
    create_summary_plots,
)


def main():
    print("Starting Utah redistricting ensemble analysis...")
    parser = argparse.ArgumentParser(description="Run Utah redistricting ensemble analysis")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file. If omitted, uses the latest in results/configurations/")
    args = parser.parse_args()

    # Build defaults dict to apply precedence: CLI > YAML > defaults
    defaults = {
        "constraints": {
            "use_cut_edges": False,
            "max_muni_splits": None,
            "max_county_splits": None,
        },
        "proposal": {
            "muni_surcharge": 9.0,
            "county_surcharge": 3.0,
            "highered_surcharge": 1.0,
            "metro_surcharge": 1.0,
            "schdist_surcharge": 0.1,
            "water_surcharge": 0.1,
            "basin_surcharge": 0.1,
        },
        "preconditioning": {
            "enable": True,
            "steps": 20,
        },
        "ensemble": {
            "num_steps": 21,
            "visualize_every": 5,
            "tilted_run": 0.5,
        },
        "election": {
            "years": "2016,2020,2024",
            "offices": "PRE,GOV,ATG,AUD,TRE",
            "vote_share_agg": "median",
        }
    }

    yaml_config = {}
    # Resolve config file path (use most recent config if not provided)
    def _latest_config():
        configs_dir = os.path.join("results", "configurations")
        if not os.path.isdir(configs_dir):
            return None
        candidates = []
        for root, _, files in os.walk(configs_dir):
            for name in files:
                if name.endswith(".yaml") or name.endswith(".yml"):
                    path = os.path.join(root, name)
                    try:
                        mtime = os.path.getmtime(path)
                    except Exception:
                        mtime = 0
                    candidates.append((mtime, path))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    config_path = args.config or _latest_config()

    if config_path is not None:
        if not os.path.exists(config_path):
            if args.config is None:
                raise FileNotFoundError("No configuration files found under results/configurations/.")
            else:
                raise FileNotFoundError(f"Config file not found: {args.config}")
        if args.config is None:
            print(f"Using latest configuration: {config_path}")
        with open(config_path, "r") as f:
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

    # Merge values from YAML with nested structure support
    def deep_merge(default_dict, yaml_dict):
        """Deep merge YAML config into defaults, preserving nested structure."""
        result = dict(default_dict)
        for key, value in yaml_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    merged = deep_merge(defaults, yaml_config)

    # Determine output directory for this run, matching config tag/name BEFORE running samplers
    # Prefer the name of the params directory to match the source config
    run_tag = None
    try:
        params_dir = os.path.dirname(config_path) if config_path else None
        if params_dir:
            run_tag = os.path.basename(params_dir)
    except Exception:
        run_tag = None
    if not run_tag:
        if isinstance(yaml_config.get("tag"), str) and yaml_config["tag"].strip():
            run_tag = yaml_config["tag"].strip()
        else:
            try:
                base = os.path.splitext(os.path.basename(config_path or ""))[0]
                run_tag = base or "run"
            except Exception:
                run_tag = "run"
    out_dir = os.path.join("results", "ensembles", run_tag)
    os.makedirs(out_dir, exist_ok=True)

    precincts, initial_plan = load_data()
    print("Loading county boundaries...")
    counties = load_county_boundaries(precincts)
    print("Loading municipality boundaries...")
    municipalities = load_municipality_boundaries(precincts)

    available_elections = detect_election_data(precincts)
    election_columns = get_election_columns(precincts)
    years = [int(x) for x in str(merged["election"]["years"]).split(',') if x.strip().isdigit()] if merged.get("election", {}).get("years") else None
    offices = [x.strip() for x in str(merged["election"]["offices"]).split(',') if x.strip()] if merged.get("election", {}).get("offices") else None
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
        use_cut_edges=bool(merged["constraints"]["use_cut_edges"]),
        max_muni_splits=merged["constraints"]["max_muni_splits"],
        max_county_splits=merged["constraints"]["max_county_splits"],
    )

    print(f"Using region surcharges:")
    print(f"  Municipality: {merged['proposal']['muni_surcharge']}")
    print(f"  County: {merged['proposal']['county_surcharge']}")
    print(f"  Higher Ed COI: {merged['proposal']['highered_surcharge']}")
    print(f"  Metro/Micro COI: {merged['proposal']['metro_surcharge']}")
    print(f"  School District COI: {merged['proposal']['schdist_surcharge']}")
    print(f"  Hydrologic Basin COI: {merged['proposal']['basin_surcharge']}")
    print(f"  Water Planning Area COI: {merged['proposal']['water_surcharge']}")
    print(f"  Cut edges constraint: {'enabled' if merged['constraints']['use_cut_edges'] else 'disabled'}")

    proposal = create_proposal(
        ideal_population,
        precincts,
        muni_surcharge=merged["proposal"]["muni_surcharge"],
        county_surcharge=merged["proposal"]["county_surcharge"],
        highered_surcharge=merged["proposal"]["highered_surcharge"],
        metro_surcharge=merged["proposal"]["metro_surcharge"],
        schdist_surcharge=merged["proposal"]["schdist_surcharge"],
        basin_surcharge=merged["proposal"]["basin_surcharge"],
        water_surcharge=merged["proposal"]["water_surcharge"],
    )

    print("\n" + "=" * 60)
    print("RUNNING PRECONDITIONING PHASE")
    print("=" * 60)
    optimized_partition = run_preconditioning(
        initial_partition,
        proposal,
        muni_surcharge=merged["proposal"]["muni_surcharge"],
        county_surcharge=merged["proposal"]["county_surcharge"],
        steps=merged["preconditioning"]["steps"],
        split_munis_tolerance=merged["constraints"]["max_muni_splits"],
        split_counties_tolerance=merged["constraints"]["max_county_splits"],
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

    # Before running the sampler, persist the exact configuration used (defaults merged with YAML)
    try:
        params_out_path = os.path.join(out_dir, "params.yaml")
        effective = {k: merged.get(k) for k in merged.keys()}
        # Also persist the resolved years/offices strings for clarity
        effective["years"] = merged.get("years")
        effective["offices"] = merged.get("offices")
        with open(params_out_path, "w") as f:
            yaml.safe_dump(effective, f, sort_keys=True)
        print(f"Saved run parameters to: {params_out_path}")
    except Exception:
        pass

    # Choose sampler based on tilted-run intensity (map to optimizer p=1-value)
    tilt_intensity = max(0.0, min(1.0, float(merged["ensemble"]["tilted_run"])) )
    if tilt_intensity <= 0.0:
        print("Using neutral sampler (tilted-run=0.0)")
        results = run_ensemble(
            ensemble_start_partition,
            proposal,
            active_constraints,
            filtered_elections,
            counties=counties,
            municipalities=municipalities,
            num_steps=int(merged["ensemble"]["num_steps"]),
            visualize_every=int(merged["ensemble"]["visualize_every"]),
            vote_share_agg=str(merged["election"]["vote_share_agg"]),
            save_visualization_fn=lambda part, step, res, counties, municipalities: save_visualization(part, step, res, counties, municipalities, base_dir=out_dir),
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
            num_steps=int(merged["ensemble"]["num_steps"]),
            visualize_every=int(merged["ensemble"]["visualize_every"]),
            vote_share_agg=str(merged["election"]["vote_share_agg"]),
            save_visualization_fn=lambda part, step, res, counties, municipalities: save_visualization(part, step, res, counties, municipalities, base_dir=out_dir),
            p=p,
        )

    # Neutral mode output when there are no elections
    neutral_mode = len(filtered_elections) == 0
    save_results(results, filtered_elections, mode=("neutral" if neutral_mode else None), out_dir=out_dir)
    print("Ensemble analysis complete!")
    summary_df = pd.read_csv(os.path.join(out_dir, "ensemble_summary.csv"))
    create_summary_plots(summary_df, out_dir=out_dir)

if __name__ == "__main__":
    main()