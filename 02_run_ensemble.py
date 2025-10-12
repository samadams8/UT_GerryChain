"""Thin CLI for running Utah redistricting ensemble analysis via utgc modules.

Uses EnsembleRunner for unified orchestration.
"""

import argparse
import os
import sys
import yaml

import warnings
import pandas as pd
warnings.filterwarnings('ignore')

from utgc.ensemble import EnsembleRunner


def main():
    print("Starting Utah redistricting ensemble analysis...")
    parser = argparse.ArgumentParser(description="Run Utah redistricting ensemble analysis")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file. If omitted, uses the latest in results/configurations/")
    args = parser.parse_args()

    # Build defaults dict to apply precedence: CLI > YAML > defaults
    defaults = {
        "initialization": {
            "nodes_data": "data/UT_precincts.geojson",
            "initial_partition": "plans/CONG/2025_UT-C/2025_UT-C.shp",
            "random_seed": 1847,
        },
        "constraints": {
            "pop_deviation": 0.001,
            "split_munis_constraint": 4,
            "split_counties_constraint": 4,
            "muni_multi_splits_constraint": 0,
            "county_multi_splits_constraint": 1,
        },
        "region_surcharges": {
            "muni": 3,
            "county": 2,
            "highered": 1,
            "metro": 1,
            "school_district": 0.1,
            "water_region": 0.1,
            "basin": 0.1,
        },
        "tilted_run": {
            "less_compact_probability": 0.5,
        },
        "ensemble": {
            "steps": 51,
            "visualize_every": 1,
        },
        "preconditioning": {
            "enable": True,
            "steps": 20,
            "max_repeats": 10,
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

    # Determine output directory for this run
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

    # Use EnsembleRunner for simplified orchestration
    print("Initializing ensemble runner...")
    
    # Check for transitability_graph parameter in config
    transitability_graph = merged.get('transitability', {}).get('precomputed_path', None)
    
    runner = EnsembleRunner(merged, transitability_graph=transitability_graph)
    
    print("Running ensemble analysis...")
    results = runner.run(output_dir=out_dir, save_config=True)
    
    print("Ensemble analysis complete!")
    print(f"Generated {len(results)} maps")
    print(f"Results saved to: {out_dir}")

if __name__ == "__main__":
    main()