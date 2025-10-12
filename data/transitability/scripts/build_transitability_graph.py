#!/usr/bin/env python3
"""Wrapper CLI: build transitability-aware graph and export artifacts.

Supports two methods:
1. roads-only: Prunes by direct road contact, then falls back to admin
   boundaries and repairs county connectivity.
2. hierarchical: The original method using a fixed hierarchy of features.
"""

from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import sys
import pandas as pd
import yaml
import networkx as nx

# Ensure project root is on sys.path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utgc.transitability import build_transitable_graph, build_roads_only_graph  # type: ignore

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the script."""
    p = argparse.ArgumentParser(
        description="Build and export transitability artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--precincts", default="data/UT_precincts.geojson", help="Path to precincts GeoJSON file.")
    p.add_argument("--out-dir", default="data/transitability", help="Directory to save output artifacts.")

    # --- Mode Selection ---
    p.add_argument(
        "--method", 
        choices=['roads-only', 'hierarchical'], 
        default='roads-only',
        help="The pruning method to use."
    )
    
    # --- Shared Options ---
    p.add_argument("--roads-path", default="data/geography_processed/UtahRoads_filtered.shp", help="Path to roads shapefile (used in both modes).")
    p.add_argument("--reproject", action="store_true", help="Reproject all layers to match the CRS of the precincts file.")

    # --- Options for Hierarchical mode ONLY ---
    hier_group = p.add_argument_group("Hierarchical Method Options")
    hier_group.add_argument("--prune-roads", action="store_true", help="[Hierarchical] Enable road-based connectivity pruning.")
    hier_group.add_argument("--road-prune-level", choices=['primary', 'secondary', 'all'], default='primary', help="[Hierarchical] Road network level for initial pruning.")
    hier_group.add_argument("--road-fallback", action="store_true", help="[Hierarchical] Enable fallback to county/municipal boundaries.")
    hier_group.add_argument("--prune-water", action="store_true", help="[Hierarchical] Enable water barrier pruning.")
    hier_group.add_argument("--lakes-path", default="data/geography_processed/UtahMajorLakes_filtered.shp", help="[Hierarchical] Path to major lakes shapefile.")
    hier_group.add_argument("--rivers-path", default="data/geography_processed/UtahMajorRivers_filtered.shp", help="[Hierarchical] Path to major rivers shapefile.")
    
    return p.parse_args()


def export_artifacts(base_graph: nx.Graph, final_graph: nx.Graph, out_dir: str, precincts_path: str, params: dict, args: argparse.Namespace):
    """Exports the removed edges to a CSV and a metadata YAML file."""
    # ... (This function remains unchanged)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    base_edges = {tuple(sorted(e)) for e in base_graph.edges}
    final_edges = {tuple(sorted(e)) for e in final_graph.edges}
    removed_edges = sorted(list(base_edges - final_edges))
    
    csv_path = out_path / "removed_edges.csv"
    pd.DataFrame(removed_edges, columns=['u', 'v']).to_csv(csv_path, index=False)
    print(f"  - Saved {len(removed_edges)} removed edges to {csv_path}")

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "precincts_file": precincts_path,
        "parameters": params,
        "cli_args": vars(args)
    }
    
    yaml_path = out_path / "metadata.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(metadata, f, default_flow_style=False, sort_keys=False)
    print(f"  - Metadata saved to {yaml_path}")
        
    return str(csv_path)


def main():
    """Main execution function."""
    args = parse_args()
    
    if args.method == 'roads-only':
        print("Building transitability-aware graph (roads-only mode)...")
        params = {
            "method": "roads-only",
            "roads_path": args.roads_path,
            "reproject": args.reproject,
        }
        base_graph, final_graph = build_roads_only_graph(args.precincts, args.roads_path, args.reproject)

    elif args.method == 'hierarchical':
        print("Building transitability-aware graph (hierarchical mode)...")
        params = {
            "method": "hierarchical",
            "prune_roads": args.prune_roads,
            "roads_path": args.roads_path,
            "road_prune_level": args.road_prune_level,
            "road_fallback": args.road_fallback,
            "prune_water": args.prune_water,
            "lakes_path": args.lakes_path,
            "rivers_path": args.rivers_path,
            "reproject": args.reproject,
        }
        base_graph, final_graph = build_transitable_graph(args.precincts, **params)
    
    # --- EXPORT ARTIFACTS --
    print("\nExporting artifacts...")
    csv_output_path = export_artifacts(base_graph, final_graph, args.out_dir, args.precincts, params, args)

    print("\nArtifacts:")
    print(f"  - removed_edges_csv: {csv_output_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()
