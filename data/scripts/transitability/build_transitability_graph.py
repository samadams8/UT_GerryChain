#!/usr/bin/env python3
"""Wrapper CLI: build transitability-aware graph via utgc.transitability and export artifacts.

Outputs to data/transitability (by default):
- transitability_graph_nodes.parquet (node_id)
- transitability_graph_edges.parquet (u,v)
- transitability.graphml
- transitability.json
- metadata.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import sys

import geopandas as gpd
import yaml

import networkx as nx

# Ensure project root is on sys.path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utgc.transitability import build_transitable_graph  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build transitability artifacts (wrapper)")
    p.add_argument("--precincts", default="data/UT_precincts.geojson")
    p.add_argument("--out-dir", default="data/transitability")
    # Parameters passed to utgc.transitability.build_transitable_graph
    p.add_argument("--enable", action="store_true", help="Enable transitability modifications")
    p.add_argument("--remove-water-barriers", dest="remove_water_barriers", action="store_true", help="Prune major water crossings")
    p.add_argument("--no-remove-water-barriers", dest="remove_water_barriers", action="store_false")
    p.set_defaults(remove_water_barriers=True)
    p.add_argument("--verify-road-connectivity", dest="verify_road_connectivity", action="store_true", help="Apply road connectivity checks")
    p.add_argument("--no-verify-road-connectivity", dest="verify_road_connectivity", action="store_false")
    p.set_defaults(verify_road_connectivity=True)
    p.add_argument("--road-buffer-meters", type=float, default=500.0)
    p.add_argument("--water-threshold", type=float, default=0.5)
    p.add_argument("--export-formats", default="parquet,graphml,json")
    p.add_argument("--verbose", action="store_true", help="Print detailed progress information")
    return p.parse_args()


def write_metadata(out_dir: Path, precincts: gpd.GeoDataFrame, params: Dict, args: argparse.Namespace) -> None:
    meta = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "crs": str(precincts.crs),
        "params": params,
        "inputs": {
            "precincts": str(Path(args.precincts).resolve()),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load precincts
    precincts = gpd.read_file(args.precincts)
    if args.verbose:
        print(f"Loaded {len(precincts):,} precincts from {args.precincts}")

    # Build transitability-aware graph
    params: Dict = {
        "enable": bool(args.enable or True),
        "remove_water_barriers": bool(args.remove_water_barriers),
        "verify_road_connectivity": bool(args.verify_road_connectivity),
        "road_buffer_meters": float(args.road_buffer_meters),
        "water_threshold": float(args.water_threshold),
    }
    if args.verbose:
        print("Transitability params:")
        for k, v in params.items():
            print(f"  {k}: {v}")

    graph = build_transitable_graph(precincts, transitability_params=params)
    if args.verbose:
        print(f"Graph built: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # Export nodes/edges parquet (disabled per request)
    # nodes_path = out_dir / "transitability_graph_nodes.parquet"
    # edges_path = out_dir / "transitability_graph_edges.parquet"
    # import pandas as pd
    # pd.DataFrame({"node_id": list(graph.nodes())}).to_parquet(nodes_path)
    # pd.DataFrame([(u, v) for u, v in graph.edges()], columns=["u", "v"]).to_parquet(edges_path)

    # Export GraphML and JSON edge list
    # Sanitize graph attributes: GraphML cannot serialize shapely geometries or complex objects.
    simple_graph = nx.Graph()
    simple_graph.add_nodes_from(list(graph.nodes()))
    simple_graph.add_edges_from(list(graph.edges()))
    gml_path = out_dir / "transitability.graphml"
    nx.write_graphml(simple_graph, gml_path)
    json_path = out_dir / "transitability.json"
    import json
    with open(json_path, "w") as f:
        json.dump([{ "source": int(u), "target": int(v)} for u, v in simple_graph.edges()], f)

    # Metadata
    write_metadata(out_dir, precincts, params, args)

    print("Artifacts:")
    print(f"  - graphml: {gml_path}")
    print(f"  - json: {json_path}")
    print(f"  - metadata: {out_dir / 'metadata.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


