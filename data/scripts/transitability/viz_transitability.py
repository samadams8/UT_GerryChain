#!/usr/bin/env python3
"""Visualize transitability artifacts (JSON edges) in 4 panels:

1. Base adjacency (from precinct adjacency)
2. Final (from transitability.json edge list)
3. Removed (Base minus Final)
4. Overlay: precinct boundaries, rivers/lakes, and final graph
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import json
from gerrychain import Graph

# Ensure project root on path for absolute imports if needed
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize transitability artifacts")
    p.add_argument("--precincts", default="data/UT_precincts.geojson")
    p.add_argument("--artifacts-dir", default="data/transitability")
    p.add_argument("--lakes", default="data/geography_processed/UtahMajorLakes_filtered.shp")
    p.add_argument("--rivers", default="data/geography_processed/UtahMajorRivers_filtered.shp")
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def plot_edges(precincts: gpd.GeoDataFrame, edges: pd.DataFrame, ax, color: str, alpha: float, lw: float):
    for _, r in edges.iterrows():
        try:
            p1 = precincts.loc[int(r["u"]), "geometry"].centroid
            p2 = precincts.loc[int(r["v"]), "geometry"].centroid
            ax.plot([p1.x, p2.x], [p1.y, p2.y], color=color, alpha=alpha, linewidth=lw)
        except Exception:
            continue


def main() -> int:
    args = parse_args()
    art = Path(args.artifacts_dir)

    precincts = gpd.read_file(args.precincts)
    # Build base adjacency from precincts
    base_graph = Graph.from_geodataframe(precincts)
    edges_base = pd.DataFrame([(int(u), int(v)) for u, v in base_graph.edges()], columns=["u", "v"])

    # Load final edges from JSON
    json_path = art / "transitability.json"
    with open(json_path, "r") as f:
        edgelist = json.load(f)
    edges_final = pd.DataFrame([(int(e["source"]), int(e["target"])) for e in edgelist], columns=["u", "v"])

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    ax = axes[0][0]
    precincts.plot(ax=ax, color="white", edgecolor="black", linewidth=0.2)
    plot_edges(precincts, edges_base, ax, color="tab:blue", alpha=0.6, lw=0.6)
    ax.set_aspect("equal")
    ax.set_title("Base Adjacency")

    ax = axes[0][1]
    precincts.plot(ax=ax, color="white", edgecolor="black", linewidth=0.2)
    plot_edges(precincts, edges_final, ax, color="tab:red", alpha=0.6, lw=0.6)
    ax.set_aspect("equal")
    ax.set_title("Final")

    ax = axes[1][0]
    precincts.plot(ax=ax, color="white", edgecolor="black", linewidth=0.2)
    base_set = set((int(u), int(v)) for u, v in zip(edges_base["u"], edges_base["v"]))
    final_set = set((int(u), int(v)) for u, v in zip(edges_final["u"], edges_final["v"]))
    removed = pd.DataFrame(list(base_set - final_set), columns=["u", "v"])  # edges removed by transitability
    plot_edges(precincts, removed, ax, color="tab:purple", alpha=0.8, lw=1.2)
    ax.set_aspect("equal")
    ax.set_title("Removed (Base - Final)")

    # Overlay panel: precincts (boundaries), rivers/lakes, final graph
    ax = axes[1][1]
    lakes = gpd.read_file(args.lakes)
    rivers = gpd.read_file(args.rivers)
    # Reproject water to match precincts
    if lakes.crs != precincts.crs:
        lakes = lakes.to_crs(precincts.crs)
    if rivers.crs != precincts.crs:
        rivers = rivers.to_crs(precincts.crs)
    # Plot order: precinct boundaries (no fill), water, then graph
    precincts.boundary.plot(ax=ax, color="black", linewidth=0.2)
    try:
        lakes.plot(ax=ax, color="#6baed6", edgecolor="#3182bd", linewidth=0.3, alpha=0.6)
    except Exception:
        pass
    try:
        rivers.plot(ax=ax, color="#3182bd", linewidth=0.4)
    except Exception:
        pass
    plot_edges(precincts, edges_final, ax, color="tab:red", alpha=0.6, lw=0.6)
    ax.set_aspect("equal")
    ax.set_title("Overlay: Precincts + Water + Graph")

    plt.tight_layout()
    if args.save:
        (art / "plots").mkdir(parents=True, exist_ok=True)
        out = art / "plots" / "transitability_overview.png"
        plt.savefig(out, dpi=250, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
