#!/usr/bin/env python3
"""
Visualize transitability artifacts in 4 panels using a CSV of removed edges.

1. Base adjacency (from precinct geometry)
2. Final (Transitable) graph (Base - Removed)
3. Removed edges (from the CSV file)
4. Overlay: precinct boundaries, water features, and the final graph
"""
import argparse
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from gerrychain import Graph
import os

def find_and_set_index(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Heuristically finds a unique ID column and sets it as the index."""
    CANDIDATE_ID_COLS = ["ID", "GEOID", "GEOID20", "VTDST20"]
    
    for col in CANDIDATE_ID_COLS:
        if col in gdf.columns:
            if gdf[col].is_unique:
                print(f"Found unique ID column: '{col}'. Setting as index.")
                return gdf.set_index(col)
            
    print("Warning: No unique standard ID column found ('ID', 'GEOID', etc.). Using existing index.")
    return gdf

def plot_edges(precincts: gpd.GeoDataFrame, edges: pd.DataFrame, ax, color: str, alpha: float, lw: float):
    """Helper function to plot graph edges between precinct centroids."""
    for _, row in edges.iterrows():
        try:
            p1 = precincts.loc[row["u"]].geometry.centroid
            p2 = precincts.loc[row["v"]].geometry.centroid
            line = [(p1.x, p1.y), (p2.x, p2.y)]
            (xs, ys) = zip(*line)
            ax.plot(xs, ys, color=color, alpha=alpha, linewidth=lw, solid_capstyle='round', zorder=5)
        except KeyError:
            # This can happen if the edges file and precincts file are mismatched
            # We'll just skip plotting the problematic edge.
            continue

def main():
    parser = argparse.ArgumentParser(
        description="Visualize transitability results from a removed_edges.csv file."
    )
    parser.add_argument(
        "--precincts", 
        default="data/UT_precincts.geojson",
        help="Path to the precincts GeoJSON or shapefile."
    )
    parser.add_argument(
        "--removed_edges_csv", 
        default="data/transitability/removed_edges.csv",
        help="Path to the CSV file containing edges to be removed."
    )
    parser.add_argument(
        "--lakes", 
        default="data/geography_processed/UtahMajorLakes_filtered.shp",
        help="Path to the shapefile for major lakes."
    )
    parser.add_argument(
        "--rivers", 
        default="data/geography_processed/UtahMajorRivers_filtered.shp",
        help="Path to the shapefile for major rivers."
    )
    parser.add_argument(
        "--save_path", 
        help="Path to save the output visualization PNG."
    )
    args = parser.parse_args()

    # --- Data Loading and Graph Processing ---
    print("Loading precinct data...")
    precincts_raw = gpd.read_file(args.precincts)
    precincts = find_and_set_index(precincts_raw)
    
    # Create the full base graph from precinct adjacencies
    base_graph = Graph.from_geodataframe(precincts)
    base_edges_set = {tuple(sorted(e)) for e in base_graph.edges}
    
    # Load the set of edges that were removed
    print(f"Loading removed edges from '{args.removed_edges_csv}'...")
    removed_edges_df = pd.read_csv(args.removed_edges_csv)
    removed_edges_set = {tuple(sorted(x)) for x in removed_edges_df.to_numpy()}
    
    # The final set of edges is the base set minus the removed set
    final_edges_set = base_edges_set - removed_edges_set
    
    print(f"Base graph has {len(base_edges_set)} edges.")
    print(f"Removed {len(removed_edges_set)} edges.")
    print(f"Final graph has {len(final_edges_set)} edges.")

    # Prepare DataFrames for plotting
    edges_base_df = pd.DataFrame(list(base_edges_set), columns=["u", "v"])
    edges_final_df = pd.DataFrame(list(final_edges_set), columns=["u", "v"])
    edges_removed_df = removed_edges_df # Use the dataframe directly

    # --- Plotting ---
    print("Generating 4-panel visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle("Transitability Analysis", fontsize=20, y=0.98)

    # Panel 1: Base adjacency
    ax = axes[0, 0]
    precincts.plot(ax=ax, color="whitesmoke", edgecolor="gray", linewidth=0.3, zorder=1)
    plot_edges(precincts, edges_base_df, ax, color="tab:blue", alpha=0.5, lw=0.5)
    ax.set_title("1. Base Adjacency Graph")

    # Panel 2: Final (Transitable) Edges
    ax = axes[0, 1]
    precincts.plot(ax=ax, color="whitesmoke", edgecolor="gray", linewidth=0.3, zorder=1)
    plot_edges(precincts, edges_final_df, ax, color="tab:green", alpha=0.6, lw=0.6)
    ax.set_title("2. Final (Transitable) Graph")

    # Panel 3: Removed Edges
    ax = axes[1, 0]
    precincts.plot(ax=ax, color="whitesmoke", edgecolor="gray", linewidth=0.3, zorder=1)
    plot_edges(precincts, edges_removed_df, ax, color="tab:purple", alpha=0.7, lw=0.8)
    ax.set_title("3. Removed Edges")

    # Panel 4: Overlay
    ax = axes[1, 1]
    lakes = gpd.read_file(args.lakes).to_crs(precincts.crs)
    rivers = gpd.read_file(args.rivers).to_crs(precincts.crs)
    precincts.boundary.plot(ax=ax, color="black", linewidth=0.2, zorder=2)
    lakes.plot(ax=ax, color="#a6cee3", edgecolor="#1f78b4", linewidth=0.4, alpha=0.7, zorder=3)
    rivers.plot(ax=ax, color="#1f78b4", linewidth=0.5, zorder=4)
    plot_edges(precincts, edges_final_df, ax, color="tab:red", alpha=0.7, lw=0.7)
    ax.set_title("4. Overlay (Final Graph + Water Features)")

    # Clean up axes
    for ax_row in axes:
        for ax_col in ax_row:
            ax_col.set_xticks([])
            ax_col.set_yticks([])
            ax_col.set_aspect('equal')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save or show the plot
    if args.save_path:
        output_folder = os.path.dirname(args.save_path)
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
        plt.savefig(args.save_path, dpi=300)
        print(f"Saved visualization to '{args.save_path}'")
    else:
        plt.show()

if __name__ == "__main__":
    main()

