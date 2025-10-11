import argparse
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

def find_county_name_column(gdf):
    """
    Finds the correct column name for counties in a GeoDataFrame by checking
    a list of common candidates, then performing a heuristic search.
    """
    target_names = {"WEBER", "DAVIS", "SALT LAKE", "UTAH"}
    candidate_cols = [
        "NAME", "County", "COUNTY", "COUNTYNAME", "COUNTY_NA",
        "COUNTYNAM", "CNTY_NAME", "NAMELSAD", "NAMELSAD20"
    ]
    
    # --- Pass 1: Check common candidate names first ---
    for col in candidate_cols:
        if col in gdf.columns:
            try:
                upper_vals = gdf[col].astype(str).str.upper()
                if upper_vals.isin(target_names).any():
                    print(f"Found county name column: '{col}'")
                    return col
            except Exception:
                continue

    # --- Pass 2: Heuristic search if no common name is found ---
    print("Could not find a common county name column. Performing a heuristic search...")
    for col in gdf.columns:
        if gdf[col].dtype == 'object':
            try:
                upper_vals = gdf[col].astype(str).str.upper()
                if upper_vals.isin(target_names).sum() >= 3:
                    print(f"Heuristically identified county name column: '{col}'")
                    return col
            except Exception:
                continue

    return None

def get_county_data(county_name, counties_gdf, county_name_col):
    """Retrieves the geometry GeoDataFrame and bounding box for a specific county."""
    if not county_name_col:
        print("Error: Could not determine the county name column in the counties file.")
        return None, None
        
    county_gdf = counties_gdf[counties_gdf[county_name_col].str.upper() == county_name]
    if not county_gdf.empty:
        # Return the GeoDataFrame itself for plotting, and its bounds
        bounds = county_gdf.total_bounds
        return county_gdf, bounds
    
    print(f"Warning: County '{county_name}' not found in counties file.")
    return None, None

def plot_optional_layers(ax, lakes_gdf, rivers_gdf, roads_gdf, crs):
    """Plots optional geographic layers like water and roads with a consistent style."""
    if lakes_gdf is not None:
        lakes_gdf = lakes_gdf.to_crs(crs)
        lakes_gdf.plot(ax=ax, color="lightblue", edgecolor="lightblue", linewidth=0.25, alpha=0.75, zorder=2)
    if rivers_gdf is not None:
        rivers_gdf = rivers_gdf.to_crs(crs)
        rivers_gdf.plot(ax=ax, color="lightblue", edgecolor="lightblue", linewidth=0.5, alpha=0.5, zorder=3)
    if roads_gdf is not None:
        roads_gdf = roads_gdf.to_crs(crs)
        roads_gdf.plot(ax=ax, color='olive', linewidth=0.5, alpha=0.75, zorder=4)

def plot_edges(precincts_gdf, edges_df, ax, **kwargs):
    """Plots edges on a map, connecting the centroids of precincts."""
    if edges_df is None or edges_df.empty:
        return
        
    for _, edge in edges_df.iterrows():
        try:
            u, v = edge['u'], edge['v']
            p1 = precincts_gdf.loc[u].geometry.representative_point()
            p2 = precincts_gdf.loc[v].geometry.representative_point()
            ax.plot([p1.x, p2.x], [p1.y, p2.y], **kwargs)
        except (KeyError, IndexError):
            # Skip edges where one of the nodes is not in the precincts GDF
            continue



def plot_statewide_map(precincts_gdf, output_path, edges_df=None, lakes_gdf=None, rivers_gdf=None, roads_gdf=None):
    """Generates and saves a plot of all precincts in the state."""
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # Plot precincts as boundaries only to serve as the base grid
    precincts_gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5, facecolor='none', zorder=1)

    # Plot optional layers on top of the precinct grid
    plot_optional_layers(ax, lakes_gdf, rivers_gdf, roads_gdf, precincts_gdf.crs)
    
    # Plot connectivity graph on top of everything
    plot_edges(precincts_gdf, edges_df, ax, color='red', linewidth=1.5, alpha=0.5, zorder=5)

    for index, row in precincts_gdf.iterrows():
        centroid = row.geometry.representative_point()
        ax.text(centroid.x, centroid.y, str(index), ha='center', va='center', fontsize=10, zorder=6,
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.1'))

    ax.set_title("Statewide Precinct Map", fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_county_map(county_name, precincts_gdf, county_gdf, bounds, output_path, edges_df=None, lakes_gdf=None, rivers_gdf=None, roads_gdf=None):
    """Generates and saves a plot for a specific county."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    precincts_gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5, facecolor='none', zorder=1)
    
    # Plot optional layers 
    plot_optional_layers(ax, lakes_gdf, rivers_gdf, roads_gdf, precincts_gdf.crs)
    
    # Plot connectivity graph on top
    plot_edges(precincts_gdf, edges_df, ax, color='red', linewidth=1.5, alpha=0.5, zorder=5)

    # Get precincts within county
    county_precincts = precincts_gdf[precincts_gdf['COUNTY'].str.upper() == county_name.upper()]

    for index, row in county_precincts.iterrows():
        centroid = row.geometry.representative_point()
        ax.text(centroid.x, centroid.y, str(index), ha='center', va='center', fontsize=12, zorder=6,
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.1'))

    ax.set_title(f"{county_name.title()} County Precincts", fontsize=16)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def load_connectivity_edges(precincts_gdf, connectivity_csv_path):
    """Loads connectivity data and determines the set of edges to plot."""
    from gerrychain import Graph

    if not connectivity_csv_path or not os.path.exists(connectivity_csv_path):
        print("Connectivity CSV not found. Plotting without connectivity graph.")
        return None

    print(f"Loading connectivity data from {connectivity_csv_path}...")
    base_graph = Graph.from_geodataframe(precincts_gdf)
    base_edges = {tuple(sorted(e)) for e in base_graph.edges}
    
    removed_edges_df = pd.read_csv(connectivity_csv_path)
    removed_edges = {tuple(sorted(x)) for x in removed_edges_df.to_numpy()}
    
    final_edges = base_edges - removed_edges
    
    edges_df = pd.DataFrame(list(final_edges), columns=['u', 'v'])
    print(f"Loaded {len(edges_df)} edges for plotting.")
    return edges_df

def main(precincts_file, counties_file, output_dir, connectivity_csv, lakes_file, rivers_file, roads_file):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading geospatial data...")
    precincts_gdf = gpd.read_file(precincts_file)
    # Ensure a unique ID column is the index. Fallback to default if none found.
    if 'ID' in precincts_gdf.columns and precincts_gdf['ID'].is_unique:
        precincts_gdf = precincts_gdf.set_index('ID')
    else:
        print("Warning: 'ID' column not found or not unique. Using default index.")

    counties_gdf = gpd.read_file(counties_file)
    
    # --- CRS Alignment (Fix for clipping issues) ---
    if precincts_gdf.crs != counties_gdf.crs:
        print("Aligning CRS between precincts and counties files...")
        counties_gdf = counties_gdf.to_crs(precincts_gdf.crs)
        
    county_name_col = find_county_name_column(counties_gdf)

    # Load optional layers
    lakes_gdf = gpd.read_file(lakes_file) if lakes_file and os.path.exists(lakes_file) else None
    rivers_gdf = gpd.read_file(rivers_file) if rivers_file and os.path.exists(rivers_file) else None
    roads_gdf = gpd.read_file(roads_file) if roads_file and os.path.exists(roads_file) else None
    
    # Load connectivity data if provided
    edges_df = load_connectivity_edges(precincts_gdf, connectivity_csv)

    # --- Generate Statewide Map ---
    print("\nGenerating statewide map...")
    state_filename = os.path.join(output_dir, "statewide_precincts.png")
    plot_statewide_map(precincts_gdf, state_filename, edges_df, lakes_gdf, rivers_gdf, roads_gdf)

    # --- Generate County Maps ---
    if not county_name_col:
        print("\nSkipping county maps because a valid county name column could not be identified.")
    else:
        all_counties = sorted(counties_gdf[county_name_col].unique())
        for county_name in all_counties:
            print(f"\nGenerating map for {county_name.title()} County...")
            county_gdf, bounds = get_county_data(county_name.upper(), counties_gdf, county_name_col)
            if county_gdf is not None and bounds is not None:
                county_filename = f"{county_name.replace(' ', '_').lower()}_precincts.png"
                output_path = os.path.join(output_dir, county_filename)
                plot_county_map(county_name, precincts_gdf, county_gdf, bounds, output_path, edges_df, lakes_gdf, rivers_gdf, roads_gdf)
            else:
                print(f"Could not find or process geometry for {county_name.title()} County.")
    
    print("\nAll maps have been generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates plots of Utah precincts with optional connectivity graph overlay."
    )
    parser.add_argument("precincts_file", type=str, help="Path to the GeoJSON file for precincts.")
    parser.add_argument("counties_file", type=str, help="Path to the GeoJSON/shapefile for counties.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/geography/figures",
        help="Directory to save the output figures."
    )
    parser.add_argument(
        "--connectivity",
        type=str,
        default=None,
        help="Optional path to a CSV file with removed precinct edges."
    )
    parser.add_argument(
        "--lakes",
        type=str,
        default=None,
        help="Optional path to a shapefile with lake features."
    )
    parser.add_argument(
        "--rivers",
        type=str,
        default=None,
        help="Optional path to a shapefile with river features."
    )
    parser.add_argument(
        "--roads",
        type=str,
        default=None,
        help="Optional path to a shapefile with road features."
    )
    args = parser.parse_args()
    
    main(
        args.precincts_file, args.counties_file, args.output_dir, 
        args.connectivity, args.lakes, args.rivers, args.roads
    )