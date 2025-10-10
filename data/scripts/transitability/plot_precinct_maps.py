import argparse
import os
import json
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

def get_county_bounds(county_name, counties_gdf, name_col):
    """
    Gets the geometry and padded bounding box for a single county.
    """
    try:
        upper_names = counties_gdf[name_col].astype(str).str.upper()
        county_geom = counties_gdf[upper_names == county_name].union_all()
        if county_geom.is_empty:
            return None, None
        
        minx, miny, maxx, maxy = county_geom.bounds
        pad_x = (maxx - minx) * 0.05
        pad_y = (maxy - miny) * 0.05
        bounds = (minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y)
        return county_geom, bounds
    except Exception:
        return None, None

def plot_optional_layers(ax, layers):
    """
    Plots optional geographic layers like water and roads on the given axes.
    """
    if 'lakes' in layers and not layers['lakes'].empty:
        layers['lakes'].plot(ax=ax, color='lightblue', edgecolor='lightblue', linewidth=0.25, alpha=0.75, zorder=2)
    
    if 'rivers' in layers and not layers['rivers'].empty:
        layers['rivers'].plot(ax=ax, color='lightblue', edgecolor='lightblue', linewidth=0.5, alpha=0.75, zorder=3)
        
    if 'roads' in layers and not layers['roads'].empty:
        layers['roads'].plot(ax=ax, color='olive', linewidth=0.5, alpha=0.75, zorder=4)

def plot_edges(precincts_gdf, edges_df, ax, **kwargs):
    """
    Plots the connectivity graph edges on the given axes.
    """
    for _, row in edges_df.iterrows():
        try:
            # Use node index from GeoDataFrame if it exists, otherwise fall back to integer index
            u_lookup = row['source']
            v_lookup = row['target']
            
            u_geom = precincts_gdf.loc[u_lookup].geometry
            v_geom = precincts_gdf.loc[v_lookup].geometry
            
            p1 = u_geom.representative_point()
            p2 = v_geom.representative_point()
            
            ax.plot([p1.x, p2.x], [p1.y, p2.y], **kwargs)
        except (KeyError, IndexError):
            # Skip edges where one of the nodes doesn't exist in the precincts file
            continue

def plot_statewide_map(precincts_gdf, output_path, edges_df=None, optional_layers=None):
    """Generates and saves a single plot of the entire state."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_title("Statewide Precinct View", fontsize=16)

    precincts_gdf.plot(ax=ax, edgecolor='black', linewidth=1.5, facecolor='none', zorder=1)

    if optional_layers:
        plot_optional_layers(ax, optional_layers)
    
    if edges_df is not None:
        plot_edges(precincts_gdf, edges_df, ax, color='red', alpha=0.5, linewidth=1.5, zorder=5)
    
    state_label_threshold = 5e8  # 500 million m^2
    for index, row in precincts_gdf.iterrows():
        if row['area_m2'] > state_label_threshold:
            centroid = row.geometry.representative_point()
            ax.text(centroid.x, centroid.y, str(index), ha='center', va='center', fontsize=10, zorder=6,
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.1'))
    
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved statewide map to {output_path}")

def plot_county_map(county_name, precincts_gdf, county_geom, county_bounds, output_path, edges_df=None, optional_layers=None):
    """Generates and saves a single plot zoomed to a specific county."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_title(f"{county_name.title()} County Precincts", fontsize=16)

    precincts_gdf.plot(ax=ax, edgecolor='black', linewidth=1.5, facecolor='none', zorder=1)

    if optional_layers:
        plot_optional_layers(ax, optional_layers)

    minx, maxx, miny, maxy = county_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    
    if edges_df is not None:
        plot_edges(precincts_gdf, edges_df, ax, color='red', alpha=0.5, linewidth=1.5, zorder=5)

    county_label_threshold = 5e5 # 0.5 million m^2
    precincts_in_county = precincts_gdf[precincts_gdf.geometry.representative_point().within(county_geom)]

    for index, row in precincts_in_county.iterrows():
        if row['area_m2'] > county_label_threshold:
            centroid = row.geometry.representative_point()
            ax.text(centroid.x, centroid.y, str(index), ha='center', va='center', fontsize=12, zorder=6,
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.1'))
    
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {county_name.title()} County map to {output_path}")

def main(precincts_path, counties_path, output_dir, connectivity_path, lakes_path, rivers_path, roads_path):
    """
    Main function to load data and generate all requested maps.
    """
    print("Loading geographic data...")
    precincts_gdf = gpd.read_file(precincts_path)
    counties_gdf = gpd.read_file(counties_path)
    
    # Load optional geographic layers
    optional_layers = {}
    if lakes_path:
        print(f"Loading lakes data from {lakes_path}...")
        optional_layers['lakes'] = gpd.read_file(lakes_path)
    if rivers_path:
        print(f"Loading rivers data from {rivers_path}...")
        optional_layers['rivers'] = gpd.read_file(rivers_path)
    if roads_path:
        print(f"Loading roads data from {roads_path}...")
        optional_layers['roads'] = gpd.read_file(roads_path)

    edges_df = None
    if connectivity_path:
        print(f"Loading connectivity data from {connectivity_path}...")
        with open(connectivity_path, 'r') as f:
            edges_data = json.load(f)
        edges_df = pd.DataFrame(edges_data)
    
    os.makedirs(output_dir, exist_ok=True)

    if precincts_gdf.crs != counties_gdf.crs:
        print("Notice: Aligning CRS between precinct and county files.")
        precincts_gdf = precincts_gdf.to_crs(counties_gdf.crs)

    # Align CRS for optional layers
    for name, gdf in optional_layers.items():
        if gdf.crs != precincts_gdf.crs:
            print(f"Notice: Aligning CRS for {name} layer.")
            optional_layers[name] = gdf.to_crs(precincts_gdf.crs)

    gdf_proj = precincts_gdf.to_crs("EPSG:32612") # Project to compute area in meters
    precincts_gdf['area_m2'] = gdf_proj.geometry.area

    print("\nGenerating statewide map...")
    statewide_path = os.path.join(output_dir, "statewide_precincts.png")
    plot_statewide_map(precincts_gdf, statewide_path, edges_df=edges_df, optional_layers=optional_layers)

    county_name_col = find_county_name_column(counties_gdf)
    if not county_name_col:
        print("\n---")
        print("Error: Could not automatically find the county name column.")
        print("Available columns are:")
        for col in counties_gdf.columns:
            print(f"  - {col}")
        print("---\n")
        return

    all_counties = sorted(counties_gdf[county_name_col].unique())
    for county_name in all_counties:
        print(f"\nGenerating map for {county_name.title()} County...")
        geom, bounds = get_county_bounds(county_name.upper(), counties_gdf, county_name_col)
        if geom and bounds:
            county_filename = f"{county_name.replace(' ', '_').lower()}_precincts.png"
            output_path = os.path.join(output_dir, county_filename)
            plot_county_map(
                county_name, precincts_gdf, geom, bounds, output_path,
                edges_df=edges_df, optional_layers=optional_layers
            )
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
        help="Optional path to a JSON file with precinct connectivity data (e.g., transitability.json)."
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
        args.precincts_file, args.counties_file, args.output_dir, args.connectivity,
        args.lakes, args.rivers, args.roads
    )