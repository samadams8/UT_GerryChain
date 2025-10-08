from pathlib import Path
import os
import sys
import geopandas as gpd
import maup
import numpy as np

def load_data(nodes_data_path=None, initial_partition_path=None):
    """Load precinct data and initial congressional plan.
    
    Args:
        nodes_data_path: Path to precincts GeoJSON file. Defaults to "data/UT_precincts.geojson"
        initial_partition_path: Path to initial plan shapefile. Defaults to "plans/CONG/2025_UT-C/2025_UT-C.shp"
    
    Returns:
        tuple: (precincts, initial_plan) GeoDataFrames
    """
    print("Loading data...")

    # Use provided paths or fall back to defaults for backward compatibility
    precincts_path = nodes_data_path or "data/UT_precincts.geojson"
    if not os.path.exists(precincts_path):
        print(f"Error: {precincts_path} not found. Run 02_compile_precincts.py first.")
        sys.exit(1)

    precincts = gpd.read_file(precincts_path)
    print(f"Loaded {len(precincts)} precincts")

    # Create unique IDs for unincorporated municipalities
    if any(precincts["MUNIID"] == ""):
        print("Found %d nodes assigned to %d incorporated municipalities" % (
            (precincts["MUNIID"] != "").sum(),
            len(set(precincts[precincts["MUNIID"] != ""]["MUNIID"]))
        ))
        print("Assigning unique IDs to unincorporated nodes...")
        
        # Get existing numeric MUNIIDs
        existing_muniids = precincts[precincts["MUNIID"] != ""]["MUNIID"]
        if len(existing_muniids) > 0:
            max_id = int(existing_muniids.astype(int).max())
        else:
            max_id = 0

        # Generate unique sequential IDs for unincorporated areas
        unincorporated_mask = precincts["MUNIID"] == ""
        unincorporated_count = unincorporated_mask.sum()
        if unincorporated_count > 0:
            precincts.loc[unincorporated_mask, "MUNIID"] = np.arange(max_id + 1, max_id + 1 + unincorporated_count)
            print(f"Assigned unique IDs to {unincorporated_count} unincorporated nodes")
        
        # Print final municipality count
        num_unique_munis = len(set(precincts["MUNIID"]))
        print(f"Total unique MUNIIDs: {num_unique_munis}")
    
    initial_plan_path = initial_partition_path or "plans/CONG/2025_UT-C/2025_UT-C.shp"
    if not os.path.exists(initial_plan_path):
        print(f"Error: {initial_plan_path} not found.")
        sys.exit(1)

    initial_plan = gpd.read_file(initial_plan_path)
    print(f"Loaded initial plan with {len(initial_plan)} districts")

    if precincts.crs != initial_plan.crs:
        initial_plan = initial_plan.to_crs(precincts.crs)

    precincts["CONGDIST"] = maup.assign(precincts, initial_plan)
    precincts["area"] = precincts.geometry.area

    return precincts, initial_plan


def load_county_boundaries(precincts):
    """Load county boundaries for visualization overlay."""
    county_path = "data/cois/UtahCountyBoundaries/ut_cnty_2020_bound.shp"
    counties = None
    if os.path.exists(county_path):
        print(f"Loading county boundaries from {county_path}...")
        counties = gpd.read_file(county_path)
        counties = counties.to_crs(precincts.crs)
        print(f"Loaded {len(counties)} counties")
    else:
        print(f"Warning: {county_path} not found.")
    return counties


def load_municipality_boundaries(precincts):
    """Load municipality boundaries for visualization overlay."""
    muni_path = "data/cois/UtahMunicipalBoundaries/Municipalities.shp"
    municipalities = None
    if os.path.exists(muni_path):
        print(f"Loading municipality boundaries from {muni_path}...")
        municipalities = gpd.read_file(muni_path)
        municipalities = municipalities.to_crs(precincts.crs)
        print(f"Loaded {len(municipalities)} municipalities")
    else:
        print(f"Warning: {muni_path} not found.")
    return municipalities


def detect_election_data(precincts):
    """Detect available election data in precincts."""
    election_years = [2016, 2018, 2020, 2024]
    offices = ["PRE", "GOV", "ATG", "AUD", "TRE", "USS"]

    available_elections = []

    for year in election_years:
        for office in offices:
            dem_col = f"{year%100:02d}{office}D"
            rep_col = f"{year%100:02d}{office}R"
            if dem_col in precincts.columns and rep_col in precincts.columns:
                try:
                    dem_total = float(precincts[dem_col].fillna(0).sum())
                    rep_total = float(precincts[rep_col].fillna(0).sum())
                    if dem_total > 0 and rep_total > 0:
                        available_elections.append(f"{year}_{office}")
                except Exception:
                    pass

    return available_elections


def filter_elections(available_elections, years=None, offices=None):
    """Filter the detected elections by selected years and offices."""
    if years:
        years_set = set(int(y) for y in years)
    else:
        years_set = None
    if offices:
        offices_set = set(offices)
    else:
        offices_set = None
    filtered = []
    for e in available_elections:
        try:
            y_str, office = e.split('_')
            y = int(y_str)
            if (years_set is None or y in years_set) and (offices_set is None or office in offices_set):
                filtered.append(e)
        except Exception:
            if years_set is None and offices_set is None:
                filtered.append(e)
    return filtered


def get_election_columns(precincts):
    """Get all election-related columns from precincts data."""
    election_columns = []
    for col in precincts.columns:
        if len(col) == 6 and col[:2].isdigit() and col[2:5] in ["PRE", "GOV", "ATG", "AUD", "TRE", "USS"] and col[5] in ["R", "D", "O"]:
            election_columns.append(col)
    return sorted(election_columns)


