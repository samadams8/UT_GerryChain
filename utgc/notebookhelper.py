import os
import yaml
import geopandas as gpd
import pandas as pd
from IPython.display import display
import ipywidgets as widgets
from PIL import Image
from typing import Optional

def get_district_count(shapefile_path):
    """
    Get the number of districts from a shapefile.
    """
    shapefile = gpd.read_file(shapefile_path)
    return len(shapefile)

def load_boundaries_from_shapefiles(
    bounds_dir: str = "data/bounds",
    target_crs: str = "EPSG:26912",
    county_shapefile: Optional[str] = None,
    muni_shapefile: Optional[str] = None,
    county_path: Optional[str] = None,
    muni_path: Optional[str] = None,
):
    """
    Load municipality and county boundary GeoDataFrames from shapefiles.
    
    Parameters
    ----------
    bounds_dir : str, optional
        Base directory containing boundary shapefiles, by default "data/bounds".
        Only used if county_path/muni_path are not provided.
    target_crs : str, optional
        Target CRS to project to, by default "EPSG:26912"
    county_shapefile : str, optional
        Relative path to county shapefile from bounds_dir, by default 
        "UtahCountyBoundaries/ut_cnty_2020_bound.shp". Ignored if county_path is provided.
    muni_shapefile : str, optional
        Relative path to municipality shapefile from bounds_dir, by default 
        "UtahMunicipalBoundaries/Municipalities.shp". Ignored if muni_path is provided.
    county_path : str, optional
        Absolute path to county shapefile. If provided, takes precedence over 
        bounds_dir + county_shapefile.
    muni_path : str, optional
        Absolute path to municipality shapefile. If provided, takes precedence over 
        bounds_dir + muni_shapefile.
    
    Returns
    -------
    tuple
        (municipalities_gdf, counties_gdf) GeoDataFrames, or (None, None) if files not found
    """
    municipalities = None
    counties = None
    
    # Load county boundaries
    if county_path is None:
        if county_shapefile is None:
            county_shapefile = "UtahCountyBoundaries/ut_cnty_2020_bound.shp"
        county_path = os.path.join(bounds_dir, county_shapefile)
    
    if os.path.exists(county_path):
        counties = gpd.read_file(county_path)
        # Project to target CRS if needed
        if counties.crs != target_crs:
            counties = counties.to_crs(target_crs)
        print(f"Loaded {len(counties)} counties from {county_path}")
    else:
        print(f"Warning: County shapefile not found at {county_path}")
    
    # Load municipality boundaries
    if muni_path is None:
        if muni_shapefile is None:
            muni_shapefile = "UtahMunicipalBoundaries/Municipalities.shp"
        muni_path = os.path.join(bounds_dir, muni_shapefile)
    
    if os.path.exists(muni_path):
        municipalities = gpd.read_file(muni_path)
        # Project to target CRS if needed
        if municipalities.crs != target_crs:
            municipalities = municipalities.to_crs(target_crs)
        print(f"Loaded {len(municipalities)} municipalities from {muni_path}")
    else:
        print(f"Warning: Municipality shapefile not found at {muni_path}")
    
    return municipalities, counties

def generate_boundaries_from_geodata(geodata, muni_column="MUNIID", county_column="COUNTYID"):
    """
    Generate municipality and county boundary GeoDataFrames by dissolving the geodata.
    
    This function is deprecated in favor of load_boundaries_from_shapefiles() which uses
    the official boundary shapefiles in data/bounds.
    
    Parameters
    ----------
    geodata : gpd.GeoDataFrame
        The input geodata with geometry and ID columns
    muni_column : str, optional
        Column name for municipality IDs, by default "MUNIID"
    county_column : str, optional
        Column name for county IDs, by default "COUNTYID"
    
    Returns
    -------
    tuple
        (municipalities_gdf, counties_gdf) GeoDataFrames
    """
    municipalities = None
    counties = None
    
    if muni_column in geodata.columns:
        # Filter out empty MUNIIDs and dissolve
        muni_data = geodata[geodata[muni_column] != ""].copy()
        if len(muni_data) > 0:
            # Dissolve by municipality, keeping first value for name columns
            municipalities = muni_data.dissolve(by=muni_column, aggfunc='first')
            # Keep only geometry and relevant columns
            keep_cols = [c for c in municipalities.columns if c in ['MUNINAME', 'geometry'] or c == muni_column]
            municipalities = municipalities[[c for c in keep_cols if c in municipalities.columns]].reset_index()
            # Ensure CRS is preserved
            municipalities = municipalities.set_crs(geodata.crs, allow_override=True)
    
    if county_column in geodata.columns:
        # Dissolve by county, keeping first value for name columns
        counties = geodata.dissolve(by=county_column, aggfunc='first')
        # Keep only geometry and relevant columns
        keep_cols = [c for c in counties.columns if c in ['COUNTYNAME', 'geometry'] or c == county_column]
        counties = counties[[c for c in keep_cols if c in counties.columns]].reset_index()
        # Ensure CRS is preserved
        counties = counties.set_crs(geodata.crs, allow_override=True)
    
    return municipalities, counties

def load_config(config_path=""):
    """
    Load a configuration file. If none is provided, we'll retrieve the latest in results/configurations/
    """
    if config_path == "":
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
        config_path = candidates[0][1]
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def map_viewer_widget(image_dir):
    # Load PNGs that begin with "step_" in the config_dir directory
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith(".png") and f.startswith("step_")]
    )
    # Get step numbers from the filenames
    steps = [int(f.split("_")[-1].split(".")[0]) for f in image_files]

    img = widgets.Image(format='png')

    steps2index = {s: i for i, s in enumerate(steps)}
    # Slider shows actual step numbers
    stepper = widgets.BoundedIntText(value=steps[0], min=min(steps), max=max(steps), step=steps[1] - steps[0], description="Step:")

    frames = []
    for fname in image_files:
        with open(os.path.join(image_dir, fname), "rb") as f:
            data = f.read()
        frames.append(data)

    # Ensure no duplicate observers if you re-run this cell
    try:
        stepper.unobserve_all()
    except Exception: pass

    def on_change(value):
        img.value = frames[steps2index[value]]

    widgets.interactive(on_change, value=stepper)
    img.value = frames[steps2index[stepper.value]]

    # Create and display the widget
    widget_box = widgets.VBox([stepper, img])
    display(widget_box)