import os
import yaml
import geopandas as gpd
import pandas as pd
from IPython.display import display
import ipywidgets as widgets
from PIL import Image
from typing import Optional, Literal
import matplotlib.pyplot as plt
import utgc.results as gcres
import utgc.plotting as gcplt

import warnings

def suppress_colab_warnings():
    # Suppress the datetime deprecation warning from Jupyter
    warnings.filterwarnings(
        "ignore", 
        category=DeprecationWarning, 
        module="jupyter_client"
    )

    # Suppress the NA values warnings from GerryChain
    warnings.filterwarnings(
        "ignore", 
        category=UserWarning, 
        module="gerrychain"
    )

def get_notebook_params(
    map_type: Literal["us_house", "ut_house", "ut_senate"],
    repo_dir: str = ".",
    data_option: Literal["blocks", "d-capped"] = "d-capped",
):
    if map_type == "us_house":
        params = {
            "prefix": "us_house",
            "data_tag": "d4-cap",
            "data_path": os.path.join(repo_dir, "data/UT_capped_d4_eps1e-3.zip"),
            "transitability_path": os.path.join(repo_dir, "data/UT_capped_d4_eps1e-3_transitability.csv"),
            "init_plan_path": os.path.join(repo_dir, "maps/US-House/preconditioned_1000.zip"),
            "pop_tolerance": 0.001,
            "comparison_maps": {
                "Map C": os.path.join(repo_dir, "maps/US-House/2025_USH_Leg-C/2025_USH_Leg-C.shp"),
                "Plaintiff 1": os.path.join(repo_dir, "maps/US-House/2025_USH_Plaintiff-1/2025_USH_Plaintiff-1.shp"),
                "Plaintiff 2": os.path.join(repo_dir, "maps/US-House/2025_USH_Plaintiff-2/2025_USH_Plaintiff-2.shp"),
                "2021 Enacted": os.path.join(repo_dir, "maps/US-House/2021_USH_Enacted/2021_USH_Enacted.shp"),
                "UIRC Orange": os.path.join(repo_dir, "maps/US-House/2021_USH_UIRC-Orange/2021_USH_UIRC-Orange.shp"),
                "UIRC Purple": os.path.join(repo_dir, "maps/US-House/2021_USH_UIRC-Purple/2021_USH_UIRC-Purple.shp"),
                "UIRC Public": os.path.join(repo_dir, "maps/US-House/2021_USH_UIRC-Public/2021_USH_UIRC-Public.shp"),
            },
        }
    elif map_type == "ut_senate":
        params = {
            "prefix": "ut_senate",
            "data_tag": "d29-cap",
            "data_path": os.path.join(repo_dir, "data/UT_capped_d29_eps1e-3.zip"),
            "transitability_path": os.path.join(repo_dir, "data/UT_capped_d29_eps1e-3_transitability.csv"),
            "init_plan_path": os.path.join(repo_dir, "maps/UT-Senate/preconditioned_3000.zip"),
            "pop_tolerance": 0.01,
            "comparison_maps": {
                "Enacted": os.path.join(repo_dir, "maps/UT-Senate/2021_UTS_Enacted/2021_UTS_Enacted.shp"),
                "UIRC Green": os.path.join(repo_dir, "maps/UT-Senate/2021_UTS_UIRC-Green/2021_UTS_UIRC-Green.shp"),
                "UIRC Orange": os.path.join(repo_dir, "maps/UT-Senate/2021_UTS_UIRC-Orange/2021_UTS_UIRC-Orange.shp"),
                "UIRC Purple": os.path.join(repo_dir, "maps/UT-Senate/2021_UTS_UIRC-Purple/2021_UTS_UIRC-Purple.shp"),
            },
        }
    elif map_type == "ut_house":
        params = {
            "prefix": "ut_house",
            "data_tag": "d75-cap",
            "data_path": os.path.join(repo_dir, "data/UT_capped_d75_eps1e-3.zip"),
            "transitability_path": os.path.join(repo_dir, "data/UT_capped_d75_eps1e-3_transitability.csv"),
            "init_plan_path": os.path.join(repo_dir, "maps/UT-House/preconditioned_10000.zip"),
            "pop_tolerance": 0.01,
            "comparison_maps": {
                "Enacted": os.path.join(repo_dir, "maps/UT-House/2021_UTH_Enacted/2021_UTH_Enacted.shp"),
                "UIRC Green": os.path.join(repo_dir, "maps/UT-House/2021_UTH_UIRC-Green/2021_UTH_UIRC-Green.shp"),
                "UIRC Orange": os.path.join(repo_dir, "maps/UT-House/2021_UTH_UIRC-Orange/2021_UTH_UIRC-Orange.shp"),
                "UIRC Purple": os.path.join(repo_dir, "maps/UT-House/2021_UTH_UIRC-Purple/2021_UTH_UIRC-Purple.shp"),
            },
        }
    
    if data_option == "blocks":
        params.update({
            "data_tag": "blocks",
            "data_path": os.path.join(repo_dir, "data/UT_blocks.zip"),
            "transitability_path": os.path.join(repo_dir, "data/UT_blocks_transitability.csv"),
        })
    
    return params

def get_district_count(shapefile_path):
    """
    Get the number of districts from a shapefile.
    """
    shapefile = gpd.read_file(shapefile_path)
    return len(shapefile)

def load_boundaries_from_shapefiles(
    bounds_dir: Optional[str] = None,
    target_crs: str = "EPSG:26912",
    county_shapefile: Optional[str] = None,
    muni_shapefile: Optional[str] = None,
    county_path: Optional[str] = None,
    muni_path: Optional[str] = None,
    repo_dir: str = ".",
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
    repo_dir : str, optional
        Path to the repository directory, by default "..".
    
    Returns
    -------
    tuple
        (municipalities_gdf, counties_gdf) GeoDataFrames, or (None, None) if files not found
    """
    municipalities = None
    counties = None
    
    if bounds_dir is None:
        bounds_dir = os.path.join(repo_dir, "data/bounds")
    
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
        configs_dir = _resolve_path(os.path.join("results", "configurations"))
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

def save_partition(partition, filepath: str, geodata: Optional[gpd.GeoDataFrame] = None):
    """
    Save a GerryChain Partition to a GeoJSON or Shapefile.
    
    Parameters
    ----------
    partition : gerrychain.Partition
        The partition to save.
    filepath : str
        The path to save the output file to. Supported extensions are .geojson and .shp.
    geodata : gpd.GeoDataFrame, optional
        The original geodataframe used to build the graph. If not provided,
        this function will attempt to reconstruct it from partition.graph.nodes.
    """
    if geodata is None:
        nodes_data = []
        for node_id, node_data in partition.graph.nodes(data=True):
            data = node_data.copy()
            data["node_id"] = node_id
            data["assignment"] = partition.assignment.get(node_id)
            nodes_data.append(data)
        
        gdf = gpd.GeoDataFrame(nodes_data)
        if "geometry" not in gdf.columns:
            raise ValueError("No geometry found in partition.graph.nodes and no geodata provided.")
        gdf = gdf.set_geometry("geometry")
        
        if hasattr(partition.graph, "crs"):
            gdf.crs = partition.graph.crs
        elif hasattr(partition.graph, "graph") and "crs" in partition.graph.graph:
            gdf.crs = partition.graph.graph["crs"]
    else:
        gdf = geodata.copy()
        gdf["assignment"] = gdf.index.map(dict(partition.assignment))

    gdf = gdf.dropna(subset=["assignment"])
    districts = gdf.dissolve(by="assignment").reset_index()
    
    keep_cols = ["assignment", "geometry"]
    districts = districts[[c for c in keep_cols if c in districts.columns]]

    filepath_lower = filepath.lower()
    if filepath_lower.endswith(".geojson"):
        districts.to_file(filepath, driver="GeoJSON")
    elif filepath_lower.endswith(".shp"):
        # Note: A .shp file is actually a collection of files (.shp, .shx, .dbf, .prj).
        # This will create all of those files at the given path.
        districts.to_file(filepath)
    elif filepath_lower.endswith(".zip"):
        # Save shapefile components to a temporary directory and zip them up
        import tempfile
        import shutil
        
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = os.path.join(tmpdir, f"{base_name}.shp")
            districts.to_file(shp_path)
            
            zip_base = os.path.splitext(filepath)[0]
            shutil.make_archive(zip_base, 'zip', tmpdir)
    else:
        raise ValueError("Unsupported file extension. Please use .geojson, .shp, or .zip")

def get_updater_values(partition, updaters_to_save):
    """
    Extract values from a gerrychain Partition for saving.
    
    Parameters
    ----------
    partition : gerrychain.Partition
        The partition to extract values from.
    updaters_to_save : list
        List of updater names to extract values for.
        
    Returns
    -------
    dict
        Dictionary of updater values.
    """
    data = {}
    for name in updaters_to_save:
        value = partition[name]
        if isinstance(value, dict):
            data[name] = {str(k): v for k, v in sorted(value.items())}
        else:
            data[name] = value
    
    return data

def multi_districts_figure(setnames, metricname, output_path, comparison_maps, highlight_interval=[0.025, 0.975], relative_to_median=False, hline_value=None):
    fig, axes = plt.subplots(len(setnames), 1, dpi=300)
    for setindex, setname in enumerate(setnames):
        if hline_value is not None:
            axes[setindex].axhline(hline_value, color='black', linestyle='--')
        datakey = setname + "_majority_partisan_shares"
        party_shares = gcres.read_jsonl_table(output_path, datakey)
        party_shares = gcres.sort_subentries(party_shares, datakey)
        gcplt.district_plot(
            party_shares,
            highlight_interval=[0.025, 0.975],
            reference_values={
                k: sorted(v[datakey].values()) for k, v in comparison_maps.items()
            },
            relative_to_median=False,
            ax=axes[setindex]
        )
        axes[setindex].set_ylabel(setname)

    min_y = min(ax.get_ylim()[0] for ax in axes)
    max_y = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(min_y, max_y)

    axes[-1].set_xlabel(metricname.replace("_", " "))
    
    return fig

def multi_distribution_figure(setnames, metricname, output_path, comparison_maps, highlight_interval=[0.025, 0.975], relative_to_median=False):
    fig, axes = plt.subplots(len(setnames), 1, dpi=300)
    for setindex, setname in enumerate(setnames):
        datakey = setname + "_" + metricname
        vals = gcres.read_jsonl_table(output_path, datakey)
        gcplt.distribution_plot(
            vals[datakey],
            highlight_interval=highlight_interval,
            reference_values={
                mapname: stats[datakey]
                for mapname, stats in comparison_maps.items()
            },
            relative_to_median=relative_to_median,
            ax=axes[setindex]
        )
        axes[setindex].set_ylabel(setname.replace("_", " "))

    min_x = min(ax.get_xlim()[0] for ax in axes)
    max_x = max(ax.get_xlim()[1] for ax in axes)
    for ax in axes:
        ax.set_xlim(min_x, max_x)

    axes[-1].set_xlabel(metricname.replace("_", " "))

    return fig