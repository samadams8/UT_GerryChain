import os
import yaml
import geopandas as gpd
import pandas as pd

def get_district_count(shapefile_path):
    """
    Get the number of districts from a shapefile.
    """
    shapefile = gpd.read_file(shapefile_path)
    return len(shapefile)

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