import geopandas as gpd
import pandas as pd

def get_district_count(shapefile_path):
    """
    Get the number of districts from a shapefile.
    """
    shapefile = gpd.read_file(shapefile_path)
    return len(shapefile)