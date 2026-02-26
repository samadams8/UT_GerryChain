import geopandas as gpd

from utgc.geography import GeographyManager
from utgc.configuration import ConfigurationManager

# Configuration files and example maps will be saved to a directory with the current date and time plus an optional user-defined tag
config_tag = "polish_test"  

initial_plan = "maps/US-House/2025_USH_Leg-C/2025_USH_Leg-C.shp"

cfg = ConfigurationManager(
    random_seed=0,
    pop_column="TOTPOP",
    pop_tolerance=0.01,
)

geo = GeographyManager(
    pop_data={
        "blocks": "data/UT_blocks.geojson",
        "d4-cap": "data/UT_capped_d4_eps1e-3.geojson",
    },
    crs="EPSG:26912"
)