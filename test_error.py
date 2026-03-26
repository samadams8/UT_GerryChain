import geopandas as gpd
gdf = gpd.read_file('data/UT_blocks.geojson')
print("Total rows:", len(gdf))
# Print the value of the index 10073 if it exists
if 10073 in gdf.index:
    print("Row 10073 geometry:", gdf.loc[10073].geometry)
