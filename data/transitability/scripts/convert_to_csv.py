import argparse
import json
import pandas as pd
import geopandas as gpd
from gerrychain import Graph

def convert_json_to_csv_removed_edges(precincts_file, old_json_file, output_csv_file):
    """
    Converts an old transitability.json file (edges to keep) to a new
    removed_edges.csv file (edges to remove) by comparing it against the
    base precinct adjacency graph.

    Args:
        precincts_file (str): Path to the precincts GeoJSON/shapefile.
        old_json_file (str): Path to the old transitability.json file.
        output_csv_file (str): Path to save the new removed_edges.csv file.
    """
    print(f"Loading precincts from '{precincts_file}' to build base graph...")
    precincts_gdf = gpd.read_file(precincts_file)
    base_graph = Graph.from_geodataframe(precincts_gdf)
    base_edges = {tuple(sorted(edge)) for edge in base_graph.edges}
    print(f"Base graph contains {len(base_edges)} edges.")

    print(f"Loading final edges to keep from '{old_json_file}'...")
    with open(old_json_file, 'r') as f:
        final_edges_list = json.load(f)
    
    final_edges_set = {tuple(sorted((d['source'], d['target']))) for d in final_edges_list}
    print(f"JSON file contains {len(final_edges_set)} edges to keep.")

    # Find the edges that are in the base graph but not in the final graph
    removed_edges = base_edges - final_edges_set
    print(f"Calculated {len(removed_edges)} removed edges.")

    # Create and save the CSV
    removed_df = pd.DataFrame(list(removed_edges), columns=['u', 'v'])
    removed_df.to_csv(output_csv_file, index=False)
    print(f"Successfully saved removed edges to '{output_csv_file}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert old JSON transitability files to the new CSV format of removed edges."
    )
    parser.add_argument(
        "precincts_file", 
        type=str, 
        help="Path to the GeoJSON/shapefile for precincts (e.g., data/UT_precincts.geojson)."
    )
    parser.add_argument(
        "old_json_file", 
        type=str, 
        help="Path to the old transitability.json file containing edges to keep."
    )
    parser.add_argument(
        "output_csv_file", 
        type=str, 
        help="Path for the new output CSV file (e.g., data/transitability/removed_edges.csv)."
    )
    
    args = parser.parse_args()
    
    convert_json_to_csv_removed_edges(
        args.precincts_file,
        args.old_json_file,
        args.output_csv_file
    )
