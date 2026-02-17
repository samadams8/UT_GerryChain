"""
Build graph and load geodata for redistricting runs.

This module provides reusable functions to load population geodata and
initial plan, optionally project to a CRS, assign initial plan to geodata,
and build a GerryChain Graph. Used by ConfigurationManager and available
for notebooks that need the same graph without full config setup.
"""
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np

import maup
from gerrychain import Graph


def load_geodata_and_build_graph(
    pop_geodata_path: str,
    initial_plan_path: str,
    crs: Optional[str] = "EPSG:26912",
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Graph]:
    """
    Load population geodata and initial plan, optionally project to CRS,
    assign initial plan to geodata, and build a GerryChain Graph.

    Parameters
    ----------
    pop_geodata_path : str
        Path to the population geodata file (e.g. GeoJSON or shapefile).
    initial_plan_path : str
        Path to the initial plan (district boundaries).
    crs : str or None, optional
        Target CRS for projection (e.g. "EPSG:26912"). If None, no projection is applied.

    Returns
    -------
    geodata : geopandas.GeoDataFrame
        Loaded and optionally projected population geodata with "initial_plan"
        and "area" columns added.
    initial_plan : geopandas.GeoDataFrame
        Loaded and optionally projected initial plan geometries.
    graph : gerrychain.Graph
        Graph built from geodata via Graph.from_geodataframe(geodata).
    """
    geodata = gpd.read_file(pop_geodata_path)
    print(f"Loaded {len(geodata)} segments from {pop_geodata_path}")
    initial_plan = gpd.read_file(initial_plan_path)
    print(f"Loaded {len(initial_plan)} districts from {initial_plan_path}")

    if crs:
        print(f"Projecting to {crs}")
        geodata = geodata.to_crs(crs)
        initial_plan = initial_plan.to_crs(crs)

    # Create unique IDs for unincorporated municipalities
    if "MUNIID" in geodata.columns and any(geodata["MUNIID"] == ""):
        print(
            "Found %d nodes assigned to %d incorporated municipalities"
            % (
                (geodata["MUNIID"] != "").sum(),
                len(set(geodata[geodata["MUNIID"] != ""]["MUNIID"])),
            )
        )
        print("Assigning unique IDs to unincorporated nodes...")

        existing_muniids = geodata[geodata["MUNIID"] != ""]["MUNIID"]
        if len(existing_muniids) > 0:
            max_id = int(existing_muniids.astype(int).max())
        else:
            max_id = 0

        unincorporated_mask = geodata["MUNIID"] == ""
        unincorporated_count = unincorporated_mask.sum()
        if unincorporated_count > 0:
            geodata.loc[unincorporated_mask, "MUNIID"] = np.arange(
                max_id + 1, max_id + 1 + unincorporated_count
            )
            print(f"Assigned unique IDs to {unincorporated_count} unincorporated nodes")

        num_unique_munis = len(set(geodata["MUNIID"]))
        print(f"Total unique MUNIIDs: {num_unique_munis}")

    # Assign initial plan to geodata
    geodata["initial_plan"] = maup.assign(geodata, initial_plan)
    if "area" not in geodata.columns:
        geodata["area"] = geodata.geometry.area

    graph = Graph.from_geodataframe(geodata)
    print(f"  Graph built with {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    return geodata, initial_plan, graph
