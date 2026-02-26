"""
Build graph and load geodata for redistricting runs.

This module provides a backward-compatible thin wrapper around GeographyManager.
For new code, use GeographyManager directly to manage multiple datasets.
"""
from typing import Optional, Tuple

import geopandas as gpd
import maup

from gerrychain import Graph

from .geography import GeographyManager


def load_geodata_and_build_graph(
    pop_geodata_path: str,
    initial_plan_path: str,
    crs: Optional[str] = "EPSG:26912",
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Graph]:
    """
    Load population geodata and initial plan, optionally project to CRS,
    assign initial plan to geodata, and build a GerryChain Graph.
    Delegates to GeographyManager for loading (backward compatible).
    Returns geodata with "initial_plan" and "area" and a graph with the same.

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
    geo = GeographyManager(crs=crs or "EPSG:26912")
    geo.register_pop_dataset("_", pop_geodata_path)
    geo.register_plan_dataset("_", initial_plan_path)
    pop = geo.get_pop_geodata("_")
    plan = geo.get_plan_geodata("_")
    geodata = pop.copy()
    geodata["initial_plan"] = maup.assign(geodata, plan)
    if "area" not in geodata.columns:
        geodata["area"] = geodata.geometry.area
    graph = Graph.from_geodataframe(geodata)
    return geodata, plan, graph
