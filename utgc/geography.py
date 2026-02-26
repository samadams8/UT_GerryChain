"""
Geography manager for redistricting runs.

Loads and harmonizes geodata, builds graphs, and provides graph/partition
interfaces for ConfigurationManager and EnsembleRunner. All geodata and
geometry-specific logic lives here; config and chain work only with graph
and partition.
"""
from typing import Optional, Dict, Any, Set, List, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import maup

from gerrychain import Graph, GeographicPartition

from .partition_builder import build_initial_partition

# Type for registry value: path string or cached GeoDataFrame
_GeoDataEntry = Union[str, gpd.GeoDataFrame]

def _load_and_harmonize(
    pop_geodata_path: str,
    initial_plan_path: str,
    crs: Optional[str] = "EPSG:26912",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Graph]:
    """
    Load geodata and initial plan, project to CRS, harmonize (MUNIID, area),
    assign initial plan, and build graph. Used by graph_builder for backward
    compatibility only.
    """
    geodata = gpd.read_file(pop_geodata_path)
    print(f"Loaded {len(geodata)} segments from {pop_geodata_path}")
    initial_plan = gpd.read_file(initial_plan_path)
    print(f"Loaded {len(initial_plan)} districts from {initial_plan_path}")

    if crs:
        print(f"Projecting to {crs}")
        geodata = geodata.to_crs(crs)
        initial_plan = initial_plan.to_crs(crs)

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
        print(f"Total unique MUNIIDs: {len(set(geodata['MUNIID']))}")

    geodata["initial_plan"] = maup.assign(geodata, initial_plan)
    if "area" not in geodata.columns:
        geodata["area"] = geodata.geometry.area

    graph = Graph.from_geodataframe(geodata)
    print(f"  Graph built with {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    return geodata, initial_plan, graph


def _assignment_column_name(plan_key: str, existing_columns: Set[str]) -> str:
    """
    Return the column name to use for plan assignment. Use plan_key unless
    it conflicts or is too long, then use a fallback.
    """
    if plan_key not in existing_columns and len(plan_key) <= 64:
        return plan_key
    fallback = f"plan_{plan_key}"[:64]
    if fallback in existing_columns:
        fallback = f"plan_{hash(plan_key) % 10**8}"
    return fallback


class GeographyManager:
    """
    Manages multiple population and plan geodata datasets by key. Responsible
    for loading, harmonization (CRS, unique IDs, area), and graph/partition
    construction. ConfigurationManager and Runner use only graph/partition
    from here.
    """

    def __init__(self, crs: str = "EPSG:26912"):
        """
        Parameters
        ----------
        crs : str, optional
            Shared CRS for projection when loading datasets (e.g. "EPSG:26912").
            Can be changed later via self.crs.
        """
        self.crs = crs
        self._pop_datasets: Dict[str, _GeoDataEntry] = {}
        self._plan_datasets: Dict[str, _GeoDataEntry] = {}

    # --- Registration ---

    def register_pop_datasets(self, pop_data_paths: Dict[str, str]) -> None:
        """Register a dictionary of population geodata paths under the given keys."""
        for key, path in pop_data_paths.items():
            self.register_pop_dataset(key, path)

    def register_plan_datasets(self, plan_data_paths: Dict[str, str]) -> None:
        """Register a dictionary of plan geodata paths under the given keys."""
        for key, path in plan_data_paths.items():
            self.register_plan_dataset(key, path)

    def register_pop_dataset(self, key: str, pop_geodata_path: str) -> None:
        """Register a population geodata path under the given key."""
        self._pop_datasets[key] = pop_geodata_path

    def register_plan_dataset(self, key: str, plan_geodata_path: str) -> None:
        """Register a plan (district boundaries) geodata path under the given key."""
        self._plan_datasets[key] = plan_geodata_path

    # --- Lazy loading (private) ---

    def _load_pop_geodata(self, key: str) -> gpd.GeoDataFrame:
        """Load and cache population geodata for key; project to self.crs, add area if missing."""
        entry = self._pop_datasets.get(key)
        if entry is None:
            raise KeyError(f"Pop dataset '{key}' not found. Available: {list(self._pop_datasets.keys())}")
        if isinstance(entry, gpd.GeoDataFrame):
            return entry
        path = entry
        gdf = gpd.read_file(path)
        if self.crs:
            gdf = gdf.to_crs(self.crs)
        if "area" not in gdf.columns:
            gdf["area"] = gdf.geometry.area
        self._pop_datasets[key] = gdf
        return gdf

    def _load_plan_geodata(self, key: str) -> gpd.GeoDataFrame:
        """Load and cache plan geodata for key; project to self.crs, add area if missing."""
        entry = self._plan_datasets.get(key)
        if entry is None:
            raise KeyError(f"Plan dataset '{key}' not found. Available: {list(self._plan_datasets.keys())}")
        if isinstance(entry, gpd.GeoDataFrame):
            return entry
        path = entry
        gdf = gpd.read_file(path)
        if self.crs:
            gdf = gdf.to_crs(self.crs)
        if "area" not in gdf.columns:
            gdf["area"] = gdf.geometry.area
        self._plan_datasets[key] = gdf
        return gdf

    def _ensure_unique_ids(self, geodata: gpd.GeoDataFrame, cols: List[str]) -> None:
        """Ensure each column in cols has unique, non-empty values (in-place)."""
        for col in cols:
            if col not in geodata.columns:
                continue
            series = geodata[col]
            empty = series.isna() | (series.astype(str).str.strip() == "")
            if not empty.any():
                # Check duplicates
                if series.nunique() == len(series):
                    continue
            # Assign new unique values where empty or duplicated
            existing = series[~empty]
            if len(existing) > 0:
                try:
                    max_id = int(pd.to_numeric(existing, errors="coerce").max())
                except (TypeError, ValueError):
                    max_id = 0
            else:
                max_id = 0
            need_id = empty | series.duplicated(keep="first")
            n_need = need_id.sum()
            if n_need > 0:
                new_ids = np.arange(max_id + 1, max_id + 1 + n_need)
                geodata.loc[need_id, col] = new_ids

    def _build_graph(
        self,
        pop_geodata_key: str,
        plan_assignment_key: str,
        unique_id_cols: Optional[List[str]] = None,
    ) -> Graph:
        """Build graph from pop geodata with plan assigned; no caching."""
        pop_geodata = self._load_pop_geodata(pop_geodata_key).copy()
        plan_geodata = self._load_plan_geodata(plan_assignment_key)

        assignment_col = _assignment_column_name(
            plan_assignment_key, set(pop_geodata.columns)
        )
        pop_geodata[assignment_col] = maup.assign(pop_geodata, plan_geodata)

        if unique_id_cols:
            self._ensure_unique_ids(pop_geodata, unique_id_cols)

        return Graph.from_geodataframe(pop_geodata)

    # --- Public accessors ---

    def get_pop_columns_for_key(self, key: str) -> List[str]:
        """Return the columns of the population geodata for the given key."""
        return list(self._load_pop_geodata(key).columns)

    def get_plan_columns_for_key(self, key: str) -> List[str]:
        """Return the columns of the plan geodata for the given key."""
        return list(self._load_plan_geodata(key).columns)

    def get_pop_geodata(self, key: str) -> gpd.GeoDataFrame:
        """Return population geodata for the given key (loaded lazily)."""
        return self._load_pop_geodata(key)

    def get_plan_geodata(self, key: str) -> gpd.GeoDataFrame:
        """Return plan geodata for the given key (loaded lazily)."""
        return self._load_plan_geodata(key)

    def get_graph(
        self,
        pop_geodata_key: str,
        plan_assignment_key: str,
        unique_id_cols: Optional[List[str]] = None,
    ) -> Graph:
        """Build and return graph for the given pop and plan keys (no cache)."""
        return self._build_graph(pop_geodata_key, plan_assignment_key, unique_id_cols)

    def build_partition(
        self,
        pop_geodata_key: str,
        plan_assignment_key: str,
        updaters: Optional[Dict] = None,
        repair_contiguity: bool = True,
        unique_id_cols: Optional[List[str]] = None,
    ) -> GeographicPartition:
        """
        Build a GeographicPartition from the given pop and plan keys.
        Optionally repairs contiguity by default.
        """
        updaters = updaters or {}
        pop_geodata = self._load_pop_geodata(pop_geodata_key)
        assignment_col = _assignment_column_name(
            plan_assignment_key, set(pop_geodata.columns)
        )
        graph = self._build_graph(
            pop_geodata_key, plan_assignment_key, unique_id_cols
        )
        plan_gdf = self._load_plan_geodata(plan_assignment_key)
        num_districts = len(plan_gdf)

        return build_initial_partition(
            graph,
            assignment=assignment_col,
            updaters=updaters,
            num_districts=num_districts,
            repair=repair_contiguity,
        )

    def list_pop_keys(self) -> List[str]:
        """Return list of all registered population dataset keys."""
        return sorted(set(self._pop_datasets))

    def list_plan_keys(self) -> List[str]:
        """Return list of all registered plan dataset keys."""
        return sorted(set(self._plan_datasets))

    def get_geography_config(
        self, default_dataset: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Return a dict suitable for the 'geography' section of run metadata.
        Includes crs, datasets (key -> pop_geodata_path, plan_geodata_path),
        and optional default_dataset.
        """
        all_keys = set(self._pop_datasets) | set(self._plan_datasets)
        datasets: Dict[str, Dict[str, Optional[str]]] = {}
        for key in sorted(all_keys):
            pop_entry = self._pop_datasets.get(key)
            plan_entry = self._plan_datasets.get(key)
            pop_path = pop_entry if isinstance(pop_entry, str) else None
            plan_path = plan_entry if isinstance(plan_entry, str) else None
            datasets[key] = {
                "pop_geodata_path": pop_path,
                "plan_geodata_path": plan_path,
            }
        out: Dict[str, Any] = {"crs": self.crs, "datasets": datasets}
        if default_dataset is not None:
            out["default_dataset"] = default_dataset
        return out

    def make_total_column(
        self,
        total_col: str,
        all_election_columns: List[str],
        key: str,
    ) -> None:
        """
        Add a total-vote column to the population geodata for the given key.
        Operates in place on the cached geodata; the next graph build will
        include this column.
        """
        geodata = self._load_pop_geodata(key)
        if total_col in geodata.columns:
            return
        party_cols = [c for c in all_election_columns if c in geodata.columns]
        if party_cols:
            geodata[total_col] = 0
            for col in party_cols:
                try:
                    geodata[total_col] = (
                        geodata[total_col].fillna(0) + geodata[col].fillna(0)
                    )
                except Exception:
                    geodata[total_col] = (
                        geodata[total_col].fillna(0)
                        + pd.to_numeric(geodata[col], errors="coerce").fillna(0)
                    )

    def get_election_columns_for_key(
        self,
        key: str,
        years: List[int] = [],
        offices: List[str] = [],
    ) -> List[str]:
        """
        Return election column names present in the population geodata for key
        that match the given years and offices (e.g. G20USP format). If no years (or offices) are provided, returns election columns that match all available years (or offices).
        """
        geodata = self._load_pop_geodata(key)
        available_columns = set(geodata.columns)
        available_elections = []
        for column in available_columns:
            if isinstance(column, str) and column.startswith("G"):
                try:
                    year = int(column[1:3]) + 2000
                    office = column[3:6]
                    if (
                        (len(years) == 0 or year in years)
                        and (len(offices) == 0 or office in offices)
                    ):
                        available_elections.append(column)
                except ValueError:
                    pass
        return sorted(available_elections)

    def compute_metrics_for_map(
        self,
        shapefile_path: str,
        pop_geodata_key: str,
        plan_assignment_key: str,
        updaters: Optional[Dict] = None,
        ignored_updaters: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute partition updater values for a map (shapefile). Uses pop geodata
        and graph for the given keys; assigns the user map to geodata and
        builds a temporary partition for updater evaluation.
        """
        updaters = updaters or {}
        ignored_updaters = ignored_updaters or set()
        geodata = self._load_pop_geodata(pop_geodata_key)
        graph = self._build_graph(pop_geodata_key, plan_assignment_key)

        user_map = gpd.read_file(shapefile_path)
        if geodata.crs != user_map.crs:
            user_map = user_map.to_crs(geodata.crs)
        geodata_copy = geodata.copy()
        geodata_copy["user_assignment"] = maup.assign(geodata_copy, user_map)
        for node in graph.nodes:
            if node in geodata_copy.index:
                graph.nodes[node]["user_assignment"] = geodata_copy.loc[
                    node, "user_assignment"
                ]
        partition = GeographicPartition(
            graph,
            assignment="user_assignment",
            updaters=updaters,
        )
        data = {}
        for updater_name in updaters.keys():
            if updater_name in ignored_updaters:
                continue
            value = partition[updater_name]
            if isinstance(value, dict):
                data[updater_name] = {str(k): v for k, v in sorted(value.items())}
            else:
                data[updater_name] = value
        return data
