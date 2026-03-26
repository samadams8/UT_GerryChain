"""
Geography manager for redistricting runs.

Loads and harmonizes geodata, builds graphs, and provides graph/partition
interfaces for ConfigurationManager and EnsembleRunner. All geodata and
geometry-specific logic lives here; config and chain work only with graph
and partition.
"""
import random
from typing import Optional, Dict, Any, Set, List, Union
from warnings import warn

import geopandas as gpd
import numpy as np
import pandas as pd
import maup

from gerrychain import Graph, GeographicPartition, Partition
from gerrychain.constraints import contiguous

def repair_contiguity(
    partition: Partition,
    num_districts: int,
) -> Dict:
    """
    Repair non-contiguous districts by reassigning disconnected components
    to adjacent districts. Iterates until contiguity is achieved or
    max_iterations is reached.

    Parameters
    ----------
    partition : Partition
        Partition that may have non-contiguous districts.
    num_districts : int
        Number of districts (used to bound repair iterations).

    Returns
    -------
    dict
        Repaired assignment mapping node -> district.
    """
    import networkx as nx

    graph = partition.graph
    repaired_assignment = dict(partition.assignment)

    def _is_contiguous(assignment_dict):
        temp_partition = GeographicPartition(
            graph,
            assignment=assignment_dict,
            updaters={},
        )
        return contiguous(temp_partition)

    for iteration in range(2 * num_districts):
        if _is_contiguous(repaired_assignment):
            break

        # Handle unassigned nodes first
        unassigned_nodes = [
            node
            for node in graph.nodes
            if repaired_assignment[node] is None or pd.isna(repaired_assignment[node])
        ]

        for node in unassigned_nodes:
            neighbors = list(graph.neighbors(node))
            adjacent_districts = []
            for neighbor in neighbors:
                neighbor_dist = repaired_assignment.get(neighbor)
                if neighbor_dist is not None and not pd.isna(neighbor_dist):
                    adjacent_districts.append(neighbor_dist)

            if adjacent_districts:
                repaired_assignment[node] = max(
                    set(adjacent_districts), key=adjacent_districts.count
                )
            else:
                available_districts = [
                    d
                    for d in set(repaired_assignment.values())
                    if d is not None and not pd.isna(d)
                ]
                if available_districts:
                    repaired_assignment[node] = random.choice(available_districts)

        districts_to_check = set(repaired_assignment.values())
        districts_to_check.discard(None)

        repairs_made = False
        for district in districts_to_check:
            district_nodes = [
                node
                for node in graph.nodes
                if repaired_assignment[node] == district
            ]

            if not district_nodes:
                continue

            district_subgraph = graph.subgraph(district_nodes)
            components = list(nx.connected_components(district_subgraph))

            if len(components) <= 1:
                continue

            repairs_made = True
            components = sorted(components, key=len, reverse=True)

            for component in components[1:]:
                neighbor_districts = []
                for node in component:
                    for neighbor in graph.neighbors(node):
                        neighbor_dist = repaired_assignment.get(neighbor)
                        if (
                            neighbor_dist is not None
                            and neighbor_dist != district
                            and not pd.isna(neighbor_dist)
                        ):
                            neighbor_districts.append(neighbor_dist)

                if neighbor_districts:
                    target_district = max(
                        set(neighbor_districts), key=neighbor_districts.count
                    )
                else:
                    available_districts = [
                        d for d in districts_to_check if d != district
                    ]
                    if available_districts:
                        target_district = random.choice(available_districts)
                    else:
                        target_district = district

                for node in component:
                    repaired_assignment[node] = target_district

        if not repairs_made:
            break

    return repaired_assignment


def build_initial_partition(
    graph,
    assignment: Union[str, Dict],
    updaters: Dict,
    num_districts: int,
    repair: bool = True,
) -> Partition:
    """
    Build a GeographicPartition from a graph, assignment, and updaters,
    optionally repairing contiguity.

    Parameters
    ----------
    graph : gerrychain.Graph
        Graph built from geodata.
    assignment : str or dict
        Column name in graph nodes (e.g. "initial_plan") or assignment dict.
    updaters : dict
        Partition updaters (e.g. population Tally and others).
    num_districts : int
        Number of districts (used for contiguity repair iteration bound).
    repair : bool, optional
        If True (default), attempt to repair non-contiguous districts.

    Returns
    -------
    Partition
        GeographicPartition, repaired for contiguity when repair=True and needed.
    """
    part = GeographicPartition(
        graph,
        assignment=assignment,
        updaters=updaters,
    )

    if repair and not contiguous(part):
        repaired_assignment = repair_contiguity(part, num_districts)
        part = GeographicPartition(
            graph,
            assignment=repaired_assignment,
            updaters=updaters,
        )
        if not contiguous(part):
            warn(
                "Contiguity repair may not have fully resolved all issues. "
                "You should check your initial plan or population geodata for compatibility issues."
            )

    return part


class GeographyManager:
    """
    Manages population geographic data for redistricting simulations.

    Loads datasets by path, transforms to a common CRS, and provides
    column reporting, total columns, election columns, graph/partition
    building, and optional fill of empty ID columns.
    """

    def __init__(self, pop_data: Dict[str, str], crs: str = "EPSG:26912"):
        """
        Load each dataset in pop_data, transform to crs, and store by key.

        Parameters
        ----------
        pop_data : dict
            Mapping of dataset key to file path (e.g. GeoJSON or shapefile).
        crs : str, optional
            Target CRS for all loaded datasets (default "EPSG:26912").
        """
        self._pop_data = dict(pop_data)
        self._crs = crs
        self._datasets: Dict[str, gpd.GeoDataFrame] = {}
        for key, path in self._pop_data.items():
            self._datasets[key] = self._load_one(path)

    def _load_one(self, path: str) -> gpd.GeoDataFrame:
        """Load one file, transform to self._crs, add area if missing."""
        gdf = gpd.read_file(path)
        if self._crs:
            gdf = gdf.to_crs(self._crs)
        if "area" not in gdf.columns:
            gdf["area"] = gdf.geometry.area
        return gdf

    @property
    def pop_data(self) -> Dict[str, str]:
        """Read-only mapping of key -> path passed at init."""
        return dict(self._pop_data)

    @property
    def crs(self) -> str:
        """Read-only CRS used for loaded datasets."""
        return self._crs

    def get_pop_geodata(self, key: str) -> gpd.GeoDataFrame:
        """Return the loaded population GeoDataFrame for the given key."""
        if key not in self._datasets:
            raise KeyError(
                f"Pop dataset '{key}' not found. Available: {list(self._datasets.keys())}"
            )
        return self._datasets[key]

    def list_pop_keys(self) -> List[str]:
        """Return the list of population dataset keys."""
        return sorted(self._datasets.keys())

    def shared_columns(self) -> List[str]:
        """Return column names common to all loaded population datasets."""
        if not self._datasets:
            return []
        sets = [set(gdf.columns) for gdf in self._datasets.values()]
        common = set.intersection(*sets)
        return sorted(common)

    def columns_unique_to(self, key: str) -> List[str]:
        """Return column names that appear only in the dataset for this key."""
        if key not in self._datasets:
            raise KeyError(
                f"Pop dataset '{key}' not found. Available: {list(self._datasets.keys())}"
            )
        all_cols = set(self._datasets[key].columns)
        others = set()
        for k, gdf in self._datasets.items():
            if k != key:
                others |= set(gdf.columns)
        unique = all_cols - others
        return sorted(unique)

    def add_total_column(
        self,
        new_col_name: str,
        source_columns: List[str],
        key: str,
    ) -> None:
        """
        Add a new column as the sum of the given source columns for the given key.
        Operates in place on the loaded geodata.
        """
        gdf = self.get_pop_geodata(key)
        if new_col_name in gdf.columns:
            return
        available = [c for c in source_columns if c in gdf.columns]
        if not available:
            return
        gdf[new_col_name] = 0
        for col in available:
            try:
                gdf[new_col_name] = (
                    gdf[new_col_name].fillna(0) + gdf[col].fillna(0)
                )
            except Exception:
                gdf[new_col_name] = (
                    gdf[new_col_name].fillna(0)
                    + pd.to_numeric(gdf[col], errors="coerce").fillna(0)
                )

    def get_election_columns(
        self,
        key: str,
        years: Optional[List[int]] = None,
        offices: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Return election column names (G20USP format) for the key, optionally
        filtered by years and/or offices. Year from digits 1:3 as 20xx, office from 3:6.
        """
        years = years or []
        offices = offices or []
        gdf = self.get_pop_geodata(key)
        available = set(gdf.columns)
        out = []
        for col in available:
            if not isinstance(col, str) or not col.startswith("G"):
                continue
            try:
                year = int(col[1:3]) + 2000
                office = col[3:6]
                if (len(years) == 0 or year in years) and (
                    len(offices) == 0 or office in offices
                ):
                    out.append(col)
            except (ValueError, IndexError):
                pass
        return sorted(out)

    def get_graph(self, key: str) -> Graph:
        """Return a GerryChain Graph from the population geodata for the given key."""
        gdf = self.get_pop_geodata(key)
        return Graph.from_geodataframe(gdf)

    def build_partition(
        self,
        pop_key: str,
        plan: Union[str, gpd.GeoDataFrame],
        updaters: Optional[Dict] = None,
        repair_contiguity: bool = True,
    ) -> GeographicPartition:
        """
        Build a GeographicPartition from pop geodata and a caller-provided plan.
        Plan may be a file path or a GeoDataFrame. Optionally repairs contiguity (default True).
        """
        updaters = updaters or {}
        gdf = self.get_pop_geodata(pop_key)
        if isinstance(plan, str):
            plan_gdf = gpd.read_file(plan)
        else:
            plan_gdf = plan
        if gdf.crs != plan_gdf.crs:
            plan_gdf = plan_gdf.to_crs(gdf.crs)
        gdf_with_plan = gdf.copy()
        gdf_with_plan["PLAN_ASSIGNMENT"] = maup.assign(gdf_with_plan, plan_gdf)
        graph = Graph.from_geodataframe(gdf_with_plan)
        num_districts = len(plan_gdf)
        return build_initial_partition(
            graph,
            assignment="PLAN_ASSIGNMENT",
            updaters=updaters,
            num_districts=num_districts,
            repair=repair_contiguity,
        )

    def fill_empty_ids(self, key: str, columns: List[str]) -> None:
        """
        Assign unique IDs only to entries that are None or empty (blank/NaN).
        Rows that already have a value are unchanged, including duplicates.
        """
        gdf = self.get_pop_geodata(key)
        for col in columns:
            if col not in gdf.columns:
                continue
            series = gdf[col]
            empty = series.isna() | (series.astype(str).str.strip() == "")
            if not empty.any():
                continue
            existing = series[~empty]
            max_id = 0
            if len(existing) > 0:
                try:
                    max_id = int(pd.to_numeric(existing, errors="coerce").max())
                except (TypeError, ValueError):
                    pass
            need_id = empty
            n_need = need_id.sum()
            if n_need > 0:
                new_ids = np.arange(max_id + 1, max_id + 1 + int(n_need))
                gdf.loc[need_id, col] = new_ids

    def build_election_dicts(
        self,
        pop_key: str,
        years: List[int],
        offices: List[str],
        parties: List[str] = ["D", "R", "-"],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, str]]:
        """
        Build ``{election_name: {party_label: column_name}}`` dicts for
        ``ConfigurationManager.add_election_updaters()``.

        Uses ``get_election_columns()`` for column discovery, then groups
        columns by election and maps party initials to labels.

        Parameters
        ----------
        pop_key : str
            Key of the population dataset to inspect for election columns.
        years : list of int
            Years to include (e.g. ``[2016, 2020, 2024]``).
        offices : list of str
            Three-character office codes to include (e.g. ``["PRE", "GOV"]``).
        parties : list of str, optional
            Party labels to retain. Default ``["D", "R", "-"]``.
        overrides : dict, optional
            Mapping of ``election_name -> {party_label: column_name}`` to
            apply after auto-discovery. Override entries are merged (not
            replaced) into the discovered mapping.

        Returns
        -------
        dict
            ``{election_name: {party_label: column_name}}`` suitable for
            ``ConfigurationManager.add_election_updaters(elections=...)``.

        Notes
        -----
        Column naming convention: ``G{YY}{OFFICE}{PARTY_INITIAL}{LASTNAME}``
        Party initial at position 6: ``D`` → Democratic, ``R`` → Republican,
        anything else → ``"-"`` (other/third party).
        """
        overrides = overrides or {}
        columns = self.get_election_columns(pop_key, years=years, offices=offices)

        elections: Dict[str, Dict[str, str]] = {}

        for col in columns:
            if len(col) < 7:
                continue
            year = int(col[1:3]) + 2000
            office = col[3:6]
            election_name = f"{year}{office}"
            party_char = col[6]

            if party_char == "D":
                party_label = "D"
            elif party_char == "R":
                party_label = "R"
            else:
                party_label = "-"

            if party_label not in parties:
                continue

            if election_name not in elections:
                elections[election_name] = {}

            # Build cumulative label for multi-candidate parties (e.g. R1, R2)
            existing_for_party = [k for k in elections[election_name] if k.startswith(party_label)]
            label = f"{party_label}{len(existing_for_party) + 1}"
            elections[election_name][label] = col

        # Apply overrides (merge, not replace)
        for election_name, override_mapping in overrides.items():
            if election_name not in elections:
                elections[election_name] = {}
            elections[election_name].update(override_mapping)

        return elections


# class GeographyManager_legacy:
#     """
#     Manages multiple population and plan geodata datasets by key. Responsible
#     for loading, harmonization (CRS, unique IDs, area), and graph/partition
#     construction. ConfigurationManager and Runner use only graph/partition
#     from here.
#     """

#     def __init__(self, crs: str = "EPSG:26912"):
#         """
#         Parameters
#         ----------
#         crs : str, optional
#             Shared CRS for projection when loading datasets (e.g. "EPSG:26912").
#             Can be changed later via self.crs.
#         """
#         self.crs = crs
#         self._pop_datasets: Dict[str, _GeoDataEntry] = {}
#         self._plan_datasets: Dict[str, _GeoDataEntry] = {}

#     # --- Registration ---

#     def register_pop_datasets(self, pop_data_paths: Dict[str, str]) -> None:
#         """Register a dictionary of population geodata paths under the given keys."""
#         for key, path in pop_data_paths.items():
#             self.register_pop_dataset(key, path)

#     def register_plan_datasets(self, plan_data_paths: Dict[str, str]) -> None:
#         """Register a dictionary of plan geodata paths under the given keys."""
#         for key, path in plan_data_paths.items():
#             self.register_plan_dataset(key, path)

#     def register_pop_dataset(self, key: str, pop_geodata_path: str) -> None:
#         """Register a population geodata path under the given key."""
#         self._pop_datasets[key] = pop_geodata_path

#     def register_plan_dataset(self, key: str, plan_geodata_path: str) -> None:
#         """Register a plan (district boundaries) geodata path under the given key."""
#         self._plan_datasets[key] = plan_geodata_path

#     # --- Lazy loading (private) ---

#     def _load_pop_geodata(self, key: str) -> gpd.GeoDataFrame:
#         """Load and cache population geodata for key; project to self.crs, add area if missing."""
#         entry = self._pop_datasets.get(key)
#         if entry is None:
#             raise KeyError(f"Pop dataset '{key}' not found. Available: {list(self._pop_datasets.keys())}")
#         if isinstance(entry, gpd.GeoDataFrame):
#             return entry
#         path = entry
#         gdf = gpd.read_file(path)
#         if self.crs:
#             gdf = gdf.to_crs(self.crs)
#         if "area" not in gdf.columns:
#             gdf["area"] = gdf.geometry.area
#         self._pop_datasets[key] = gdf
#         return gdf

#     def _load_plan_geodata(self, key: str) -> gpd.GeoDataFrame:
#         """Load and cache plan geodata for key; project to self.crs, add area if missing."""
#         entry = self._plan_datasets.get(key)
#         if entry is None:
#             raise KeyError(f"Plan dataset '{key}' not found. Available: {list(self._plan_datasets.keys())}")
#         if isinstance(entry, gpd.GeoDataFrame):
#             return entry
#         path = entry
#         gdf = gpd.read_file(path)
#         if self.crs:
#             gdf = gdf.to_crs(self.crs)
#         if "area" not in gdf.columns:
#             gdf["area"] = gdf.geometry.area
#         self._plan_datasets[key] = gdf
#         return gdf

#     def _ensure_unique_ids(self, geodata: gpd.GeoDataFrame, cols: List[str]) -> None:
#         """Ensure each column in cols has unique, non-empty values (in-place)."""
#         for col in cols:
#             if col not in geodata.columns:
#                 continue
#             series = geodata[col]
#             empty = series.isna() | (series.astype(str).str.strip() == "")
#             if not empty.any():
#                 # Check duplicates
#                 if series.nunique() == len(series):
#                     continue
#             # Assign new unique values where empty or duplicated
#             existing = series[~empty]
#             if len(existing) > 0:
#                 try:
#                     max_id = int(pd.to_numeric(existing, errors="coerce").max())
#                 except (TypeError, ValueError):
#                     max_id = 0
#             else:
#                 max_id = 0
#             need_id = empty | series.duplicated(keep="first")
#             n_need = need_id.sum()
#             if n_need > 0:
#                 new_ids = np.arange(max_id + 1, max_id + 1 + n_need)
#                 geodata.loc[need_id, col] = new_ids

#     def _build_graph(
#         self,
#         pop_geodata_key: str,
#         plan_assignment_key: str,
#         unique_id_cols: Optional[List[str]] = None,
#     ) -> Graph:
#         """Build graph from pop geodata with plan assigned; no caching."""
#         pop_geodata = self._load_pop_geodata(pop_geodata_key).copy()
#         plan_geodata = self._load_plan_geodata(plan_assignment_key)

#         assignment_col = _assignment_column_name(
#             plan_assignment_key, set(pop_geodata.columns)
#         )
#         pop_geodata[assignment_col] = maup.assign(pop_geodata, plan_geodata)

#         if unique_id_cols:
#             self._ensure_unique_ids(pop_geodata, unique_id_cols)

#         return Graph.from_geodataframe(pop_geodata)

#     # --- Public accessors ---

#     def get_pop_columns_for_key(self, key: str) -> List[str]:
#         """Return the columns of the population geodata for the given key."""
#         return list(self._load_pop_geodata(key).columns)

#     def get_plan_columns_for_key(self, key: str) -> List[str]:
#         """Return the columns of the plan geodata for the given key."""
#         return list(self._load_plan_geodata(key).columns)

#     def get_pop_geodata(self, key: str) -> gpd.GeoDataFrame:
#         """Return population geodata for the given key (loaded lazily)."""
#         return self._load_pop_geodata(key)

#     def get_plan_geodata(self, key: str) -> gpd.GeoDataFrame:
#         """Return plan geodata for the given key (loaded lazily)."""
#         return self._load_plan_geodata(key)

#     def get_graph(
#         self,
#         pop_geodata_key: str,
#         plan_assignment_key: str,
#         unique_id_cols: Optional[List[str]] = None,
#     ) -> Graph:
#         """Build and return graph for the given pop and plan keys (no cache)."""
#         return self._build_graph(pop_geodata_key, plan_assignment_key, unique_id_cols)

#     def build_partition(
#         self,
#         pop_geodata_key: str,
#         plan_assignment_key: str,
#         updaters: Optional[Dict] = None,
#         repair_contiguity: bool = True,
#         unique_id_cols: Optional[List[str]] = None,
#     ) -> GeographicPartition:
#         """
#         Build a GeographicPartition from the given pop and plan keys.
#         Optionally repairs contiguity by default.
#         """
#         updaters = updaters or {}
#         pop_geodata = self._load_pop_geodata(pop_geodata_key)
#         assignment_col = _assignment_column_name(
#             plan_assignment_key, set(pop_geodata.columns)
#         )
#         graph = self._build_graph(
#             pop_geodata_key, plan_assignment_key, unique_id_cols
#         )
#         plan_gdf = self._load_plan_geodata(plan_assignment_key)
#         num_districts = len(plan_gdf)

#         return build_initial_partition(
#             graph,
#             assignment=assignment_col,
#             updaters=updaters,
#             num_districts=num_districts,
#             repair=repair_contiguity,
#         )

#     def list_pop_keys(self) -> List[str]:
#         """Return list of all registered population dataset keys."""
#         return sorted(set(self._pop_datasets))

#     def list_plan_keys(self) -> List[str]:
#         """Return list of all registered plan dataset keys."""
#         return sorted(set(self._plan_datasets))

#     def get_geography_config(
#         self, default_dataset: Optional[str] = None
#     ) -> Dict[str, Any]:
#         """
#         Return a dict suitable for the 'geography' section of run metadata.
#         Includes crs, datasets (key -> pop_geodata_path, plan_geodata_path),
#         and optional default_dataset.
#         """
#         all_keys = set(self._pop_datasets) | set(self._plan_datasets)
#         datasets: Dict[str, Dict[str, Optional[str]]] = {}
#         for key in sorted(all_keys):
#             pop_entry = self._pop_datasets.get(key)
#             plan_entry = self._plan_datasets.get(key)
#             pop_path = pop_entry if isinstance(pop_entry, str) else None
#             plan_path = plan_entry if isinstance(plan_entry, str) else None
#             datasets[key] = {
#                 "pop_geodata_path": pop_path,
#                 "plan_geodata_path": plan_path,
#             }
#         out: Dict[str, Any] = {"crs": self.crs, "datasets": datasets}
#         if default_dataset is not None:
#             out["default_dataset"] = default_dataset
#         return out

#     def make_total_column(
#         self,
#         total_col: str,
#         all_election_columns: List[str],
#         key: str,
#     ) -> None:
#         """
#         Add a total-vote column to the population geodata for the given key.
#         Operates in place on the cached geodata; the next graph build will
#         include this column.
#         """
#         geodata = self._load_pop_geodata(key)
#         if total_col in geodata.columns:
#             return
#         party_cols = [c for c in all_election_columns if c in geodata.columns]
#         if party_cols:
#             geodata[total_col] = 0
#             for col in party_cols:
#                 try:
#                     geodata[total_col] = (
#                         geodata[total_col].fillna(0) + geodata[col].fillna(0)
#                     )
#                 except Exception:
#                     geodata[total_col] = (
#                         geodata[total_col].fillna(0)
#                         + pd.to_numeric(geodata[col], errors="coerce").fillna(0)
#                     )

#     def get_election_columns_for_key(
#         self,
#         key: str,
#         years: List[int] = [],
#         offices: List[str] = [],
#     ) -> List[str]:
#         """
#         Return election column names present in the population geodata for key
#         that match the given years and offices (e.g. G20USP format). If no years (or offices) are provided, returns election columns that match all available years (or offices).
#         """
#         geodata = self._load_pop_geodata(key)
#         available_columns = set(geodata.columns)
#         available_elections = []
#         for column in available_columns:
#             if isinstance(column, str) and column.startswith("G"):
#                 try:
#                     year = int(column[1:3]) + 2000
#                     office = column[3:6]
#                     if (
#                         (len(years) == 0 or year in years)
#                         and (len(offices) == 0 or office in offices)
#                     ):
#                         available_elections.append(column)
#                 except ValueError:
#                     pass
#         return sorted(available_elections)

#     def compute_metrics_for_map(
#         self,
#         shapefile_path: str,
#         pop_geodata_key: str,
#         plan_assignment_key: str,
#         updaters: Optional[Dict] = None,
#         ignored_updaters: Optional[Set[str]] = None,
#     ) -> Dict[str, Any]:
#         """
#         Compute partition updater values for a map (shapefile). Uses pop geodata
#         and graph for the given keys; assigns the user map to geodata and
#         builds a temporary partition for updater evaluation.
#         """
#         updaters = updaters or {}
#         ignored_updaters = ignored_updaters or set()
#         geodata = self._load_pop_geodata(pop_geodata_key)
#         graph = self._build_graph(pop_geodata_key, plan_assignment_key)

#         user_map = gpd.read_file(shapefile_path)
#         if geodata.crs != user_map.crs:
#             user_map = user_map.to_crs(geodata.crs)
#         geodata_copy = geodata.copy()
#         geodata_copy["user_assignment"] = maup.assign(geodata_copy, user_map)
#         for node in graph.nodes:
#             if node in geodata_copy.index:
#                 graph.nodes[node]["user_assignment"] = geodata_copy.loc[
#                     node, "user_assignment"
#                 ]
#         partition = GeographicPartition(
#             graph,
#             assignment="user_assignment",
#             updaters=updaters,
#         )
#         data = {}
#         for updater_name in updaters.keys():
#             if updater_name in ignored_updaters:
#                 continue
#             value = partition[updater_name]
#             if isinstance(value, dict):
#                 data[updater_name] = {str(k): v for k, v in sorted(value.items())}
#             else:
#                 data[updater_name] = value
#         return data
