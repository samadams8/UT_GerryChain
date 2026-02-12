import os
import random
import yaml
import json
import math
from typing import Optional, Dict, Any, List, Callable, Tuple, Literal, Union
from warnings import warn
from functools import partial
from datetime import datetime
from statistics import mean

import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np

import maup
from gerrychain import Graph, GeographicPartition, Partition, updaters
from gerrychain.constraints import contiguous, UpperBound
from gerrychain.metrics import polsby_popper
from gerrychain.proposals import recom
from gerrychain.tree import bipartition_tree
from gerrychain.updaters.locality_split_scores import LocalitySplits
from gerrychain.optimization import SingleMetricOptimizer

import utgc.metrics as utmetrics
from . import run_utils as rutil
from .optimization import LexicographicOptimizer, OptimizationMetric

class ConfigurationManager:
    def __init__(
        self,
        pop_geodata_path: str,
        initial_plan_path: str,
        random_seed: Optional[int] = None,
        pop_column: Optional[str] = "TOTPOP",
    ):
        self.init_params = {}
        if not pop_geodata_path:
            raise ValueError("Population geodata path must be provided.")
        if not initial_plan_path:
            raise ValueError("Initial plan path must be provided.")
        self.init_params["pop_geodata_path"] = pop_geodata_path
        self.init_params["initial_plan_path"] = initial_plan_path
        
        if random_seed:
            self.init_params["random_seed"] = random_seed
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.geodata, self.initial_plan = self._load_geodata(pop_geodata_path, initial_plan_path)
        self.graph = Graph.from_geodataframe(self.geodata)

        total_population = sum(self.geodata[pop_column])
        num_districts = len(self.initial_plan)

        print(f"  Graph built with {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")

        self.population_params = {
            "column_id": pop_column,
            "total_pop": total_population,
            "num_districts": num_districts,
            "ideal_pop": total_population / num_districts,
            "pop_tolerance": 0.01,
        }

        # Record which configuration methods were used
        self.construction_history = []

        self.region_surcharges = {}
        self.edge_penalty_params = {}
        self.edge_penalties = {}

        self.constraint_params = {}
        self.constraints = [contiguous,]

        self.optimization_scheme_params = {}
        self.ignored_updaters = set()
        
        self.lex_metrics = []
        self.lex_burst_lengths = []
        self.lex_num_bursts = []
        self.lex_preoptimization_limit = 0

        self.updaters = {
            "population": updaters.Tally(pop_column, alias="population"),
        }

    # Properties
    @property
    def pop_tolerance(self) -> float:
        return self.population_params.get("pop_tolerance", 0.01)
    
    @property
    def initial_partition(self) -> Partition:
        tmp = GeographicPartition(
            self.graph, assignment="initial_plan", updaters=self.updaters
        )

        if not contiguous(tmp):
            repaired_assignment = self._repair_contiguity(tmp)
            
            # Create a new partition with the repaired assignment
            tmp = GeographicPartition(
                self.graph,
                assignment=repaired_assignment,
                updaters=self.updaters
            )

        if not contiguous(tmp):
            warn("Contiguity repair may not have fully resolved all issues. You should check your initial plan or population geodata for compatibility issues.")
        
        return tmp

    # --- Geodata and Graph Utilities ---

    def _load_geodata(self,
        pop_geodata_path: str,
        initial_plan_path: str,
        crs: Optional[str] = "EPSG:26912",
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
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
            print("Found %d nodes assigned to %d incorporated municipalities" % (
                (geodata["MUNIID"] != "").sum(),
                len(set(geodata[geodata["MUNIID"] != ""]["MUNIID"]))
            ))
            print("Assigning unique IDs to unincorporated nodes...")
            
            # Get existing numeric MUNIIDs
            existing_muniids = geodata[geodata["MUNIID"] != ""]["MUNIID"]
            if len(existing_muniids) > 0:
                max_id = int(existing_muniids.astype(int).max())
            else:
                max_id = 0

            # Generate unique sequential IDs for unincorporated areas
            unincorporated_mask = geodata["MUNIID"] == ""
            unincorporated_count = unincorporated_mask.sum()
            if unincorporated_count > 0:
                geodata.loc[unincorporated_mask, "MUNIID"] = np.arange(max_id + 1, max_id + 1 + unincorporated_count)
                print(f"Assigned unique IDs to {unincorporated_count} unincorporated nodes")
            
            # Print final municipality count
            num_unique_munis = len(set(geodata["MUNIID"]))
            print(f"Total unique MUNIIDs: {num_unique_munis}")

        # Assign initial plan to geodata
        geodata["initial_plan"] = maup.assign(geodata, initial_plan)
        if "area" not in geodata.columns:
            geodata["area"] = geodata.geometry.area

        return geodata, initial_plan

    def _repair_contiguity(self, partition: Partition) -> Dict:
        """
        Repair non-contiguous districts by reassigning disconnected components
        to adjacent districts. This function iterates until contiguity is achieved
        or max_iterations is reached.
        """
        graph = partition.graph
        repaired_assignment = dict(partition.assignment)
        
        # Create a temporary partition to check contiguity
        def _is_contiguous(assignment_dict):
            temp_partition = GeographicPartition(
                graph,
                assignment=assignment_dict,
                updaters={}
            )
            return contiguous(temp_partition)
        
        for iteration in range(2*self.population_params["num_districts"]):
            if _is_contiguous(repaired_assignment):
                break
            
            # Handle unassigned nodes first
            unassigned_nodes = [
                node for node in graph.nodes 
                if repaired_assignment[node] is None or pd.isna(repaired_assignment[node])
            ]
            
            for node in unassigned_nodes:
                # Find adjacent districts
                neighbors = list(graph.neighbors(node))
                adjacent_districts = []
                for neighbor in neighbors:
                    neighbor_dist = repaired_assignment.get(neighbor)
                    if neighbor_dist is not None and not pd.isna(neighbor_dist):
                        adjacent_districts.append(neighbor_dist)
                
                if adjacent_districts:
                    # Assign to the most common adjacent district
                    repaired_assignment[node] = max(set(adjacent_districts), 
                                                   key=adjacent_districts.count)
                else:
                    # No adjacent districts found, assign to first available district
                    available_districts = [d for d in set(repaired_assignment.values()) 
                                         if d is not None and not pd.isna(d)]
                    if available_districts:
                        repaired_assignment[node] = random.choice(available_districts)
            
            # Now repair non-contiguous districts
            districts_to_check = set(repaired_assignment.values())
            districts_to_check.discard(None)
            
            repairs_made = False
            for district in districts_to_check:
                # Get all nodes in this district
                district_nodes = [node for node in graph.nodes 
                                if repaired_assignment[node] == district]
                
                if not district_nodes:
                    continue
                    
                # Find connected components within this district
                district_subgraph = graph.subgraph(district_nodes)
                components = list(nx.connected_components(district_subgraph))
                
                if len(components) <= 1:
                    # District is already contiguous
                    continue
                
                repairs_made = True
                # Sort components by size (largest first)
                components = sorted(components, key=len, reverse=True)
                
                # Keep the largest component in the original district
                # Reassign smaller components to adjacent districts
                for component in components[1:]:
                    # Find adjacent districts for this component
                    neighbor_districts = []
                    for node in component:
                        neighbors = list(graph.neighbors(node))
                        for neighbor in neighbors:
                            neighbor_dist = repaired_assignment.get(neighbor)
                            if (neighbor_dist is not None and 
                                neighbor_dist != district and 
                                not pd.isna(neighbor_dist)):
                                neighbor_districts.append(neighbor_dist)
                    
                    if neighbor_districts:
                        # Use most common adjacent district
                        target_district = max(set(neighbor_districts), 
                                            key=neighbor_districts.count)
                    else:
                        # No adjacent districts found, try to find any available district
                        available_districts = [d for d in districts_to_check if d != district]
                        if available_districts:
                            target_district = random.choice(available_districts)
                        else:
                            # Fallback: keep in original district (shouldn't happen)
                            target_district = district
                    
                    # Reassign all nodes in this component
                    for node in component:
                        repaired_assignment[node] = target_district
            
            # If no repairs were made, break to avoid infinite loop
            if not repairs_made:
                break
        
        return repaired_assignment
    
    # --- Configuration Methods ---

    # Population
    def set_pop_dev_tolerance(self, tolerance: float = 0.01) -> 'ConfigurationManager':
        self.construction_history.append({
            "method": "set_pop_dev_tolerance",
            "kwargs": { "tolerance": tolerance }
        })

        self.population_params["pop_tolerance"] = tolerance
        print(f"Population deviation tolerance: {tolerance:%}")
        return self

    def add_pop_dev_updater(self, name: str = "pop_dev", ignore_output: bool = True) -> 'ConfigurationManager':
        self.construction_history.append({
            "method": "add_pop_dev_updater",
            "kwargs": { "name": name, "ignore_output": ignore_output }
        })

        ideal_pop = self.population_params["ideal_pop"]

        self.updaters[name] = lambda p, target=ideal_pop: {
            k: v - target for k, v in p["population"].items()
        }
        print(f"  Added population deviation updater: '{name}'")

        if ignore_output: 
            self.ignore_output(name)

        return self

    # Constraints
    def constrain_region_splits(self,
        name: Optional[str] = None,
        column_id: Optional[str] = None,
        num_split: Optional[int] = None,
        num_multi_splits: Optional[int] = None,
        create_updater: bool = True
    ) -> 'ConfigurationManager':
        self.construction_history.append({
            "method": "constrain_region_splits",
            "kwargs": {
                "name": name,
                "column_id": column_id,
                "num_split": num_split,
                "num_multi_splits": num_multi_splits,
                "create_updater": False, # Captured separately
            }
        })

        if not name:
            name = column_id or "unknown_region"

        if num_split is not None:
            constraint_name = f"split_{name}"
            self.constraints.append(UpperBound(
                    lambda p, name=constraint_name: p[name],
                    num_split
                ))
            self.constraint_params[constraint_name] = num_split
            print(f"Constraint: split max {num_split} {name}")
        if num_multi_splits is not None:
            constraint_name = f"{name}_multi_splits"
            self.constraints.append(UpperBound(
                    lambda p, name=constraint_name: p[name],
                    num_multi_splits
                ))
            self.constraint_params[constraint_name] = num_multi_splits
            print(f"Constraint: max {num_multi_splits} multi-splits of {name}")

        if create_updater:
            # Need strict=False or similar potentially if already exists?
            # actually add_locality_splits_updater checks if updaters exist usually or overwrites.
            self.add_locality_splits_updater(name=name, column_id=column_id)

        return self

    def constrain_not_equal(self,
        not_equal_constraint: bool = True,
        create_updater: bool = True,
        ignore_output: bool = True,
    ) -> 'ConfigurationManager':
        self.construction_history.append({
            "method": "constrain_not_equal",
            "kwargs": {
                "not_equal_constraint": not_equal_constraint,
                "create_updater": create_updater,
                "ignore_output": ignore_output
            }
        })

        self.constraint_params["not_equal_constraint"] = not_equal_constraint

        # Remove existing NotEqual if present (to avoid dups or reset)
        self.constraints = [c for c in self.constraints if not isinstance(c, rutil.NotEqual)]

        if not_equal_constraint:
            self.constraints.append(rutil.NotEqual())
            print(f"Constraint: prevent same map from being generated twice in a row")

            if create_updater and "assignment_hash" not in self.updaters:
                self.updaters["assignment_hash"] = rutil._assignment_hash
                print(f"  Added 'assignment_hash' updater")
                if ignore_output:
                    self.ignore_output("assignment_hash")
        else:
            print(f"Constraint: allow same map to be generated twice in a row")
        
        return self

    # Region surcharges
    def surcharge_region(self,
        column_id: str,
        surcharge: float,
    ) -> 'ConfigurationManager':
        if surcharge <= 0:
            return self

        self.construction_history.append({
            "method": "surcharge_region",
            "kwargs": {
                "column_id": column_id,
                "surcharge": surcharge
            }
        })

        self.region_surcharges[column_id] = surcharge
        print(f"Surcharge: {surcharge} for {column_id}")

        return self

    # Edge penalties
    def penalize_edges_from_csv(self,
        csv_path: str,
        penalty: float,
        weight_column: str = "w",
    ) -> 'ConfigurationManager':
        if not os.path.exists(csv_path):
            warn(f"Transitability edge file not found: {csv_path}. Skipping edge penalties.")
            return self
        if penalty <= 0:
            return self
        
        self.construction_history.append({
            "method": "penalize_edges_from_csv",
            "kwargs": {
                "csv_path": csv_path,
                "penalty": penalty,
                "weight_column": weight_column,
            },
        })

        edges_df = pd.read_csv(csv_path)
        
        # Check if the weight column exists
        has_weights = weight_column in edges_df.columns
        if not has_weights:
            warn(f"Weight column '{weight_column}' not found in {os.path.basename(csv_path)}. Using constant penalty for all edges in file.")

        for _, row in edges_df.iterrows():
            edge = tuple(sorted((int(row["u"]), int(row["v"]))))
            
            if has_weights:
                edge_weight = row[weight_column] * penalty
            else:
                edge_weight = penalty

            if edge in self.edge_penalties:
                self.edge_penalties[edge] += edge_weight
            else:
                self.edge_penalties[edge] = edge_weight

        self.edge_penalty_params[csv_path] = penalty

        print(f"Penalizing edges from {os.path.basename(csv_path)} with factor {penalty}")

        return self

    # Updaters
    def add_locality_splits_updater(self,
        name: Optional[str] = None,
        column_id: str = "",
        ignore_ls_output: bool = True,
    ) -> 'ConfigurationManager':
        self.construction_history.append({
            "method": "add_locality_splits_updater",
            "kwargs": {
                "name": name,
                "column_id": column_id,
                "ignore_ls_output": ignore_ls_output
            },
        })

        if not name:
            # Fallback if column_id is somehow also empty? Though type hint says str.
            name = column_id or "unknown"
            
        ls_name = f"ls_{name}"
        self.updaters[ls_name] = LocalitySplits(
            name=ls_name,
            col_id=column_id,
            pop_col=self.population_params["column_id"],
            scores_to_compute=["num_split_localities", "num_parts"],
        )
        print(f"  Added locality split updater: '{ls_name}'")
        if ignore_ls_output:
            self.ignore_output(ls_name)

        # Count number of localities
        num_localities = len(set(dict(self.graph.nodes(data=column_id)).values()))

        sname = f"split_{name}"
        self.updaters[sname] = \
            lambda p, ls=ls_name: p[ls].get("num_split_localities", 0)
        print(f"  Added split updater: '{sname}'")
        
        msname = f"{name}_multi_splits"
        self.updaters[msname] = \
            lambda p, ls=ls_name, sn=sname, nl=num_localities: p[ls].get("num_parts", 0) - p[sn] - nl
        print(f"  Added multi-split updater: '{msname}'")

        return self

    def make_total_column(
        self,
        total_col: str,
        all_election_columns: List[str],
    ) -> 'ConfigurationManager':
        if total_col not in self.geodata.columns:
            party_cols = [col for col in all_election_columns if col in self.geodata.columns]
            
            if party_cols:
                self.geodata[total_col] = 0
                for col in party_cols:
                    try:
                        self.geodata[total_col] = self.geodata[total_col].fillna(0) + self.geodata[col].fillna(0)
                    except Exception:
                        self.geodata[total_col] = self.geodata[total_col].fillna(0) + pd.to_numeric(self.geodata[col], errors='coerce').fillna(0)
            
            if total_col in self.geodata.columns:
                for node in self.graph.nodes:
                    if node in self.geodata.index:
                        self.graph.nodes[node][total_col] = self.geodata.loc[node, total_col]
        
        return self

    def available_election_columns(self, years: List[int], offices: List[str]) -> List[str]:
        available_columns = set(self.geodata.columns)
        available_elections = []

        for column in available_columns:
            if element_is_election_column := (isinstance(column, str) and column.startswith("G")):
                try:
                    year = int(column[1:3]) + 2000
                    office = column[3:6]
                    if year in years and office in offices:
                        available_elections.append(column)
                except ValueError:
                    pass

        return sorted(available_elections)

    def add_election_updater(
        self,
        name: str,
        parties_to_columns: Dict[str, str],
        ignore_output: bool = True,
    ) -> 'ConfigurationManager':
        self.construction_history.append({
            "method": "add_election_updater",
            "kwargs": {
                "name": name,
                "parties_to_columns": parties_to_columns,
                "ignore_output": ignore_output
            },
        })

        self.updaters[name] = updaters.Election(
            name=name,
            parties_to_columns=parties_to_columns,
        )
        print(f"  Added election updater: '{name}'")
        print(f"    Parties to columns: {parties_to_columns}")
        if ignore_output:
            self.ignore_output(name)

        for column in parties_to_columns.values():
            if column not in self.geodata.columns:
                print(f"  WARNING: Column '{column}' not found in geodata. Please check the column names and try again.")

        return self

    def add_election_updaters(
        self,
        years: Optional[List[int]] = None,
        elections: Optional[List[str]] = None,
        parties: Optional[List[str]] = ['R', 'D', '-'],
        parties_to_columns_override: Optional[Dict[str, Dict[str, str]]] = {},
        skip_if_missing_parties: bool = True,
        ignore_output: bool = True,
    ) -> 'ConfigurationManager':
        # Logic matches Runner.add_election_updaters exactly
        columns = []
        for c in self.geodata.columns:
            if isinstance(c, str) and len(c) >= 3 and c.startswith("G") and c[1:3].isdigit():
                columns.append(c)

        current_year = datetime.now().year
        def two_digit_to_year(yy_str: str) -> int:
            yy = int(yy_str)
            candidate = 2000 + yy
            if candidate <= current_year:
                return candidate
            return 1900 + yy

        discovered_years = sorted({two_digit_to_year(c[1:3]) for c in columns if len(c) >= 3 and c[1:3].isdigit()})
        discovered_offices = sorted({c[3:6] for c in columns if len(c) >= 6})

        years = years or discovered_years
        elections = elections or discovered_offices

        election_to_cols: Dict[str, List[str]] = {}
        for col in columns:
            try:
                yy_str = col[1:3]
                if len(col) < 6:
                    continue
                office = col[3:6]
                if not (yy_str.isdigit() and len(office) == 3):
                    continue
                year = two_digit_to_year(yy_str)
            except Exception:
                continue

            if (year in years) and (office in elections):
                ename = f"{year:04d}{office}"
                election_to_cols.setdefault(ename, []).append(col)

        for ename, cols in sorted(election_to_cols.items()):
            year = int(ename[:4])
            office = ename[4:7]
            yy = f"{year % 100:02d}"

            party_initial_to_cols: Dict[str, List[str]] = {}
            for c in cols:
                if len(c) < 7:
                    continue
                # Handle cases where party might be missing or different
                party_initial = c[6]
                party_initial_to_cols.setdefault(party_initial, []).append(c)

            parties_to_columns: Dict[str, str] = {}
            for initial, plist in party_initial_to_cols.items():
                if len(plist) == 1:
                    parties_to_columns[initial] = plist[0]
                else:
                    for i, pc in enumerate(sorted(plist)):
                        parties_to_columns[f"{initial}{i+1}"] = pc

            total_col = f"G{yy}{office}-TOT"
            parties_to_columns["-"] = total_col

            if ename in parties_to_columns_override:
                override_columns = set(parties_to_columns_override[ename].values())
                keys_to_remove = [
                    key for key, col in parties_to_columns.items()
                    if col in override_columns and key != "-"
                ]
                for key in keys_to_remove:
                    del parties_to_columns[key]
                
                for k, v in parties_to_columns_override[ename].items():
                    parties_to_columns[k] = v

            if parties is not None:
                filtered_parties_to_columns: Dict[str, str] = {}
                for party_key, col_name in parties_to_columns.items():
                    if party_key == "-":
                        if "-" in parties:
                            filtered_parties_to_columns[party_key] = col_name
                    else:
                        for party_init in parties:
                            if party_key.startswith(party_init):
                                filtered_parties_to_columns[party_key] = col_name
                                break
                parties_to_columns = filtered_parties_to_columns

            if "-" in parties_to_columns:
                total_col = parties_to_columns["-"]
                self.make_total_column(total_col, cols)

            if (
                not skip_if_missing_parties
                or all( (
                    p in parties_to_columns.keys()
                    or any( c.startswith(p) for c in parties_to_columns.keys() )
                    ) for p in parties
                )
            ):
                self.add_election_updater(
                    name=ename,
                    parties_to_columns=parties_to_columns,
                    ignore_output=ignore_output,
                )

        return self

    def add_election_aggregator(self,
        name: str,
        elections: List[str],
        parties: List[str] = ["D", "R", "-"],
        ignore_table_output: bool = True,
        ignore_agg_output: bool = True,
    ) -> 'ConfigurationManager':
        self.construction_history.append({
            "method": "add_election_aggregator",
            "kwargs": {
                "name": name,
                "elections": elections,
                "parties": parties,
                "ignore_table_output": ignore_table_output,
                "ignore_agg_output": ignore_agg_output
            },
        })

        self.updaters[f"{name}_table"] = lambda p: utmetrics.tabulate_partisan_data(p, elections, parties)
        print(f"  Added partisan data tabulator: '{name}_table'")

        if ignore_table_output:
            self.ignore_output(f"{name}_table")

        self.updaters[name] = lambda p: utmetrics.aggregate_partisan_metrics(p[f"{name}_table"])
        print(f"  Added partisan data aggregator: '{name}'")

        if ignore_agg_output:
            self.ignore_output(name)

        return self

    def add_election_metric_updaters(self,
        aggregator_name: str,
        metrics: List[str],
        prepend_agg_name: bool = False,
    ) -> 'ConfigurationManager':
        self.construction_history.append({
            "method": "add_election_metric_updaters",
            "kwargs": {
                "aggregator_name": aggregator_name,
                "metrics": metrics,
                "prepend_agg_name": prepend_agg_name
            },
        })

        if aggregator_name not in self.updaters:
            warn(f"Election metric updater requires election data aggregator '{aggregator_name}' to be added, but it was not found.")

        for metric in metrics:
            mname = f"{aggregator_name}_{metric}" if prepend_agg_name else metric

            if metric == "partisan_bias_utah":
                self.updaters[mname] = lambda p, an=aggregator_name: utmetrics.partisan_bias_utah(p[an])
            elif metric == "partisan_bias":
                self.updaters[mname] = lambda p, an=aggregator_name: utmetrics.partisan_bias(p[an])
            elif metric == "mean_median":
                self.updaters[mname] = lambda p, an=aggregator_name: utmetrics.mean_median(p[an])
            elif metric == "efficiency_gap":
                self.updaters[mname] = lambda p, an=aggregator_name: utmetrics.efficiency_gap(p[an])
            elif metric == "stdev_partisan_share":
                self.updaters[mname] = lambda p, an=aggregator_name: utmetrics.stdev_partisan_share(p[an])
            elif metric == "majority_partisan_shares":
                self.updaters[mname] = lambda p, an=aggregator_name: utmetrics.majority_partisan_shares(p[an])
            elif metric == "majority_seats":
                self.updaters[mname] = lambda p, an=aggregator_name: utmetrics.majority_seats(p[an])
            else:
                raise ValueError(f"Unknown election metric: '{metric}'")
            print(f"  Added election metric updater: '{mname}'")

        return self

    def add_updater_function(self,
        name: str,
        function: Callable[[Partition], Any]
    ) -> 'ConfigurationManager':
        if isinstance(function, str):
            print(f"!!! Updater function '{name} ({function})' is a string. Please use .add_updater_function() to add this function manually.")
            return self

        self.construction_history.append({
            "method": "add_updater_function",
            "kwargs": {
                "name": name,
                "function": function.__name__,
            },
        })

        return self._add_updater_function(name=name, function=function)

    def _add_updater_function(self,
        name: str,
        function: Callable[[Partition], Any],
        ignore_output: bool = False,
    ) -> 'ConfigurationManager':
        if name in self.updaters:
            print(f"Updater function '{name}' already exists. Overwriting...")
        
        self.updaters[name] = function
        print(f"  Added updater function: '{name}'")

        if ignore_output: 
            self.ignore_output(name)

        return self

    def ignore_output(self, name: str) -> 'ConfigurationManager':
        self.ignored_updaters.add(name)
        print(f"    Ignoring updater in output: '{name}'")
        return self

    # Shape metrics
    def add_shape_metrics(self, metrics: List[str]) -> 'ConfigurationManager':
        self.construction_history.append({
            "method": "add_shape_metrics",
            "kwargs": {
                "metrics": metrics,
            }
        })

        for metric in metrics:
            if metric not in self.updaters:
                if metric == "polsby_popper":
                    if "perimeter" not in self.updaters:
                        self._add_updater_function("perimeter", updaters.perimeter, ignore_output=True)
                    if "area" not in self.updaters:
                         self._add_updater_function("area", updaters.Tally("area", alias="area"), ignore_output=True)
                    if "polsby_popper" not in self.updaters:
                        self._add_updater_function("polsby_popper", polsby_popper)
                elif metric == "reock_score":
                    self._add_updater_function("reock_score", rutil._reock_score)
                    print(f"  Added Reock score shape metric")
                else:
                    raise ValueError(f"Unknown shape metric: '{metric}'")

        return self

    # Optimization
    def add_optimization_scheme(self,
        scheme: str,
        updater: str,
        **kwargs
    ) -> 'ConfigurationManager':
        self.construction_history.append({
            "method": "add_optimization_scheme",
            "kwargs": {
                "scheme": scheme,
                "updater": updater,
                **kwargs
            }
        })

        if scheme not in ["neutral", "tilted", "short_bursts"]:
            raise ValueError(f"Unknown optimization scheme: '{scheme}'. Options: 'neutral', 'tilted', 'short_bursts'")

        if updater not in self.updaters:
            raise ValueError(f"Updater '{updater}' not found. Must be added to the runner first.")

        self.optimization_scheme_params = {
            "scheme": scheme,
            "updater": updater,
            **kwargs
        }
        print(f"Optimization scheme: {scheme}")
        if kwargs:
            print(f"  Parameters: {kwargs}")

        return self

    def add_lexicographic_metric(self,
        score_updater: str,
        reduce: Optional[Literal["sum", "max", "min", "mean", "L1", "L2", "absmax", "absmin"]] = None,
        maximize: bool = False,
        optimal_bound: Optional[float] = None,
        acceptance_threshold: Optional[float] = None,
        is_inclusive: bool = False,
        tolerance: float = 1e-6,
        burst_length: int = 50,
        num_bursts: int = 10,
    ) -> 'ConfigurationManager':
        self.construction_history.append({
            "method": "add_lexicographic_metric",
            "kwargs": {
                "score_updater": score_updater,
                "reduce": reduce,
                "maximize": maximize,
                "optimal_bound": optimal_bound,
                "acceptance_threshold": acceptance_threshold,
                "is_inclusive": is_inclusive,
                "tolerance": tolerance,
                "burst_length": burst_length,
                "num_bursts": num_bursts,
            }
        })

        if score_updater not in self.updaters:
            warn(f"Lexicographic metric optimization requires an updater '{score_updater}', but it was not found.")

        def reduce_fn(value, reduction):
            if reduction is None:
                return value
            
            if isinstance(value, dict):
                v = list(value.values())
            elif isinstance(value, tuple):
                v = list(value)
            elif isinstance(value, np.ndarray):
                v = value.tolist()
            else:
                v = value
        
            if reduction == "min":
                return min(v)
            elif reduction == "max":
                return max(v)
            elif reduction == "absmax":
                return max([abs(t) for t in v])
            elif reduction == "absmin":
                return min([abs(t) for t in v])
            elif reduction == "sum":
                return sum(v)
            elif reduction == "mean":
                return mean(v)
            elif reduction == "L1":
                return sum(abs(v))
            elif reduction == "L2":
                return sum([t**2 for t in v])**0.5
            else:
                raise ValueError(f"Unknown reduction '{reduction}'.")

        self.lex_metrics.append(
            OptimizationMetric(
                lambda p: reduce_fn(p[score_updater], reduce),
                maximize,
                optimal_bound,
                acceptance_threshold,
                is_inclusive,
                tolerance
            )
        )
        self.lex_burst_lengths.append(burst_length)
        self.lex_num_bursts.append(num_bursts)

        print(f"Added lexicographic optimization metric '{score_updater}'")
        return self

    def add_lexicographic_preoptimization(self, limit: int = 10000) -> 'ConfigurationManager':
        self.construction_history.append({
            "method": "add_lexicographic_preoptimization",
            "kwargs": {
                "limit": limit,
            }
        })
        self.lex_preoptimization_limit = limit
        return self

    # --- Proposal Creation ---
    def proposal(self, initial_partition: Optional[Partition] = None) -> Callable[[Partition], Partition]:
        if not initial_partition:
            initial_partition = self.initial_partition

        num_districts = len(initial_partition)
        ideal_population = self.population_params["ideal_pop"]

        spanning_tree_fn = partial(
            rutil.random_spanning_tree_with_edge_penalties,
            edge_penalties=self.edge_penalties
        )

        proposal = partial(
            recom,
            pop_col=self.population_params["column_id"],
            pop_target=ideal_population,
            epsilon=self.pop_tolerance,
            node_repeats=num_districts,
            region_surcharge=self.region_surcharges,
            method=partial(
                bipartition_tree,
                max_attempts=1000,
                allow_pair_reselection=True,
                spanning_tree_fn=spanning_tree_fn,
            )
        )
        return proposal

    @classmethod
    def from_config(cls, config_path: str) -> 'ConfigurationManager':
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        init_params = config_data.get("initialization", {})
        
        # Helper to resolve relative paths if they don't exist
        config_dir = os.path.dirname(os.path.abspath(config_path))
        def resolve_path(path):
            if os.path.exists(path):
                return path
            # Try relative to config file
            rel_path = os.path.join(config_dir, path)
            if os.path.exists(rel_path):
                return rel_path
            # Try relative to project root (heuristically backing up)
            # Assuming config is in output/TAG/run/config.yaml
            # Project root is likely 3 levels up
            root_path = os.path.join(config_dir, "../../..", path)
            if os.path.exists(root_path):
                return os.path.abspath(root_path)
            return path # Return original if resolution fails

        if "pop_geodata_path" in init_params:
            init_params["pop_geodata_path"] = resolve_path(init_params["pop_geodata_path"])
        if "initial_plan_path" in init_params:
            init_params["initial_plan_path"] = resolve_path(init_params["initial_plan_path"])

        try:
            instance = cls(**init_params)
        except TypeError as e:
            raise ValueError(f"Error authenticating ConfigurationManager with params {init_params}: {e}")

        # Replay construction history
        construction_history = config_data.get("construction", [])
        for step in construction_history:
            method_name = step.get("method")
            kwargs = step.get("kwargs", {})
            
            if hasattr(instance, method_name):
                method = getattr(instance, method_name)
                # Handle potential path arguments in kwargs (e.g. edge penalties)
                if "csv_path" in kwargs:
                    kwargs["csv_path"] = resolve_path(kwargs["csv_path"])
                
                try:
                    method(**kwargs)
                except Exception as e:
                    print(f"Warning: Failed to replay configuration method '{method_name}': {e}")
            else:
                print(f"Warning: Method '{method_name}' not found on ConfigurationManager. Skipping.")

        return instance
