import os
from warnings import warn
import maup
import pandas as pd
import geopandas as gpd
from functools import partial
import random
from typing import Optional, Dict, Any, List, Callable, Tuple, Literal
import yaml
from math import ceil
import json
from datetime import datetime

from gerrychain import Graph, GeographicPartition, MarkovChain, Partition, updaters, accept, optimization
from gerrychain.proposals import recom
from gerrychain.tree import bipartition_tree
from gerrychain.constraints import contiguous, UpperBound
from gerrychain.metrics import polsby_popper
from gerrychain.updaters.locality_split_scores import LocalitySplits
import numpy as np

import utgc.metrics as utmetrics
from . import run_utils as rutil

# --- ENSEMBLE RUNNER ---
class EnsembleRunner:
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

        self.geodata, self.initial_plan = self._load_geodata(pop_geodata_path, initial_plan_path)
        self.graph = Graph.from_geodataframe(self.geodata)

        total_population = sum(self.geodata[pop_column])
        num_districts = len(self.initial_plan)

        print(f"  Graph built with {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")

        self._population_params = {
            "column_id": pop_column,
            "total_pop": total_population,
            "num_districts": num_districts,
            "ideal_pop": total_population / num_districts,
            "pop_tolerance": 0.01,
        }

        # Record which constructors were used to create the runner
        self._construction_history = []

        # self._region_surcharge_params = {}
        self._region_surcharges = {}
        self._region_name_to_column = {}
        self._edge_penalty_params = {}
        self._edge_penalties = {}

        self._constraint_params = {}
        self._constraints = [contiguous,]

        self._tilted_run_params = {}
        self._optimization_scheme_params = {}
        # Updaters to ignore from the output file
        self._ignored_updaters = set()

        self._precondition_params = {}
        self.preconditioned_partition = None

        self._callbacks = {}

        self._updaters = {
            "population": updaters.Tally(pop_column, alias="population"),
        }

    @classmethod
    def from_config(cls, config_path: str) -> 'EnsembleRunner':
        print(f"Initializing runner from configuration: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        init_params = config.get('initialization', {})
        runner = cls(
            pop_geodata_path=init_params.get('pop_geodata_path'),
            initial_plan_path=init_params.get('initial_plan_path'),
            random_seed=init_params.get('random_seed')
        )

        # Iterate over the entries in the construction history and call the corresponding method
        for entry in config.get('construction', []):
            method = entry['method']
            kwargs = entry['kwargs']
            getattr(runner, method)(**kwargs)

        # Remind the user that run parameters cannot be initialized from a config file
        if 'run' in config:
            print(f"!!! Config included run parameters: {json.dumps(config['run'], indent=2)}\nNote that run parameters must be passed as arguments to precondition() and run().")

        return runner

    # Convenience computed properties
    @property
    def pop_tolerance(self) -> float:
        return self._population_params.get("pop_tolerance", 0.01)
    
    @property
    def initial_partition(self) -> Partition:
        return GeographicPartition(
            self.graph, assignment="initial_plan", updaters=self._updaters
        )

    def available_election_columns(self, years: List[int], offices: List[str]) -> List[str]:
        available_columns = set(self.geodata.columns)
        available_elections = []

        for column in available_columns:
            if column.startswith("G"):
                year = int(column[1:3]) + 2000
                office = column[3:6]
                if year in years and office in offices:
                    available_elections.append(column)

        return sorted(available_elections)
    
    def election_names_parties(self, years: List[int], offices: List[str]) -> List[Dict[str, str]]:
        available_elections = self.available_election_columns(years, offices)
        election_names_parties = []

        for year in years:
            for office in offices:
                name = f"{year:04d}{office}"
                parties_to_columns = {}

                # Get a list of all the columns for this year and office
                columns = [c for c in available_elections if c.startswith(f"G{year%100:02d}{office}")]
                parties = [c[6] for c in columns]
                # Whether each entry is the only one for that party
                is_unique = [parties.count(p) == 1 for p in parties]

                # Create entries in the parties to columns dictionary for each party
                for idx, column in enumerate(columns):
                    if is_unique[idx]:
                        parties_to_columns[parties[idx]] = column
                    else:
                        key = column[6:]
                        parties_to_columns[key] = column

                election_names_parties.append({
                    "name": name,
                    "parties_to_columns": parties_to_columns
                })

        return election_names_parties
    
    def _proposal(self,
        initial_partition: Optional[Partition] = None
    ) -> Callable[[Partition], Partition]:
        if not initial_partition:
            initial_partition = self.initial_partition

        num_districts = len(initial_partition)
        ideal_population = self._population_params["ideal_pop"]

        # Spanning tree function including edge penalites
        # nb region surcharges are passed down by the caller
        spanning_tree_fn = partial(
            rutil.random_spanning_tree_with_edge_penalties,
            edge_penalties=self._edge_penalties
        )

        proposal = partial(
            recom,
            pop_col=self._population_params["column_id"],
            pop_target=ideal_population,
            epsilon=self.pop_tolerance,
            node_repeats=num_districts,
            region_surcharge=self._region_surcharges,
            method=partial(
                bipartition_tree,
                max_attempts=1000,
                allow_pair_reselection=True,
                spanning_tree_fn=spanning_tree_fn,
            )
        )
        return proposal

    # Population
    def set_pop_dev_tolerance(self, tolerance: float = 0.01) -> 'EnsembleRunner':
        # Record the construction history
        self._construction_history.append({
            "method": "set_pop_dev_tolerance",
            "kwargs": { "tolerance": tolerance }
        })

        self._population_params["pop_tolerance"] = tolerance
        print(f"Population deviation tolerance: {tolerance:%}")

        return self

    # Constraints
    def constrain_region_splits(self,
        name: Optional[str] = None,
        column_id: Optional[str] = None,
        num_split: Optional[int] = None,
        num_multi_splits: Optional[int] = None,
        create_updater: bool = True
    ) -> 'EnsembleRunner':
        """Constrain the number of splits for a given region.

        Parameters
        ----------
        region_name : Optional[str], optional
            The name of the region to constrain, by default None
        column_id : Optional[str], required
            The column ID of the region to constrain, by default None
        num_split : Optional[int], optional
            The maximum number of splits for the region, by default None
        num_multi_splits : Optional[int], optional
            The maximum number of extra splits for regions of this type, total across the map. Multi splits count the total number of pieces formed by region and district boundaries and subtract out the number of regions and the first split for each region.
        create_updater : bool, optional
            Whether to automatically create an updater to count the number of splits for the region, by default True. Set to False if you want to manually create an updater, or it has already been created; the updater is required for the constraint to function.

        Returns
        -------
        self
        """
        # Record the construction history
        self._construction_history.append({
            "method": "constrain_region_splits",
            "kwargs": {
                "name": name,
                "column_id": column_id,
                "num_split": num_split,
                "num_multi_splits": num_multi_splits,
                # Override the create_updater parameter--the updater creation will be stored in the construction history, if it was created
                "create_updater": False,
            }
        })

        if not name:
            name = column_id

        if num_split is not None:
            constraint_name = f"split_{name}"
            self._constraints.append(UpperBound(
                    lambda p: p[constraint_name],
                    num_split
                ))
            self._constraint_params[constraint_name] = num_split
            print(f"Constraint: split max {num_split} {name}")
        if num_multi_splits is not None:
            constraint_name = f"{name}_multi_splits"
            self._constraints.append(UpperBound(
                    lambda p: p[constraint_name],
                    num_multi_splits
                ))
            self._constraint_params[constraint_name] = num_multi_splits
            print(f"Constraint: max {num_multi_splits} multi-splits of {name}")

        if create_updater:
            self.add_locality_splits_updater(name=name, column_id=column_id)

        return self

    def constrain_not_equal(self,
        not_equal_constraint: bool = True,
        create_updater: bool = True,
        ignore_output: bool = True,
    ) -> 'EnsembleRunner':
        # Record the construction history
        self._construction_history.append({
            "method": "constrain_not_equal",
            "kwargs": {
                "not_equal_constraint": not_equal_constraint,
                "create_updater": create_updater,
                "ignore_output": ignore_output
            }
        })

        self._constraint_params["not_equal_constraint"] = not_equal_constraint

        if not_equal_constraint:
            self._constraints.append(rutil.NotEqual())
            print(f"Constraint: prevent same map from being generated twice in a row")

            if create_updater and "assignment_hash" not in self._updaters:
                self._updaters["assignment_hash"] = rutil._assignment_hash
                print(f"  Added 'assignment_hash' updater")
                if ignore_output:
                    self.ignore_output("assignment_hash")
        else:
            self._constraints.remove(rutil.NotEqual())
            print(f"Constraint: allow same map to be generated twice in a row")
        
        return self

    # Region surcharges
    def surcharge_region(self,
        column_id: str,
        surcharge: float,
    ) -> 'EnsembleRunner':
        if surcharge <= 0:
            return self

        # Record the construction history
        self._construction_history.append({
            "method": "surcharge_region",
            "kwargs": {
                "column_id": column_id,
                "surcharge": surcharge
            }
        })

        self._region_surcharges[column_id] = surcharge
        print(f"Surcharge: {surcharge} for {column_id}")

        return self

    # Edge penalties
    def penalize_edges_from_csv(self,
        csv_path: str,
        penalty: float,
    ) -> 'EnsembleRunner':
        if not os.path.exists(csv_path):
            warn(f"Transitability edge file not found: {csv_path}. Skipping edge penalties.")
            return self
        if penalty <= 0:
            return self
        
        # Record the construction history
        self._construction_history.append({
            "method": "penalize_edges_from_csv",
            "kwargs": {
                "csv_path": csv_path,
                "penalty": penalty
            },
        })

        edges_df = pd.read_csv(csv_path)
        for _, row in edges_df.iterrows():
            # Check whether the edge already has a penalty assigned
            edge = tuple(sorted((int(row["u"]), int(row["v"]))))
            if edge in self._edge_penalties:
                # Add the penalty to the existing penalty
                self._edge_penalties[edge] += penalty
            else:
                # Assign the penalty to the edge
                self._edge_penalties[edge] = penalty

        self._edge_penalty_params[csv_path] = penalty

        print(f"Penalizing edges from {os.path.basename(csv_path)} with weight {penalty}")

        return self

    # Updaters
    def add_locality_splits_updater(self,
        name: Optional[str] = None,
        column_id: str = "",
        ignore_ls_output: bool = True,
    ) -> 'EnsembleRunner':
        # Record the construction history
        self._construction_history.append({
            "method": "add_locality_splits_updater",
            "kwargs": {
                "name": name,
                "column_id": column_id,
                "ignore_ls_output": ignore_ls_output
            },
        })

        if not name:
            name = column_id
        ls_name = f"ls_{name}"
        self._updaters[ls_name] = LocalitySplits(
            name=ls_name,
            col_id=column_id,
            pop_col=self._population_params["column_id"],
            scores_to_compute=["num_split_localities", "num_parts"],
        )
        print(f"  Added locality split updater: '{ls_name}'")
        if ignore_ls_output:
            self.ignore_output(ls_name)

        # Count the number of localities in the partition
        num_localities = len(set(dict(self.graph.nodes(data=column_id)).values()))

        # Add convenience updaters
        sname = f"split_{name}"
        self._updaters[sname] = \
            lambda p: p[ls_name].get("num_split_localities", 0)
        print(f"  Added split updater: '{sname}'")
        
        msname = f"{name}_multi_splits"   
        self._updaters[msname] = \
            lambda p: p[ls_name].get("num_parts", 0) - p[sname] - num_localities
        print(f"  Added multi-split updater: '{msname}'")

        return self

    def make_total_column(
        self,
        total_col: str,
        all_election_columns: List[str],
    ) -> 'EnsembleRunner':
        """
        Create a total votes column by summing all election columns and sync it to graph nodes.
        
        Parameters
        ----------
        total_col : str
            Name of the total column to create
        all_election_columns : List[str]
            List of all column names for this election to sum together
        
        Returns
        -------
        self
        """
        if total_col not in self.geodata.columns:
            # Compute total from ALL columns for this election
            party_cols = [col for col in all_election_columns if col in self.geodata.columns]
            
            if party_cols:
                self.geodata[total_col] = 0
                for col in party_cols:
                    try:
                        self.geodata[total_col] = self.geodata[total_col].fillna(0) + self.geodata[col].fillna(0)
                    except Exception:
                        # If non-numeric, coerce then sum
                        self.geodata[total_col] = self.geodata[total_col].fillna(0) + pd.to_numeric(self.geodata[col], errors='coerce').fillna(0)
            
            # Sync the new column to graph nodes
            if total_col in self.geodata.columns:
                for node in self.graph.nodes:
                    if node in self.geodata.index:
                        self.graph.nodes[node][total_col] = self.geodata.loc[node, total_col]
        
        return self

    def add_election_updater(
        self,
        name: str,
        parties_to_columns: Dict[str, str],
        ignore_output: bool = True,
    ) -> 'EnsembleRunner':
        # Record the construction history
        self._construction_history.append({
            "method": "add_election_updater",
            "kwargs": {
                "name": name,
                "parties_to_columns": parties_to_columns,
                "ignore_output": ignore_output
            },
        })

        self._updaters[name] = updaters.Election(
            name=name,
            parties_to_columns=parties_to_columns,
        )
        print(f"  Added election updater: '{name}'")
        print(f"    Parties to columns: {parties_to_columns}")
        if ignore_output:
            self.ignore_output(name)

        # Create a warning if the columns are not found in the geodata
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
    ) -> 'EnsembleRunner':
        """
        Auto-detect and add election updaters for all elections that match the
        provided year and office filters. If no filters are provided, all
        detectable elections are included.

        Parameters
        ----------
        years : Optional[List[int]]
            Four-digit years to include (e.g., [2016, 2020]). If None, all
            years present in the geodata will be included.
        elections : Optional[List[str]]
            Three-letter office codes to include (e.g., ['GOV','PRE']). If None,
            all offices present in the geodata will be included.
        parties : Optional[List[str]]
            Party initials to track. Parties are matched if their key starts
            with any of the provided initials, or if the key is exactly '-'.
            If None, defaults to ['R', 'D', '-']. Pass an empty list to track
            all parties (not recommended).
        parties_to_columns_override : Optional[Dict[str, Dict[str, str]]]
            Optional mapping that overrides the auto-detected party-to-column
            mapping for specific elections. Keys must be election names in the
            form '{YYYY}{OFFICE}' (e.g., '2024GOV'), and values are mappings
            of party keys to column names (e.g., {'R1': 'G24GOVRHEN'}).
        skip_if_missing_parties : bool
            Whether to skip adding updaters if any requested parties are missing from the geodata.
        ignore_output : bool
            Whether to ignore these updaters in the serialized output.

        Returns
        -------
        self : EnsembleRunner
            Enables chaining.
        """
        # Record the construction history
        # self._construction_history.append({
        #     "method": "add_election_updaters",
        #     "kwargs": {
        #         "years": years,
        #         "elections": elections,
        #         "parties": parties,
        #         "parties_to_columns_override": parties_to_columns_override,
        #         "skip_if_missing_parties": skip_if_missing_parties,
        #         "ignore_output": ignore_output,
        #     },
        # })

        # Collect all election-like columns from geodata
        # Filter: must start with 'G' followed immediately by two digits
        columns = []
        for c in self.geodata.columns:
            if isinstance(c, str) and len(c) >= 3 and c.startswith("G") and c[1:3].isdigit():
                columns.append(c)

        # Helper function to convert two-digit year to most recent past year
        current_year = datetime.now().year
        def two_digit_to_year(yy_str: str) -> int:
            """Convert two-digit year to most recent past year."""
            yy = int(yy_str)
            # Try current century first
            candidate = 2000 + yy
            if candidate <= current_year:
                return candidate
            # Try previous century
            return 1900 + yy

        # Discover available years and offices if not supplied
        discovered_years = sorted({two_digit_to_year(c[1:3]) for c in columns if len(c) >= 3 and c[1:3].isdigit()})
        discovered_offices = sorted({c[3:6] for c in columns if len(c) >= 6})

        years = years or discovered_years
        elections = elections or discovered_offices

        # Group columns by election name {YYYY}{OFFICE}
        # Example: 'G24GOVRHEN' -> year=2024 (most recent past), office='GOV'
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

        # Build party-to-column mapping per election, compute totals, apply overrides, and register updaters
        for ename, cols in sorted(election_to_cols.items()):
            # Determine year and office for naming auxiliary columns
            year = int(ename[:4])
            office = ename[4:7]
            yy = f"{year % 100:02d}"

            # Build party keys: if multiple columns share the same first party letter,
            # assign sequential keys (e.g., R1, R2). If unique, use single-letter key (e.g., R).
            party_initial_to_cols: Dict[str, List[str]] = {}
            for c in cols:
                if len(c) < 7:
                    continue
                party_initial = c[6]
                party_initial_to_cols.setdefault(party_initial, []).append(c)

            parties_to_columns: Dict[str, str] = {}
            for initial, plist in party_initial_to_cols.items():
                if len(plist) == 1:
                    parties_to_columns[initial] = plist[0]
                else:
                    # Stable order for deterministic keys
                    for i, pc in enumerate(sorted(plist)):
                        parties_to_columns[f"{initial}{i+1}"] = pc

            # Create total votes column name (will be computed in add_election_updater)
            total_col = f"G{yy}{office}-TOT"
            # Attach total votes under '-' key
            parties_to_columns["-"] = total_col

            # Apply per-election overrides if provided
            if ename in parties_to_columns_override:
                # First, identify columns that are being overridden
                override_columns = set(parties_to_columns_override[ename].values())
                
                # Remove any existing keys that point to columns being overridden
                # (to avoid double-counting when override adds new keys for same columns)
                keys_to_remove = [
                    key for key, col in parties_to_columns.items()
                    if col in override_columns and key != "-"
                ]
                for key in keys_to_remove:
                    del parties_to_columns[key]
                
                # Now apply the overrides
                for k, v in parties_to_columns_override[ename].items():
                    parties_to_columns[k] = v

            # Filter parties if specified
            if parties is not None:
                # Filter to only include parties that:
                # 1. Start with any of the provided party initials, OR
                # 2. Are exactly '-' (for total votes)
                filtered_parties_to_columns: Dict[str, str] = {}
                for party_key, col_name in parties_to_columns.items():
                    if party_key == "-":
                        # Always include total votes if '-' is in the parties list
                        if "-" in parties:
                            filtered_parties_to_columns[party_key] = col_name
                    else:
                        # Check if party key starts with any of the provided initials
                        for party_init in parties:
                            if party_key.startswith(party_init):
                                filtered_parties_to_columns[party_key] = col_name
                                break
                parties_to_columns = filtered_parties_to_columns

            # Create total column if needed (before registering the updater)
            if "-" in parties_to_columns:
                total_col = parties_to_columns["-"]
                self.make_total_column(total_col, cols)

            # Register the election updater
            if (
                not skip_if_missing_parties
                or all( (
                    # Check if the party is included in the selected columns
                    p in parties_to_columns.keys()
                    # Check if the party has a multi-column assignment
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
    ) -> 'EnsembleRunner':
        # Record the construction history
        self._construction_history.append({
            "method": "add_election_aggregator",
            "kwargs": {
                "name": name,
                "elections": elections,
                "parties": parties,
                "ignore_table_output": ignore_table_output,
                "ignore_agg_output": ignore_agg_output
            },
        })

        # Create the table 
        self._updaters[f"{name}_table"] = lambda p: utmetrics.tabulate_partisan_data(p, elections, parties)
        print(f"  Added partisan data tabulator: '{f"{name}_table"}'")

        if ignore_table_output:
            self.ignore_output(f"{name}_table")

        # Create the aggregator
        self._updaters[name] = lambda p: utmetrics.aggregate_partisan_metrics(p[f"{name}_table"])
        print(f"  Added partisan data aggregator: '{name}'")

        if ignore_agg_output:
            self.ignore_output(name)

        return self
    
    def add_election_metric_updaters(self,
        aggregator_name: str,
        metrics: List[str],
        prepend_agg_name: bool = False,
    ) -> 'EnsembleRunner':
        # Record the construction history
        self._construction_history.append({
            "method": "add_election_metric_updaters",
            "kwargs": {
                "aggregator_name": aggregator_name,
                "metrics": metrics,
                "prepend_agg_name": prepend_agg_name
            },
        })

        # Election metrics require an election data aggregator to be added first
        if aggregator_name not in self._updaters:
            warn(f"Election metric updater requires election data aggregator '{aggregator_name}' to be added, but it was not found.")

        for metric in metrics:
            if metric == "partisan_bias_utah":
                if prepend_agg_name: mname = f"{aggregator_name}_{metric}"
                else: mname = metric
                self._updaters[mname] = lambda p: utmetrics.partisan_bias_utah(p[aggregator_name])
            elif metric == "partisan_bias":
                if prepend_agg_name: mname = f"{aggregator_name}_{metric}"
                else: mname = metric
                self._updaters[mname] = lambda p: utmetrics.partisan_bias(p[aggregator_name])
            elif metric == "mean_median":
                if prepend_agg_name: mname = f"{aggregator_name}_{metric}"
                else: mname = metric
                self._updaters[mname] = lambda p: utmetrics.mean_median(p[aggregator_name])
            elif metric == "efficiency_gap":
                if prepend_agg_name: mname = f"{aggregator_name}_{metric}"
                else: mname = metric
                self._updaters[mname] = lambda p: utmetrics.efficiency_gap(p[aggregator_name])
            elif metric == "stdev_partisan_share":
                if prepend_agg_name: mname = f"{aggregator_name}_{metric}"
                else: mname = metric
                self._updaters[mname] = lambda p: utmetrics.stdev_partisan_share(p[aggregator_name])
            elif metric == "majority_partisan_shares":
                if prepend_agg_name: mname = f"{aggregator_name}_{metric}"
                else: mname = metric
                self._updaters[mname] = lambda p: utmetrics.majority_partisan_shares(p[aggregator_name])
            elif metric == "majority_seats":
                if prepend_agg_name: mname = f"{aggregator_name}_{metric}"
                else: mname = metric
                self._updaters[mname] = lambda p: utmetrics.majority_seats(p[aggregator_name])
            else:
                raise ValueError(f"Unknown election metric: '{metric}'")
            print(f"  Added election metric updater: '{mname}'")

        return self

    # Updaters
    def add_updater_function(self,
        name: str,
        function: Callable[[Partition], Any]
    ) -> 'EnsembleRunner':
        """
        Add a custom updater function to the runner.

        Parameters
        ----------
        name : str
            Name of the updater function.
        function : Callable[[Partition], Any]
            Function to add to the runner.

        Returns
        -------
        EnsembleRunner
            Self.
        """
        if isinstance(function, str):
            print(f"!!! Updater function '{name} ({function})' is a string. This usually occurs when restoring a runner from a config file. Please use .add_updater_function() to add this function manually.")
            return self
        else:
            # Record the construction history
            self._construction_history.append({
                "method": "add_updater_function",
                "kwargs": {
                    "name": name,
                    "function": function.__name__,
                },
            })

        self._add_updater_function(name=name, function=function)

        return self

    def _add_updater_function(self,
        name: str,
        function: Callable[[Partition], Any],
        ignore_output: bool = False,
    ) -> 'EnsembleRunner':
        """
        Internal method to add an updater function to the runner. Does not record the construction history (assumes this is done by the caller).

        Parameters
        ----------
        name : str
            Name of the updater function.
        function : Callable[[Partition], Any]
            Function to add to the runner.
        ignore_output : bool, optional
            Whether to ignore the output of the updater function in the serialized output. Defaults to False.

        Returns
        -------
        EnsembleRunner
            Self.
        """
        if name in self._updaters:
            print(f"Updater function '{name}' already exists. Overwriting...")
        
        self._updaters[name] = function
        print(f"  Added updater function: '{name}'")

        if ignore_output: 
            self.ignore_output(name)

        return self

    def ignore_output(self,
        name: str,
    ) -> 'EnsembleRunner':
        """
        Indicate that the results for the named updater should not be included in the output file.
        """
        self._ignored_updaters.add(name)
        print(f"    Ignoring updater in output: '{name}'")
        return self

    # Shape metrics
    def add_shape_metrics(self, metrics: List[str]) -> 'EnsembleRunner':
        """
        Add shape metrics (Polsby-Popper and Reock score) as updaters.
        These metrics are computed independently of optimization.

        Parameters
        ----------
        metrics : List[str]
            List of shape metrics to add. Options:
            - "polsby_popper"
            - "reock_score"

        Returns
        -------
        EnsembleRunner
            Self.
        """
        # Record the construction history
        self._construction_history.append({
            "method": "add_shape_metrics",
            "kwargs": {
                "metrics": metrics,
            }
        })

        for metric in metrics:
            if metric not in self._updaters:
                if metric == "polsby_popper":
                    if "perimeter" not in self._updaters:
                        self._add_updater_function("perimeter", updaters.perimeter, ignore_output=True)
                    if "area" not in self._updaters:
                        self._add_updater_function("area", updaters.Tally("area", alias="area"), ignore_output=True)
                    if "polsby_popper" not in self._updaters:
                        self._add_updater_function("polsby_popper", polsby_popper)
                elif metric == "reock_score":
                    self._add_updater_function("reock_score", rutil._reock_score)
                    print(f"  Added Reock score shape metric")
                else:
                    raise ValueError(f"Unknown shape metric: '{metric}'")

        return self

    # Optimization schemes
    def add_optimization_scheme(self,
        scheme: str,
        updater: str,
        **kwargs
    ) -> 'EnsembleRunner':
        """
        Set the optimization scheme for the ensemble run.

        Parameters
        ----------
        scheme : str
            Name of the optimization scheme. Options:
            - "neutral": Standard MarkovChain (default behavior)
            - "tilted": SingleMetricOptimizer with tilted_run
            - "short_bursts": SingleMetricOptimizer with short_bursts iterator
        updater: str
            Name of the updater to use for optimization. Must be added to the runner first.
        **kwargs
            Scheme-specific parameters:
            - For "tilted": requires "less_compact_probability" (< 1.0), optional "maximize" (default False)
            - For "short_bursts": accepts "burst_length" (default 100) and "num_bursts" (default ceil(num_steps / burst_length))

        Returns
        -------
        self
        """
        # Record the construction history
        self._construction_history.append({
            "method": "add_optimization_scheme",
            "kwargs": {
                "scheme": scheme,
                "updater": updater,
                **kwargs
            }
        })

        if scheme not in ["neutral", "tilted", "short_bursts"]:
            raise ValueError(f"Unknown optimization scheme: '{scheme}'. Options: 'neutral', 'tilted', 'short_bursts'")

        if updater not in self._updaters:
            raise ValueError(f"Updater '{updater}' not found. Must be added to the runner first.")

        self._optimization_scheme_params = {
            "scheme": scheme,
            "updater": updater,
            **kwargs
        }
        print(f"Optimization scheme: {scheme}")
        if kwargs:
            print(f"  Parameters: {kwargs}")

        return self
    
    # Callbacks
    def add_runtime_callback(self,
        name: str,
        frequency: int = 1,
        action: Callable[[Partition, int, str], None] = None,
    ) -> 'EnsembleRunner':
        if frequency <= 0:
            raise ValueError("Callback frequency must be a positive integer.")
        if not action:
            raise ValueError("Callback action must be provided.")
        if isinstance(action, str):
            print(f"!!! Callback action '{name} ({action})' is a string. This usually occurs when restoring a runner from a config file. Please use .add_runtime_callback() to add this action manually.")
            return self

        # Record the construction history
        self._construction_history.append({
            "method": "add_runtime_callback",
            "kwargs": {
                "name": name,
                "frequency": frequency,
                "action": action.__name__
            },
        })

        self._callbacks[name] = {"frequency": frequency, "action": action}
        print(f"Registered callback '{name}' ({action.__name__}) to run every {frequency} steps.")

        return self

    # --- Internal Builder Methods ---
    def _load_geodata(self,
        pop_geodata_path: str,
        initial_plan_path: str,
        crs: Optional[str] = "EPSG:26912",
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        geodata = gpd.read_file(pop_geodata_path)
        print(f"Loaded {len(geodata)} segments from {pop_geodata_path}")
        initial_plan = gpd.read_file(initial_plan_path)
        print(f"Loaded {len(initial_plan)} districts from {initial_plan_path}")

        print(f"Projecting to {crs}")
        geodata = geodata.to_crs(crs)
        initial_plan = initial_plan.to_crs(crs)

        # Create unique IDs for unincorporated municipalities
        if any(geodata["MUNIID"] == ""):
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
        geodata["area"] = geodata.geometry.area

        return geodata, initial_plan

    # --- Preconditioning Method ---
    def precondition(self,
        steps: int = 50,
        max_attempts: int = 1,
    ) -> 'EnsembleRunner':
        """
        Runs a preconditioning phase to find a better starting plan.
        This method should be called BEFORE .run().
        """
        print("=== Preconditioning ===")
        # Check if preconditioning has already been run
        if self.preconditioned_partition:
            print("Preconditioning has already been run. Overwriting preconditioned partition.")
            self.preconditioned_partition = None

        self._precondition_params = {
            "steps": steps,
            "max_attempts": max_attempts,
        }

        # Preconditioning requires the graph and initial partition to be built.
        initial_partition = self.initial_partition
        proposal = self._proposal(initial_partition)

        # Define metrics for the optimization objective
        def _optimization_metric(partition):
            def _pop_dev(partition):
                max_deviation = 0
                for pop in partition["population"].values():
                    max_deviation = max(max_deviation, abs(float(pop) - self._population_params["ideal_pop"]) / self._population_params["ideal_pop"])
                return max_deviation

            def _ceiling_objective(value, ceiling):
                if value > ceiling:
                    return abs(value - ceiling) ** 2
                else:
                    return 0

            components = [
                _ceiling_objective(_pop_dev(partition)/self.pop_tolerance, 1),
            ]
            for name, ceiling in self._constraint_params.items():
                if "split" in name:
                    try:
                        num_splits = partition[name]
                    except Exception:
                        print(f"  WARNING: {name} not found in partition updaters!")
                        num_splits = 0
                    components.append(_ceiling_objective(num_splits, ceiling))
            
            return sum(components)

        for attempt in range(max_attempts):
            if attempt > 0:
                print(f"  Retrying preconditioning (attempt {attempt + 1}/{max_attempts})...")
            else:
                print("Starting preconditioning...")

            constraints = [contiguous,]
            if "not_equal_constraint" in self._constraint_params and self._constraint_params["not_equal_constraint"] is True:
                constraints.append(rutil.NotEqual())

            optimizer = optimization.SingleMetricOptimizer(
                proposal=proposal,
                constraints=constraints,
                initial_state=initial_partition,
                optimization_metric=_optimization_metric,
                maximize=False,
            )

            for _ in optimizer.short_bursts(
                5,
                ceil(steps / 5),
                with_progress_bar=True,
            ):
                pass
            optimized_partition = optimizer.best_part

            # If the optimized partition passes all constraints, return it
            passed_constraints = [c(optimized_partition) for c in self._constraints]
            # print(f"  Passed constraints: {passed_constraints}")
            if all(passed_constraints):
                print("  Preconditioning successful! All tolerances met.")
                break
            else:
                print(f"  Preconditioning failed to meet all tolerances.")

                # split_counties_names, split_munis_names = compute_split_name_lists(optimized_partition)
                # print(f"  Split counties: {split_counties_names}")
                # print(f"  Split municipalities: {split_munis_names}")

                initial_partition = optimized_partition

        # Rehydrate the partition with the full set of updaters for the main run
        self.preconditioned_partition = GeographicPartition(
            self.graph,
            assignment=optimized_partition.assignment,
            parent=None,
            updaters=self._updaters
        )

        return self

    # --- Internal Run Helper Methods ---
    def _create_optimization_metric(self, updater_name: str):
        """
        Create an optimization metric function from an updater name.
        
        Handles different updater value types:
        - dict: sums the values (e.g., polsby_popper)
        - set/list: uses length (e.g., cut_edges)
        - scalar: uses directly
        
        Parameters
        ----------
        updater_name : str
            Name of the updater to use for optimization
            
        Returns
        -------
        Callable
            Function that takes a partition and returns a numeric optimization metric
        """
        def optimization_metric(partition):
            value = partition[updater_name]
            if isinstance(value, dict):
                # For dict updaters (e.g., polsby_popper), sum the values
                return sum(value.values())
            elif isinstance(value, (set, list)):
                # For set/list updaters (e.g., cut_edges), use length
                return len(value)
            else:
                # For scalar values, use directly
                return value
        return optimization_metric

    def _create_partition_iterator(self,
        proposal: Callable,
        initial_partition: Partition,
        num_steps: int,
    ):
        """
        Create a partition iterator based on the configured optimization scheme.

        Parameters
        ----------
        proposal : Callable
            The proposal function for generating new partitions
        initial_partition : Partition
            The starting partition
        num_steps : int
            Number of steps to run

        Returns
        -------
        Iterator
            Iterator over partitions
        """
        # Determine which scheme to use
        # Default is "neutral" (no optimization) unless explicitly set
        scheme_params = self._optimization_scheme_params.copy()
        scheme = scheme_params.get("scheme", "neutral")

        if scheme == "neutral":
            chain = MarkovChain(
                proposal=proposal,
                constraints=self._constraints,
                accept=accept.always_accept,
                initial_state=initial_partition,
                total_steps=num_steps
            )
            partition_iterator = chain.with_progress_bar()
            print(f"Configured neutral run with {num_steps} steps")

        elif scheme == "tilted":
            updater_name = scheme_params.get("updater")
            if updater_name is None:
                raise ValueError("No updater specified for tilted optimization scheme.")

            p_less_compact = scheme_params.get("less_compact_probability")
            if p_less_compact is None or p_less_compact >= 1.0:
                raise ValueError("Tilted scheme requires 'less_compact_probability' < 1.0 (via add_optimization_scheme(...)), got: {}".format(p_less_compact))

            maximize = scheme_params.get("maximize", False)

            # Create optimization metric from updater name
            optimization_metric = self._create_optimization_metric(updater_name)

            optimizer = optimization.SingleMetricOptimizer(
                proposal=proposal,
                constraints=self._constraints,
                initial_state=initial_partition,
                optimization_metric=optimization_metric,
                maximize=maximize,
            )
            partition_iterator = optimizer.tilted_run(
                num_steps,
                p=p_less_compact,
                with_progress_bar=True
            )
            print(f"Configured tilted run with {num_steps} steps and p={p_less_compact}")

        elif scheme == "short_bursts":
            updater_name = scheme_params.get("updater")
            if updater_name is None:
                raise ValueError("No updater specified for short_bursts optimization scheme.")

            maximize = scheme_params.get("maximize", False)

            # Create optimization metric from updater name
            optimization_metric = self._create_optimization_metric(updater_name)

            optimizer = optimization.SingleMetricOptimizer(
                proposal=proposal,
                constraints=self._constraints,
                initial_state=initial_partition,
                optimization_metric=optimization_metric,
                maximize=maximize,
            )

            burst_length = scheme_params.get("burst_length", 100)
            num_bursts = scheme_params.get("num_bursts", ceil(num_steps / burst_length))

            partition_iterator = optimizer.short_bursts(
                burst_length,
                num_bursts,
                with_progress_bar=True
            )
            print(f"Configured short_bursts run with {num_bursts} bursts of {burst_length} steps each")

        else:
            raise ValueError(f"Unknown optimization scheme: '{scheme}'")

        return partition_iterator

    # --- Run Execution ---
    def run(self,
        name: Optional[str] = None,
        num_steps: int = 5000,
        output_dir: Optional[str] = None,
        use_preconditioned_partition: bool = True,
    ):
        if name is None:
            save_dir = output_dir
        else:
            save_dir = os.path.join(output_dir, name)
            os.makedirs(save_dir, exist_ok=True)

        print(f"=== MCMC {name if name else '\b'} ===")

        if use_preconditioned_partition:
            if not self.preconditioned_partition:
                raise ValueError("Preconditioned partition not found. Please run .precondition() first or set use_preconditioned_partition to False.")
            initial_partition = self.preconditioned_partition
        else:
            initial_partition = self.initial_partition
        
        proposal = self._proposal(initial_partition)

        # Create partition iterator using configured optimization scheme
        partition_iterator = self._create_partition_iterator(
            proposal=proposal,
            initial_partition=initial_partition,
            num_steps=num_steps
        )

        # Create the output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save run configuration
        metadata = {
            "initialization": self.init_params,
            "population": self._population_params,
            "region_surcharges": self._region_surcharges,
            "edge_penalties": self._edge_penalty_params,
            "optimization_scheme": self._optimization_scheme_params,
            "updater_names": list(self._updaters.keys()),
            "ignored_updaters": list(self._ignored_updaters),
            "constraints": self._constraint_params,
            "callback_names": list(self._callbacks.keys()),
            "run": {
                "num_steps": num_steps,
                "use_preconditioned_partition": use_preconditioned_partition,
                "preconditioning": self._precondition_params,
            },
            "construction": self._construction_history,
        }

        with open(os.path.join(save_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(metadata, f, sort_keys=False)
        
        # Set results path and open file for writing
        output_file = os.path.join(save_dir, "output.jsonl")
        output_file_handle = open(output_file, "w")
        print("Running Markov chain...")
        for iter, partition in enumerate(partition_iterator):
            # Use (iter + 1) so that step numbers correspond to how many
            # iterations have been completed at the time of the callback
            step_number = iter + 1

            # Collect updater data from this step
            data = {'step': step_number}
            for updater_name in self._updaters.keys():
                if updater_name in self._ignored_updaters:
                    continue

                value = partition[updater_name]
                if isinstance(value, dict):
                    # Sort the dictionary by key
                    data[updater_name] = {
                        k: v for k, v in sorted(value.items())
                    }
                else:
                    # Make the value a string
                    data[updater_name] = str(value)
            output_file_handle.write(json.dumps(data) + "\n")
            output_file_handle.flush()

            # Run callbacks
            for _, value in self._callbacks.items():
                if step_number % value['frequency'] == 0:
                    value['action'](partition, step_number, save_dir)
            
            # Clear parent partition to save memory
            partition.parent = None
        
        output_file_handle.close()

    def compute_metrics_for_map(self, shapefile_path: str) -> Dict[str, Any]:
        """
        Compute metrics for a user-defined map (shapefile) using the same
        updater system as ensemble generation.
        
        :param shapefile_path: Path to the shapefile containing the map
        :return: Dictionary of metric values matching the format of ensemble results
        """
        import geopandas as gpd
        import maup
        
        # Load the user map
        user_map = gpd.read_file(shapefile_path)
        
        # Ensure same CRS
        if self.geodata.crs != user_map.crs:
            user_map = user_map.to_crs(self.geodata.crs)
        
        # Assign districts to geodata using maup
        geodata_copy = self.geodata.copy()
        geodata_copy['user_assignment'] = maup.assign(geodata_copy, user_map)
        
        # Sync assignment to graph nodes
        for node in self.graph.nodes:
            if node in geodata_copy.index:
                self.graph.nodes[node]['user_assignment'] = geodata_copy.loc[node, 'user_assignment']
        
        # Create partition with user assignment and all updaters
        partition = GeographicPartition(
            self.graph,
            assignment="user_assignment",
            updaters=self._updaters
        )
        
        # Collect metrics, matching the format used in ensemble generation
        metrics = {}
        for updater_name in self._updaters.keys():
            if updater_name in self._ignored_updaters:
                continue
            
            try:
                value = partition[updater_name]
                if isinstance(value, dict):
                    # Sort dictionary by key for consistency
                    metrics[updater_name] = {
                        k: v for k, v in sorted(value.items())
                    }
                else:
                    # Convert to string to match JSONL serialization format
                    metrics[updater_name] = str(value)
            except Exception as e:
                # If updater fails, skip it (some updaters may not work on arbitrary partitions)
                print(f"  Warning: Could not compute '{updater_name}': {e}")
                metrics[updater_name] = None
        
        return metrics
