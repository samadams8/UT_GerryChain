from codecs import namereplace_errors
import os
from warnings import warn
import maup
from numpy import str_
import pandas as pd
import geopandas as gpd
from functools import partial
import random
import networkx as nx
from typing import Iterable, Optional, Dict, Any, List, Callable, Tuple, Literal
from pandas.core.indexes.accessors import NoNewAttributesMixin
import yaml
from math import ceil, sqrt
import json

from gerrychain import Graph, GeographicPartition, MarkovChain, Partition, updaters, constraints, accept, optimization
from gerrychain.proposals import recom
from gerrychain.tree import bipartition_tree, random_spanning_tree
from networkx.algorithms import tree as nx_tree
from gerrychain.constraints import contiguous, Validator, UpperBound
from gerrychain.metrics import polsby_popper
from gerrychain.updaters.locality_split_scores import LocalitySplits
import numpy as np

from .results import ResultSet
from .metrics import calculate_partisan_metrics, compute_split_name_lists

# --- HELPER CLASSES AND FUNCTIONS ---
def random_spanning_tree_with_edge_penalties(
    graph: nx.Graph,
    edge_penalties: Optional[Dict[Tuple[int, int], float]] = None,
    region_surcharge: Optional[Dict[str, float]] = None,
) -> nx.Graph:
    """
    Builds a spanning tree using Kruskal's method with random weights,
    allowing for region-based surcharges (standard GerryChain behavior) and specific edge penalties (e.g., to impose transitability constraints).

    This function is a flexible replacement for GerryChain's default
    `random_spanning_tree`, enabling more complex weighting schemes.

    :param graph: The input graph to build the spanning tree from.
    :param region_surcharge: A dictionary where keys are column names in the graph nodes
        (e.g., 'county_id') and values are surcharges for edges crossing
        those regional boundaries.
    :param edge_penalties: A dictionary where keys are edge tuples (u, v)
        and values are penalty weights to be added to those specific edges.
        The function checks for edges in a canonical (sorted) form.
    :returns: The minimum spanning tree based on the calculated random weights.
    """
    edge_penalties = edge_penalties or {}
    region_surcharge = region_surcharge or {}

    # print("  DEBUG: random spanning tree with edge penalties")
    # print(f"  DEBUG: edge penalties: {edge_penalties}")
    # print(f"  DEBUG: region surcharge: {region_surcharge}")

    for u, v in graph.edges():
        weight = random.random()

        # Apply region surcharges (GerryChain standard behavior)
        for key, value in region_surcharge.items():
            if (
                graph.nodes[u].get(key) != graph.nodes[v].get(key)
                or graph.nodes[u].get(key) is None
                or graph.nodes[v].get(key) is None
            ):
                weight += value

        # Apply specific edge penalties from our configuration
        edge_canonical = tuple(sorted((u, v)))
        if edge_canonical in edge_penalties:
            weight += edge_penalties[edge_canonical]

        graph.edges[(u, v)]["random_weight"] = weight

    return nx_tree.minimum_spanning_tree(
        graph, algorithm="kruskal", weight="random_weight"
    )

def _assignment_hash(partition):
    return hash(frozenset(partition.assignment.items()))

class NotEqual(Validator):
    """
    A constraint that is satisfied if the proposed partition is not the same as
    the partition it is being compared to (its parent). It uses a hash of the
    assignment to perform this check efficiently.

    Requires the `assignment_hash` updater to be active.
    """
    def __init__(self):
        """
        Initializes the NotEqual constraint.
        """
        pass

    def __call__(self, partition):
        """
        Checks if the current partition's assignment hash is different from
        its parent's.

        :param partition: The proposed partition to check.
        :return: True if the partition is different from its parent, False otherwise.
        """
        if partition.parent is None:
            return True  # The initial partition is always valid

        # The 'assignment_hash' must be in the updaters for this to work.
        # This check provides a helpful error message if the updater is missing.
        if "assignment_hash" not in partition.updaters:
            raise KeyError(
                "The 'NotEqual' constraint requires the 'assignment_hash' updater. "
                "Please add it to your Partition's updaters."
            )

        return partition["assignment_hash"] != partition.parent["assignment_hash"]

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
            print(f"!!!Config included run parameters: {config['run']}. Note that run parameters are passed as arguments to precondition() and run().")

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
            random_spanning_tree_with_edge_penalties,
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
            self._constraints.append(NotEqual())
            print(f"Constraint: prevent same map from being generated twice in a row")

            if create_updater and "assignment_hash" not in self._updaters:
                self._updaters["assignment_hash"] = _assignment_hash
                print(f"  Added 'assignment_hash' updater")
                if ignore_output:
                    self.ignore_output("assignment_hash")
        else:
            self._constraints.remove(NotEqual())
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
            raise FileNotFoundError(f"Edge file not found: {csv_path}")
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

    def add_election_updater(
        self,
        name: str,
        parties_to_columns: Dict[str, str],
    ) -> 'EnsembleRunner':
        # Record the construction history
        self._construction_history.append({
            "method": "add_election_updater",
            "kwargs": {
                "name": name,
                "parties_to_columns": parties_to_columns
            },
        })

        self._updaters[name] = updaters.Election(
            name=name,
            parties_to_columns=parties_to_columns,
        )
        print(f"  Added election updater: '{name}'")

        return self

    def add_updater_function(self,
        name: str,
        function: Callable[[Partition], Any],
        ignore_output: bool = False,
    ) -> 'EnsembleRunner':
        if isinstance(function, str):
            print(f"!!! Updater function '{name} ({function})' is a string. This usually occurs when restoring a runner from a config file. Please use .add_updater_function() to add this function manually.")
            return self

        # Record the construction history
        self._construction_history.append({
            "method": "add_updater_function",
            "kwargs": {
                "name": name,
                "function": function.__name__,
                "ignore_output": ignore_output
            },
        })
    
        if name in self._updaters:
            print(f"Updater function '{name}' already exists. Overwriting...")
        self._updaters[name] = function.__name__
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
        print(f"  Ignoring updater in output: '{name}'")
        return self

    # Tilted run
    def make_tilted_run(self,
        less_compact_probability: float = 1.0,
        compactness_score: Literal["cut_edges", "polsby_popper"] = "polsby_popper"
    ) -> 'EnsembleRunner':
        # Record the construction history
        self._construction_history.append({
            "method": "make_tilted_run",
            "kwargs": {
                "less_compact_probability": less_compact_probability,
                "compactness_score": compactness_score
            }
        })

        if less_compact_probability < 1.0:
            self._tilted_run_params["less_compact_probability"] = less_compact_probability
            self._tilted_run_params["compactness_score"] = compactness_score
            print(f"Tilted run: less compact probability {less_compact_probability}, compactness score {compactness_score}")

            # Add updaters for whichever score was selected
            if compactness_score == "cut_edges":
                self.add_updater_function("cut_edges", updaters.cut_edges)
            elif compactness_score == "polsby_popper":
                if "perimeter" not in self._updaters:
                    self.add_updater_function("perimeter", updaters.perimeter)
                if "area" not in self._updaters:
                    self.add_updater_function("area", updaters.Tally("area", alias="area"))
                if "polsby_popper" not in self._updaters:
                    self.add_updater_function("polsby_popper", polsby_popper)
        else:
            self._tilted_run_params = {}
            print(f"Tilted run: disabled")
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
                constraints.append(NotEqual())

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

        if (self._tilted_run_params
            and self._tilted_run_params["less_compact_probability"] < 1.0
            ):
            # Whether to minimize or maximize the compactness score; depends on which score is being used
            maximize = (
                self._tilted_run_params["compactness_score"] == "polsby_popper"
            )

            if self._tilted_run_params["compactness_score"] == "cut_edges":
                optim = lambda p: len(p["cut_edges"])
            elif self._tilted_run_params["compactness_score"] == "polsby_popper":
                optim = lambda p: sum(p["polsby_popper"].values())

            optimizer = optimization.SingleMetricOptimizer(
                proposal=proposal,
                constraints=self._constraints,
                initial_state=initial_partition,
                optimization_metric=optim,
                maximize=maximize,
            )
            partition_iterator = optimizer.tilted_run(
                num_steps,
                p=self._tilted_run_params["less_compact_probability"],
                with_progress_bar=True
            )
            print(f"Configured tilted run with {num_steps} steps and p={self._tilted_run_params['less_compact_probability']}")
        else:
            chain = MarkovChain(
                proposal=proposal,
                constraints=self._constraints,
                accept=accept.always_accept,
                initial_state=initial_partition,
                total_steps=num_steps
            )
            partition_iterator = chain.with_progress_bar()
            print(f"Configured neutral run with {num_steps} steps")

        # Create the output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save run configuration
        metadata = {
            "initialization": self.init_params,
            "population": self._population_params,
            "region_surcharges": self._region_surcharges,
            "edge_penalties": self._edge_penalty_params,
            "tilted_run": self._tilted_run_params,
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
                    data[updater_name] = value
            output_file_handle.write(json.dumps(data) + "\n")
            output_file_handle.flush()

            # Run callbacks
            for _, value in self._callbacks.items():
                if step_number % value['frequency'] == 0:
                    value['action'](partition, step_number, save_dir)
                    
        output_file_handle.close()
