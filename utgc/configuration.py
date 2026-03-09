import os
import yaml
from typing import Optional, Dict, Any, List, Callable, Literal
from warnings import warn

import pandas as pd
import numpy as np

from gerrychain import Partition, updaters
from gerrychain.constraints import contiguous, UpperBound
from gerrychain.metrics import polsby_popper
from gerrychain.updaters.locality_split_scores import LocalitySplits

import utgc.metrics as utmetrics
from . import run_utils as rutil
from .proposals import create_recom_proposal


class ConfigurationManager:
    """
    Geography-free fluent builder for GerryChain MCMC building blocks.

    Provides only what is needed to construct native GerryChain MCMC runs:
    proposal, constraints, and updaters. Can construct itself from a serialized
    YAML description. Graph/partition are supplied at runtime by the caller.

    **GerryChain inputs (use directly in notebooks or runner):**
    - ``proposal(partition, total_population=..., num_districts=..., pop_tolerance=...)``
    - ``constraints``: List of constraint callables.
    - ``updaters``: Dict of partition updaters.
    """

    def __init__(self) -> None:
        """No required arguments. Supply pop_column, population, etc. when needed."""
        self._verbose = False
        self.population_params: Dict[str, Any] = {
            "column_id": "TOTPOP",
            "total_pop": None,
            "num_districts": None,
            "ideal_pop": None,
        }
        self.construction_history: List[Dict[str, Any]] = []
        self.region_surcharges: Dict[str, float] = {}
        self.edge_penalties: Dict[tuple, float] = {}
        self.constraints: List[Callable] = [contiguous]
        self.updaters: Dict[str, Callable] = {
            "population": updaters.Tally("TOTPOP", alias="population"),
        }

    def _log(self, msg: str) -> None:
        """Print only when verbose (e.g. when loading one config to confirm)."""
        if self._verbose:
            print(msg)

    def set_pop_column(self, pop_column: str) -> "ConfigurationManager":
        """Set the population column id used by population updater and locality splits."""
        self.construction_history.append({"method": "set_pop_column", "kwargs": {"pop_column": pop_column}})
        self.population_params["column_id"] = pop_column
        self.updaters["population"] = updaters.Tally(pop_column, alias="population")
        return self

    def add_pop_dev_updater(self, name: str = "pop_dev", ignore_output: bool = True) -> "ConfigurationManager":
        """Add population deviation updater. ideal_pop is derived from partition at runtime."""
        self.construction_history.append({
            "method": "add_pop_dev_updater",
            "kwargs": {"name": name, "ignore_output": ignore_output},
        })

        def _pop_dev_fn(p: Partition) -> Dict:
            pop = p["population"]
            if not pop:
                return {}
            ideal = sum(pop.values()) / len(pop)
            return {k: v - ideal for k, v in pop.items()}

        self.updaters[name] = _pop_dev_fn
        self._log(f"  Added population deviation updater: '{name}'")
        return self

    # Constraints
    def constrain_region_splits(self,
        name: Optional[str] = None,
        column_id: Optional[str] = None,
        num_split: Optional[int] = None,
        num_multi_splits: Optional[int] = None,
        create_updater: bool = True,
    ) -> "ConfigurationManager":
        self.construction_history.append({
            "method": "constrain_region_splits",
            "kwargs": {
                "name": name,
                "column_id": column_id,
                "num_split": num_split,
                "num_multi_splits": num_multi_splits,
                "create_updater": create_updater,
            },
        })

        if not name:
            name = column_id or "unknown_region"

        if num_split is not None:
            constraint_name = f"split_{name}"
            self.constraints.append(UpperBound(
                lambda p, n=constraint_name: p[n],
                num_split,
            ))
            self._log(f"Constraint: split max {num_split} {name}")
        if num_multi_splits is not None:
            constraint_name = f"{name}_multi_splits"
            self.constraints.append(UpperBound(
                lambda p, n=constraint_name: p[n],
                num_multi_splits,
            ))
            self._log(f"Constraint: max {num_multi_splits} multi-splits of {name}")

        if create_updater:
            self.add_locality_splits_updater(name=name, column_id=column_id or "")

        return self

    def constrain_not_equal(self,
        not_equal_constraint: bool = True,
        create_updater: bool = True,
        ignore_output: bool = True,
    ) -> "ConfigurationManager":
        self.construction_history.append({
            "method": "constrain_not_equal",
            "kwargs": {
                "not_equal_constraint": not_equal_constraint,
                "create_updater": create_updater,
                "ignore_output": ignore_output,
            },
        })

        self.constraints = [c for c in self.constraints if not isinstance(c, rutil.NotEqual)]

        if not_equal_constraint:
            self.constraints.append(rutil.NotEqual())
            self._log("Constraint: prevent same map from being generated twice in a row")
            if create_updater and "assignment_hash" not in self.updaters:
                self.updaters["assignment_hash"] = rutil._assignment_hash
                self._log("  Added 'assignment_hash' updater")
        else:
            self._log("Constraint: allow same map to be generated twice in a row")
        return self

    # Region surcharges
    def surcharge_region(self, column_id: str, surcharge: float) -> "ConfigurationManager":
        if surcharge <= 0:
            return self
        self.construction_history.append({
            "method": "surcharge_region",
            "kwargs": {"column_id": column_id, "surcharge": surcharge},
        })
        self.region_surcharges[column_id] = surcharge
        self._log(f"Surcharge: {surcharge} for {column_id}")
        return self

    # Edge penalties
    def penalize_edges_from_csv(self,
        csv_path: str,
        penalty: float,
        weight_column: str = "w",
    ) -> "ConfigurationManager":
        if not os.path.exists(csv_path):
            warn(f"Transitability edge file not found: {csv_path}. Skipping edge penalties.")
            return self
        if penalty <= 0:
            return self
        self.construction_history.append({
            "method": "penalize_edges_from_csv",
            "kwargs": {"csv_path": csv_path, "penalty": penalty, "weight_column": weight_column},
        })
        edges_df = pd.read_csv(csv_path)
        has_weights = weight_column in edges_df.columns
        if not has_weights:
            warn(f"Weight column '{weight_column}' not found in {os.path.basename(csv_path)}. Using constant penalty.")
        for _, row in edges_df.iterrows():
            edge = tuple(sorted((int(row["u"]), int(row["v"]))))
            edge_weight = row[weight_column] * penalty if has_weights else penalty
            self.edge_penalties[edge] = self.edge_penalties.get(edge, 0) + edge_weight
        self._log(f"Penalizing edges from {os.path.basename(csv_path)} with factor {penalty}")
        return self

    # Updaters
    def add_locality_splits_updater(self,
        name: Optional[str] = None,
        column_id: str = "",
        ignore_ls_output: bool = True,
    ) -> "ConfigurationManager":
        self.construction_history.append({
            "method": "add_locality_splits_updater",
            "kwargs": {"name": name, "column_id": column_id, "ignore_ls_output": ignore_ls_output},
        })
        if not name:
            name = column_id or "unknown"
        ls_name = f"ls_{name}"
        self.updaters[ls_name] = LocalitySplits(
            name=ls_name,
            col_id=column_id,
            pop_col=self.population_params["column_id"],
            scores_to_compute=["num_split_localities", "num_parts"],
        )
        self._log(f"  Added locality split updater: '{ls_name}'")
        sname = f"split_{name}"
        self.updaters[sname] = lambda p, ls=ls_name: p[ls].get("num_split_localities", 0)
        msname = f"{name}_multi_splits"

        def _multi_splits_fn(p, ls=ls_name, sn=sname, col=column_id):
            num_localities = len(set(dict(p.graph.nodes(data=col)).values()))
            return p[ls].get("num_parts", 0) - p[sn] - num_localities

        self.updaters[msname] = _multi_splits_fn
        self._log(f"  Added multi-split updater: '{msname}'")
        return self

    def add_election_updater(self,
        name: str,
        parties_to_columns: Dict[str, str],
        ignore_output: bool = True,
    ) -> "ConfigurationManager":
        self.construction_history.append({
            "method": "add_election_updater",
            "kwargs": {"name": name, "parties_to_columns": parties_to_columns, "ignore_output": ignore_output},
        })
        self.updaters[name] = updaters.Election(name=name, parties_to_columns=parties_to_columns)
        self._log(f"  Added election updater: '{name}'")
        return self

    def add_election_updaters(
        self,
        elections: Dict[str, Dict[str, str]],
        ignore_output: bool = True,
        skip_if_missing_parties: bool = False,
    ) -> 'ConfigurationManager':
        """
        Add election updaters for multiple elections from an explicit mapping.

        Parameters
        ----------
        elections : dict
            Mapping from election name (e.g. \"2012PRE\") to parties_to_columns
            dict suitable for `add_election_updater`.
        ignore_output : bool, optional
            If True, hide each election updater from output by default.
        skip_if_missing_parties : bool, optional
            If True, skip elections whose parties_to_columns mapping is empty.
        """
        self.construction_history.append({
            "method": "add_election_updaters",
            "kwargs": {
                "elections": elections,
                "ignore_output": ignore_output,
                "skip_if_missing_parties": skip_if_missing_parties,
            },
        })

        for name, parties_to_columns in elections.items():
            if skip_if_missing_parties and not parties_to_columns:
                continue
            self.add_election_updater(
                name=name,
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
    ) -> "ConfigurationManager":
        self.construction_history.append({
            "method": "add_election_aggregator",
            "kwargs": {
                "name": name,
                "elections": elections,
                "parties": parties,
                "ignore_table_output": ignore_table_output,
                "ignore_agg_output": ignore_agg_output,
            },
        })
        self.updaters[f"{name}_table"] = lambda p: utmetrics.tabulate_partisan_data(p, elections, parties)
        self.updaters[name] = lambda p: utmetrics.aggregate_partisan_metrics(p[f"{name}_table"])
        self._log(f"  Added partisan data aggregator: '{name}'")
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
            self._log(f"  Added election metric updater: '{mname}'")
        return self

    def add_updater_function(self, name: str, function: Callable[[Partition], Any]) -> "ConfigurationManager":
        if isinstance(function, str):
            self._log(f"!!! Updater function '{name}' is a string. Add the callable manually.")
            return self
        self.construction_history.append({
            "method": "add_updater_function",
            "kwargs": {"name": name, "function": function.__name__},
        })
        return self._add_updater_function(name=name, function=function)

    def _add_updater_function(self,
        name: str,
        function: Callable[[Partition], Any],
        ignore_output: bool = False,
    ) -> "ConfigurationManager":
        if name in self.updaters:
            self._log(f"Updater function '{name}' already exists. Overwriting...")
        self.updaters[name] = function
        self._log(f"  Added updater function: '{name}'")
        return self

    # Shape metrics
    def add_shape_metrics(self, metrics: List[str]) -> "ConfigurationManager":
        self.construction_history.append({"method": "add_shape_metrics", "kwargs": {"metrics": metrics}})
        for metric in metrics:
            if metric not in self.updaters:
                if metric == "polsby_popper":
                    if "perimeter" not in self.updaters:
                        self._add_updater_function("perimeter", updaters.perimeter)
                    if "area" not in self.updaters:
                        self._add_updater_function("area", updaters.Tally("area", alias="area"))
                    if "polsby_popper" not in self.updaters:
                        self._add_updater_function("polsby_popper", polsby_popper)
                elif metric == "reock_score":
                    self._add_updater_function("reock_score", rutil._reock_score)
                    self._log("  Added Reock score shape metric")
                else:
                    raise ValueError(f"Unknown shape metric: '{metric}'")
        return self

    # --- Proposal Creation ---
    def proposal(self,
        initial_partition: Partition,
        total_population: Optional[float] = None,
        num_districts: Optional[int] = None,
        pop_tolerance: float = 0.01,
    ) -> Callable[[Partition], Partition]:
        """
        Create a ReCom proposal. Population and tolerance are supplied at call time.

        Parameters
        ----------
        initial_partition : Partition
            Current partition (used for num_districts if num_districts not given).
        total_population : float, optional
            Total population (required unless already set on config).
        num_districts : int, optional
            Number of districts (defaults to len(initial_partition)).
        pop_tolerance : float, optional
            Population deviation tolerance (epsilon) for ReCom.

        Returns
        -------
        callable
            A proposal function (partition) -> partition.
        """
        num_d = num_districts if num_districts is not None else len(initial_partition)
        if total_population is None:
            raise ValueError("proposal() requires total_population=... at call time.")
        ideal_pop = total_population / num_d
        population_params = {
            "column_id": self.population_params["column_id"],
            "ideal_pop": ideal_pop,
            "num_districts": num_d,
            "total_pop": total_population,
            "pop_tolerance": pop_tolerance,
        }
        return create_recom_proposal(
            population_params,
            self.region_surcharges,
            self.edge_penalties,
            num_districts=num_d,
            pop_tolerance=pop_tolerance,
        )

    def to_config(self, config_path: str) -> None:
        """Write a configuration-only YAML file that can be loaded with from_config()."""
        data: Dict[str, Any] = {
            "initialization": {},
            "construction": list(self.construction_history),
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    @classmethod
    def from_config(cls, config_path: str, verbose: bool = False) -> "ConfigurationManager":
        """
        Construct a ConfigurationManager from a configuration-only YAML file.

        Parameters
        ----------
        config_path : str
            Path to the YAML file.
        verbose : bool, optional
            If True, log when replaying construction steps. Default False for bulk deserialization.

        Returns
        -------
        ConfigurationManager
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f) or {}
        config_dir = os.path.dirname(os.path.abspath(config_path))

        def resolve_path(path: Optional[str]) -> Optional[str]:
            if path is None or not path:
                return path
            if os.path.exists(path):
                return path
            rel = os.path.join(config_dir, path)
            if os.path.exists(rel):
                return rel
            root = os.path.join(config_dir, "../../..", path)
            if os.path.exists(root):
                return os.path.abspath(root)
            return path

        instance = cls()
        instance._verbose = verbose
        init_section = config_data.get("initialization", {})
        if init_section.get("pop_column"):
            instance.set_pop_column(init_section["pop_column"])

        construction_history = config_data.get("construction", [])
        skip_methods = {"add_optimization_scheme", "add_lexicographic_metric", "add_lexicographic_preoptimization",
                       "set_pop_dev_tolerance", "set_population_from_partition", "build_initial_partition_from_graph", "ignore_output"}
        for step in construction_history:
            method_name = step.get("method")
            kwargs = dict(step.get("kwargs") or {})
            if not isinstance(method_name, str) or method_name in skip_methods:
                continue
            if not hasattr(instance, method_name):
                if verbose:
                    print(f"Warning: Method '{method_name}' not found. Skipping.")
                continue
            method = getattr(instance, method_name)
            if "csv_path" in kwargs:
                kwargs["csv_path"] = resolve_path(kwargs["csv_path"])
            try:
                method(**kwargs)
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to replay '{method_name}': {e}")
        return instance
