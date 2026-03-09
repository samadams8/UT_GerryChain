import os
import json
import yaml
from typing import Optional, Dict, Any, List, Callable, Union

from gerrychain import Partition

from .configuration import ConfigurationManager
from . import run_utils as rutil
from .preconditioning import precondition as run_precondition
from .chain import create_partition_iterator
from .optimization import LexicographicOptimizer


class EnsembleRunner:
    def __init__(
        self,
        config_manager: ConfigurationManager,
        geography: GeographyManager_legacy,
        dataset_key: str,
    ):
        """
        Parameters
        ----------
        config_manager : ConfigurationManager
            Configuration for constraints, updaters, proposals, etc.
        geography : GeographyManager
            Geography manager providing graphs/partitions by dataset key.
        dataset_key : str
            Key of the dataset in the geography manager to use for this run.
        """
        self.config = config_manager
        self.geography = geography
        self.dataset_key = dataset_key
        self.preconditioned_partition = None
        self.callbacks: Dict[str, Dict[str, Any]] = {}
        self._precondition_params: Dict[str, Any] = {}

    @classmethod
    def from_run_directory(cls, run_dir: str) -> "EnsembleRunner":
        """
        Reconstruct an EnsembleRunner from a run directory saved by run().

        Expects:
        - config.yaml: configuration-only file written by ConfigurationManager.to_config()
        - run_metadata.yaml: run metadata including a 'geography' section with crs,
          datasets (key -> pop_geodata_path, plan_geodata_path), and default_dataset.
        """
        config_path = os.path.join(run_dir, "config.yaml")
        metadata_path = os.path.join(run_dir, "run_metadata.yaml")

        config = ConfigurationManager.from_config(config_path, verbose=False)

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Run metadata file not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f) or {}

        geography_section = metadata.get("geography", {})
        if not isinstance(geography_section, dict) or not geography_section:
            raise ValueError("Run metadata does not contain a valid 'geography' section.")

        datasets_config = geography_section.get("datasets", {})
        crs = geography_section.get("crs", "EPSG:26912")

        geo = GeographyManager_legacy(crs=crs)
        for key, spec in datasets_config.items():
            if not isinstance(spec, dict):
                continue
            p1 = spec.get("pop_geodata_path")
            p2 = spec.get("plan_geodata_path")
            if p1:
                geo.register_pop_dataset(key, p1)
            if p2:
                geo.register_plan_dataset(key, p2)

        dataset_key = geography_section.get("default_dataset")
        if not dataset_key and geo.list_keys():
            dataset_key = geo.list_keys()[0]
        if not dataset_key:
            raise ValueError("No dataset_key found in geography metadata or datasets.")

        return cls(config, geo, dataset_key)

    def add_runtime_callback(self,
        name: str,
        frequency: int = 1,
        action: Callable[[Partition, int, str], None] = None,
    ) -> 'EnsembleRunner':
        if frequency <= 0:
            raise ValueError("Callback frequency must be a positive integer.")
        if not action:
            raise ValueError("Callback action must be provided.")
        
        self.callbacks[name] = {"frequency": frequency, "action": action}
        print(f"Registered callback '{name}' ({action.__name__}) to run every {frequency} steps.")
        return self

    @staticmethod
    def _constraint_params_from_history(construction_history: List) -> Dict[str, Any]:
        """Build constraint_params dict from construction_history for precondition."""
        out: Dict[str, Any] = {}
        for step in construction_history:
            if step.get("method") == "constrain_region_splits":
                kwargs = step.get("kwargs") or {}
                name = kwargs.get("name") or kwargs.get("column_id") or "unknown"
                if kwargs.get("num_split") is not None:
                    out[f"split_{name}"] = kwargs["num_split"]
                if kwargs.get("num_multi_splits") is not None:
                    out[f"{name}_multi_splits"] = kwargs["num_multi_splits"]
            elif step.get("method") == "constrain_not_equal":
                out["not_equal_constraint"] = (step.get("kwargs") or {}).get("not_equal_constraint", True)
        return out

    # --- Preconditioning ---
    def precondition(self,
        steps: int = 50,
        max_attempts: int = 1,
        pop_tolerance: float = 0.01,
    ) -> "EnsembleRunner":
        print("=== Preconditioning ===")
        if self.preconditioned_partition:
            print("Preconditioning has already been run. Overwriting preconditioned partition.")
            self.preconditioned_partition = None

        self._precondition_params = {"steps": steps, "max_attempts": max_attempts}

        initial_partition = self.geography.build_partition(
            self.dataset_key,
            self.dataset_key,
            updaters=self.config.updaters,
            repair_contiguity=True,
        )
        total_pop = sum(initial_partition["population"].values())
        num_districts = len(initial_partition)
        ideal_pop = total_pop / num_districts
        population_params = {
            "ideal_pop": ideal_pop,
            "pop_tolerance": pop_tolerance,
            "column_id": self.config.population_params["column_id"],
            "num_districts": num_districts,
            "total_pop": total_pop,
        }
        proposal = self.config.proposal(
            initial_partition,
            total_population=total_pop,
            num_districts=num_districts,
            pop_tolerance=pop_tolerance,
        )
        constraint_params = self._constraint_params_from_history(self.config.construction_history)

        self.preconditioned_partition = run_precondition(
            initial_partition=initial_partition,
            proposal=proposal,
            constraints=self.config.constraints,
            constraint_params=constraint_params,
            population_params=population_params,
            graph=self.geography.get_graph(self.dataset_key, self.dataset_key),
            updaters=self.config.updaters,
            steps=steps,
            max_attempts=max_attempts,
        )

        return self

    # --- Internal Run Helper Methods ---
    def _lexicographic_optimize(self, partition: Partition) -> Partition:
        """Lexicographic polish: config no longer holds lex params; use getattr for backward compat."""
        lex_metrics = getattr(self.config, "lex_metrics", None)
        if not lex_metrics:
            return partition
        active_constraints = [c for c in self.config.constraints if not isinstance(c, rutil.NotEqual)]
        total_pop = sum(partition["population"].values())
        num_districts = len(partition)
        proposal = self.config.proposal(
            partition, total_population=total_pop, num_districts=num_districts, pop_tolerance=0.01
        )
        optimizer = LexicographicOptimizer(
            proposal=proposal,
            constraints=active_constraints,
            initial_state=partition,
            metrics=lex_metrics,
        )
        from collections import deque

        burst_lengths = getattr(self.config, "lex_burst_lengths", [50])
        num_bursts = getattr(self.config, "lex_num_bursts", [10])
        preopt_limit = getattr(self.config, "lex_preoptimization_limit", 0)
        deque(
            optimizer.sequential_short_bursts(
                burst_lengths=burst_lengths,
                num_bursts=num_bursts,
                preoptimization_limit=preopt_limit,
                verbose=True,
            ),
            maxlen=0,
        )
        return optimizer.best_part

    # --- Run Execution ---
    def run(self,
        name: Optional[str] = None,
        num_steps: int = 5000,
        output_dir: Optional[str] = None,
        use_preconditioned_partition: bool = True,
        lexicographic_polish: bool = False,
    ):
        if name is None:
            save_dir = output_dir
        else:
            save_dir = os.path.join(output_dir, name)
            os.makedirs(save_dir, exist_ok=True)

        label = name if name else ""
        print(f"=== MCMC {label} ===")

        if use_preconditioned_partition:
            if not self.preconditioned_partition:
                raise ValueError("Preconditioned partition not found. Please run .precondition() first or set use_preconditioned_partition to False.")
            initial_partition = self.preconditioned_partition
        else:
            initial_partition = self.geography.build_partition(
                self.dataset_key,
                self.dataset_key,
                updaters=self.config.updaters,
                repair_contiguity=True,
            )

        total_pop = sum(initial_partition["population"].values())
        num_districts = len(initial_partition)
        proposal = self.config.proposal(
            initial_partition,
            total_population=total_pop,
            num_districts=num_districts,
            pop_tolerance=0.01,
        )
        optimization_scheme_params = getattr(self.config, "optimization_scheme_params", None) or {"scheme": "neutral"}

        partition_iterator = create_partition_iterator(
            proposal=proposal,
            initial_partition=initial_partition,
            constraints=self.config.constraints,
            optimization_scheme_params=optimization_scheme_params,
            num_steps=num_steps,
        )

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save configuration (config-only) and run metadata (including geography).
        if save_dir is None:
            save_dir = output_dir or "."

        os.makedirs(save_dir, exist_ok=True)

        # Configuration-only file for ConfigurationManager.from_config
        config_out_path = os.path.join(save_dir, "config.yaml")
        self.config.to_config(config_out_path)

        total_pop = sum(initial_partition["population"].values())
        num_districts = len(initial_partition)
        ideal_pop = total_pop / num_districts
        population_snapshot = {
            **self.config.population_params,
            "total_pop": total_pop,
            "num_districts": num_districts,
            "ideal_pop": ideal_pop,
        }
        constraint_params = self._constraint_params_from_history(self.config.construction_history)
        metadata = {
            "initialization": {},
            "geography": self.geography.get_geography_config(default_dataset=self.dataset_key),
            "population": population_snapshot,
            "region_surcharges": self.config.region_surcharges,
            "updater_names": list(self.config.updaters.keys()),
            "constraints": constraint_params,
            "callback_names": list(self.callbacks.keys()),
            "run": {
                "num_steps": num_steps,
                "use_preconditioned_partition": use_preconditioned_partition,
                "preconditioning": self._precondition_params,
                "lexicographic_polish": lexicographic_polish,
            },
            "construction": self.config.construction_history,
        }

        with open(os.path.join(save_dir, "run_metadata.yaml"), "w") as f:
            yaml.safe_dump(metadata, f, sort_keys=False)
        
        output_file = os.path.join(save_dir, "output.jsonl")
        print("Running Markov chain...")
        
        with open(output_file, "w") as output_file_handle:
            for iter, partition in enumerate(partition_iterator):
                step_number = iter + 1

                if lexicographic_polish:
                    output_partition = self._lexicographic_optimize(partition)
                else:
                    output_partition = partition

                data = {"step": step_number}
                for updater_name in self.config.updaters.keys():
                    value = output_partition[updater_name]
                    if isinstance(value, dict):
                        data[updater_name] = {
                            str(k): v for k, v in sorted(value.items())
                        }
                    else:
                        data[updater_name] = value
                output_file_handle.write(json.dumps(data) + "\n")
                output_file_handle.flush()

                for _, value in self.callbacks.items():
                    if step_number % value['frequency'] == 0:
                        value['action'](output_partition, step_number, save_dir)
                
                partition.parent = None
        
        output_file_handle.close()

    def compute_metrics_for_map(self, shapefile_path: str) -> Dict[str, Any]:
        """
        Compute updater values for a user-provided map (shapefile) using the
        active geography and the runner's configuration updaters.
        """
        return self.geography.compute_metrics_for_map(
            shapefile_path,
            self.dataset_key,
            self.dataset_key,
            updaters=self.config.updaters,
            ignored_updaters=set(),
        )
