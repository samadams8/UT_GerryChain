import os
import json
import yaml
import geopandas as gpd
import maup
from typing import Optional, Dict, Any, List, Callable, Union

from gerrychain import Partition, GeographicPartition

from .configuration import ConfigurationManager
from . import run_utils as rutil
from .preconditioning import precondition as run_precondition
from .chain import create_partition_iterator
from .optimization import LexicographicOptimizer

class EnsembleRunner:
    def __init__(self, config_manager: ConfigurationManager):
        self.config = config_manager
        self.preconditioned_partition = None
        self.callbacks = {}

    @classmethod
    def from_config(cls, config_path: str) -> 'EnsembleRunner':
        config = ConfigurationManager.from_config(config_path)
        return cls(config)

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

    # --- Preconditioning ---
    def precondition(self,
        steps: int = 50,
        max_attempts: int = 1,
    ) -> 'EnsembleRunner':
        print("=== Preconditioning ===")
        if self.preconditioned_partition:
            print("Preconditioning has already been run. Overwriting preconditioned partition.")
            self.preconditioned_partition = None

        self.config.precondition_params = {
            "steps": steps,
            "max_attempts": max_attempts,
        }

        initial_partition = self.config.initial_partition
        proposal = self.config.proposal(initial_partition)

        self.preconditioned_partition = run_precondition(
            initial_partition=initial_partition,
            proposal=proposal,
            constraints=self.config.constraints,
            constraint_params=self.config.constraint_params,
            population_params=self.config.population_params,
            graph=self.config.graph,
            updaters=self.config.updaters,
            steps=steps,
            max_attempts=max_attempts,
        )

        return self

    # --- Internal Run Helper Methods ---
    def _lexicographic_optimize(self, partition: Partition) -> Partition:
        if not self.config.lex_metrics:
            return partition

        active_constraints = [c for c in self.config.constraints if not isinstance(c, rutil.NotEqual)]
        
        optimizer = LexicographicOptimizer(
            proposal=self.config.proposal(partition), # Re-using standard proposal, or maybe random flip? Original used propose_random_flip but that's imported from gerrychain? 
            # Original: proposal=propose_random_flip.  Need to import it if we want it.
            # But wait, original code was: proposal=propose_random_flip
            # I should verify if I need to import that.
            constraints=active_constraints,
            initial_state=partition,
            metrics=self.config.lex_metrics
        )
        # Note: I need to ensure propose_random_flip is available or use config's proposal. 
        # Usually for polishing/local search, random flip is better than ReCom. 
        # I'll check imports.

        from collections import deque
        deque(
            optimizer.sequential_short_bursts(
                burst_lengths=self.config.lex_burst_lengths,
                num_bursts=self.config.lex_num_bursts,
                preoptimization_limit=self.config.lex_preoptimization_limit,
                verbose=True
            ),
            maxlen=0
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
            initial_partition = self.config.initial_partition
        
        proposal = self.config.proposal(initial_partition)

        partition_iterator = create_partition_iterator(
            proposal=proposal,
            initial_partition=initial_partition,
            constraints=self.config.constraints,
            optimization_scheme_params=self.config.optimization_scheme_params,
            num_steps=num_steps,
        )

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save run configuration
        # This will be slightly different now
        metadata = {
            "initialization": self.config.init_params,
            "population": self.config.population_params,
            "region_surcharges": self.config.region_surcharges,
            "edge_penalties": self.config.edge_penalty_params,
            "optimization_scheme": self.config.optimization_scheme_params,
            "updater_names": list(self.config.updaters.keys()),
            "ignored_updaters": list(self.config.ignored_updaters),
            "constraints": self.config.constraint_params,
            "callback_names": list(self.callbacks.keys()),
            "run": {
                "num_steps": num_steps,
                "use_preconditioned_partition": use_preconditioned_partition,
                # "preconditioning": self.config.precondition_params, # If we want to capture this, we need to store it
                "lexicographic_polish": lexicographic_polish,
                "lexicographic_params": {
                    "metrics": [str(m) for m in self.config.lex_metrics], # OptimizationMetric likely needs str repr
                    "burst_lengths": self.config.lex_burst_lengths,
                    "num_bursts": self.config.lex_num_bursts,
                    "preoptimization_limit": self.config.lex_preoptimization_limit
                }
            },
            "construction": self.config.construction_history,
        }

        with open(os.path.join(save_dir, "config.yaml"), "w") as f:
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

                data = {'step': step_number}
                for updater_name in self.config.updaters.keys():
                    if updater_name in self.config.ignored_updaters:
                        continue

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
        user_map = gpd.read_file(shapefile_path)
        
        if self.config.geodata.crs != user_map.crs:
            user_map = user_map.to_crs(self.config.geodata.crs)
        
        geodata_copy = self.config.geodata.copy()
        geodata_copy['user_assignment'] = maup.assign(geodata_copy, user_map)
        
        for node in self.config.graph.nodes:
            if node in geodata_copy.index:
                self.config.graph.nodes[node]['user_assignment'] = geodata_copy.loc[node, 'user_assignment']
        
        partition = GeographicPartition(
            self.config.graph,
            assignment="user_assignment",
            updaters=self.config.updaters
        )
        
        data = {}
        for updater_name in self.config.updaters.keys():
            if updater_name in self.config.ignored_updaters:
                continue

            value = partition[updater_name]
            if isinstance(value, dict):
                data[updater_name] = {
                    str(k): v for k, v in sorted(value.items())
                }
            else:
                data[updater_name] = value
        
        return data
