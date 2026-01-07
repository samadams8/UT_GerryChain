from typing import List, Dict, Literal, Union, Callable, Optional, Tuple, Sequence, Any
import random
from dataclasses import dataclass, asdict
import yaml
from tqdm.auto import tqdm
import data

from gerrychain.chain import MarkovChain
from gerrychain.partition import Partition
from gerrychain.accept import always_accept
from gerrychain.optimization import SingleMetricOptimizer

@dataclass
class OptimizationMetric:
    # Callable that takes a partition and returns a score
    score: Callable[[Partition], float]
    # Whether to maximize or minimize the score
    maximize: bool = False
    # If the score is known to be bounded, early_stopping_bound allows the optimizer to stop searching along that axis once it reaches the bound
    early_stopping_bound: Optional[float] = None
    # Threshold above/below which the solution should not be accepted
    acceptance_threshold: Optional[float] = None
    # Whether the early termination and acceptance thresholds are inclusive or exclusive
    is_inclusive: bool = False 
    # Amount of difference permitted between two scores before they are considered different
    tolerance: Optional[float] = None 

    def is_equivalent(self, score1: float, score2: float) -> bool:
        """
        Returns True if the scores are within the tolerance, False otherwise.
        """
        if self.tolerance is None:
            return score1 == score2
        else:
            return abs(score1 - score2) <= self.tolerance
    
    def is_preferred(self, score1: float, score2: float) -> bool:
        """
        Returns True if score1 is preferred to score2, False otherwise.
        """
        if self.maximize:
            return score1 > score2
        else:
            return score1 < score2

    def is_acceptable(self, s: float) -> bool:
        """
        Returns True if the score s is acceptable, False otherwise.
        """
        if self.acceptance_threshold is None:
            return True
        
        # Determine direction of inequality based on optimization goal
        if self.maximize:
            # Maximizing: must be greater than threshold
            return (
                s > self.acceptance_threshold
                or (self.is_inclusive and self.is_equivalent(s, self.acceptance_threshold))
            )
        else:
            # Minimizing: must be less than threshold
            return (
                s < self.acceptance_threshold
                or (self.is_inclusive and self.is_equivalent(s, self.acceptance_threshold))
            )

    def within_early_stopping_bound(self, s: float) -> bool:
        """
        Returns True if the score s is within the early stopping bound, False otherwise.
        """
        if self.early_stopping_bound is None:
            return False
        
        if self.maximize:
            return (
                s > self.early_stopping_bound
                or ( self.is_inclusive
                and self.is_equivalent(s, self.early_stopping_bound) )
            )
        else:
            return (
                s < self.early_stopping_bound
                or ( self.is_inclusive
                and self.is_equivalent(s, self.early_stopping_bound) )
            )

    @classmethod
    def to_yaml(cls, representer, data):
        dct = asdict(data)
        if "score" in dct:
            del dct["score"]
        return representer.represent_mapping('tag:yaml.org,2002:map', dct)

yaml.add_representer(OptimizationMetric, OptimizationMetric.to_yaml, Dumper=yaml.SafeDumper)

class LexicographicOptimizer:
    """
    Class of algorithms that optimize partitions based on a lexicographic ordering of two or more metrics.
    """
    def __init__(
        self,
        constraints: Union[
            Callable[[Partition], bool],
            List[Callable[[Partition], bool]],
        ],
        initial_state: Partition,
        metrics: List[OptimizationMetric],
        step_indexer: str = "step",
    ):
        self._initial_part = initial_state
        self._constraints = constraints
        self._metrics = metrics
        self._step_indexer = step_indexer

        self._best_part = None
        self._best_lex_score = None

        if self._step_indexer not in self._initial_part.updaters:
            step_updater = lambda p: (
                0 if p.parent is None else p.parent[self._step_indexer] + 1
            )
            self._initial_part.updaters[self._step_indexer] = step_updater
    
    @property
    def best_part(self) -> Partition:
        return self._best_part

    @property
    def best_lex_score(self) -> Tuple[float, ...]:
        return self._best_lex_score

    def lex_score(self, part: Partition) -> Tuple[float, ...]:
        return tuple([metric.score(part) for metric in self._metrics])

    def lex_geq(self, score1: Tuple[float, ...], score2: Tuple[float, ...], depth: Optional[int] = None) -> bool:
        """
        Evaluates whether score1 is at least as preferred as score2 lexicographically.
        
        If depth is provided, only the first `depth` metrics are considered.
        """
        metrics_to_check = self._metrics if depth is None else self._metrics[:depth]
        
        for i, m in enumerate(metrics_to_check):
            s1 = score1[i]
            s2 = score2[i]
            
            # If they are equivalent, continue to next metric
            if m.is_equivalent(s1, s2):
                continue
            
            # If not equivalent, checks if s1 is strictly preferred
            return m.is_preferred(s1, s2)
            
        # If we went through all metrics and they were all equivalent
        return True

    def optimize(
        self,
        proposals: Sequence[Callable[[Partition], Partition]],
        optimizer_configs: Sequence[Dict[str, Any]],
        preoptimization_limit: int = 0,
        verbose: bool = False,
    ):
        """
        Optimizes the metrics sequentially (as described in [1]) using specified GerryChain optimizers.
        
        First, an optimization pass over metric[0] is performed. Then metric[1] is optimized, under the constraint that metric[0] maintains its optimal value. This process is repeated for metric[2], etc.
        
        [1] https://en.wikipedia.org/wiki/Lexicographic_optimization#Sequential_algorithm_for_general_objectives
        
        :param proposals: A list of proposal functions to use for each phase of optimization.
        :param optimizer_configs: A list of dictionaries, where each dictionary configures the optimizer for the corresponding metric. Each dict must have a "method" key (e.g., "short_bursts", "simulated_annealing") and other keys as arguments.
        :param preoptimization_limit: Number of steps to pre-optimize if a metric is not acceptable.
        :param verbose: Whether to print progress.
        """
        self._best_part = self._initial_part
        self._best_lex_score = self.lex_score(self._best_part)

        if len(optimizer_configs) != len(self._metrics):
            raise ValueError("optimizer_configs and metrics must have the same length")

        metric_iter = zip(optimizer_configs, self._metrics)
        if verbose:
            metric_iter = tqdm(metric_iter, total=len(self._metrics), desc="Lexicographic Optimization")

        for i, (config, metric) in enumerate(metric_iter):
            comparison_depth = i + 1
            
            def make_lex_constraint(depth, best_score):
                def constraint(partition):
                    score = self.lex_score(partition)
                    if depth == 0:
                        return True
                    return self.lex_geq(score, best_score, depth=depth)
                return constraint
            
            baseline_score = self._best_lex_score
            phase_constraint = make_lex_constraint(i, baseline_score)
            
            constraints = self._constraints
            if isinstance(constraints, list):
                constraints = constraints + [phase_constraint]
            else:
                constraints = [constraints, phase_constraint]
            
            smo = SingleMetricOptimizer(
                proposal=proposals[i],
                constraints=constraints,
                initial_state=self._best_part,
                optimization_metric=metric.score,
                maximize=metric.maximize
            )
            
            # Call the requested method
            method_name = config.get("method", "short_bursts")
            kwargs = {k: v for k, v in config.items() if k != "method"}
            
            # Optimization methods in SingleMetricOptimizer (short_bursts, simulated_annealing, etc.) return a generator
            if not hasattr(smo, method_name):
                 raise ValueError(f"Unknown optimization method: {method_name}")
            
            optimizer_method = getattr(smo, method_name)
            
            # Run the optimizer
            # We wrap in tqdm if verbose specific to this burst
            iterator = optimizer_method(**kwargs)

            for part in iterator:
                # Check if this partition improves the GLOBAL lexicographic score up to current depth
                part_score = self.lex_score(part)

                if verbose:
                    print(f"Score: {part_score}")
                
                if self.lex_geq(part_score, self._best_lex_score, depth=comparison_depth):
                    self._best_part = part
                    self._best_lex_score = part_score
                    if verbose:
                        print(f"  New best score: {part_score}")
                    
                    # Check early stopping for current metric
                    if metric.within_early_stopping_bound(part_score[i]):
                        break
        
        return self._best_part