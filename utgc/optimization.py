from typing import List, Dict, Literal, Union, Callable, Optional
import random

from gerrychain.chain import MarkovChain
from gerrychain.partition import Partition
from gerrychain.accept import always_accept

@dataclass
class OptimizationMetric:
    # Callable that takes a partition and returns a score
    score: Callable[[Partition], float]
    # Whether to maximize or minimize the score
    maximize: bool = False
    # If the score is known to be bounded, optimal_bound allows the optimizer to stop searching along that axis once it reaches the bound
    optimal_bound: Optional[float] = None
    # Threshold above/below which the solution should not be accepted
    acceptance_threshold: Optional[float] = None
    # Whether the acceptance threshold is inclusive or exclusive
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
        else:
            if self.is_inclusive:
                return s >= self.acceptance_threshold
            else:
                return s > self.acceptance_threshold

    def within_optimal_bound(self, s: float) -> bool:
        """
        Returns True if the score s is within the optimal bound, False otherwise.
        """
        if self.optimal_bound is None:
            return False
        else:
            if self.maximize:
                return s >= self.optimal_bound
            else:
                return s <= self.optimal_bound

class LexicographicOptimizer:
    """
    Class of algorithms that optimize partitions based on a lexicographic ordering of two or more metrics.
    """
    def __init__(
        self,
        proposal: Callable[[Partition], Partition],
        constraints: Union[
            Callable[[Partition], bool],
            List[Callable[[Partition], bool]],
        ],
        initial_state: Partition,
        metrics: List[OptimizationMetric],
        step_indexer: str = "step",
    ):
        self._initial_part = initial_state
        self._proposal = proposal
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
    def best_lex_score(self) -> Tuple[float]:
        return self._best_lex_score

    def lex_score(self, part: Partition) -> Tuple[float]:
        return tuple([metric.score(part) for metric in self._metrics])

    def lex_geq(self, score1: Tuple[float], score2: Tuple[float]) -> bool:
        """
        Evaluates whether score1 is at least as preferred as score2, analagous to >= for sets of lexicographic preferences.
        """
        for s1, s2, m in zip(score1, score2, self._metrics):
            if not (m.is_equivalent(s1, s2) or m.is_preferred(s1, s2)):
                return False
        return True

    def sequential_short_bursts(
        self,
        burst_lengths: Union[int, List[int]],
        num_bursts: Union[int, List[int]],
    ):
        """
        Optimizes the metrics sequentially.
        
        First, an optimization pass over metric[0] is performed. Then metric[1] is optimized, under the constraint that metric[0] maintains its optimal value. This process is repeated for metric[2], etc.
        
        The number of bursts and burst lengths are specified by the user. If a single integer is provided, it is used for all metrics. If a list is provided, it is used for the corresponding step in the sequence, and should have the same number of entries as the number of metrics.
        """
        self._best_part = self._initial_part
        self._best_lex_score = self.lex_score(self._best_part)

        if isinstance(burst_lengths, int):
            burst_lengths = [burst_lengths] * len(self._metrics)
        if isinstance(num_bursts, int):
            num_bursts = [num_bursts] * len(self._metrics)

        if len(burst_lengths) != len(num_bursts) or len(burst_lengths) != len(self._metrics):
            raise ValueError("burst_lengths, num_bursts, and LexicographicOptimizer metrics must have the same length")
        
        # For each metric, perform an optimization run
        for bursts, length, metric in zip(
            num_bursts, burst_lengths, self._metrics
        ):
            # For each burst, perform a short burst
            for _ in range(bursts):
                chain = MarkovChain(
                    proposal=self._proposal,
                    constraints=self._constraints,
                    accept=always_accept,
                    initial_state=self._best_part,
                    total_steps=length,
                )

                for part in chain:
                    yield part
                    part_score = self.lex_score(part)

                    if self.lex_geq(part_score, self._best_lex_score):
                        self._best_part = part
                        self._best_lex_score = part_score
                    
                        if metric.within_optimal_bound(metric.score(part)):
                            break