"""
Build partition iterators (Markov chain or optimization-based) for runs.

Standalone so notebooks and the runner can get an iterator over partitions
for neutral, tilted, or short_bursts schemes without going through EnsembleRunner.
"""
from math import ceil
from typing import Callable, Dict, Iterator, List

from gerrychain import MarkovChain, Partition
from gerrychain.accept import always_accept
from gerrychain.optimization import SingleMetricOptimizer

def coupon_collector_expectation(num_coupons):
    ''' 
    Calculates the expected number of steps to collect `num_coupons` unique coupons.
    '''
    return num_coupons * sum(1/i for i in range(1, num_coupons + 1))

class CouponCollectorChain(MarkovChain):
    """
    A MarkovChain subclass that runs multiple ReCom micro-steps for each macro-step yielded.
    """
    def __init__(self, proposal: Callable, constraints: List[Callable], accept: Callable, initial_state: Partition, micro_steps_per_yield: int, num_macro_steps: int):
        if not isinstance(micro_steps_per_yield, int) or not isinstance(num_macro_steps, int):
            raise TypeError("micro_steps_per_yield and num_macro_steps must be integers.")
        self.micro_steps_per_yield = micro_steps_per_yield
        self.num_macro_steps = num_macro_steps
        self.macro_counter = 0
        # The base chain needs enough capacity to handle all the micro-steps.
        total_micro_steps = num_macro_steps * micro_steps_per_yield
        super().__init__(
            proposal=proposal,
            constraints=constraints,
            accept=accept,
            initial_state=initial_state,
            total_steps=total_micro_steps,
        )

    def __len__(self):
        return self.num_macro_steps

    def __next__(self):
        if self.macro_counter >= self.num_macro_steps:
            raise StopIteration
            
        for _ in range(self.micro_steps_per_yield):
            state = super().__next__()
            
        self.macro_counter += 1
        return state


def _optimization_metric_from_updater(updater_name: str) -> Callable:
    """Build a single-value optimization metric from a partition updater name."""

    def optimization_metric(partition):
        value = partition[updater_name]
        if isinstance(value, dict):
            return sum(value.values())
        if isinstance(value, (set, list)):
            return len(value)
        return value

    return optimization_metric


def create_partition_iterator(
    proposal: Callable[[Partition], Partition],
    initial_partition: Partition,
    constraints: List[Callable[[Partition], bool]],
    optimization_scheme_params: Dict,
    num_steps: int,
) -> Iterator[Partition]:
    """
    Create an iterator over partitions for the given scheme (neutral, tilted, short_bursts).

    Parameters
    ----------
    proposal : callable
        Proposal function (partition) -> partition.
    initial_partition : Partition
        Starting partition.
    constraints : list of callables
        Constraint functions for the chain/optimizer.
    optimization_scheme_params : dict
        Must contain "scheme" ("neutral", "tilted", or "short_bursts").
        For "tilted": "updater", "less_compact_probability", optional "maximize".
        For "short_bursts": "updater", optional "maximize", "burst_length", "num_bursts".
    num_steps : int
        Total steps (neutral) or steps per run (tilted/short_bursts).

    Returns
    -------
    iterator
        Yields partitions. Use in a for loop or pass to runner for saving.
    """
    scheme_params = optimization_scheme_params.copy()
    scheme = scheme_params.get("scheme", "neutral")

    if scheme == "neutral":
        chain = MarkovChain(
            proposal=proposal,
            constraints=constraints,
            accept=always_accept,
            initial_state=initial_partition,
            total_steps=num_steps,
        )
        partition_iterator = chain.with_progress_bar()
        print(f"Configured neutral run with {num_steps} steps")

    elif scheme == "tilted":
        updater_name = scheme_params.get("updater")
        p_less_compact = scheme_params.get("less_compact_probability")
        maximize = scheme_params.get("maximize", False)
        optimization_metric = _optimization_metric_from_updater(updater_name)
        optimizer = SingleMetricOptimizer(
            proposal=proposal,
            constraints=constraints,
            initial_state=initial_partition,
            optimization_metric=optimization_metric,
            maximize=maximize,
        )
        partition_iterator = optimizer.tilted_run(
            num_steps,
            p=p_less_compact,
            with_progress_bar=True,
        )
        print(f"Configured tilted run with {num_steps} steps and p={p_less_compact}")

    elif scheme == "short_bursts":
        updater_name = scheme_params.get("updater")
        maximize = scheme_params.get("maximize", False)
        optimization_metric = _optimization_metric_from_updater(updater_name)
        optimizer = SingleMetricOptimizer(
            proposal=proposal,
            constraints=constraints,
            initial_state=initial_partition,
            optimization_metric=optimization_metric,
            maximize=maximize,
        )
        burst_length = scheme_params.get("burst_length", 100)
        num_bursts = scheme_params.get("num_bursts", ceil(num_steps / burst_length))
        partition_iterator = optimizer.short_bursts(
            burst_length,
            num_bursts,
            with_progress_bar=True,
        )
        print(
            f"Configured short_bursts run with {num_bursts} bursts of {burst_length} steps each"
        )

    elif scheme == "coupon_collector":
        micro_steps_per_yield = scheme_params.get("micro_steps_per_yield", 100)
        num_macro_steps = scheme_params.get("num_macro_steps", num_steps)
        chain = CouponCollectorChain(
            proposal=proposal,
            constraints=constraints,
            accept=always_accept,
            initial_state=initial_partition,
            micro_steps_per_yield=micro_steps_per_yield,
            num_macro_steps=num_macro_steps,
        )
        partition_iterator = chain.with_progress_bar()
        print(
            f"Configured coupon_collector run with {num_macro_steps} macro steps of {micro_steps_per_yield} micro steps each"
        )

    else:
        raise ValueError(f"Unknown optimization scheme: '{scheme}'")

    return partition_iterator
