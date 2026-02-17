"""
Precondition an initial partition to satisfy constraints before a run.

Standalone function so notebooks and the runner can reuse the same
preconditioning logic. Uses SingleMetricOptimizer to find a partition
that meets population and split tolerances, then verifies full constraints.
"""
from math import ceil
from typing import Callable, Dict, List

from gerrychain import GeographicPartition, Partition
from gerrychain.constraints import contiguous
from gerrychain.optimization import SingleMetricOptimizer

from . import run_utils as rutil


def precondition(
    initial_partition: Partition,
    proposal: Callable[[Partition], Partition],
    constraints: List[Callable[[Partition], bool]],
    constraint_params: Dict,
    population_params: Dict,
    graph,
    updaters: Dict,
    steps: int = 50,
    max_attempts: int = 1,
) -> Partition:
    """
    Precondition an initial partition so it satisfies the configured constraints
    (population deviation, split ceilings, etc.). Uses short bursts of
    SingleMetricOptimizer and optionally retries.

    Parameters
    ----------
    initial_partition : Partition
        Starting partition (e.g. config.initial_partition).
    proposal : callable
        Proposal function (e.g. from config.proposal(initial_partition)).
    constraints : list of callables
        Full constraint list used to validate the final partition.
    constraint_params : dict
        Map from constraint name to ceiling (e.g. split_muni -> 2); used
        to build the preconditioning optimization metric.
    population_params : dict
        Must contain "ideal_pop", "pop_tolerance" (or use pop_tolerance arg).
    graph : gerrychain.Graph
        Graph (for building the returned partition).
    updaters : dict
        Partition updaters (for building the returned partition).
    steps : int, optional
        Total preconditioning steps (split into 5 bursts).
    max_attempts : int, optional
        Number of attempts; if constraints are not all met, retry from
        the optimized partition.

    Returns
    -------
    Partition
        A GeographicPartition that satisfies all constraints (or best effort).
    """
    ideal_pop = population_params["ideal_pop"]
    pop_tolerance = population_params.get("pop_tolerance", 0.01)

    def _optimization_metric(partition):
        def _pop_dev(partition):
            max_deviation = 0
            for pop in partition["population"].values():
                max_deviation = max(
                    max_deviation,
                    abs(float(pop) - ideal_pop) / ideal_pop,
                )
            return max_deviation

        def _ceiling_objective(value, ceiling):
            if value > ceiling:
                return abs(value - ceiling) ** 2
            return 0

        components = [
            _ceiling_objective(_pop_dev(partition) / pop_tolerance, 1),
        ]
        for name, ceiling in constraint_params.items():
            if "split" in name:
                try:
                    num_splits = partition[name]
                except Exception:
                    print(f"  WARNING: {name} not found in partition updaters!")
                    num_splits = 0
                components.append(_ceiling_objective(num_splits, ceiling))

        return sum(components)

    precondition_constraints = [contiguous]
    if constraint_params.get("not_equal_constraint") is True:
        precondition_constraints.append(rutil.NotEqual())

    current_partition = initial_partition
    optimized_partition = None

    for attempt in range(max_attempts):
        if attempt > 0:
            print(f"  Retrying preconditioning (attempt {attempt + 1}/{max_attempts})...")
        else:
            print("Starting preconditioning...")

        optimizer = SingleMetricOptimizer(
            proposal=proposal,
            constraints=precondition_constraints,
            initial_state=current_partition,
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

        if all(c(optimized_partition) for c in constraints):
            print("  Preconditioning successful! All tolerances met.")
            break
        print("  Preconditioning failed to meet all tolerances.")
        current_partition = optimized_partition

    return GeographicPartition(
        graph,
        assignment=optimized_partition.assignment,
        parent=None,
        updaters=updaters,
    )
