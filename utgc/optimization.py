from math import ceil
from gerrychain.optimization import SingleMetricOptimizer
from gerrychain.constraints import contiguous


def get_num_split_munis(partition):
    try:
        muni_ls = partition["muni_locality_splits"]
        return int(muni_ls.get("num_split_localities", 0))
    except Exception:
        return 0


def get_num_split_counties(partition):
    try:
        county_ls = partition["county_locality_splits"]
        return int(county_ls.get("num_split_localities", 0))
    except Exception:
        return 0


def population_deviation_objective(partition):
    total_population = sum(partition["population"].values())
    ideal_population = total_population / len(partition)
    max_deviation = 0
    for pop in partition["population"].values():
        max_deviation = max(max_deviation, abs(float(pop) - ideal_population) / ideal_population)
    return max_deviation


def combined_optimization_objective(
    partition,
    muni_surcharge=9.0,
    muni_splits_tolerance=None,
    county_surcharge=3.0,
    county_splits_tolerance=None,
    pop_tolerance=0.001,
    num_districts=4,
):
    pop_dev = population_deviation_objective(partition)
    muni_splits = get_num_split_munis(partition)
    county_splits = get_num_split_counties(partition)

    if muni_splits_tolerance is None:
        muni_splits_tolerance = 2 * num_districts
    if county_splits_tolerance is None:
        county_splits_tolerance = 2 * num_districts

    pop_component = pop_dev / pop_tolerance
    muni_component = muni_splits / muni_splits_tolerance
    county_component = county_splits / county_splits_tolerance

    return pop_component + muni_component + county_component


def run_optimization(
    initial_partition,
    proposal,
    muni_surcharge=9.0,
    county_surcharge=3.0,
    popdev_tolerance=0.001,
    optimization_steps=20,
    split_munis_tolerance=None,
    split_counties_tolerance=None,
    max_attempts=5,
):
    num_districts = len(initial_partition)
    print(f"Running optimization for {optimization_steps} steps...")
    if split_munis_tolerance is not None and split_counties_tolerance is not None:
        print(
            f"Tolerance thresholds: pop_dev={popdev_tolerance:.4f}, muni_splits={split_munis_tolerance}, county_splits={split_counties_tolerance}"
        )
    else:
        print(
            f"Tolerance thresholds: pop_dev={popdev_tolerance:.4f}, muni_splits={'unlimited' if split_munis_tolerance is None else split_munis_tolerance}, county_splits={'unlimited' if split_counties_tolerance is None else split_counties_tolerance}"
        )

    def objective_function(partition):
        return combined_optimization_objective(
            partition,
            muni_surcharge=muni_surcharge,
            county_surcharge=county_surcharge,
            pop_tolerance=popdev_tolerance,
            muni_splits_tolerance=split_munis_tolerance,
            county_splits_tolerance=split_counties_tolerance,
            num_districts=num_districts,
        )

    optimized_partition = initial_partition
    for attempt in range(max_attempts):
        if attempt > 0:
            print(f"Retrying optimization (attempt {attempt + 1}/{max_attempts})...")

        optimizer = SingleMetricOptimizer(
            proposal=proposal,
            constraints=[contiguous],
            initial_state=optimized_partition,
            optimization_metric=objective_function,
            maximize=False,
        )

        if attempt == 0:
            print("Starting optimization...")

        for i, partition in enumerate(
            optimizer.short_bursts(5, ceil(optimization_steps / 5), with_progress_bar=True)
        ):
            pass

        print(f"Optimized score: {optimizer.best_score}")

        optimized_partition = optimizer.best_part

        pop_dev = population_deviation_objective(optimized_partition)
        muni_splits = get_num_split_munis(optimized_partition)
        county_splits = get_num_split_counties(optimized_partition)

        pop_passes = pop_dev <= popdev_tolerance
        muni_passes = (split_munis_tolerance is None) or (muni_splits <= split_munis_tolerance)
        county_passes = (split_counties_tolerance is None) or (county_splits <= split_counties_tolerance)

        if pop_passes and muni_passes and county_passes:
            if attempt > 0:
                print(f"✓ Optimization successful on attempt {attempt + 1}! All tolerances met.")
            else:
                print(f"✓ Optimization successful! All tolerances met.")
            print(f"Final population deviation: {pop_dev:.6f}")
            print(f"Final municipality splits: {muni_splits}")
            print(f"Final county splits: {county_splits}")
            return optimized_partition
        else:
            if attempt < max_attempts - 1:
                print(f"✗ Attempt {attempt + 1} failed tolerance tests, retrying...")

    print(f"⚠️  WARNING: Optimization failed to meet tolerance requirements after {max_attempts} attempts")
    return optimized_partition


