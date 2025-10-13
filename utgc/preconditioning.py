from math import ceil
from gerrychain.optimization import SingleMetricOptimizer
from gerrychain.constraints import contiguous
from .metrics import build_locality_name_maps, compute_split_name_lists


def get_num_split_munis(partition):
    try:
        return int(partition["split_munis"])  # central updater
    except Exception:
        return 0


def get_num_split_counties(partition):
    try:
        return int(partition["split_counties"])  # central updater
    except Exception:
        return 0


def population_deviation_objective(partition):
    total_population = sum(partition["population"].values())
    ideal_population = total_population / len(partition)
    max_deviation = 0
    for pop in partition["population"].values():
        max_deviation = max(max_deviation, abs(float(pop) - ideal_population) / ideal_population)
    return max_deviation


def combined_preconditioning_objective(
    partition,
    muni_splits_tolerance=None,
    muni_multi_splits_tolerance=None,
    county_splits_tolerance=None,
    county_multi_splits_tolerance=None,
    pop_tolerance=0.001,
):
    pop_dev = population_deviation_objective(partition)
    muni_splits = get_num_split_munis(partition)
    county_splits = get_num_split_counties(partition)
    
    # Get multi-splits from central updaters
    try:
        muni_multi_splits = int(partition["muni_multi_splits"]) if "muni_multi_splits" in partition.updaters else 0
    except Exception:
        muni_multi_splits = 0
    try:
        county_multi_splits = int(partition["county_multi_splits"]) if "county_multi_splits" in partition.updaters else 0
    except Exception:
        county_multi_splits = 0

    def _ceiling_objective(value, ceiling):
        # When the value is above the ceiling, return a positive value that grows with squared error
        if value > ceiling:
            return abs(value - ceiling) ** 2
        else:
            return 0

    pop_component = _ceiling_objective(pop_dev/pop_tolerance, 1)
    muni_component = _ceiling_objective(muni_splits, muni_splits_tolerance)
    muni_multi_component = _ceiling_objective(muni_multi_splits, muni_multi_splits_tolerance)
    county_component = _ceiling_objective(county_splits, county_splits_tolerance)
    county_multi_component = _ceiling_objective(county_multi_splits, county_multi_splits_tolerance)

    return pop_component + muni_component + muni_multi_component + county_component + county_multi_component


def run_preconditioning(
    initial_partition,
    proposal,
    popdev_tolerance=0.001,
    num_steps=20,
    split_munis_tolerance=None,
    split_counties_tolerance=None,
    muni_multi_splits_tolerance=None,
    county_multi_splits_tolerance=None,
    max_attempts=5,
    # Optional: automatically increase region surcharges tied to failing constraints
    auto_adjust_region_surcharge=True,
    region_adjust_factor=1.25,
    region_surcharge_max=None,
):
    num_districts = len(initial_partition)
    print(f"Running preconditioning for {num_steps} steps...")
    if split_munis_tolerance is not None and split_counties_tolerance is not None:
        print(
            f"Tolerance thresholds: pop_dev={popdev_tolerance:.4f}, muni_splits={split_munis_tolerance}, county_splits={split_counties_tolerance}"
        )
    else:
        print(
            f"Tolerance thresholds: pop_dev={popdev_tolerance:.4f}, muni_splits={'unlimited' if split_munis_tolerance is None else split_munis_tolerance}, county_splits={'unlimited' if split_counties_tolerance is None else split_counties_tolerance}"
        )

    def objective_function(partition):
        return combined_preconditioning_objective(
            partition,
            pop_tolerance=popdev_tolerance,
            muni_splits_tolerance=split_munis_tolerance,
            county_splits_tolerance=split_counties_tolerance,
            muni_multi_splits_tolerance=muni_multi_splits_tolerance,
            county_multi_splits_tolerance=county_multi_splits_tolerance,
        )

    optimized_partition = initial_partition
    current_proposal = proposal

    # Extract initial surcharges from the proposal's region_surcharge
    region_surcharge_params = None
    if auto_adjust_region_surcharge:
        try:
            kw = getattr(proposal, 'keywords', {}) or {}
            rs = kw.get('region_surcharge') or {}
            # Map core columns back to the friendly keys we adjust here
            inferred = {}
            if 'MUNIID' in rs:
                inferred['muni'] = rs['MUNIID']
            if 'COUNTYID' in rs:
                inferred['county'] = rs['COUNTYID']
            # Keep other existing keys if present and numeric
            for k_col, v in rs.items():
                if k_col not in ('MUNIID', 'COUNTYID'):
                    # Pass-through by best-effort using original column key
                    inferred[k_col] = v
            if inferred:
                region_surcharge_params = inferred
                print(f"Inferred initial region surcharges from proposal: {region_surcharge_params}")
        except Exception:
            pass

    for attempt in range(max_attempts):
        if attempt > 0:
            print(f"Retrying preconditioning (attempt {attempt + 1}/{max_attempts})...")

        optimizer = SingleMetricOptimizer(
            proposal=current_proposal,
            constraints=[contiguous],
            initial_state=optimized_partition,
            optimization_metric=objective_function,
            maximize=False,
        )

        if attempt == 0:
            print("Starting preconditioning...")

        for _ in optimizer.short_bursts(
            5, ceil(num_steps / 5), with_progress_bar=True):
            pass

        print(f"Preconditioned score: {optimizer.best_score}")

        optimized_partition = optimizer.best_part

        pop_dev = population_deviation_objective(optimized_partition)
        muni_splits = get_num_split_munis(optimized_partition)
        county_splits = get_num_split_counties(optimized_partition)
        
        # Get multi-splits from central updaters
        try:
            muni_multi_splits = int(optimized_partition["muni_multi_splits"])
        except (KeyError, TypeError):
            muni_multi_splits = 0
        
        try:
            county_multi_splits = int(optimized_partition["county_multi_splits"])
        except (KeyError, TypeError):
            county_multi_splits = 0

        # Report which municipalities and counties are split and multi-split
        try:
            county_id_to_name, muni_id_to_name = build_locality_name_maps(optimized_partition)
            split_counties_names, split_munis_names = compute_split_name_lists(optimized_partition, county_id_to_name, muni_id_to_name)
            # multi-split = municipalities appearing in 3+ districts
            muni_to_districts = {}
            for node in optimized_partition.graph.nodes:
                node_data = optimized_partition.graph.nodes[node]
                muni_id = node_data.get("MUNIID")
                if muni_id:
                    dist = optimized_partition.assignment[node]
                    if muni_id not in muni_to_districts:
                        muni_to_districts[muni_id] = set()
                    muni_to_districts[muni_id].add(dist)
            multi_split_muni_ids = sorted([mid for mid, dists in muni_to_districts.items() if len(dists) > 2])
            multi_split_muni_names = [muni_id_to_name.get(mid, str(mid)) for mid in multi_split_muni_ids]
            print("Split municipalities:", ", ".join(split_munis_names) if split_munis_names else "None")
            print("Multi-split municipalities:", ", ".join(sorted(multi_split_muni_names)) if multi_split_muni_names else "None")
            # multi-split = counties appearing in 3+ districts
            county_to_districts = {}
            for node in optimized_partition.graph.nodes:
                node_data = optimized_partition.graph.nodes[node]
                county_id = node_data.get("COUNTYID")
                if county_id:
                    dist = optimized_partition.assignment[node]
                    if county_id not in county_to_districts:
                        county_to_districts[county_id] = set()
                    county_to_districts[county_id].add(dist)
            multi_split_county_ids = sorted([cid for cid, dists in county_to_districts.items() if len(dists) > 2])
            multi_split_county_names = [county_id_to_name.get(cid, str(cid)) for cid in multi_split_county_ids]
            print("Split counties:", ", ".join(split_counties_names) if split_counties_names else "None")
            print("Multi-split counties:", ", ".join(sorted(multi_split_county_names)) if multi_split_county_names else "None")
        except Exception:
            pass

        pop_passes = pop_dev <= popdev_tolerance
        muni_passes = (split_munis_tolerance is None) or (muni_splits <= split_munis_tolerance)
        muni_multi_passes = (muni_multi_splits_tolerance is None) or (muni_multi_splits <= muni_multi_splits_tolerance)
        county_passes = (split_counties_tolerance is None) or (county_splits <= split_counties_tolerance)
        county_multi_passes = (county_multi_splits_tolerance is None) or (county_multi_splits <= county_multi_splits_tolerance)

        if pop_passes and muni_passes and muni_multi_passes and county_passes and county_multi_passes:
            if attempt > 0:
                print(f"✓ Preconditioning successful on attempt {attempt + 1}! All tolerances met.")
            else:
                print(f"✓ Preconditioning successful! All tolerances met.")
            print(f"Final population deviation: {pop_dev:.6f}")
            print(f"Final municipality splits: {muni_splits}")
            print(f"Final municipality multi-splits: {muni_multi_splits}")
            print(f"Final county splits: {county_splits}")
            print(f"Final county multi-splits: {county_multi_splits}")
            return optimized_partition
        else:
            if attempt < max_attempts - 1:
                print(f"✗ Attempt {attempt + 1} failed tolerance tests, retrying...")
                # Optionally escalate region surcharges associated with failing constraints
                if auto_adjust_region_surcharge and region_surcharge_params is not None:
                    updated = False
                    # Increase municipality-related surcharge if muni constraints failed
                    if (split_munis_tolerance is not None and muni_splits > split_munis_tolerance) or (
                        muni_multi_splits_tolerance is not None and muni_multi_splits > muni_multi_splits_tolerance
                    ):
                        if 'muni' in region_surcharge_params:
                            old_val = region_surcharge_params['muni']
                            new_val = old_val * float(region_adjust_factor)
                            if region_surcharge_max is not None:
                                try:
                                    new_val = min(new_val, float(region_surcharge_max))
                                except Exception:
                                    pass
                            region_surcharge_params['muni'] = new_val
                            print(f"  ↳ Increasing municipality region surcharge: ({old_val}) -> ({region_surcharge_params['muni']})")
                            updated = True
                    # Increase county-related surcharge if county constraints failed
                    if (split_counties_tolerance is not None and county_splits > split_counties_tolerance) or (
                        county_multi_splits_tolerance is not None and county_multi_splits > county_multi_splits_tolerance
                    ):
                        if 'county' in region_surcharge_params:
                            old_val = region_surcharge_params['county']
                            new_val = old_val * float(region_adjust_factor)
                            if region_surcharge_max is not None:
                                try:
                                    new_val = min(new_val, float(region_surcharge_max))
                                except Exception:
                                    pass
                            region_surcharge_params['county'] = new_val
                            print(f"  ↳ Increasing county region surcharge: ({old_val}) -> ({region_surcharge_params['county']})")
                            updated = True
                    # Update proposal in-place if anything changed
                    if updated:
                        try:
                            kw = getattr(current_proposal, 'keywords', {}) or {}
                            rs = kw.get('region_surcharge')
                            if isinstance(rs, dict):
                                # Update known columns in-place
                                if 'muni' in region_surcharge_params:
                                    rs['MUNIID'] = region_surcharge_params['muni']
                                if 'county' in region_surcharge_params:
                                    rs['COUNTYID'] = region_surcharge_params['county']
                                print(f"  ↳ Updated proposal.region_surcharge in-place: {rs}")
                            else:
                                print("  ↳ Could not access proposal.region_surcharge for in-place update")
                        except Exception as e:
                            print(f"  ↳ Failed to update proposal in-place after surcharge update: {e}")

    print(f"⚠️  WARNING: Preconditioning failed to meet tolerance requirements after {max_attempts} attempts")
    return optimized_partition
