from gerrychain import MarkovChain, accept
from gerrychain.optimization import SingleMetricOptimizer
from .metrics import build_locality_name_maps, compute_split_name_lists, calculate_partisan_metrics

def run_ensemble(initial_partition, proposal, constraints_list, available_elections, counties=None, municipalities=None, num_steps=5000, visualize_every=10, vote_share_agg="median", save_visualization_fn=None):
    """Run the ensemble analysis."""
    print(f"Running ensemble analysis with {num_steps} steps...")

    chain = MarkovChain(
        proposal=proposal,
        constraints=constraints_list,
        accept=accept.always_accept,
        initial_state=initial_partition,
        total_steps=num_steps,
    )

    results = []

    for i, partition in enumerate(chain.with_progress_bar()):
        pop_dict = dict(partition["population"]) if "population" in partition.updaters else {}
        if pop_dict:
            ideal_pop = sum(pop_dict.values()) / len(pop_dict)
            pop_dev = {k: (abs(float(v) - ideal_pop) / ideal_pop if ideal_pop > 0 else None) for k, v in pop_dict.items()}
        else:
            pop_dev = {}
        step_results = {
            "step": i,
            "population_deviation": pop_dev,
            "vote_share_agg": vote_share_agg,
            "num_cut_edges": len(partition["cut_edges"]),
        }

        try:
            county_ls = partition["county_locality_splits"]
            step_results["split_counties_count"] = county_ls.get("num_split_localities", 0)
            total_counties = len(set(partition.graph.nodes[node].get("COUNTYID") for node in partition.graph.nodes if partition.graph.nodes[node].get("COUNTYID")))
            step_results["split_counties_extra_parts"] = county_ls.get("num_parts", 0) - total_counties
        except Exception as e:
            step_results["split_counties_count"] = 0
            step_results["split_counties_extra_parts"] = 0

        try:
            muni_ls = partition["muni_locality_splits"]
            step_results["split_munis_count"] = muni_ls.get("num_split_localities", 0)
            total_munis = len(set(partition.graph.nodes[node].get("MUNIID") for node in partition.graph.nodes if partition.graph.nodes[node].get("MUNIID")))
            step_results["split_munis_extra_parts"] = muni_ls.get("num_parts", 0) - total_munis
        except Exception as e:
            step_results["split_munis_count"] = 0
            step_results["split_munis_extra_parts"] = 0

        county_id_to_name, muni_id_to_name = build_locality_name_maps(partition)
        split_counties_names, split_munis_names = compute_split_name_lists(partition, county_id_to_name, muni_id_to_name)
        step_results["split_counties_names"] = split_counties_names
        step_results["split_munis_names"] = split_munis_names

        rep_shares_matrix = []
        for election in available_elections:
            if election in partition.updaters:
                election_results = partition[election]
                try:
                    rep_votes = list(election_results.votes("Republican"))
                    dem_votes = list(election_results.votes("Democratic"))
                except Exception as e:
                    rep_votes = []
                    dem_votes = []

                shares = []
                n = min(len(rep_votes), len(dem_votes))
                for j in range(n):
                    r = rep_votes[j] or 0
                    d = dem_votes[j] or 0
                    total = r + d
                    shares.append((r / total) if total > 0 else None)

                if vote_share_agg == "none":
                    try:
                        step_results[f"{election}_Republican_total"] = sum(rep_votes)
                    except Exception:
                        step_results[f"{election}_Republican_total"] = 0
                    step_results[f"{election}_Republican_votes_by_district"] = rep_votes
                    try:
                        step_results[f"{election}_Republican_wins"] = election_results.wins("Republican")
                    except Exception:
                        step_results[f"{election}_Republican_wins"] = 0
                    step_results[f"{election}_Republican_share_by_district"] = shares
                    margins_pct = []
                    for j in range(n):
                        r = rep_votes[j] or 0
                        d = dem_votes[j] or 0
                        total = r + d
                        margins_pct.append(((r - d) / total) if total > 0 else None)
                    step_results[f"{election}_margin_pct_by_district"] = margins_pct
                else:
                    if shares:
                        rep_shares_matrix.append(shares)

        if vote_share_agg in ("median", "mean") and len(available_elections) > 0:
            import statistics
            try:
                district_ids = list(partition.parts)
                num_districts = len(district_ids)
                rep_agg = []
                if rep_shares_matrix:
                    n = min(len(row) for row in rep_shares_matrix)
                    for j in range(n):
                        vals = [row[j] for row in rep_shares_matrix if row[j] is not None]
                        if len(vals) == 0:
                            rep_agg.append(None)
                        else:
                            rep_agg.append(statistics.median(vals) if vote_share_agg == "median" else sum(vals) / len(vals))
                    rep_agg_sorted = sorted(rep_agg)
                    step_results["Republican_agg_share_by_district"] = rep_agg_sorted
                    rep_seats = sum(1 for v in rep_agg_sorted if v is not None and v > 0.5)
                    step_results["Republican_agg_seats"] = int(rep_seats)
                    valid_shares = [v for v in rep_agg_sorted if v is not None]
                    if len(valid_shares) > 0:
                        try:
                            mean_share = sum(valid_shares) / len(valid_shares)
                            median_share = statistics.median(valid_shares)
                            step_results["mean_median"] = float(mean_share - median_share)
                            above_mean = sum(1 for v in valid_shares if v > mean_share)
                            step_results["partisan_bias"] = float(above_mean / len(valid_shares) - 0.5)
                            wasted_R = 0.0
                            wasted_D = 0.0
                            for s in valid_shares:
                                if s > 0.5:
                                    wasted_R += s - 0.5
                                    wasted_D += 1 - s
                                else:
                                    wasted_R += s
                                    wasted_D += 0.5 - s
                            step_results["efficiency_gap"] = float((wasted_D - wasted_R) / len(valid_shares))
                            sorted_shares = sorted(valid_shares)
                            n = len(sorted_shares)
                            if n > 1:
                                gini = 1.0 - 2.0 * sum((j + 1) * x for j, x in enumerate(sorted_shares)) / (n * sum(sorted_shares))
                                step_results["partisan_gini"] = float(gini)
                            else:
                                step_results["partisan_gini"] = None
                        except Exception as e:
                            pass
            except Exception as e:
                pass

        if vote_share_agg == "none":
            partisan_metrics = calculate_partisan_metrics(partition, available_elections)
            step_results.update(partisan_metrics)

        results.append(step_results)

        if i % visualize_every == 0 and save_visualization_fn is not None:
            save_visualization_fn(partition, i, step_results, counties, municipalities)

    return results

def run_ensemble_tilted(
    initial_partition,
    proposal,
    constraints_list,
    available_elections,
    counties=None,
    municipalities=None,
    num_steps=5000,
    visualize_every=10,
    vote_share_agg="median",
    save_visualization_fn=None,
    p=0.125,
):
    """Run ensemble analysis using a tilted-run optimizer that minimizes cut edges.

    This prefers more compact plans by minimizing the number of cut edges, following
    the tilted-run method described in the GerryChain optimization docs.
    """
    print(f"Running tilted-run ensemble (minimize cut edges) with {num_steps} steps and p={p}...")

    optimizer = SingleMetricOptimizer(
        proposal=proposal,
        constraints=constraints_list,
        initial_state=initial_partition,
        optimization_metric=lambda partition: len(partition["cut_edges"]),
        maximize=False,
    )

    results = []

    for i, partition in enumerate(optimizer.tilted_run(num_steps, p=p, with_progress_bar=True)):
        pop_dict = dict(partition["population"]) if "population" in partition.updaters else {}
        if pop_dict:
            ideal_pop = sum(pop_dict.values()) / len(pop_dict)
            pop_dev = {k: (abs(float(v) - ideal_pop) / ideal_pop if ideal_pop > 0 else None) for k, v in pop_dict.items()}
        else:
            pop_dev = {}
        step_results = {
            "step": i,
            "population_deviation": pop_dev,
            "vote_share_agg": vote_share_agg,
            "num_cut_edges": len(partition["cut_edges"]),
        }

        try:
            county_ls = partition["county_locality_splits"]
            step_results["split_counties_count"] = county_ls.get("num_split_localities", 0)
            total_counties = len(set(partition.graph.nodes[node].get("COUNTYID") for node in partition.graph.nodes if partition.graph.nodes[node].get("COUNTYID")))
            step_results["split_counties_extra_parts"] = county_ls.get("num_parts", 0) - total_counties
        except Exception as e:
            step_results["split_counties_count"] = 0
            step_results["split_counties_extra_parts"] = 0

        try:
            muni_ls = partition["muni_locality_splits"]
            step_results["split_munis_count"] = muni_ls.get("num_split_localities", 0)
            total_munis = len(set(partition.graph.nodes[node].get("MUNIID") for node in partition.graph.nodes if partition.graph.nodes[node].get("MUNIID")))
            step_results["split_munis_extra_parts"] = muni_ls.get("num_parts", 0) - total_munis
        except Exception as e:
            step_results["split_munis_count"] = 0
            step_results["split_munis_extra_parts"] = 0

        county_id_to_name, muni_id_to_name = build_locality_name_maps(partition)
        split_counties_names, split_munis_names = compute_split_name_lists(partition, county_id_to_name, muni_id_to_name)
        step_results["split_counties_names"] = split_counties_names
        step_results["split_munis_names"] = split_munis_names

        rep_shares_matrix = []
        for election in available_elections:
            if election in partition.updaters:
                election_results = partition[election]
                try:
                    rep_votes = list(election_results.votes("Republican"))
                    dem_votes = list(election_results.votes("Democratic"))
                except Exception as e:
                    rep_votes = []
                    dem_votes = []

                shares = []
                n = min(len(rep_votes), len(dem_votes))
                for j in range(n):
                    r = rep_votes[j] or 0
                    d = dem_votes[j] or 0
                    total = r + d
                    shares.append((r / total) if total > 0 else None)

                if vote_share_agg == "none":
                    try:
                        step_results[f"{election}_Republican_total"] = sum(rep_votes)
                    except Exception:
                        step_results[f"{election}_Republican_total"] = 0
                    step_results[f"{election}_Republican_votes_by_district"] = rep_votes
                    try:
                        step_results[f"{election}_Republican_wins"] = election_results.wins("Republican")
                    except Exception:
                        step_results[f"{election}_Republican_wins"] = 0
                    step_results[f"{election}_Republican_share_by_district"] = shares
                    margins_pct = []
                    for j in range(n):
                        r = rep_votes[j] or 0
                        d = dem_votes[j] or 0
                        total = r + d
                        margins_pct.append(((r - d) / total) if total > 0 else None)
                    step_results[f"{election}_margin_pct_by_district"] = margins_pct
                else:
                    if shares:
                        rep_shares_matrix.append(shares)

        if vote_share_agg in ("median", "mean") and len(available_elections) > 0:
            import statistics
            try:
                district_ids = list(partition.parts)
                num_districts = len(district_ids)
                rep_agg = []
                if rep_shares_matrix:
                    n = min(len(row) for row in rep_shares_matrix)
                    for j in range(n):
                        vals = [row[j] for row in rep_shares_matrix if row[j] is not None]
                        if len(vals) == 0:
                            rep_agg.append(None)
                        else:
                            rep_agg.append(statistics.median(vals) if vote_share_agg == "median" else sum(vals) / len(vals))
                    rep_agg_sorted = sorted(rep_agg)
                    step_results["Republican_agg_share_by_district"] = rep_agg_sorted
                    rep_seats = sum(1 for v in rep_agg_sorted if v is not None and v > 0.5)
                    step_results["Republican_agg_seats"] = int(rep_seats)
                    valid_shares = [v for v in rep_agg_sorted if v is not None]
                    if len(valid_shares) > 0:
                        try:
                            mean_share = sum(valid_shares) / len(valid_shares)
                            median_share = statistics.median(valid_shares)
                            step_results["mean_median"] = float(mean_share - median_share)
                            above_mean = sum(1 for v in valid_shares if v > mean_share)
                            step_results["partisan_bias"] = float(above_mean / len(valid_shares) - 0.5)
                            wasted_R = 0.0
                            wasted_D = 0.0
                            for s in valid_shares:
                                if s > 0.5:
                                    wasted_R += s - 0.5
                                    wasted_D += 1 - s
                                else:
                                    wasted_R += s
                                    wasted_D += 0.5 - s
                            step_results["efficiency_gap"] = float((wasted_D - wasted_R) / len(valid_shares))
                            sorted_shares = sorted(valid_shares)
                            n = len(sorted_shares)
                            if n > 1:
                                gini = 1.0 - 2.0 * sum((j + 1) * x for j, x in enumerate(sorted_shares)) / (n * sum(sorted_shares))
                                step_results["partisan_gini"] = float(gini)
                            else:
                                step_results["partisan_gini"] = None
                        except Exception as e:
                            pass
            except Exception as e:
                pass

        if vote_share_agg == "none":
            partisan_metrics = calculate_partisan_metrics(partition, available_elections)
            step_results.update(partisan_metrics)

        results.append(step_results)

        if i % visualize_every == 0 and save_visualization_fn is not None:
            save_visualization_fn(partition, i, step_results, counties, municipalities)

    return results

