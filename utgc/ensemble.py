from gerrychain import MarkovChain, accept
from gerrychain.optimization import SingleMetricOptimizer
from .metrics import build_locality_name_maps, compute_split_name_lists, calculate_partisan_metrics
from .data_io import load_data, load_county_boundaries, load_municipality_boundaries, get_election_columns, detect_election_data, filter_elections
from .build import create_graph, create_updaters, create_initial_partition, create_constraints, create_proposal, set_random_seed
from .preconditioning import run_preconditioning
from .reporting import save_visualization, save_results, create_summary_plots
import os
import yaml

def run_ensemble(initial_partition, proposal, constraints_list, available_elections, counties=None, municipalities=None, num_steps=5000, visualize_every=10, vote_share_agg="median", save_visualization_fn=None, tilted_probability=1.0):
    """Run ensemble analysis with support for both neutral and tilted sampling.
    
    Args:
        initial_partition: Starting partition
        proposal: ReCom proposal function
        constraints_list: List of constraints
        available_elections: List of election columns
        counties: County boundaries for visualization
        municipalities: Municipality boundaries for visualization
        num_steps: Number of steps to run
        visualize_every: How often to save visualizations
        vote_share_agg: How to aggregate vote shares ("median", "mean", "none")
        save_visualization_fn: Function to save visualizations
        tilted_probability: Probability for tilted sampling (1.0 = neutral, <1.0 = tilted)
    """
    # Decision logic: use tilted sampling if probability < 1.0
    if tilted_probability < 1.0:
        print(f"Running tilted-run ensemble (minimize cut edges) with {num_steps} steps and p={tilted_probability}...")
        optimizer = SingleMetricOptimizer(
            proposal=proposal,
            constraints=constraints_list,
            initial_state=initial_partition,
            optimization_metric=lambda partition: len(partition["cut_edges"]),
            maximize=False,
        )
        partition_iterator = optimizer.tilted_run(num_steps, p=tilted_probability, with_progress_bar=True)
    else:
        print(f"Running neutral ensemble analysis with {num_steps} steps...")
        chain = MarkovChain(
            proposal=proposal,
            constraints=constraints_list,
            accept=accept.always_accept,
            initial_state=initial_partition,
            total_steps=num_steps,
        )
        partition_iterator = chain.with_progress_bar()

    results = []
    for i, partition in enumerate(partition_iterator):
        step_results = _collect_step_metrics(partition, i, available_elections, vote_share_agg)
        results.append(step_results)

        if i % visualize_every == 0 and save_visualization_fn is not None:
            save_visualization_fn(partition, i, step_results, counties, municipalities)

    return results


def _collect_step_metrics(partition, step, available_elections, vote_share_agg):
    """Extract shared metrics collection logic for both neutral and tilted runs."""
    pop_dict = dict(partition["population"]) if "population" in partition.updaters else {}
    if pop_dict:
        ideal_pop = sum(pop_dict.values()) / len(pop_dict)
        pop_dev = {k: (abs(float(v) - ideal_pop) / ideal_pop if ideal_pop > 0 else None) for k, v in pop_dict.items()}
    else:
        pop_dev = {}
        
    step_results = {
        "step": step,
        "population_deviation": pop_dev,
        "vote_share_agg": vote_share_agg,
        "num_cut_edges": len(partition["cut_edges"]),
    }

    # County splits
    try:
        county_ls = partition["county_locality_splits"]
        step_results["split_counties_count"] = county_ls.get("num_split_localities", 0)
        total_counties = len(set(partition.graph.nodes[node].get("COUNTYID") for node in partition.graph.nodes if partition.graph.nodes[node].get("COUNTYID")))
        step_results["split_counties_extra_parts"] = county_ls.get("num_parts", 0) - county_ls.get("num_split_localities", 0) - total_counties
    except Exception:
        step_results["split_counties_count"] = 0
        step_results["split_counties_extra_parts"] = 0

    # Municipality splits
    try:
        muni_ls = partition["muni_locality_splits"]
        step_results["split_munis_count"] = muni_ls.get("num_split_localities", 0)
        total_munis = len(set(partition.graph.nodes[node].get("MUNIID") for node in partition.graph.nodes if partition.graph.nodes[node].get("MUNIID")))
        step_results["split_munis_extra_parts"] = muni_ls.get("num_parts", 0) - muni_ls.get("num_split_localities", 0) - total_munis
    except Exception:
        step_results["split_munis_count"] = 0
        step_results["split_munis_extra_parts"] = 0

    # Split names for reporting
    county_id_to_name, muni_id_to_name = build_locality_name_maps(partition)
    split_counties_names, split_munis_names = compute_split_name_lists(partition, county_id_to_name, muni_id_to_name)
    step_results["split_counties_names"] = split_counties_names
    step_results["split_munis_names"] = split_munis_names

    # Election metrics
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

    return step_results

class EnsembleRunner:
    """Unified orchestrator for ensemble analysis that eliminates code duplication."""
    
    def __init__(self, config):
        """Initialize runner from config dict matching notebook structure.
        
        Config structure:
        {
            'initialization': {'nodes_data': '...', 'initial_partition': '...', 'random_seed': 1847},
            'constraints': {'pop_deviation': 0.001, 'split_munis_constraint': 4, ...},
            'region_surcharges': {'muni': 3, 'county': 2, ...},
            'tilted_run': {'less_compact_probability': 0.5},
            'ensemble': {'steps': 51, 'visualize_every': 1},
            'preconditioning': {'enable': True, 'steps': 20, ...},
            'election': {'years': '2016,2020,2024', 'offices': 'PRE,GOV,ATG,AUD,TRE', 'vote_share_agg': 'median'}
        }
        """
        self.config = config
        
        # Set random seed if provided
        if 'random_seed' in config.get('initialization', {}):
            set_random_seed(config['initialization']['random_seed'])
        
        # Load data from config paths
        self.precincts, self.initial_plan = load_data(
            nodes_data_path=config['initialization']['nodes_data'],
            initial_partition_path=config['initialization']['initial_partition']
        )
        
        # Load boundary data for visualization
        self.counties = load_county_boundaries(self.precincts)
        self.municipalities = load_municipality_boundaries(self.precincts)
        
        # Detect and filter elections
        available_elections = detect_election_data(self.precincts)
        election_columns = get_election_columns(self.precincts)
        
        # Filter elections based on config
        election_config = config.get('election', {})
        years = [int(x) for x in str(election_config.get('years', '')).split(',') if x.strip().isdigit()] if election_config.get('years') else None
        offices = [x.strip() for x in str(election_config.get('offices', '')).split(',') if x.strip()] if election_config.get('offices') else None
        self.available_elections = filter_elections(available_elections, years=years, offices=offices)
        self.vote_share_agg = election_config.get('vote_share_agg', 'median')
        
        # Build graph and partition
        self.graph = create_graph(self.precincts)
        self.updaters = create_updaters(elections=self.available_elections, 
                                       election_columns=election_columns)
        self.initial_partition = create_initial_partition(self.graph, 
                                                         self.precincts, 
                                                         self.updaters)
        
        # Build proposal and constraints from config
        self.ideal_population = sum(self.initial_partition["population"].values()) / len(self.initial_partition)
        self.proposal = create_proposal(self.ideal_population, 
                                       self.precincts, 
                                       config['region_surcharges'])
        self.constraints = create_constraints(self.initial_partition, 
                                             **config['constraints'])
        
    def run(self, output_dir=None, save_config=True):
        """Orchestrate full pipeline: preconditioning -> ensemble -> save results."""
        print("Starting ensemble analysis...")
        
        # Run preconditioning if enabled
        start_partition = self._run_preconditioning()
        
        # Run ensemble (neutral or tilted based on config)
        results = self._run_ensemble_internal(start_partition)
        
        # Save results and visualizations
        if output_dir:
            self._save_results(results, output_dir)
            if save_config:
                self._save_config(output_dir)
        
        return results
        
    def _run_preconditioning(self):
        """Run preconditioning with merged constraint parameters."""
        if not self.config.get('preconditioning', {}).get('enable', False):
            print("Preconditioning disabled, using initial partition")
            return self.initial_partition
            
        print("Running preconditioning...")
        
        # Merge preconditioning and constraint params
        precond_params = {**self.config['preconditioning'], **self.config['constraints']}
        
        # Extract parameters that run_preconditioning expects, including surcharges
        valid_params = {
            'steps': precond_params.get('steps', 20),
            'split_munis_tolerance': precond_params.get('split_munis_constraint'),
            'split_counties_tolerance': precond_params.get('split_counties_constraint'),
            'muni_multi_splits_tolerance': precond_params.get('muni_multi_splits_constraint'),
            'county_multi_splits_tolerance': precond_params.get('county_multi_splits_constraint'),
        }
        
        optimized_partition = run_preconditioning(
            self.initial_partition,
            self.proposal,
            **valid_params
        )
        
        # Check if optimized partition satisfies constraints
        def _satisfies_all(partition, constraints_list):
            for c in constraints_list:
                try:
                    if not c(partition):
                        return False
                except Exception:
                    return False
            return True

        if _satisfies_all(optimized_partition, self.constraints):
            print("Using optimized partition as starting point.")
            return optimized_partition
        elif _satisfies_all(self.initial_partition, self.constraints):
            print("Optimized partition failed constraints; using initial partition.")
            return self.initial_partition
        else:
            print("Neither optimized nor initial partition meets all constraints; proceeding with initial partition.")
            return self.initial_partition
        
    def _run_ensemble_internal(self, start_partition):
        """Single implementation for both neutral and tilted runs."""
        ensemble_params = self.config['ensemble']
        steps = ensemble_params['steps']
        visualize_every = ensemble_params['visualize_every']
        
        # Get tilted probability from config (default 1.0 for neutral)
        tilted_params = self.config.get('tilted_run', {})
        tilted_probability = tilted_params.get('less_compact_probability', 1.0)
        
        # Use unified run_ensemble function with single probability parameter
        return run_ensemble(
            initial_partition=start_partition,
            proposal=self.proposal,
            constraints_list=self.constraints,
            available_elections=self.available_elections,
            counties=self.counties,
            municipalities=self.municipalities,
            num_steps=steps,
            visualize_every=visualize_every,
            vote_share_agg=self.vote_share_agg,
            save_visualization_fn=self._save_visualization,
            tilted_probability=tilted_probability
        )
    
    
    
    def _save_visualization(self, partition, step, step_results, counties=None, municipalities=None):
        """Save visualization for current step."""
        # Use provided counties/municipalities or fall back to instance variables
        if counties is None:
            counties = self.counties
        if municipalities is None:
            municipalities = self.municipalities
        save_visualization(partition, step, step_results, counties, municipalities)
    
    def _save_results(self, results, output_dir):
        """Save results and create visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine mode based on whether elections are available
        mode = "neutral" if len(self.available_elections) == 0 else None
        
        # Save results using existing reporting functions
        save_results(results, self.available_elections, mode=mode, out_dir=output_dir)
        
        # Create summary plots
        import pandas as pd
        summary_df = pd.read_csv(os.path.join(output_dir, "ensemble_summary.csv"))
        create_summary_plots(summary_df, out_dir=output_dir)
        
        print(f"Results saved to {output_dir}")
    
    def _save_config(self, output_dir):
        """Save the configuration used for this run."""
        config_path = os.path.join(output_dir, "params.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(self.config, f, sort_keys=True)
        print(f"Configuration saved to {config_path}")
