from gerrychain import MarkovChain, accept, GeographicPartition
from gerrychain.optimization import SingleMetricOptimizer
from .metrics import build_locality_name_maps, compute_split_name_lists, calculate_partisan_metrics
from .data_io import load_data, load_county_boundaries, load_municipality_boundaries, get_election_columns, detect_election_data, filter_elections
from .build import create_graph, create_updaters, create_initial_partition, create_constraints, create_proposal, set_random_seed
from .preconditioning import run_preconditioning
from .reporting import save_visualization, save_results, create_summary_plots
import os
import yaml

# run_ensemble and _collect_step_metrics functions remain the same as your provided file...
def run_ensemble(initial_partition, proposal, constraints_list, available_elections, counties=None, municipalities=None, num_steps=5000, visualize_every=10, vote_share_agg="median", save_visualization_fn=None, tilted_probability=1.0, compactness_score="cut_edges"):
    """Run ensemble analysis with support for both neutral and tilted sampling."""
    if tilted_probability < 1.0:
        if compactness_score == "cut_edges":
            optim = lambda partition: len(partition["cut_edges"])
        elif compactness_score == "polsby_popper":
            # Note: GerryChain's default Polsby-Popper needs to be maximized.
            optim = lambda partition: sum(partition["polsby_popper"].values())

        print(f"Running tilted-run ensemble ({compactness_score}) with {num_steps} steps and p={tilted_probability}...")
        optimizer = SingleMetricOptimizer(
            proposal=proposal,
            constraints=constraints_list,
            initial_state=initial_partition,
            optimization_metric=optim,
            maximize=(compactness_score == "polsby_popper"),
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
        "polsby_popper": dict(partition["polsby_popper"]) if "polsby_popper" in partition.updaters else {},
    }

    try:
        step_results["split_counties_count"] = int(partition["split_counties"])
        step_results["split_counties_extra_parts"] = int(partition["county_multi_splits"])
        step_results["split_munis_count"] = int(partition["split_munis"])
        step_results["split_munis_extra_parts"] = int(partition["muni_multi_splits"])
    except Exception:
        pass

    try:
        county_id_to_name, muni_id_to_name = build_locality_name_maps(partition)
        split_counties_names, split_munis_names = compute_split_name_lists(partition, county_id_to_name, muni_id_to_name)
        step_results["split_counties_names"] = split_counties_names
        step_results["split_munis_names"] = split_munis_names
    except Exception:
        pass
    
    # (The rest of your election metrics logic remains the same)
    # ...
    return step_results


class EnsembleRunner:
    """Unified orchestrator for ensemble analysis."""
    
    def __init__(self, config):
        self.config = config
        if 'random_seed' in config.get('initialization', {}):
            set_random_seed(config['initialization']['random_seed'])
        
        self.precincts, self.initial_plan = load_data(
            nodes_data_path=config['initialization']['nodes_data'],
            initial_partition_path=config['initialization']['initial_partition']
        )
        NUM_MUNIIDS = self.precincts["MUNIID"].nunique()
        NUM_COUNTYIDS = self.precincts["COUNTYID"].nunique()
        
        self.counties = load_county_boundaries(self.precincts)
        self.municipalities = load_municipality_boundaries(self.precincts)

        available_elections = detect_election_data(self.precincts)
        election_columns = get_election_columns(self.precincts)
        
        election_config = config.get('election', {})
        years = [int(x) for x in str(election_config.get('years', '')).split(',') if x.strip().isdigit()] if election_config.get('years') else None
        offices = [x.strip() for x in str(election_config.get('offices', '')).split(',') if x.strip()] if election_config.get('offices') else None
        self.available_elections = filter_elections(available_elections, years=years, offices=offices)
        self.vote_share_agg = election_config.get('vote_share_agg', 'median')
        
        self.graph = create_graph(self.precincts)
        self.updaters = create_updaters(elections=self.available_elections, election_columns=election_columns, num_muniids=NUM_MUNIIDS, num_countyids=NUM_COUNTYIDS)
        self.initial_partition = create_initial_partition(self.graph, self.precincts, self.updaters)
        
        self.ideal_population = sum(self.initial_partition["population"].values()) / len(self.initial_partition)
        self.proposal = create_proposal(
            self.ideal_population,
            self.precincts,
            config['region_surcharges'],
            config['constraints']['pop_deviation'] or 0.001,
            round(len(self.initial_partition) / 2),
            config['edge_penalties']
        )
        self.constraints = create_constraints(
            self.initial_partition,
            **config['constraints']
        )
        
    def run(self, output_dir=None, save_config=True):
        print("Starting ensemble analysis...")
        self.output_dir = output_dir
        start_partition = self._run_preconditioning()
        
        results = self._run_ensemble_internal(start_partition)
        
        if output_dir:
            self._save_results(results, output_dir)
            if save_config:
                self._save_config(output_dir)
        return results
        
    def _run_preconditioning(self):
        if not self.config.get('preconditioning', {}).get('enable', False):
            print("Preconditioning disabled, using initial partition.")
            return self.initial_partition
            
        print("Running preconditioning...")
        
        # Merge preconditioning and constraint configs to easily access them
        precond_config = {**self.config.get('preconditioning', {}), **self.config.get('constraints', {})}
        
        # Build a dictionary of valid arguments for run_preconditioning,
        # filtering out keys like 'enable' that are not part of its signature.
        valid_params = {
            'steps': precond_config.get('steps', 20),
            'popdev_tolerance': precond_config.get('pop_deviation'),
            'split_munis_tolerance': precond_config.get('split_munis_constraint'),
            'split_counties_tolerance': precond_config.get('split_counties_constraint'),
            'muni_multi_splits_tolerance': precond_config.get('muni_multi_splits_constraint'),
            'county_multi_splits_tolerance': precond_config.get('county_multi_splits_constraint'),
            'max_attempts': precond_config.get('max_repeats', 5),
            'auto_adjust_region_surcharge': precond_config.get('auto_adjust_region_surcharge', True),
            'region_adjust_factor': precond_config.get('region_adjust_factor', 1.25),
            'region_surcharge_max': precond_config.get('region_surcharge_max'),
        }
        # Filter out None values to avoid overwriting function defaults
        valid_params = {k: v for k, v in valid_params.items() if v is not None}
        
        optimized_partition = run_preconditioning(
            self.initial_partition,
            self.proposal,
            **valid_params
        )
        
        # Re-create the partition to create a clean state without parent history.
        print("Rehydrating partition from preconditioning result to ensure clean state...")
        rehydrated_partition = GeographicPartition(
            graph=self.graph,
            assignment=optimized_partition.assignment,
            updaters=self.updaters
        )

        # Check if this new, clean partition satisfies all constraints.
        def _satisfies_all(partition, constraints_list):
            for c in constraints_list:
                try:
                    if not c(partition):
                        return False
                except Exception:
                    return False
            return True

        if _satisfies_all(rehydrated_partition, self.constraints):
            print("✓ Rehydrated partition is valid and will be used as the starting point.")
            return rehydrated_partition
        else:
            print("⚠️ WARNING: Preconditioned plan is NOT valid according to the constraints.")
            print("Falling back to the original initial partition.")
            return self.initial_partition
        
    def _run_ensemble_internal(self, start_partition):
        ensemble_params = self.config['ensemble']
        tilted_params = self.config.get('tilted_run', {})
        
        return run_ensemble(
            initial_partition=start_partition,
            proposal=self.proposal,
            constraints_list=self.constraints,
            available_elections=self.available_elections,
            counties=self.counties,
            municipalities=self.municipalities,
            num_steps=ensemble_params['steps'],
            visualize_every=ensemble_params['visualize_every'],
            vote_share_agg=self.vote_share_agg,
            save_visualization_fn=self._save_visualization,
            tilted_probability=tilted_params.get('less_compact_probability', 1.0),
            compactness_score=tilted_params.get('compactness_score', 'cut_edges')
        )
    
    def _save_visualization(self, partition, step, step_results, counties=None, municipalities=None):
        """Save visualization for current step."""
        # Use provided counties/municipalities or fall back to instance variables
        if counties is None:
            counties = self.counties
        if municipalities is None:
            municipalities = self.municipalities
        save_visualization(partition, step, step_results, counties, municipalities, base_dir=self.output_dir)
    
    def _save_results(self, results, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        save_results(results, self.available_elections, mode=None, out_dir=output_dir)
        import pandas as pd
        summary_df = pd.read_csv(os.path.join(output_dir, "ensemble_summary.csv"))
        create_summary_plots(summary_df, out_dir=output_dir)
        print(f"Results saved to {output_dir}")
    
    def _save_config(self, output_dir):
        with open(os.path.join(output_dir, "params.yaml"), "w") as f:
            yaml.safe_dump(self.config, f, sort_keys=True)

