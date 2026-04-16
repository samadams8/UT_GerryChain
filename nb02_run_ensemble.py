"""
Script version of 02_run_ensemble.ipynb, refactored to use the modern utgc API.

Copy each section below into the corresponding notebook cell.

Cell structure:
  [CELL 1]  Basic settings (imports + tag)
  [CELL 2]  Election updaters on top of a saved config
  [CELL 3]  Run setup (output_updaters + boundaries)
  [CELL 4]  Preconditioning + MCMC loop
  [CELL 5]  Compute metrics for comparison maps
  [CELL 6]  Load comparison maps from file (if not already in memory)
  [CELL 7]  End-of-ensemble figures (splits, compactness, partisan metrics)
"""


# ── [CELL 1]  Basic settings ─────────────────────────────────────────────────
import os
import json

from utgc import GeographyManager, ConfigurationManager, precondition, create_partition_iterator
import utgc.notebookhelper as nbh

tag = "test"  # <-- Match the tag used in notebook 01


# ── [CELL 2]  Load config, add election updaters ─────────────────────────────
# Load the configuration that was saved during notebook 01 (configure_sampling).
cfg = ConfigurationManager.from_config(f"output/{tag}/testrun/config.yaml")

# Geography (needed for election columns and to rebuild the partition)
geo = GeographyManager(
    pop_data={"d4-cap": "data/UT_capped_d4_eps1e-3.geojson"},
    crs="EPSG:26912",
)
geo.fill_empty_ids("d4-cap", ["MUNIID"])

initial_plan = "maps/US-House/2025_USH_Leg-C/2025_USH_Leg-C.shp"
num_districts = nbh.get_district_count(initial_plan)

# Build election dictionaries from the geospatial columns.
election_dicts = geo.build_election_dicts(
    "d4-cap",
    years=[2016, 2020, 2024],
    offices=["PRE", "GOV", "ATG", "AUD", "TRE"],
    parties=["D", "R", "-"],
    overrides={"2024GOV": {"R1": "G24GOVRHEN", "R2": "G24GOVNCLA"}},
)

cfg = (cfg
    .add_election_updaters(elections=election_dicts, skip_if_missing_parties=True)
    .add_election_aggregator(
        name="sb1011_data",
        elections=[
            "2016PRE", "2016GOV", "2016ATG", "2016AUD", "2016TRE",
            "2020PRE", "2020GOV", "2020ATG",
            "2024PRE", "2024GOV", "2024ATG", "2024AUD", "2024TRE",
        ],
        parties=["D", "R", "-"],
    )
    .add_election_metric_updaters(
        "sb1011_data",
        [
            "partisan_bias_utah", "partisan_bias", "mean_median",
            "efficiency_gap", "stdev_partisan_share",
            "majority_partisan_shares", "majority_seats",
        ],
    )
)


# ── [CELL 3]  Run setup: output_updaters + boundary files ───────────────────
import utgc.plotting as gcplt

munis, counties = nbh.load_boundaries_from_shapefiles()

run_name = "ensemble"
save_dir = f"output/{tag}/{run_name}"
os.makedirs(os.path.join(save_dir, "maps"), exist_ok=True)

# Serializable, analysis-relevant updaters to write to output.jsonl.
# Excludes intermediates: population, pop_dev, perimeter, area,
# ls_muni, ls_county, sb1011_data_table, sb1011_data, raw election updaters.
output_updaters = {
    "split_muni", "split_county", "muni_multi_splits", "county_multi_splits",
    "assignment_hash", "polsby_popper",
    "majority_partisan_shares", "stdev_partisan_share", "efficiency_gap",
    "mean_median", "partisan_bias", "partisan_bias_utah", "majority_seats",
}
print(f"Output will be saved to {save_dir}")


# ── [CELL 4]  Preconditioning + MCMC loop ───────────────────────────────────
print("=== Preconditioning ===")

initial_partition = geo.build_partition(
    pop_key="d4-cap",
    plan=initial_plan,
    updaters=cfg.updaters,
)
total_pop = sum(initial_partition["population"].values())
pop_tolerance = 0.01

proposal = cfg.proposal(
    initial_partition,
    total_population=total_pop,
    pop_tolerance=pop_tolerance,
)

initial_partition = precondition(
    initial_partition=initial_partition,
    proposal=proposal,
    constraints=cfg.constraints,
    constraint_params=cfg.get_constraint_params(),
    population_params={
        "ideal_pop": total_pop / num_districts,
        "pop_tolerance": pop_tolerance,
    },
    graph=geo.get_graph("d4-cap"),
    updaters=cfg.updaters,
    steps=100,
    max_attempts=1,
)

# ── MCMC ─────────────────────────────────────────────────────────────────────
num_steps = 1000
proposal = cfg.proposal(
    initial_partition,
    total_population=total_pop,
    pop_tolerance=pop_tolerance,
)
partition_iterator = create_partition_iterator(
    proposal=proposal,
    initial_partition=initial_partition,
    constraints=cfg.constraints,
    optimization_scheme_params={"scheme": "neutral"},
    num_steps=num_steps,
)

print(f"=== MCMC {run_name} ===")
print("Running Markov chain...")
output_path = os.path.join(save_dir, "output.jsonl")
assignments_path = os.path.join(save_dir, "assignments.jsonl")
with open(output_path, "w") as f, open(assignments_path, "w") as af:
    for step_number, partition in enumerate(partition_iterator, 1):
        # Save metrics (output_updaters subset only)
        data = {"step": step_number}
        for name in output_updaters:
            if name not in cfg.updaters:
                continue
            value = partition[name]
            if isinstance(value, dict):
                data[name] = {str(k): v for k, v in sorted(value.items())}
            else:
                data[name] = value
        f.write(json.dumps(data) + "\n")
        f.flush()
        # Save district assignment
        af.write(json.dumps({
            "step": step_number,
            "assignment": {str(k): int(v) for k, v in partition.assignment.items()},
        }) + "\n")
        af.flush()
        # Render map every 10 steps
        if step_number % 10 == 0:
            gcplt.visualize_partition(
                partition, step_number, os.path.join(save_dir, "maps"),
                counties=counties, municipalities=munis,
                split_munis_count=partition["split_muni"],
                split_counties_count=partition["split_county"],
            )
        partition.parent = None
print("Done.")


# ── [CELL 5]  Compute metrics for comparison maps ────────────────────────────
# NOTE: compute_metrics_for_map has been removed from ConfigurationManager.
# Rebuild comparison-map metrics by creating a partition for each shapefile,
# then reading the updaters directly.

def compute_metrics_for_map(shapefile_path, geo, cfg, pop_key="d4-cap"):
    """Return a dict of output_updater values for a given electoral map."""
    partition = geo.build_partition(
        pop_key=pop_key,
        plan=shapefile_path,
        updaters=cfg.updaters,
    )
    result = {}
    for name in output_updaters:
        if name not in cfg.updaters:
            continue
        value = partition[name]
        if isinstance(value, dict):
            result[name] = {str(k): v for k, v in sorted(value.items())}
        else:
            result[name] = value
    return result

comparison_maps = {
    "Map C": compute_metrics_for_map(
        "maps/US-House/2025_USH_Leg-C/2025_USH_Leg-C.shp", geo, cfg
    ),
    "Plaintiff 1": compute_metrics_for_map(
        "maps/US-House/2025_USH_Plaintiff-1/2025_USH_Plaintiff-1.shp", geo, cfg
    ),
    "Plaintiff 2": compute_metrics_for_map(
        "maps/US-House/2025_USH_Plaintiff-2/2025_USH_Plaintiff-2.shp", geo, cfg
    ),
    "2021 Enacted": compute_metrics_for_map(
        "maps/US-House/2021_USH_Enacted/2021_USH_Enacted.shp", geo, cfg
    ),
    "UIRC Orange": compute_metrics_for_map(
        "maps/US-House/2021_USH_UIRC-Orange/2021_USH_UIRC-Orange.shp", geo, cfg
    ),
    "UIRC Purple": compute_metrics_for_map(
        "maps/US-House/2021_USH_UIRC-Purple/2021_USH_UIRC-Purple.shp", geo, cfg
    ),
    "UIRC Public": compute_metrics_for_map(
        "maps/US-House/2021_USH_UIRC-Public/2021_USH_UIRC-Public.shp", geo, cfg
    ),
}

with open(f"output/{tag}/ensemble/comparison_maps.json", "w") as f:
    json.dump(comparison_maps, f, indent=2)


# ── [CELL 6]  Load comparison maps (if not already in memory) ────────────────
if "comparison_maps" not in dir():
    with open(f"output/{tag}/ensemble/comparison_maps.json", "r") as f:
        comparison_maps = json.load(f)


# ── [CELL 7]  End-of-ensemble figures ───────────────────────────────────────
import utgc.results as gcres
import matplotlib.pyplot as plt

output_path = os.path.join(save_dir, "output.jsonl")

# ── Splits histogram ──────────────────────────────────────────────────────────
splits = gcres.read_jsonl_table(
    output_path, ["split_muni", "split_county", "muni_multi_splits", "county_multi_splits"]
)
splits.plot.hist(alpha=0.6, subplots=True, layout=(2, 2))
plt.show()

# ── Compactness district plot ─────────────────────────────────────────────────
compactness = gcres.read_jsonl_table(output_path, "polsby_popper")
compactness = gcres.sort_subentries(compactness, "polsby_popper")

plt.figure(dpi=300)
gcplt.district_plot(compactness, relative_to_median=False)
plt.show()

# ── Partisan metrics ──────────────────────────────────────────────────────────
partisan_metrics = [
    "efficiency_gap",
    "mean_median",
    "partisan_bias",
    "partisan_bias_utah",
    "stdev_partisan_share",
    "majority_seats",
]

for metric in partisan_metrics:
    data = gcres.read_jsonl_table(output_path, metric)
    plt.figure(dpi=150)
    plt.title(metric)
    if data.shape[1] == 1:
        # Scalar metric – histogram
        data.plot.hist(alpha=0.7, legend=False)
    else:
        # Per-district metric – district plot
        data = gcres.sort_subentries(data, metric)
        gcplt.district_plot(data, relative_to_median=False)
    # Overlay comparison maps
    for label, metrics in comparison_maps.items():
        if metric in metrics:
            val = metrics[metric]
            if isinstance(val, (int, float)):
                plt.axvline(val, linestyle="--", label=label)
    plt.legend()
    plt.show()
