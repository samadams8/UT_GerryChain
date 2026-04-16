"""
Script version of 01_configure_sampling.ipynb, refactored to use the modern utgc API.

Copy each section below into the corresponding notebook cell.

Cell structure mirrors the original notebook:
  [CELL 1]  Imports & geography setup
  [CELL 2]  Constraints
  [CELL 3]  Region surcharges
  [CELL 4]  Shape metrics (Tilted Run)
  [CELL 5]  Lexicographic polishing
  [CELL 6]  Preconditioning
  [CELL 7]  Run setup (output_updaters + boundaries)
  [CELL 8]  MCMC loop
  [CELL 9]  Map viewer widget
  [CELL 10] End-of-ensemble figures
  [CELL 11] Uniqueness check
"""


# ── [CELL 1]  Imports & geography setup ─────────────────────────────────────
import os
from datetime import datetime as dt
from math import floor

from utgc import GeographyManager, ConfigurationManager, precondition, create_partition_iterator
import utgc.notebookhelper as nbh

# Configuration files and example maps will be saved to a directory with
# the current date and time plus an optional user-defined tag.
config_tag = "test"  # <-- Change this to something descriptive if desired

# Where the configuration file and all related results will be saved
config_dir = os.path.join("output", config_tag or dt.now().strftime("%Y%m%d%H%M%S"))

# ── Geography setup ──────────────────────────────────────────────────────────
# Data file path – the d4-cap (census block, capped epsilon=1e-3) dataset
geo = GeographyManager(
    pop_data={"d4-cap": "data/UT_capped_d4_eps1e-3.geojson"},
    crs="EPSG:26912",
)
geo.fill_empty_ids("d4-cap", ["MUNIID"])

initial_plan = "maps/US-House/2025_USH_Leg-C/2025_USH_Leg-C.shp"
num_districts = nbh.get_district_count(initial_plan)
print(f"Number of districts: {num_districts}")

# ── Bootstrap ConfigurationManager ──────────────────────────────────────────
cfg = ConfigurationManager()
cfg = cfg.set_pop_column("TOTPOP")
cfg = cfg.add_pop_dev_updater()

# ── [CELL 2]  Constraints ────────────────────────────────────────────────────
# Population deviation tolerance is passed to proposal() and precondition() directly.
# That call uses pop_tolerance=0.001 (0.1%). No separate set_pop_dev_tolerance method.
cfg = (cfg
    # How can municipalities be split?
    .constrain_region_splits(
        name="muni",
        column_id="MUNIID",
        # How many municipalities can be split total?
        num_split=floor(num_districts / 2),
        # How many splits past the first for each muni are allowed (map wide)?
        num_multi_splits=floor(num_districts / 6),
    )
    # How can counties be split?
    .constrain_region_splits(
        name="county",
        column_id="COUNTYID",
        # How many counties can be split total?
        num_split=round(num_districts / 2) + 1,
        # How many splits past the first for each county are allowed (map wide)?
        num_multi_splits=floor(num_districts / 5),
    )
    # Whether to prevent the algorithm from drawing the same map twice in a row
    .constrain_not_equal(not_equal_constraint=True)
)

# ── [CELL 3]  Region surcharges (and commented-out edge penalties) ───────────
cfg = (cfg
    ### Region surcharge configuration ###
    # Municipalities
    .surcharge_region(column_id="MUNIID", surcharge=1)
    # Counties
    .surcharge_region(column_id="COUNTYID", surcharge=0.5)
    # Institutions of higher education
    .surcharge_region(column_id="HIGHEREDID", surcharge=0.1)
    # Indian Reservations (American Indian / Alaska Native Areas)
    .surcharge_region(column_id="AIANNHID", surcharge=0.1)
    # Military Installations
    .surcharge_region(column_id="MILITID", surcharge=0.1)
    # Metro/micropolitan areas (CBSA – Core Based Statistical Areas)
    .surcharge_region(column_id="CBSAID", surcharge=0.1)

    ### Edge penalty configuration ###
    # Transitability
    # .penalize_edges_from_csv(
    #     csv_path="data/UT_blocks_transitability.csv",
    #     penalty=0.3,
    # )
)

# ── [CELL 4]  Tilted Run / Shape metrics ────────────────────────────────────
cfg = (cfg
    .add_shape_metrics(["polsby_popper"])
    # .add_optimization_scheme(
    #     scheme='short_bursts',
    #     updater='polsby_popper',
    #     burst_length=10,
    #     maximize=True,
    # )
)


# ── [CELL 5]  Lexicographic polishing ───────────────────────────────────────
# NOTE: add_lexicographic_metric and add_lexicographic_preoptimization no longer
# exist on the modern ConfigurationManager — they are legacy methods. Polishing
# is handled internally by precondition(). Nothing to configure here.


# ── [CELL 6]  Preconditioning ────────────────────────────────────────────────
print("=== Preconditioning ===")

initial_partition = geo.build_partition(
    pop_key="d4-cap",
    plan=initial_plan,
    updaters=cfg.updaters,
)
total_pop = sum(initial_partition["population"].values())
pop_tolerance = 0.001

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
    steps=25,
    max_attempts=1,
)


# ── [CELL 7]  Run setup: output_updaters + boundary files ───────────────────
import json
import utgc.plotting as gcplt

munis, counties = nbh.load_boundaries_from_shapefiles()

run_name = "testrun"
save_dir = os.path.join(config_dir, run_name)
os.makedirs(os.path.join(save_dir, "maps"), exist_ok=True)
cfg.to_config(os.path.join(save_dir, "config.yaml"))

# Serializable, analysis-relevant updaters to write to output.jsonl.
# Excludes intermediates such as population, pop_dev, perimeter, area,
# ls_muni, ls_county, raw election updaters, sb1011_data_table, etc.
output_updaters = {
    "split_muni", "split_county", "muni_multi_splits", "county_multi_splits",
    "assignment_hash", "polsby_popper",
}
print(f"Output will be saved to {save_dir}")


# ── [CELL 8]  MCMC loop ──────────────────────────────────────────────────────
num_steps = 50
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
        # Render map every step (frequency=1 for configure notebook)
        gcplt.visualize_partition(
            partition, step_number, os.path.join(save_dir, "maps"),
            counties=counties, municipalities=munis,
            split_munis_count=partition["split_muni"],
            split_counties_count=partition["split_county"],
        )
        partition.parent = None
print("Done.")


# ── [CELL 9]  Interactive map viewer widget ──────────────────────────────────
nbh.map_viewer_widget(os.path.join(config_dir, "testrun/maps"))


# ── [CELL 10]  End-of-ensemble figures ──────────────────────────────────────
import utgc.results as gcres
import matplotlib.pyplot as plt

output_path = os.path.join(config_dir, "testrun/output.jsonl")

splits = gcres.read_jsonl_table(
    output_path, ["split_muni", "split_county", "muni_multi_splits", "county_multi_splits"]
)
compactness = gcres.read_jsonl_table(output_path, "polsby_popper")
compactness = gcres.sort_subentries(compactness, "polsby_popper")

# Histogram for each column in the splits dataframe
splits.plot.hist(alpha=0.6, subplots=True, layout=(2, 2))
plt.show()

plt.figure(dpi=300)
gcplt.district_plot(compactness, relative_to_median=False)
plt.show()


# ── [CELL 11]  Uniqueness check ─────────────────────────────────────────────
hashes = gcres.read_jsonl_table(output_path, "assignment_hash")
print(f"Unique maps: {len(hashes.drop_duplicates()) / len(hashes):.1%}")
