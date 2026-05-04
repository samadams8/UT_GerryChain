import os
from datetime import datetime as dt
from math import floor, ceil
import json

from utgc import GeographyManager, ConfigurationManager, precondition
import utgc.notebookhelper as nbh
from utgc.chain import CouponCollectorChain, coupon_collector_expectation
from gerrychain.chain import MarkovChain

params = nbh.get_notebook_params("us_house")
num_steps = 100

# Configuration files and example maps will be saved to a tagged directory
config_tag = params["prefix"] + ""  # <-- Change this to something descriptive if desired

# Where the configuration file and all related results will be saved
config_dir = os.path.join("output", config_tag or dt.now().strftime("%Y%m%d%H%M%S"))
run_name = "ensemble"
save_dir = os.path.join(config_dir, run_name)

# Data file path
geo = GeographyManager(
    pop_data={
        params["data_tag"]: {
            "data": params["data_path"],
            "transitability": params["transitability_path"],
        }
    },
    crs="EPSG:26912",
)
geo.fill_empty_ids(params["data_tag"], ["MUNIID"])

import glob

initial_plan = params["init_plan_path"]
start_step = 0
shapefiles_dir = os.path.join(save_dir, "maps", "shapefiles")
if os.path.exists(shapefiles_dir):
    saved_zips = glob.glob(os.path.join(shapefiles_dir, "step_*.zip"))
    if saved_zips:
        last_file = max(saved_zips)
        filename = os.path.basename(last_file)
        start_step = int(filename.split("_")[1].split(".")[0]) + 1
        initial_plan = last_file
        print(f"Resuming from step {start_step-1} using {last_file}")

num_districts = nbh.get_district_count(initial_plan)
print(f"Number of districts: {num_districts}")

cfg = ConfigurationManager()
cfg = cfg.set_pop_column("TOTPOP")
cfg = cfg.add_pop_dev_updater()

# Population deviation tolerance is passed to proposal() and precondition() directly.
# That call uses pop_tolerance=0.001 (0.1%). No separate set_pop_dev_tolerance method.
cfg = (cfg
    # How can municipalities be split?
    .constrain_region_splits(
        name="muni",
        column_id="MUNIID",
        # How many municipalities can be split total?
        num_split=num_districts-1, #floor(num_districts / 2),
        # How many splits past the first for each muni are allowed (map wide)?
        num_multi_splits=num_districts-1, #floor(num_districts / 6),
    )
    # No muni split constraint--just track it.
    # .add_locality_splits_updater(
    #     name="muni",
    #     column_id="MUNIID",
    # )
    # How can counties be split?
    .constrain_region_splits(
        name="county",
        column_id="COUNTYID",
        # How many counties can be split total?
        num_split=num_districts-1, #round(num_districts / 2) + 1,
        # How many splits past the first for each county are allowed (map wide)?
        num_multi_splits=num_districts-1, #floor(num_districts / 5),
    )
    # Whether to prevent the algorithm from drawing the same map twice in a row
    .constrain_not_equal(not_equal_constraint=True)
)

cfg = (cfg
    ### Region surcharge configuration ###
    # Municipalities
    .surcharge_region(column_id="MUNIID", surcharge=1)
    # Counties
    # .surcharge_region(column_id="COUNTYID", surcharge=0.8)
    .surcharge_region(column_id="COUNTYID", surcharge=0.5)
    # Institutions of higher education
    .surcharge_region(column_id="HIGHEREDID", surcharge=0.2)
    # Indian Reservations (American Indian / Alaska Native Areas)
    .surcharge_region(column_id="AIANNHID", surcharge=0.2)
    # Military Installations
    .surcharge_region(column_id="MILITID", surcharge=0.2)
    # Metro/micropolitan areas (CBSA – Core Based Statistical Areas)
    .surcharge_region(column_id="CBSAID", surcharge=0.1)

    ### Edge penalty configuration ###
    # Transitability
    .set_edge_penalty_scale(0.3)
)

cfg = (cfg
    .add_shape_metrics(["polsby_popper"])
)

# Build election dictionaries from the geospatial columns.
election_dicts = geo.build_election_dicts(
    params["data_tag"],
    years=[2016, 2020, 2024],
    offices=["PRE", "GOV", "ATG", "AUD", "TRE"],
    parties=["D", "R", "-"],
    overrides={"2024GOV": {"R1": "G24GOVRHEN", "R2": "G24GOVNCLA"}},
)

cfg = (cfg
    .add_election_updaters(
        elections=election_dicts,
        skip_if_missing_parties=True
    )
    .add_election_aggregator(
        name="2020",
        elections=["2020PRE"],
        parties=["D", "R", "-"],
    )
    .add_election_metric_updaters(
        "2020", 
        [
            "efficiency_gap", "stdev_partisan_share",
            "majority_partisan_shares", "majority_seats",
        ],
        prepend_agg_name=True
    )
    .add_election_aggregator(
        name="2024",
        elections=["2024PRE"],
        parties=["D", "R", "-"],
    )
    .add_election_metric_updaters(
        "2024",
        [
            "efficiency_gap", "stdev_partisan_share",
            "majority_partisan_shares", "majority_seats",
        ],
        prepend_agg_name=True
    )
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
        prepend_agg_name=True,
    )
)

save_dir = f"output/{config_tag}/{run_name}"
os.makedirs(os.path.join(save_dir, "maps"), exist_ok=True)

print(f"Output will be saved to {save_dir}")
cfg.to_config(os.path.join(save_dir, "config.yaml"))

comparison_maps_path = os.path.join(save_dir, "comparison_maps.json")
if start_step > 0 and os.path.exists(comparison_maps_path):
    print("Loading pre-computed comparison maps performance")
    with open(comparison_maps_path, "r") as f:
        comparison_maps = json.load(f)
else:
    print("Computing performance of comparison maps")
    def compute_metrics_for_map(shapefile_path, geo, cfg, pop_key=params["data_tag"]):
        """Return a dict of output_updater values for a given electoral map."""
        partition = geo.build_partition(
            pop_key=pop_key,
            plan=shapefile_path,
            updaters=cfg.updaters,
        )
        return nbh.get_updater_values(partition, cfg.updaters_to_save)

    comparison_maps = {}
    for k, v in params["comparison_maps"].items():
        comparison_maps[k] = compute_metrics_for_map(v, geo, cfg)

    with open(comparison_maps_path, "w") as f:
        json.dump(comparison_maps, f, indent=2)

munis, counties = nbh.load_boundaries_from_shapefiles()

from gerrychain.accept import always_accept
import utgc.plotting as gcplt

initial_partition = geo.build_partition(
    pop_key=params["data_tag"],
    plan=initial_plan,
    updaters=cfg.updaters,
)
total_pop = sum(initial_partition["population"].values())
pop_tolerance = params["pop_tolerance"]

remaining_steps = max(0, num_steps - start_step)
proposal = cfg.proposal(
    initial_partition,
    total_population=total_pop,
    pop_tolerance=pop_tolerance,
    edge_weights=geo.get_edge_weights(params["data_tag"]),
)
chain = CouponCollectorChain(
    proposal=proposal,
    constraints=cfg.constraints,
    accept=always_accept,
    initial_state=initial_partition,
    micro_steps_per_yield=ceil(coupon_collector_expectation(num_districts)),
    num_macro_steps=remaining_steps,
)

print(f"=== MCMC {run_name} ===")
print("Running Markov chain (Batched/Coupon Collector)...")
partition_iterator = chain.with_progress_bar()

output_path = os.path.join(save_dir, "output.jsonl")

# import utgc.results as gcres
# import utgc.plotting as gcplt
# import matplotlib.pyplot as plt

# def update_metrics_plot(output_path, comparison_maps, handles, in_jupyter):
#     import os
#     if not os.path.exists(output_path):
#         return handles
    
#     datasets = ["sb1011_data", "2020", "2024"]
#     metrics = ["majority_partisan_shares", "stdev_partisan_share", "majority_seats"]
    
#     columns_to_read = [f"{ds}_{m}" for ds in datasets for m in metrics]
#     try:
#         import pandas as pd
#         df = gcres.read_jsonl_table(output_path, columns_to_read)
#         df = df.apply(pd.to_numeric, errors='coerce')
#     except Exception as e:
#         print(f"Error reading jsonl: {e}")
#         return handles
        
#     if df.empty:
#         return handles
        
#     figs = []
#     axes_list = []
    
#     if in_jupyter:
#         for _ in range(3):
#             fig, axes = plt.subplots(3, 1, figsize=(6, 12), dpi=100)
#             figs.append(fig)
#             axes_list.append(axes)
#     else:
#         if handles is not None and isinstance(handles, list) and all(isinstance(h, plt.Figure) for h in handles):
#             figs = handles
#             for fig in figs:
#                 fig.clf()
#                 axes_list.append(fig.subplots(4, 1))
#         else:
#             for _ in range(3):
#                 fig = plt.figure()
#                 axes_list.append(fig.subplots(4, 1))
#                 figs.append(fig)
    
#     for i, ds in enumerate(datasets):
#         # majority_partisan_shares
#         col_mps = f"{ds}_majority_partisan_shares"
#         if f"{col_mps}_0" in df.columns:
#             df_mps = gcres.sort_subentries(df, col_mps)
#             mps_cols = [c for c in df_mps.columns if c.startswith(f"{col_mps}_")]
            
#             ref_vals_mps = {}
#             for k, v in comparison_maps.items():
#                 if col_mps in v and isinstance(v[col_mps], dict):
#                     ref_vals_mps[k] = sorted(v[col_mps].values())
            
#             gcplt.district_plot(
#                 df_mps[mps_cols],
#                 reference_values=ref_vals_mps,
#                 ax=axes_list[0][i]
#             )
#             axes_list[0][i].set_ylabel(ds, fontsize=12, fontweight='bold')
#             if i == 0:
#                 axes_list[0][i].set_title("Ranked Partisan Share")
                
#         # stdev_partisan_share
#         col_stdev = f"{ds}_stdev_partisan_share"
#         if col_stdev in df.columns:
#             ref_vals_stdev = {}
#             for k, v in comparison_maps.items():
#                 if col_stdev in v and not isinstance(v[col_stdev], dict):
#                     ref_vals_stdev[k] = v[col_stdev]
                    
#             gcplt.distribution_plot(
#                 df[col_stdev].dropna(),
#                 highlight_interval=[0.025, 0.975],
#                 reference_values=ref_vals_stdev,
#                 ax=axes_list[1][i]
#             )
#             axes_list[1][i].set_ylabel(ds, fontsize=12, fontweight='bold')
#             if i == 0:
#                 axes_list[1][i].set_title("Stdev Partisan Share")
                
#         # majority_seats
#         col_seats = f"{ds}_majority_seats"
#         if col_seats in df.columns:
#             ref_vals_seats = {}
#             for k, v in comparison_maps.items():
#                 if col_seats in v and not isinstance(v[col_seats], dict):
#                     ref_vals_seats[k] = v[col_seats]
                    
#             gcplt.distribution_plot(
#                 df[col_seats].dropna(),
#                 highlight_interval=[0.025, 0.975],
#                 reference_values=ref_vals_seats,
#                 ax=axes_list[2][i]
#             )
#             axes_list[2][i].set_ylabel(ds, fontsize=12, fontweight='bold')
#             if i == 0:
#                 axes_list[2][i].set_title("Majority Seats")
                
#     for f in figs:
#         f.tight_layout()
        
#     if in_jupyter:
#         if handles is not None and len(handles) == 3 and hasattr(handles[0], 'update'):
#             for j in range(3):
#                 handles[j].update(figs[j])
#         else:
#             try:
#                 from IPython.display import clear_output, display
#                 clear_output(wait=True)
#                 for f in figs:
#                     display(f)
#             except Exception:
#                 pass
#         for f in figs:
#             plt.close(f)
#         return handles
#     else:
#         plt.show(block=False)
#         plt.pause(0.1)
#         return figs

# in_jupyter = False
# try:
#     from IPython import get_ipython
#     if get_ipython() is not None and get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
#         in_jupyter = True
# except Exception:
#     pass

# plot_handles = None
# if in_jupyter:
#     try:
#         from IPython.display import display
#         plot_handles = [
#             display("Initializing Partisan Share plots...", display_id=True),
#             display("Initializing Stdev plots...", display_id=True),
#             display("Initializing Seats plots...", display_id=True)
#         ]
#     except Exception:
#         pass
# else:
#     plt.ion()

# plot_handles = update_metrics_plot(output_path, comparison_maps, plot_handles, in_jupyter)

file_mode = "a" if start_step > 0 else "w"
with open(output_path, file_mode) as f:
    for idx, partition in enumerate(partition_iterator):
        step_number = idx + start_step
        # Save metrics (output_updaters subset only)
        data = (
            {"step": step_number}
            | nbh.get_updater_values(partition, cfg.updaters_to_save)
        )
        f.write(json.dumps(data) + "\n")
        f.flush()

        nbh.save_partition(
            partition,
            os.path.join(save_dir,"maps","shapefiles",f"step_{step_number:05d}.zip"),
            geo.get_pop_geodata(params["data_tag"])
        )

        # Plot ordinary map
        gcplt.visualize_partition(
            partition, step_number, os.path.join(save_dir, "maps"),
            counties=counties, municipalities=munis,
            split_munis_count=partition["split_muni"],
            split_counties_count=partition["split_county"],
        )
        # Plot map with partisanship
        gcplt.visualize_partition(
            partition, step_number, os.path.join(save_dir, "maps", "partisanship"),
            counties=counties, municipalities=munis,
            split_munis_count=partition["split_muni"],
            split_counties_count=partition["split_county"],
            color_by="2024_majority_partisan_shares",
            colormap="coolwarm"
        )
        
        # if step_number > start_step and (step_number - start_step) % 5 == 0:
        #     plot_handles = update_metrics_plot(output_path, comparison_maps, plot_handles, in_jupyter)
            
        partition.parent = None

print("Done!")

import utgc.results as gcres
import utgc.plotting as gcplt
import matplotlib.pyplot as plt

if "comparison_maps" not in locals():
    with open(f"output/{config_tag}/ensemble/comparison_maps.json", "r") as f:
        comparison_maps = json.load(f)

output_path = f"output/{config_tag}/ensemble/output.jsonl"

# Partisan vote shares
party_shares = gcres.read_jsonl_table(output_path, "majority_partisan_shares")
party_shares = gcres.sort_subentries(party_shares, "majority_partisan_shares")

plt.figure(dpi=300, figsize=(6,4))
gcplt.district_plot(
    party_shares,
    reference_values={
        k: sorted(v["majority_partisan_shares"].values()) for k, v in comparison_maps.items()
    },
    relative_to_median=False
)
plt.xlabel("Ranked Partisan Share")
plt.savefig(os.path.join(save_dir, "ensemble_partisan_shares.png"), dpi=300, bbox_inches='tight', facecolor='white')

plt.figure(dpi=300, figsize=(6,3))
gcplt.distribution_plot(
    party_shares["majority_partisan_shares_0"],
    highlight_interval=[0.025, 0.975],
    reference_values={
        mapname: min(stats["majority_partisan_shares"].values())
        for mapname, stats in comparison_maps.items()
    },
    relative_to_median=False,
)
plt.xlabel("Least-Republican District")
plt.savefig(os.path.join(save_dir, "ensemble_least_rep_district.png"), dpi=300, bbox_inches='tight', facecolor='white')

sdvs = gcres.read_jsonl_table(output_path, "stdev_partisan_share")
plt.figure(dpi=300, figsize=(6,3))
gcplt.distribution_plot(
    sdvs["stdev_partisan_share"],
    highlight_interval=[0.025, 0.975],
    reference_values={
        mapname: stats["stdev_partisan_share"]
        for mapname, stats in comparison_maps.items()
    },
    relative_to_median=False,
)
plt.xlabel("Standard Deviation of Partisan Vote Share")
plt.savefig(os.path.join(save_dir, "ensemble_sdv_partisan_share.png"), dpi=300, bbox_inches='tight', facecolor='white')

eg = gcres.read_jsonl_table(output_path, "efficiency_gap")
plt.figure(dpi=300, figsize=(6,3))
gcplt.distribution_plot(
    eg["efficiency_gap"],
    highlight_interval=[0.025, 0.975],
    reference_values={
        mapname: stats["efficiency_gap"]
        for mapname, stats in comparison_maps.items()
    },
    relative_to_median=False,
)
plt.savefig(os.path.join(save_dir, "ensemble_efficiency_gap.png"), dpi=300, bbox_inches='tight', facecolor='white')

ranked_means = party_shares.mean(axis=0)

def ranked_marginal_deviation(party_shares, ranked_means):
    rmd = ((party_shares - ranked_means) ** 2)
    if isinstance(party_shares, list):
        rmd = rmd.sum()
    else:
        rmd = rmd.sum(axis=1)
    return (rmd / len(ranked_means)) ** 0.5

plt.figure(dpi=300, figsize=(6,3))
gcplt.distribution_plot(
    ranked_marginal_deviation(party_shares, ranked_means),
    reference_values={
        mapname: ranked_marginal_deviation(
            sorted(stats["majority_partisan_shares"].values()),
            ranked_means
        ) for mapname, stats in comparison_maps.items()
    },
    highlight_interval=[0, 0.95],
    relative_to_median=False,
)
plt.xlabel("Ranked Marginal Deviation")
# plt.xlim(left=0)
plt.savefig(os.path.join(save_dir, "ensemble_ranked_marginal_deviation.png"), dpi=300, bbox_inches='tight', facecolor='white')

hashes = gcres.read_jsonl_table(output_path, "assignment_hash")
print(f"Unique maps: {len(hashes.drop_duplicates()) / len(hashes):.1%}")