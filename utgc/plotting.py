import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from gerrychain import GeographicPartition
from typing import List, Optional

def plot_partisan_summary(summary_df: pd.DataFrame, elections: List[str], output_dir: str):
    """
    Creates a two-panel summary plot showing Democratic vote share distributions
    and the distribution of Republican-won seats for a given election.

    :param summary_df: The DataFrame from a ResultSet.
    :param elections: A list of election names to plot (e.g., ['20_PRE']).
                      Currently, it plots the first election in the list.
    :param output_dir: The directory to save the plot image.
    """
    if not elections:
        print("Plotting skipped: No elections were tracked in this run.")
        return

    # For this initial version, we'll plot the first tracked election.
    # This can be expanded to loop or select specific elections.
    election_to_plot = elections[0]
    
    # Define column names based on the selected election
    dem_share_cols = [col for col in summary_df.columns if col.startswith(f"{election_to_plot}_d_share_")]
    rep_seats_col = f"{election_to_plot}_R_seats"

    if not dem_share_cols or rep_seats_col not in summary_df.columns:
        print(f"Plotting skipped: Missing data for election '{election_to_plot}'.")
        return

    num_districts = len(dem_share_cols)

    # Create the two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Ensemble Analysis Summary for Election: {election_to_plot}", fontsize=16, fontweight='bold')

    # --- Panel 1: Boxplot of Democratic Vote Shares ---
    sns.boxplot(data=summary_df[dem_share_cols], ax=ax1, orient='h', whis=[1, 99])
    ax1.set_title("Distribution of Democratic Vote Share by District", fontsize=12)
    ax1.set_xlabel("Democratic Vote Share")
    ax1.set_ylabel("District")
    ax1.axvline(0.5, color='r', linestyle='--', alpha=0.7, label='50% Threshold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # --- Panel 2: Histogram of Republican Seats ---
    seats_data = summary_df[rep_seats_col]
    sns.histplot(data=seats_data, ax=ax2, discrete=True, stat="proportion")
    ax2.set_title("Distribution of Republican Seats Won", fontsize=12)
    ax2.set_xlabel("Number of Republican Seats")
    ax2.set_ylabel("Proportion of Plans")
    mean_val = seats_data.mean()
    median_val = seats_data.median()
    ax2.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
    ax2.axvline(median_val, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_val:.0f}')
    ax2.legend()
    ax2.set_xlim(-0.5, num_districts + 0.5)
    ax2.set_xticks(range(num_districts + 1))

    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    save_path = os.path.join(output_dir, "partisan_summary_plot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"✓ Partisan summary plot saved to {os.path.basename(save_path)}")

def _find_wasatch_front_bounds(counties_gdf: gpd.GeoDataFrame) -> Optional[tuple]:
    """
    Return padded, custom bounds for a Wasatch Front zoom.
    Returns None if bounds cannot be determined.
    """
    if counties_gdf is None or len(counties_gdf) == 0:
        return None

    target_names = {"WEBER", "DAVIS", "SALT LAKE", "UTAH"}
    # Find the column with county names
    name_col = next((col for col in ["NAME", "County", "COUNTY"] if col in counties_gdf.columns), None)
    if name_col is None: return None

    wf_counties = counties_gdf[counties_gdf[name_col].str.upper().isin(target_names)]
    if wf_counties.empty: return None

    minx, miny, maxx, maxy = wf_counties.total_bounds
    x_pad = (maxx - minx) * 0.1
    y_pad = (maxy - miny) * 0.1
    return (minx - x_pad, maxx + x_pad, miny - y_pad, maxy + y_pad)


def plot_partition_on_map(
    partition: GeographicPartition,
    output_path: str,
    title: str = "Districting Plan",
    counties_gdf: Optional[gpd.GeoDataFrame] = None,
    zoom_to_wasatch_front: bool = False
):
    """
    Plots a map of the districting plan and saves it to a file.
    Optionally overlays county boundaries and can zoom to the Wasatch Front.

    :param partition: The GerryChain GeographicPartition to plot.
    :param output_path: The full path where the image file will be saved.
    :param title: The title for the map plot.
    :param counties_gdf: Optional GeoDataFrame of county boundaries to overlay.
    :param zoom_to_wasatch_front: If True, zooms the map to the Wasatch Front.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the districts from the partition's geodataframe
    partition.plot(ax=ax, cmap="tab20", edgecolor="#444", linewidth=0.5)

    # Plot county boundaries if provided
    if counties_gdf is not None:
        counties_gdf.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=1.2, alpha=0.8)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_axis_off()

    if zoom_to_wasatch_front:
        bounds = _find_wasatch_front_bounds(counties_gdf)
        if bounds:
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[2], bounds[3])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

