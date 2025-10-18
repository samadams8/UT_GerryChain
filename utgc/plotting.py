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

def _find_wasatch_front_bounds(counties_gdf):
        """Return padded, custom bounds for a Wasatch Front zoom.

        Rules requested:
        - Show the combined extent of Weber, Davis, Salt Lake, and Utah counties
          (both east-west and north-south bounds from the union of these)

        Returns None if bounds cannot be determined.
        """
        if counties_gdf is None or len(counties_gdf) == 0:
            return None

        target_names = {"WEBER", "DAVIS", "SALT LAKE", "UTAH"}
        candidate_cols = [
            "NAME",
            "County",
            "COUNTY",
            "COUNTYNAME",
            "COUNTY_NA",
            "COUNTYNAM",
            "CNTY_NAME",
        ]
        name_col = None
        for col in candidate_cols:
            if col in counties_gdf.columns:
                name_col = col
                break

        if name_col is None:
            # Try any object dtype column by heuristic
            for col in counties_gdf.columns:
                try:
                    if counties_gdf[col].dtype == object:
                        upper_vals = counties_gdf[col].astype(str).str.upper()
                        if upper_vals.isin(target_names).any():
                            name_col = col
                            break
                except Exception:
                    continue

        if name_col is None:
            return None

        try:
            upper_names = counties_gdf[name_col].astype(str).str.upper()
            # Extract individual counties
            def _geom_of(county_upper_name):
                m = upper_names == county_upper_name
                g = counties_gdf[m]
                if len(g) == 0:
                    return None
                try:
                    # GeoPandas >= 0.14 preferred API
                    return g.geometry.union_all()
                except Exception:
                    # Fallback for older GeoPandas
                    return g.unary_union

            geom_weber = _geom_of("WEBER")
            geom_davis = _geom_of("DAVIS")
            geom_salt_lake = _geom_of("SALT LAKE")
            geom_utah = _geom_of("UTAH")
            # Require at least Weber, Davis, Salt Lake to proceed
            required_geoms = [geom_weber, geom_davis, geom_salt_lake]
            if any(g is None for g in required_geoms):
                return None

            # Core union (Weber, Davis, Salt Lake, Utah if available)
            from shapely.ops import unary_union
            core_geoms = [g for g in [geom_weber, geom_davis, geom_salt_lake, geom_utah] if g is not None]
            core_union = unary_union(core_geoms)
            c_minx, c_miny, c_maxx, c_maxy = core_union.bounds

            # East-West and North-South bounds: use core union of the four counties
            minx = c_minx
            maxx = c_maxx
            miny = c_miny
            maxy = c_maxy

            # Ensure minx < maxx; if inverted due to data quirks, fallback to core bbox
            if not (minx < maxx):
                minx, maxx = c_minx, c_maxx

            # Apply modest padding
            pad_x = (maxx - minx) * 0.05
            pad_y = (maxy - miny) * 0.05
            return (minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y)
        except Exception:
            return None

def visualize_partition(
    partition: GeographicPartition,
    step: int,
    output_dir: str,
    municipalities: Optional[gpd.GeoDataFrame] = None,
    counties: Optional[gpd.GeoDataFrame] = None,
    split_munis_count: Optional[int] = None,
    split_counties_count: Optional[int] = None,
):
    # Prepare figure with two panels: full map (left) and Wasatch Front zoom (right)
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 8))

    # Left: full map
    partition.plot(ax=ax_full, cmap='tab20c', edgecolor='none')
    if municipalities is not None:
        municipalities.boundary.plot(
            ax=ax_full, color='black', linewidth=0.25, alpha=0.5
        )
    if counties is not None:
        counties.boundary.plot(
            ax=ax_full, color='black', linewidth=1, alpha=0.5
        )
    title = f"Step {step}"
    if split_munis_count is not None:
        title += f", Muni Splits={split_munis_count}"
    if split_counties_count is not None:
        title += f", County Splits={split_counties_count}"
    ax_full.set_title(title, fontsize=12, fontweight='bold')
    ax_full.set_xticks([])
    ax_full.set_yticks([])
    ax_full.set_aspect('equal')

    # Right: Wasatch Front zoom (fallback to full extent if bounds unresolved)
    partition.plot(ax=ax_zoom, cmap='tab20c', edgecolor='none')
    if municipalities is not None:
        municipalities.boundary.plot(
            ax=ax_zoom, color='black', linewidth=0.25, alpha=0.5
        )
    if counties is not None:
        counties.boundary.plot(
            ax=ax_zoom, color='black', linewidth=1, alpha=0.5
        )

    wf_bounds = _find_wasatch_front_bounds(counties)
    if wf_bounds is not None:
        minx, maxx, miny, maxy = wf_bounds
        ax_zoom.set_xlim(minx, maxx)
        ax_zoom.set_ylim(miny, maxy)
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    ax_zoom.set_aspect('equal')

    plt.tight_layout()
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"step_{step:05d}.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()