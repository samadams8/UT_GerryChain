import os
import warnings
from math import ceil
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
import numpy as np
from gerrychain import GeographicPartition
from typing import List, Optional, Dict, Any, Tuple, Union

from seaborn.utils import despine

def distribution_plot(
    results,
    reference_values={},
    highlight_interval=[],
    relative_to_median=False,
    use_kde=False,
    ):
    if not results.ndim == 1:
        raise ValueError("distribution_plot only supports 1D results")

    # Calculate medians per district
    medians = results.median()
    if relative_to_median:
        plot_df = results - medians
        plt_medians = medians - medians
    else:
        plot_df = results
        plt_medians = medians

    x_median = plt_medians

    if use_kde:
        sns.kdeplot(data=plot_df, fill=True, color='black')
        sns.rugplot(data=plot_df, color='black')
    else:
        sns.histplot(data=plot_df, color='gray')

    # Median
    plt.axvline(x_median, linestyle='--', color='white' if highlight_interval else 'black')
    plt.text(x_median, plt.gca().get_ylim()[1], ' 50%', ha='center', va='bottom', fontsize=10, rotation=90)

    if highlight_interval:
        percentiles = plot_df.quantile(highlight_interval)

        x_low = percentiles.loc[highlight_interval[0]]
        x_high = percentiles.loc[highlight_interval[1]]
        plt.axvspan(x_low, x_high, color='gray', alpha=0.3)
        # Lower bound
        if highlight_interval[0] > 0:
            plt.text(x_low, plt.gca().get_ylim()[1], f' {highlight_interval[0]:.1%}', ha='center', va='bottom', fontsize=10, rotation=90)

        # Upper bound
        if highlight_interval[1] < 1:
            plt.text(x_high, plt.gca().get_ylim()[1], f' {highlight_interval[1]:.1%}', ha='center', va='bottom', fontsize=10, rotation=90)

    y_max = max(plt.gca().get_ylim())

    if reference_values:
        markers = ['o', '^', 's', 'v', 'D'] * ceil(len(reference_values) / 6)
        colors = plt.cm.tab10.colors

        for i, (label, ref_val) in enumerate(reference_values.items()):
            if relative_to_median:
                ref_val = ref_val - medians
            plt.plot(ref_val, y_max/20, markers[i], mec=colors[i], mfc="None", label=label)

        plt.legend()

def district_plot(
        results,
        reference_values={},
        highlight_interval=[],
        relative_to_median=False
    ):

    # Calculate medians per district
    medians = results.median()
    if relative_to_median:
        plot_df = results.sub(medians, axis=1)
        plt_medians = medians.sub(medians, axis=0)
    else:
        plot_df = results
        plt_medians = medians

    sns.violinplot(data=plot_df, orient='v', inner=None, color='gray')
    plt.xticks(range(len(plot_df.columns)), range(1,len(plot_df.columns)+1))

    if highlight_interval:
        percentiles = plot_df.quantile(highlight_interval)

        for i, col in enumerate(plot_df.columns):
            y_low = percentiles.loc[highlight_interval[0], col]
            y_high = percentiles.loc[highlight_interval[1], col]
            plt.plot([i, i], [y_low, y_high], color='black', lw=2)
            y_median = plt_medians[col]
            plt.plot(i, y_median, marker='o', mfc='white', mec='k', markersize=5)
    
    if reference_values:
        markers = ['o', '^', 's', 'v', 'D'] * ceil(len(reference_values) / 6)
        colors = plt.cm.tab10.colors
        for i, (label, ref_values) in enumerate(reference_values.items()):
            # No sorting; reference_values must conform to the district column order externally
            if isinstance(ref_values, dict):
                ref_values = [ref_values.get(col.split('_', 1)[-1], None) for col in results.columns]
            for rank_idx, value in enumerate(ref_values):
                if value is None:
                    continue  # skip if data not provided for this district
                if relative_to_median:
                    value = value - medians.iloc[rank_idx]
                plt.plot(rank_idx, value, markers[i], mec=colors[i], mfc="None", label=label if rank_idx == 0 else "")

        plt.legend()

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

def _build_value_gdf(partition, updater_name: str) -> gpd.GeoDataFrame:
    """
    Build a GeoDataFrame with a ``color_val`` column from a district-level updater.

    Each row corresponds to a graph node (VTD/precinct). The ``color_val`` column
    holds the scalar updater value for that node's assigned district.

    Parameters
    ----------
    partition : GeographicPartition
        The active partition. Must have ``.geometries`` (GeoDataFrame indexed by
        node id), ``.assignment`` (dict node_id -> district_id), and support
        ``partition[updater_name]`` returning a ``dict[district_id -> float]``.
    updater_name : str
        Key of the district-level updater to extract values from.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of ``partition.geometries`` with an added ``color_val`` column.
    """
    district_values: dict = partition[updater_name]
    assignment: dict = dict(partition.assignment)
    gdf = gpd.GeoDataFrame(geometry=partition.graph.geometry)
    
    # Convert keys to strings to avoid type mismatch (e.g., int vs str district IDs)
    str_district_values = {str(k): v for k, v in district_values.items()}
    gdf['_district'] = gdf.index.map(assignment).astype(str)
    gdf['color_val'] = gdf['_district'].map(str_district_values)
    
    return gdf


def visualize_partition(
    partition: GeographicPartition,
    step: int,
    output_dir: str,
    municipalities: Optional[gpd.GeoDataFrame] = None,
    counties: Optional[gpd.GeoDataFrame] = None,
    split_munis_count: Optional[int] = None,
    split_counties_count: Optional[int] = None,
    bounds_dir: str = "data/bounds",
    auto_load_boundaries: bool = True,
    county_path: Optional[str] = None,
    muni_path: Optional[str] = None,
    dpi: Optional[int] = 300,
    colormap: Optional[str] = 'Spectral',
    color_by: Optional[str] = None,
    color_by_norm: Optional[Tuple[float, float]] = None,
):
    """
    Visualize a partition with optional municipality and county boundaries.
    
    Parameters
    ----------
    partition : GeographicPartition
        The partition to visualize
    step : int
        Step number for the title
    output_dir : str
        Directory to save the output image
    municipalities : Optional[gpd.GeoDataFrame], optional
        Municipality boundaries. If None and auto_load_boundaries=True, will load from bounds_dir or muni_path.
    counties : Optional[gpd.GeoDataFrame], optional
        County boundaries. If None and auto_load_boundaries=True, will load from bounds_dir or county_path.
    split_munis_count : Optional[int], optional
        Number of municipality splits to display in title
    split_counties_count : Optional[int], optional
        Number of county splits to display in title
    bounds_dir : str, optional
        Directory containing boundary shapefiles, by default "data/bounds".
        Only used if county_path/muni_path are not provided.
    auto_load_boundaries : bool, optional
        If True and boundaries are None, automatically load from bounds_dir or paths, by default True
    county_path : str, optional
        Absolute path to county shapefile. If provided, takes precedence over bounds_dir.
    muni_path : str, optional
        Absolute path to municipality shapefile. If provided, takes precedence over bounds_dir.
    colormap : str, optional
        Colormap to use for the partition, by default ``'Spectral'``.
    color_by : str, optional
        Name of a district-level updater whose value is a ``dict[district_id ->
        float]``. When provided, districts are colored by the scalar value rather
        than by district ID. A diverging colormap (e.g. ``'coolwarm'``) is
        recommended. Defaults to ``None`` (categorical coloring).
    color_by_norm : tuple of (float, float), optional
        ``(vmin, vmax)`` normalization range for ``color_by`` coloring. Defaults
        to ``(0.0, 1.0)`` when ``color_by`` is set.
    """
    # Auto-load boundaries if not provided
    if auto_load_boundaries:
        if counties is None or municipalities is None:
            # Import here to avoid circular imports
            from .notebookhelper import load_boundaries_from_shapefiles
            loaded_munis, loaded_counties = load_boundaries_from_shapefiles(
                bounds_dir=bounds_dir,
                county_path=county_path,
                muni_path=muni_path
            )
            if municipalities is None:
                municipalities = loaded_munis
            if counties is None:
                counties = loaded_counties
    
    # Resolve color_by GeoDataFrame and normalization up front
    value_gdf = None
    if color_by is not None:
        try:
            value_gdf = _build_value_gdf(partition, color_by)
        except (KeyError, Exception) as exc:
            warnings.warn(
                f"visualize_partition: color_by='{color_by}' failed ({exc}); "
                "falling back to categorical colormap."
            )
            value_gdf = None

    if value_gdf is not None:
        vmin, vmax = color_by_norm if color_by_norm is not None else (0.0, 1.0)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # Prepare figure with two panels: full map (left) and Wasatch Front zoom (right)
    # Leave room on the right for a colorbar when using choropleth
    if value_gdf is not None:
        fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(13, 8))
    else:
        fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 8))

    num_parts = len(partition)
    cmap = plt.get_cmap(colormap, num_parts)

    def _plot_boundaries(ax):
        """Overlay municipality and county boundaries onto ax."""
        if municipalities is not None:
            municipalities.boundary.plot(
                ax=ax, color='black', linewidth=0.25, alpha=0.5
            )
        if counties is not None:
            counties.boundary.plot(
                ax=ax, color='black', linewidth=1, alpha=0.5
            )

    # Left: full map
    if value_gdf is not None:
        value_gdf.plot(ax=ax_full, column='color_val', cmap=colormap,
                       norm=norm, edgecolor='none', legend=False)
    else:
        partition.plot(ax=ax_full, cmap=cmap, edgecolor='none')
    _plot_boundaries(ax_full)

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
    if value_gdf is not None:
        value_gdf.plot(ax=ax_zoom, column='color_val', cmap=colormap,
                       norm=norm, edgecolor='none', legend=False)
    else:
        partition.plot(ax=ax_zoom, cmap=cmap, edgecolor='none')
    _plot_boundaries(ax_zoom)

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
    plt.savefig(os.path.join(output_dir, f"step_{step:05d}.png"), dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_time_series(df: pd.DataFrame,
metric_names: Union[str, List[str]],
output_path: str, 
sort_districts: bool = False,
line_styles: Optional[List[str]] = None):
    """
    Plot time series of one or more metrics over steps.
    
    :param df: DataFrame containing the ensemble data
    :param metric_names: Name(s) of the metric(s) to plot
    :param output_path: Path to save the plot
    :param sort_districts: If True, sort district values at each step
    :param line_styles: Optional list of line styles for multiple metrics
    """
    # Handle single metric as string
    if isinstance(metric_names, str):
        metric_names = [metric_names]
    
    # Validate all metrics exist
    for metric_name in metric_names:
        if metric_name not in df.columns:
            raise ValueError(f"Metric '{metric_name}' not found in data")
    
    plt.figure(figsize=(12, 8))
    
    # Default line styles
    if line_styles is None:
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    
    # Ensure we have enough line styles
    while len(line_styles) < len(metric_names):
        line_styles.extend(['-', '--', '-.', ':'])
    
    colors = plt.colormaps['tab10'](np.linspace(0, 1, len(metric_names)))
    
    for i, metric_name in enumerate(metric_names):
        # Check if metric is district-level (dict) or map-level (scalar)
        sample_value = df[metric_name].dropna().iloc[0] if not df[metric_name].dropna().empty else None
        
        if isinstance(sample_value, dict):
            # District-level metric
            if sort_districts:
                # Sort districts at each step
                sorted_data = []
                for _, row in df.iterrows():
                    if pd.notna(row[metric_name]) and isinstance(row[metric_name], dict):
                        sorted_values = sorted(row[metric_name].values())
                        sorted_data.append(sorted_values)
                    else:
                        sorted_data.append([])
                
                # Create DataFrame for sorted data
                max_districts = max(len(step_data) for step_data in sorted_data) if sorted_data else 0
                sorted_df = pd.DataFrame(sorted_data, columns=[f'Rank_{i+1}' for i in range(max_districts)])
                
                # Plot each rank
                for j, col in enumerate(sorted_df.columns):
                    if not sorted_df[col].dropna().empty:
                        plt.plot(df['step'], sorted_df[col], 
                               linestyle=line_styles[i % len(line_styles)],
                               color=colors[i], 
                               label=f'{metric_name} - {col}', 
                               alpha=0.7)
            else:
                # Plot each district separately
                all_districts = set()
                for _, row in df.iterrows():
                    if pd.notna(row[metric_name]) and isinstance(row[metric_name], dict):
                        all_districts.update(row[metric_name].keys())
                
                # Create district-specific colors and styles
                district_colors = plt.colormaps['tab10'](np.linspace(0, 1, len(all_districts)))
                district_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
                
                for j, district in enumerate(sorted(all_districts)):
                    values = []
                    for _, row in df.iterrows():
                        if pd.notna(row[metric_name]) and isinstance(row[metric_name], dict):
                            values.append(row[metric_name].get(district, np.nan))
                        else:
                            values.append(np.nan)
                    
                    if not all(pd.isna(values)):
                        plt.plot(df['step'], values, 
                               linestyle=district_styles[j % len(district_styles)],
                               color=district_colors[j],
                               label=f'{metric_name} - District {district}', 
                               alpha=0.7)
        else:
            # Map-level metric
            plt.plot(df['step'], df[metric_name], 
                   linestyle=line_styles[i % len(line_styles)],
                   color=colors[i],
                   linewidth=2, 
                   label=metric_name)
    
    plt.xlabel('Step')
    if len(metric_names) == 1:
        plt.ylabel(metric_names[0])
        plt.title(f'{metric_names[0]} Over Time')
    else:
        plt.ylabel('Value')
        plt.title(f'Multiple Metrics Over Time')
    
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Time series plot saved to {os.path.basename(output_path)}")

def draw_time_series(ax, df: pd.DataFrame, metric_names: Union[str, List[str]], 
                    sort_districts: bool = False, line_styles: Optional[List[str]] = None):
    """
    Draw time series plot on the given axes.
    
    :param ax: Matplotlib axes to draw on
    :param df: DataFrame containing the ensemble data
    :param metric_names: Name(s) of the metric(s) to plot
    :param sort_districts: If True, sort district values at each step
    :param line_styles: Optional list of line styles for multiple metrics
    """
    # Handle single metric as string
    if isinstance(metric_names, str):
        metric_names = [metric_names]
    
    # Validate all metrics exist
    for metric_name in metric_names:
        if metric_name not in df.columns:
            raise ValueError(f"Metric '{metric_name}' not found in data")
    
    # Default line styles
    if line_styles is None:
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    
    # Ensure we have enough line styles
    while len(line_styles) < len(metric_names):
        line_styles.extend(['-', '--', '-.', ':'])
    
    colors = plt.colormaps['tab10'](np.linspace(0, 1, len(metric_names)))
    
    for i, metric_name in enumerate(metric_names):
        # Check if metric is district-level (dict) or map-level (scalar)
        sample_value = df[metric_name].dropna().iloc[0] if not df[metric_name].dropna().empty else None
        
        if isinstance(sample_value, dict):
            # District-level metric
            if sort_districts:
                # Sort districts at each step
                sorted_data = []
                for _, row in df.iterrows():
                    if pd.notna(row[metric_name]) and isinstance(row[metric_name], dict):
                        sorted_values = sorted(row[metric_name].values())
                        sorted_data.append(sorted_values)
                    else:
                        sorted_data.append([])
                
                # Create DataFrame for sorted data
                max_districts = max(len(step_data) for step_data in sorted_data) if sorted_data else 0
                sorted_df = pd.DataFrame(sorted_data, columns=[f'Rank_{i+1}' for i in range(max_districts)])
                
                # Plot each rank
                for j, col in enumerate(sorted_df.columns):
                    if not sorted_df[col].dropna().empty:
                        ax.plot(df['step'], sorted_df[col], 
                               linestyle=line_styles[i % len(line_styles)],
                               color=colors[i], 
                               label=f'{metric_name} - {col}', 
                               alpha=0.7)
            else:
                # Plot each district separately
                all_districts = set()
                for _, row in df.iterrows():
                    if pd.notna(row[metric_name]) and isinstance(row[metric_name], dict):
                        all_districts.update(row[metric_name].keys())
                
                # Create district-specific colors and styles
                district_colors = plt.colormaps['tab10'](np.linspace(0, 1, len(all_districts)))
                district_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
                
                for j, district in enumerate(sorted(all_districts)):
                    values = []
                    for _, row in df.iterrows():
                        if pd.notna(row[metric_name]) and isinstance(row[metric_name], dict):
                            values.append(row[metric_name].get(district, np.nan))
                        else:
                            values.append(np.nan)
                    
                    if not all(pd.isna(values)):
                        ax.plot(df['step'], values, 
                               linestyle=district_styles[j % len(district_styles)],
                               color=district_colors[j],
                               label=f'{metric_name} - District {district}', 
                               alpha=0.7)
        else:
            # Map-level metric
            ax.plot(df['step'], df[metric_name], 
                   linestyle=line_styles[i % len(line_styles)],
                   color=colors[i],
                   linewidth=2, 
                   label=metric_name)
    
    ax.set_xlabel('Step')
    if len(metric_names) == 1:
        ax.set_ylabel(metric_names[0])
        ax.set_title(f'{metric_names[0]} Over Time')
    else:
        ax.set_ylabel('Value')
        ax.set_title('Multiple Metrics Over Time')
        
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

def plot_distribution_histogram(values: pd.Series, metric_name: str, output_path: str, 
                               reference_values: Optional[Union[List[Any], Dict[str, Any]]] = None):
    """
    Plot histogram of a map-level metric.
    
    :param values: Series of values to plot
    :param metric_name: Name of the metric
    :param output_path: Path to save the plot
    :param reference_values: Optional list or dict of reference values to overlay.
                            If dict, keys are map labels and values are scalar values.
                            If list, will be labeled as "Reference 1", "Reference 2", etc.
    """
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    plt.hist(values, bins=30, alpha=0.7, edgecolor='black', density=True)
    
    # Add reference lines if provided
    if reference_values:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Handle both list (backward compatibility) and dict (new format)
        if isinstance(reference_values, dict):
            items = list(reference_values.items())
        else:
            # Convert list to dict with auto-generated labels
            items = [(f"Reference {i+1}", val) for i, val in enumerate(reference_values)]
        
        for i, (label, ref_val) in enumerate(items):
            if ref_val is not None:
                try:
                    # Try to convert to float if it's a string
                    if isinstance(ref_val, str):
                        ref_val = float(ref_val)
                    color = colors[i % len(colors)]
                    plt.axvline(ref_val, color=color, linestyle='--', linewidth=2, 
                               label=f'{label}: {ref_val:.3f}')
                except (ValueError, TypeError):
                    # Skip if value can't be converted to float
                    continue
    
    plt.xlabel(metric_name)
    plt.ylabel('Density')
    plt.title(f'Distribution of {metric_name}')
    plt.grid(True, alpha=0.3)
    
    if reference_values:
        plt.legend()
    
    plt.tight_layout()
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Histogram plot saved to {os.path.basename(output_path)}")

def plot_distribution_violin(df: pd.DataFrame, metric_name: str, output_path: str,
                            reference_values: Optional[Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]] = None,
                            sort_districts: bool = False):
    """
    Plot violin plot of a district-level metric.
    
    :param df: DataFrame containing the ensemble data
    :param metric_name: Name of the metric to plot
    :param output_path: Path to save the plot
    :param reference_values: Optional list or dict of reference value dictionaries.
                            If dict, keys are map labels and values are district-level dicts.
                            If list, will be labeled as "Reference 1", "Reference 2", etc.
    :param sort_districts: If True, sort district values at each step
    """
    if metric_name not in df.columns:
        raise ValueError(f"Metric '{metric_name}' not found in data")
    
    # Extract district-level data
    district_data = []
    district_labels = []
    
    if sort_districts:
        # Sort districts at each step
        for _, row in df.iterrows():
            if pd.notna(row[metric_name]) and isinstance(row[metric_name], dict):
                sorted_values = sorted(row[metric_name].values())
                district_data.append(sorted_values)
        
        # Create DataFrame for sorted data
        max_districts = max(len(step_data) for step_data in district_data) if district_data else 0
        sorted_df = pd.DataFrame(district_data, columns=[f'Rank_{i+1}' for i in range(max_districts)])
        
        # Plot violin plot
        plt.figure(figsize=(12, 8))
        sns.violinplot(data=sorted_df, orient='v')
        plt.xlabel('District Rank')
        plt.ylabel(metric_name)
        plt.title(f'Distribution of {metric_name} (Sorted by Rank)')
        
    else:
        # Extract data for each district
        all_districts = set()
        for _, row in df.iterrows():
            if pd.notna(row[metric_name]) and isinstance(row[metric_name], dict):
                all_districts.update(row[metric_name].keys())
        
        district_values = {}
        for district in sorted(all_districts):
            values = []
            for _, row in df.iterrows():
                if pd.notna(row[metric_name]) and isinstance(row[metric_name], dict):
                    values.append(row[metric_name].get(district, np.nan))
                else:
                    values.append(np.nan)
            
            # Remove NaN values
            clean_values = [v for v in values if not pd.isna(v)]
            if clean_values:
                district_values[f'District {district}'] = clean_values
        
        if not district_values:
            raise ValueError(f"No valid data found for metric '{metric_name}'")
        
        # Create DataFrame for violin plot
        plot_data = []
        for district, values in district_values.items():
            for value in values:
                plot_data.append({'District': district, metric_name: value})
        
        plot_df = pd.DataFrame(plot_data)
        
        plt.figure(figsize=(12, 8))
        sns.violinplot(data=plot_df, x='District', y=metric_name)
        plt.xlabel('District')
        plt.ylabel(metric_name)
        plt.title(f'Distribution of {metric_name} by District')
        plt.xticks(rotation=45)
    
    # Add reference lines if provided
    if reference_values:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Handle both list (backward compatibility) and dict (new format)
        if isinstance(reference_values, dict):
            items = list(reference_values.items())
        else:
            # Convert list to dict with auto-generated labels
            items = [(f"Reference {i+1}", val) for i, val in enumerate(reference_values)]
        
        labels_shown = set()  # Track which labels we've already shown
        for i, (label, ref_dict) in enumerate(items):
            if ref_dict and isinstance(ref_dict, dict):
                color = colors[i % len(colors)]
                
                if sort_districts:
                    # For sorted plots, sort the reference values and show them at each rank
                    sorted_ref_values = sorted(ref_dict.values())
                    for rank_idx, value in enumerate(sorted_ref_values):
                        if not pd.isna(value):
                            try:
                                # Try to convert to float if it's a string
                                if isinstance(value, str):
                                    value = float(value)
                                # Show one horizontal line per reference map with label
                                if label not in labels_shown:
                                    plt.axhline(value, color=color, linestyle='--', alpha=0.7, 
                                               label=label)
                                    labels_shown.add(label)
                                else:
                                    plt.axhline(value, color=color, linestyle='--', alpha=0.7)
                            except (ValueError, TypeError):
                                continue
                else:
                    # For unsorted plots, show reference lines for each district
                    for district, value in ref_dict.items():
                        if not pd.isna(value):
                            try:
                                # Try to convert to float if it's a string
                                if isinstance(value, str):
                                    value = float(value)
                                
                                # Find the district in the plot
                                district_label = f'District {district}'
                                if district_label in plot_df['District'].values:
                                    # Show horizontal line for this district
                                    plt.axhline(value, color=color, linestyle='--', alpha=0.7)
                                    if label not in labels_shown:
                                        # Add label to legend only once per map
                                        plt.axhline(value, color=color, linestyle='--', 
                                                   alpha=0.7, label=label)
                                        labels_shown.add(label)
                            except (ValueError, TypeError):
                                # Skip if value can't be converted to float
                                continue
    
    plt.grid(True, alpha=0.3)
    if reference_values:
        plt.legend()
    
    plt.tight_layout()
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Violin plot saved to {os.path.basename(output_path)}")