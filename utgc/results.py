import os
import json
import pandas as pd
import yaml
from typing import Dict, Any, List, Optional, Union

# Import the new plotting functionality
from .plotting import plot_partisan_summary

class ResultSet:
    """
    A container for the results of an ensemble run.

    This object holds the summary DataFrame of all generated plans and provides
    methods for analysis, plotting, and saving.
    """
    def __init__(self, output_file: str):
        """
        Initializes the ResultSet.

        :param output_file: Path to the JSONL output file from an ensemble run.
        """
        self.output_file = output_file
        self.output_dir = os.path.dirname(output_file)
        
        # Parse JSONL into DataFrame 
        self._parse_jsonl()
        
        # Load config metadata
        self._load_config()
        
        print("✓ ResultSet created.")
        print(f"  - Contains results for {len(self.df)} plans.")
        print(f"  - Output directory: {self.output_dir}")

    def _parse_jsonl(self):
        """Parse JSONL file into a pandas DataFrame."""
        data = []
        with open(self.output_file, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        
        self.df = pd.DataFrame(data)
        
        # Store original data for reference
        self.output = data

    def _load_config(self):
        """Load configuration metadata from config.yaml file."""
        config_path = os.path.join(self.output_dir, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.metadata = yaml.safe_load(f)
        else:
            self.metadata = {}
            print("  Warning: No config.yaml found in output directory")

    def _classify_metric_type(self, metric_name: str) -> str:
        """
        Classify a metric as either 'map_level' (scalar) or 'district_level' (dict).
        
        :param metric_name: Name of the metric to classify
        :return: 'map_level' or 'district_level'
        """
        if metric_name not in self.df.columns:
            raise ValueError(f"Metric '{metric_name}' not found in data")
        
        # Check if the metric contains dict values (district-level)
        sample_value = self.df[metric_name].dropna().iloc[0] if not self.df[metric_name].dropna().empty else None
        
        if isinstance(sample_value, dict):
            return 'district_level'
        else:
            return 'map_level'

    def __repr__(self) -> str:
        """Provides a string representation of the ResultSet."""
        return f"<ResultSet with {len(self.df)} plans>"

    def save(self) -> None:
        """
        Saves the result summary to a CSV file in the run's output directory.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        path = os.path.join(self.output_dir, "ensemble_summary.csv")
        self.df.to_csv(path, index=False)
        print(f"  - Results summary saved to {path}")

    # --------------------------------------------------------------------------
    # Plotting and Analysis Methods
    # --------------------------------------------------------------------------

    def plot_partisan_summary(self):
        """
        Generates and saves a standard summary plot of partisan metrics.
        """
        # This method acts as a high-level, user-friendly API that calls
        # the more detailed, stateless function from the plotting library.
        plot_partisan_summary(
            summary_df=self.df,
            elections=self.metadata.get("tracked_elections", []),
            output_dir=self.output_dir
        )
        return self # Allow for chaining
        
    def plot_metric_over_steps(self,
        metrics: Union[str, List[str]], 
        save_path: Optional[str] = None, show: bool = False, 
        sort_districts: bool = False,
        line_styles: Optional[List[str]] = None,
        fig_name: Optional[str] = None,
        ax: Optional[Any] = None
    ):
        """
        Plot one or more metrics over time (steps).
        
        :param metrics: Name of the metric(s) to plot (string or list of strings)
        :param save_path: Optional path to save the plot (defaults to output directory)
        :param show: Whether to display the plot interactively
        :param sort_districts: If True, sort district values at each step (for district-level metrics)
        :param line_styles: Optional list of line styles for multiple metrics (e.g., ['-', '--', ':'])
        :param fig_name: Optional custom name for the saved file (without extension)
        :param ax: Optional matplotlib axes to draw on (for subplots)
        """
        from .plotting import plot_time_series, draw_time_series
        import matplotlib.pyplot as plt
        
        # Handle single metric as string
        if isinstance(metrics, str):
            metrics = [metrics]
        else:
            metrics = metrics
        
        if ax is not None:
            # Draw on provided axes (for subplots)
            draw_time_series(ax, self.df, metrics, sort_districts, line_styles)
            return self
        
        # Generate save path if not provided
        if save_path is None:
            if fig_name:
                save_path = os.path.join(self.output_dir, f"{fig_name}.png")
            elif len(metrics) == 1:
                save_path = os.path.join(self.output_dir, f"{metrics[0]}_over_steps.png")
            else:
                # Create underscore-separated name from metric names
                combined_name = "_".join(metrics)
                save_path = os.path.join(self.output_dir, f"{combined_name}_over_steps.png")
        
        plot_time_series(
            df=self.df, 
            metric_names=metrics, 
            output_path=save_path,
            sort_districts=sort_districts,
            line_styles=line_styles
        )
        
        if show:
            plt.show()
        
        return self

    def plot_metric_histogram(self, metric_name: str, user_maps: Optional[List[str]] = None, 
                             save_path: Optional[str] = None, show: bool = False,
                             fig_name: Optional[str] = None):
        """
        Plot histogram of a map-level metric.
        
        :param metric_name: Name of the metric to plot
        :param user_maps: Optional list of shapefile paths to overlay as reference lines
        :param save_path: Optional path to save the plot (defaults to output directory)
        :param show: Whether to display the plot interactively
        :param fig_name: Optional custom name for the saved file (without extension)
        """
        from .plotting import plot_distribution_histogram
        
        if save_path is None:
            if fig_name:
                save_path = os.path.join(self.output_dir, f"{fig_name}.png")
            else:
                save_path = os.path.join(self.output_dir, f"{metric_name}_distribution.png")
        
        # Extract values for the metric
        values = self.df[metric_name].dropna()
        
        # Compute reference values if user maps provided
        reference_values = None
        if user_maps:
            reference_values = []
            for map_path in user_maps:
                user_metrics = self.compute_metrics_for_map(map_path)
                if metric_name in user_metrics:
                    reference_values.append(user_metrics[metric_name])
        
        plot_distribution_histogram(
            values=values,
            metric_name=metric_name,
            reference_values=reference_values,
            output_path=save_path
        )
        
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        
        return self

    def plot_metric_violin(self, metric_name: str, user_maps: Optional[List[str]] = None, 
                          save_path: Optional[str] = None, show: bool = False, 
                          sort_districts: bool = False, fig_name: Optional[str] = None):
        """
        Plot violin plot of a district-level metric.
        
        :param metric_name: Name of the metric to plot
        :param user_maps: Optional list of shapefile paths to overlay as reference lines
        :param save_path: Optional path to save the plot (defaults to output directory)
        :param show: Whether to display the plot interactively
        :param sort_districts: If True, sort district values at each step before plotting
        :param fig_name: Optional custom name for the saved file (without extension)
        """
        from .plotting import plot_distribution_violin
        
        if save_path is None:
            if fig_name:
                save_path = os.path.join(self.output_dir, f"{fig_name}.png")
            else:
                save_path = os.path.join(self.output_dir, f"{metric_name}_distribution.png")
        
        # Compute reference values if user maps provided
        reference_values = None
        if user_maps:
            reference_values = []
            for map_path in user_maps:
                user_metrics = self.compute_metrics_for_map(map_path)
                if metric_name in user_metrics:
                    reference_values.append(user_metrics[metric_name])
        
        plot_distribution_violin(
            df=self.df,
            metric_name=metric_name,
            reference_values=reference_values,
            output_path=save_path,
            sort_districts=sort_districts
        )
        
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        
        return self

    def compute_metrics_for_map(self, shapefile_path: str) -> Dict[str, Any]:
        """
        Compute metrics for a user-defined map (shapefile).
        
        :param shapefile_path: Path to the shapefile containing the map
        :return: Dictionary of metric values
        """
        import geopandas as gpd
        import maup
        from gerrychain import Graph, GeographicPartition
        
        # Load the user map
        user_map = gpd.read_file(shapefile_path)
        
        # Reconstruct the graph and geodata from config
        # This is a simplified version - in practice, you'd need to reconstruct
        # the full graph with all the updaters from the original run
        geodata_path = self.metadata.get('initialization', {}).get('pop_geodata_path')
        if not geodata_path:
            raise ValueError("Cannot reconstruct graph: pop_geodata_path not found in config")
        
        geodata = gpd.read_file(geodata_path)
        
        # Ensure same CRS
        if geodata.crs != user_map.crs:
            user_map = user_map.to_crs(geodata.crs)
        
        # Assign districts to geodata
        geodata['user_assignment'] = maup.assign(geodata, user_map)
        
        # Build graph
        graph = Graph.from_geodataframe(geodata)
        
        # Create partition with basic updaters
        # Note: This is simplified - you'd need to reconstruct all updaters from config
        updaters = {
            "population": lambda p: {node: geodata.loc[node, 'TOTPOP'] for node in graph.nodes}
        }
        
        partition = GeographicPartition(
            graph, 
            assignment="user_assignment", 
            updaters=updaters
        )
        
        # Extract metrics (simplified - would need to match original updaters)
        metrics = {}
        for metric_name in self.df.columns:
            if metric_name == 'step':
                continue
            try:
                if self._classify_metric_type(metric_name) == 'map_level':
                    # For map-level metrics, we'd need to implement the actual computation
                    # This is a placeholder
                    metrics[metric_name] = None
                else:
                    # For district-level metrics, we'd need to implement the actual computation
                    # This is a placeholder
                    metrics[metric_name] = {}
            except:
                metrics[metric_name] = None
        
        return metrics

    def generate_standard_report(self):
        """
        Generates a standard set of plots and saves all results.
        
        This demonstrates how multiple analysis/plotting calls can be
        composed into a single, convenient method.
        """
        print("\nGenerating standard report...")
        self.save()
        self.plot_partisan_summary()
        print("✓ Report generation complete.")
        return self
