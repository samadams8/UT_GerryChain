import os
import pandas as pd
from typing import Dict, Any

# Import the new plotting functionality
from .plotting import plot_partisan_summary

class ResultSet:
    """
    A container for the results of an ensemble run.

    This object holds the summary DataFrame of all generated plans and provides
    methods for analysis, plotting, and saving.
    """
    def __init__(self, summary_df: pd.DataFrame, metadata: Dict[str, Any], output_dir: str):
        """
        Initializes the ResultSet.

        :param summary_df: A pandas DataFrame where each row represents one
                           districting plan from the ensemble.
        :param metadata: A dictionary containing configuration details of the run.
        :param output_dir: The directory where results should be saved.
        """
        self.summary_df = summary_df
        self.metadata = metadata
        self.output_dir = output_dir
        
        print("✓ ResultSet created.")
        print(f"  - Contains results for {len(self.summary_df)} plans.")

    def __repr__(self) -> str:
        """Provides a string representation of the ResultSet."""
        return f"<ResultSet with {len(self.summary_df)} plans>"

    def save(self) -> None:
        """
        Saves the result summary to a CSV file in the run's output directory.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        path = os.path.join(self.output_dir, "ensemble_summary.csv")
        self.summary_df.to_csv(path, index=False)
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
            summary_df=self.summary_df,
            elections=self.metadata.get("tracked_elections", []),
            output_dir=self.output_dir
        )
        return self # Allow for chaining
        
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
