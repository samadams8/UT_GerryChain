import unittest
import json
import pandas as pd
from typing import Dict, Any

# Import the collect_metrics function directly since the rest of cong_equalpop.py requires setup
from cong_equalpop import collect_metrics

class MockPartition:
    """A mock partition object for testing updater collection."""
    def __init__(self, updaters: Dict[str, Any]):
        self.updaters = updaters

    def __getitem__(self, key: str) -> Any:
        return self.updaters[key]

class TestCongEqualPop(unittest.TestCase):
    def test_collect_metrics_dataframe_serialization(self):
        """Test that collect_metrics properly handles pandas DataFrames."""
        
        # Create a sample DataFrame similar to what sb1011_data would produce
        df = pd.DataFrame({
            "metrics": ["partisan_bias", "mean_median", "efficiency_gap"],
            "value": [0.05, 0.02, 0.08]
        })

        # Mock partition with a dataframe and some standard updaters
        partition = MockPartition({
            "step": 1,
            "population": {"1": 100, "2": 100},
            "sb1011_data": df
        })
        
        updater_names = ["population", "sb1011_data"]
        
        # Collect the metrics
        metrics = collect_metrics(partition, 1, updater_names)
        
        # Attempt to serialize to JSON
        # This should succeed and not raise a TypeError
        try:
            json_str = json.dumps(metrics)
            # Verify the dataframe was serialized reasonably
            parsed = json.loads(json_str)
            self.assertIn("sb1011_data", parsed)
            self.assertTrue(isinstance(parsed["sb1011_data"], (dict, list)))
        except TypeError as e:
            self.fail(f"collect_metrics produced a dictionary that is not JSON serializable: {e}")

if __name__ == '__main__':
    unittest.main()
