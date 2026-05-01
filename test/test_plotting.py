"""
Tests for plotting.py: visualize_partition with color_by support.
"""
import os
import tempfile
import unittest
from unittest.mock import MagicMock

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # headless
from shapely.geometry import Polygon

from utgc.plotting import _build_value_gdf, visualize_partition


def _make_partition_mock(assignment: dict, values: dict, updater_name: str) -> MagicMock:
    """
    Build a minimal mock GeographicPartition.

    Parameters
    ----------
    assignment : dict
        Mapping of node_id -> district_id.
    values : dict
        Mapping of district_id -> float (the updater values).
    updater_name : str
        The updater key whose value is ``values``.

    Returns
    -------
    MagicMock
        Mock partition with .geometries, .assignment, and [] access.
    """
    # Build a tiny GeoDataFrame indexed by node_id
    polys = {
        node: Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)])
        for i, node in enumerate(assignment.keys())
    }
    gdf = gpd.GeoDataFrame({"geometry": list(polys.values())}, index=list(polys.keys()))

    mock_part = MagicMock()
    mock_part.graph = MagicMock()
    mock_part.graph.geometry = gdf.geometry
    mock_part.graph.nodes = list(assignment.keys())
    mock_part.assignment = assignment
    mock_part.__len__ = MagicMock(return_value=len(set(assignment.values())))

    def _getitem(key):
        if key == updater_name:
            return values
        raise KeyError(key)

    mock_part.__getitem__ = MagicMock(side_effect=_getitem)

    return mock_part


class TestBuildValueGdf(unittest.TestCase):
    """Unit tests for the _build_value_gdf helper."""

    def setUp(self):
        self.assignment = {0: "A", 1: "A", 2: "B", 3: "B"}
        self.values = {"A": 0.2, "B": 0.8}
        self.updater_name = "partisan_shares"
        self.partition = _make_partition_mock(
            self.assignment, self.values, self.updater_name
        )

    def test_returns_geodataframe(self):
        """_build_value_gdf must return a GeoDataFrame."""
        result = _build_value_gdf(self.partition, self.updater_name)
        self.assertIsInstance(result, gpd.GeoDataFrame)

    def test_color_val_column_present(self):
        """Result must contain a 'color_val' column."""
        result = _build_value_gdf(self.partition, self.updater_name)
        self.assertIn("color_val", result.columns)

    def test_color_val_values_correct(self):
        """color_val must equal the updater value for each node's district."""
        result = _build_value_gdf(self.partition, self.updater_name)
        for node, district in self.assignment.items():
            expected = self.values[district]
            actual = result.loc[node, "color_val"]
            self.assertAlmostEqual(actual, expected)

    def test_geometry_preserved(self):
        """Original geometry column must be present and valid."""
        result = _build_value_gdf(self.partition, self.updater_name)
        self.assertIn("geometry", result.columns)
        self.assertTrue(result.geometry.is_valid.all())


class TestVisualizePartitionColorBy(unittest.TestCase):
    """Integration tests for visualize_partition with color_by argument."""

    def _make_full_partition_mock(self):
        """Create a mock partition suitable for visualize_partition."""
        assignment = {0: 1, 1: 1, 2: 2, 3: 2}
        values = {1: 0.3, 2: 0.7}
        updater_name = "majority_partisan_shares"
        mock = _make_partition_mock(assignment, values, updater_name)

        # partition.plot is called internally; stub it out
        mock.plot = MagicMock()
        return mock, updater_name

    def test_output_file_created_with_color_by(self):
        """visualize_partition with color_by must save a PNG to output_dir."""
        mock_part, updater_name = self._make_full_partition_mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            visualize_partition(
                partition=mock_part,
                step=1,
                output_dir=tmpdir,
                municipalities=None,
                counties=None,
                auto_load_boundaries=False,
                color_by=updater_name,
            )
            output_path = os.path.join(tmpdir, "step_00001.png")
            self.assertTrue(
                os.path.exists(output_path),
                f"Expected PNG not found at {output_path}",
            )

    def test_output_file_created_without_color_by(self):
        """visualize_partition without color_by must still save a PNG (regression)."""
        assignment = {0: 1, 1: 1, 2: 2, 3: 2}
        values = {1: 0.3, 2: 0.7}
        mock_part = _make_partition_mock(assignment, values, "unused")
        mock_part.plot = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            visualize_partition(
                partition=mock_part,
                step=2,
                output_dir=tmpdir,
                municipalities=None,
                counties=None,
                auto_load_boundaries=False,
            )
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "step_00002.png")))

    def test_fallback_on_missing_updater_key(self):
        """If color_by names a non-existent updater, fall back without raising."""
        assignment = {0: 1, 1: 2}
        values = {1: 0.4, 2: 0.6}
        mock_part = _make_partition_mock(assignment, values, "real_updater")
        mock_part.plot = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise even though "bad_key" doesn't exist
            try:
                visualize_partition(
                    partition=mock_part,
                    step=3,
                    output_dir=tmpdir,
                    municipalities=None,
                    counties=None,
                    auto_load_boundaries=False,
                    color_by="bad_key",
                )
            except Exception as exc:
                self.fail(
                    f"visualize_partition raised an unexpected exception: {exc}"
                )

    def test_color_by_norm_default_saves_file(self):
        """
        When color_by is set without color_by_norm, visualize_partition must
        still save a PNG (implicitly verifying norm defaults to (0.0, 1.0)
        without raising).
        """
        mock_part, updater_name = self._make_full_partition_mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            visualize_partition(
                partition=mock_part,
                step=4,
                output_dir=tmpdir,
                municipalities=None,
                counties=None,
                auto_load_boundaries=False,
                color_by=updater_name,
            )
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "step_00004.png")))

    def test_color_by_norm_custom_range_saves_file(self):
        """When color_by_norm is provided, visualize_partition must save a PNG."""
        mock_part, updater_name = self._make_full_partition_mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            visualize_partition(
                partition=mock_part,
                step=5,
                output_dir=tmpdir,
                municipalities=None,
                counties=None,
                auto_load_boundaries=False,
                color_by=updater_name,
                color_by_norm=(0.25, 0.75),
            )
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "step_00005.png")))


if __name__ == "__main__":
    unittest.main()
