"""
Unit tests for GeographyManager.

TDD: These tests define the expected API and behavior. They are intended to fail
until GeographyManager is implemented in utgc.geography.
"""
import os
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from gerrychain import Graph, GeographicPartition, updaters
from gerrychain.constraints import contiguous


def _minimal_geodata_path(extra_columns=None, crs="EPSG:4326", include_area=False):
    """Write a minimal GeoDataFrame to a temp GeoJSON; return path. Optional extra columns dict."""
    extra_columns = extra_columns or {}
    # Two adjacent boxes so graph has an edge
    geoms = [box(0, 0, 1, 1), box(1, 0, 2, 1)]
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2], **extra_columns},
        geometry=geoms,
        crs=crs,
    )
    if include_area:
        gdf["area"] = gdf.geometry.area
    fd, path = tempfile.mkstemp(suffix=".geojson")
    os.close(fd)
    gdf.to_file(path, driver="GeoJSON")
    return path


def _minimal_plan_geodata_path(num_districts=2, crs="EPSG:4326"):
    """Write a minimal plan (district polygons) to a temp GeoJSON; return path."""
    # Two districts aligned with _minimal_geodata_path pop units (box(0,0,1,1), box(1,0,2,1))
    geoms = [box(0, 0, 1, 1), box(1, 0, 2, 1)]
    gdf = gpd.GeoDataFrame(
        {"district": list(range(num_districts))},
        geometry=geoms,
        crs=crs,
    )
    fd, path = tempfile.mkstemp(suffix=".geojson")
    os.close(fd)
    gdf.to_file(path, driver="GeoJSON")
    return path


from utgc.geography import GeographyManager


class TestGeographyManagerInit(unittest.TestCase):
    """Initialization and readonly getters."""

    def setUp(self):
        self.path1 = _minimal_geodata_path(extra_columns={"TOTPOP": [100, 200]})
        self.path2 = _minimal_geodata_path(extra_columns={"TOTPOP": [50, 150]})
        self.addCleanup(lambda: os.path.exists(self.path1) and os.remove(self.path1))
        self.addCleanup(lambda: os.path.exists(self.path2) and os.remove(self.path2))

    def test_load_and_store_by_key_transformed_to_crs(self):
        pop_data = {"k1": self.path1, "k2": self.path2}
        manager = GeographyManager(pop_data=pop_data, crs="EPSG:26912")
        # Assert via public API: get_pop_geodata returns data in manager CRS
        gdf1 = manager.get_pop_geodata("k1")
        gdf2 = manager.get_pop_geodata("k2")
        self.assertEqual(str(gdf1.crs), "EPSG:26912")
        self.assertEqual(str(gdf2.crs), "EPSG:26912")
        self.assertEqual(len(gdf1), 2)
        self.assertEqual(len(gdf2), 2)

    def test_default_crs(self):
        pop_data = {"k1": self.path1}
        manager = GeographyManager(pop_data=pop_data)
        self.assertEqual(manager.crs, "EPSG:26912")

    def test_custom_crs(self):
        pop_data = {"k1": self.path1}
        manager = GeographyManager(pop_data=pop_data, crs="EPSG:4326")
        self.assertEqual(manager.crs, "EPSG:4326")
        gdf = manager.get_pop_geodata("k1")
        self.assertEqual(str(gdf.crs), "EPSG:4326")

    def test_pop_data_getter_returns_passed_mapping(self):
        pop_data = {"k1": self.path1, "k2": self.path2}
        manager = GeographyManager(pop_data=pop_data, crs="EPSG:26912")
        got = manager.pop_data
        self.assertEqual(got, pop_data)

    def test_pop_data_readonly_mutation_does_not_affect_manager(self):
        pop_data = {"k1": self.path1}
        manager = GeographyManager(pop_data=pop_data, crs="EPSG:26912")
        got = manager.pop_data
        original_len = len(manager.list_pop_keys())
        got["k2"] = "nonexistent"
        # Manager should still only know about k1 (internal view unchanged)
        self.assertEqual(len(manager.list_pop_keys()), original_len)

    def test_crs_getter_no_setter(self):
        pop_data = {"k1": self.path1}
        manager = GeographyManager(pop_data=pop_data, crs="EPSG:26912")
        self.assertEqual(manager.crs, "EPSG:26912")
        with self.assertRaises((AttributeError, TypeError)):
            manager.crs = "EPSG:4326"


class TestGeographyManagerColumns(unittest.TestCase):
    """Shared and unique column reporting."""

    def setUp(self):
        self.path_common = _minimal_geodata_path(extra_columns={"TOTPOP": [1, 2], "A": [10, 20]})
        self.path_extra = _minimal_geodata_path(extra_columns={"TOTPOP": [1, 2], "B": [30, 40]})
        self.addCleanup(lambda: os.path.exists(self.path_common) and os.remove(self.path_common))
        self.addCleanup(lambda: os.path.exists(self.path_extra) and os.remove(self.path_extra))

    def test_shared_columns_returns_intersection(self):
        manager = GeographyManager(
            pop_data={"c": self.path_common, "e": self.path_extra},
            crs="EPSG:26912",
        )
        shared = manager.shared_columns()
        self.assertIn("geometry", shared)
        self.assertIn("TOTPOP", shared)
        self.assertNotIn("A", shared)
        self.assertNotIn("B", shared)

    def test_columns_unique_to_returns_per_key_unique_columns(self):
        manager = GeographyManager(
            pop_data={"c": self.path_common, "e": self.path_extra},
            crs="EPSG:26912",
        )
        unique_c = manager.columns_unique_to("c")
        unique_e = manager.columns_unique_to("e")
        self.assertIn("A", unique_c)
        self.assertIn("B", unique_e)
        self.assertNotIn("B", unique_c)
        self.assertNotIn("A", unique_e)


class TestGeographyManagerTotalColumn(unittest.TestCase):
    """Create new column as sum of user-specified columns."""

    def setUp(self):
        self.path = _minimal_geodata_path(
            extra_columns={"v1": [10, 20], "v2": [5, 15]},
            include_area=True,
        )
        self.addCleanup(lambda: os.path.exists(self.path) and os.remove(self.path))

    def test_add_total_column_two_columns(self):
        manager = GeographyManager(pop_data={"k": self.path}, crs="EPSG:26912")
        manager.add_total_column("total_v", ["v1", "v2"], "k")
        gdf = manager.get_pop_geodata("k")
        self.assertIn("total_v", gdf.columns)
        self.assertEqual(gdf["total_v"].tolist(), [15, 35])

    def test_add_total_column_single_column(self):
        manager = GeographyManager(pop_data={"k": self.path}, crs="EPSG:26912")
        manager.add_total_column("copy_v1", ["v1"], "k")
        gdf = manager.get_pop_geodata("k")
        self.assertIn("copy_v1", gdf.columns)
        self.assertEqual(gdf["copy_v1"].tolist(), [10, 20])


class TestGeographyManagerElectionColumns(unittest.TestCase):
    """Election columns with optional year/office filters (G20USP format)."""

    def setUp(self):
        self.path_with_elections = _minimal_geodata_path(
            extra_columns={
                "G20USP": [100, 200],
                "G24USS": [50, 150],
                "G20USS": [80, 120],
                "TOTPOP": [1, 2],
            },
            include_area=True,
        )
        self.path_no_elections = _minimal_geodata_path(
            extra_columns={"TOTPOP": [1, 2]},
            include_area=True,
        )
        self.addCleanup(
            lambda: os.path.exists(self.path_with_elections) and os.remove(self.path_with_elections)
        )
        self.addCleanup(
            lambda: os.path.exists(self.path_no_elections) and os.remove(self.path_no_elections)
        )

    def test_get_election_columns_no_filter(self):
        manager = GeographyManager(
            pop_data={"k": self.path_with_elections},
            crs="EPSG:26912",
        )
        cols = manager.get_election_columns("k")
        self.assertIn("G20USP", cols)
        self.assertIn("G24USS", cols)
        self.assertIn("G20USS", cols)
        self.assertEqual(len(cols), 3)

    def test_get_election_columns_filter_years(self):
        manager = GeographyManager(
            pop_data={"k": self.path_with_elections},
            crs="EPSG:26912",
        )
        cols = manager.get_election_columns("k", years=[2020])
        self.assertIn("G20USP", cols)
        self.assertIn("G20USS", cols)
        self.assertNotIn("G24USS", cols)

    def test_get_election_columns_filter_offices(self):
        manager = GeographyManager(
            pop_data={"k": self.path_with_elections},
            crs="EPSG:26912",
        )
        cols = manager.get_election_columns("k", offices=["USP"])
        self.assertIn("G20USP", cols)
        self.assertNotIn("G20USS", cols)
        self.assertNotIn("G24USS", cols)

    def test_get_election_columns_both_filters(self):
        manager = GeographyManager(
            pop_data={"k": self.path_with_elections},
            crs="EPSG:26912",
        )
        cols = manager.get_election_columns("k", years=[2020], offices=["USP"])
        self.assertEqual(cols, ["G20USP"])

    def test_get_election_columns_key_with_none_returns_empty(self):
        manager = GeographyManager(
            pop_data={"k": self.path_no_elections},
            crs="EPSG:26912",
        )
        cols = manager.get_election_columns("k")
        self.assertEqual(cols, [])


class TestGeographyManagerPartition(unittest.TestCase):
    """GeographicPartition from pop key and caller-provided plan."""

    def setUp(self):
        self.pop_path = _minimal_geodata_path(
            extra_columns={"TOTPOP": [100, 200]},
            include_area=True,
        )
        self.plan_path = _minimal_plan_geodata_path(num_districts=2)
        self.addCleanup(lambda: os.path.exists(self.pop_path) and os.remove(self.pop_path))
        self.addCleanup(lambda: os.path.exists(self.plan_path) and os.remove(self.plan_path))

    def test_build_partition_plan_as_geodataframe(self):
        manager = GeographyManager(pop_data={"p": self.pop_path}, crs="EPSG:26912")
        plan_gdf = gpd.read_file(self.plan_path).to_crs("EPSG:26912")
        partition = manager.build_partition("p", plan=plan_gdf, updaters={})
        self.assertIsInstance(partition, GeographicPartition)
        self.assertEqual(len(partition), 2)

    def test_build_partition_plan_as_file_path(self):
        manager = GeographyManager(pop_data={"p": self.pop_path}, crs="EPSG:26912")
        partition = manager.build_partition("p", plan=self.plan_path, updaters={})
        self.assertIsInstance(partition, GeographicPartition)
        self.assertEqual(len(partition), 2)

    def test_build_partition_contiguity_repair_default(self):
        manager = GeographyManager(pop_data={"p": self.pop_path}, crs="EPSG:26912")
        partition = manager.build_partition("p", plan=self.plan_path, updaters={})
        self.assertTrue(contiguous(partition))

    def test_build_partition_repair_contiguity_false(self):
        manager = GeographyManager(pop_data={"p": self.pop_path}, crs="EPSG:26912")
        partition = manager.build_partition(
            "p", plan=self.plan_path, updaters={}, repair_contiguity=False
        )
        self.assertIsInstance(partition, GeographicPartition)

    def test_build_partition_with_updaters(self):
        manager = GeographyManager(pop_data={"p": self.pop_path}, crs="EPSG:26912")
        up = {"population": updaters.Tally("TOTPOP", alias="population")}
        partition = manager.build_partition("p", plan=self.plan_path, updaters=up)
        self.assertIn("population", partition.updaters)
        self.assertIsNotNone(partition["population"])


class TestGeographyManagerGraph(unittest.TestCase):
    """get_graph(pop_key) returns Graph from loaded geodata."""

    def setUp(self):
        self.path = _minimal_geodata_path(
            extra_columns={"TOTPOP": [100, 200]},
            include_area=True,
        )
        self.addCleanup(lambda: os.path.exists(self.path) and os.remove(self.path))

    def test_get_graph_returns_graph_with_expected_nodes_and_attributes(self):
        manager = GeographyManager(pop_data={"blocks": self.path}, crs="EPSG:26912")
        graph = manager.get_graph("blocks")
        self.assertIsInstance(graph, Graph)
        self.assertEqual(graph.number_of_nodes(), 2)
        for node in graph.nodes:
            self.assertIn("TOTPOP", graph.nodes[node])


class TestGeographyManagerGeodataAccess(unittest.TestCase):
    """get_pop_geodata(key) returns loaded GeoDataFrame in CRS."""

    def setUp(self):
        self.path = _minimal_geodata_path(
            extra_columns={"TOTPOP": [100, 200]},
            include_area=True,
        )
        self.addCleanup(lambda: os.path.exists(self.path) and os.remove(self.path))

    def test_get_pop_geodata_returns_geodataframe_in_crs_with_columns(self):
        manager = GeographyManager(pop_data={"blocks": self.path}, crs="EPSG:26912")
        gdf = manager.get_pop_geodata("blocks")
        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        self.assertEqual(str(gdf.crs), "EPSG:26912")
        self.assertIn("geometry", gdf.columns)
        self.assertIn("TOTPOP", gdf.columns)


class TestGeographyManagerGeodataAccessWithDataDir(unittest.TestCase):
    """When data/ exists, get_pop_geodata with data/UT_blocks.geojson."""

    @unittest.skipUnless(
        os.path.exists("data/UT_blocks.geojson"),
        "data/UT_blocks.geojson not found",
    )
    def test_get_pop_geodata_blocks_returns_geodataframe(self):
        manager = GeographyManager(
            pop_data={"blocks": "data/UT_blocks.geojson"},
            crs="EPSG:26912",
        )
        gdf = manager.get_pop_geodata("blocks")
        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        self.assertEqual(str(gdf.crs), "EPSG:26912")
        self.assertIn("geometry", gdf.columns)


class TestGeographyManagerListKeys(unittest.TestCase):
    """list_pop_keys() returns population dataset keys."""

    def setUp(self):
        self.path1 = _minimal_geodata_path(extra_columns={"TOTPOP": [1, 2]})
        self.path2 = _minimal_geodata_path(extra_columns={"TOTPOP": [3, 4]})
        self.addCleanup(lambda: os.path.exists(self.path1) and os.remove(self.path1))
        self.addCleanup(lambda: os.path.exists(self.path2) and os.remove(self.path2))

    def test_list_pop_keys_returns_expected_keys(self):
        manager = GeographyManager(
            pop_data={"a": self.path1, "b": self.path2},
            crs="EPSG:26912",
        )
        keys = manager.list_pop_keys()
        self.assertEqual(set(keys), {"a", "b"})
        self.assertEqual(len(keys), 2)


class TestGeographyManagerAreaColumn(unittest.TestCase):
    """Area column added on load when missing."""

    def setUp(self):
        # Do not pass include_area so file has no "area" column
        self.path = _minimal_geodata_path(extra_columns={"TOTPOP": [1, 2]})
        self.addCleanup(lambda: os.path.exists(self.path) and os.remove(self.path))

    def test_loaded_geodata_has_area_column_when_missing_in_file(self):
        manager = GeographyManager(pop_data={"k": self.path}, crs="EPSG:26912")
        gdf = manager.get_pop_geodata("k")
        self.assertIn("area", gdf.columns)
        pd.testing.assert_series_equal(
            gdf["area"],
            gdf.geometry.area,
            check_names=False,
        )


class TestGeographyManagerFillEmptyIds(unittest.TestCase):
    """fill_empty_ids(key, columns) fills only empty/None; leaves non-empty and duplicates unchanged."""

    def setUp(self):
        # Four adjacent boxes; MUNIID has empty, NaN, and duplicate non-empty "101"
        geoms = [box(i, 0, i + 1, 1) for i in range(4)]
        gdf = gpd.GeoDataFrame(
            {
                "MUNIID": ["", pd.NA, "101", "101"],
                "TOTPOP": [10, 20, 30, 40],
            },
            geometry=geoms,
            crs="EPSG:4326",
        )
        fd, self.path = tempfile.mkstemp(suffix=".geojson")
        os.close(fd)
        gdf.to_file(self.path, driver="GeoJSON")
        self.addCleanup(lambda: os.path.exists(self.path) and os.remove(self.path))

    def test_fill_empty_ids_fills_only_empty_and_nan(self):
        manager = GeographyManager(pop_data={"k": self.path}, crs="EPSG:26912")
        manager.fill_empty_ids("k", ["MUNIID"])
        gdf = manager.get_pop_geodata("k")
        col = gdf["MUNIID"]
        self.assertFalse(col.isna().any())
        self.assertFalse((col.astype(str).str.strip() == "").any())
        # Duplicate "101" must still be present (two rows with same value)
        self.assertEqual(col.iloc[2], col.iloc[3])
        self.assertEqual(col.iloc[2], "101")


if __name__ == "__main__":
    unittest.main()
