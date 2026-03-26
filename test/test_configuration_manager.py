"""
TDD tests for ConfigurationManager.

ConfigurationManager is geography-free and provides only what is needed to
construct native GerryChain MCMC runs: proposal, constraints, updaters.
It can construct itself from a serialized YAML description.
"""
import os
import tempfile
import unittest

from gerrychain import Graph, GeographicPartition, Partition, updaters
from gerrychain.constraints import contiguous

import utgc.run_utils as rutil
from utgc.configuration import ConfigurationManager


class TestConfigurationManagerInit(unittest.TestCase):
    """Init: no arguments; valid minimal state."""

    def test_init_no_args(self):
        config = ConfigurationManager()
        self.assertIsInstance(config.construction_history, list)
        self.assertEqual(len(config.construction_history), 0)
        self.assertIsInstance(config.region_surcharges, dict)
        self.assertIsInstance(config.constraints, list)
        self.assertIn(contiguous, config.constraints)
        self.assertIsInstance(config.updaters, dict)
        # Population updater may be present with default pop_column or added later
        self.assertIsInstance(config.population_params, dict)
        self.assertIn("column_id", config.population_params)

    def test_init_minimal_state(self):
        config = ConfigurationManager()
        self.assertEqual(config.region_surcharges, {})
        self.assertEqual(len(config.constraints), 1)  # contiguous only
        self.assertIsInstance(config.edge_penalties, dict)


class TestProposalAndFluentSetters(unittest.TestCase):
    """Proposal with args when needed; fluent setters for pop_column and pop_dev."""

    def test_set_pop_column(self):
        config = ConfigurationManager().set_pop_column("POP")
        self.assertEqual(config.population_params["column_id"], "POP")

    def test_add_pop_dev_updater(self):
        config = ConfigurationManager()
        config.set_pop_column("TOTPOP")
        # pop_dev needs ideal_pop at call time; add_pop_dev_updater can store
        # a closure that gets ideal_pop from partition or we pass at proposal time.
        # For now we need config to support add_pop_dev_updater; it may require
        # ideal_pop to be set (e.g. via a setter) before use.
        config.add_pop_dev_updater(name="pop_dev")
        self.assertIn("pop_dev", config.updaters)

    def test_proposal_requires_population_args(self):
        """proposal(partition, total_population=..., num_districts=..., pop_tolerance=...)"""
        config = ConfigurationManager()
        config.set_pop_column("TOTPOP")
        config.surcharge_region("MUNIID", 0.5)
        # Minimal graph with an internal node (degree > 1) so ReCom can run
        graph = Graph()
        for i in range(4):
            graph.add_node(i, TOTPOP=100)
        graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
        part = GeographicPartition(
            graph,
            assignment={0: 0, 1: 0, 2: 1, 3: 1},
            updaters={"population": updaters.Tally("TOTPOP", alias="population")},
        )
        total_pop = 400
        num_districts = 2
        pop_tolerance = 0.01
        prop = config.proposal(part, total_population=total_pop, num_districts=num_districts, pop_tolerance=pop_tolerance)
        self.assertTrue(callable(prop))
        next_part = prop(part)
        self.assertIsInstance(next_part, Partition)


class TestRegionSurcharges(unittest.TestCase):
    """surcharge_region(column_id, surcharge); assert region_surcharges."""

    def test_surcharge_region(self):
        config = ConfigurationManager().surcharge_region("MUNIID", 1.0)
        self.assertEqual(config.region_surcharges["MUNIID"], 1.0)

    def test_surcharge_region_multiple(self):
        config = (
            ConfigurationManager()
            .surcharge_region("MUNIID", 1.0)
            .surcharge_region("COUNTYID", 0.5)
        )
        self.assertEqual(config.region_surcharges["MUNIID"], 1.0)
        self.assertEqual(config.region_surcharges["COUNTYID"], 0.5)

    def test_surcharge_zero_skipped(self):
        config = ConfigurationManager().surcharge_region("MUNIID", 0)
        self.assertNotIn("MUNIID", config.region_surcharges)


class TestLocalitySplits(unittest.TestCase):
    """add_locality_splits_updater(name, column_id); ls_*, split_*, *_multi_splits."""

    def test_add_locality_splits_updater(self):
        config = ConfigurationManager()
        config.set_pop_column("TOTPOP")
        config.add_locality_splits_updater(name="muni", column_id="MUNIID")
        self.assertIn("ls_muni", config.updaters)
        self.assertIn("split_muni", config.updaters)
        self.assertIn("muni_multi_splits", config.updaters)


class TestConstraints(unittest.TestCase):
    """constrain_region_splits, constrain_not_equal; assert constraints list."""

    def test_constrain_region_splits_creates_constraints(self):
        config = ConfigurationManager()
        config.set_pop_column("TOTPOP")
        config.constrain_region_splits(
            name="muni",
            column_id="MUNIID",
            num_split=2,
            num_multi_splits=0,
            create_updater=True,
        )
        self.assertGreater(len(config.constraints), 1)  # contiguous + at least one UpperBound
        self.assertIn("constrain_region_splits", [s.get("method") for s in config.construction_history])

    def test_constrain_not_equal(self):
        config = ConfigurationManager().constrain_not_equal(not_equal_constraint=True, create_updater=True)
        constraint_types = [type(c) for c in config.constraints]
        self.assertIn(rutil.NotEqual, constraint_types)
        self.assertIn("assignment_hash", config.updaters)


class TestElections(unittest.TestCase):
    """add_election_updater, add_election_aggregator, add_election_metric_updaters."""

    def test_add_election_updater(self):
        config = (
            ConfigurationManager()
            .add_election_updater("2020GEN", parties_to_columns={"D": "G20PRED", "R": "G20PRER"})
        )
        self.assertIn("2020GEN", config.updaters)

    def test_add_election_aggregator(self):
        config = ConfigurationManager()
        config.add_election_updater("2020GEN", parties_to_columns={"D": "G20PRED", "R": "G20PRER"})
        config.add_election_aggregator("pres", elections=["2020GEN"], parties=["D", "R", "-"])
        self.assertIn("pres_table", config.updaters)
        self.assertIn("pres", config.updaters)


class TestShapeMetrics(unittest.TestCase):
    """add_shape_metrics([...]); perimeter, area, polsby_popper or reock."""

    def test_add_shape_metrics_polsby_popper(self):
        config = ConfigurationManager().add_shape_metrics(["polsby_popper"])
        self.assertIn("polsby_popper", config.updaters)
        self.assertIn("perimeter", config.updaters)
        self.assertIn("area", config.updaters)


class TestToConfigFromConfigRoundTrip(unittest.TestCase):
    """to_config / from_config round-trip; compare serializable state."""

    def test_round_trip_surcharges_and_updaters(self):
        config = (
            ConfigurationManager()
            .set_pop_column("TOTPOP")
            .surcharge_region("MUNIID", 1.0)
            .add_locality_splits_updater(name="muni", column_id="MUNIID")
        )
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        config.to_config(path)
        loaded = ConfigurationManager.from_config(path, verbose=False)
        self.assertEqual(loaded.region_surcharges, config.region_surcharges)
        self.assertIn("ls_muni", loaded.updaters)
        self.assertIn("split_muni", loaded.updaters)
        self.assertEqual(len(loaded.constraints), len(config.constraints))

    def test_from_config_verbose_default_no_print(self):
        """Optional logging: from_config(..., verbose=False) does not burden bulk deserialization."""
        config = ConfigurationManager().surcharge_region("X", 0.1)
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        config.to_config(path)
        # Should not raise; verbose=False or default should minimize output
        loaded = ConfigurationManager.from_config(path, verbose=False)
        self.assertEqual(loaded.region_surcharges, {"X": 0.1})


class TestEdgePenalties(unittest.TestCase):
    """penalize_edges_from_csv; edge_penalties and from_config path resolution."""

    def test_penalize_edges_from_csv(self):
        fd, csv_path = tempfile.mkstemp(suffix=".csv")
        os.write(fd, b"u,v,w\n0,1,1.0\n1,2,0.5\n")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(csv_path) and os.remove(csv_path))
        config = ConfigurationManager().penalize_edges_from_csv(csv_path, penalty=0.3)
        self.assertIn((0, 1), config.edge_penalties)
        self.assertIn((1, 2), config.edge_penalties)

    def test_from_config_resolves_relative_path(self):
        fd, csv_path = tempfile.mkstemp(suffix=".csv")
        os.write(fd, b"u,v,w\n0,1,1.0\n")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(csv_path) and os.remove(csv_path))
        config = ConfigurationManager().penalize_edges_from_csv(csv_path, penalty=0.2)
        config_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: os.path.exists(config_dir) and os.rmdir(config_dir) if os.listdir(config_dir) == [] else None)
        yaml_path = os.path.join(config_dir, "config.yaml")
        config.to_config(yaml_path)
        # from_config should resolve path relative to config file dir
        loaded = ConfigurationManager.from_config(yaml_path, verbose=False)
        self.assertGreater(len(loaded.edge_penalties), 0)


class TestColumnCoordination(unittest.TestCase):
    """Same column id for locality splits and surcharges; pop_column shared."""

    def test_column_coordination(self):
        config = (
            ConfigurationManager()
            .set_pop_column("TOTPOP")
            .surcharge_region("MUNIID", 0.5)
            .add_locality_splits_updater(name="muni", column_id="MUNIID")
        )
        self.assertEqual(config.region_surcharges["MUNIID"], 0.5)
        # LocalitySplits uses col_id=MUNIID and pop_col from config
        self.assertEqual(config.population_params["column_id"], "TOTPOP")
        self.assertIn("ls_muni", config.updaters)


class TestGetConstraintParams(unittest.TestCase):
    """get_constraint_params() derives ceiling dict from construction_history."""

    def test_empty_history_returns_empty_dict(self):
        config = ConfigurationManager()
        self.assertEqual(config.get_constraint_params(), {})

    def test_single_split_constraint(self):
        config = ConfigurationManager()
        config.set_pop_column("TOTPOP")
        config.constrain_region_splits(name="muni", column_id="MUNIID", num_split=2)
        params = config.get_constraint_params()
        self.assertEqual(params["split_muni"], 2)
        self.assertNotIn("muni_multi_splits", params)

    def test_split_and_multi_split_constraints(self):
        config = ConfigurationManager()
        config.set_pop_column("TOTPOP")
        config.constrain_region_splits(
            name="muni", column_id="MUNIID", num_split=2, num_multi_splits=1
        )
        params = config.get_constraint_params()
        self.assertEqual(params["split_muni"], 2)
        self.assertEqual(params["muni_multi_splits"], 1)

    def test_multiple_region_constraints(self):
        config = ConfigurationManager()
        config.set_pop_column("TOTPOP")
        config.constrain_region_splits(name="muni", column_id="MUNIID", num_split=2)
        config.constrain_region_splits(name="county", column_id="COUNTYID", num_split=3)
        params = config.get_constraint_params()
        self.assertEqual(params["split_muni"], 2)
        self.assertEqual(params["split_county"], 3)

    def test_not_equal_constraint_included(self):
        config = ConfigurationManager()
        config.constrain_not_equal(not_equal_constraint=True, create_updater=True)
        params = config.get_constraint_params()
        self.assertIn("not_equal_constraint", params)
        self.assertTrue(params["not_equal_constraint"])

    def test_surcharges_do_not_appear_in_constraint_params(self):
        config = ConfigurationManager()
        config.surcharge_region("MUNIID", 1.0)
        params = config.get_constraint_params()
        self.assertNotIn("MUNIID", params)
        self.assertEqual(params, {})


if __name__ == "__main__":
    unittest.main()
