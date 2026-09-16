"""The benchmark inventory: what the suite would measure, described exactly.

Structural work on the benchmark tree is only safe if the matrix it produces
can be compared before and after. These tests pin the shape of that
description — the fields every row carries, the statuses it may report, and
the agreement between rows and their summary — so that a full inventory taken
across a refactor is comparing like with like.

Building every case allocates operands for thousands of combinations and is
far too slow for the unit suite, so the real build here is limited to two
cheap suites. Coverage of the rest is asserted structurally: every registered
suite must resolve to groups.
"""

import unittest

import tensors as ts

from benchmarks import inventory
from benchmarks import registry

#: Cheap suites that still exercise both a successful build and a decline.
SAMPLE_SUITES = ("framework", "shape")

REQUIRED_FIELDS = frozenset(
    {
        "name",
        "backend",
        "suite",
        "group",
        "layer",
        "family",
        "dtype",
        "shape",
        "elements",
        "work_items",
        "tags",
        "memory",
        "single_shot",
        "status",
        "reason",
    }
)

STATUSES = frozenset({"supported", "unsupported", "error"})


class InventoryRowTests(unittest.TestCase):
    """Every row describes one case on one backend, completely."""

    @classmethod
    def setUpClass(cls):
        cls.rows = inventory.build(SAMPLE_SUITES)

    def test_the_sample_suites_produce_rows(self):
        self.assertTrue(self.rows)
        self.assertEqual(
            sorted({row["suite"] for row in self.rows}), sorted(SAMPLE_SUITES)
        )

    def test_every_row_carries_every_field(self):
        for row in self.rows:
            with self.subTest(name=row["name"], backend=row["backend"]):
                self.assertEqual(set(row), REQUIRED_FIELDS)

    def test_every_row_reports_a_known_status(self):
        for row in self.rows:
            with self.subTest(name=row["name"]):
                self.assertIn(row["status"], STATUSES)

    def test_a_declined_case_says_why(self):
        """An exclusion is part of the matrix, so it has to be legible."""
        declined = [row for row in self.rows if row["status"] != "supported"]
        for row in declined:
            with self.subTest(name=row["name"]):
                self.assertTrue(row["reason"], f"{row['name']} declined silently")

    def test_no_case_reports_a_build_error(self):
        errors = [
            f"{row['backend']}::{row['name']}: {row['reason']}"
            for row in self.rows
            if row["status"] == "error"
        ]
        self.assertEqual(errors, [], "\n".join(errors))

    def test_rows_cover_every_available_backend(self):
        self.assertEqual(
            sorted({row["backend"] for row in self.rows}),
            sorted(ts.available_backends()),
        )

    def test_shapes_are_json_representable(self):
        """A tuple shape becomes a list so a stored inventory round-trips."""
        for row in self.rows:
            with self.subTest(name=row["name"]):
                self.assertNotIsInstance(row["shape"], tuple)

    def test_rows_are_ordered_deterministically(self):
        """Two inventories must diff as a diff, not as a reordering."""
        again = inventory.build(SAMPLE_SUITES)
        self.assertEqual(
            [(row["backend"], row["suite"], row["name"]) for row in again],
            [(row["backend"], row["suite"], row["name"]) for row in self.rows],
        )


class InventorySummaryTests(unittest.TestCase):
    """The summary is a view of the rows, not a second measurement."""

    @classmethod
    def setUpClass(cls):
        cls.rows = inventory.build(SAMPLE_SUITES)
        cls.totals = inventory.summarize(cls.rows)

    def test_the_total_counts_the_rows(self):
        self.assertEqual(self.totals["total"], len(self.rows))

    def test_each_breakdown_sums_to_the_total(self):
        for axis in ("by_status", "by_layer"):
            with self.subTest(axis=axis):
                self.assertEqual(sum(self.totals[axis].values()), self.totals["total"])

    def test_the_suite_breakdown_sums_to_the_total(self):
        counted = sum(
            count
            for counts in self.totals["by_suite"].values()
            for count in counts.values()
        )
        self.assertEqual(counted, self.totals["total"])


class SuiteCoverageTests(unittest.TestCase):
    """Every registered suite resolves, without building its cases."""

    def test_every_registered_suite_produces_groups(self):
        for name in registry.DEFAULT_SUITES:
            with self.subTest(suite=name):
                self.assertTrue(registry.suite_groups(name), f"{name} is empty")

    def test_every_group_names_its_suite(self):
        for name in registry.DEFAULT_SUITES:
            for group in registry.suite_groups(name):
                with self.subTest(group=group.name):
                    self.assertEqual(group.suite, name)


if __name__ == "__main__":
    unittest.main()
