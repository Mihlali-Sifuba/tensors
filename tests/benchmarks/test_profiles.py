"""Profiles select from the matrix; they never add to it.

The point of a profile is that a suite keeps saying which sizes and dtypes
its question needs, while the profile says how much of that to run today. So
the property that matters is one-directional: every profile's selection is a
subset of what the suites declare, and ``standard`` is the whole of it.
"""

import unittest

from benchmarks import profiles
from benchmarks.workloads import ELEMENTWISE_SIZES, NUMERIC_DTYPES, ceiling_for


class ScaleTests(unittest.TestCase):
    """Every element count falls in exactly one category."""

    def test_each_size_has_one_category(self):
        for size in ELEMENTWISE_SIZES:
            with self.subTest(size=size):
                self.assertIn(profiles.scale_of(size), profiles.SCALE_NAMES)

    def test_the_categories_ascend_with_size(self):
        order = {name: index for index, name in enumerate(profiles.SCALE_NAMES)}
        ranks = [order[profiles.scale_of(size)] for size in ELEMENTWISE_SIZES]
        self.assertEqual(ranks, sorted(ranks))

    def test_a_size_beyond_every_bound_is_the_largest_category(self):
        self.assertEqual(profiles.scale_of(10**12), profiles.SCALE_NAMES[-1])


class ProfileSelectionTests(unittest.TestCase):
    """A profile filters a declared curve rather than supplying one."""

    def test_standard_admits_everything_a_suite_declares(self):
        with profiles.use("standard"):
            self.assertEqual(
                profiles.selected_sizes(ELEMENTWISE_SIZES), ELEMENTWISE_SIZES
            )
            self.assertEqual(profiles.selected_dtypes(NUMERIC_DTYPES), NUMERIC_DTYPES)

    def test_every_profile_selects_a_subset(self):
        for name in profiles.PROFILE_NAMES:
            with self.subTest(profile=name), profiles.use(name):
                sizes = profiles.selected_sizes(ELEMENTWISE_SIZES)
                dtypes = profiles.selected_dtypes(NUMERIC_DTYPES)
                self.assertTrue(set(sizes) <= set(ELEMENTWISE_SIZES))
                self.assertTrue(set(dtypes) <= set(NUMERIC_DTYPES))

    def test_a_selection_keeps_the_declared_order(self):
        with profiles.use("quick"):
            sizes = profiles.selected_sizes(ELEMENTWISE_SIZES)
        self.assertEqual(list(sizes), sorted(sizes))

    def test_quick_is_smaller_than_standard(self):
        with profiles.use("standard"):
            full = profiles.selected_sizes(ELEMENTWISE_SIZES)
        with profiles.use("quick"):
            narrowed = profiles.selected_sizes(ELEMENTWISE_SIZES)
        self.assertLess(len(narrowed), len(full))

    def test_an_unfiltered_profile_admits_an_unknown_dtype(self):
        """An empty selection is no restriction, not an empty result."""
        with profiles.use("standard"):
            self.assertEqual(profiles.selected_dtypes(("bfloat16",)), ("bfloat16",))


class CeilingTests(unittest.TestCase):
    """The tighter of the workload's ceiling and the profile's wins."""

    def test_a_profile_may_only_lower_a_ceiling(self):
        declared = {"python": 100_000, "numpy": 10_000_000, "cuda": 10_000_000}
        for name in profiles.PROFILE_NAMES:
            for backend, limit in declared.items():
                with self.subTest(profile=name, backend=backend), profiles.use(name):
                    self.assertLessEqual(ceiling_for(backend, declared), limit)

    def test_standard_leaves_a_declared_ceiling_alone(self):
        declared = {"python": 100_000, "numpy": 10_000_000, "cuda": 10_000_000}
        with profiles.use("standard"):
            for backend, limit in declared.items():
                with self.subTest(backend=backend):
                    self.assertEqual(ceiling_for(backend, declared), limit)


class ProfileRegistryTests(unittest.TestCase):
    """Each named profile exists, is distinct, and describes itself."""

    def test_every_name_loads(self):
        for name in profiles.PROFILE_NAMES:
            with self.subTest(profile=name):
                self.assertEqual(profiles.load(name).name, name)

    def test_an_unknown_name_is_refused(self):
        with self.assertRaisesRegex(KeyError, "unknown profile"):
            profiles.load("thorough")

    def test_each_profile_states_rounds_a_target_and_a_description(self):
        for name in profiles.PROFILE_NAMES:
            with self.subTest(profile=name):
                profile = profiles.load(name)
                self.assertGreater(profile.rounds, 0)
                self.assertGreater(profile.target_seconds, 0.0)
                self.assertTrue(profile.description)

    def test_comprehensive_measures_harder_than_quick(self):
        quick, comprehensive = profiles.load("quick"), profiles.load("comprehensive")
        self.assertGreater(comprehensive.rounds, quick.rounds)
        self.assertGreater(comprehensive.target_seconds, quick.target_seconds)
        self.assertTrue(comprehensive.collect_memory)

    def test_the_default_profile_is_standard(self):
        self.assertEqual(profiles.active().name, "standard")

    def test_use_restores_the_previous_profile(self):
        with profiles.use("quick"):
            self.assertEqual(profiles.active().name, "quick")
            with profiles.use("comprehensive"):
                self.assertEqual(profiles.active().name, "comprehensive")
            self.assertEqual(profiles.active().name, "quick")
        self.assertEqual(profiles.active().name, "standard")

    def test_the_active_profile_is_reportable(self):
        with profiles.use("quick"):
            recorded = profiles.settings()
        self.assertEqual(recorded["name"], "quick")
        self.assertEqual(recorded["rounds"], 3)
        self.assertIn("ceilings", recorded)


if __name__ == "__main__":
    unittest.main()
