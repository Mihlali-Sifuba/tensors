"""Direct coverage for the numerical primitives shared below both layers.

These used to be private helpers inside ``tensors.math``, reached only through
an operation. They now have their own neutral home, and their contracts are
subtle enough that a behavioural test through a public operation would not say
which primitive broke. Expected values are derived with exact arithmetic
rather than written as literals.
"""

import math
import unittest
from fractions import Fraction

from tensors.utils.deviation import scaled_deviations
from tensors.utils.normalization import shifted_normalization
from tensors.utils.reductions import (
    keepdims_shape,
    normalize_axes,
    reduction_groups,
    reduction_shape,
    reduction_size,
)
from tensors.utils.summation import (
    stable_float_mean,
    stable_float_sum,
    stable_product_sum,
    sum_exact_ratios,
)


class StableFloatSumTests(unittest.TestCase):
    """Summation survives cancellation and an overflowing partial total."""

    def test_extreme_cancellation_keeps_the_small_terms(self):
        values = [1e16, 1.0, -1e16, 1.0, 1e16, -1.0, -1e16, -1.0]

        self.assertEqual(stable_float_sum(values), 0.0)

    def test_small_terms_survive_a_dominant_magnitude(self):
        values = [1e16, 1.0, -1e16]

        self.assertEqual(stable_float_sum(values), 1.0)

    def test_an_overflowing_partial_sum_still_returns_a_finite_total(self):
        values = [1e308, 1e308, -1e308]

        result = stable_float_sum(values)

        self.assertTrue(math.isfinite(result))
        self.assertEqual(result, 1e308)

    def test_infinities_of_both_signs_are_not_a_number(self):
        self.assertTrue(math.isnan(stable_float_sum([math.inf, -math.inf])))

    def test_a_single_infinity_propagates_with_its_sign(self):
        self.assertEqual(stable_float_sum([math.inf, 1.0]), math.inf)
        self.assertEqual(stable_float_sum([-math.inf, 1.0]), -math.inf)

    def test_any_nan_propagates(self):
        self.assertTrue(math.isnan(stable_float_sum([1.0, math.nan, 2.0])))

    def test_an_empty_sum_is_zero(self):
        self.assertEqual(stable_float_sum([]), 0.0)


class StableProductSumTests(unittest.TestCase):
    """Products are summed even when each product leaves float range."""

    def test_products_that_overflow_individually_still_sum(self):
        factors = [(1e200, 1e200), (1e200, -1e200)]

        # Each product is inf and -inf on its own; the exact sum is zero.
        self.assertEqual(factors[0][0] * factors[0][1], math.inf)
        self.assertEqual(stable_product_sum(factors), 0.0)

    def test_a_small_term_survives_products_that_left_float_range(self):
        factors = [(1e200, 1e200), (1e200, -1e200), (1.0, 2.0)]

        # Naively the first two products are inf and -inf, so their sum is
        # NaN and the 2.0 is lost. Exactly, they cancel and 2.0 remains.
        self.assertTrue(math.isnan(sum(left * right for left, right in factors)))
        self.assertEqual(stable_product_sum(factors), 2.0)

    def test_products_below_float_range_round_to_zero(self):
        factors = [(1e-200, 1e-200), (1e-200, 1e-200)]

        # The exact total is about 2e-400, which no float represents.
        self.assertEqual(stable_product_sum(factors), 0.0)

    def test_a_single_representable_product_is_returned_directly(self):
        self.assertEqual(stable_product_sum([(2.5, 4.0)]), 10.0)

    def test_ordinary_products_match_exact_arithmetic(self):
        factors = [(0.1, 0.2), (0.3, 0.4), (-0.5, 0.6)]
        expected = float(
            sum(Fraction(left) * Fraction(right) for left, right in factors)
        )

        self.assertAlmostEqual(stable_product_sum(factors), expected, places=15)


class SumExactRatiosTests(unittest.TestCase):
    """Binary ratios convert to the float their exact sum rounds to."""

    def test_ratios_sum_exactly(self):
        ratios = [(1, 2), (1, 4), (1, 8)]

        self.assertEqual(sum_exact_ratios(ratios), 0.875)

    def test_a_divisor_is_applied_to_the_exact_total(self):
        ratios = [(1, 1), (1, 1), (1, 1)]

        self.assertEqual(sum_exact_ratios(ratios, divisor=3), 1.0)

    def test_an_unrepresentable_total_saturates_with_its_sign(self):
        huge = (10**400, 1)

        self.assertEqual(sum_exact_ratios([huge]), math.inf)
        self.assertEqual(sum_exact_ratios([(-(10**400), 1)]), -math.inf)


class StableFloatMeanTests(unittest.TestCase):
    """Means avoid both an overflowing sum and an underflowing term."""

    def test_a_mean_of_extremes_does_not_overflow(self):
        values = [1e308, 1e308]

        self.assertEqual(stable_float_mean(values), 1e308)

    def test_a_mean_matches_exact_arithmetic(self):
        values = [0.1, 0.2, 0.3]
        expected = float(sum(Fraction(value) for value in values) / len(values))

        self.assertEqual(stable_float_mean(values), expected)

    def test_an_empty_mean_is_not_a_number(self):
        self.assertTrue(math.isnan(stable_float_mean([])))

    def test_a_non_finite_value_falls_back_to_the_stable_sum(self):
        self.assertEqual(stable_float_mean([math.inf, 1.0]), math.inf)


class ShiftedNormalizationTests(unittest.TestCase):
    """The shifted normalizer keeps tails a naive exponential would lose."""

    def test_probabilities_sum_to_one(self):
        _, _, probabilities, _ = shifted_normalization([1.0, 2.0, 3.0])

        self.assertAlmostEqual(math.fsum(probabilities), 1.0)

    def test_a_large_common_offset_does_not_overflow(self):
        _, _, probabilities, _ = shifted_normalization([1000.0, 1001.0])

        self.assertAlmostEqual(probabilities[0], 1.0 / (1.0 + math.e))
        self.assertAlmostEqual(probabilities[1], math.e / (1.0 + math.e))

    def test_the_maximum_reports_the_shift(self):
        maximum, _, _, _ = shifted_normalization([-5.0, -1.0, -3.0])

        self.assertEqual(maximum, -1.0)

    def test_a_dominant_maximum_keeps_its_complement_precise(self):
        _, _, probabilities, complements = shifted_normalization([0.0, -100.0])

        # ``1 - p`` rounds to zero here; the complement must not.
        self.assertEqual(1.0 - probabilities[0], 0.0)
        self.assertGreater(complements[0], 0.0)
        self.assertAlmostEqual(complements[0] / math.exp(-100.0), 1.0, places=6)

    def test_tied_maxima_share_the_mass(self):
        _, _, probabilities, _ = shifted_normalization([2.0, 2.0])

        self.assertAlmostEqual(probabilities[0], 0.5)
        self.assertAlmostEqual(probabilities[1], 0.5)


class ScaledDeviationsTests(unittest.TestCase):
    """Deviations are centered and scaled so squaring cannot overflow."""

    def test_extreme_magnitudes_produce_a_finite_deviation(self):
        values = [1e200, -1e200, 1e200, -1e200]

        scale, centered, deviation = scaled_deviations(values, range(len(values)))

        self.assertTrue(math.isfinite(deviation))
        self.assertTrue(all(math.isfinite(item) for item in centered))
        self.assertEqual(scale * deviation, 1e200)

    def test_identical_values_have_no_deviation(self):
        scale, centered, deviation = scaled_deviations([3.0] * 5, range(5))

        self.assertEqual(scale, 0.0)
        self.assertEqual(centered, [0.0] * 5)
        self.assertEqual(deviation, 0.0)

    def test_a_group_selects_only_its_own_indices(self):
        values = [0.0, 1.0, 100.0, 3.0, 200.0]

        scale, _, deviation = scaled_deviations(values, [1, 3])

        # Only 1.0 and 3.0 participate: mean 2.0, population deviation 1.0.
        self.assertAlmostEqual(scale * deviation, 1.0)

    def test_a_non_finite_value_makes_every_result_not_a_number(self):
        scale, centered, deviation = scaled_deviations([1.0, math.inf], range(2))

        self.assertTrue(math.isnan(scale))
        self.assertTrue(math.isnan(deviation))
        self.assertTrue(all(math.isnan(item) for item in centered))

    def test_an_ordinary_group_matches_the_textbook_deviation(self):
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]

        scale, _, deviation = scaled_deviations(values, range(len(values)))

        self.assertAlmostEqual(scale * deviation, 2.0)


class ReductionGroupingTests(unittest.TestCase):
    """Axis metadata is resolved from a shape, with no values involved."""

    def test_axes_are_normalized_and_sorted(self):
        self.assertEqual(normalize_axes(3, None), (0, 1, 2))
        self.assertEqual(normalize_axes(3, -1), (2,))
        self.assertEqual(normalize_axes(3, [2, 0]), (0, 2))

    def test_invalid_axes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            normalize_axes(2, 5)
        with self.assertRaisesRegex(ValueError, "Duplicate axis"):
            normalize_axes(3, (1, 1))
        with self.assertRaisesRegex(TypeError, "axis must be"):
            normalize_axes(3, True)

    def test_output_shapes_follow_keepdims(self):
        self.assertEqual(reduction_shape((2, 3, 4), (1,), False), (2, 4))
        self.assertEqual(reduction_shape((2, 3, 4), (1,), True), (2, 1, 4))
        self.assertEqual(keepdims_shape((2, 3, 4), 1), (2, 1, 4))
        self.assertEqual(reduction_size((2, 3, 4), (0, 2)), 8)

    def test_groups_partition_every_input_index_exactly_once(self):
        _, output_shape, groups = reduction_groups((2, 3), 1, False)

        self.assertEqual(output_shape, (2,))
        self.assertEqual(groups, [[0, 1, 2], [3, 4, 5]])

    def test_grouping_over_all_axes_collects_one_group(self):
        axes, output_shape, groups = reduction_groups((2, 2), None, False)

        self.assertEqual(axes, (0, 1))
        self.assertEqual(output_shape, ())
        self.assertEqual(groups, [[0, 1, 2, 3]])

    def test_scalar_output_can_be_requested_as_a_vector(self):
        _, output_shape, groups = reduction_groups(
            (2, 2), None, False, scalar_as_vector=True
        )

        self.assertEqual(output_shape, (1,))
        self.assertEqual(groups, [[0, 1, 2, 3]])

    def test_an_empty_axis_produces_empty_groups(self):
        _, output_shape, groups = reduction_groups((0, 3), 0, False)

        self.assertEqual(output_shape, (3,))
        self.assertEqual(groups, [[], [], []])

    def test_a_reduction_over_an_empty_result_has_no_groups(self):
        _, output_shape, groups = reduction_groups((3, 0), 0, False)

        self.assertEqual(output_shape, (0,))
        self.assertEqual(groups, [])

    def test_grouping_reads_a_shape_rather_than_a_tensor(self):
        """A plain tuple is enough; no value sequence is required."""
        _, output_shape, groups = reduction_groups((4,), 0, True)

        self.assertEqual(output_shape, (1,))
        self.assertEqual(groups, [[0, 1, 2, 3]])


if __name__ == "__main__":
    unittest.main()
