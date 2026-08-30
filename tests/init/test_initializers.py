import math
import unittest

import tensors as ts
from tensors.init._utils import calculate_fan_in_and_fan_out


def _mean(values):
    return math.fsum(values) / len(values)


def _variance(values):
    average = _mean(values)
    return math.fsum((value - average) ** 2 for value in values) / len(values)


class InitializerTests(unittest.TestCase):
    def setUp(self):
        self.previous_backend = ts.get_backend()
        ts.set_backend("python")
        ts.random.seed(1234)

    def tearDown(self):
        ts.random.seed(None)
        ts.set_backend(self.previous_backend)

    def test_public_facade_contains_initializers_without_root_aliases(self):
        names = (
            "variance_scaling",
            "xavier_uniform",
            "xavier_normal",
            "he_uniform",
            "he_normal",
            "lecun_uniform",
            "lecun_normal",
            "truncated_normal",
            "orthogonal",
        )
        for name in names:
            self.assertTrue(callable(getattr(ts.init, name)))
            self.assertFalse(hasattr(ts, name))

    def test_fans_use_matrix_and_channel_last_kernel_conventions(self):
        self.assertEqual(calculate_fan_in_and_fan_out((128, 64)), (128, 64))
        self.assertEqual(
            calculate_fan_in_and_fan_out((3, 5, 16, 32)),
            (240, 480),
        )

    def test_fan_calculation_rejects_ambiguous_and_zero_shapes(self):
        for shape in ((), (3,), (2, 0), (0, 2, 3)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "fan_in"):
                    calculate_fan_in_and_fan_out(shape)

    def test_all_initializers_preserve_shape_and_dtype(self):
        initializers = (
            ts.init.xavier_uniform,
            ts.init.xavier_normal,
            ts.init.he_uniform,
            ts.init.he_normal,
            ts.init.lecun_uniform,
            ts.init.lecun_normal,
            ts.init.truncated_normal,
            ts.init.orthogonal,
        )
        for initializer in initializers:
            with self.subTest(initializer=initializer.__name__):
                value = initializer((8, 4), dtype=ts.float32)
                self.assertEqual(value.shape, (8, 4))
                self.assertIs(value.dtype, ts.float32)

    def test_uniform_initializer_bounds_follow_definitions(self):
        shape = (128, 64)
        cases = (
            (ts.init.xavier_uniform, math.sqrt(6.0 / (128 + 64))),
            (ts.init.he_uniform, math.sqrt(6.0 / 128)),
            (ts.init.lecun_uniform, math.sqrt(3.0 / 128)),
        )
        for initializer, bound in cases:
            with self.subTest(initializer=initializer.__name__):
                values = initializer(shape).tolist()
                self.assertTrue(all(-bound <= value < bound for value in values))

    def test_normal_initializer_variances_follow_definitions(self):
        shape = (256, 512)
        cases = (
            (ts.init.xavier_normal, 2.0 / (256 + 512)),
            (ts.init.he_normal, 2.0 / 256),
            (ts.init.lecun_normal, 1.0 / 256),
        )
        for initializer, expected in cases:
            with self.subTest(initializer=initializer.__name__):
                values = initializer(shape).tolist()
                self.assertAlmostEqual(
                    _variance(values),
                    expected,
                    delta=expected * 0.06,
                )

    def test_variance_scaling_supports_all_modes_and_distributions(self):
        for mode in ("fan_in", "fan_out", "fan_avg"):
            for distribution in ("uniform", "normal", "truncated_normal"):
                with self.subTest(mode=mode, distribution=distribution):
                    value = ts.init.variance_scaling(
                        (64, 32),
                        scale=1.5,
                        mode=mode,
                        distribution=distribution,
                    )
                    self.assertEqual(value.shape, (64, 32))

    def test_truncated_normal_obeys_bounds_and_has_expected_shape(self):
        value = ts.init.truncated_normal(
            (100_000,),
            mean=1.0,
            stddev=2.0,
            lower=-1.0,
            upper=3.0,
        )
        values = value.tolist()

        self.assertTrue(all(-1.0 <= item <= 3.0 for item in values))
        self.assertAlmostEqual(_mean(values), 1.0, delta=0.03)
        self.assertGreater(_variance(values), 1.0)
        self.assertLess(_variance(values), 2.0)

    def test_truncated_normal_supports_empty_tensors(self):
        value = ts.init.truncated_normal((2, 0, 3))

        self.assertEqual(value.shape, (2, 0, 3))
        self.assertEqual(value.tolist(), [])

    def test_orthogonal_tall_and_wide_matrices(self):
        for shape in ((8, 3), (3, 8), (2, 3, 4)):
            with self.subTest(shape=shape):
                value = ts.init.orthogonal(shape, gain=1.5)
                rows = shape[0]
                columns = math.prod(shape[1:])
                matrix = [
                    value.tolist()[row * columns:(row + 1) * columns]
                    for row in range(rows)
                ]
                if rows <= columns:
                    vectors = matrix
                else:
                    vectors = [
                        [matrix[row][column] for row in range(rows)]
                        for column in range(columns)
                    ]
                for left, first in enumerate(vectors):
                    for right, second in enumerate(vectors):
                        dot = math.fsum(a * b for a, b in zip(first, second))
                        expected = 2.25 if left == right else 0.0
                        self.assertAlmostEqual(dot, expected, delta=1e-10)

    def test_initializer_seed_is_reproducible_and_advances(self):
        ts.random.seed(19)
        first = ts.init.he_normal((32, 16)).tolist()
        second = ts.init.he_normal((32, 16)).tolist()
        ts.random.seed(19)

        self.assertNotEqual(first, second)
        self.assertEqual(ts.init.he_normal((32, 16)).tolist(), first)

    def test_invalid_arguments_are_rejected(self):
        invalid_calls = (
            lambda: ts.init.variance_scaling((2, 3), scale=0.0),
            lambda: ts.init.variance_scaling((2, 3), mode="missing"),
            lambda: ts.init.variance_scaling((2, 3), distribution="missing"),
            lambda: ts.init.he_normal((3,)),
            lambda: ts.init.he_normal((2, 0)),
            lambda: ts.init.he_normal((2, 3), dtype=ts.int32),
            lambda: ts.init.truncated_normal((2,), stddev=0.0),
            lambda: ts.init.truncated_normal((2,), lower=1.0, upper=0.0),
            lambda: ts.init.truncated_normal(
                (2,), lower=100.0, upper=101.0
            ),
            lambda: ts.init.orthogonal((4,)),
            lambda: ts.init.orthogonal((2, 0)),
            lambda: ts.init.orthogonal((2, 2), gain=float("inf")),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, ValueError)):
                    call()


if __name__ == "__main__":
    unittest.main()
