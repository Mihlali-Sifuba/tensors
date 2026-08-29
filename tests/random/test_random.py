import random as python_random
import unittest

import tensors as ts
from tensors.storage import CudaStorage, NumPyStorage, PythonStorage


class RandomTests(unittest.TestCase):
    def setUp(self):
        self.previous_backend = ts.get_backend()
        ts.set_backend("python")

    def tearDown(self):
        ts.random.seed(None)
        ts.set_backend(self.previous_backend)

    def test_generators_preserve_shape_dtype_and_ranges(self):
        uniform = ts.random.uniform((2, 3), -2.0, 4.0, dtype=ts.float32)
        normal = ts.random.normal((3, 2), 1.5, 0.25)
        integers = ts.random.randint((100,), -3, 5, dtype=ts.int16)

        self.assertEqual(uniform.shape, (2, 3))
        self.assertIs(uniform.dtype, ts.float32)
        self.assertTrue(all(-2.0 <= value < 4.0 for value in uniform.tolist()))
        self.assertEqual(normal.shape, (3, 2))
        self.assertIs(normal.dtype, ts.float64)
        self.assertIs(integers.dtype, ts.int16)
        self.assertTrue(all(-3 <= value < 5 for value in integers.tolist()))

    def test_randint_supports_single_bound(self):
        ts.random.seed(11)
        values = ts.random.randint((100,), 4)

        self.assertTrue(all(0 <= value < 4 for value in values.tolist()))

    def test_random_constructors_support_scalar_and_empty_shapes(self):
        scalar = ts.random.uniform(())
        empty = ts.random.normal((2, 0, 3))
        integers = ts.random.randint((0,), 5)

        self.assertEqual(scalar.shape, ())
        self.assertEqual(scalar.size, 1)
        self.assertEqual(empty.shape, (2, 0, 3))
        self.assertEqual(empty.tolist(), [])
        self.assertEqual(integers.tolist(), [])

    def test_zero_normal_deviation_returns_the_mean(self):
        value = ts.random.normal((8,), mean=3.5, stddev=0.0)

        self.assertEqual(value.tolist(), [3.5] * 8)

    def test_seed_reproduces_a_sequence_and_state_advances(self):
        ts.random.seed(42)
        first = ts.random.uniform((16,)).tolist()
        second = ts.random.uniform((16,)).tolist()
        self.assertNotEqual(first, second)

        ts.random.seed(42)
        self.assertEqual(ts.random.uniform((16,)).tolist(), first)
        self.assertEqual(ts.random.uniform((16,)).tolist(), second)

    def test_different_seeds_produce_different_sequences(self):
        ts.random.seed(1)
        first = ts.random.normal((16,)).tolist()
        ts.random.seed(2)
        second = ts.random.normal((16,)).tolist()

        self.assertNotEqual(first, second)

    def test_seed_does_not_modify_python_global_random_state(self):
        python_random.seed(8675309)
        before = python_random.getstate()
        ts.random.seed(23)
        ts.random.uniform((32,))
        ts.random.normal((32,))
        ts.random.randint((32,), 10)

        self.assertEqual(python_random.getstate(), before)

    def test_python_backend_uses_python_native_storage(self):
        value = ts.random.normal((64,))

        self.assertIsInstance(value._storage, PythonStorage)

    def test_argument_validation_is_explicit(self):
        invalid_calls = (
            lambda: ts.random.seed(True),
            lambda: ts.random.seed(-1),
            lambda: ts.random.uniform((2,), 1.0, 1.0),
            lambda: ts.random.uniform((2,), dtype=ts.int32),
            lambda: ts.random.normal((2,), stddev=-1.0),
            lambda: ts.random.randint((2,), 3, 3),
            lambda: ts.random.randint((2,), 0, 300, dtype=ts.uint8),
            lambda: ts.random.randint((2,), 3, dtype=ts.float32),
            lambda: ts.random.uniform((-1,)),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, ValueError)):
                    call()


@unittest.skipUnless(
    "numpy" in ts.available_backends(),
    "NumPy is not installed",
)
class NumPyRandomTests(unittest.TestCase):
    def tearDown(self):
        ts.random.seed(None)

    def test_numpy_generation_is_native_and_reproducible(self):
        with ts.use_backend("numpy"):
            ts.random.seed(101)
            first = ts.random.uniform((128,), dtype=ts.float32)
            ts.random.seed(101)
            second = ts.random.uniform((128,), dtype=ts.float32)
            integers = ts.random.randint((128,), -4, 7)

        self.assertIsInstance(first._storage, NumPyStorage)
        self.assertIsInstance(integers._storage, NumPyStorage)
        self.assertEqual(first.tolist(), second.tolist())


@unittest.skipUnless(
    "cuda" in ts.available_backends(),
    "CUDA is not available",
)
class CudaRandomTests(unittest.TestCase):
    def tearDown(self):
        ts.random.seed(None)

    def test_cuda_generation_is_device_native_and_reproducible(self):
        with ts.use_backend("cuda"):
            ts.random.seed(101)
            first = ts.random.normal((128,), dtype=ts.float32)
            ts.random.seed(101)
            second = ts.random.normal((128,), dtype=ts.float32)
            integers = ts.random.randint((128,), -4, 7)

        self.assertIsInstance(first._storage, CudaStorage)
        self.assertIsInstance(integers._storage, CudaStorage)
        self.assertEqual(first.tolist(), second.tolist())


if __name__ == "__main__":
    unittest.main()
