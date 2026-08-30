import math
import unittest

import tensors as ts
from tensors.storage import CudaStorage, NumPyStorage


def _variance(values):
    mean = math.fsum(values) / len(values)
    return math.fsum((value - mean) ** 2 for value in values) / len(values)


@unittest.skipUnless(
    "numpy" in ts.available_backends(),
    "NumPy is not installed",
)
class NumPyInitializerTests(unittest.TestCase):
    def tearDown(self):
        ts.random.seed(None)

    def test_numpy_initializers_retain_native_storage(self):
        with ts.use_backend("numpy"):
            normal = ts.init.he_normal((128, 64))
            truncated = ts.init.truncated_normal((128, 64))
            orthogonal = ts.init.orthogonal((128, 64))

        self.assertIsInstance(normal._storage, NumPyStorage)
        self.assertIsInstance(truncated._storage, NumPyStorage)
        self.assertIsInstance(orthogonal._storage, NumPyStorage)

    def test_numpy_statistics_match_reference_semantics(self):
        with ts.use_backend("numpy"):
            ts.random.seed(212)
            values = ts.init.he_normal((512, 256)).tolist()

        expected = 2.0 / 512
        self.assertAlmostEqual(
            _variance(values),
            expected,
            delta=expected * 0.06,
        )


@unittest.skipUnless(
    "cuda" in ts.available_backends(),
    "CUDA is not available",
)
class CudaInitializerTests(unittest.TestCase):
    def tearDown(self):
        ts.random.seed(None)

    def test_cuda_initializers_remain_device_resident(self):
        with ts.use_backend("cuda"):
            normal = ts.init.he_normal((128, 64))
            truncated = ts.init.truncated_normal((128, 64))
            orthogonal = ts.init.orthogonal((128, 64))

        self.assertIsInstance(normal._storage, CudaStorage)
        self.assertIsInstance(truncated._storage, CudaStorage)
        self.assertIsInstance(orthogonal._storage, CudaStorage)

    def test_cuda_statistics_match_reference_semantics(self):
        with ts.use_backend("cuda"):
            ts.random.seed(212)
            values = ts.init.he_normal((512, 256)).tolist()

        expected = 2.0 / 512
        self.assertAlmostEqual(
            _variance(values),
            expected,
            delta=expected * 0.06,
        )


if __name__ == "__main__":
    unittest.main()
