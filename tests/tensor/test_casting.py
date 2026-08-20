import unittest

import tensors as ts
from tensors.casting import cast_values


class TensorCastingTests(unittest.TestCase):
    def test_cast_values_converts_floating_values_to_integers(self):
        values = cast_values(
            [1.9, -2.1],
            source_dtype=ts.float64,
            target_dtype=ts.int32,
        )

        self.assertEqual(values, [1, -2])

    def test_cast_values_converts_integer_values_to_floats(self):
        values = cast_values(
            [1, 2],
            source_dtype=ts.int32,
            target_dtype=ts.float32,
        )

        self.assertEqual(values, [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
