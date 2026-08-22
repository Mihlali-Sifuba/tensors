import unittest

import tensors as ts
from tensors.utils.broadcasting import (
    broadcast_shape,
    broadcast_tensors,
    broadcast_to,
)


class BroadcastingUtilityTests(unittest.TestCase):
    def test_broadcast_shape_combines_leading_and_singleton_dimensions(self):
        self.assertEqual(broadcast_shape((3, 1), (1, 4)), (3, 4))
        self.assertEqual(broadcast_shape((5, 1, 4), (3, 4)), (5, 3, 4))

    def test_scalar_shape_broadcasts_to_any_valid_shape(self):
        self.assertEqual(broadcast_shape((), (2, 3)), (2, 3))

    def test_broadcast_shape_rejects_incompatible_dimensions(self):
        with self.assertRaisesRegex(ValueError, "cannot be broadcast"):
            broadcast_shape((2, 3), (2, 4))

    def test_broadcast_shape_validates_dimensions(self):
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            broadcast_shape((2, -1), (1, 1))

    def test_broadcast_to_returns_the_original_tensor_for_the_same_shape(self):
        tensor = ts.Tensor([[1.0, 2.0]])

        self.assertIs(broadcast_to(tensor, tensor.shape), tensor)

    def test_broadcast_to_repeats_a_vector_across_a_leading_dimension(self):
        tensor = ts.Tensor([1.0, 2.0, 3.0], dtype=ts.float32)

        result = broadcast_to(tensor, (2, 3))

        self.assertEqual(result.shape, (2, 3))
        self.assertIs(result.dtype, ts.float32)
        self.assertEqual(result.tolist(), [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])

    def test_broadcast_to_repeats_singleton_dimensions(self):
        tensor = ts.Tensor([[1], [2]], dtype=ts.int32)

        result = broadcast_to(tensor, (2, 3))

        self.assertEqual(result.tolist(), [1, 1, 1, 2, 2, 2])
        self.assertIs(result.dtype, ts.int32)

    def test_broadcast_to_materializes_a_scalar_shape(self):
        tensor = ts.Tensor([7.0], shape=())

        result = broadcast_to(tensor, (2, 2))

        self.assertEqual(result.tolist(), [7.0, 7.0, 7.0, 7.0])

    def test_broadcast_to_handles_an_empty_target_dimension(self):
        tensor = ts.Tensor([], shape=(1, 0))

        result = broadcast_to(tensor, (3, 0))

        self.assertEqual(result.shape, (3, 0))
        self.assertEqual(result.tolist(), [])

    def test_broadcast_to_rejects_rank_reduction(self):
        with self.assertRaisesRegex(ValueError, "cannot be broadcast"):
            broadcast_to(ts.Tensor([[1.0]]), (1,))

    def test_broadcast_to_rejects_an_incompatible_target(self):
        with self.assertRaisesRegex(ValueError, "cannot be broadcast"):
            broadcast_to(ts.Tensor([1.0, 2.0]), (3,))

    def test_broadcast_tensors_returns_the_shared_shape(self):
        left, right = broadcast_tensors(
            ts.Tensor([[1.0], [2.0]]),
            ts.Tensor([[10.0, 20.0, 30.0]]),
        )

        self.assertEqual(left.shape, (2, 3))
        self.assertEqual(right.shape, (2, 3))
        self.assertEqual(left.tolist(), [1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        self.assertEqual(right.tolist(), [10.0, 20.0, 30.0] * 2)


if __name__ == "__main__":
    unittest.main()
