"""``broadcast_to`` as a transformation in its own right.

Broadcasting used to be entangled with computation: ``broadcast_binary_values``
took two tensors and an operation, walked the broadcast offsets and applied
the operation as it went. Splitting the two means ``broadcast_to`` can be
tested on its own terms — what shape it produces, what values, and what it
leaves untouched — without an arithmetic result standing in for the answer.

These tests exercise the utility directly. The operations that consume it keep
their own tests; nothing here asserts an arithmetic, comparison or selection
result.
"""

from __future__ import annotations

import unittest

import tensors as ts
from tensors.shape import Shape
from tensors.utils.broadcasting import broadcast_tensors, broadcast_to


def values_of(tensor):
    return list(tensor._data)


class TheOutputShape(unittest.TestCase):
    """What shape comes back, for every way a source can expand."""

    def test_an_equal_shape_returns_the_source_unchanged(self):
        """The shortcut is observable: the very same object comes back."""
        source = ts.Tensor([1.0, 2.0, 3.0])
        produced = broadcast_to(source, Shape(3))
        self.assertIs(produced, source)

    def test_an_equal_shape_given_as_a_tuple(self):
        source = ts.Tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertIs(broadcast_to(source, (2, 2)), source)

    def test_a_singleton_dimension_expands(self):
        produced = broadcast_to(ts.Tensor([[1.0], [2.0]]), (2, 3))
        self.assertEqual(produced.shape, (2, 3))

    def test_a_leading_dimension_is_added(self):
        produced = broadcast_to(ts.Tensor([1.0, 2.0, 3.0]), (4, 3))
        self.assertEqual(produced.shape, (4, 3))

    def test_several_ranks_are_added_at_once(self):
        produced = broadcast_to(ts.Tensor([1.0, 2.0]), (2, 3, 4, 2))
        self.assertEqual(produced.shape, (2, 3, 4, 2))

    def test_a_scalar_tensor_expands_to_any_shape(self):
        for shape in ((3,), (2, 3), (2, 3, 4), ()):
            with self.subTest(shape=shape):
                produced = broadcast_to(ts.Tensor(5.0), shape)
                self.assertEqual(produced.shape, shape)

    def test_a_one_element_tensor_of_higher_rank_expands(self):
        produced = broadcast_to(ts.Tensor([[[7.0]]]), (2, 3, 4))
        self.assertEqual(produced.shape, (2, 3, 4))
        self.assertEqual(values_of(produced), [7.0] * 24)


class TheOutputValues(unittest.TestCase):
    """Which source element each output position takes."""

    def test_a_row_repeats_down_the_added_dimension(self):
        produced = broadcast_to(ts.Tensor([1.0, 2.0, 3.0]), (2, 3))
        self.assertEqual(values_of(produced), [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])

    def test_a_column_repeats_across_a_singleton(self):
        produced = broadcast_to(ts.Tensor([[1.0], [2.0]]), (2, 3))
        self.assertEqual(values_of(produced), [1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

    def test_two_singletons_expand_independently(self):
        produced = broadcast_to(ts.Tensor([[[1.0], [2.0]]]), (2, 2, 3))
        self.assertEqual(
            values_of(produced),
            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        )

    def test_a_scalar_fills_every_position(self):
        produced = broadcast_to(ts.Tensor(2.5), (2, 3))
        self.assertEqual(values_of(produced), [2.5] * 6)

    def test_the_values_match_a_hand_written_index_walk(self):
        """An independent derivation of which element lands where."""
        source = ts.Tensor([[1.0], [2.0], [3.0]])
        target = (3, 4)
        produced = broadcast_to(source, target)

        expected = []
        for row in range(target[0]):
            for column in range(target[1]):
                # The second axis is a singleton in the source, so every
                # column of a row takes the same element.
                expected.append(source._data[row])
        self.assertEqual(values_of(produced), expected)


class RankAndCompatibility(unittest.TestCase):
    """What is refused, and with what."""

    def test_rank_reduction_is_refused(self):
        with self.assertRaises(ValueError):
            broadcast_to(ts.Tensor([[1.0, 2.0], [3.0, 4.0]]), (2,))

    def test_an_incompatible_dimension_is_refused(self):
        with self.assertRaises(ValueError):
            broadcast_to(ts.Tensor([1.0, 2.0, 3.0]), (2, 4))

    def test_a_non_singleton_cannot_shrink(self):
        with self.assertRaises(ValueError):
            broadcast_to(ts.Tensor([[1.0, 2.0, 3.0]]), (1, 2))

    def test_the_error_names_both_shapes(self):
        with self.assertRaises(ValueError) as raised:
            broadcast_to(ts.Tensor([1.0, 2.0, 3.0]), (2, 4))
        message = str(raised.exception)
        self.assertIn("(3,)", message)
        self.assertIn("(2, 4)", message)


class EmptyDimensions(unittest.TestCase):
    """A zero extent is a legitimate shape, not an error."""

    def test_an_empty_target_dimension(self):
        produced = broadcast_to(ts.Tensor([[1.0]]), (0, 1))
        self.assertEqual(produced.shape, (0, 1))
        self.assertEqual(values_of(produced), [])

    def test_an_empty_source_stays_empty(self):
        source = ts.Tensor([], dtype=ts.float64, shape=(0,))
        produced = broadcast_to(source, (0,))
        self.assertEqual(produced.shape, (0,))
        self.assertEqual(values_of(produced), [])

    def test_an_empty_dimension_beside_an_expanding_one(self):
        produced = broadcast_to(ts.Tensor([], dtype=ts.float64, shape=(1, 0)), (3, 0))
        self.assertEqual(produced.shape, (3, 0))
        self.assertEqual(values_of(produced), [])


class DtypeIsPreserved(unittest.TestCase):
    """Broadcasting moves values; it never resolves a dtype."""

    def test_every_public_dtype_survives(self):
        for name in ("uint8", "int8", "int16", "int32", "int64", "float32", "float64"):
            with self.subTest(dtype=name):
                dtype = getattr(ts, name)
                source = ts.Tensor([1, 2], dtype=dtype)
                produced = broadcast_to(source, (3, 2))
                self.assertIs(produced.dtype, dtype)

    def test_the_equal_shape_shortcut_preserves_the_dtype(self):
        source = ts.Tensor([1, 2], dtype=ts.int16)
        self.assertIs(broadcast_to(source, (2,)).dtype, ts.int16)

    def test_float32_values_are_not_widened(self):
        """A binary32 value must come back as itself, not as a binary64 one."""
        subnormal = 1.401298464324817e-45
        source = ts.Tensor([subnormal], dtype=ts.float32)
        produced = broadcast_to(source, (4,))
        self.assertIs(produced.dtype, ts.float32)
        self.assertEqual(values_of(produced), [subnormal] * 4)


class LayoutsAndOffsets(unittest.TestCase):
    """Logical values are the source of truth, not the physical buffer."""

    def test_a_strided_view_broadcasts_by_its_logical_values(self):
        source = ts.Tensor([float(value) for value in range(8)])[::2]
        self.assertEqual(values_of(source), [0.0, 2.0, 4.0, 6.0])

        produced = broadcast_to(source, (2, 4))
        self.assertEqual(produced.shape, (2, 4))
        self.assertEqual(values_of(produced), [0.0, 2.0, 4.0, 6.0, 0.0, 2.0, 4.0, 6.0])

    def test_a_view_with_a_nonzero_offset(self):
        source = ts.Tensor([float(value) for value in range(8)])[3:6]
        self.assertEqual(values_of(source), [3.0, 4.0, 5.0])

        produced = broadcast_to(source, (2, 3))
        self.assertEqual(values_of(produced), [3.0, 4.0, 5.0, 3.0, 4.0, 5.0])

    def test_a_transposed_tensor_broadcasts_by_its_logical_order(self):
        source = ts.transpose(ts.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        self.assertEqual(source.shape, (3, 2))
        self.assertEqual(values_of(source), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])

        produced = broadcast_to(source, (2, 3, 2))
        self.assertEqual(values_of(produced), values_of(source) * 2)

    def test_a_strided_singleton_expands(self):
        source = ts.Tensor([float(value) for value in range(8)])[::8]
        self.assertEqual(values_of(source), [0.0])
        produced = broadcast_to(source, (3,))
        self.assertEqual(values_of(produced), [0.0, 0.0, 0.0])


class TheSourceIsNotMutated(unittest.TestCase):
    """The invariant every caller relies on."""

    def test_the_source_keeps_its_shape_values_and_dtype(self):
        source = ts.Tensor([[1.0], [2.0]], dtype=ts.float32)
        before_shape = source.shape
        before_values = values_of(source)
        before_dtype = source.dtype
        before_version = source.version

        broadcast_to(source, (2, 5))

        self.assertEqual(source.shape, before_shape)
        self.assertEqual(values_of(source), before_values)
        self.assertIs(source.dtype, before_dtype)
        self.assertEqual(source.version, before_version)

    def test_writing_to_the_result_does_not_reach_the_source(self):
        """No aliasing is introduced: the expansion materialises its values."""
        source = ts.Tensor([1.0, 2.0, 3.0])
        produced = broadcast_to(source, (2, 3))
        self.assertIsNot(produced, source)

        produced[0, 0] = 99.0
        self.assertEqual(values_of(source), [1.0, 2.0, 3.0])

    def test_the_equal_shape_shortcut_returns_the_source_itself(self):
        """Documented behaviour, and the one case that does alias.

        ``broadcast_to`` returns the source unchanged when the shape already
        matches, so the result *is* the input. Callers that mutate a
        broadcast result must not rely on it being a copy; none in the
        package does.
        """
        source = ts.Tensor([1.0, 2.0, 3.0])
        self.assertIs(broadcast_to(source, (3,)), source)


class BroadcastTensorsPairs(unittest.TestCase):
    """The pair form, which agrees the shape and then expands both."""

    def test_both_operands_reach_the_common_shape(self):
        left, right = broadcast_tensors(
            ts.Tensor([[1.0], [2.0]]), ts.Tensor([10.0, 20.0, 30.0])
        )
        self.assertEqual(left.shape, (2, 3))
        self.assertEqual(right.shape, (2, 3))
        self.assertEqual(values_of(left), [1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        self.assertEqual(values_of(right), [10.0, 20.0, 30.0, 10.0, 20.0, 30.0])

    def test_the_shape_agrees_with_broadcast_with(self):
        """The pair form adds no rule of its own."""
        a = ts.Tensor([[1.0], [2.0]])
        b = ts.Tensor([10.0, 20.0, 30.0])
        left, right = broadcast_tensors(a, b)
        self.assertEqual(left.shape, a.shape.broadcast_with(b.shape))
        self.assertEqual(right.shape, a.shape.broadcast_with(b.shape))

    def test_incompatible_operands_are_refused(self):
        with self.assertRaises(ValueError):
            broadcast_tensors(ts.Tensor([1.0, 2.0]), ts.Tensor([1.0, 2.0, 3.0]))

    def test_dtypes_are_left_alone(self):
        """The pair form resolves no result dtype; each keeps its own."""
        left, right = broadcast_tensors(
            ts.Tensor([1.0], dtype=ts.float32), ts.Tensor([1, 2], dtype=ts.int32)
        )
        self.assertIs(left.dtype, ts.float32)
        self.assertIs(right.dtype, ts.int32)


class BroadcastingDoesNotComputeOrDispatch(unittest.TestCase):
    """The separation this refactor establishes, asserted directly."""

    def test_the_module_exports_only_broadcasting(self):
        """Two transformations and the mapping they share, and nothing else.

        ``broadcast_source_indices`` states which source position each
        result position reads. That is broadcasting itself rather than an
        addition to it: the transformations apply the mapping to values,
        and a caller holding values of its own applies it where they are.
        Neither form computes or dispatches, which is what this class is
        guarding.
        """
        from tensors.utils import broadcasting

        self.assertEqual(
            sorted(broadcasting.__all__),
            ["broadcast_source_indices", "broadcast_tensors", "broadcast_to"],
        )

    def test_the_shared_mapping_reads_no_values_and_selects_no_backend(self):
        """It takes two shapes, so there is nothing for it to compute with."""
        import inspect

        from tensors.utils.broadcasting import broadcast_source_indices

        parameters = list(inspect.signature(broadcast_source_indices).parameters)
        self.assertEqual(parameters, ["source_shape", "target_shape"])
        self.assertEqual(broadcast_source_indices((2, 1), (2, 3)), [0, 0, 0, 1, 1, 1])
        self.assertEqual(broadcast_source_indices((3,), (2, 3)), [0, 1, 2, 0, 1, 2])
        with self.assertRaises(ValueError):
            broadcast_source_indices((2, 3), (3,))

    def test_the_combined_helper_is_gone(self):
        """``broadcast_binary_values`` mixed broadcasting with computation."""
        from tensors.utils import broadcasting

        self.assertFalse(hasattr(broadcasting, "broadcast_binary_values"))

    def test_broadcast_to_takes_no_operation(self):
        """Its signature admits a tensor and a shape, and nothing else."""
        import inspect

        parameters = list(inspect.signature(broadcast_to).parameters)
        self.assertEqual(parameters, ["tensor", "shape"])

    def test_broadcasting_runs_under_every_backend_selection(self):
        """It selects no backend: the result is the same wherever it runs."""
        expected = None
        for backend in ts.available_backends():
            with self.subTest(backend=backend), ts.use_backend(backend):
                produced = broadcast_to(ts.Tensor([[1.0], [2.0]]), (2, 3))
                values = values_of(produced)
                if expected is None:
                    expected = values
                self.assertEqual(values, expected)
                self.assertEqual(produced.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
