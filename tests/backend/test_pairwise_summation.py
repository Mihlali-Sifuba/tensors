"""The deterministic pairwise summation contract, on every backend.

docs/summation-semantics.md specifies floating summation as a tree: a group in
logical order, adjacent pairs added with one rounding to the declared dtype, an
odd final value carried unchanged, repeated until one value remains, then the
non-finite classification and the canonical zero. Contractions form each
product in the dtype and reduce the products with the same tree.

Every expectation here comes from the independent oracle in
``tests/_pairwise_oracle.py`` or from a derivation written next to it, never
from another backend. Where the tree differs from the exact sum, the test
asserts the tree.
"""

import itertools
import math
import random
import struct
import unittest
from fractions import Fraction
from unittest.mock import patch

import tensors as ts
from tensors.graph.state import reset_graph_state
from tests._pairwise_oracle import (
    error_bound,
    left_to_right_sum,
    pairwise_dot,
    pairwise_sum,
    round32,
)

FLOATS = (ts.float32, ts.float64)


def bits(value: float, dtype) -> bytes:
    """The IEEE bit pattern of ``value`` in ``dtype``; NaN compares as one."""
    if math.isnan(value):
        return b"nan"
    return struct.pack("<f" if dtype is ts.float32 else "<d", value)


def exact(values) -> Fraction:
    return sum((Fraction(value) for value in values), Fraction(0))


class PairwiseTestCase(unittest.TestCase):
    def setUp(self):
        self.previous_backend = ts.get_backend()
        reset_graph_state()

    def tearDown(self):
        ts.set_backend(self.previous_backend)
        reset_graph_state()

    def assertTreeValue(self, produced: float, expected: float, dtype) -> None:
        self.assertEqual(bits(produced, dtype), bits(expected, dtype))

    def each_backend(self):
        for backend in ts.available_backends():
            with self.subTest(backend=backend):
                reset_graph_state()
                with ts.use_backend(backend):
                    yield backend


class SumTests(PairwiseTestCase):
    """``ts.sum`` of one group, against the oracle."""

    def check(self, values, dtype=ts.float64):
        expected = pairwise_sum(values, dtype.name)
        for backend in self.each_backend():
            result = ts.sum(ts.Tensor(values, dtype=dtype))
            self.assertEqual(result.backend_storage.kind, backend)
            self.assertIs(result.dtype, dtype)
            self.assertEqual(tuple(result.shape), (1,))
            self.assertTreeValue(result.item(), expected, dtype)
        return expected

    def test_an_empty_group_is_positive_zero(self):
        for backend in self.each_backend():
            result = ts.sum(ts.Tensor([])).item()
            self.assertEqual(result, 0.0)
            self.assertEqual(math.copysign(1.0, result), 1.0)

    def test_one_element_is_preserved(self):
        for value in (0.1, -3.5, 1e-310, math.inf):
            self.assertEqual(self.check([value]), value)

    def test_two_elements_are_one_rounded_addition(self):
        self.assertEqual(self.check([0.1, 0.2]), 0.1 + 0.2)

    def test_three_elements_carry_the_odd_value(self):
        """``[0.1, 0.2, 0.3]`` is ``fl(fl(0.1 + 0.2) + 0.3)``, not ``fsum``."""
        expected = self.check([0.1, 0.2, 0.3])
        self.assertEqual(expected, (0.1 + 0.2) + 0.3)
        self.assertEqual(expected, 0.6000000000000001)
        self.assertNotEqual(expected, math.fsum([0.1, 0.2, 0.3]))

    def test_larger_odd_and_even_groups_use_several_levels(self):
        rng = random.Random(1)
        for size in (4, 5, 7, 8, 9, 31, 64, 65, 1000, 1023):
            with self.subTest(size=size):
                self.check([rng.uniform(-1.0, 1.0) for _ in range(size)])

    def test_float32_rounds_after_every_addition(self):
        """Per-addition binary32 rounding differs from a binary64 accumulation.

        This group reduces to ``-1.002000093460083`` when every addition is
        rounded to binary32, as specified, and to ``-1.0019999742507935`` if
        the group were accumulated in binary64 and narrowed once.
        """
        values = [
            -1.0,
            5.960464477539063e-08,
            -5.960464477539063e-08,
            -0.0010000000474974513,
            -0.0010000000474974513,
        ]
        self.assertNotEqual(
            pairwise_sum(values, "float32"), round32(pairwise_sum(values, "float64"))
        )
        self.assertEqual(self.check(values, ts.float32), -1.002000093460083)

    def test_float64_uses_the_same_tree(self):
        rng = random.Random(2)
        self.check([rng.uniform(-1e3, 1e3) for _ in range(37)], ts.float64)

    def test_cancellation_asserts_the_tree_not_the_exact_sum(self):
        """``[-1e16, 1, 1e16, 1, -2]``: tree -2, left-to-right -1, exact 0."""
        values = [-1e16, 1.0, 1e16, 1.0, -2.0]
        self.assertEqual(left_to_right_sum(values), -1.0)
        self.assertEqual(exact(values), 0)
        self.assertEqual(self.check(values), -2.0)

    def test_intermediate_overflow_is_part_of_the_algorithm(self):
        """``+inf`` and ``-inf`` in the first round, NaN in the second."""
        self.assertTrue(math.isnan(self.check([1e308, 1e308, -1e308, -1e308])))

    def test_nonfinite_inputs_are_classified(self):
        cases = (
            ([math.nan, 1.0], math.nan),
            ([math.inf, 1.0, 2.0], math.inf),
            ([-math.inf, 1.0], -math.inf),
            ([math.inf, -math.inf], math.nan),
            ([math.nan, math.inf, -math.inf], math.nan),
            # Finite overflow would give NaN, but +inf with no -inf is +inf.
            ([-1e308, -1e308, math.inf], math.inf),
        )
        for values, expected in cases:
            with self.subTest(values=values):
                produced = self.check(values)
                if math.isnan(expected):
                    self.assertTrue(math.isnan(produced))
                else:
                    self.assertEqual(produced, expected)

    def test_a_zero_result_is_positive_zero(self):
        for values in ([-0.0], [-0.0, -0.0], [1.0, -1.0], [-0.0, -0.0, -0.0]):
            with self.subTest(values=values):
                for backend in self.each_backend():
                    result = ts.sum(ts.Tensor(values)).item()
                    self.assertEqual(result, 0.0)
                    self.assertEqual(math.copysign(1.0, result), 1.0)

    def test_subnormals_underflow_gradually(self):
        tiny64 = 5e-324
        self.assertEqual(self.check([tiny64, tiny64, tiny64]), 3 * tiny64)
        tiny32 = struct.unpack("<f", struct.pack("<I", 1))[0]
        produced = self.check([tiny32, tiny32, tiny32, -tiny32, tiny32], ts.float32)
        self.assertEqual(produced, round32(3 * tiny32))
        self.assertGreater(produced, 0.0)


class AxisTests(PairwiseTestCase):
    """Groups run over the reduced axes, in increasing order, row-major."""

    VALUES = [math.sin(index) * 10.0 ** (index % 7 - 3) for index in range(120)]

    def expected(self, shape, axes, keepdims, dtype):
        rank = len(shape)
        axes = tuple(range(rank)) if axes is None else tuple(sorted(a % rank for a in axes))
        kept = [axis for axis in range(rank) if axis not in axes]
        strides = [math.prod(shape[axis + 1 :]) for axis in range(rank)]
        results = []
        for kept_index in itertools.product(*(range(shape[a]) for a in kept)):
            group = []
            for reduced_index in itertools.product(*(range(shape[a]) for a in axes)):
                coordinate = [0] * rank
                for axis, value in zip(kept, kept_index):
                    coordinate[axis] = value
                for axis, value in zip(axes, reduced_index):
                    coordinate[axis] = value
                offset = sum(c * s for c, s in zip(coordinate, strides))
                group.append(self.VALUES[offset])
            results.append(pairwise_sum(group, dtype.name))
        if keepdims:
            out_shape = tuple(1 if a in axes else shape[a] for a in range(rank))
        else:
            out_shape = tuple(shape[a] for a in kept) or (1,)
        return results, out_shape

    def test_axes_keepdims_and_order(self):
        shape = (2, 3, 4, 5)
        for dtype in FLOATS:
            for axes in ((0,), (2,), (3,), (1, 3), (0, 2), (3, 1), (-1, 0), None):
                for keepdims in (False, True):
                    expected, out_shape = self.expected(shape, axes, keepdims, dtype)
                    for backend in self.each_backend():
                        with self.subTest(dtype=dtype.name, axes=axes, keepdims=keepdims):
                            value = ts.Tensor(self.VALUES, dtype=dtype, shape=shape)
                            result = ts.sum(value, axis=axes, keepdims=keepdims)
                            self.assertEqual(result.backend_storage.kind, backend)
                            self.assertEqual(tuple(result.shape), out_shape)
                            produced = result.tolist()
                            self.assertEqual(
                                [bits(v, dtype) for v in produced],
                                [bits(round32(v) if dtype is ts.float32 else v, dtype)
                                 for v in expected],
                            )

    def test_a_non_contiguous_view_is_reduced_in_logical_order(self):
        """A transposed tensor's logical order, not its storage order, decides."""
        for backend in self.each_backend():
            base = ts.Tensor(self.VALUES[:12], shape=(3, 4))
            transposed = ts.transpose(base)
            result = ts.sum(transposed, axis=1)
            logical = transposed.tolist()
            expected = [pairwise_sum(logical[row * 3 : row * 3 + 3]) for row in range(4)]
            self.assertEqual(result.tolist(), expected)
            self.assertEqual(result.backend_storage.kind, backend)


class ContractionTests(PairwiseTestCase):
    """``dot`` and ``matmul``: products in the dtype, then the tree."""

    def test_contraction_lengths(self):
        rng = random.Random(5)
        for dtype in FLOATS:
            for length in (0, 1, 2, 3, 7, 8, 16, 33):
                left = [rng.uniform(-2.0, 2.0) for _ in range(3 * length)]
                right = [rng.uniform(-2.0, 2.0) for _ in range(length * 4)]
                if dtype is ts.float32:
                    left = [round32(v) for v in left]
                    right = [round32(v) for v in right]
                expected = [
                    pairwise_dot(
                        left[row * length : (row + 1) * length],
                        [right[k * 4 + column] for k in range(length)],
                        dtype.name,
                    )
                    for row in range(3)
                    for column in range(4)
                ]
                for backend in self.each_backend():
                    with self.subTest(dtype=dtype.name, length=length):
                        result = ts.matmul(
                            ts.Tensor(left, dtype=dtype, shape=(3, length)),
                            ts.Tensor(right, dtype=dtype, shape=(length, 4)),
                        )
                        self.assertEqual(result.backend_storage.kind, backend)
                        self.assertIs(result.dtype, dtype)
                        self.assertEqual(
                            [bits(v, dtype) for v in result.tolist()],
                            [bits(v, dtype) for v in expected],
                        )

    def test_dot_distinguishes_the_tree_from_other_orders(self):
        """Products ``[-1e16, 1, 1e16, 1, -2]``: tree -2, sequential -1, exact 0."""
        left = [-1e16, 1.0, 1e16, 1.0, -2.0]
        right = [1.0] * 5
        self.assertEqual(pairwise_dot(left, right), -2.0)
        for backend in self.each_backend():
            result = ts.dot(ts.Tensor(left), ts.Tensor(right))
            self.assertEqual(result.item(), -2.0)
            self.assertEqual(result.backend_storage.kind, backend)

    def test_ordinary_fractional_values(self):
        left = [0.1 * index for index in range(1, 10)]
        right = [math.cos(index) for index in range(9)]
        expected = pairwise_dot(left, right)
        for backend in self.each_backend():
            self.assertEqual(ts.dot(ts.Tensor(left), ts.Tensor(right)).item(), expected)


def conv1d_oracle(inputs, kernel, *, padding=0, stride=1, bias=None, dtype="float64"):
    """The dense-receptive-field definition for one channel, one batch.

    Every kernel tap is a term: an inside tap is ``input * weight`` rounded to
    the dtype, and a tap in the padding is an exact zero.
    """
    length, width = len(inputs), len(kernel)
    outputs = (length + 2 * padding - width) // stride + 1
    result = []
    for position in range(outputs):
        terms = []
        for offset in range(width):
            source = position * stride - padding + offset
            terms.append((inputs[source], kernel[offset]) if 0 <= source < length else (0.0, 0.0))
        total = pairwise_dot([a for a, _ in terms], [b for _, b in terms], dtype)
        if bias is not None:
            total = pairwise_sum([total, bias], dtype)
        result.append(total)
    return result


class ConvolutionTests(PairwiseTestCase):
    def test_whole_number_convolution(self):
        for backend in self.each_backend():
            inputs = ts.Tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
            kernel = ts.Tensor([[[[1.0, 0.0], [0.0, -1.0]]]])
            result = ts.conv2d(inputs, kernel)
            self.assertEqual(result.tolist(), [-4.0, -4.0, -4.0, -4.0])
            self.assertEqual(result.backend_storage.kind, backend)

    def test_sine_data_executes_with_a_nine_term_contraction(self):
        """The reproduction that NumPy and CUDA used to refuse."""
        values = [math.sin(index) for index in range(64)]
        weights = [1.0, 0.0, 0.5, 0.0, -1.0, 0.25, 2.0, 1.0, -0.5]
        for dtype in FLOATS:
            rounded = [round32(v) for v in values] if dtype is ts.float32 else values
            expected = []
            for row in range(6):
                for column in range(6):
                    patch_values = [
                        rounded[(row + dr) * 8 + column + dc]
                        for dr in range(3)
                        for dc in range(3)
                    ]
                    expected.append(pairwise_dot(patch_values, weights, dtype.name))
            for backend in self.each_backend():
                with self.subTest(dtype=dtype.name):
                    result = ts.conv2d(
                        ts.Tensor(values, dtype=dtype, shape=(1, 8, 8)),
                        ts.Tensor(weights, dtype=dtype, shape=(1, 1, 3, 3)),
                    )
                    self.assertEqual(result.backend_storage.kind, backend)
                    self.assertEqual(
                        [bits(v, dtype) for v in result.tolist()],
                        [bits(v, dtype) for v in expected],
                    )

    def test_padding_stride_and_bias(self):
        inputs = [math.sin(index) for index in range(9)]
        kernel = [0.3, -1.25, 0.7, 2.0]
        expected = conv1d_oracle(inputs, kernel, padding=2, stride=2, bias=0.1)
        for backend in self.each_backend():
            result = ts.conv1d(
                ts.Tensor(inputs, shape=(1, 1, 9)),
                ts.Tensor(kernel, shape=(1, 1, 4)),
                bias=ts.Tensor([0.1]),
                stride=2,
                padding=2,
            )
            self.assertEqual(result.tolist(), expected)
            self.assertEqual(result.backend_storage.kind, backend)


class GradientTests(PairwiseTestCase):
    def test_broadcast_gradient_reduction_of_fractional_addends(self):
        """``d sum(a + b) / d a`` sums seven fractional seed values per row."""
        seed = [[0.1 * (row + 1) + 0.01 * column for column in range(7)] for row in range(3)]
        expected = [pairwise_sum(row) for row in seed]
        for backend in self.each_backend():
            a = ts.Variable(ts.Tensor([1.0, 2.0, 3.0], shape=(3, 1)))
            b = ts.Variable(ts.Tensor([0.0] * 7, shape=(1, 7)))
            gradient = ts.grad(a + b, a, grad_outputs=ts.Tensor(seed))
            self.assertEqual(gradient.tolist(), expected)
            self.assertEqual(gradient.backend_storage.kind, backend)

    def test_broadcast_product_gradient_of_fractional_terms(self):
        """``d sum(a * b) / d a`` forms ``seed * b`` then sums it pairwise."""
        factor = [math.cos(index) for index in range(64)]
        expected = [pairwise_dot([1.0] * 64, factor)] * 3
        for backend in self.each_backend():
            a = ts.Variable(ts.Tensor([math.sin(i) for i in range(3)], shape=(3, 1)))
            b = ts.Variable(ts.Tensor(factor, shape=(1, 64)))
            gradient = ts.grad(ts.sum(a * b), a)
            self.assertEqual(gradient.tolist(), expected)
            self.assertEqual(gradient.backend_storage.kind, backend)

    def test_convolution_backward_with_ordinary_floats(self):
        """Kernel and bias VJPs of a 1-D convolution, against their definition.

        Without padding, the kernel VJP at offset ``k`` is the pairwise sum
        over output positions of ``grad[p] * input[p + k]``, and the bias VJP
        is the pairwise sum of ``grad``.
        """
        inputs = [math.sin(index) for index in range(11)]
        kernel = [0.5, -0.25, 1.5]
        upstream = [math.cos(index) for index in range(9)]
        kernel_expected = [
            pairwise_dot(upstream, inputs[offset : offset + 9]) for offset in range(3)
        ]
        bias_expected = [pairwise_sum(upstream)]
        for backend in self.each_backend():
            x = ts.Variable(ts.Tensor(inputs, shape=(1, 1, 11)))
            w = ts.Variable(ts.Tensor(kernel, shape=(1, 1, 3)))
            c = ts.Variable(ts.Tensor([0.2]))
            output = ts.conv1d(x, w, bias=c)
            kernel_gradient, bias_gradient = ts.grad(
                output, [w, c], grad_outputs=ts.Tensor(upstream, shape=(1, 1, 9))
            )
            self.assertEqual(kernel_gradient.tolist(), kernel_expected)
            self.assertEqual(bias_gradient.tolist(), bias_expected)
            self.assertEqual(kernel_gradient.backend_storage.kind, backend)
            self.assertEqual(bias_gradient.backend_storage.kind, backend)


class CrossBackendTests(PairwiseTestCase):
    """The same specified tree runs everywhere, so results are bit-identical."""

    def outputs(self, dtype):
        rng = random.Random(9)
        values = [rng.uniform(-1.0, 1.0) * 10.0 ** rng.randint(-6, 6) for _ in range(96)]
        with ts.use_backend(ts.get_backend()):
            x = ts.Tensor(values, dtype=dtype, shape=(2, 3, 16))
            w = ts.Tensor(values[:48], dtype=dtype, shape=(16, 3))
            image = ts.Variable(ts.Tensor(values[:50], dtype=dtype, shape=(2, 1, 5, 5)))
            kernel = ts.Variable(ts.Tensor(values[50:68], dtype=dtype, shape=(2, 1, 3, 3)))
            convolved = ts.conv2d(image, kernel, padding=1)
            gradients = ts.grad(ts.sum(convolved * convolved), [image, kernel])
            results = [
                ts.sum(x),
                ts.sum(x, axis=(0, 2), keepdims=True),
                x @ w,
                convolved.data,
                *gradients,
            ]
        return results

    def test_results_are_bit_identical_and_resident(self):
        backends = ts.available_backends()
        for dtype in FLOATS:
            with self.subTest(dtype=dtype.name):
                reference = None
                for backend in backends:
                    with ts.use_backend(backend):
                        produced = self.outputs(dtype)
                    for result in produced:
                        self.assertEqual(result.backend_storage.kind, backend)
                        self.assertIs(result.dtype, dtype)
                    encoded = [
                        (tuple(result.shape), [bits(v, dtype) for v in result.tolist()])
                        for result in produced
                    ]
                    if reference is None:
                        reference = encoded
                    else:
                        self.assertEqual(encoded, reference, backend)

    def test_no_backend_answers_through_the_python_reference(self):
        import tensors.backend.python.kernels as python_kernels

        for backend in ts.available_backends():
            if backend == "python":
                continue
            with (
                self.subTest(backend=backend),
                patch.object(
                    python_kernels, "reduce_sum", side_effect=AssertionError("fallback")
                ),
                patch.object(
                    python_kernels, "matmul", side_effect=AssertionError("fallback")
                ),
                ts.use_backend(backend),
            ):
                self.assertEqual(
                    ts.sum(ts.Tensor([0.1, 0.2, 0.3])).item(), 0.6000000000000001
                )
                ts.matmul(ts.Tensor([[0.1, 0.2, 0.3]]), ts.Tensor([[1.0], [2.0], [3.0]]))


class ErrorBoundTests(unittest.TestCase):
    """The documented bound holds for the specified tree, float32 and float64."""

    def cases(self, dtype_name):
        rng = random.Random(13 if dtype_name == "float64" else 17)
        tiny = 2.0**-149 if dtype_name == "float32" else 2.0**-1074
        rnd = round32 if dtype_name == "float32" else (lambda v: v)
        for size in (2, 3, 5, 8, 17, 64, 255, 256, 1000):
            yield [rnd(rng.uniform(-1.0, 1.0)) for _ in range(size)]
            yield [rnd(rng.choice((-1.0, 1.0)) * 10.0 ** rng.randint(-20, 20)) for _ in range(size)]
            yield [rnd(v) for v in ([1e8, 1.0, -1e8, 0.5] * size)[:size]]
            yield [rng.randint(-5, 5) * tiny for _ in range(size)]

    def test_the_bound_holds(self):
        for dtype_name in ("float32", "float64"):
            for values in self.cases(dtype_name):
                with self.subTest(dtype=dtype_name, size=len(values)):
                    produced = Fraction(pairwise_sum(values, dtype_name))
                    error = abs(produced - exact(values))
                    self.assertLessEqual(error, Fraction(error_bound(values, dtype_name)))

    def test_every_backend_meets_the_bound(self):
        for dtype, dtype_name in ((ts.float32, "float32"), (ts.float64, "float64")):
            for values in list(self.cases(dtype_name))[::5]:
                bound = Fraction(error_bound(values, dtype_name))
                for backend in ts.available_backends():
                    with self.subTest(dtype=dtype_name, backend=backend, size=len(values)):
                        with ts.use_backend(backend):
                            produced = ts.sum(ts.Tensor(values, dtype=dtype)).item()
                        self.assertLessEqual(abs(Fraction(produced) - exact(values)), bound)


if __name__ == "__main__":
    unittest.main()
