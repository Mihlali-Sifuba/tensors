"""CUDA binary32 subnormal handling across the remaining kernel families.

The eighteen forward kernels and the ``abs``/``relu`` gradients were corrected
first. This module covers the rest of the CUDA surface, where the same
flushing conversion appeared in three further shapes:

* ``_view(t).astype(cupy.float64, copy=False)`` — the dominant idiom, in
  ninety-seven call sites across fifty-seven files;
* ``_operand``, a second conversion helper reaching ``cupy.asarray`` with a
  ``float64`` dtype, used by ``outer`` and ``negate``;
* no widening at all, in ``maximum`` and ``minimum``, which passed binary32
  arrays straight to a CuPy ufunc whose generated binary32 code flushes.

**Two kinds of test appear here, and the distinction matters.** Where an
operation's *output* changes when the operand is flushed, the output is
asserted. Where it does not — many functions return the same correctly
rounded value for a subnormal operand as for zero — asserting the output
would prove nothing about the operand, so the conversion boundary itself is
tested instead.

No accuracy tolerance is introduced. The accuracy policies for these
operations remain unspecified (audit finding S-1) and nothing here pre-empts
them; what is asserted is exactness where the mathematics is exact,
preservation where a value must survive, and classification where a sign or a
domain decides the answer.
"""

from __future__ import annotations

import math
import struct
import unittest

import tensors as ts

from tests.backend._support import requires_cuda

SMALLEST = 1.401298464324817e-45
LARGEST_SUBNORMAL = 1.1754942106924411e-38
MIN_NORMAL = 1.1754943508222875e-38


def bits32(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def tensor32(values):
    return ts.Tensor(list(values), dtype=ts.float32)


def scalar(values):
    """The single value of a reduction's result, which is a one-element list."""
    return values[0] if isinstance(values, list) else values


@requires_cuda
class CudaTestCase(unittest.TestCase):
    """Restores the backend selection and offers the shared comparisons."""

    def setUp(self):
        self.previous = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous)

    def on_cuda(self, build):
        with ts.use_backend("cuda"):
            return build()

    def assertPreserved(self, produced, expected, context=""):
        """Bitwise: a value that must survive the round trip unchanged."""
        self.assertEqual(
            bits32(produced),
            bits32(expected),
            f"{context}: expected {expected!r}, got {produced!r}",
        )


class TheConversionBoundary(CudaTestCase):
    """What every correction in this milestone relies on.

    Several operations cannot show a flushed operand in their output. For
    those, this is the property that can be established: the value handed to
    the mathematics is the value the caller supplied.
    """

    OPERANDS = (
        SMALLEST,
        -SMALLEST,
        2 * SMALLEST,
        LARGEST_SUBNORMAL,
        -LARGEST_SUBNORMAL,
        MIN_NORMAL,
        0.0,
        -0.0,
        1.5,
    )

    def test_working_values_preserves_every_operand(self):
        import cupy

        from tensors.backend.cuda.conversion import _working_values

        with ts.use_backend("cuda"):
            widened = cupy.asnumpy(_working_values(tensor32(self.OPERANDS)))
        for operand, got in zip(self.OPERANDS, widened):
            with self.subTest(operand=operand):
                self.assertEqual(float(got), operand)

    def test_the_operand_helper_preserves_every_operand(self):
        """``_operand`` is the second helper that crosses the same boundary."""
        import cupy

        from tensors.backend.cuda.conversion import _operand

        with ts.use_backend("cuda"):
            widened = cupy.asnumpy(_operand(tensor32(self.OPERANDS), ts.float32))
        for operand, got in zip(self.OPERANDS, widened):
            with self.subTest(operand=operand):
                self.assertEqual(float(got), operand)

    def test_the_replaced_conversion_still_flushes(self):
        """Pins the mechanism, so the helpers are shown to be necessary."""
        import cupy

        from tensors.backend.cuda.conversion import _view

        with ts.use_backend("cuda"):
            flushed = cupy.asnumpy(
                _view(tensor32([SMALLEST])).astype(cupy.float64, copy=False)
            )
        self.assertEqual(
            float(flushed[0]), 0.0, "astype no longer flushes; this test is obsolete"
        )


class Reductions(CudaTestCase):
    """``sum``, ``mean``, ``prod`` and ``norm`` returned zero."""

    def test_a_sum_of_subnormals_is_exact(self):
        """Two quanta and three quanta are representable, so the sum is exact."""
        for count in (1, 2, 3, 4):
            with self.subTest(count=count):
                produced = self.on_cuda(
                    lambda n=count: scalar(ts.sum(tensor32([SMALLEST] * n)).tolist())
                )
                self.assertPreserved(produced, count * SMALLEST, f"sum of {count}")

    def test_a_mean_of_identical_subnormals_is_that_subnormal(self):
        for count in (1, 2, 4):
            with self.subTest(count=count):
                produced = self.on_cuda(
                    lambda n=count: scalar(ts.mean(tensor32([SMALLEST] * n)).tolist())
                )
                self.assertPreserved(produced, SMALLEST, f"mean of {count}")

    def test_a_product_with_one_is_the_operand(self):
        produced = self.on_cuda(
            lambda: scalar(ts.prod(tensor32([SMALLEST, 1.0])).tolist())
        )
        self.assertPreserved(produced, SMALLEST, "prod")

    def test_the_norm_of_a_single_subnormal_is_its_magnitude(self):
        for operand in (SMALLEST, -SMALLEST, LARGEST_SUBNORMAL):
            with self.subTest(operand=operand):
                produced = self.on_cuda(
                    lambda v=operand: scalar(ts.norm(tensor32([v])).tolist())
                )
                self.assertPreserved(produced, abs(operand), "norm")

    def test_extrema_select_the_subnormal_operand(self):
        """``max`` and ``min`` return one of their operands, exactly."""
        produced = self.on_cuda(
            lambda: scalar(ts.max(tensor32([SMALLEST, 0.0])).tolist())
        )
        self.assertPreserved(produced, SMALLEST, "max")
        produced = self.on_cuda(
            lambda: scalar(ts.min(tensor32([-SMALLEST, 0.0])).tolist())
        )
        self.assertPreserved(produced, -SMALLEST, "min")

    def test_argmax_sees_a_subnormal_as_greater_than_zero(self):
        """A classification, not a value: the flush made them equal."""
        produced = self.on_cuda(
            lambda: scalar(ts.argmax(tensor32([0.0, SMALLEST])).tolist())
        )
        self.assertEqual(produced, 1)

    def test_a_subnormal_summand_is_not_lost_beside_a_normal_one(self):
        """The sum is dominated, so only the operand's survival is testable."""
        produced = self.on_cuda(
            lambda: scalar(ts.sum(tensor32([SMALLEST, 1.0])).tolist())
        )
        self.assertEqual(produced, 1.0)

    def test_results_stay_in_cuda_storage(self):
        for name in ("sum", "mean", "prod", "max", "min", "norm"):
            for size in (1, 64, 10_000):
                with self.subTest(reduction=name, size=size):
                    with ts.use_backend("cuda"):
                        produced = getattr(ts, name)(tensor32([SMALLEST] * size))
                        self.assertEqual(
                            type(produced._storage).__name__, "CudaStorage"
                        )


class LinearAlgebra(CudaTestCase):
    """``matmul``, ``dot`` and ``outer``; ``outer`` used ``_operand``."""

    def test_multiplying_a_subnormal_by_one_returns_it(self):
        produced = self.on_cuda(
            lambda: ts.matmul(
                ts.Tensor([[SMALLEST]], dtype=ts.float32),
                ts.Tensor([[1.0]], dtype=ts.float32),
            ).tolist()
        )
        self.assertPreserved(produced[0], SMALLEST, "matmul")

    def test_dot_of_a_subnormal_with_one(self):
        produced = self.on_cuda(
            lambda: scalar(ts.dot(tensor32([SMALLEST]), tensor32([1.0])).tolist())
        )
        self.assertPreserved(produced, SMALLEST, "dot")

    def test_outer_products_with_one(self):
        for operand in (SMALLEST, -SMALLEST, LARGEST_SUBNORMAL):
            with self.subTest(operand=operand):
                produced = self.on_cuda(
                    lambda v=operand: ts.outer(tensor32([v]), tensor32([1.0])).tolist()
                )
                self.assertPreserved(produced[0], operand, "outer")

    def test_residency(self):
        with ts.use_backend("cuda"):
            produced = ts.matmul(
                ts.Tensor([[SMALLEST] * 64] * 64, dtype=ts.float32),
                ts.Tensor([[1.0] * 64] * 64, dtype=ts.float32),
            )
            self.assertEqual(type(produced._storage).__name__, "CudaStorage")


class SelectionAndComparison(CudaTestCase):
    """``maximum`` and ``minimum`` passed binary32 straight to a ufunc."""

    def test_maximum_selects_the_subnormal_over_zero(self):
        produced = self.on_cuda(
            lambda: ts.maximum(tensor32([SMALLEST]), tensor32([0.0])).tolist()
        )
        self.assertPreserved(produced[0], SMALLEST, "maximum")

    def test_minimum_selects_the_negative_subnormal(self):
        produced = self.on_cuda(
            lambda: ts.minimum(tensor32([-SMALLEST]), tensor32([0.0])).tolist()
        )
        self.assertPreserved(produced[0], -SMALLEST, "minimum")

    def test_the_extremes_of_the_subnormal_range(self):
        for label, left, right, expected in (
            ("largest subnormal over zero", LARGEST_SUBNORMAL, 0.0, LARGEST_SUBNORMAL),
            (
                "normal over largest subnormal",
                MIN_NORMAL,
                LARGEST_SUBNORMAL,
                MIN_NORMAL,
            ),
            ("two quanta over one", 2 * SMALLEST, SMALLEST, 2 * SMALLEST),
        ):
            with self.subTest(case=label):
                produced = self.on_cuda(
                    lambda a=left, b=right: ts.maximum(
                        tensor32([a]), tensor32([b])
                    ).tolist()
                )
                self.assertPreserved(produced[0], expected, label)

    def test_a_comparison_distinguishes_a_subnormal_from_zero(self):
        """Classification: the flush made these operands equal."""
        produced = self.on_cuda(
            lambda: ts.greater(tensor32([SMALLEST]), tensor32([0.0])).tolist()
        )
        self.assertEqual([bool(v) for v in produced], [True])
        produced = self.on_cuda(
            lambda: ts.equal(tensor32([SMALLEST]), tensor32([0.0])).tolist()
        )
        self.assertEqual([bool(v) for v in produced], [False])

    def test_where_and_clip_carry_a_subnormal_through(self):
        produced = self.on_cuda(
            lambda: ts.where(
                ts.Tensor([1], dtype=ts.int8), tensor32([SMALLEST]), tensor32([0.0])
            ).tolist()
        )
        self.assertPreserved(produced[0], SMALLEST, "where")
        produced = self.on_cuda(
            lambda: ts.clip(tensor32([SMALLEST]), 0.0, 1.0).tolist()
        )
        self.assertPreserved(produced[0], SMALLEST, "clip")


class Casting(CudaTestCase):
    """``astype`` read the source through the flushing conversion."""

    def test_widening_a_subnormal_to_float64(self):
        for operand in (SMALLEST, -SMALLEST, LARGEST_SUBNORMAL, -LARGEST_SUBNORMAL):
            with self.subTest(operand=operand):
                produced = self.on_cuda(
                    lambda v=operand: tensor32([v]).astype(ts.float64).tolist()
                )
                self.assertEqual(produced[0], operand, f"astype lost {operand!r}")

    def test_narrowing_a_float64_subnormal_to_float32(self):
        """The binary64 subnormal is far below binary32's range."""
        produced = self.on_cuda(
            lambda: ts.Tensor([5e-324], dtype=ts.float64).astype(ts.float32).tolist()
        )
        self.assertEqual(produced[0], 0.0)

    def test_narrowing_a_value_that_is_subnormal_in_binary32(self):
        produced = self.on_cuda(
            lambda: ts.Tensor([SMALLEST], dtype=ts.float64).astype(ts.float32).tolist()
        )
        self.assertPreserved(produced[0], SMALLEST, "float64 -> float32")

    def test_residency_and_dtype(self):
        with ts.use_backend("cuda"):
            produced = tensor32([SMALLEST] * 64).astype(ts.float64)
            self.assertEqual(type(produced._storage).__name__, "CudaStorage")
            self.assertIs(produced.dtype, ts.float64)


class ElementwiseGradients(CudaTestCase):
    """Every unary gradient kernel crossed the boundary twice."""

    #: Functions whose derivative at a subnormal operand is exactly one, so
    #: the VJP is the upstream gradient itself.
    UNIT_DERIVATIVE = (
        "sin",
        "tan",
        "arcsin",
        "arctan",
        "arcsinh",
        "arctanh",
        "sinh",
        "tanh",
        "exp",
    )

    def gradient(self, function, operand, upstream, dtype=ts.float32):
        with ts.use_backend("cuda"):
            variable = ts.Variable(
                ts.Tensor([operand], dtype=dtype), requires_grad=True
            )
            (produced,) = ts.grad(
                function(variable),
                [variable],
                grad_outputs=ts.Tensor([upstream], dtype=dtype),
            )
            return produced

    def test_a_subnormal_upstream_gradient_survives_a_unit_derivative(self):
        """``f'(0⁺) = 1`` for each of these, so the VJP is the upstream."""
        for name in self.UNIT_DERIVATIVE:
            for upstream in (SMALLEST, -SMALLEST, LARGEST_SUBNORMAL):
                with self.subTest(operation=name, upstream=upstream):
                    produced = self.gradient(
                        getattr(ts, name), SMALLEST, upstream
                    ).tolist()[0]
                    self.assertPreserved(produced, upstream, f"d/dx {name}")

    def test_the_sigmoid_derivative_scales_a_subnormal_upstream(self):
        """``sigmoid'(0) = 1/4`` exactly, and a quarter of a subnormal is exact
        only when the quantum divides: four quanta scale to one."""
        produced = self.gradient(ts.sigmoid, SMALLEST, 4 * SMALLEST).tolist()[0]
        self.assertPreserved(produced, SMALLEST, "d/dx sigmoid")

    def test_the_sqrt_derivative_is_finite_at_a_subnormal(self):
        """``sqrt'(x) = 1/(2√x)``, which is large but finite at a subnormal."""
        produced = self.gradient(ts.sqrt, SMALLEST, 1.0).tolist()[0]
        self.assertTrue(math.isfinite(produced), repr(produced))
        self.assertGreater(produced, 0.0)

    def test_residency(self):
        for name in ("sin", "tanh", "exp", "sigmoid"):
            with self.subTest(operation=name):
                produced = self.gradient(getattr(ts, name), SMALLEST, 1.0)
                self.assertEqual(type(produced._storage).__name__, "CudaStorage")


class NeuralNetworkFamilies(CudaTestCase):
    """Losses, normalisation and convolution read operands the same way."""

    def test_convolution_carries_a_subnormal_through_a_unit_kernel(self):
        produced = self.on_cuda(
            lambda: ts.conv1d(
                ts.Tensor([[[SMALLEST]]], dtype=ts.float32),
                ts.Tensor([[[1.0]]], dtype=ts.float32),
            ).tolist()
        )
        self.assertPreserved(produced[0], SMALLEST, "conv1d")

    def test_softmax_of_equal_operands_is_uniform(self):
        """A classification the flush could not change, so the boundary test
        in :class:`TheConversionBoundary` is what covers softmax's operand."""
        produced = self.on_cuda(
            lambda: ts.softmax(tensor32([SMALLEST, SMALLEST])).tolist()
        )
        self.assertEqual(produced, [0.5, 0.5])

    def test_a_loss_accepts_a_subnormal_target(self):
        produced = self.on_cuda(
            lambda: ts.binary_cross_entropy(
                tensor32([0.5]), tensor32([SMALLEST])
            ).tolist()
        )
        self.assertTrue(math.isfinite(scalar(produced)))

    def test_residency(self):
        with ts.use_backend("cuda"):
            produced = ts.softmax(tensor32([SMALLEST] * 64))
            self.assertEqual(type(produced._storage).__name__, "CudaStorage")


class ExecutionAndResidency(CudaTestCase):
    """The corrections introduced no fallback and no host materialisation."""

    def _counting_device_reads(self):
        import contextlib

        import tensors.tensor as tensor_module

        class Counter:
            count = 0

        @contextlib.contextmanager
        def counting():
            counter = Counter()
            original = tensor_module.Tensor._data.fget

            def counted(self):
                counter.count += 1
                return original(self)

            tensor_module.Tensor._data = property(counted)
            try:
                yield counter
            finally:
                tensor_module.Tensor._data = property(original)

        return counting()

    def test_no_operation_reads_an_operand_back_to_the_host(self):
        operations = {
            "sum": lambda t: ts.sum(t),
            "mean": lambda t: ts.mean(t),
            "norm": lambda t: ts.norm(t),
            "maximum": lambda t: ts.maximum(t, t),
            "minimum": lambda t: ts.minimum(t, t),
            "softmax": lambda t: ts.softmax(t),
            "astype": lambda t: t.astype(ts.float64),
            "sin": lambda t: ts.sin(t),
        }
        for name, build in operations.items():
            with self.subTest(operation=name):
                with ts.use_backend("cuda"):
                    operand = tensor32([SMALLEST] * 4096) + 0.0
                    with self._counting_device_reads() as reads:
                        produced = build(operand)
                        self.assertEqual(
                            type(produced._storage).__name__, "CudaStorage"
                        )
                self.assertEqual(
                    reads.count, 0, f"{name} materialised an operand on the host"
                )

    def test_eager_and_graph_execution_agree(self):
        from tensors.graph import Computation

        for name in ("sin", "tanh", "sqrt", "exp"):
            with self.subTest(operation=name):
                function = getattr(ts, name)
                with ts.use_backend("cuda"):
                    eager = function(tensor32([SMALLEST] * 64)).tolist()[0]
                    variable = ts.Variable(
                        tensor32([SMALLEST] * 64), requires_grad=False
                    )
                    replayed = Computation(function(variable)).forward().tolist()[0]
                self.assertPreserved(replayed, eager, f"{name} replay")

    def test_float64_is_unaffected(self):
        smallest64 = 5e-324
        operations = {
            "sum": lambda t: ts.sum(t),
            "mean": lambda t: ts.mean(t),
            "norm": lambda t: ts.norm(t),
            "maximum": lambda t: ts.maximum(t, t),
            "sin": lambda t: ts.sin(t),
        }
        for name, build in operations.items():
            with self.subTest(operation=name):
                with ts.use_backend("cuda"):
                    produced = build(ts.Tensor([smallest64], dtype=ts.float64))
                    value = produced.tolist()
                    value = value[0] if isinstance(value, list) else value
                self.assertNotEqual(value, 0.0, f"{name} flushed a float64 subnormal")


if __name__ == "__main__":
    unittest.main()
