import math
import unittest
from pathlib import Path
from unittest.mock import patch
import tensors as ts
from tensors.backend.config import (
    BackendMismatchError,
    BackendOperationUnsupportedError,
)
from tensors.backend import dispatch as backend_dispatch
from tensors.graph import Computation
from tensors.graph.state import reset_graph_state
from tests._pairwise_oracle import pairwise_dot, pairwise_sum


def _identity_case():
    """Return a 2x2 input and the 2x2 diagonal kernel used across tests."""
    return (
        ts.Tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),
        ts.Tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
    )


class Conv2dTests(unittest.TestCase):

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_conv2d_correlates_without_reversing_the_kernel(self):
        inputs, kernel = _identity_case()
        result = ts.conv2d(inputs, kernel)
        self.assertEqual(result.shape, (1, 1, 1, 1))
        self.assertEqual(result.tolist(), [5.0])

    def test_conv2d_is_available_in_the_math_namespace(self):
        inputs, kernel = _identity_case()
        self.assertEqual(ts.math.conv2d(inputs, kernel).tolist(), [5.0])

    def test_conv2d_adds_a_per_output_channel_bias(self):
        inputs, kernel = _identity_case()
        result = ts.conv2d(inputs, kernel, ts.Tensor([10.0]))
        self.assertEqual(result.tolist(), [15.0])

    def test_conv2d_pads_both_ends_of_each_spatial_axis(self):
        inputs, kernel = _identity_case()
        result = ts.conv2d(inputs, kernel, padding=1)
        self.assertEqual(result.shape, (1, 1, 3, 3))
        self.assertEqual(result.tolist(), [1.0, 2.0, 0.0, 3.0, 5.0, 2.0, 0.0, 3.0, 4.0])

    def test_conv2d_accepts_per_axis_stride_padding_and_dilation(self):
        inputs = ts.Tensor([[[[float(value) for value in range(6)]] * 5]])
        kernel = ts.Tensor([[[[1.0, 1.0]]]])
        result = ts.conv2d(
            inputs, kernel, stride=(2, 1), padding=(1, 0), dilation=(1, 2)
        )
        self.assertEqual(result.shape, (1, 1, 4, 4))

    def test_conv2d_output_extent_follows_the_documented_formula(self):
        inputs = ts.zeros((2, 3, 9, 11))
        kernel = ts.zeros((4, 3, 3, 2))
        result = ts.conv2d(inputs, kernel, stride=2, padding=1, dilation=2)
        self.assertEqual(result.shape, (2, 4, 4, 6))

    def test_conv2d_promotes_operand_dtypes(self):
        inputs = ts.Tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=ts.float32)
        kernel = ts.Tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=ts.float64)
        self.assertIs(ts.conv2d(inputs, kernel).dtype, ts.float64)

    def test_conv2d_records_its_operands_and_backpropagates(self):
        inputs = ts.Variable([[[[1.0, 2.0], [3.0, 4.0]]]], name="x")
        kernel = ts.Variable([[[[1.0, 0.0], [0.0, 1.0]]]], name="w")
        bias = ts.Variable([10.0], name="b")
        result = ts.conv2d(inputs, kernel, bias)
        self.assertEqual(result.node.producer.label, "conv2d")
        self.assertEqual(
            result.node.producer.inputs, [inputs.node, kernel.node, bias.node]
        )
        ts.backward(result)
        self.assertEqual(inputs.grad.tolist(), [1.0, 0.0, 0.0, 1.0])
        self.assertEqual(kernel.grad.tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(bias.grad.tolist(), [1.0])

    def test_conv2d_replays_through_a_recorded_computation(self):
        inputs = ts.Variable([[[[1.0, 2.0], [3.0, 4.0]]]])
        kernel = ts.Variable([[[[1.0, 0.0], [0.0, 1.0]]]])
        result = ts.conv2d(inputs, kernel)
        self.assertEqual(Computation(result).forward().tolist(), [5.0])

    def test_conv2d_becomes_a_variable_when_only_the_bias_requires_grad(self):
        inputs, kernel = _identity_case()
        bias = ts.Variable([1.0], name="b")
        result = ts.conv2d(inputs, kernel, bias)
        ts.backward(result)
        self.assertIsInstance(result, ts.Variable)
        self.assertEqual(bias.grad.tolist(), [1.0])

    def test_conv2d_gradients_match_finite_differences(self):
        inputs = ts.Tensor([[[[0.4, -0.2, 0.7], [1.1, 0.3, -0.9], [0.5, 0.8, -0.1]]]])
        kernel = ts.Tensor([[[[0.6, -0.3], [0.2, 0.9]]]])
        bias = ts.Tensor([0.25])
        ts.gradcheck(
            lambda x, w, b: ts.conv2d(x, w, b, padding=1), [inputs, kernel, bias]
        )

    @unittest.skip("Higher-order convolution gradients are outside current support")
    def test_convolution_builds_differentiable_higher_order_gradients(self):
        inputs = ts.Variable([[[1.0, 2.0, 3.0]]])
        kernel = ts.Variable([[[1.0, -1.0]]])
        output = ts.sum(ts.conv1d(inputs, kernel))
        input_gradient, kernel_gradient = ts.grad(
            output, (inputs, kernel), create_graph=True
        )
        kernel_cross_gradient = ts.grad(ts.sum(input_gradient), kernel)
        input_cross_gradient = ts.grad(ts.sum(kernel_gradient), inputs)
        self.assertEqual(input_gradient.data.tolist(), [1.0, 0.0, -1.0])
        self.assertEqual(kernel_gradient.data.tolist(), [3.0, 5.0])
        self.assertEqual(kernel_cross_gradient.tolist(), [2.0, 2.0])
        self.assertEqual(input_cross_gradient.tolist(), [1.0, 2.0, 1.0])


class Conv2dGroupTests(unittest.TestCase):

    def test_grouped_channels_are_convolved_independently(self):
        inputs = ts.Tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
        kernel = ts.Tensor([[[[1.0, 1.0]]], [[[2.0, 2.0]]]])
        result = ts.conv2d(inputs, kernel, groups=2)
        self.assertEqual(result.shape, (1, 2, 1, 1))
        self.assertEqual(result.tolist(), [3.0, 14.0])

    def test_depthwise_convolution_uses_one_kernel_per_channel(self):
        inputs = ts.zeros((1, 4, 5, 5))
        kernel = ts.zeros((4, 1, 3, 3))
        result = ts.conv2d(inputs, kernel, groups=4)
        self.assertEqual(result.shape, (1, 4, 3, 3))

    def test_grouped_gradients_match_finite_differences(self):
        inputs = ts.Tensor([[[[0.5, -0.4, 0.9]], [[1.2, 0.3, -0.7]]]])
        kernel = ts.Tensor([[[[0.8, -0.2]]], [[[0.4, 0.6]]]])
        ts.gradcheck(lambda x, w: ts.conv2d(x, w, groups=2), [inputs, kernel])


class Conv1dTests(unittest.TestCase):

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_conv1d_slides_a_kernel_along_the_spatial_axis(self):
        inputs = ts.Tensor([[[1.0, 2.0, 3.0, 4.0]]])
        kernel = ts.Tensor([[[1.0, -1.0]]])
        result = ts.conv1d(inputs, kernel)
        self.assertEqual(result.shape, (1, 1, 3))
        self.assertEqual(result.tolist(), [-1.0, -1.0, -1.0])

    def test_conv1d_is_available_in_the_math_namespace(self):
        inputs = ts.Tensor([[[1.0, 2.0]]])
        kernel = ts.Tensor([[[1.0, 1.0]]])
        self.assertEqual(ts.math.conv1d(inputs, kernel).tolist(), [3.0])

    def test_conv1d_applies_stride_and_dilation(self):
        inputs = ts.Tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
        kernel = ts.Tensor([[[1.0, 1.0]]])
        strided = ts.conv1d(inputs, kernel, stride=2)
        dilated = ts.conv1d(inputs, kernel, dilation=2)
        self.assertEqual(strided.tolist(), [3.0, 7.0])
        self.assertEqual(dilated.tolist(), [4.0, 6.0, 8.0])

    def test_conv1d_groups_split_channels(self):
        inputs = ts.Tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        kernel = ts.Tensor([[[1.0, 1.0]], [[2.0, 2.0]]])
        result = ts.conv1d(inputs, kernel, groups=2)
        self.assertEqual(result.tolist(), [3.0, 5.0, 18.0, 22.0])

    def test_conv1d_records_its_label_and_backpropagates(self):
        inputs = ts.Variable([[[1.0, 2.0, 3.0]]], name="x")
        kernel = ts.Variable([[[1.0, -1.0]]], name="w")
        result = ts.conv1d(inputs, kernel)
        self.assertEqual(result.node.producer.label, "conv1d")
        ts.backward(result)
        self.assertEqual(inputs.grad.tolist(), [1.0, 0.0, -1.0])
        self.assertEqual(kernel.grad.tolist(), [3.0, 5.0])

    def test_conv1d_gradients_match_finite_differences(self):
        inputs = ts.Tensor([[[0.3, -0.6, 1.2, 0.4], [0.9, 0.1, -0.5, 0.7]]])
        kernel = ts.Tensor([[[0.5, -0.8]], [[0.2, 0.6]]])
        ts.gradcheck(
            lambda x, w: ts.conv1d(x, w, padding=1, groups=2), [inputs, kernel]
        )


class Conv3dAndUnbatchedTests(unittest.TestCase):

    def test_conv1d_accepts_an_unbatched_signal(self):
        result = ts.conv1d(ts.Tensor([[1.0, 2.0, 3.0]]), ts.Tensor([[[1.0, -1.0]]]))
        self.assertEqual(result.shape, (1, 2))
        self.assertEqual(result.tolist(), [-1.0, -1.0])

    def test_conv2d_accepts_an_unbatched_image(self):
        inputs, kernel = _identity_case()
        result = ts.conv2d(ts.reshape(inputs, (1, 2, 2)), kernel)
        self.assertEqual(result.shape, (1, 1, 1))
        self.assertEqual(result.tolist(), [5.0])

    def test_conv3d_correlates_a_volume(self):
        inputs = ts.Tensor([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]])
        kernel = ts.ones((1, 1, 2, 2, 2))
        result = ts.conv3d(inputs, kernel, ts.Tensor([1.0]))
        self.assertEqual(result.shape, (1, 1, 1, 1, 1))
        self.assertEqual(result.tolist(), [37.0])
        self.assertEqual(ts.math.conv3d(inputs, kernel).tolist(), [36.0])

    def test_conv3d_accepts_unbatched_grouped_volumes(self):
        inputs = ts.Tensor([[[[1.0, 2.0]]], [[[3.0, 4.0]]]])
        kernel = ts.Tensor([[[[[1.0, 1.0]]]], [[[[2.0, 2.0]]]]])
        result = ts.conv3d(inputs, kernel, groups=2)
        self.assertEqual(result.shape, (2, 1, 1, 1))
        self.assertEqual(result.tolist(), [3.0, 14.0])

    def test_unbatched_conv3d_gradients_match_finite_differences(self):
        inputs = ts.Tensor([[[[0.2, -0.4], [0.5, 0.7]], [[0.1, 0.3], [-0.2, 0.8]]]])
        kernel = ts.Tensor([[[[[0.6, -0.1], [0.2, 0.4]], [[-0.3, 0.5], [0.7, -0.2]]]]])
        ts.gradcheck(lambda x, w: ts.conv3d(x, w, padding=1), [inputs, kernel])

    def test_conv3d_replays_through_a_recorded_computation(self):
        inputs = ts.Variable(ts.ones((1, 1, 2, 2, 2)))
        kernel = ts.Variable(ts.ones((1, 1, 2, 2, 2)))
        result = ts.conv3d(inputs, kernel)
        self.assertEqual(result.node.producer.label, "conv3d")
        self.assertEqual(Computation(result).forward().tolist(), [8.0])


class ConvolutionExactnessTests(unittest.TestCase):

    def test_integer_convolution_keeps_exact_python_intermediates(self):
        inputs = ts.Tensor([[[2**60, 2**60 + 1]]], dtype=ts.int64)
        kernel = ts.Tensor([[[1, 1]]], dtype=ts.int64)
        result = ts.conv1d(inputs, kernel)
        self.assertIs(result.dtype, ts.int64)
        self.assertEqual(result.tolist(), [2**61 + 1])

    def test_convolution_rounds_each_product_before_summing(self):
        """``fl(1e300 * 1e300)`` is ``+inf`` and its negation ``-inf``.

        The products are formed in the dtype and then summed pairwise, so the
        group holds both infinities and the specified result is NaN.
        """
        inputs = ts.Tensor([[[1e300, 1e300]]])
        kernel = ts.Tensor([[[1e300, -1e300]]])
        result = ts.conv1d(inputs, kernel)
        self.assertTrue(math.isnan(result.item()))


class ConvolutionValidationTests(unittest.TestCase):

    def test_input_rank_must_match_the_convolution_rank(self):
        with self.assertRaisesRegex(ValueError, "3 unbatched.*4 batched"):
            ts.conv2d(ts.zeros((4,)), ts.zeros((1, 1, 2, 2)))

    def test_kernel_rank_must_match_the_convolution_rank(self):
        with self.assertRaisesRegex(ValueError, "conv1d kernel must have 3"):
            ts.conv1d(ts.zeros((1, 1, 4)), ts.zeros((1, 1, 2, 2)))

    def test_kernel_input_channels_must_match_the_group_width(self):
        with self.assertRaisesRegex(ValueError, "input channels per group"):
            ts.conv2d(ts.zeros((1, 3, 4, 4)), ts.zeros((2, 2, 2, 2)))

    def test_channels_must_divide_evenly_into_groups(self):
        with self.assertRaisesRegex(ValueError, "not divisible by groups"):
            ts.conv2d(ts.zeros((1, 3, 4, 4)), ts.zeros((2, 1, 2, 2)), groups=2)

    def test_kernel_span_may_not_exceed_the_padded_input(self):
        with self.assertRaisesRegex(ValueError, "exceeds the padded input"):
            ts.conv2d(ts.zeros((1, 1, 2, 2)), ts.zeros((1, 1, 5, 2)))

    def test_bias_shape_must_match_the_output_channels(self):
        with self.assertRaisesRegex(ValueError, "does not match the expected"):
            ts.conv2d(ts.zeros((1, 1, 2, 2)), ts.zeros((3, 1, 2, 2)), ts.zeros((2,)))

    def test_stride_and_dilation_must_be_positive(self):
        inputs = ts.zeros((1, 1, 4, 4))
        kernel = ts.zeros((1, 1, 2, 2))
        with self.assertRaisesRegex(ValueError, "stride entries"):
            ts.conv2d(inputs, kernel, stride=0)
        with self.assertRaisesRegex(ValueError, "dilation entries"):
            ts.conv2d(inputs, kernel, dilation=0)
        with self.assertRaisesRegex(ValueError, "padding entries"):
            ts.conv2d(inputs, kernel, padding=-1)

    def test_per_axis_arguments_must_match_the_rank(self):
        with self.assertRaisesRegex(ValueError, "must contain 2 values"):
            ts.conv2d(ts.zeros((1, 1, 4, 4)), ts.zeros((1, 1, 2, 2)), stride=(1, 1, 1))

    def test_boolean_and_non_integer_arguments_are_rejected(self):
        inputs = ts.zeros((1, 1, 4, 4))
        kernel = ts.zeros((1, 1, 2, 2))
        with self.assertRaisesRegex(TypeError, "stride must be an integer"):
            ts.conv2d(inputs, kernel, stride=True)
        with self.assertRaisesRegex(TypeError, "stride must be an integer"):
            ts.conv2d(inputs, kernel, stride=1.0)
        with self.assertRaisesRegex(TypeError, "groups must be an integer"):
            ts.conv2d(inputs, kernel, groups=True)

    def test_groups_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "groups must be at least 1"):
            ts.conv2d(ts.zeros((1, 1, 4, 4)), ts.zeros((1, 1, 2, 2)), groups=0)


@unittest.skipUnless(
    len(ts.available_backends()) > 1, "No accelerated backend is installed"
)
class ConvolutionBackendParityTests(unittest.TestCase):
    """Every installed backend must agree with the Python reference."""

    CASES = (
        ((2, 3, 6, 7), (4, 3, 3, 2), True, 1, 0, 1, 1),
        ((2, 4, 8, 8), (6, 2, 3, 3), True, 2, 1, 1, 2),
        ((1, 4, 9, 9), (4, 1, 3, 3), True, 1, 2, 2, 4),
        ((3, 2, 7, 5), (5, 2, 2, 3), True, (2, 1), (1, 2), (2, 1), 1),
    )

    @staticmethod
    def _ramp(shape, scale):
        size = ts.Shape.from_iterable(shape).size
        values = [scale * (index * 37 % 19 - 9) / 8.0 for index in range(size)]
        return ts.Tensor(values, shape=shape)

    def test_forward_matches_the_python_reference(self):
        for case in self.CASES:
            shape, kernel_shape, use_bias, stride, padding, dilation, groups = case
            with self.subTest(case=case):
                options = dict(
                    stride=stride, padding=padding, dilation=dilation, groups=groups
                )
                with ts.use_backend("python"):
                    inputs = self._ramp(shape, 0.5)
                    kernel = self._ramp(kernel_shape, 0.25)
                    bias = self._ramp((kernel_shape[0],), 1.0) if use_bias else None
                    expected = ts.conv2d(inputs, kernel, bias, **options)
                for backend in ts.available_backends():
                    with ts.use_backend(backend):
                        inputs = self._ramp(shape, 0.5)
                        kernel = self._ramp(kernel_shape, 0.25)
                        bias = self._ramp((kernel_shape[0],), 1.0) if use_bias else None
                        result = ts.conv2d(inputs, kernel, bias, **options)
                    self.assertEqual(result.shape, expected.shape)
                    for actual, want in zip(result.tolist(), expected.tolist()):
                        self.assertAlmostEqual(actual, want, places=10)

    def test_gradients_match_the_python_reference(self):
        for case in self.CASES:
            shape, kernel_shape, use_bias, stride, padding, dilation, groups = case
            with self.subTest(case=case):
                options = dict(
                    stride=stride, padding=padding, dilation=dilation, groups=groups
                )

                def gradients(backend):
                    reset_graph_state()
                    with ts.use_backend(backend):
                        inputs = ts.Variable(self._ramp(shape, 0.5))
                        kernel = ts.Variable(self._ramp(kernel_shape, 0.25))
                        operands = [inputs, kernel]
                        if use_bias:
                            operands.append(
                                ts.Variable(self._ramp((kernel_shape[0],), 1.0))
                            )
                        output = ts.conv2d(*operands, **options)
                        seed = self._ramp(tuple(output.shape), 0.75)
                        ts.backward(output, seed)
                        return [operand.grad.tolist() for operand in operands]

                expected = gradients("python")
                for backend in ts.available_backends():
                    for actual, want in zip(gradients(backend), expected):
                        for value, target in zip(actual, want):
                            self.assertAlmostEqual(value, target, places=10)

    def test_unbatched_and_three_dimensional_forward_paths_match_python(self):
        cases = (
            (ts.conv1d, (2, 8), (3, 2, 3), dict(stride=2, padding=1)),
            (ts.conv2d, (2, 5, 6), (4, 2, 2, 3), dict(stride=(2, 1), padding=(1, 0))),
            (
                ts.conv3d,
                (2, 2, 4, 5, 6),
                (3, 2, 2, 2, 3),
                dict(stride=(1, 2, 1), padding=1),
            ),
            (
                ts.conv3d,
                (2, 4, 4, 5),
                (2, 1, 2, 2, 2),
                dict(groups=2, dilation=(1, 2, 1)),
            ),
        )
        for operation, shape, kernel_shape, options in cases:
            with self.subTest(operation=operation.__name__, shape=shape):
                with ts.use_backend("python"):
                    inputs = self._ramp(shape, 0.5)
                    kernel = self._ramp(kernel_shape, 0.25)
                    expected = operation(inputs, kernel, **options)
                for backend in ts.available_backends():
                    with ts.use_backend(backend):
                        inputs = self._ramp(shape, 0.5)
                        kernel = self._ramp(kernel_shape, 0.25)
                        result = operation(inputs, kernel, **options)
                    self.assertEqual(result.shape, expected.shape)
                    for actual, want in zip(result.tolist(), expected.tolist()):
                        self.assertAlmostEqual(actual, want, places=10)

    def test_float32_convolution_stays_float32_on_accelerated_backends(self):
        with ts.use_backend("python"):
            inputs = ts.Tensor(
                [float(index % 13) / 8.0 for index in range(2 * 3 * 8 * 8)],
                dtype=ts.float32,
                shape=(2, 3, 8, 8),
            )
            kernel = ts.Tensor(
                [float(index % 11) / 16.0 for index in range(4 * 3 * 3 * 3)],
                dtype=ts.float32,
                shape=(4, 3, 3, 3),
            )
            expected = ts.conv2d(inputs, kernel, padding=1)
        for backend in ts.available_backends():
            with ts.use_backend(backend):
                inputs = ts.Tensor(
                    [float(index % 13) / 8.0 for index in range(2 * 3 * 8 * 8)],
                    dtype=ts.float32,
                    shape=(2, 3, 8, 8),
                )
                kernel = ts.Tensor(
                    [float(index % 11) / 16.0 for index in range(4 * 3 * 3 * 3)],
                    dtype=ts.float32,
                    shape=(4, 3, 3, 3),
                )
                result = ts.conv2d(inputs, kernel, padding=1)
            self.assertIs(result.dtype, ts.float32)
            for actual, want in zip(result.tolist(), expected.tolist()):
                self.assertAlmostEqual(actual, want, places=4)

    def test_conv3d_gradients_match_the_python_backend(self):
        shape = (2, 4, 4, 4)
        kernel_shape = (2, 1, 2, 2, 2)
        options = dict(groups=2, padding=1)

        def gradients(backend):
            reset_graph_state()
            with ts.use_backend(backend):
                inputs = ts.Variable(self._ramp(shape, 0.5))
                kernel = ts.Variable(self._ramp(kernel_shape, 0.25))
                output = ts.conv3d(inputs, kernel, **options)
                ts.backward(output, self._ramp(tuple(output.shape), 0.75))
                return (inputs.grad.tolist(), kernel.grad.tolist())

        expected = gradients("python")
        for backend in ts.available_backends():
            for actual, want in zip(gradients(backend), expected):
                for value, target in zip(actual, want):
                    self.assertAlmostEqual(value, target, places=10)

    def test_accelerated_convolution_matches_when_forced_across_small_tiles(self):
        import importlib
        from unittest.mock import patch

        options = dict(stride=(2, 1), padding=(1, 2), dilation=(2, 1))
        with ts.use_backend("python"):
            inputs = self._ramp((2, 3, 7, 8), 0.5)
            kernel = self._ramp((4, 3, 3, 2), 0.25)
            expected = ts.conv2d(inputs, kernel, **options)
        for backend in ts.available_backends():
            if backend == "python":
                continue
            common = importlib.import_module(
                f"tensors.backend.{backend}.kernels.convolution.common"
            )
            with (
                patch.object(common, "_CONVOLUTION_COLUMN_MAX_ELEMENTS", 64),
                ts.use_backend(backend),
            ):
                inputs = self._ramp((2, 3, 7, 8), 0.5)
                kernel = self._ramp((4, 3, 3, 2), 0.25)
                result = ts.conv2d(inputs, kernel, **options)
            self.assertEqual(result.shape, expected.shape)
            for actual, want in zip(result.tolist(), expected.tolist()):
                self.assertAlmostEqual(actual, want, places=10)


class ConvolutionSelectedBackendExecutionTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_dispatchers_and_kernels_have_strict_native_boundaries(self):
        repository = Path(__file__).resolve().parents[3]
        for filename in ("convolution.py", "convolution_gradient.py"):
            source = (
                repository
                / "tensors"
                / "backend"
                / "dispatch"
                / "convolution"
                / filename
            ).read_text(encoding="utf-8")
            for forbidden in (
                "_array_work_is_large_enough",
                "_NUMPY_MATMUL_MIN_WORK",
                "_backend_kernel",
                "python.kernels",
                " as reference",
            ):
                self.assertNotIn(forbidden, source)
            self.assertIn("validate_backend_residency", source)
            self.assertIn("BackendOperationUnsupportedError", source)

        for backend in ("python", "numpy", "cuda"):
            for filename in ("convolution.py", "convolution_gradient.py"):
                source = (
                    repository
                    / "tensors"
                    / "backend"
                    / backend
                    / "kernels"
                    / "convolution"
                    / filename
                ).read_text(encoding="utf-8")
                for forbidden in (
                    "Tensor",
                    ".get_host_values(",
                    "tensor_to_logical_array",
                    "get_backend",
                ):
                    self.assertNotIn(forbidden, source)
                if backend != "python" and filename == "convolution_gradient.py":
                    self.assertNotIn("destinations = list(", source)
                    self.assertNotIn("positions = tuple(", source)

    def test_small_forward_and_first_order_vjps_stay_selected_backend_native(self):
        for backend in ts.available_backends():
            with self.subTest(backend=backend), ts.use_backend(backend):
                inputs = ts.Variable(
                    ts.Tensor([1.0, 2.0, 3.0, 4.0], shape=(1, 1, 2, 2))
                )
                kernel = ts.Variable(ts.Tensor([1.0], shape=(1, 1, 1, 1)))
                bias = ts.Variable(ts.Tensor([0.5]))
                output = ts.conv2d(inputs, kernel, bias)
                ts.backward(output, ts.ones(output.shape))
                self.assertEqual(output.data.backend_storage.kind, backend)
                self.assertEqual(inputs.grad.backend_storage.kind, backend)
                self.assertEqual(kernel.grad.backend_storage.kind, backend)
                self.assertEqual(bias.grad.backend_storage.kind, backend)
                self.assertEqual(output.data.tolist(), [1.5, 2.5, 3.5, 4.5])
                self.assertEqual(inputs.grad.tolist(), [1.0, 1.0, 1.0, 1.0])
                self.assertEqual(kernel.grad.tolist(), [10.0])
                self.assertEqual(bias.grad.tolist(), [4.0])

    @unittest.skipUnless("numpy" in ts.available_backends(), "NumPy unavailable")
    def test_decline_and_foreign_residency_raise_without_fallback(self):
        from tensors.backend.loading import load_backend

        numpy_backend = load_backend("numpy")
        with ts.use_backend("numpy"):
            inputs = ts.Tensor([1.0], shape=(1, 1, 1))
            kernel = ts.Tensor([1.0], shape=(1, 1, 1))
            with patch.object(numpy_backend, "convolution", return_value=None):
                with self.assertRaises(BackendOperationUnsupportedError):
                    ts.conv1d(inputs, kernel)

        with ts.use_backend("python"):
            foreign = ts.Tensor([1.0], shape=(1, 1, 1))
        with ts.use_backend("numpy"):
            kernel = ts.Tensor([1.0], shape=(1, 1, 1))
            with self.assertRaises(BackendMismatchError):
                ts.conv1d(foreign, kernel)

    def test_empty_contribution_groups_produce_native_positive_zero(self):
        for backend in ts.available_backends():
            with self.subTest(backend=backend), ts.use_backend(backend):
                inputs = ts.Tensor([], shape=(0, 1, 3))
                kernel = ts.Tensor([1.0, 2.0], shape=(2, 1, 1))
                output = ts.conv1d(inputs, kernel)
                self.assertEqual(output.shape, (0, 2, 3))
                grad = ts.Tensor([], shape=output.shape)
                input_storage, kernel_storage, bias_storage = (
                    backend_dispatch.execute_convolution_gradient(
                        grad,
                        inputs,
                        kernel,
                        stride=(1,),
                        padding=(0,),
                        dilation=(1,),
                        groups=1,
                        include_bias=True,
                        needs_input_grad=(True, True, True),
                    )
                )
                self.assertEqual(input_storage.kind, backend)
                self.assertEqual(kernel_storage.kind, backend)
                self.assertEqual(bias_storage.kind, backend)
                self.assertEqual(list(kernel_storage.buffer), [0.0, 0.0])
                self.assertEqual(list(bias_storage.buffer), [0.0, 0.0])
                self.assertTrue(
                    all(
                        value == 0.0 and math.copysign(1.0, value) == 1.0
                        for value in kernel_storage.buffer
                    )
                )
                self.assertTrue(
                    all(
                        value == 0.0 and math.copysign(1.0, value) == 1.0
                        for value in bias_storage.buffer
                    )
                )

    def test_zero_output_channels_produce_empty_native_results(self):
        for backend in ts.available_backends():
            with self.subTest(backend=backend), ts.use_backend(backend):
                inputs = ts.Tensor([1.0, 2.0, 3.0], shape=(1, 1, 3))
                kernel = ts.Tensor([], shape=(0, 1, 1))
                bias = ts.Tensor([], shape=(0,))
                output = ts.conv1d(inputs, kernel, bias)
                self.assertEqual(output.shape, (1, 0, 3))
                self.assertEqual(output.backend_storage.kind, backend)
                grad = ts.Tensor([], shape=output.shape)
                input_storage, kernel_storage, bias_storage = (
                    backend_dispatch.execute_convolution_gradient(
                        grad,
                        inputs,
                        kernel,
                        stride=(1,),
                        padding=(0,),
                        dilation=(1,),
                        groups=1,
                        include_bias=True,
                        needs_input_grad=(True, True, True),
                    )
                )
                self.assertEqual(input_storage.kind, backend)
                self.assertEqual(kernel_storage.kind, backend)
                self.assertEqual(bias_storage.kind, backend)
                self.assertEqual(list(input_storage.buffer), [0.0, 0.0, 0.0])
                self.assertEqual(list(kernel_storage.buffer), [])
                self.assertEqual(list(bias_storage.buffer), [])
                self.assertTrue(
                    all(
                        value == 0.0 and math.copysign(1.0, value) == 1.0
                        for value in input_storage.buffer
                    )
                )

    def test_input_destinations_without_factors_are_positive_zero(self):
        for backend in ts.available_backends():
            with self.subTest(backend=backend), ts.use_backend(backend):
                inputs = ts.Tensor([1.0, 2.0, 3.0, 4.0], shape=(1, 1, 4))
                kernel = ts.Tensor([1.0], shape=(1, 1, 1))
                grad = ts.Tensor([1.0, 1.0], shape=(1, 1, 2))
                input_storage, _ = backend_dispatch.execute_convolution_gradient(
                    grad,
                    inputs,
                    kernel,
                    stride=(2,),
                    padding=(0,),
                    dilation=(1,),
                    groups=1,
                    include_bias=False,
                    needs_input_grad=(True, False),
                )
                self.assertEqual(list(input_storage.buffer), [1.0, 0.0, 1.0, 0.0])
                for value in list(input_storage.buffer)[1::2]:
                    self.assertEqual(math.copysign(1.0, value), 1.0)

    def test_padded_kernel_vjp_excludes_nonfinite_out_of_bounds_terms(self):
        for backend in ts.available_backends():
            with self.subTest(backend=backend), ts.use_backend(backend):
                inputs = ts.Tensor([1.0], shape=(1, 1, 1))
                kernel = ts.Tensor([1.0], shape=(1, 1, 1))
                grad = ts.Tensor([math.inf, 1.0, math.inf], shape=(1, 1, 3))
                _, kernel_storage = backend_dispatch.execute_convolution_gradient(
                    grad,
                    inputs,
                    kernel,
                    stride=(1,),
                    padding=(1,),
                    dilation=(1,),
                    groups=1,
                    include_bias=False,
                    needs_input_grad=(False, True),
                )
                self.assertEqual(list(kernel_storage.buffer), [1.0])

    def test_requested_storage_decline_fails_the_whole_vjp(self):
        import importlib

        for backend in ts.available_backends():
            if backend == "python":
                continue
            kernel_module = importlib.import_module(
                f"tensors.backend.{backend}.kernels.convolution.convolution_gradient"
            )
            with self.subTest(backend=backend), ts.use_backend(backend):
                inputs = ts.Tensor([1.0, 2.0], shape=(1, 1, 2))
                kernel = ts.Tensor([1.0], shape=(1, 1, 1))
                grad = ts.Tensor([1.0, 1.0], shape=(1, 1, 2))
                with patch.object(
                    kernel_module, "_convolution_storage", return_value=None
                ):
                    with self.assertRaises(BackendOperationUnsupportedError):
                        backend_dispatch.execute_convolution_gradient(
                            grad,
                            inputs,
                            kernel,
                            stride=(1,),
                            padding=(0,),
                            dilation=(1,),
                            groups=1,
                            include_bias=False,
                            needs_input_grad=(True, False),
                        )

    def test_unrequested_vjp_branch_remains_none(self):
        for backend in ts.available_backends():
            if backend == "python":
                continue
            with self.subTest(backend=backend), ts.use_backend(backend):
                inputs = ts.Tensor([1.0, 2.0], shape=(1, 1, 2))
                kernel = ts.Tensor([1.0], shape=(1, 1, 1))
                grad = ts.Tensor([1.0, 1.0], shape=(1, 1, 2))
                input_storage, kernel_storage = (
                    backend_dispatch.execute_convolution_gradient(
                        grad,
                        inputs,
                        kernel,
                        stride=(1,),
                        padding=(0,),
                        dilation=(1,),
                        groups=1,
                        include_bias=False,
                        needs_input_grad=(False, True),
                    )
                )
                self.assertIsNone(input_storage)
                self.assertIsNotNone(kernel_storage)

    def test_bias_only_vjp_does_not_allocate_input_or_kernel_destinations(self):
        import importlib

        for backend in ts.available_backends():
            if backend == "python":
                continue
            gradient_module = importlib.import_module(
                f"tensors.backend.{backend}.kernels.convolution.convolution_gradient"
            )
            provider = getattr(
                gradient_module, "numpy" if backend == "numpy" else "cupy"
            )
            original_zeros = provider.zeros
            with (
                self.subTest(backend=backend),
                patch.object(provider, "zeros", side_effect=original_zeros) as zeros,
                ts.use_backend(backend),
            ):
                inputs = ts.Tensor([1.0, 2.0, 3.0], shape=(1, 1, 3))
                kernel = ts.Tensor([1.0], shape=(1, 1, 1))
                grad = ts.Tensor([1.0, 1.0, 1.0], shape=(1, 1, 3))
                result = backend_dispatch.execute_convolution_gradient(
                    grad,
                    inputs,
                    kernel,
                    stride=(1,),
                    padding=(0,),
                    dilation=(1,),
                    groups=1,
                    include_bias=True,
                    needs_input_grad=(False, False, True),
                )
            allocated_shapes = [call.args[0] for call in zeros.call_args_list]
            self.assertNotIn((1, 1, 3), allocated_shapes)
            self.assertNotIn((1, 1, 1), allocated_shapes)
            self.assertIsNone(result[0])
            self.assertIsNone(result[1])
            self.assertEqual(list(result[2].buffer), [3.0])

    @unittest.skipUnless("numpy" in ts.available_backends(), "NumPy unavailable")
    def test_wrong_backend_storage_and_forced_gradient_decline_raise(self):
        from tensors.backend.loading import load_backend
        from tensors.backend.python.storage import PythonStorage

        numpy_backend = load_backend("numpy")
        with ts.use_backend("numpy"):
            inputs = ts.Tensor([1.0], shape=(1, 1, 1))
            kernel = ts.Tensor([1.0], shape=(1, 1, 1))
            grad = ts.Tensor([1.0], shape=(1, 1, 1))
            foreign = PythonStorage.from_values([1.0], ts.float64)
            with patch.object(numpy_backend, "convolution", return_value=foreign):
                with self.assertRaises(BackendMismatchError):
                    ts.conv1d(inputs, kernel)
            with patch.object(numpy_backend, "convolution_gradient", return_value=None):
                with self.assertRaises(BackendOperationUnsupportedError):
                    backend_dispatch.execute_convolution_gradient(
                        grad,
                        inputs,
                        kernel,
                        stride=(1,),
                        padding=(0,),
                        dilation=(1,),
                        groups=1,
                        include_bias=False,
                        needs_input_grad=(True, False),
                    )

    def test_accelerated_kernels_receive_native_arrays(self):
        import importlib
        from tensors.backend.loading import load_backend

        for backend in ts.available_backends():
            if backend == "python":
                continue
            provider = importlib.import_module(
                "numpy" if backend == "numpy" else "cupy"
            )
            backend_module = load_backend(backend)
            with (
                self.subTest(backend=backend),
                patch.object(
                    backend_module,
                    "convolution",
                    wraps=backend_module.convolution,
                ) as forward,
                patch.object(
                    backend_module,
                    "convolution_gradient",
                    wraps=backend_module.convolution_gradient,
                ) as gradient,
                ts.use_backend(backend),
            ):
                inputs = ts.Tensor([1.0, 2.0], shape=(1, 1, 2))
                kernel = ts.Tensor([1.0], shape=(1, 1, 1))
                output = ts.conv1d(inputs, kernel)
                grad = ts.Tensor([1.0, 1.0], shape=output.shape)
                backend_dispatch.execute_convolution_gradient(
                    grad,
                    inputs,
                    kernel,
                    stride=(1,),
                    padding=(0,),
                    dilation=(1,),
                    groups=1,
                    include_bias=False,
                    needs_input_grad=(True, True),
                )
            for value in forward.call_args.args[:3]:
                if value is not None:
                    self.assertIsInstance(value, provider.ndarray)
            for value in gradient.call_args.args[:3]:
                self.assertIsInstance(value, provider.ndarray)

    def test_small_budget_exercises_all_certification_tiling(self):
        import importlib

        for backend in ts.available_backends():
            if backend == "python":
                continue
            common = importlib.import_module(
                f"tensors.backend.{backend}.kernels.convolution.common"
            )
            forward_module = importlib.import_module(
                f"tensors.backend.{backend}.kernels.convolution.convolution"
            )
            gradient_module = importlib.import_module(
                f"tensors.backend.{backend}.kernels.convolution.convolution_gradient"
            )
            with (
                self.subTest(backend=backend),
                patch.object(common, "_CONVOLUTION_COLUMN_MAX_ELEMENTS", 4),
                ts.use_backend(backend),
            ):
                inputs = ts.Tensor([1.0, 1.0], shape=(1, 1, 2))
                kernel = ts.Tensor([1.0] * 4, shape=(4, 1, 1))
                with patch.object(
                    forward_module,
                    "pairwise_matmul",
                    wraps=forward_module.pairwise_matmul,
                ) as certified:
                    output = ts.conv1d(inputs, kernel)
                self.assertGreaterEqual(certified.call_count, 2)
                grad = ts.Tensor([1.0] * 8, shape=output.shape)
                with patch.object(
                    gradient_module,
                    "pairwise_float_sum",
                    wraps=gradient_module.pairwise_float_sum,
                ) as certified:
                    backend_dispatch.execute_convolution_gradient(
                        grad,
                        inputs,
                        kernel,
                        stride=(1,),
                        padding=(0,),
                        dilation=(1,),
                        groups=1,
                        include_bias=False,
                        needs_input_grad=(True, False),
                    )
                self.assertGreaterEqual(certified.call_count, 2)
                with patch.object(
                    gradient_module,
                    "pairwise_float_sum",
                    wraps=gradient_module.pairwise_float_sum,
                ) as certified:
                    backend_dispatch.execute_convolution_gradient(
                        grad,
                        inputs,
                        kernel,
                        stride=(1,),
                        padding=(0,),
                        dilation=(1,),
                        groups=1,
                        include_bias=False,
                        needs_input_grad=(False, True),
                    )
                self.assertGreaterEqual(certified.call_count, 2)

    def test_hostile_finite_forward_cancellation_follows_the_pairwise_tree(self):
        """The products ``[1e16, 1, -1e16]`` are summed pairwise.

        ``fl(1e16 + 1)`` is ``1e16``, because the spacing of doubles at
        ``1e16`` is 2, and adding ``-1e16`` then gives exactly zero. The exact
        sum is 1; the specified result is 0 on every backend.
        """
        self.assertEqual(pairwise_dot([1e16, 1.0, -1e16], [1.0, 1.0, 1.0]), 0.0)
        for backend in ts.available_backends():
            with self.subTest(backend=backend), ts.use_backend(backend):
                result = ts.conv1d(
                    ts.Tensor([1e16, 1.0, -1e16], shape=(1, 1, 3)),
                    ts.Tensor([1.0, 1.0, 1.0], shape=(1, 1, 3)),
                )
                self.assertEqual(result.tolist(), [0.0])
                self.assertEqual(result.backend_storage.kind, backend)

    def test_hostile_vjp_cancellation_follows_the_pairwise_tree(self):
        """Each VJP element is a fixed term sequence summed pairwise.

        Input VJP, padding 1, unit upstream: position ``c`` runs over kernel
        offsets 0, 1, 2 and each offset reaches output ``c + 1 - offset`` or
        contributes an exact zero term. The sequences are
        ``[1e16, 1, 0]``, ``[1e16, 1, -1e16]`` and ``[0, 1, -1e16]``, which
        reduce to ``1e16``, ``0`` and ``-1e16``.

        Kernel VJP: the one weight's gradient runs over the three positions,
        ``[1e16, 1, -1e16]``, which reduces to ``0``.
        """
        cases = (
            (
                [0.0, 0.0, 0.0],
                (1, 1, 3),
                [1e16, 1.0, -1e16],
                (1, 1, 3),
                [1.0] * 3,
                (1, 1, 3),
                (True, False),
                (1,),
                [1e16, 0.0, -1e16],
            ),
            (
                [1e16, 1.0, -1e16],
                (1, 1, 3),
                [1.0],
                (1, 1, 1),
                [1.0] * 3,
                (1, 1, 3),
                (False, True),
                (0,),
                [0.0],
            ),
        )
        for case in cases:
            (
                input_values,
                input_shape,
                kernel_values,
                kernel_shape,
                grad_values,
                grad_shape,
                needs,
                padding,
                expected,
            ) = case
            for backend in ts.available_backends():
                with (
                    self.subTest(backend=backend, needs=needs),
                    ts.use_backend(backend),
                ):
                    inputs = ts.Tensor(input_values, shape=input_shape)
                    kernel = ts.Tensor(kernel_values, shape=kernel_shape)
                    grad = ts.Tensor(grad_values, shape=grad_shape)
                    result = backend_dispatch.execute_convolution_gradient(
                        grad,
                        inputs,
                        kernel,
                        stride=(1,),
                        padding=padding,
                        dilation=(1,),
                        groups=1,
                        include_bias=False,
                        needs_input_grad=needs,
                    )
                    storage = result[0] if needs[0] else result[1]
                    self.assertEqual(storage.kind, backend)
                    self.assertEqual(list(storage.buffer), expected)

    def test_hostile_bias_vjp_reduction_follows_the_pairwise_tree(self):
        """The bias gradient is the upstream ``[1e16, 1, -1e16]`` summed pairwise.

        ``fl(1e16 + 1) + (-1e16)`` is exactly zero on every backend.
        """
        self.assertEqual(pairwise_sum([1e16, 1.0, -1e16]), 0.0)
        for backend in ts.available_backends():
            with self.subTest(backend=backend), ts.use_backend(backend):
                inputs = ts.Tensor([0.0, 0.0, 0.0], shape=(1, 1, 3))
                kernel = ts.Tensor([1.0], shape=(1, 1, 1))
                grad = ts.Tensor([1e16, 1.0, -1e16], shape=(1, 1, 3))
                result = backend_dispatch.execute_convolution_gradient(
                    grad,
                    inputs,
                    kernel,
                    stride=(1,),
                    padding=(0,),
                    dilation=(1,),
                    groups=1,
                    include_bias=True,
                    needs_input_grad=(False, False, True),
                )
                self.assertEqual(list(result[2].buffer), [0.0])
                self.assertEqual(result[2].kind, backend)

    def test_product_underflow_is_rounded_like_any_product(self):
        """``fl(1e-300 * 1e-300)`` underflows to zero on every backend.

        The product is formed once in the dtype; it is a result, not a
        reason to refuse the convolution.
        """
        for backend in ts.available_backends():
            with self.subTest(backend=backend), ts.use_backend(backend):
                inputs = ts.Tensor([1e-300], shape=(1, 1, 1))
                kernel = ts.Tensor([1e-300], shape=(1, 1, 1))
                result = ts.conv1d(inputs, kernel)
                self.assertEqual(result.tolist(), [0.0])
                self.assertEqual(result.backend_storage.kind, backend)

    def test_nonfinite_padded_kernel_declines_instead_of_using_padding_as_data(self):
        for backend in ts.available_backends():
            if backend == "python":
                continue
            with self.subTest(backend=backend), ts.use_backend(backend):
                inputs = ts.Tensor([1.0], shape=(1, 1, 1))
                kernel = ts.Tensor([math.inf], shape=(1, 1, 1))
                with self.assertRaises(BackendOperationUnsupportedError):
                    ts.conv1d(inputs, kernel, padding=1)

    @unittest.skipUnless("numpy" in ts.available_backends(), "NumPy unavailable")
    def test_accelerated_integer_convolution_is_explicitly_unsupported(self):
        with ts.use_backend("numpy"):
            inputs = ts.Tensor([1, 2], dtype=ts.int64, shape=(1, 1, 2))
            kernel = ts.Tensor([1], dtype=ts.int64, shape=(1, 1, 1))
            with self.assertRaises(BackendOperationUnsupportedError):
                ts.conv1d(inputs, kernel)


if __name__ == "__main__":
    unittest.main()
