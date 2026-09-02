import unittest

import tensors as ts
from tensors.graph import Computation
from tensors.graph.state import reset_graph_state


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
        self.assertEqual(
            result.tolist(),
            [1.0, 2.0, 0.0, 3.0, 5.0, 2.0, 0.0, 3.0, 4.0],
        )

    def test_conv2d_accepts_per_axis_stride_padding_and_dilation(self):
        inputs = ts.Tensor([[[[float(value) for value in range(6)]] * 5]])
        kernel = ts.Tensor([[[[1.0, 1.0]]]])

        result = ts.conv2d(
            inputs,
            kernel,
            stride=(2, 1),
            padding=(1, 0),
            dilation=(1, 2),
        )

        self.assertEqual(result.shape, (1, 1, 4, 4))

    def test_conv2d_output_extent_follows_the_documented_formula(self):
        inputs = ts.zeros((2, 3, 9, 11))
        kernel = ts.zeros((4, 3, 3, 2))

        result = ts.conv2d(
            inputs, kernel, stride=2, padding=1, dilation=2
        )

        # (9 + 2 - 2 * 2 - 1) // 2 + 1 = 4, (11 + 2 - 2 - 1) // 2 + 1 = 6
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

        self.assertEqual(result.node.label, "conv2d")
        self.assertEqual(
            result.node.inputs,
            [inputs.node, kernel.node, bias.node],
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
        inputs = ts.Tensor(
            [[[[0.4, -0.2, 0.7], [1.1, 0.3, -0.9], [0.5, 0.8, -0.1]]]]
        )
        kernel = ts.Tensor([[[[0.6, -0.3], [0.2, 0.9]]]])
        bias = ts.Tensor([0.25])

        ts.gradcheck(
            lambda x, w, b: ts.conv2d(x, w, b, padding=1),
            [inputs, kernel, bias],
        )

    def test_convolution_builds_differentiable_higher_order_gradients(self):
        inputs = ts.Variable([[[1.0, 2.0, 3.0]]])
        kernel = ts.Variable([[[1.0, -1.0]]])

        output = ts.sum(ts.conv1d(inputs, kernel))
        input_gradient, kernel_gradient = ts.grad(
            output,
            (inputs, kernel),
            create_graph=True,
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

        ts.gradcheck(
            lambda x, w: ts.conv2d(x, w, groups=2),
            [inputs, kernel],
        )


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
        self.assertEqual(result.node.label, "conv1d")

        ts.backward(result)

        self.assertEqual(inputs.grad.tolist(), [1.0, 0.0, -1.0])
        self.assertEqual(kernel.grad.tolist(), [3.0, 5.0])

    def test_conv1d_gradients_match_finite_differences(self):
        inputs = ts.Tensor([[[0.3, -0.6, 1.2, 0.4], [0.9, 0.1, -0.5, 0.7]]])
        kernel = ts.Tensor([[[0.5, -0.8]], [[0.2, 0.6]]])

        ts.gradcheck(
            lambda x, w: ts.conv1d(x, w, padding=1, groups=2),
            [inputs, kernel],
        )


class Conv3dAndUnbatchedTests(unittest.TestCase):
    def test_conv1d_accepts_an_unbatched_signal(self):
        result = ts.conv1d(
            ts.Tensor([[1.0, 2.0, 3.0]]),
            ts.Tensor([[[1.0, -1.0]]]),
        )

        self.assertEqual(result.shape, (1, 2))
        self.assertEqual(result.tolist(), [-1.0, -1.0])

    def test_conv2d_accepts_an_unbatched_image(self):
        inputs, kernel = _identity_case()

        result = ts.conv2d(
            ts.reshape(inputs, (1, 2, 2)),
            kernel,
        )

        self.assertEqual(result.shape, (1, 1, 1))
        self.assertEqual(result.tolist(), [5.0])

    def test_conv3d_correlates_a_volume(self):
        inputs = ts.Tensor(
            [[[[[1.0, 2.0], [3.0, 4.0]],
               [[5.0, 6.0], [7.0, 8.0]]]]]
        )
        kernel = ts.ones((1, 1, 2, 2, 2))

        result = ts.conv3d(inputs, kernel, ts.Tensor([1.0]))

        self.assertEqual(result.shape, (1, 1, 1, 1, 1))
        self.assertEqual(result.tolist(), [37.0])
        self.assertEqual(ts.math.conv3d(inputs, kernel).tolist(), [36.0])

    def test_conv3d_accepts_unbatched_grouped_volumes(self):
        inputs = ts.Tensor([
            [[[1.0, 2.0]]],
            [[[3.0, 4.0]]],
        ])
        kernel = ts.Tensor([
            [[[[1.0, 1.0]]]],
            [[[[2.0, 2.0]]]],
        ])

        result = ts.conv3d(inputs, kernel, groups=2)

        self.assertEqual(result.shape, (2, 1, 1, 1))
        self.assertEqual(result.tolist(), [3.0, 14.0])

    def test_unbatched_conv3d_gradients_match_finite_differences(self):
        inputs = ts.Tensor([
            [[[0.2, -0.4], [0.5, 0.7]],
             [[0.1, 0.3], [-0.2, 0.8]]],
        ])
        kernel = ts.Tensor([[[[[0.6, -0.1], [0.2, 0.4]],
                              [[-0.3, 0.5], [0.7, -0.2]]]]])

        ts.gradcheck(
            lambda x, w: ts.conv3d(x, w, padding=1),
            [inputs, kernel],
        )

    def test_conv3d_replays_through_a_recorded_computation(self):
        inputs = ts.Variable(ts.ones((1, 1, 2, 2, 2)))
        kernel = ts.Variable(ts.ones((1, 1, 2, 2, 2)))
        result = ts.conv3d(inputs, kernel)

        self.assertEqual(result.node.label, "conv3d")
        self.assertEqual(Computation(result).forward().tolist(), [8.0])


class ConvolutionExactnessTests(unittest.TestCase):
    def test_integer_convolution_keeps_exact_python_intermediates(self):
        inputs = ts.Tensor([[[2 ** 60, 2 ** 60 + 1]]], dtype=ts.int64)
        kernel = ts.Tensor([[[1, 1]]], dtype=ts.int64)

        result = ts.conv1d(inputs, kernel)

        self.assertIs(result.dtype, ts.int64)
        self.assertEqual(result.tolist(), [2 ** 61 + 1])

    def test_convolution_recovers_from_a_temporary_product_overflow(self):
        inputs = ts.Tensor([[[1e300, 1e300]]])
        kernel = ts.Tensor([[[1e300, -1e300]]])

        result = ts.conv1d(inputs, kernel)

        self.assertEqual(result.tolist(), [0.0])


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
            ts.conv2d(
                ts.zeros((1, 1, 2, 2)),
                ts.zeros((3, 1, 2, 2)),
                ts.zeros((2,)),
            )

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
            ts.conv2d(
                ts.zeros((1, 1, 4, 4)),
                ts.zeros((1, 1, 2, 2)),
                stride=(1, 1, 1),
            )

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
    len(ts.available_backends()) > 1,
    "No accelerated backend is installed",
)
class ConvolutionBackendParityTests(unittest.TestCase):
    """Every installed backend must agree with the Python reference."""

    CASES = (
        # inputs shape, kernel shape, bias, stride, padding, dilation, groups
        ((2, 3, 6, 7), (4, 3, 3, 2), True, 1, 0, 1, 1),
        ((2, 4, 8, 8), (6, 2, 3, 3), True, 2, 1, 1, 2),
        ((1, 4, 9, 9), (4, 1, 3, 3), True, 1, 2, 2, 4),
        ((3, 2, 7, 5), (5, 2, 2, 3), True, (2, 1), (1, 2), (2, 1), 1),
    )

    @staticmethod
    def _ramp(shape, scale):
        size = ts.Shape.from_iterable(shape).size
        values = [
            scale * (((index * 37) % 19) - 9) / 7.0 for index in range(size)
        ]
        return ts.Tensor(values, shape=shape)

    def test_forward_matches_the_python_reference(self):
        for case in self.CASES:
            shape, kernel_shape, use_bias, stride, padding, dilation, groups = case
            with self.subTest(case=case):
                inputs = self._ramp(shape, 0.5)
                kernel = self._ramp(kernel_shape, 0.25)
                bias = self._ramp((kernel_shape[0],), 1.0) if use_bias else None
                options = dict(
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )

                with ts.use_backend("python"):
                    expected = ts.conv2d(inputs, kernel, bias, **options)

                for backend in ts.available_backends():
                    with ts.use_backend(backend):
                        result = ts.conv2d(inputs, kernel, bias, **options)
                    self.assertEqual(result.shape, expected.shape)
                    for actual, want in zip(
                        result.tolist(), expected.tolist()
                    ):
                        self.assertAlmostEqual(actual, want, places=10)

    def test_gradients_match_the_python_reference(self):
        for case in self.CASES:
            shape, kernel_shape, use_bias, stride, padding, dilation, groups = case
            with self.subTest(case=case):
                options = dict(
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
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
                        return [
                            operand.grad.tolist() for operand in operands
                        ]

                expected = gradients("python")
                for backend in ts.available_backends():
                    for actual, want in zip(gradients(backend), expected):
                        for value, target in zip(actual, want):
                            self.assertAlmostEqual(value, target, places=10)

    def test_unbatched_and_three_dimensional_forward_paths_match_python(self):
        cases = (
            (
                ts.conv1d,
                (2, 8),
                (3, 2, 3),
                dict(stride=2, padding=1),
            ),
            (
                ts.conv2d,
                (2, 5, 6),
                (4, 2, 2, 3),
                dict(stride=(2, 1), padding=(1, 0)),
            ),
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
                inputs = self._ramp(shape, 0.5)
                kernel = self._ramp(kernel_shape, 0.25)
                with ts.use_backend("python"):
                    expected = operation(inputs, kernel, **options)

                for backend in ts.available_backends():
                    with ts.use_backend(backend):
                        result = operation(inputs, kernel, **options)
                    self.assertEqual(result.shape, expected.shape)
                    for actual, want in zip(
                        result.tolist(),
                        expected.tolist(),
                    ):
                        self.assertAlmostEqual(actual, want, places=10)

    def test_float32_convolution_stays_float32_on_accelerated_backends(self):
        inputs = ts.Tensor(
            [float(index % 13) / 7.0 for index in range(2 * 3 * 8 * 8)],
            dtype=ts.float32,
            shape=(2, 3, 8, 8),
        )
        kernel = ts.Tensor(
            [float(index % 11) / 9.0 for index in range(4 * 3 * 3 * 3)],
            dtype=ts.float32,
            shape=(4, 3, 3, 3),
        )
        with ts.use_backend("python"):
            expected = ts.conv2d(inputs, kernel, padding=1)

        for backend in ts.available_backends():
            with ts.use_backend(backend):
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
                return inputs.grad.tolist(), kernel.grad.tolist()

        expected = gradients("python")
        for backend in ts.available_backends():
            for actual, want in zip(gradients(backend), expected):
                for value, target in zip(actual, want):
                    self.assertAlmostEqual(value, target, places=10)


    def test_accelerated_convolution_matches_when_forced_across_small_tiles(self):
        from tensors.backend import _array

        inputs = self._ramp((2, 3, 7, 8), 0.5)
        kernel = self._ramp((4, 3, 3, 2), 0.25)
        options = dict(stride=(2, 1), padding=(1, 2), dilation=(2, 1))
        with ts.use_backend("python"):
            expected = ts.conv2d(inputs, kernel, **options)

        original_limit = _array._CONVOLUTION_COLUMN_MAX_ELEMENTS
        _array._CONVOLUTION_COLUMN_MAX_ELEMENTS = 64
        try:
            for backend in ts.available_backends():
                if backend == "python":
                    continue
                with ts.use_backend(backend):
                    result = ts.conv2d(inputs, kernel, **options)
                self.assertEqual(result.shape, expected.shape)
                for actual, want in zip(result.tolist(), expected.tolist()):
                    self.assertAlmostEqual(actual, want, places=10)
        finally:
            _array._CONVOLUTION_COLUMN_MAX_ELEMENTS = original_limit



if __name__ == "__main__":
    unittest.main()
