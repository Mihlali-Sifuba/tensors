import unittest
from unittest.mock import patch

import tensors as ts
import tensors.backend.numpy as numpy_backend

from ._support import NumPyParityTestCase, requires_numpy


@requires_numpy
class NumPyLayoutTests(NumPyParityTestCase):
    """Construction, layout, slicing, and cast kernels."""

    def test_creation_layout_and_optimizers_dispatch_to_numpy(self):
        with (
            patch.object(
                numpy_backend,
                "full",
                wraps=numpy_backend.full,
            ) as full,
            patch.object(
                numpy_backend,
                "eye",
                wraps=numpy_backend.eye,
            ) as eye,
            patch.object(
                numpy_backend,
                "arange",
                wraps=numpy_backend.arange,
            ) as arange,
            patch.object(
                numpy_backend,
                "linspace",
                wraps=numpy_backend.linspace,
            ) as linspace,
            patch.object(
                numpy_backend,
                "transpose",
                wraps=numpy_backend.transpose,
            ) as transpose,
            patch.object(
                numpy_backend,
                "concat",
                wraps=numpy_backend.concat,
            ) as concat,
            patch.object(
                numpy_backend,
                "stack",
                wraps=numpy_backend.stack,
            ) as stack,
            patch.object(
                numpy_backend,
                "outer",
                wraps=numpy_backend.outer,
            ) as outer,
            patch.object(
                numpy_backend,
                "outer_gradient",
                wraps=numpy_backend.outer_gradient,
            ) as outer_gradient,
            patch.object(
                numpy_backend,
                "sgd_update",
                wraps=numpy_backend.sgd_update,
            ) as sgd_update,
            patch.object(
                numpy_backend,
                "adam_update",
                wraps=numpy_backend.adam_update,
            ) as adam_update,
            patch.object(
                numpy_backend,
                "rmsprop_update",
                wraps=numpy_backend.rmsprop_update,
            ) as rmsprop_update,
        ):
            with ts.use_backend("numpy"):
                ts.full((64,), 2.0)
                ts.eye(8)
                ts.arange(64)
                ts.linspace(1.0, 2.0, 64)

                matrix = ts.full((8, 8), 2.0)
                ts.transpose(matrix)
                ts.concat([matrix, matrix], axis=0)
                ts.stack([matrix, matrix], axis=0)

                left = ts.Variable(ts.full((8,), 2.0))
                right = ts.Variable(ts.full((8,), 3.0))
                product = ts.outer(left, right)
                ts.grad(
                    product,
                    (left, right),
                    grad_outputs=ts.full((8, 8), 1.0),
                )

                for optimizer_type in (
                    ts.optim.SGD,
                    ts.optim.Adam,
                    ts.optim.RMSprop,
                ):
                    parameter = ts.Variable(ts.full((64,), 1.0))
                    parameter.grad = ts.full((64,), 1.0)
                    optimizer_type([parameter], learning_rate=0.1).step()

        self.assertGreaterEqual(full.call_count, 1)
        eye.assert_called_once()
        arange.assert_called_once()
        linspace.assert_called_once()
        transpose.assert_called_once()
        concat.assert_called_once()
        stack.assert_called_once()
        outer.assert_called_once()
        outer_gradient.assert_called_once()
        sgd_update.assert_called_once()
        adam_update.assert_called_once()
        rmsprop_update.assert_called_once()
    def test_creation_layout_and_optimizer_results_match_python_backend(self):
        def evaluate(backend):
            with ts.use_backend(backend):
                matrix = ts.reshape(ts.arange(64), (8, 8))
                creations = (
                    ts.full((64,), 1.25),
                    ts.eye(8, k=1),
                    ts.arange(0.0, 6.4, 0.1),
                    ts.linspace(1.0, 2.0, 64),
                )
                layouts = (
                    ts.transpose(matrix),
                    ts.concat([matrix, matrix], axis=1),
                    ts.stack([matrix, matrix], axis=1),
                    ts.outer(ts.ones((8,)), ts.arange(8)),
                )
                parameters = []
                for optimizer_type in (
                    ts.optim.SGD,
                    ts.optim.Adam,
                    ts.optim.RMSprop,
                ):
                    parameter = ts.Variable(ts.full((64,), 1.0))
                    parameter.grad = ts.full((64,), 0.5)
                    optimizer_type([parameter], learning_rate=0.1).step()
                    parameters.append(parameter.data)
                return creations + layouts + tuple(parameters)

        expected = evaluate("python")
        actual = evaluate("numpy")
        for actual_tensor, expected_tensor in zip(actual, expected):
            self.assertEqual(actual_tensor.shape, expected_tensor.shape)
            self.assertIs(actual_tensor.dtype, expected_tensor.dtype)
            for actual_item, expected_item in zip(
                actual_tensor._data,
                expected_tensor._data,
            ):
                self.assertAlmostEqual(actual_item, expected_item)
    def test_slice_dispatches_to_numpy(self):
        value = ts.Variable(ts.full((64, 2), 2.0))
        with patch.object(
            numpy_backend,
            "slice_tensor",
            wraps=numpy_backend.slice_tensor,
        ) as slice_tensor:
            self._evaluate("numpy", lambda: value[:, 1])

        slice_tensor.assert_called_once()
    def test_slice_scatter_dispatches_to_numpy(self):
        with patch.object(
            numpy_backend,
            "slice_scatter",
            wraps=numpy_backend.slice_scatter,
        ) as slice_scatter:
            with ts.use_backend("numpy"):
                value = ts.Variable(ts.full((128,), 2.0))
                output = ts.sum(value[::2])
                ts.grad(output, value, create_graph=True)

        slice_scatter.assert_called_once()
    def test_cast_dispatches_to_numpy(self):
        value = ts.full((64,), 1.25)
        with patch.object(
            numpy_backend,
            "cast_tensor",
            wraps=numpy_backend.cast_tensor,
        ) as cast_tensor:
            self._evaluate("numpy", lambda: value.astype(ts.int32))

        cast_tensor.assert_called_once()
    def test_slice_and_cast_match_python_backend(self):
        value = ts.Variable([[1.25, 2.75], [3.5, 4.125]])

        self.assertOperationParity(lambda: value[:, 1])
        self.assertOperationParity(
            lambda: value.astype(ts.float32),
        )


if __name__ == "__main__":
    unittest.main()
