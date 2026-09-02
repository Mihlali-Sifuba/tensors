import unittest
from unittest.mock import patch

import tensors as ts
import tensors.backend.numpy as numpy_backend

from ._support import NumPyParityTestCase, requires_numpy


@requires_numpy
class NumPyElementwiseTests(NumPyParityTestCase):
    """Binary, unary, and scalar kernels dispatch and match Python."""

    def test_every_binary_operation_dispatches_to_numpy(self):
        left = ts.full((32, 1), 2.0)
        right = ts.full((1, 32), 3.0)

        with patch.object(
            numpy_backend,
            "binary",
            wraps=numpy_backend.binary,
        ) as binary:
            with ts.use_backend("numpy"):
                _ = left + right
                _ = left - right
                _ = left * right
                _ = left / right
                _ = left ** right
                _ = 2.0 / left
                _ = 2.0 ** left

        self.assertEqual(
            [call.args[0] for call in binary.call_args_list],
            [
                "add",
                "subtract",
                "multiply",
                "divide",
                "power",
                "divide",
                "power",
            ],
        )
    def test_negation_dispatches_to_numpy(self):
        with patch.object(
            numpy_backend,
            "negate",
            wraps=numpy_backend.negate,
        ) as negate:
            self._evaluate("numpy", lambda: -ts.full((64,), 2.0))

        negate.assert_called_once()
    def test_every_unary_operation_dispatches_to_numpy(self):
        operations = (
            ("abs", ts.abs, 0.5),
            ("sqrt", ts.sqrt, 0.5),
            ("exp", ts.exp, 0.5),
            ("log", ts.log, 0.5),
            ("sin", ts.sin, 0.5),
            ("cos", ts.cos, 0.5),
            ("tan", ts.tan, 0.5),
            ("arcsin", ts.arcsin, 0.5),
            ("arccos", ts.arccos, 0.5),
            ("arctan", ts.arctan, 0.5),
            ("sinh", ts.sinh, 0.5),
            ("cosh", ts.cosh, 0.5),
            ("arcsinh", ts.arcsinh, 0.5),
            ("arccosh", ts.arccosh, 1.5),
            ("arctanh", ts.arctanh, 0.5),
            ("sign", ts.sign, 0.5),
            ("relu", ts.relu, 0.5),
            ("sigmoid", ts.sigmoid, 0.5),
            ("tanh", ts.tanh, 0.5),
            ("softplus", ts.softplus, 0.5),
        )
        with (
            patch.object(
                numpy_backend,
                "unary",
                wraps=numpy_backend.unary,
            ) as unary,
            patch.object(
                numpy_backend,
                "unary_gradient",
                wraps=numpy_backend.unary_gradient,
            ) as unary_gradient,
        ):
            with ts.use_backend("numpy"):
                for _, function, item in operations:
                    value = ts.Variable(ts.full((64,), item))
                    output = function(value)
                    ts.grad(ts.sum(output), value)

        expected = [name for name, _, _ in operations]
        self.assertEqual(
            [call.args[0] for call in unary.call_args_list],
            expected,
        )
        self.assertEqual(
            [call.args[0] for call in unary_gradient.call_args_list],
            expected,
        )
    def test_unary_operations_and_gradients_match_python_backend(self):
        operations = (
            (ts.abs, 0.5),
            (ts.sqrt, 0.5),
            (ts.exp, 0.5),
            (ts.log, 0.5),
            (ts.sin, 0.5),
            (ts.cos, 0.5),
            (ts.tan, 0.5),
            (ts.arcsin, 0.5),
            (ts.arccos, 0.5),
            (ts.arctan, 0.5),
            (ts.sinh, 0.5),
            (ts.cosh, 0.5),
            (ts.arcsinh, 0.5),
            (ts.arccosh, 1.5),
            (ts.arctanh, 0.5),
            (ts.sign, 0.5),
            (ts.relu, 0.5),
            (ts.sigmoid, 0.5),
            (ts.tanh, 0.5),
            (ts.softplus, 0.5),
        )

        def evaluate(backend, function, item):
            with ts.use_backend(backend):
                value = ts.Variable(ts.full((64,), item))
                output = function(value)
                gradient = ts.grad(ts.sum(output), value)
                return output.data, gradient

        for function, item in operations:
            with self.subTest(operation=function.__name__):
                expected = evaluate("python", function, item)
                actual = evaluate("numpy", function, item)
                for actual_tensor, expected_tensor in zip(actual, expected):
                    self.assertEqual(actual_tensor.shape, expected_tensor.shape)
                    self.assertIs(actual_tensor.dtype, expected_tensor.dtype)
                    for actual_item, expected_item in zip(
                        actual_tensor._data,
                        expected_tensor._data,
                    ):
                        self.assertAlmostEqual(actual_item, expected_item)
    def test_unary_kernels_preserve_domain_errors(self):
        with ts.use_backend("numpy"):
            with self.assertRaisesRegex(
                ValueError,
                "sqrt is only defined for non-negative values",
            ):
                ts.sqrt(ts.full((64,), -1.0))
            value = ts.Variable(ts.full((64,), 1.0))
            with self.assertRaisesRegex(
                ValueError,
                "arcsin derivative is undefined at -1 and 1",
            ):
                ts.grad(ts.sum(ts.arcsin(value)), value)
    def test_broadcast_arithmetic_matches_python_backend(self):
        left = ts.Tensor([[1.5], [2.5]])
        right = ts.Tensor([[3.0, 4.0]])

        for operation in (
            lambda: left + right,
            lambda: left - right,
            lambda: left * right,
            lambda: left / right,
            lambda: left ** right,
        ):
            with self.subTest(operation=operation):
                self.assertOperationParity(operation)
    def test_reverse_scalar_operations_match_python_backend(self):
        value = ts.Tensor([1.0, 2.0, 4.0])

        self.assertOperationParity(lambda: 8.0 / value)
        self.assertOperationParity(lambda: 2.0 ** value)
    def test_numpy_division_validates_tensor_denominators_in_kernel(self):
        numerator = ts.Tensor([1.0] * 512)
        denominator = ts.Tensor([2.0] * 511 + [0.0])

        with patch.object(
            numpy_backend,
            "binary",
            wraps=numpy_backend.binary,
        ) as binary:
            with ts.use_backend("numpy"):
                with self.assertRaisesRegex(ZeroDivisionError, "Division by zero"):
                    ts.divide(numerator, denominator)

        binary.assert_called_once()
    def test_integer_arithmetic_and_unsigned_negation_match(self):
        left = ts.Tensor([1, 2, 3], dtype=ts.int32)
        right = ts.Tensor([4, 5, 6], dtype=ts.int32)
        unsigned = ts.Tensor([1, 255], dtype=ts.uint8)

        self.assertOperationParity(lambda: left + right)
        self.assertOperationParity(lambda: left * right)
        self.assertOperationParity(lambda: left ** 3)
        self.assertOperationParity(lambda: -unsigned)


if __name__ == "__main__":
    unittest.main()
