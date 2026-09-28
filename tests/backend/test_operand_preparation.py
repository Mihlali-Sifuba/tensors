"""Arithmetic kernels receive lowered values and perform numerical work only."""

import unittest
from array import array
from itertools import repeat

import tensors as ts
from tensors.backend.python.kernels.arithmetic.add import add
from tensors.backend.python.kernels.arithmetic.divide import divide
from tensors.backend.python.kernels.arithmetic.multiply import multiply
from tensors.backend.python.kernels.arithmetic.subtract import subtract
from tensors.backend.python.storage import PythonStorage
from tests.backend._support import BackendTestCase, requires_numpy


class PythonKernelExecutionTests(BackendTestCase):
    """A kernel evaluates prepared operands. It knows nothing about Tensors."""

    SHAPE = (3,)

    def _values(self, storage):
        return list(storage.buffer)

    def test_add_evaluates_prepared_pairs(self):
        storage = add(
            [1.0, 2.0, 3.0], [10.0, 20.0, 30.0],
            dtype=ts.float64, output_shape=self.SHAPE,
        )
        self.assertIsInstance(storage, PythonStorage)
        self.assertEqual(self._values(storage), [11.0, 22.0, 33.0])

    def test_subtract_multiply_and_divide_evaluate_prepared_pairs(self):
        left, right = [6.0, 8.0, 10.0], [2.0, 4.0, 5.0]
        self.assertEqual(
            self._values(
                subtract(left, right, dtype=ts.float64, output_shape=self.SHAPE)
            ),
            [4.0, 4.0, 5.0],
        )
        self.assertEqual(
            self._values(
                multiply(left, right, dtype=ts.float64, output_shape=self.SHAPE)
            ),
            [12.0, 32.0, 50.0],
        )
        self.assertEqual(
            self._values(
                divide(left, right, dtype=ts.float64, output_shape=self.SHAPE)
            ),
            [3.0, 2.0, 2.0],
        )

    def test_a_kernel_accepts_any_iterable_of_values(self):
        """Prepared operands are values; their container is not the contract."""
        storage = add(
            array("d", [1.0, 2.0, 3.0]),
            repeat(1.0),
            dtype=ts.float64,
            output_shape=self.SHAPE,
        )
        self.assertEqual(self._values(storage), [2.0, 3.0, 4.0])

    def test_a_kernel_applies_the_declared_dtype_to_its_result(self):
        storage = add(
            [1, 2, 3], [10, 20, 30], dtype=ts.int32, output_shape=self.SHAPE
        )
        self.assertIs(storage.dtype, ts.int32)
        self.assertEqual(self._values(storage), [11, 22, 33])

    def test_a_kernel_wraps_an_integer_result_at_the_declared_width(self):
        """Section 4.2: arithmetic wraps, and the kernel is where it happens."""
        storage = add([127], [1], dtype=ts.int8, output_shape=(1,))
        self.assertIs(storage.dtype, ts.int8)
        self.assertEqual(self._values(storage), [-128])

    def test_division_by_zero_delivers_the_ieee_value(self):
        """Section 7.2: floating division never raises."""
        storage = divide(
            [1.0, -1.0, 0.0], [0.0, 0.0, 0.0],
            dtype=ts.float64, output_shape=self.SHAPE,
        )
        values = self._values(storage)
        self.assertEqual(values[0], float("inf"))
        self.assertEqual(values[1], float("-inf"))
        self.assertNotEqual(values[2], values[2])  # NaN

    def test_a_kernel_does_not_broadcast(self):
        """Mismatched lengths are a preparation failure, never a kernel one."""
        storage = add(
            [1.0, 2.0, 3.0], [10.0], dtype=ts.float64, output_shape=(1,)
        )
        # zip stops at the shorter operand: the kernel pairs, it does not expand.
        self.assertEqual(self._values(storage), [11.0])


class ArrayKernelExecutionTests(BackendTestCase):
    """The array kernels evaluate prepared arrays and return native storage."""

    @requires_numpy
    def test_the_numpy_kernel_evaluates_prepared_arrays(self):
        import numpy

        from tensors.backend.numpy.kernels.arithmetic.add import add as numpy_add
        from tensors.backend.numpy.storage import NumPyStorage

        left = numpy.asarray([1.0, 2.0], dtype=numpy.float64)
        right = numpy.asarray([10.0, 20.0], dtype=numpy.float64)

        with ts.use_backend("numpy"):
            storage = numpy_add(left, right, dtype=ts.float64, output_shape=(2,))

        self.assertIsInstance(storage, NumPyStorage)
        self.assertIs(storage.dtype, ts.float64)
        self.assertEqual(list(storage.buffer), [11.0, 22.0])

    @requires_numpy
    def test_the_numpy_kernel_lets_the_array_api_broadcast(self):
        import numpy

        from tensors.backend.numpy.kernels.arithmetic.add import add as numpy_add

        left = numpy.asarray([[1.0], [2.0]], dtype=numpy.float64)
        right = numpy.asarray([10.0, 20.0], dtype=numpy.float64)

        with ts.use_backend("numpy"):
            storage = numpy_add(left, right, dtype=ts.float64, output_shape=(2, 2))

        self.assertEqual(list(storage.buffer), [11.0, 21.0, 12.0, 22.0])


class EndToEndEquivalenceTests(BackendTestCase):
    """Preparation and execution together still produce the documented result."""

    def test_every_backend_agrees_with_the_python_reference(self):
        cases = (
            ([[1.0], [2.0]], [10.0, 20.0], (2, 2)),
            ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], (3,)),
        )
        for left_values, right_values, shape in cases:
            with ts.use_backend("python"):
                expected = (
                    ts.Tensor(left_values) + ts.Tensor(right_values)
                ).tolist()
            for backend in ts.available_backends():
                with self.subTest(backend=backend, shape=shape):
                    with ts.use_backend(backend):
                        try:
                            produced = (
                                ts.Tensor(left_values) + ts.Tensor(right_values)
                            )
                        except ts.BackendOperationUnsupportedError:
                            continue
                        self.assertEqual(produced.shape, shape)
                        self.assertEqual(produced.tolist(), expected)

    def test_scalar_operands_agree_across_backends(self):
        with ts.use_backend("python"):
            expected = (ts.Tensor([1.0, 2.0, 3.0]) * 2.0).tolist()
        for backend in ts.available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    try:
                        produced = ts.Tensor([1.0, 2.0, 3.0]) * 2.0
                    except ts.BackendOperationUnsupportedError:
                        continue
                    self.assertEqual(produced.tolist(), expected)


if __name__ == "__main__":
    unittest.main()
