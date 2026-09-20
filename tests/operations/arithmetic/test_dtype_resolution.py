"""Dtype resolution as a responsibility separate from operand normalization.

`docs/arithmetic-semantics.md` sections 6.2, 6.5 and 7.3. Resolving a result
dtype is a question about two *declarations*; converting a Python scalar is a
question about one *value*. These tests hold the two apart:

- :func:`~tensors.dtype.resolve_result_dtype` takes two ``DataType`` values
  and returns one, and does nothing else;
- :func:`~tensors.dtype.convert_scalar` normalizes a scalar against the
  tensor's declared dtype, and is called from the operation boundary;
- :func:`~tensors.dtype.true_division_dtype` adapts one already-resolved
  dtype for true division, and never resolves a pair a second time.

Expected results come from :mod:`_spec`, which derives them from the document
rather than from the package.
"""

from __future__ import annotations

import inspect
import unittest
from unittest import mock

import tensors as ts
from tensors.dtype import (
    DataType,
    DtypePromotionError,
    convert_scalar,
    resolve_result_dtype,
    true_division_dtype,
)

from . import _spec
from ._support import BACKENDS, DTYPE, ArithmeticTestCase, tensor

BINARY = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
}

#: Operation modules that resolve a section 6 result dtype.
ARITHMETIC_MODULES = (
    "tensors.operations.arithmetic.add",
    "tensors.operations.arithmetic.subtract",
    "tensors.operations.arithmetic.multiply",
    "tensors.operations.arithmetic.divide",
)


def representative(dtype_name):
    """A small value every dtype can hold, so dtype alone is under test."""
    return tensor(dtype_name, [2])


class ResolverInterfaceTests(unittest.TestCase):
    """The resolver answers one question and takes only declarations."""

    def test_it_accepts_exactly_two_dtype_parameters(self):
        parameters = list(inspect.signature(resolve_result_dtype).parameters.values())
        self.assertEqual(len(parameters), 2)
        for parameter in parameters:
            self.assertIs(parameter.annotation, DataType)
            self.assertIs(parameter.default, inspect.Parameter.empty)
            self.assertEqual(parameter.kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)

    def test_it_returns_one_dtype_and_never_a_tuple(self):
        for left in _spec.DTYPES:
            for right in _spec.DTYPES:
                if _spec.promote(left, right) == _spec.CAST:
                    continue
                with self.subTest(left=left, right=right):
                    resolved = resolve_result_dtype(DTYPE[left], DTYPE[right])
                    self.assertIsInstance(resolved, DataType)
                    self.assertNotIsInstance(resolved, tuple)

    def test_it_has_no_operation_or_division_parameter(self):
        parameters = inspect.signature(resolve_result_dtype).parameters
        for forbidden in ("division", "operation", "op", "divide"):
            self.assertNotIn(forbidden, parameters)

    def test_operands_and_scalars_are_not_part_of_its_interface(self):
        """It resolves declarations; anything else must not resolve quietly."""
        for label, right in (
            ("tensor", ts.Tensor([1, 2], dtype=ts.int32)),
            ("int", 5),
            ("float", 5.0),
            ("variable", ts.Variable([1.0], requires_grad=False)),
        ):
            with self.subTest(operand=label):
                with self.assertRaises((TypeError, AttributeError)):
                    resolve_result_dtype(ts.int32, right)

    def test_it_performs_no_scalar_conversion(self):
        """A tensor/tensor operation never reaches the scalar rules."""
        module = "tensors.operations.arithmetic.add"
        with mock.patch(f"{module}.convert_scalar") as converter:
            ts.Tensor([1], dtype=ts.int32) + ts.Tensor([2], dtype=ts.int32)
        converter.assert_not_called()


class ResolvedDtypeTests(ArithmeticTestCase):
    """Section 6.2, resolved directly rather than through an operation."""

    def test_same_dtype_pairs_are_preserved(self):
        for name in _spec.DTYPES:
            with self.subTest(dtype=name):
                self.assertIs(
                    resolve_result_dtype(DTYPE[name], DTYPE[name]), DTYPE[name]
                )

    def test_every_cell_matches_the_specification(self):
        for left in _spec.DTYPES:
            for right in _spec.DTYPES:
                expected = _spec.promote(left, right)
                with self.subTest(left=left, right=right):
                    call = lambda: resolve_result_dtype(DTYPE[left], DTYPE[right])
                    if expected == _spec.CAST:
                        self.assertRaises(DtypePromotionError, call)
                    else:
                        self.assertIs(call(), DTYPE[expected])

    def test_mixed_integers_use_the_smallest_containing_range(self):
        for left in _spec.INTEGER_DTYPES:
            for right in _spec.INTEGER_DTYPES:
                low = min(_spec.integer_range(left)[0], _spec.integer_range(right)[0])
                high = max(_spec.integer_range(left)[1], _spec.integer_range(right)[1])
                with self.subTest(left=left, right=right):
                    resolved = resolve_result_dtype(DTYPE[left], DTYPE[right])
                    got_low, got_high = _spec.integer_range(resolved.name)
                    self.assertLessEqual(got_low, low)
                    self.assertGreaterEqual(got_high, high)

    def test_mixed_floating_pairs_use_the_wider_dtype(self):
        self.assertIs(
            resolve_result_dtype(ts.float32, ts.float64), ts.float64
        )
        self.assertIs(
            resolve_result_dtype(ts.float64, ts.float32), ts.float64
        )

    def test_integer_with_floating_obeys_exact_domain_representability(self):
        for integer in _spec.INTEGER_DTYPES:
            for floating in _spec.FLOAT_DTYPES:
                expected = _spec.promote(integer, floating)
                with self.subTest(integer=integer, floating=floating):
                    call = lambda: resolve_result_dtype(
                        DTYPE[integer], DTYPE[floating]
                    )
                    if expected == _spec.CAST:
                        self.assertRaises(DtypePromotionError, call)
                    else:
                        self.assertIs(call(), DTYPE[expected])
                        self.assertTrue(
                            _spec.dtype_fits_in_float(integer, expected)
                        )

    def test_every_documented_cast_cell_raises(self):
        cast_cells = [
            (left, right)
            for left in _spec.DTYPES
            for right in _spec.DTYPES
            if _spec.promote(left, right) == _spec.CAST
        ]
        self.assertTrue(cast_cells, "the specification defines cast cells")
        for left, right in cast_cells:
            with self.subTest(left=left, right=right):
                with self.assertRaises(DtypePromotionError):
                    resolve_result_dtype(DTYPE[left], DTYPE[right])

    def test_resolution_is_symmetric(self):
        for left in _spec.DTYPES:
            for right in _spec.DTYPES:
                with self.subTest(left=left, right=right):
                    try:
                        forward = resolve_result_dtype(DTYPE[left], DTYPE[right])
                    except DtypePromotionError:
                        with self.assertRaises(DtypePromotionError):
                            resolve_result_dtype(DTYPE[right], DTYPE[left])
                        continue
                    self.assertIs(
                        forward, resolve_result_dtype(DTYPE[right], DTYPE[left])
                    )

    def test_element_values_cannot_reach_the_resolver(self):
        """Only declarations are passed, so no value can change the result."""
        for values in ([0, 0], [1, 2], [2**31 - 1, -(2**31)]):
            with self.subTest(values=values):
                left = ts.Tensor(values, dtype=ts.int32)
                right = ts.Tensor([1, 1], dtype=ts.int32)
                self.assertIs(
                    resolve_result_dtype(left.dtype, right.dtype), ts.int32
                )
                self.assertIs((left + right).dtype, ts.int32)


class TrueDivisionAdjustmentTests(ArithmeticTestCase):
    """Section 7.3: a one-dtype adjustment applied after resolution."""

    def test_it_accepts_exactly_one_dtype_parameter(self):
        parameters = list(inspect.signature(true_division_dtype).parameters.values())
        self.assertEqual(len(parameters), 1)
        self.assertIs(parameters[0].annotation, DataType)

    def test_it_never_returns_an_integer_dtype(self):
        for name in _spec.DTYPES:
            with self.subTest(dtype=name):
                try:
                    adjusted = true_division_dtype(DTYPE[name])
                except DtypePromotionError:
                    self.assertEqual(name, "int64")
                    continue
                self.assertEqual(adjusted.kind, "floating")

    def test_floating_dtypes_are_preserved(self):
        for name in _spec.FLOAT_DTYPES:
            with self.subTest(dtype=name):
                self.assertIs(true_division_dtype(DTYPE[name]), DTYPE[name])

    def test_composition_reproduces_the_documented_division_table(self):
        """Resolve once, then adjust — never resolve the pair twice."""
        for left in _spec.DTYPES:
            for right in _spec.DTYPES:
                expected = _spec.divide_result(left, right)
                with self.subTest(left=left, right=right):
                    def call():
                        resolved = resolve_result_dtype(DTYPE[left], DTYPE[right])
                        return true_division_dtype(resolved)

                    if expected == _spec.CAST:
                        self.assertRaises(DtypePromotionError, call)
                    else:
                        self.assertIs(call(), DTYPE[expected])


class ScalarNormalizationTests(ArithmeticTestCase):
    """Section 6.5: the scalar rules stay outside dtype resolution."""

    def test_representable_scalars_convert_against_the_tensor_dtype(self):
        cases = (
            ("int32", 5, 5),
            ("int32", 5.0, 5),
            ("uint8", 255, 255),
            ("float32", 2, 2.0),
            ("float64", 0.5, 0.5),
        )
        for name, value, expected in cases:
            with self.subTest(dtype=name, value=value):
                self.assertEqual(convert_scalar(value, DTYPE[name]), expected)

    def test_unrepresentable_scalars_raise(self):
        cases = (
            ("int32", 2**40),
            ("int32", 5.5),
            ("int8", 200),
            ("uint8", -1),
            ("float32", 1e40),
            ("float32", 2**24 + 1),
        )
        for name, value in cases:
            with self.subTest(dtype=name, value=value):
                self.assertRaises(TypeError, convert_scalar, value, DTYPE[name])

    def test_boolean_scalars_remain_unsupported(self):
        for name in _spec.DTYPES:
            with self.subTest(dtype=name):
                self.assertRaises(TypeError, convert_scalar, True, DTYPE[name])
                self.assertRaises(
                    TypeError, lambda: representative(name) + True
                )

    def test_a_scalar_never_widens_the_result_dtype(self):
        for name in _spec.DTYPES:
            for value in (0, 1, 2):
                rule, converts = _spec.scalar_rule(name, value)
                if not converts:
                    continue
                for symbol, operation in BINARY.items():
                    with self.subTest(dtype=name, value=value, op=symbol, rule=rule):
                        self.assertDtypeIs(
                            operation(representative(name), value), name
                        )

    def test_float_to_integer_restrictions_are_intact(self):
        integer_tensor = ts.Tensor([4], dtype=ts.int32)
        self.assertIs((integer_tensor + 3.0).dtype, ts.int32)
        self.assertRaises(TypeError, lambda: integer_tensor + 3.5)


class OperationStageTests(ArithmeticTestCase):
    """Each operation classifies, normalizes, resolves, then dispatches."""

    def test_tensor_pairs_resolve_through_the_shared_resolver(self):
        for module in ARITHMETIC_MODULES:
            with self.subTest(module=module):
                target = f"{module}.resolve_result_dtype"
                with mock.patch(target, wraps=resolve_result_dtype) as resolver:
                    left = ts.Tensor([4], dtype=ts.int32)
                    right = ts.Tensor([2], dtype=ts.int32)
                    {
                        "add": lambda: left + right,
                        "subtract": lambda: left - right,
                        "multiply": lambda: left * right,
                        "divide": lambda: left / right,
                    }[module.rsplit(".", 1)[1]]()
                resolver.assert_called_once_with(ts.int32, ts.int32)

    def test_scalar_operands_are_converted_not_resolved(self):
        for module in ARITHMETIC_MODULES:
            with self.subTest(module=module):
                resolver_target = f"{module}.resolve_result_dtype"
                converter_target = f"{module}.convert_scalar"
                with mock.patch(resolver_target, wraps=resolve_result_dtype) as res:
                    with mock.patch(
                        converter_target, wraps=convert_scalar
                    ) as converter:
                        left = ts.Tensor([4], dtype=ts.int32)
                        {
                            "add": lambda: left + 2,
                            "subtract": lambda: left - 2,
                            "multiply": lambda: left * 2,
                            "divide": lambda: left / 2,
                        }[module.rsplit(".", 1)[1]]()
                res.assert_not_called()
                converter.assert_called_once_with(2, ts.int32)

    def test_addition_subtraction_and_multiplication_follow_the_table(self):
        for left in _spec.DTYPES:
            for right in _spec.DTYPES:
                expected = _spec.promote(left, right)
                for symbol, operation in BINARY.items():
                    with self.subTest(left=left, right=right, op=symbol):
                        call = lambda: operation(
                            representative(left), representative(right)
                        )
                        if expected == _spec.CAST:
                            self.assertRaisesConversion(call)
                        else:
                            self.assertDtypeIs(call(), expected)

    def test_division_follows_the_documented_division_table(self):
        for left in _spec.DTYPES:
            for right in _spec.DTYPES:
                expected = _spec.divide_result(left, right)
                with self.subTest(left=left, right=right):
                    call = lambda: representative(left) / representative(right)
                    if expected == _spec.CAST:
                        self.assertRaisesConversion(call)
                    else:
                        self.assertDtypeIs(call(), expected)

    def test_floating_division_preserves_the_floating_dtype(self):
        for name in _spec.FLOAT_DTYPES:
            with self.subTest(dtype=name):
                self.assertDtypeIs(
                    representative(name) / representative(name), name
                )

    def test_integer_division_returns_the_documented_floating_dtype(self):
        self.assertDtypeIs(representative("int8") / representative("int8"), "float32")
        self.assertDtypeIs(
            representative("int32") / representative("int32"), "float64"
        )
        self.assertRaisesConversion(
            lambda: representative("int64") / representative("int64")
        )

    def test_scalar_division_converts_against_the_tensor_dtype(self):
        """The conversion target is the tensor's dtype, not the result's."""
        self.assertDtypeIs(ts.Tensor([4], dtype=ts.int32) / 2, "float64")
        # 2.5 has no int32 value, so S2 refuses it even though the result
        # dtype of the division is floating.
        self.assertRaises(TypeError, lambda: ts.Tensor([4], dtype=ts.int32) / 2.5)

    def test_reflected_forms_keep_their_documented_dtypes(self):
        self.assertDtypeIs(3 - ts.Tensor([1], dtype=ts.uint8), "uint8")
        self.assertDtypeIs(6 / ts.Tensor([2], dtype=ts.int32), "float64")
        self.assertRaises(TypeError, lambda: -1 - ts.Tensor([1], dtype=ts.uint8))

    def test_variable_arithmetic_matches_tensor_arithmetic(self):
        left = ts.Variable(ts.Tensor([4.0], dtype=ts.float32), requires_grad=True)
        self.assertIs((left + 2).dtype, ts.float32)
        self.assertIs((left * 2.0).dtype, ts.float32)
        self.assertIs((left / 2).dtype, ts.float32)
        right = ts.Variable(ts.Tensor([2.0], dtype=ts.float64), requires_grad=True)
        self.assertIs((left + right).dtype, ts.float64)
        self.assertIs((left / right).dtype, ts.float64)

    def test_power_dtype_behaviour_is_unchanged(self):
        base = ts.Tensor([2], dtype=ts.int32)
        self.assertIs((base ** ts.Tensor([3], dtype=ts.int32)).dtype, ts.int32)
        self.assertIs((base**2).dtype, ts.int32)
        self.assertIs((base**2.0).dtype, ts.int32)
        self.assertRaises(TypeError, lambda: base**0.5)
        self.assertIs((2 ** ts.Tensor([3], dtype=ts.int32)).dtype, ts.int32)
        self.assertIs((2.5 ** ts.Tensor([3], dtype=ts.int32)).dtype, ts.float64)
        self.assertRaises(
            DtypePromotionError, lambda: 2.5 ** ts.Tensor([3], dtype=ts.int64)
        )


class BackendInvarianceTests(ArithmeticTestCase):
    """The refactor is dtype-only: residency and dispatch are untouched."""

    def test_the_result_dtype_is_identical_on_every_backend(self):
        for backend in BACKENDS:
            for left in ("uint8", "int8", "int16", "int32", "float32", "float64"):
                expected = _spec.promote(left, left)
                with self.subTest(backend=backend, dtype=left):
                    with ts.use_backend(backend):
                        try:
                            result = representative(left) + representative(left)
                        except ts.BackendOperationUnsupportedError:
                            # A backend that cannot conform says so; strict
                            # selection means it must not answer from another.
                            continue
                        self.assertDtypeIs(result, expected)

    def test_results_stay_resident_on_the_selected_backend(self):
        for backend in BACKENDS:
            for operation in ("+", "-", "*"):
                with self.subTest(backend=backend, op=operation):
                    with ts.use_backend(backend):
                        try:
                            left = ts.Tensor([4, 6], dtype=ts.int32)
                            right = ts.Tensor([2, 3], dtype=ts.int32)
                            result = BINARY[operation](left, right)
                        except ts.BackendOperationUnsupportedError:
                            continue
                        self.assertEqual(left._storage.kind, backend)
                        self.assertEqual(result._storage.kind, backend)

    def test_division_stays_resident_on_the_selected_backend(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    try:
                        result = ts.Tensor([4, 6], dtype=ts.int32) / ts.Tensor(
                            [2, 3], dtype=ts.int32
                        )
                    except ts.BackendOperationUnsupportedError:
                        continue
                    self.assertEqual(result._storage.kind, backend)
                    self.assertIs(result.dtype, ts.float64)

    def test_an_operand_from_another_backend_is_still_refused(self):
        """Strict residency validation is unchanged by this refactor."""
        others = [name for name in BACKENDS if name != "python"]
        if not others:
            self.skipTest("only the Python backend is installed")
        for backend in others:
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    foreign = ts.Tensor([1, 2], dtype=ts.int32)
                resident = ts.Tensor([1, 2], dtype=ts.int32)
                with self.assertRaises(ts.BackendMismatchError):
                    resident + foreign


if __name__ == "__main__":
    unittest.main()
