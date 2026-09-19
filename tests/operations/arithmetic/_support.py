"""Shared fixtures for the arithmetic conformance tests.

These build operands and compare results. They deliberately contain no
expectations: every expected value comes from :mod:`_spec`, which is written
from the specification, never from a backend.
"""

from __future__ import annotations

import math
import struct
import unittest

import tensors as ts

from . import _spec

#: dtype name to the package's DataType.
DTYPE = {name: getattr(ts, name) for name in _spec.DTYPES}

#: Backends installed in this environment.
BACKENDS = tuple(ts.available_backends())

_FORMAT = {"float32": ("<f", "<I"), "float64": ("<d", "<Q")}


def tensor(dtype_name: str, values):
    """Build a tensor of a declared dtype from exact Python values."""
    return ts.Tensor(list(values), dtype=DTYPE[dtype_name])


def float_bits(value: float, dtype_name: str) -> int:
    """Return the bit pattern of a value in a floating format."""
    pack, unpack = _FORMAT[dtype_name]
    return struct.unpack(unpack, struct.pack(pack, value))[0]


class ArithmeticTestCase(unittest.TestCase):
    """Restores the backend selection and offers specification comparisons."""

    def setUp(self):
        self._previous_backend = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self._previous_backend)

    # -- assertions -----------------------------------------------------

    def assertDtypeIs(self, result, dtype_name: str):
        """The result carries the declared dtype, by identity."""
        self.assertIs(result.dtype, DTYPE[dtype_name])

    def assertIntegerResult(self, result, expected, dtype_name: str):
        """Integer results are compared exactly; no tolerance is permitted."""
        self.assertDtypeIs(result, dtype_name)
        self.assertEqual([int(v) for v in result.tolist()], list(expected))

    def assertFloatBitsEqual(self, result, expected, dtype_name: str):
        """Compare correctly rounded results bitwise, signed zero included.

        NaN is classified rather than compared, because the specification
        leaves NaN payloads and sign unspecified.
        """
        self.assertDtypeIs(result, dtype_name)
        produced = result.tolist()
        self.assertEqual(len(produced), len(expected))
        for index, (got, want) in enumerate(zip(produced, expected)):
            with self.subTest(element=index):
                if isinstance(want, float) and math.isnan(want):
                    self.assertTrue(math.isnan(got), f"expected NaN, got {got!r}")
                    continue
                self.assertEqual(
                    float_bits(got, dtype_name),
                    float_bits(want, dtype_name),
                    f"expected {want!r}, got {got!r}",
                )

    def assertRaisesConversion(self, callable_object):
        """A refused implicit conversion raises TypeError (section 6.5)."""
        return self.assertRaises(TypeError, callable_object)


def each_backend():
    """Yield every installed backend name."""
    return BACKENDS


__all__ = [
    "BACKENDS",
    "DTYPE",
    "ArithmeticTestCase",
    "each_backend",
    "float_bits",
    "tensor",
]
