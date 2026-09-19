"""Raw NumPy, called directly.

This is the comparison the provider rung of every ladder is taken
against: NumPy operating on its own arrays in host memory, with nothing
of this package between the call and the work.
"""

from __future__ import annotations

import importlib
from typing import Any

from ._native import build

#: The library this baseline is.
NAME = "numpy"


def module() -> Any:
    """Return the library, imported on first use."""
    return importlib.import_module(NAME)


def array(
    shape: tuple[int, ...],
    *,
    dtype_name: str = "float64",
    kind: str = "ramp",
    value: float = 1.5,
) -> Any:
    """Build a native array holding the values a Tensor would hold."""
    return build(module(), shape, dtype_name=dtype_name, kind=kind, value=value)


__all__ = ["NAME", "array", "module"]
