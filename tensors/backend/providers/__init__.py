"""Array providers with arithmetic callables bound to one array module."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol, TYPE_CHECKING

from ..kernels.arithmetic.add import add
from ..kernels.arithmetic.subtract import subtract
from ..kernels.arithmetic.multiply import multiply
from ..kernels.arithmetic.divide import divide
from ..kernels.arithmetic.power import power
from ..storage import Storage

if TYPE_CHECKING:
    from ..._typing import Scalar
    from ...dtype import DataType
    from ...tensor import Tensor


class ArithmeticKernel(Protocol):
    def __call__(
        self, left: Tensor | Scalar, right: Tensor | Scalar, *,
        dtype: DataType, output_shape: tuple[int, ...],
    ) -> Storage | None: ...


@dataclass
class ArrayBackend:
    """A fixed provider and its dedicated arithmetic functions."""

    xp: Any
    add: ArithmeticKernel
    subtract: ArithmeticKernel
    multiply: ArithmeticKernel
    divide: ArithmeticKernel
    power: ArithmeticKernel


def bind_array_backend(xp: Any) -> ArrayBackend:
    """Bind once; executing a kernel never reads backend configuration."""
    return ArrayBackend(
        xp=xp,
        add=partial(add, xp),
        subtract=partial(subtract, xp),
        multiply=partial(multiply, xp),
        divide=partial(divide, xp),
        power=partial(power, xp),
    )
