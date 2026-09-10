"""Reduction kernels and the stable-summation machinery behind them."""

from __future__ import annotations

from .extrema import (
    arg_extremum as arg_extremum,
)
from .logsumexp_ops import (
    logsumexp as logsumexp,
    logsumexp_gradient as logsumexp_gradient,
)
from .reduction_ops import (
    reduction as reduction,
    reduction_gradient as reduction_gradient,
)
from .shape import (
    sum_products_to_shape as sum_products_to_shape,
    sum_to_shape as sum_to_shape,
)


__all__ = [
    "arg_extremum",
    "logsumexp",
    "logsumexp_gradient",
    "reduction",
    "reduction_gradient",
    "sum_products_to_shape",
    "sum_to_shape",
]
