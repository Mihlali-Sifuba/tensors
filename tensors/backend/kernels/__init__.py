"""Shared NumPy/CuPy kernels backed by native array storage.

The kernel families are split by responsibility. This module is the internal
facade that the provider modules (:mod:`~tensors.backend.numpy` and
:mod:`~tensors.backend.cuda`) import from, and so is what the backend loader
ultimately resolves kernels against:

- :mod:`~tensors.backend.kernels.core` bridges Tensor/Storage and native arrays;
- :mod:`~tensors.backend.kernels.elementwise` runs elementwise work and its VJPs;
- :mod:`~tensors.backend.kernels.fusion` compiles and runs fused chains;
- :mod:`~tensors.backend.kernels.reductions` reduces values stably;
- :mod:`~tensors.backend.kernels.creation` builds arrays from parameters;
- :mod:`~tensors.backend.kernels.manipulation` reshapes and re-lays out values;
- :mod:`~tensors.backend.kernels.linalg` runs matrix and vector products;
- :mod:`~tensors.backend.kernels.nn` runs normalization and loss kernels;
- :mod:`~tensors.backend.kernels.conv` runs grouped cross-correlation;
- :mod:`~tensors.backend.kernels.optim` runs optimizer updates.

Names are re-exported as ``name as name`` so that the loader resolving
``getattr(module, name)`` and a type checker reading this package see the same
kernel surface.
"""

from __future__ import annotations

from .elementwise import (
    binary as binary,
    clip as clip,
    clip_gradient as clip_gradient,
    comparison as comparison,
    division_denominator_gradient as division_denominator_gradient,
    extremum as extremum,
    extremum_gradient as extremum_gradient,
    negate as negate,
    power_base_gradient as power_base_gradient,
    power_exponent_gradient as power_exponent_gradient,
    unary as unary,
    unary_gradient as unary_gradient,
    where as where,
    where_gradient as where_gradient,
)
from .fusion import (
    fused_elementwise as fused_elementwise,
    fused_elementwise_backward as fused_elementwise_backward,
)
from .reductions import (
    arg_extremum as arg_extremum,
    logsumexp as logsumexp,
    logsumexp_gradient as logsumexp_gradient,
    reduction as reduction,
    reduction_gradient as reduction_gradient,
    sum_products_to_shape as sum_products_to_shape,
    sum_to_shape as sum_to_shape,
)
from .creation import (
    arange as arange,
    eye as eye,
    full as full,
    linspace as linspace,
    one_hot_targets as one_hot_targets,
)
from .manipulation import (
    cast_tensor as cast_tensor,
    concat as concat,
    slice_scatter as slice_scatter,
    slice_tensor as slice_tensor,
    stack as stack,
    transpose as transpose,
)
from .linalg import (
    matmul as matmul,
    matmul_gradient as matmul_gradient,
    outer as outer,
    outer_gradient as outer_gradient,
)
from .nn import (
    binary_cross_entropy as binary_cross_entropy,
    binary_cross_entropy_gradient as binary_cross_entropy_gradient,
    cross_entropy as cross_entropy,
    cross_entropy_gradient as cross_entropy_gradient,
    distributions_valid as distributions_valid,
    normalization as normalization,
    normalization_gradient as normalization_gradient,
)
from .conv import (
    convolution as convolution,
    convolution_gradient as convolution_gradient,
)
from .optim import (
    adam_update as adam_update,
    adam_updates as adam_updates,
    rmsprop_update as rmsprop_update,
    rmsprop_updates as rmsprop_updates,
    sgd_update as sgd_update,
    sgd_updates as sgd_updates,
)


__all__ = [
    "adam_update",
    "adam_updates",
    "arange",
    "arg_extremum",
    "binary",
    "binary_cross_entropy",
    "binary_cross_entropy_gradient",
    "cast_tensor",
    "clip",
    "clip_gradient",
    "comparison",
    "concat",
    "convolution",
    "convolution_gradient",
    "cross_entropy",
    "cross_entropy_gradient",
    "distributions_valid",
    "division_denominator_gradient",
    "extremum",
    "extremum_gradient",
    "eye",
    "full",
    "fused_elementwise",
    "fused_elementwise_backward",
    "linspace",
    "logsumexp",
    "logsumexp_gradient",
    "matmul",
    "matmul_gradient",
    "negate",
    "normalization",
    "normalization_gradient",
    "one_hot_targets",
    "outer",
    "outer_gradient",
    "power_base_gradient",
    "power_exponent_gradient",
    "reduction",
    "reduction_gradient",
    "rmsprop_update",
    "rmsprop_updates",
    "sgd_update",
    "sgd_updates",
    "slice_scatter",
    "slice_tensor",
    "stack",
    "sum_products_to_shape",
    "sum_to_shape",
    "transpose",
    "unary",
    "unary_gradient",
    "where",
    "where_gradient",
]
