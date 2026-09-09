"""Internal kernel dispatch entry points.

Every ``execute_*`` function follows the same shape: apply the workload
policy, resolve the kernel for the active backend, and run it. Returning
``None`` asks the caller to run its own Python fallback instead.

The entry points are grouped by execution domain, mirroring
:mod:`tensors.backend.kernels`; this module re-exports them as the dispatch
surface the rest of ``tensors`` imports:

- :mod:`~tensors.backend.dispatch.elementwise` elementwise work and its VJPs;
- :mod:`~tensors.backend.dispatch.creation` values built from parameters;
- :mod:`~tensors.backend.dispatch.manipulation` shape, layout, and indexing;
- :mod:`~tensors.backend.dispatch.reductions` reductions and shape summation;
- :mod:`~tensors.backend.dispatch.linalg` matrix and vector products;
- :mod:`~tensors.backend.dispatch.convolution` grouped cross-correlation;
- :mod:`~tensors.backend.dispatch.fusion` fused chains, including the Python
  interpretation used when no accelerated backend is active;
- :mod:`~tensors.backend.dispatch.nn` normalization and losses;
- :mod:`~tensors.backend.dispatch.optim` optimizer updates.

Re-exports are written as ``name as name`` so a type checker reading this
package as a typed dependency treats them as part of its interface.
"""

from __future__ import annotations

from .elementwise import (
    execute_binary as execute_binary,
    execute_clip as execute_clip,
    execute_clip_gradient as execute_clip_gradient,
    execute_comparison as execute_comparison,
    execute_division_denominator_gradient as execute_division_denominator_gradient,
    execute_extremum as execute_extremum,
    execute_extremum_gradient as execute_extremum_gradient,
    execute_negate as execute_negate,
    execute_power_base_gradient as execute_power_base_gradient,
    execute_power_exponent_gradient as execute_power_exponent_gradient,
    execute_unary as execute_unary,
    execute_unary_gradient as execute_unary_gradient,
    execute_where as execute_where,
    execute_where_gradient as execute_where_gradient,
)
from .creation import (
    execute_arange as execute_arange,
    execute_eye as execute_eye,
    execute_full as execute_full,
    execute_linspace as execute_linspace,
    execute_one_hot_targets as execute_one_hot_targets,
)
from .manipulation import (
    execute_cast as execute_cast,
    execute_concat as execute_concat,
    execute_slice as execute_slice,
    execute_slice_scatter as execute_slice_scatter,
    execute_stack as execute_stack,
    execute_transpose as execute_transpose,
)
from .reductions import (
    execute_arg_extremum as execute_arg_extremum,
    execute_logsumexp as execute_logsumexp,
    execute_logsumexp_gradient as execute_logsumexp_gradient,
    execute_reduction as execute_reduction,
    execute_reduction_gradient as execute_reduction_gradient,
    execute_sum_products_to_shape as execute_sum_products_to_shape,
    execute_sum_to_shape as execute_sum_to_shape,
)
from .linalg import (
    execute_matmul as execute_matmul,
    execute_matmul_gradient as execute_matmul_gradient,
    execute_outer as execute_outer,
    execute_outer_gradient as execute_outer_gradient,
)
from .convolution import (
    execute_convolution as execute_convolution,
    execute_convolution_gradient as execute_convolution_gradient,
)
from .fusion import (
    execute_fused_elementwise as execute_fused_elementwise,
    execute_fused_elementwise_backward as execute_fused_elementwise_backward,
)
from .nn import (
    execute_binary_cross_entropy as execute_binary_cross_entropy,
    execute_binary_cross_entropy_gradient as execute_binary_cross_entropy_gradient,
    execute_cross_entropy as execute_cross_entropy,
    execute_cross_entropy_gradient as execute_cross_entropy_gradient,
    execute_normalization as execute_normalization,
    execute_normalization_gradient as execute_normalization_gradient,
    execute_validate_distributions as execute_validate_distributions,
)
from .optim import (
    execute_adam_update as execute_adam_update,
    execute_adam_updates as execute_adam_updates,
    execute_rmsprop_update as execute_rmsprop_update,
    execute_rmsprop_updates as execute_rmsprop_updates,
    execute_sgd_update as execute_sgd_update,
    execute_sgd_updates as execute_sgd_updates,
)


__all__ = [
    "execute_adam_update",
    "execute_adam_updates",
    "execute_arange",
    "execute_arg_extremum",
    "execute_binary",
    "execute_binary_cross_entropy",
    "execute_binary_cross_entropy_gradient",
    "execute_cast",
    "execute_clip",
    "execute_clip_gradient",
    "execute_comparison",
    "execute_concat",
    "execute_convolution",
    "execute_convolution_gradient",
    "execute_cross_entropy",
    "execute_cross_entropy_gradient",
    "execute_division_denominator_gradient",
    "execute_extremum",
    "execute_extremum_gradient",
    "execute_eye",
    "execute_full",
    "execute_fused_elementwise",
    "execute_fused_elementwise_backward",
    "execute_linspace",
    "execute_logsumexp",
    "execute_logsumexp_gradient",
    "execute_matmul",
    "execute_matmul_gradient",
    "execute_negate",
    "execute_normalization",
    "execute_normalization_gradient",
    "execute_one_hot_targets",
    "execute_outer",
    "execute_outer_gradient",
    "execute_power_base_gradient",
    "execute_power_exponent_gradient",
    "execute_reduction",
    "execute_reduction_gradient",
    "execute_rmsprop_update",
    "execute_rmsprop_updates",
    "execute_sgd_update",
    "execute_sgd_updates",
    "execute_slice",
    "execute_slice_scatter",
    "execute_stack",
    "execute_sum_products_to_shape",
    "execute_sum_to_shape",
    "execute_transpose",
    "execute_unary",
    "execute_unary_gradient",
    "execute_validate_distributions",
    "execute_where",
    "execute_where_gradient",
]
