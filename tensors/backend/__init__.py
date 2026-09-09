"""Numerical backend selection and internal kernel dispatch.

This package is the facade the rest of ``tensors`` imports from. The
implementation is split by responsibility:

- :mod:`~tensors.backend.types` names the backends and their operations;
- :mod:`~tensors.backend.config` selects a backend and reports availability;
- :mod:`~tensors.backend.policy` decides when a workload is worth accelerating;
- :mod:`~tensors.backend.loading` resolves a kernel for a selected backend;
- :mod:`~tensors.backend.dispatch` exposes the ``execute_*`` entry points;
- :mod:`~tensors.backend.storage` owns the backend-native representations.

Re-exports are written as ``name as name`` so a type checker reading this
package as a typed dependency treats them as an explicit part of its
interface. ``__all__`` stays the user-facing selection API: the
``execute_*`` functions and operation aliases are re-exported for use
inside ``tensors``, not as a second public surface.
"""

from __future__ import annotations

from .types import (
    ArgExtremumOperation as ArgExtremumOperation,
    BackendName as BackendName,
    BackendSelection as BackendSelection,
    BinaryOperation as BinaryOperation,
    ComparisonOperation as ComparisonOperation,
    DifferentiableReductionOperation as DifferentiableReductionOperation,
    ExtremumOperation as ExtremumOperation,
    FusedElementwiseStep as FusedElementwiseStep,
    LossReduction as LossReduction,
    NormalizationOperation as NormalizationOperation,
    ReductionOperation as ReductionOperation,
    UnaryOperation as UnaryOperation,
)

from .config import (
    BackendUnavailableError as BackendUnavailableError,
    available_backends as available_backends,
    get_backend as get_backend,
    set_backend as set_backend,
    use_backend as use_backend,
)

from .loading import (
    _clear_backend_kernel_cache as _clear_backend_kernel_cache,
)

from .dispatch import (
    execute_adam_update as execute_adam_update,
    execute_adam_updates as execute_adam_updates,
    execute_arange as execute_arange,
    execute_arg_extremum as execute_arg_extremum,
    execute_binary as execute_binary,
    execute_binary_cross_entropy as execute_binary_cross_entropy,
    execute_binary_cross_entropy_gradient as execute_binary_cross_entropy_gradient,
    execute_cast as execute_cast,
    execute_clip as execute_clip,
    execute_clip_gradient as execute_clip_gradient,
    execute_comparison as execute_comparison,
    execute_concat as execute_concat,
    execute_convolution as execute_convolution,
    execute_convolution_gradient as execute_convolution_gradient,
    execute_cross_entropy as execute_cross_entropy,
    execute_cross_entropy_gradient as execute_cross_entropy_gradient,
    execute_division_denominator_gradient as execute_division_denominator_gradient,
    execute_extremum as execute_extremum,
    execute_extremum_gradient as execute_extremum_gradient,
    execute_eye as execute_eye,
    execute_full as execute_full,
    execute_fused_elementwise as execute_fused_elementwise,
    execute_fused_elementwise_backward as execute_fused_elementwise_backward,
    execute_linspace as execute_linspace,
    execute_logsumexp as execute_logsumexp,
    execute_logsumexp_gradient as execute_logsumexp_gradient,
    execute_matmul as execute_matmul,
    execute_matmul_gradient as execute_matmul_gradient,
    execute_negate as execute_negate,
    execute_normalization as execute_normalization,
    execute_normalization_gradient as execute_normalization_gradient,
    execute_one_hot_targets as execute_one_hot_targets,
    execute_outer as execute_outer,
    execute_outer_gradient as execute_outer_gradient,
    execute_power_base_gradient as execute_power_base_gradient,
    execute_power_exponent_gradient as execute_power_exponent_gradient,
    execute_reduction as execute_reduction,
    execute_reduction_gradient as execute_reduction_gradient,
    execute_rmsprop_update as execute_rmsprop_update,
    execute_rmsprop_updates as execute_rmsprop_updates,
    execute_sgd_update as execute_sgd_update,
    execute_sgd_updates as execute_sgd_updates,
    execute_slice as execute_slice,
    execute_slice_scatter as execute_slice_scatter,
    execute_stack as execute_stack,
    execute_sum_products_to_shape as execute_sum_products_to_shape,
    execute_sum_to_shape as execute_sum_to_shape,
    execute_transpose as execute_transpose,
    execute_unary as execute_unary,
    execute_unary_gradient as execute_unary_gradient,
    execute_validate_distributions as execute_validate_distributions,
    execute_where as execute_where,
    execute_where_gradient as execute_where_gradient,
)


__all__ = [
    "BackendName",
    "BackendSelection",
    "BackendUnavailableError",
    "available_backends",
    "get_backend",
    "set_backend",
    "use_backend",
]
