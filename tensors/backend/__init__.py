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
interface. ``__all__`` stays the user-facing selection API: the ``execute_*``
functions are re-exported for use inside ``tensors``, not as a second public
surface.

What is supported is decided here, by what this module re-exports, rather than
by whether a name in an implementation module begins with an underscore. A
module names a function for the operation it performs; a helper that is not
re-exported is reachable through its own module and is not part of the
supported API. Underscores are kept for state that no caller may touch.
"""

from __future__ import annotations

from tensors.backend.config import (
    BackendMismatchError as BackendMismatchError,
    BackendOperationUnsupportedError as BackendOperationUnsupportedError,
    BackendUnavailableError as BackendUnavailableError,
    available_backends as available_backends,
    get_backend as get_backend,
    set_backend as set_backend,
    use_backend as use_backend,
)
from tensors.backend.dispatch import (
    execute_abs as execute_abs,
    execute_abs_gradient as execute_abs_gradient,
    execute_adam_update as execute_adam_update,
    execute_adam_updates as execute_adam_updates,
    execute_add as execute_add,
    execute_arange as execute_arange,
    execute_arccos as execute_arccos,
    execute_arccos_gradient as execute_arccos_gradient,
    execute_arccosh as execute_arccosh,
    execute_arccosh_gradient as execute_arccosh_gradient,
    execute_arcsin as execute_arcsin,
    execute_arcsin_gradient as execute_arcsin_gradient,
    execute_arcsinh as execute_arcsinh,
    execute_arcsinh_gradient as execute_arcsinh_gradient,
    execute_arctan as execute_arctan,
    execute_arctan_gradient as execute_arctan_gradient,
    execute_arctanh as execute_arctanh,
    execute_arctanh_gradient as execute_arctanh_gradient,
    execute_argmax as execute_argmax,
    execute_argmin as execute_argmin,
    execute_binary_cross_entropy as execute_binary_cross_entropy,
    execute_binary_cross_entropy_gradient as execute_binary_cross_entropy_gradient,
    execute_cast as execute_cast,
    execute_clip as execute_clip,
    execute_clip_gradient as execute_clip_gradient,
    execute_concat as execute_concat,
    execute_convolution as execute_convolution,
    execute_convolution_gradient as execute_convolution_gradient,
    execute_cos as execute_cos,
    execute_cos_gradient as execute_cos_gradient,
    execute_cosh as execute_cosh,
    execute_cosh_gradient as execute_cosh_gradient,
    execute_cross_entropy as execute_cross_entropy,
    execute_cross_entropy_gradient as execute_cross_entropy_gradient,
    execute_divide as execute_divide,
    execute_division_denominator_gradient as execute_division_denominator_gradient,
    execute_equal as execute_equal,
    execute_exp as execute_exp,
    execute_exp_gradient as execute_exp_gradient,
    execute_eye as execute_eye,
    execute_full as execute_full,
    execute_fused_elementwise as execute_fused_elementwise,
    execute_fused_elementwise_backward as execute_fused_elementwise_backward,
    execute_greater as execute_greater,
    execute_greater_equal as execute_greater_equal,
    execute_less as execute_less,
    execute_less_equal as execute_less_equal,
    execute_linspace as execute_linspace,
    execute_log as execute_log,
    execute_log_gradient as execute_log_gradient,
    execute_log_softmax as execute_log_softmax,
    execute_log_softmax_gradient as execute_log_softmax_gradient,
    execute_logsumexp as execute_logsumexp,
    execute_logsumexp_gradient as execute_logsumexp_gradient,
    execute_matmul as execute_matmul,
    execute_matmul_gradient as execute_matmul_gradient,
    execute_maximum as execute_maximum,
    execute_maximum_gradient as execute_maximum_gradient,
    execute_minimum as execute_minimum,
    execute_minimum_gradient as execute_minimum_gradient,
    execute_multiply as execute_multiply,
    execute_negate as execute_negate,
    execute_not_equal as execute_not_equal,
    execute_one_hot_targets as execute_one_hot_targets,
    execute_outer as execute_outer,
    execute_outer_gradient as execute_outer_gradient,
    execute_power as execute_power,
    execute_power_base_gradient as execute_power_base_gradient,
    execute_power_exponent_gradient as execute_power_exponent_gradient,
    execute_reduce_max as execute_reduce_max,
    execute_reduce_max_gradient as execute_reduce_max_gradient,
    execute_reduce_mean as execute_reduce_mean,
    execute_reduce_mean_gradient as execute_reduce_mean_gradient,
    execute_reduce_min as execute_reduce_min,
    execute_reduce_min_gradient as execute_reduce_min_gradient,
    execute_reduce_norm as execute_reduce_norm,
    execute_reduce_prod as execute_reduce_prod,
    execute_reduce_prod_gradient as execute_reduce_prod_gradient,
    execute_reduce_std as execute_reduce_std,
    execute_reduce_std_gradient as execute_reduce_std_gradient,
    execute_reduce_sum as execute_reduce_sum,
    execute_reduce_sum_gradient as execute_reduce_sum_gradient,
    execute_reduce_variance as execute_reduce_variance,
    execute_reduce_variance_gradient as execute_reduce_variance_gradient,
    execute_relu as execute_relu,
    execute_relu_gradient as execute_relu_gradient,
    execute_rmsprop_update as execute_rmsprop_update,
    execute_rmsprop_updates as execute_rmsprop_updates,
    execute_sgd_update as execute_sgd_update,
    execute_sgd_updates as execute_sgd_updates,
    execute_sigmoid as execute_sigmoid,
    execute_sigmoid_gradient as execute_sigmoid_gradient,
    execute_sign as execute_sign,
    execute_sign_gradient as execute_sign_gradient,
    execute_sin as execute_sin,
    execute_sin_gradient as execute_sin_gradient,
    execute_sinh as execute_sinh,
    execute_sinh_gradient as execute_sinh_gradient,
    execute_slice as execute_slice,
    execute_slice_scatter as execute_slice_scatter,
    execute_softmax as execute_softmax,
    execute_softmax_gradient as execute_softmax_gradient,
    execute_softplus as execute_softplus,
    execute_softplus_gradient as execute_softplus_gradient,
    execute_sqrt as execute_sqrt,
    execute_sqrt_gradient as execute_sqrt_gradient,
    execute_stack as execute_stack,
    execute_subtract as execute_subtract,
    execute_sum_products_to_shape as execute_sum_products_to_shape,
    execute_sum_to_shape as execute_sum_to_shape,
    execute_vjp_sum_to_shape as execute_vjp_sum_to_shape,
    execute_tan as execute_tan,
    execute_tan_gradient as execute_tan_gradient,
    execute_tanh as execute_tanh,
    execute_tanh_gradient as execute_tanh_gradient,
    execute_transpose as execute_transpose,
    execute_validate_distributions as execute_validate_distributions,
    execute_where as execute_where,
    execute_where_gradient as execute_where_gradient,
)
from tensors.backend.loading import (
    _clear_backend_kernel_cache as _clear_backend_kernel_cache,
)
from tensors.backend.types import (
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

__all__ = [
    "BackendName",
    "BackendSelection",
    "BackendMismatchError",
    "BackendOperationUnsupportedError",
    "BackendUnavailableError",
    "available_backends",
    "get_backend",
    "set_backend",
    "use_backend",
]
