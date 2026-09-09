"""Batched CUDA optimizer kernels, compiled on first use."""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _cuda_sgd_batch_kernel() -> Any:
    """Return a fused CUDA SGD update and validation kernel."""
    cupy = importlib.import_module("cupy")
    return cupy.ElementwiseKernel(
        "float64 parameter, float64 gradient, float64 learning_rate",
        "float64 updated, raw uint32 invalid",
        """
        updated = parameter - learning_rate * gradient;
        if (
            !isfinite(parameter)
            || !isfinite(gradient)
            || !isfinite(updated)
        ) atomicExch(&invalid[0], 1U);
        """,
        "tensors_sgd_batch",
    )

@lru_cache(maxsize=1)
def _cuda_adam_batch_kernel() -> Any:
    """Return a fused CUDA Adam update and validation kernel."""
    cupy = importlib.import_module("cupy")
    return cupy.ElementwiseKernel(
        """
        float64 parameter, float64 gradient, float64 moment,
        float64 scale, float64 normalized, float64 beta1, float64 beta2,
        float64 learning_rate, float64 epsilon,
        float64 first_correction, float64 second_correction
        """,
        """
        float64 updated, float64 new_moment, float64 visible,
        float64 new_scale, float64 new_normalized, raw uint32 invalid
        """,
        """
        double left = beta1 * moment;
        double right = (1.0 - beta1) * gradient;
        new_moment = left + right;
        new_scale = fmax(scale, fabs(gradient));
        double safe_scale = new_scale == 0.0 ? 1.0 : new_scale;
        double previous_ratio = scale / safe_scale;
        double gradient_ratio = fabs(gradient) / safe_scale;
        new_normalized = (
            beta2 * normalized * previous_ratio * previous_ratio
            + (1.0 - beta2) * gradient_ratio * gradient_ratio
        );
        if (new_scale == 0.0) new_normalized = 0.0;
        double root_correction = sqrt(second_correction);
        double root_moment = new_scale * sqrt(new_normalized);
        double denominator = first_correction * (
            root_moment + epsilon * root_correction
        );
        double ratio = new_moment * root_correction / denominator;
        updated = parameter - learning_rate * ratio;
        visible = new_scale * new_scale * new_normalized;
        if (
            !isfinite(parameter)
            || !isfinite(gradient)
            || !isfinite(moment)
            || !isfinite(scale)
            || !isfinite(normalized)
            || !isfinite(updated)
            || !isfinite(new_moment)
            || !isfinite(new_scale)
            || !isfinite(new_normalized)
        ) atomicExch(&invalid[0], 1U);
        """,
        "tensors_adam_batch",
    )

@lru_cache(maxsize=1)
def _cuda_rmsprop_batch_kernel() -> Any:
    """Return a fused CUDA RMSprop update and validation kernel."""
    cupy = importlib.import_module("cupy")
    return cupy.ElementwiseKernel(
        """
        float64 parameter, float64 gradient, float64 scale,
        float64 normalized, float64 rho, float64 learning_rate,
        float64 epsilon
        """,
        """
        float64 updated, float64 new_scale, float64 new_normalized,
        raw uint32 invalid
        """,
        """
        new_scale = fmax(scale, fabs(gradient));
        double safe_scale = new_scale == 0.0 ? 1.0 : new_scale;
        double previous_ratio = scale / safe_scale;
        double gradient_ratio = fabs(gradient) / safe_scale;
        new_normalized = (
            rho * normalized * previous_ratio * previous_ratio
            + (1.0 - rho) * gradient_ratio * gradient_ratio
        );
        if (new_scale == 0.0) new_normalized = 0.0;
        double root_moment = new_scale * sqrt(new_normalized);
        updated = parameter - (
            learning_rate * gradient / (root_moment + epsilon)
        );
        if (
            !isfinite(parameter)
            || !isfinite(gradient)
            || !isfinite(scale)
            || !isfinite(normalized)
            || !isfinite(updated)
            || !isfinite(new_scale)
            || !isfinite(new_normalized)
        ) atomicExch(&invalid[0], 1U);
        """,
        "tensors_rmsprop_batch",
    )
