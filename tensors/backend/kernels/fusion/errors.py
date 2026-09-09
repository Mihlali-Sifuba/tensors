"""Domain guards and error reporting for fused kernels."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .expressions import _fused_operand_expression

if TYPE_CHECKING:
    from ...types import FusedElementwiseStep

def _fused_domain_checks(
    step: FusedElementwiseStep,
    value: str,
    result: str,
) -> tuple[tuple[str, int], ...]:
    """Return CUDA predicates that preserve public math-domain errors."""
    operation, _, reverse, _ = step
    operand = _fused_operand_expression(step, value)
    if operation == "sqrt":
        return ((f"({value}) < 0.0", 2),)
    if operation == "log":
        return ((f"({value}) <= 0.0", 3),)
    if operation in {"arcsin", "arccos"}:
        return ((f"({value}) < -1.0 || ({value}) > 1.0", 4),)
    if operation == "arccosh":
        return ((f"({value}) < 1.0", 5),)
    if operation == "arctanh":
        return ((
            f"!isnan({value}) && (({value}) <= -1.0 || ({value}) >= 1.0)",
            6,
        ),)
    if operation in {"sin", "cos", "tan"}:
        return ((f"isinf({value})", 7),)
    if operation == "power" and operand is not None:
        left, right = (operand, value) if reverse else (value, operand)
        return (
            (
                f"({left}) < 0.0 && trunc({right}) != ({right})",
                8,
            ),
            (f"({left}) == 0.0 && ({right}) < 0.0", 8),
            (
                f"isfinite({left}) && isfinite({right}) && isinf({result})",
                9,
            ),
        )
    return ()

def _raise_fused_kernel_error(code: int) -> None:
    """Raise the public exception represented by a fused-kernel error code."""
    if code == 1:
        raise ZeroDivisionError("Division by zero")
    messages = {
        2: "sqrt is only defined for non-negative values",
        3: "log is only defined for positive values",
        4: "inverse trigonometric function is only defined between -1 and 1",
        5: "arccosh is only defined for values greater than or equal to 1",
        6: "arctanh is only defined for values strictly between -1 and 1",
        7: "trigonometric functions are undefined for infinite values",
        8: "power is not defined for these real-valued inputs",
        10: "sqrt derivative is undefined at zero",
        11: "inverse trigonometric derivative is undefined at -1 and 1",
        12: "arccosh derivative is undefined at 1",
        13: "sign derivative is undefined at zero",
        14: "power derivative is undefined at a zero base",
    }
    if code == 9:
        raise OverflowError("power result is too large to represent")
    raise ValueError(messages.get(code, "invalid value in fused CUDA operation"))

def _fused_backward_checks(
    step: FusedElementwiseStep,
    value: str,
) -> tuple[tuple[str, int], ...]:
    """Return derivative-domain checks for one fused operation."""
    operation, scalar, reverse, _ = step
    if operation == "sqrt":
        return ((f"({value}) == 0.0", 10),)
    if operation in {"arcsin", "arccos"}:
        return ((f"({value}) == -1.0 || ({value}) == 1.0", 11),)
    if operation == "arccosh":
        return ((f"({value}) == 1.0", 12),)
    if operation == "sign":
        return ((f"({value}) == 0.0", 13),)
    if operation == "power" and not reverse:
        if scalar is not None:
            if scalar != 0 and scalar < 1:
                return ((f"({value}) == 0.0", 14),)
            return ()
        operand = _fused_operand_expression(step, value)
        if operand is not None:
            # A fractional exponent below one has no derivative at a zero base.
            return ((
                f"({value}) == 0.0 && ({operand}) != 0.0 "
                f"&& ({operand}) < 1.0",
                14,
            ),)
    return ()
