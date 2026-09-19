"""Domain guards and error reporting for fused kernels."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.cuda.kernels.fusion.expressions import _fused_operand_expression

if TYPE_CHECKING:
    from tensors.backend.types import FusedElementwiseStep


def _fused_domain_checks(
    step: FusedElementwiseStep, value: str, result: str, *, storage_type: str
) -> tuple[tuple[str, int], ...]:
    """Return CUDA predicates that preserve public math-domain errors."""
    operation, _, reverse, _ = step
    operand = _fused_operand_expression(step, value, storage_type=storage_type)
    if operation == "sqrt":
        return ((f"({value}) < 0.0", 2),)
    if operation == "log":
        return ((f"({value}) <= 0.0", 3),)
    if operation in {"arcsin", "arccos"}:
        return ((f"({value}) < -1.0 || ({value}) > 1.0", 4),)
    if operation == "arccosh":
        return ((f"({value}) < 1.0", 5),)
    if operation == "arctanh":
        return ((f"!isnan({value}) && (({value}) <= -1.0 || ({value}) >= 1.0)", 6),)
    if operation in {"sin", "cos", "tan"}:
        return ((f"isinf({value})", 7),)
    # Power has no forward domain check. Sections 12.2 and 12.3 make every
    # exceptional value a result: a negative base with a non-integral exponent
    # is NaN, a zero base with a negative exponent is an infinity, and an
    # overflowing result is an infinity. Codes 8 and 9 raised for exactly
    # those, so a fused plan disagreed with the same expression evaluated
    # eagerly, which docs/autodiff.md forbids.
    return ()


def _raise_fused_kernel_error(code: int) -> None:
    """Raise the public exception represented by a fused-kernel error code.

    Code 1 was division by zero. Floating division now delivers the IEEE
    result rather than raising, so no fused kernel emits it. Code 14 was the
    power derivative at a zero base, which section 12.7.2 now classifies as a
    value rather than an error.
    """
    messages = {
        2: "sqrt is only defined for non-negative values",
        3: "log is only defined for positive values",
        4: "inverse trigonometric function is only defined between -1 and 1",
        5: "arccosh is only defined for values greater than or equal to 1",
        6: "arctanh is only defined for values strictly between -1 and 1",
        7: "trigonometric functions are undefined for infinite values",
        10: "sqrt derivative is undefined at zero",
        11: "inverse trigonometric derivative is undefined at -1 and 1",
        12: "arccosh derivative is undefined at 1",
        13: "sign derivative is undefined at zero",
    }
    raise ValueError(messages.get(code, "invalid value in fused CUDA operation"))


def _fused_backward_checks(
    step: FusedElementwiseStep, value: str, *, storage_type: str
) -> tuple[tuple[str, int], ...]:
    """Return derivative-domain checks for one fused operation."""
    operation, _, _, _ = step
    if operation == "sqrt":
        return ((f"({value}) == 0.0", 10),)
    if operation in {"arcsin", "arccos"}:
        return ((f"({value}) == -1.0 || ({value}) == 1.0", 11),)
    if operation == "arccosh":
        return ((f"({value}) == 1.0", 12),)
    if operation == "sign":
        return ((f"({value}) == 0.0", 13),)
    # Power has no backward domain check either. Code 14 raised here for a
    # zero base with an exponent below one, which section 12.7.2 classifies
    # rather than rejects: the base derivative is +inf by convention for
    # 0 < y < 1 and NaN for y < 0, and rule G2 states that differentiation
    # does not raise on a numerical condition. The VJP expression now carries
    # the whole region table, so there is nothing left for a guard to catch.
    return ()
