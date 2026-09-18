"""IEEE binary32 arithmetic on the device, including gradual underflow.

`docs/arithmetic-semantics.md` section 5.4 requires gradual underflow for both
subnormal operands and subnormal results. CuPy's generated float32 elementwise
code flushes subnormals to zero on this toolchain, so ``1e-40 * 1.0`` returns
zero and ``min_normal * 0.5`` returns zero rather than a subnormal.

The device itself is not the limitation. NVRTC emits the IEEE instruction
``mul.f32`` by default; the flush is introduced below PTX. Naming the rounded
instruction explicitly avoids it, and the results then match the host bit for
bit, subnormals included.

``.rn`` selects round-to-nearest-even, which section 5.2 requires, and writing
the instruction out also prevents the compiler from contracting a multiply and
an add into a fused multiply-add.

Only float32 needs this. Double precision already underflows gradually on the
device, so it uses CuPy's own operations.

The same applies to exponentiation, which has no PTX instruction: see
:func:`_build_power`.
"""

from __future__ import annotations

from typing import Any

import cupy

#: One PTX instruction per operation, all round-to-nearest-even.
_INSTRUCTION = {
    "add": "add.rn.f32",
    "subtract": "sub.rn.f32",
    "multiply": "mul.rn.f32",
    "divide": "div.rn.f32",
}

_KERNELS: dict[str, Any] = {}


def _build(name: str) -> Any:
    """Compile the elementwise kernel for one operation."""
    instruction = _INSTRUCTION[name]
    body = (
        f'asm volatile("{instruction} %0, %1, %2;" '
        ': "=f"(out) : "f"(left), "f"(right));'
    )
    return cupy.ElementwiseKernel(
        "float32 left, float32 right",
        "float32 out",
        body,
        f"tensors_ieee32_{name}",
    )


#: Widen and narrow through inline PTX. The compiler flushes a subnormal in
#: ``cvt.f64.f32`` and ``cvt.rn.f32.f64`` exactly as it does in an arithmetic
#: instruction, and ``--ftz=false`` does not reach it, so the conversions are
#: written out for the same reason the instructions above are.
_WIDEN = 'asm("cvt.f64.f32 %0, %1;" : "=d"(widened) : "f"({operand}));'
_NARROW = 'asm("cvt.rn.f32.f64 %0, %1;" : "=f"(out) : "d"(computed));'

_POWER_BODY = f"""
double widened_left, widened_right, computed;
{{
    double widened;
    {_WIDEN.format(operand="left")}
    widened_left = widened;
    {_WIDEN.format(operand="right")}
    widened_right = widened;
}}
computed = pow(widened_left, widened_right);
{_NARROW}
"""


def _build_power() -> Any:
    """Compile the binary32 power kernel.

    PTX has no ``pow`` instruction and ``powf`` flushes its subnormal operands,
    so the operands are widened to binary64 — where the device does not flush —
    the power is taken there, and the result is rounded once back to binary32.
    Both conversions are written in PTX so that neither flushes.

    Rounding once from a binary64 ``pow`` is the same strategy the fused
    kernels already use for binary32. It is not a claim about accuracy: the
    2 ULP and 4 ULP bounds of section 12.6 are D5's, and are not established
    here.
    """
    return cupy.ElementwiseKernel(
        "float32 left, float32 right",
        "float32 out",
        _POWER_BODY,
        "tensors_ieee32_power",
    )


def kernel(name: str) -> Any:
    """Return the binary32 kernel for an operation, compiled on first use."""
    built = _KERNELS.get(name)
    if built is None:
        built = _build_power() if name == "power" else _build(name)
        _KERNELS[name] = built
    return built


def apply(name: str, left: Any, right: Any) -> Any:
    """Evaluate one binary32 operation with IEEE semantics."""
    return kernel(name)(left, right)


__all__ = ["apply", "kernel"]
