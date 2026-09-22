"""Dispatch for shape, layout, indexing, and representation changes."""

from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING
from tensors.backend import config
from tensors.backend.dispatch._selected import run_on_selected_backend
from tensors.backend.storage import Storage
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_stack(
    values: Sequence[Tensor],
    axis: int,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Join tensor storage along a new axis, on the selected backend.

    Stacking is how a reverse pass gathers the contributions a repeated
    operand collected before reducing them, which puts it inside the
    execution contract of `docs/backends.md`: the selection decides where it
    runs, at every size, and a backend that cannot stack conformingly reports
    that rather than letting the Python reference answer.

    Under the workload-size policy this carried, the same reverse pass ran on
    the selected backend or in Python according to how many elements it held,
    and a small accumulated gradient then arrived at the next strict boundary
    residing on the wrong backend. Size is not part of what stacking means,
    so it is no longer part of where it happens.

    The values are copied into a new axis and nothing is calculated, so there
    is no numerical decision here: the axis, the dtype, the output shape and
    the logical element order are resolved by ``Stack.forward`` exactly as
    before, and each backend's kernel arranges the same values.

    The operands arrive as one sequence, and the residency check at the
    execution boundary reads its arguments one by one, so it sees that
    sequence as a single value carrying no residency and passes over the
    tensors inside it. They are checked here instead, before the kernel is
    reached and before anything is converted, so an operand residing
    elsewhere is rejected rather than quietly moved.
    """
    validate_backend_residency(values, config.get_backend())
    return run_on_selected_backend(
        "stack",
        values,
        axis,
        dtype=dtype,
        output_shape=output_shape,
        detail=f"at dtype {dtype.name}",
    )
