"""Dispatch for shape, layout, indexing, and representation changes."""

from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
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
    """
    selected = config.get_backend()
    validate_backend_residency(values, selected)
    backend: Any = load_backend(selected)
    result = backend.stack(values, axis, dtype=dtype, output_shape=output_shape)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute stack at dtype "
            f"{dtype.name} conformingly. This computation runs on the selected "
            f"backend; select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
