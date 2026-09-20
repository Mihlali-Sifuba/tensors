"""The boundary where Tensor operands become an execution-ready request.

Three layers meet here, and each is kept to one question:

- the **operation layer** knows Tensors. It resolves the result dtype and the
  output shape, converts a Python scalar, and then asks for a request. It
  never names a backend and never sees a backend-native value.
- this **preparation boundary** knows both sides. It reads the selection,
  holds the operands to the residency rule while they are still Tensors, and
  asks the selected backend to turn them into its own native values.
- the **dispatcher** knows backends. It receives the request, routes it to the
  named backend's kernel, and reports a decline. It never sees a Tensor.

The backends prepare differently and that stays inside them: Python expands a
broadcast operand and pairs a scalar lazily, NumPy and CuPy hand over native
arrays and let their own broadcasting apply, and device data is never brought
to the host. This module only arranges that work; it does not perform it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from tensors.backend import config
from tensors.backend.loading import load_backend
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors._typing import Scalar
    from tensors.backend.types import BackendName
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


class BinaryExecution(NamedTuple):
    """One elementwise binary operation, ready to execute.

    Every field is already resolved: ``backend`` is a concrete selection,
    ``left`` and ``right`` are that backend's own values rather than Tensors,
    and ``dtype`` and ``output_shape`` are the result the operation layer
    decided on. A dispatcher receiving this has nothing left to work out.
    """

    backend: BackendName
    left: Any
    right: Any
    dtype: DataType
    output_shape: tuple[int, ...]


def prepare_binary_execution(
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> BinaryExecution:
    """Return an execution-ready request for one elementwise binary operation.

    The selection is read once, here, and travels in the request, so a
    dispatcher never resolves it a second time.

    Residency is checked while the operands are still Tensors, because that is
    the only point at which a Tensor's backend is a meaningful question. By
    the time the request exists its operands are backend-native, and the only
    residency question left is the one the dispatcher asks of the result.
    """
    selected = config.get_backend()
    validate_backend_residency((left, right), selected)
    backend: Any = load_backend(selected)
    prepared_left, prepared_right = backend.prepare_binary_operands(
        left, right, dtype=dtype, output_shape=output_shape
    )
    return BinaryExecution(
        backend=selected,
        left=prepared_left,
        right=prepared_right,
        dtype=dtype,
        output_shape=output_shape,
    )


__all__ = ["BinaryExecution", "prepare_binary_execution"]
