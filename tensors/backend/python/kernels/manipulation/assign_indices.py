"""Reference in-place indexed writing for the Python backend."""

from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors._typing import Scalar


def assign_indices(
    destination: Storage,
    indices: Sequence[int],
    values: Storage | Scalar,
    *,
    source_indices: Sequence[int] | None = None,
) -> bool | None:
    """Write the values into the addressed positions of the buffer itself.

    The buffer is the tensor's authoritative storage, so the writes land
    where the tensor already lives and no representation is installed in its
    place. A value carrying no backend is one value for every position.
    """
    buffer = destination.buffer
    if not isinstance(values, Storage):
        for index in indices:
            buffer[index] = values
        return True
    source = values.buffer
    if source_indices is None:
        for position, index in enumerate(indices):
            buffer[index] = source[position]
        return True
    for index, source_index in zip(indices, source_indices):
        buffer[index] = source[source_index]
    return True
