"""Value casting between supported tensor data types."""

from typing import Iterable, List, Union

from .dtype import DataType


def cast_values(
    values: Iterable[Union[int, float]],
    source_dtype: DataType,
    target_dtype: DataType,
) -> List[Union[int, float]]:
    """Cast numeric values from ``source_dtype`` to ``target_dtype``."""
    if source_dtype.kind not in {"integer", "floating"}:
        raise TypeError(f"Unsupported source dtype: {source_dtype.name}")

    if target_dtype.kind == "integer":
        return [int(value) for value in values]
    if target_dtype.kind == "floating":
        return [float(value) for value in values]
    raise TypeError(f"Unsupported target dtype: {target_dtype.name}")


__all__ = ["cast_values"]
