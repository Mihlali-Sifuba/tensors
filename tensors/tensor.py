from __future__ import annotations

from array import array
from itertools import product
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, overload

from . import dtype as _dtype
from ._typing import (
    Scalar,
    TensorData,
    TensorIndex,
    TensorLike,
    TensorOperand,
    TensorResult,
)
from .casting import cast_values
from .utils.lists import flatten_nested_list, infer_nested_list_shape
from .utils.shape import (
    normalize_shape,
    row_major_strides,
    shape_size,
)
from .utils.slicing import flat_indices_from_ranges, slice_ranges_and_shape_from_key
from .utils.indexing import indices_to_flat_index

if TYPE_CHECKING:
    from .variable import Variable


class Tensor:
    """
    A simple tensor implementation using Python's array module.

    Supports:
    - N-dimensional tensors
    - Basic operations (add, subtract, multiply)
    - Reshaping
    - Transpose
    - Indexing and slicing
    - Scalar arithmetic
    """

    def __init__(
        self,
        data: TensorData,
        dtype: str | _dtype.DataType | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> None:
        """
        Initialize a Tensor.

        Args:
            data: The data to store (list, array, number, or another Tensor)
            dtype: Data type (e.g. ``float64``, ``float32``, ``int32``).
                   Accepts a ``DataType``, a raw array typecode string, or a
                   human-readable dtype name such as ``"float32"``.
                   Defaults to ``float64``.
            shape: Shape of the tensor. If None, inferred from data.
        """
        # Resolve dtype. Copies and raw arrays preserve their dtype unless
        # the caller explicitly requests a conversion.
        if dtype is None:
            if isinstance(data, Tensor):
                dtype = data.dtype
            elif isinstance(data, array):
                dtype = _dtype.from_typecode(data.typecode)
            else:
                dtype = _dtype.default
        elif isinstance(dtype, str):
            dtype = _dtype.from_typecode(dtype)

        if not isinstance(dtype, _dtype.DataType):
            raise TypeError(
                f"dtype must be a DataType, typecode or dtype string, got {type(dtype)}"
            )
        self._dtype = dtype

        # Handle different input types ----------------------------------
        if isinstance(data, Tensor):
            # Copy from another tensor
            self._data = self._create_storage(data._data)
            inferred_shape = data.shape

        elif isinstance(data, (int, float)):
            # Scalar value
            self._data = self._create_storage([data])
            inferred_shape = (1,)

        elif isinstance(data, list):
            # Flatten the list if it's nested
            flat_data = flatten_nested_list(data)
            self._data = self._create_storage(flat_data)

            inferred_shape = infer_nested_list_shape(data)

        elif isinstance(data, array):
            # Direct array input
            self._data = self._create_storage(data)
            inferred_shape = (len(data),)

        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        self._shape = normalize_shape(
            inferred_shape if shape is None else shape
        )
        self._ndim = len(self.shape)

        # Public in-place mutations increment this counter. Computation nodes
        # remember the counter observed during their forward pass so backward
        # can reject stale values instead of silently calculating a derivative
        # from data that no longer matches the recorded computation.
        self._version = 0

        # Verify total elements match shape
        expected_element_count = shape_size(self.shape)
        if len(self._data) != expected_element_count:
            raise ValueError(
                f"Data size {len(self._data)} does not match shape {self.shape} "
                f"(expected {expected_element_count} elements)"
            )

    def _create_storage(self, values: Iterable[Scalar]) -> array:
        """Create this tensor's backing storage."""
        if isinstance(values, array) and values.typecode == self.dtype.typecode:
            return array(values.typecode, values)
        if self.dtype.kind == "integer":
            converted = (int(value) for value in values)
        else:
            converted = (float(value) for value in values)
        return array(self.dtype.typecode, converted)

    def __getitem__(self, key: TensorIndex) -> Scalar | Tensor:
        """
        Support indexing and slicing for N-dimensional tensors.

        Examples::
            tensor[0]              # First element (1D)
            tensor[1, 2, 3]        # Element at indices (3D)
            tensor[0:2]            # First 2 elements (1D)
            tensor[0, :, 1:3]      # Mixed int/slice (3D)
        """
        # A single key indexes the first dimension for N-D tensors.
        if isinstance(key, bool):
            raise TypeError("Boolean tensor indices are not supported")
        if not isinstance(key, (int, slice, tuple)):
            raise TypeError(f"Unsupported index type: {type(key)}")

        keys = key if isinstance(key, tuple) else (key,)
        _, output_shape = slice_ranges_and_shape_from_key(keys, self.shape)
        from .backend import execute_slice

        accelerated = execute_slice(
            self,
            key,
            output_shape=output_shape,
        )
        if accelerated is not None:
            if output_shape == ():
                return accelerated[0]
            return Tensor(
                accelerated,
                dtype=self.dtype,
                shape=output_shape,
            )

        if isinstance(key, (int, slice)):
            if self.ndim != 1:
                return self._slice_from_key((key,))
            if isinstance(key, int):
                idx = indices_to_flat_index((key,), self.shape)
                return self._data[idx]
            indices = range(*key.indices(self.shape[0]))
            values = [self._data[i] for i in indices]
            return Tensor(values, dtype=self.dtype, shape=(len(indices),))

        # Tuple of indices/slices — N-dimensional
        if isinstance(key, tuple):
            if len(key) == self.ndim and all(isinstance(k, int) for k in key):
                # All ints — return a scalar
                idx = indices_to_flat_index(key, self.shape)
                return self._data[idx]

            # Mixed ints and slices — return a sub-tensor
            return self._slice_from_key(key)

        raise TypeError(f"Unsupported index type: {type(key)}")

    def _slice_from_key(
        self,
        key: tuple[int | slice, ...],
    ) -> Tensor:
        """Return an N-dimensional slice selected by mixed ints and slices."""
        ranges, new_shape = slice_ranges_and_shape_from_key(key, self.shape)
        strides = row_major_strides(self.shape)

        result_data = self._create_storage(
            self._data[sum(
                coordinate * stride
                for coordinate, stride in zip(coordinates, strides)
            )]
            for coordinates in product(*ranges)
        )

        return Tensor(result_data, dtype=self.dtype, shape=new_shape)

    def _slice_assignment_values(
        self,
        value: TensorData,
        selection_shape: tuple[int, ...],
    ) -> array:
        """Validate and materialize values for an in-place slice assignment."""
        selection_size = shape_size(selection_shape)
        if isinstance(value, (int, float)):
            return self._create_storage([value] * selection_size)

        if not isinstance(value, (Tensor, list, array)):
            raise TypeError(
                f"Slice assignment value must be a number, list, array or Tensor, "
                f"got {type(value)}"
            )

        assignment = Tensor(value, dtype=self.dtype)
        from .utils.broadcasting import broadcast_to

        try:
            assignment = broadcast_to(assignment, selection_shape)
        except ValueError as exc:
            raise ValueError(
                f"Cannot assign shape {assignment.shape} "
                f"to slice shape {selection_shape}"
            ) from exc

        return self._create_storage(assignment._data)

    def _assign_slice_from_key(
        self,
        key: tuple[int | slice, ...],
        value: TensorData,
    ) -> None:
        """Assign a scalar or broadcast-compatible values to a tensor slice."""
        ranges, selection_shape = slice_ranges_and_shape_from_key(key, self.shape)
        flat_indices = flat_indices_from_ranges(
            ranges,
            row_major_strides(self.shape),
        )
        assignment_values = self._slice_assignment_values(
            value,
            selection_shape,
        )

        if len(assignment_values) != len(flat_indices):
            raise ValueError(
                f"Slice assignment has {len(assignment_values)} values; "
                f"expected {len(flat_indices)}"
            )

        for flat_index, assignment_value in zip(flat_indices, assignment_values):
            self._data[flat_index] = assignment_value
        self._version += 1

    def __setitem__(
        self,
        key: TensorIndex,
        value: TensorData,
    ) -> None:
        """Support item assignment for N-dimensional tensors."""
        if isinstance(key, bool):
            raise TypeError("Boolean tensor indices are not supported")
        if isinstance(key, slice):
            self._assign_slice_from_key((key,), value)
            return

        if isinstance(key, int):
            if self.ndim != 1:
                raise ValueError(
                    f"Cannot assign to {self.ndim}D tensor with single integer"
                )
            idx = indices_to_flat_index((key,), self.shape)
            self._data[idx] = self._assignment_scalar(value)
            self._version += 1
            return

        if isinstance(key, tuple):
            if any(isinstance(part, bool) for part in key):
                raise TypeError("Boolean tensor indices are not supported")
            if any(isinstance(part, slice) for part in key):
                self._assign_slice_from_key(key, value)
                return

            idx = indices_to_flat_index(key, self.shape)
            self._data[idx] = self._assignment_scalar(value)
            self._version += 1
            return

        raise TypeError(f"Unsupported index type: {type(key)}")

    def _assignment_scalar(
        self,
        value: TensorData,
    ) -> Scalar:
        """Convert one value using the same rules as tensor construction."""
        if isinstance(value, (Tensor, list, array)):
            converted = Tensor(value, dtype=self.dtype)
            if converted.size != 1:
                raise ValueError("Item assignment requires exactly one value")
            return converted.item()
        if not isinstance(value, (int, float)):
            raise TypeError("Item assignment value must be numeric")
        return self._create_storage([value])[0]

    def __repr__(self) -> str:
        """String representation of the tensor."""
        dtype_str = self.dtype.name

        if self.ndim == 0:
            return str(self._data[0])
        if self.ndim == 1:
            return f"Tensor({list(self._data)}, shape={self.shape}, dtype='{dtype_str}')"
        if self.ndim > 32:
            return (
                f"Tensor({list(self._data)}, shape={self.shape}, "
                f"dtype='{dtype_str}')"
            )

        # Build recursive representation for N-dim
        lines = self._format_nested_repr(0, 0)
        bracket_repr = "[\n" + "\n".join(lines) + "\n]"
        return f"Tensor(\n{bracket_repr},\n shape={self.shape}, dtype='{dtype_str}'\n)"

    def _format_nested_repr(self, dim: int, offset: int) -> list:
        """Build repr lines recursively for N-dimensional display."""
        indent = "  " * dim
        stride = 1
        for j in range(dim + 1, self.ndim):
            stride *= self.shape[j]

        if dim == self.ndim - 1:
            # Innermost dimension — show values
            values = " ".join(
                str(self._data[offset + j]) for j in range(self.shape[dim])
            )
            return [f"{indent}[{values}]"]

        # Recurse into sub-blocks
        lines = []
        for i in range(self.shape[dim]):
            sub_lines = self._format_nested_repr(dim + 1, offset + i * stride)
            if dim == 0:
                lines.extend(sub_lines)
            else:
                lines.extend(sub_lines)
            if i < self.shape[dim] - 1 and dim < self.ndim - 2:
                lines.append("")

        if dim == 0:
            return lines

        # Wrap in brackets for inner dimensions
        wrapped = [f"{indent}["]
        for line in lines:
            wrapped.append(line)
        wrapped.append(f"{indent}]")
        return wrapped


    def __len__(self) -> int:
        """Return the size of the first dimension."""
        if self.ndim == 0:
            raise TypeError("len() of a 0-dimensional tensor")
        return self.shape[0]

    @property
    def size(self) -> int:
        """Total number of elements."""
        return len(self._data)

    @property
    def shape(self) -> tuple[int, ...]:
        """Immutable dimensions of this tensor."""
        return self._shape

    @property
    def ndim(self) -> int:
        """Number of tensor dimensions."""
        return self._ndim

    @property
    def dtype(self) -> _dtype.DataType:
        """Immutable element data type of this tensor."""
        return self._dtype

    @property
    def version(self) -> int:
        """Number of successful in-place mutations made to this tensor."""
        return self._version

    @property
    def itemsize(self) -> int:
        """Size of each element in bytes."""
        return self._data.itemsize

    def tolist(self) -> list[Scalar]:
        """Convert to Python list."""
        return list(self._data)

    def clone(self) -> Tensor:
        """Return a copy with the same data and dtype."""
        return Tensor(self)

    def astype(self, dtype: str | _dtype.DataType) -> Tensor:
        """Return a copy converted to a new dtype."""
        if isinstance(dtype, str):
            dtype = _dtype.from_typecode(dtype)

        if not isinstance(dtype, _dtype.DataType):
            raise TypeError(
                f"dtype must be a DataType, typecode or dtype string, got {type(dtype)}"
            )

        from .backend import execute_cast

        accelerated = execute_cast(self, dtype=dtype)
        if accelerated is not None:
            return Tensor(accelerated, dtype=dtype, shape=self.shape)

        values = cast_values(
            self._data,
            source_dtype=self.dtype,
            target_dtype=dtype,
        )
        return Tensor(values, dtype=dtype, shape=self.shape)

    def item(self) -> Scalar:
        """Return the Python scalar stored in a single-element tensor."""
        if self.size != 1:
            raise ValueError(
                f"item() requires a tensor with one element, got {self.size}"
            )
        return self._data[0]

    def __bool__(self) -> bool:
        """Prevent a Tensor from being used as a Python bool.

        Raises:
            TypeError: Always — to catch ``if tensor:`` bugs early.
        """
        raise TypeError(
            "Cannot convert a Tensor to a Python bool. "
            "Use tensor.item() for scalar tensors or "
            "tensor.size != 0 for emptiness checks."
        )

    def __eq__(self, other: object) -> bool:
        """Return whether another Tensor has the same shape and values."""
        if not isinstance(other, Tensor):
            return NotImplemented
        return self.shape == other.shape and self.tolist() == other.tolist()

    def __ne__(self, other: object) -> bool:
        """Return whether another Tensor differs in shape or values."""
        result = self.__eq__(other)
        return NotImplemented if result is NotImplemented else not result

    def __format__(self, format_spec: str) -> str:
        """Format a scalar tensor using normal Python numeric formatting."""
        if self.size != 1:
            raise TypeError("Only single-element tensors can be formatted as scalars")
        return format(self.item(), format_spec)

    # ---------- Operator Overloads (delegate to ops) ----------
    @overload
    def __add__(self, other: Variable) -> Variable: ...

    @overload
    def __add__(self, other: Scalar | Tensor) -> Tensor: ...

    def __add__(self, other: TensorOperand) -> TensorResult:
        from .variable import Variable
        if isinstance(other, Variable):
            return other.__radd__(self)
        from .ops import Ops
        return Ops.add(self, other)

    @overload
    def __radd__(self, other: Variable) -> Variable: ...

    @overload
    def __radd__(self, other: Scalar | Tensor) -> Tensor: ...

    def __radd__(self, other: TensorOperand) -> TensorResult:
        return self + other

    @overload
    def __sub__(self, other: Variable) -> Variable: ...

    @overload
    def __sub__(self, other: Scalar | Tensor) -> Tensor: ...

    def __sub__(self, other: TensorOperand) -> TensorResult:
        from .variable import Variable
        if isinstance(other, Variable):
            return other.__rsub__(self)
        from .ops import Ops
        return Ops.subtract(self, other)

    def __rsub__(self, other: Scalar | Tensor) -> Tensor:
        return (-self) + other

    @overload
    def __mul__(self, other: Variable) -> Variable: ...

    @overload
    def __mul__(self, other: Scalar | Tensor) -> Tensor: ...

    def __mul__(self, other: TensorOperand) -> TensorResult:
        from .variable import Variable
        if isinstance(other, Variable):
            return other.__rmul__(self)
        from .ops import Ops
        return Ops.multiply(self, other)

    def __rmul__(self, other: Scalar | Tensor) -> Tensor:
        from .ops import Ops
        return Ops.multiply(self, other)

    @overload
    def __truediv__(self, other: Variable) -> Variable: ...

    @overload
    def __truediv__(self, other: Scalar | Tensor) -> Tensor: ...

    def __truediv__(self, other: TensorOperand) -> TensorResult:
        from .variable import Variable
        if isinstance(other, Variable):
            return other.__rtruediv__(self)
        from .ops import Ops
        return Ops.divide(self, other)

    def __rtruediv__(self, other: Scalar) -> Tensor:
        from .ops import Div
        return Div.forward_reverse(self, other)

    @overload
    def __pow__(self, other: Variable) -> Variable: ...

    @overload
    def __pow__(self, other: Scalar | Tensor) -> Tensor: ...

    def __pow__(self, other: TensorOperand) -> TensorResult:
        from .variable import Variable
        if isinstance(other, Variable):
            return other.__rpow__(self)
        from .ops import Pow
        return Pow.forward(self, other)

    def __rpow__(self, other: Scalar) -> Tensor:
        from .ops import Pow
        return Pow.forward_reverse(self, other)

    def __neg__(self) -> Tensor:
        from .ops import Ops
        return Ops.neg(self)

    def __abs__(self) -> Tensor:
        from .math import abs
        return abs(self)

    @overload
    def __matmul__(self, other: Variable) -> Variable: ...

    @overload
    def __matmul__(self, other: TensorData) -> Tensor: ...

    def __matmul__(self, other: TensorLike) -> TensorResult:
        from .linalg import matmul
        return matmul(self, other)

    @overload
    def __rmatmul__(self, other: Variable) -> Variable: ...

    @overload
    def __rmatmul__(self, other: TensorData) -> Tensor: ...

    def __rmatmul__(self, other: TensorLike) -> TensorResult:
        from .linalg import matmul
        return matmul(other, self)
