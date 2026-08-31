from __future__ import annotations

from array import array
from itertools import product
from collections.abc import Iterable, Iterator
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
from .shape import Shape
from .storage import (
    CudaStorage,
    NumPyStorage,
    PythonStorage,
    Storage,
    StorageKind,
    convert_storage,
)
from .strides import Strides
from .utils.lists import flatten_nested_list, infer_nested_list_shape
from .utils.slicing import flat_indices_from_ranges, slice_ranges_and_shape_from_key
from .utils.indexing import (
    coordinates_to_storage_index,
    tensor_indices_to_storage_index,
)

if TYPE_CHECKING:
    from .variable import Variable


class Tensor:
    """
    An n-dimensional typed tensor with backend-native storage.

    Tensor operations dispatch through the configured Python, NumPy, or CUDA
    backend and preserve native storage where possible.

    Supports:
    - Explicit shape, strides, offset, and storage metadata
    - Broadcasting and scalar arithmetic
    - Matrix multiplication
    - Reshaping and transposition
    - Indexing, slicing, and mutation
    - Dtype conversion and scalar extraction

    Gradient tracking is provided by :class:`Variable`; ``Tensor`` itself is
    the non-differentiable value and storage abstraction.
    """

    def __init__(
        self,
        data: TensorData,
        dtype: str | _dtype.DataType | None = None,
        shape: Iterable[int] | None = None,
    ) -> None:
        """
        Initialize a tensor.

        Args:
            data: Scalar, nested list, array, backend storage, or tensor to
                initialize from.
            dtype: Optional ``DataType``, array typecode, or dtype name.
                Existing dtype is preserved for tensor, storage, and array
                inputs; otherwise, the default is ``float64``.
            shape: Optional tensor shape. When omitted, inferred from ``data``.
                Python scalar inputs infer the rank-zero shape ``()``.

        Raises:
            TypeError: If ``data`` or ``dtype`` is unsupported, or a storage
                dtype conflicts with ``dtype``.
            ValueError: If a dtype name is unknown or the data size does not
                match ``shape``.
        """
        # Resolve dtype. Copies and raw arrays preserve their dtype unless
        # the caller explicitly requests a conversion.
        if dtype is None:
            if isinstance(data, Tensor):
                dtype = data.dtype
            elif isinstance(data, Storage):
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
            # Preserve native storage for same-dtype copies. An explicitly
            # requested dtype still follows Tensor's normal cast semantics.
            if data.dtype == self.dtype:
                self._set_storage(
                    data._logical_storage_for(data._storage.kind).copy()
                )
            else:
                self._set_storage(
                    PythonStorage.from_values(data._data, self.dtype)
                )
            inferred_shape = data.shape

        elif isinstance(data, Storage):
            if data.dtype != self.dtype:
                raise TypeError(
                    f"storage dtype {data.dtype.name!r} does not match "
                    f"tensor dtype {self.dtype.name!r}"
                )
            self._set_storage(data)
            inferred_shape = (data.size,)

        elif isinstance(data, (int, float)):
            # Scalar value
            self._set_storage(PythonStorage.from_values([data], self.dtype))
            inferred_shape = ()

        elif isinstance(data, list):
            # Flatten the list if it's nested
            flat_data = flatten_nested_list(data)
            self._set_storage(PythonStorage.from_values(flat_data, self.dtype))

            inferred_shape = infer_nested_list_shape(data)

        elif isinstance(data, array):
            # Direct array input
            self._set_storage(PythonStorage.from_values(data, self.dtype))
            inferred_shape = (len(data),)

        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        self._shape = Shape.from_iterable(
            inferred_shape if shape is None else shape
        )
        self._strides = Strides.contiguous(self._shape)
        self._offset = 0

        # Public in-place mutations increment this counter. Computation nodes
        # remember the counter observed during their forward pass so backward
        # can reject stale values instead of silently calculating a derivative
        # from data that no longer matches the recorded computation.
        self._version = 0

        # Verify total elements match shape
        expected_element_count = self.shape.size
        if self._storage.size != expected_element_count:
            raise ValueError(
                f"Data size {self._storage.size} does not match shape {self.shape} "
                f"(expected {expected_element_count} elements)"
            )

    @classmethod
    def _from_metadata(
        cls,
        storage: Storage,
        *,
        shape: Shape | Iterable[int],
        strides: Strides | Iterable[int],
        offset: int = 0,
    ) -> Tensor:
        """Build an owning Tensor with explicit layout metadata.

        Storage is copied deliberately. This internal constructor establishes
        and tests the future view representation without introducing shared
        storage or mutation aliasing.
        """
        tensor = cls.__new__(cls)
        tensor._dtype = storage.dtype
        tensor._set_storage(storage.copy())
        tensor._shape = (
            shape if isinstance(shape, Shape) else Shape.from_iterable(shape)
        )
        tensor._strides = (
            strides
            if isinstance(strides, Strides)
            else Strides.from_iterable(strides)
        )
        tensor._offset = offset
        tensor._version = 0
        tensor._validate_layout()
        return tensor

    def _validate_layout(self) -> None:
        """Validate that all logical coordinates address owned storage."""
        if len(self.strides) != self.shape.rank:
            raise ValueError(
                f"Stride rank {len(self.strides)} does not match "
                f"shape rank {self.shape.rank}"
            )
        if isinstance(self.offset, bool) or not isinstance(self.offset, int):
            raise TypeError("offset must be an integer")

        if self.size == 0:
            if not 0 <= self.offset <= self._storage.size:
                raise ValueError("empty tensor offset is outside storage")
            return

        minimum = self.offset
        maximum = self.offset
        for dimension, stride in zip(self.shape, self.strides):
            extent = (dimension - 1) * stride
            if extent < 0:
                minimum += extent
            else:
                maximum += extent
        if minimum < 0 or maximum >= self._storage.size:
            raise ValueError(
                f"Tensor layout addresses storage range [{minimum}, {maximum}] "
                f"outside buffer of size {self._storage.size}"
            )

    def _set_storage(self, storage: Storage) -> None:
        """Install authoritative storage and invalidate other representations."""
        self._storage = storage
        self._storage_cache: dict[StorageKind, Storage] = {
            storage.kind: storage,
        }

    def _storage_for(self, kind: StorageKind) -> Storage:
        """Return a cached native representation, converting only once."""
        cached = self._storage_cache.get(kind)
        if cached is not None:
            return cached
        converted = convert_storage(self._storage, kind)
        self._storage_cache[kind] = converted
        return converted

    @property
    def _has_compact_storage(self) -> bool:
        """Whether contiguous values exactly fill storage from offset zero.

        Compact storage is stricter than a contiguous logical layout, which
        may begin at a non-zero offset.
        """
        return (
            self.offset == 0
            and self.is_contiguous
            and self._storage.size == self.size
        )

    def _logical_storage_indices(self) -> Iterator[int]:
        """Yield physical positions in logical row-major order."""
        ranges = (range(dimension) for dimension in self.shape)
        for coordinates in product(*ranges):
            yield coordinates_to_storage_index(
                coordinates,
                self.shape,
                self.strides,
                self.offset,
            )

    def _logical_storage_for(self, kind: StorageKind) -> Storage:
        """Return compact logical values in one backend-native storage kind."""
        storage = self._storage_for(kind)
        if self._has_compact_storage:
            return storage

        indices = list(self._logical_storage_indices())
        if kind == "python":
            if not isinstance(storage, PythonStorage):
                raise TypeError("Python storage conversion returned an invalid buffer")
            return PythonStorage.from_values(
                (storage.buffer[index] for index in indices),
                self.dtype,
            )

        selected = (
            storage.buffer[indices]
            if indices
            else storage.buffer[:0]
        )
        if kind == "numpy":
            return NumPyStorage(selected, self.dtype)
        return CudaStorage(selected, self.dtype)

    def _mutable_data(self) -> array:
        """Return authoritative host storage for an in-place mutation."""
        storage = self._storage_for("python")
        if not isinstance(storage, PythonStorage):
            raise TypeError("Python storage conversion returned an invalid buffer")
        self._set_storage(storage)
        return storage.buffer

    @property
    def _data(self) -> array:
        """Logical row-major host values for reference kernels."""
        storage = self._logical_storage_for("python")
        if not isinstance(storage, PythonStorage):
            raise TypeError("Python storage conversion returned an invalid buffer")
        return storage.buffer

    def _value_at_storage_index(self, index: int) -> Scalar:
        """Read one physical position from authoritative host storage."""
        storage = self._storage_for("python")
        if not isinstance(storage, PythonStorage):
            raise TypeError("Python storage conversion returned an invalid buffer")
        return storage.buffer[index]

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
                return Tensor(
                    accelerated,
                    dtype=self.dtype,
                    shape=output_shape,
                ).item()
            return Tensor(
                accelerated,
                dtype=self.dtype,
                shape=output_shape,
            )

        if isinstance(key, (int, slice)):
            if self.ndim != 1:
                return self._slice_from_key((key,))
            if isinstance(key, int):
                idx = tensor_indices_to_storage_index(
                    (key,),
                    self.shape,
                    self.strides,
                    self.offset,
                )
                return self._value_at_storage_index(idx)
            indices = range(*key.indices(self.shape[0]))
            values = [self._data[i] for i in indices]
            return Tensor(values, dtype=self.dtype, shape=(len(indices),))

        # Tuple of indices/slices — N-dimensional
        if isinstance(key, tuple):
            if len(key) == self.ndim and all(isinstance(k, int) for k in key):
                # All ints — return a scalar
                idx = tensor_indices_to_storage_index(
                    key,
                    self.shape,
                    self.strides,
                    self.offset,
                )
                return self._value_at_storage_index(idx)

            # Mixed ints and slices — return a sub-tensor
            return self._slice_from_key(key)

        raise TypeError(f"Unsupported index type: {type(key)}")

    def _slice_from_key(
        self,
        key: tuple[int | slice, ...],
    ) -> Tensor:
        """Return an N-dimensional slice selected by mixed ints and slices."""
        ranges, new_shape = slice_ranges_and_shape_from_key(key, self.shape)
        storage = self._storage_for("python")
        if not isinstance(storage, PythonStorage):
            raise TypeError("Python storage conversion returned an invalid buffer")
        result_data = self._create_storage(
            storage.buffer[coordinates_to_storage_index(
                coordinates,
                self.shape,
                self.strides,
                self.offset,
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
        selection_size = Shape.from_iterable(selection_shape).size
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
        physical_indices = flat_indices_from_ranges(
            ranges,
            self.strides,
            self.offset,
        )
        assignment_values = self._slice_assignment_values(
            value,
            selection_shape,
        )

        if len(assignment_values) != len(physical_indices):
            raise ValueError(
                f"Slice assignment has {len(assignment_values)} values; "
                f"expected {len(physical_indices)}"
            )

        mutable_data = self._mutable_data()
        for physical_index, assignment_value in zip(
            physical_indices,
            assignment_values,
        ):
            mutable_data[physical_index] = assignment_value
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
            idx = tensor_indices_to_storage_index(
                (key,),
                self.shape,
                self.strides,
                self.offset,
            )
            self._mutable_data()[idx] = self._assignment_scalar(value)
            self._version += 1
            return

        if isinstance(key, tuple):
            if any(isinstance(part, bool) for part in key):
                raise TypeError("Boolean tensor indices are not supported")
            if any(isinstance(part, slice) for part in key):
                self._assign_slice_from_key(key, value)
                return

            idx = tensor_indices_to_storage_index(
                key,
                self.shape,
                self.strides,
                self.offset,
            )
            self._mutable_data()[idx] = self._assignment_scalar(value)
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
        return self.shape.size

    @property
    def shape(self) -> Shape:
        """Immutable dimensions of this tensor."""
        return self._shape

    @property
    def strides(self) -> Strides:
        """Immutable physical movement for each logical axis."""
        return self._strides

    @property
    def offset(self) -> int:
        """Storage position corresponding to logical coordinate zero."""
        return self._offset

    @property
    def ndim(self) -> int:
        """Number of tensor dimensions."""
        return self.shape.rank

    @property
    def is_contiguous(self) -> bool:
        """Whether logical elements occupy contiguous row-major storage.

        The starting offset does not affect logical contiguity.
        """
        if self.size == 0:
            return True

        expected_stride = 1
        for dimension, stride in zip(
            reversed(self.shape),
            reversed(self.strides),
        ):
            if dimension == 1:
                continue
            if stride != expected_stride:
                return False
            expected_stride *= dimension
        return True

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
        return self.dtype.size

    def tolist(self) -> list[Scalar]:
        """Convert to Python list."""
        return list(self._data)

    def clone(self) -> Tensor:
        """Return a copy with the same data and dtype."""
        return Tensor(self)

    def contiguous(self) -> Tensor:
        """Ensure that this tensor has a contiguous logical layout.

        An already-contiguous tensor is returned unchanged, including when it
        begins at a non-zero storage offset. This method does not promise
        compact storage. Non-contiguous tensors are materialized into
        independent compact storage without transferring NumPy or CUDA values
        through the host.
        """
        if self.is_contiguous:
            return self
        storage = self._logical_storage_for(self._storage.kind)
        return Tensor(storage, dtype=self.dtype, shape=self.shape)

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
