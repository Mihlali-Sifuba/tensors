from array import array
from typing import Iterable, Union, List, Tuple, Optional

from . import dtype as _dtype


def _shape_size(shape: Tuple[int, ...]) -> int:
    """Return the number of elements described by ``shape``."""
    size = 1
    for dimension in shape:
        size *= dimension
    return size


def _strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Return row-major strides for ``shape``."""
    stride = 1
    strides = []
    for dimension in reversed(shape):
        strides.append(stride)
        stride *= dimension
    return tuple(reversed(strides))


def _coordinates(index: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Convert a row-major flat index to coordinates for ``shape``."""
    coordinates = []
    for dimension, stride in zip(shape, _strides(shape)):
        coordinate = index // stride
        coordinates.append(coordinate)
        index %= stride
    return tuple(coordinates)


def _flat_index(coordinates: Tuple[int, ...], shape: Tuple[int, ...]) -> int:
    """Convert row-major coordinates to a flat index."""
    return sum(
        coordinate * stride
        for coordinate, stride in zip(coordinates, _strides(shape))
    )


def _broadcast_shape(a_shape: Tuple[int, ...], b_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Return the NumPy-style broadcast shape for two tensor shapes."""
    dimensions = []
    for a_dimension, b_dimension in zip(reversed(a_shape), reversed(b_shape)):
        if a_dimension == b_dimension:
            dimensions.append(a_dimension)
        elif a_dimension == 1:
            dimensions.append(b_dimension)
        elif b_dimension == 1:
            dimensions.append(a_dimension)
        else:
            raise ValueError(f"Shapes {a_shape} and {b_shape} cannot be broadcast")

    longer_shape = a_shape if len(a_shape) > len(b_shape) else b_shape
    matched_dimensions = min(len(a_shape), len(b_shape))
    dimensions.extend(reversed(longer_shape[:len(longer_shape) - matched_dimensions]))
    return tuple(reversed(dimensions))


def _broadcast_to(tensor: 'Tensor', shape: Tuple[int, ...]) -> 'Tensor':
    """Return ``tensor`` expanded to ``shape`` using singleton dimensions."""
    if tensor.shape == shape:
        return tensor
    if len(tensor.shape) > len(shape):
        raise ValueError(f"Shape {tensor.shape} cannot be broadcast to {shape}")

    padded_shape = (1,) * (len(shape) - tensor.ndim) + tensor.shape
    for source_dimension, target_dimension in zip(padded_shape, shape):
        if source_dimension not in {1, target_dimension}:
            raise ValueError(f"Shape {tensor.shape} cannot be broadcast to {shape}")

    values = []
    padding = len(shape) - tensor.ndim
    for output_index in range(_shape_size(shape)):
        output_coordinates = _coordinates(output_index, shape)
        source_coordinates = tuple(
            0 if source_dimension == 1 else coordinate
            for source_dimension, coordinate in zip(padded_shape, output_coordinates)
        )[padding:]
        values.append(tensor._data[_flat_index(source_coordinates, tensor.shape)])

    return Tensor(values, dtype=tensor.dtype, shape=shape)


def _broadcast_tensors(a: 'Tensor', b: 'Tensor') -> Tuple['Tensor', 'Tensor']:
    """Broadcast two tensors to a shared NumPy-style shape."""
    shape = _broadcast_shape(a.shape, b.shape)
    return _broadcast_to(a, shape), _broadcast_to(b, shape)


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
        data: Union[List, 'Tensor', array, int, float],
        dtype: Union[str, _dtype.DataType] = None,
        shape: Optional[Tuple[int, ...]] = None
    ):
        """
        Initialize a Tensor.

        Args:
            data: The data to store (list, array, number, or another Tensor)
            dtype: Data type (e.g. ``float64``, ``float32``, ``int32``).
                   Accepts a ``DataType`` or a raw array typecode string.
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
            raise TypeError(f"dtype must be a DataType or typecode, got {type(dtype)}")
        self.dtype = dtype

        # Handle different input types ----------------------------------
        if isinstance(data, Tensor):
            # Copy from another tensor
            self._data = self._make_array(data._data)
            inferred_shape = data.shape

        elif isinstance(data, (int, float)):
            # Scalar value
            self._data = self._make_array([data])
            inferred_shape = (1,)

        elif isinstance(data, list):
            # Flatten the list if it's nested
            flat_data = self._flatten_list(data)
            self._data = self._make_array(flat_data)

            inferred_shape = self._infer_shape(data)

        elif isinstance(data, array):
            # Direct array input
            self._data = self._make_array(data)
            inferred_shape = (len(data),)

        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        self.shape = self._validate_shape(inferred_shape if shape is None else shape)
        self.ndim = len(self.shape)

        # Verify total elements match shape
        total_elements = self._get_total_elements()
        if len(self._data) != total_elements:
            raise ValueError(
                f"Data size {len(self._data)} does not match shape {self.shape} "
                f"(expected {total_elements} elements)"
            )

    def _make_array(self, values: Iterable) -> array:
        """Create this tensor's backing storage."""
        return array(self.dtype.typecode, values)

    def _flatten_list(self, nested_list: List) -> List:
        """Flatten a nested list into a single list (iterative, no recursion limit)."""
        result: List = []
        stack: List = [nested_list]
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                # Reverse so original order is preserved when popping
                for sub in reversed(item):
                    stack.append(sub)
            else:
                result.append(item)
        return result

    def _infer_shape(self, nested_list: List) -> Tuple[int, ...]:
        """Infer shape and reject ragged nested lists."""
        if not isinstance(nested_list, list):
            return ()

        if not nested_list:
            return (0,)

        sub_shape = self._infer_shape(nested_list[0])
        for item in nested_list[1:]:
            item_shape = self._infer_shape(item)
            if item_shape != sub_shape:
                raise ValueError(
                    "Ragged nested lists are not valid tensor data: "
                    f"expected child shape {sub_shape}, got {item_shape}"
                )
        return (len(nested_list),) + sub_shape

    @staticmethod
    def _validate_shape(shape) -> Tuple[int, ...]:
        """Return a normalized shape tuple after validating dimensions."""
        try:
            normalized = tuple(shape)
        except TypeError as exc:
            raise TypeError("shape must be an iterable of non-negative integers") from exc
        for dim in normalized:
            if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
                raise ValueError(
                    f"Invalid shape {normalized}: dimensions must be non-negative integers"
                )
        return normalized

    def _get_total_elements(self) -> int:
        """Calculate total number of elements from shape."""
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    def _calculate_index(self, indices: Tuple[int, ...]) -> int:
        """Convert N-dimensional indices to flat index (row-major order)."""
        if len(indices) != self.ndim:
            raise IndexError(
                f"Expected {self.ndim} indices, got {len(indices)}"
            )

        flat_idx = 0
        for i, idx in enumerate(indices):
            if idx < 0:
                idx += self.shape[i]
            if not (0 <= idx < self.shape[i]):
                raise IndexError("Index out of range")
            # Stride = product of all remaining dimensions
            stride = 1
            for j in range(i + 1, self.ndim):
                stride *= self.shape[j]
            flat_idx += idx * stride

        return flat_idx

    def __getitem__(self, key):
        """
        Support indexing and slicing for N-dimensional tensors.

        Examples::
            tensor[0]              # First element (1D)
            tensor[1, 2, 3]        # Element at indices (3D)
            tensor[0:2]            # First 2 elements (1D)
            tensor[0, :, 1:3]      # Mixed int/slice (3D)
        """
        # A single key indexes the first dimension for N-D tensors.
        if isinstance(key, (int, slice)):
            if self.ndim != 1:
                return self._nd_slice((key,))
            if isinstance(key, int):
                idx = self._calculate_index((key,))
                return self._data[idx]
            indices = range(*key.indices(self.shape[0]))
            values = [self._data[i] for i in indices]
            return Tensor(values, dtype=self.dtype, shape=(len(indices),))

        # Tuple of indices/slices — N-dimensional
        if isinstance(key, tuple):
            if len(key) == self.ndim and all(isinstance(k, int) for k in key):
                # All ints — return a scalar
                idx = self._calculate_index(key)
                return self._data[idx]

            # Mixed ints and slices — return a sub-tensor
            return self._nd_slice(key)

        raise TypeError(f"Unsupported index type: {type(key)}")

    def _nd_slice(self, key: tuple):
        """General N-dimensional slicing with mixed ints and slices."""
        if len(key) > self.ndim:
            raise IndexError(
                f"Too many indices: {len(key)} for {self.ndim}D tensor"
            )

        # Build ranges for each dimension and compute output shape
        ranges = []
        new_shape = []
        for dim_idx, k in enumerate(key):
            if isinstance(k, int):
                idx = k if k >= 0 else k + self.shape[dim_idx]
                if not (0 <= idx < self.shape[dim_idx]):
                    raise IndexError("Index out of range")
                ranges.append(range(idx, idx + 1))
                # Int collapses the dimension → not added to new_shape
            elif isinstance(k, slice):
                dim_range = range(*k.indices(self.shape[dim_idx]))
                ranges.append(dim_range)
                new_shape.append(len(dim_range))
            else:
                raise TypeError(
                    f"Unsupported index type in tuple: {type(k)}"
                )

        # If key is shorter than ndim, remaining dims are taken fully
        while len(ranges) < self.ndim:
            dim_idx = len(ranges)
            dim_range = range(self.shape[dim_idx])
            ranges.append(dim_range)
            new_shape.append(self.shape[dim_idx])

        # Precompute strides for each dimension
        strides = []
        for d in range(self.ndim):
            s = 1
            for j in range(d + 1, self.ndim):
                s *= self.shape[j]
            strides.append(s)

        # Extract data by iterating over all index combinations
        result_data = self._make_array([])
        self._extract_nd(0, 0, ranges, strides, result_data)

        return Tensor(result_data, dtype=self.dtype, shape=tuple(new_shape))

    def _extract_nd(self, dim: int, flat_offset: int, ranges: list,
                    strides: list, result: array):
        """Recursively iterate over N-dimensional index ranges."""
        if dim == len(ranges):
            result.append(self._data[flat_offset])
            return

        for idx in ranges[dim]:
            self._extract_nd(
                dim + 1,
                flat_offset + idx * strides[dim],
                ranges, strides, result
            )

    def __setitem__(self, key, value):
        """Support item assignment for N-dimensional tensors."""
        if isinstance(key, int):
            if self.ndim != 1:
                raise ValueError(
                    f"Cannot assign to {self.ndim}D tensor with single integer"
                )
            idx = self._calculate_index((key,))
            self._data[idx] = value
            return

        if isinstance(key, tuple):
            idx = self._calculate_index(key)
            self._data[idx] = value
            return

        raise TypeError(f"Unsupported index type: {type(key)}")

    def __repr__(self) -> str:
        """String representation of the tensor."""
        dtype_str = self.dtype.name

        if self.ndim == 0:
            return str(self._data[0])
        if self.ndim == 1:
            return f"Tensor({list(self._data)}, shape={self.shape}, dtype='{dtype_str}')"

        # Build recursive representation for N-dim
        lines = self._repr_recursive(0, 0)
        bracket_repr = "[\n" + "\n".join(lines) + "\n]"
        return f"Tensor(\n{bracket_repr},\n shape={self.shape}, dtype='{dtype_str}'\n)"

    def _repr_recursive(self, dim: int, offset: int) -> list:
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
            sub_lines = self._repr_recursive(dim + 1, offset + i * stride)
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
        return self.shape[0]

    @property
    def size(self) -> int:
        """Total number of elements."""
        return len(self._data)

    @property
    def itemsize(self) -> int:
        """Size of each element in bytes."""
        return self._data.itemsize

    def tolist(self) -> List:
        """Convert to Python list."""
        return list(self._data)

    def clone(self) -> 'Tensor':
        """Return a copy with the same data and dtype."""
        return Tensor(self)

    def astype(self, dtype: Union[str, _dtype.DataType]) -> 'Tensor':
        """Return a copy converted to a new dtype."""
        if isinstance(dtype, str):
            dtype = _dtype.from_typecode(dtype)
        if dtype.typecode in {"b", "B", "h", "i", "q"}:
            values = [int(x) for x in self._data]
        elif dtype.typecode in {"f", "d"}:
            values = [float(x) for x in self._data]
        else:
            values = list(self._data)
        return Tensor(values, dtype=dtype, shape=self.shape)

    def item(self) -> Union[int, float]:
        """Return the Python scalar stored in a single-element tensor."""
        if self.size != 1:
            raise ValueError(
                f"item() requires a tensor with one element, got {self.size}"
            )
        return self._data[0]

    def __format__(self, format_spec: str) -> str:
        """Format a scalar tensor using normal Python numeric formatting."""
        if self.size != 1:
            raise TypeError("Only single-element tensors can be formatted as scalars")
        return format(self.item(), format_spec)

    # ---------- Operator Overloads (delegate to ops) ----------
    def __add__(self, other):
        from .ops import Ops
        return Ops.add(self, other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        from .ops import Ops
        return Ops.subtract(self, other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        from .ops import Ops
        return Ops.multiply(self, other)

    def __rmul__(self, other):
        from .ops import Ops
        return Ops.multiply(self, other)

    def __truediv__(self, other):
        from .ops import Ops
        return Ops.divide(self, other)

    def __rtruediv__(self, other):
        from .ops import Ops
        numerator = Tensor([other] * self.size, shape=self.shape)
        return Ops.divide(numerator, self)

    def __neg__(self):
        from .ops import Ops
        return Ops.neg(self)

    def __matmul__(self, other):
        from .linalg import matmul
        return matmul(self, other)

    def __rmatmul__(self, other):
        from .linalg import matmul
        return matmul(other, self)
