from array import array
from typing import Union, List, Tuple, Optional

from . import dtype as _dtype


class Tensor:
    """
    A simple tensor implementation using Python's array module.

    Supports:
    - N-dimensional tensors
    - Basic operations (add, subtract, multiply)
    - Reshaping
    - Transpose
    - Indexing and slicing
    - Broadcasting (basic)
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
        # Resolve dtype -------------------------------------------------
        if dtype is None:
            dtype = _dtype.default
        elif isinstance(dtype, str):
            dtype = _dtype.from_typecode(dtype)
        self.dtype = dtype

        # Handle different input types ----------------------------------
        if isinstance(data, Tensor):
            # Copy from another tensor
            self._data = dtype.make_array(data._data)
            self.shape = data.shape
            self.ndim = data.ndim

        elif isinstance(data, (int, float)):
            # Scalar value
            self._data = dtype.make_array([data])
            self.shape = (1,)
            self.ndim = 1

        elif isinstance(data, list):
            # Flatten the list if it's nested
            flat_data = self._flatten_list(data)
            self._data = dtype.make_array(flat_data)

            # Infer shape if not provided
            if shape is None:
                self.shape = self._infer_shape(data)
            else:
                self.shape = shape
            self.ndim = len(self.shape)

        elif isinstance(data, array):
            # Direct array input
            self._data = dtype.make_array(data)
            if shape is None:
                self.shape = (len(data),)
            else:
                self.shape = shape
            self.ndim = len(self.shape)

        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        # Verify total elements match shape
        total_elements = self._get_total_elements()
        if len(self._data) != total_elements:
            raise ValueError(
                f"Data size {len(self._data)} does not match shape {self.shape} "
                f"(expected {total_elements} elements)"
            )
    
    def _flatten_list(self, nested_list: List) -> List:
        """Recursively flatten a nested list into a single list."""
        result = []
        for item in nested_list:
            if isinstance(item, list):
                result.extend(self._flatten_list(item))
            else:
                result.append(item)
        return result
    
    def _infer_shape(self, nested_list: List) -> Tuple[int, ...]:
        """Infer the shape of a nested list."""
        if not isinstance(nested_list, list):
            return ()
        
        if not nested_list or not isinstance(nested_list[0], list):
            return (len(nested_list),)
        
        # Get shape of first sublist
        sub_shape = self._infer_shape(nested_list[0])
        return (len(nested_list),) + sub_shape
    
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
        # Single int or slice for 1D tensor
        if isinstance(key, (int, slice)):
            if self.ndim != 1:
                raise ValueError(
                    f"Cannot use single {'int' if isinstance(key, int) else 'slice'} "
                    f"on {self.ndim}D tensor"
                )
            if isinstance(key, int):
                idx = self._calculate_index((key,))
                return self._data[idx]
            # Slice
            start, stop, step = key.start, key.stop, key.step
            if start is None:
                start = 0
            if stop is None:
                stop = self.shape[0]
            if step is None:
                step = 1
            if start < 0:
                start += self.shape[0]
            if stop < 0:
                stop += self.shape[0]
            return Tensor(self._data[start:stop:step])

        # Tuple of indices/slices — N-dimensional
        if isinstance(key, tuple):
            if all(isinstance(k, int) for k in key):
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
                start, stop, step = k.start, k.stop, k.step
                if start is None:
                    start = 0
                if stop is None:
                    stop = self.shape[dim_idx]
                if step is None:
                    step = 1
                if start < 0:
                    start += self.shape[dim_idx]
                if stop < 0:
                    stop += self.shape[dim_idx]
                dim_range = range(start, stop, step)
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
        result_data = self.dtype.make_array([])
        self._extract_nd(0, 0, ranges, strides, result_data)

        return Tensor(result_data, shape=tuple(new_shape))

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
            return f"Tensor({list(self._data)}, shape=({self.shape[0],}), dtype='{dtype_str}')"

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

        return f"Tensor(shape={self.shape}, dtype='{self.dtype.name}')"
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

    # ---------- Operator Overloads (delegate to ops) ----------
    def __add__(self, other):
        from .ops import Ops
        return Ops.add(self, other)

    def __sub__(self, other):
        from .ops import Ops
        return Ops.subtract(self, other)

    def __mul__(self, other):
        from .ops import Ops
        return Ops.multiply(self, other)

    def __rmul__(self, other):
        from .ops import Ops
        return Ops.multiply(self, other)

    def __truediv__(self, other):
        from .ops import Ops
        return Ops.divide(self, other)

    def __neg__(self):
        from .ops import Ops
        return Ops.multiply(self, -1)
