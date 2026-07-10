from array import array
from typing import Union, List, Tuple, Optional

from . import dtype as _dtype


class Tensor:
    """
    A simple tensor implementation using Python's array module.

    Supports:
    - 1D and 2D tensors
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
        """
        Convert multi-dimensional indices to flat index.
        Works for 1D and 2D tensors.
        """
        if len(indices) != self.ndim:
            raise IndexError(
                f"Expected {self.ndim} indices, got {len(indices)}"
            )
        
        # Simple case: 1D tensor
        if self.ndim == 1:
            idx = indices[0]
            if idx < 0:
                idx += self.shape[0]
            if not (0 <= idx < self.shape[0]):
                raise IndexError("Index out of range")
            return idx
        
        # 2D tensor: row-major order
        if self.ndim == 2:
            row, col = indices
            if row < 0:
                row += self.shape[0]
            if col < 0:
                col += self.shape[1]
            if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
                raise IndexError("Index out of range")
            return row * self.shape[1] + col
        
        raise NotImplementedError(f"Indexing for {self.ndim}D tensors not implemented")
    
    def __getitem__(self, key):
        """
        Support indexing and slicing.
        
        Examples:
            tensor[0]           # Get first element (1D)
            tensor[0, 1]        # Get element at row 0, col 1 (2D)
            tensor[0, :]        # Get first row (2D)
            tensor[:, 0]        # Get first column (2D)
        """
        # Handle single integer for 1D tensor
        if isinstance(key, int):
            if self.ndim != 1:
                raise ValueError(f"Cannot index {self.ndim}D tensor with single integer")
            idx = self._calculate_index((key,))
            return self._data[idx]
        
        # Handle slice object (1D)
        if isinstance(key, slice):
            if self.ndim != 1:
                raise ValueError(f"Slicing {self.ndim}D tensor with single slice not supported")
            
            start, stop, step = key.start, key.stop, key.step
            if start is None:
                start = 0
            if stop is None:
                stop = self.shape[0]
            if step is None:
                step = 1
            
            # Handle negative indices
            if start < 0:
                start += self.shape[0]
            if stop < 0:
                stop += self.shape[0]
            
            slice_data = self._data[start:stop:step]
            return Tensor(slice_data)
        
        # Handle tuple of indices (multi-dimensional)
        if isinstance(key, tuple):
            # Handle slice notation
            if any(isinstance(k, slice) for k in key):
                # For simplicity, only handle full slices for 2D tensors
                # (e.g., tensor[0, :] returns a 1D tensor)
                return self._handle_slice_tuple(key)
            
            # Standard indexing
            idx = self._calculate_index(key)
            return self._data[idx]
        
        raise TypeError(f"Unsupported index type: {type(key)}")
    
    def _handle_slice_tuple(self, key: tuple):
        """
        Handle slicing with tuple notation.
        
        For 2D tensors:
            tensor[0, :]    -> returns row 0 as 1D tensor
            tensor[:, 0]    -> returns column 0 as 1D tensor
        """
        if self.ndim != 2:
            raise NotImplementedError(f"Slicing for {self.ndim}D tensors not implemented")
        
        row_key, col_key = key
        
        # Case 1: row slice, col slice -> 2D tensor
        if isinstance(row_key, slice) and isinstance(col_key, slice):
            return self._handle_full_slice(row_key, col_key)
        
        # Case 2: int row, slice col -> row as 1D tensor
        if isinstance(row_key, int) and isinstance(col_key, slice):
            row = row_key if row_key >= 0 else row_key + self.shape[0]
            if not (0 <= row < self.shape[0]):
                raise IndexError("Row index out of range")
            
            start, stop, step = col_key.start, col_key.stop, col_key.step
            if start is None:
                start = 0
            if stop is None:
                stop = self.shape[1]
            if step is None:
                step = 1
            
            if start < 0:
                start += self.shape[1]
            if stop < 0:
                stop += self.shape[1]
            
            # Extract data for this row and slice columns
            row_start = row * self.shape[1]
            row_end = (row + 1) * self.shape[1]
            row_data = self._data[row_start:row_end]
            slice_data = row_data[start:stop:step]
            return Tensor(slice_data)
        
        # Case 3: slice row, int col -> column as 1D tensor
        if isinstance(row_key, slice) and isinstance(col_key, int):
            col = col_key if col_key >= 0 else col_key + self.shape[1]
            if not (0 <= col < self.shape[1]):
                raise IndexError("Column index out of range")
            
            # Extract data for this column
            col_data = self.dtype.make_array([])
            for row in range(self.shape[0]):
                col_data.append(self._data[row * self.shape[1] + col])
            return Tensor(col_data)
        
        raise NotImplementedError(f"Unsupported slice combination: {key}")
    
    def _handle_full_slice(self, row_slice: slice, col_slice: slice):
        """Handle full 2D slicing (both dimensions are slices)."""
        # Process row slice
        r_start, r_stop, r_step = row_slice.start, row_slice.stop, row_slice.step
        if r_start is None:
            r_start = 0
        if r_stop is None:
            r_stop = self.shape[0]
        if r_step is None:
            r_step = 1
        
        if r_start < 0:
            r_start += self.shape[0]
        if r_stop < 0:
            r_stop += self.shape[0]
        
        # Process col slice
        c_start, c_stop, c_step = col_slice.start, col_slice.stop, col_slice.step
        if c_start is None:
            c_start = 0
        if c_stop is None:
            c_stop = self.shape[1]
        if c_step is None:
            c_step = 1
        
        if c_start < 0:
            c_start += self.shape[1]
        if c_stop < 0:
            c_stop += self.shape[1]
        
        # Extract data
        result_data = self.dtype.make_array([])
        new_rows = len(range(r_start, r_stop, r_step))
        new_cols = len(range(c_start, c_stop, c_step))
        
        for row in range(r_start, r_stop, r_step):
            for col in range(c_start, c_stop, c_step):
                idx = row * self.shape[1] + col
                result_data.append(self._data[idx])
        
        return Tensor(result_data, shape=(new_rows, new_cols))
    
    def __setitem__(self, key, value):
        """Support item assignment."""
        if isinstance(key, int):
            if self.ndim != 1:
                raise ValueError(f"Cannot assign to {self.ndim}D tensor with single integer")
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
        if self.ndim == 0:
            return str(self._data[0])
        
        # Show shape
        shape_str = " × ".join(str(d) for d in self.shape)
        dtype_str = self.dtype.name

        # Build a nice representation
        if self.ndim == 1:
            return f"Tensor({list(self._data)}, shape=({self.shape[0],}), dtype='{dtype_str}')"

        if self.ndim == 2:
            # Build grid representation
            rows = []
            for i in range(self.shape[0]):
                row_values = []
                for j in range(self.shape[1]):
                    row_values.append(str(self._data[i * self.shape[1] + j]))
                rows.append("[" + " ".join(row_values) + "]")
            matrix_str = "[\n  " + "\n  ".join(rows) + "\n]"
            return f"Tensor(\n{matrix_str},\n shape=({self.shape[0]}, {self.shape[1]}), dtype='{dtype_str}'\n)"

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
