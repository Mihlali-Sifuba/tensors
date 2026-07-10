"""Operations for Tensor objects.

All operations live on the Ops class as static methods.
"""

import builtins
import math
from typing import Union

from .tensor import Tensor


class Ops:
    """Tensor operations. All methods are static."""

    # ---------- Arithmetic ----------

    @staticmethod
    def add(a: Tensor, b: Union[Tensor, int, float]) -> Tensor:
        """Element-wise addition."""
        if isinstance(b, (int, float)):
            new_data = a.dtype.make_array(x + b for x in a._data)
            return Tensor(new_data, shape=a.shape)
        if isinstance(b, Tensor):
            if a.shape != b.shape:
                raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
            new_data = a.dtype.make_array([])
            for x, y in zip(a._data, b._data):
                new_data.append(x + y)
            return Tensor(new_data, shape=a.shape)
        raise TypeError(f"Unsupported operand type: {type(b)}")

    @staticmethod
    def subtract(a: Tensor, b: Union[Tensor, int, float]) -> Tensor:
        """Element-wise subtraction."""
        if isinstance(b, (int, float)):
            new_data = a.dtype.make_array(x - b for x in a._data)
            return Tensor(new_data, shape=a.shape)
        if isinstance(b, Tensor):
            if a.shape != b.shape:
                raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
            new_data = a.dtype.make_array([])
            for x, y in zip(a._data, b._data):
                new_data.append(x - y)
            return Tensor(new_data, shape=a.shape)
        raise TypeError(f"Unsupported operand type: {type(b)}")

    @staticmethod
    def multiply(a: Tensor, b: Union[Tensor, int, float]) -> Tensor:
        """Element-wise multiplication."""
        if isinstance(b, (int, float)):
            new_data = a.dtype.make_array(x * b for x in a._data)
            return Tensor(new_data, shape=a.shape)
        if isinstance(b, Tensor):
            if a.shape != b.shape:
                raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
            new_data = a.dtype.make_array([])
            for x, y in zip(a._data, b._data):
                new_data.append(x * y)
            return Tensor(new_data, shape=a.shape)
        raise TypeError(f"Unsupported operand type: {type(b)}")

    @staticmethod
    def divide(a: Tensor, b: Union[Tensor, int, float]) -> Tensor:
        """Element-wise division."""
        if isinstance(b, (int, float)):
            if b == 0:
                raise ZeroDivisionError("Division by zero")
            new_data = a.dtype.make_array(x / b for x in a._data)
            return Tensor(new_data, shape=a.shape)
        if isinstance(b, Tensor):
            if a.shape != b.shape:
                raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
            new_data = a.dtype.make_array([])
            for x, y in zip(a._data, b._data):
                if y == 0:
                    raise ZeroDivisionError("Division by zero")
                new_data.append(x / y)
            return Tensor(new_data, shape=a.shape)
        raise TypeError(f"Unsupported operand type: {type(b)}")

    # ---------- Linear Algebra ----------

    @staticmethod
    def dot(a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiplication (dot product) for 2D tensors.

        Args:
            a: A 2D tensor (left operand)
            b: A 2D tensor (right operand)

        Returns:
            A new tensor with shape (a.shape[0], b.shape[1])
        """
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Dot product only supported for 2D tensors")

        if a.shape[1] != b.shape[0]:
            raise ValueError(
                f"Cannot multiply {a.shape} with {b.shape}: "
                f"inner dimensions must match"
            )

        result_data = a.dtype.make_array([])
        a_cols = a.shape[1]
        b_cols = b.shape[1]

        for i in range(a.shape[0]):
            for j in range(b_cols):
                total = 0
                for k in range(a_cols):
                    a_idx = i * a_cols + k
                    b_idx = k * b_cols + j
                    total += a._data[a_idx] * b._data[b_idx]
                result_data.append(total)

        return Tensor(result_data, shape=(a.shape[0], b.shape[1]))

    # ---------- Shape ----------

    @staticmethod
    def transpose(tensor: Tensor) -> Tensor:
        """Transpose a 2D tensor (swap rows and columns)."""
        if tensor.ndim != 2:
            raise NotImplementedError("Transpose only implemented for 2D tensors")

        new_data = tensor.dtype.make_array([])
        rows, cols = tensor.shape

        for j in range(cols):
            for i in range(rows):
                idx = i * cols + j
                new_data.append(tensor._data[idx])

        return Tensor(new_data, shape=(cols, rows))

    @staticmethod
    def reshape(tensor: Tensor, *new_shape: int) -> Tensor:
        """Reshape a tensor to a new shape (total elements must match)."""
        total_elements = tensor._get_total_elements()
        new_total = 1
        for dim in new_shape:
            new_total *= dim

        if total_elements != new_total:
            raise ValueError(
                f"Cannot reshape tensor of size {total_elements} to shape {new_shape}"
            )

        return Tensor(tensor._data, shape=new_shape)

    # ---------- Statistics ----------

    @staticmethod
    def sum(tensor: Tensor) -> float:
        """Sum of all elements."""
        return builtins.sum(tensor._data)

    @staticmethod
    def mean(tensor: Tensor) -> float:
        """Mean (average) of all elements."""
        if tensor.size == 0:
            return 0
        return Ops.sum(tensor) / tensor.size

    @staticmethod
    def min(tensor: Tensor) -> float:
        """Minimum value in the tensor."""
        if tensor.size == 0:
            raise ValueError("Cannot compute min of empty tensor")
        return builtins.min(tensor._data)

    @staticmethod
    def max(tensor: Tensor) -> float:
        """Maximum value in the tensor."""
        if tensor.size == 0:
            raise ValueError("Cannot compute max of empty tensor")
        return builtins.max(tensor._data)

    @staticmethod
    def std(tensor: Tensor) -> float:
        """Standard deviation of all elements."""
        if tensor.size == 0:
            return 0
        m = Ops.mean(tensor)
        variance = builtins.sum((x - m) ** 2 for x in tensor._data) / tensor.size
        return math.sqrt(variance)
