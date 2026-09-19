"""Reference kernels evaluated with ordinary Python arithmetic."""

from tensors.backend.python.kernels.manipulation.cast_tensor import (
    cast_tensor as cast_tensor,
)
from tensors.backend.python.kernels.manipulation.concat import concat as concat
from tensors.backend.python.kernels.manipulation.slice_scatter import (
    slice_scatter as slice_scatter,
)
from tensors.backend.python.kernels.manipulation.slice_tensor import (
    slice_tensor as slice_tensor,
)
from tensors.backend.python.kernels.manipulation.stack import stack as stack
from tensors.backend.python.kernels.manipulation.transpose import transpose as transpose
