"""Kernels evaluated with CuPy device arrays."""

from tensors.backend.cuda.kernels.nn.binary_cross_entropy import (
    binary_cross_entropy as binary_cross_entropy,
)
from tensors.backend.cuda.kernels.nn.binary_cross_entropy_gradient import (
    binary_cross_entropy_gradient as binary_cross_entropy_gradient,
)
from tensors.backend.cuda.kernels.nn.cross_entropy import cross_entropy as cross_entropy
from tensors.backend.cuda.kernels.nn.cross_entropy_gradient import (
    cross_entropy_gradient as cross_entropy_gradient,
)
from tensors.backend.cuda.kernels.nn.distributions_valid import (
    distributions_valid as distributions_valid,
)
from tensors.backend.cuda.kernels.nn.log_softmax import log_softmax as log_softmax
from tensors.backend.cuda.kernels.nn.log_softmax_gradient import (
    log_softmax_gradient as log_softmax_gradient,
)
from tensors.backend.cuda.kernels.nn.softmax import softmax as softmax
from tensors.backend.cuda.kernels.nn.softmax_gradient import (
    softmax_gradient as softmax_gradient,
)
