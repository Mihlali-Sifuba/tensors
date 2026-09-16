"""Kernels evaluated with CuPy device arrays."""

from tensors.backend.cuda.kernels.optim.adam_update import adam_update as adam_update
from tensors.backend.cuda.kernels.optim.adam_updates import adam_updates as adam_updates
from tensors.backend.cuda.kernels.optim.rmsprop_update import (
    rmsprop_update as rmsprop_update,
)
from tensors.backend.cuda.kernels.optim.rmsprop_updates import (
    rmsprop_updates as rmsprop_updates,
)
from tensors.backend.cuda.kernels.optim.sgd_update import sgd_update as sgd_update
from tensors.backend.cuda.kernels.optim.sgd_updates import sgd_updates as sgd_updates
