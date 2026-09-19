"""Kernels evaluated with CuPy device arrays."""

from tensors.backend.cuda.kernels.creation.arange import arange as arange
from tensors.backend.cuda.kernels.creation.eye import eye as eye
from tensors.backend.cuda.kernels.creation.full import full as full
from tensors.backend.cuda.kernels.creation.linspace import linspace as linspace
from tensors.backend.cuda.kernels.creation.one_hot_targets import (
    one_hot_targets as one_hot_targets,
)
